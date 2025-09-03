from pathlib import Path
from typing import Optional, Any, Tuple, List, Literal, Dict
import io

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

# ---- path & counters ----
def _compute_shard_path(
    *,
    base_dir: Path,
    shard_prefix: str,
    meta: Optional[Dict[str, Any]],
    fallback_idx: int,
) -> Path:
    """
    Prefer deterministic meta['ordinal'] from read_images; else fallback counter.
    """
    ord_ = None
    if isinstance(meta, dict):
        ord_ = meta.get("ordinal")
        if ord_ is None and "meta" and isinstance(meta["meta"], dict):
            ord_ = meta["meta"].get("ordinal")
    if ord_ is None:
        ord_ = fallback_idx
    name = f"{shard_prefix}-{int(ord_):06d}.parquet"
    return (Path(base_dir) / name).resolve()

# ---- table construction ----
def _normalize_images_batch(
    data: Any,
    meta: Optional[Dict[str, Any]],
) -> Tuple[List[Any], Optional[List[Any]], List[str]]:
    """
    Accept one of:
      - data = (images, labels, paths)
      - data = (images, labels)
      - data = images
    Returns (images_list, labels_list|None, paths_list)
    """
    if isinstance(data, tuple):
        if len(data) == 3:
            images, labels, paths = data
        elif len(data) == 2:
            images, labels = data
            paths = (meta or {}).get("paths", [])
        elif len(data) == 1:
            images = data[0]
            labels, paths = None, (meta or {}).get("paths", [])
        else:
            raise ValueError("Unsupported data tuple shape for images.")
    else:
        images = data
        labels, paths = None, (meta or {}).get("paths", [])

    images = list(images)
    labels = list(labels) if labels is not None else None
    paths = list(paths) if paths is not None else []

    if labels is not None and len(labels) != len(images):
        raise ValueError(f"labels length {len(labels)} != images length {len(images)}")
    if paths and len(paths) != len(images):
        # Placeholder to add warning, not a fatal mismatch
        pass

    return images, labels, paths

def _to_numpy(arr: Any) -> np.ndarray:
    """
    Accept np.ndarray, PIL.Image, TF/Torch tensor -> numpy array.
    """
    # PIL
    if isinstance(arr, Image.Image):
        return np.array(arr)
    # Torch
    try:
        import torch
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
    except Exception:
        pass
    # TF
    try:
        import tensorflow as tf
        if isinstance(arr, (tf.Tensor, tf.Variable)):
            arr = arr.numpy()
    except Exception:
        pass
    return np.asarray(arr)

def _ensure_uint8_rgb(arr: Any) -> np.ndarray: 
    """
    If use_blase_path: place under run output root (…/data/<subdir>)
    Else: use target_dir as absolute/relative path.
    """
    arr = _to_numpy(arr)

    # Handle channel-last 2D greyscale
    if arr.ndim == 2:
        arr = arr[:, :, None]

    # DType to unit8
    if arr.dtype != np.uint8:
        if arr.dtype.kind == "f":
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0).round().astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)

    # Channels normalization
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] == 4: # RGBA -> RGB
        arr = arr[..., :3]
    elif arr.shape[-1] == 3:
        pass
    else:
        raise ValueError(f"Unsupported channel count: {arr.shape}")
    
    return arr

def _encode_image(
    arr_u8_rgb: np.ndarray, 
    fmt: Literal["jpeg","png"], 
    quality: int
) -> bytes:
    """Encode to JPEG/PNG with Pillow."""
    if fmt not in ("jpeg", "png"):
        raise ValueError(f"Unsupported encode fmt: {fmt}")
    with io.BytesIO() as buf:
        if fmt == "jpeg":
            Image.fromarray(arr_u8_rgb, mode="RGB").save(buf, format="JPEG", quality=int(quality))
        else:
            Image.fromarray(arr_u8_rgb, mode="RGB").save(buf, format="RGB", optimize=True)
        return buf.getvalue()            

def _build_parquet_table_from_images(
    *,
    data: Any,
    meta: Optional[Dict[str, Any]],
    encode: Literal["jpeg","png"],
    jpeg_quality: int,
    include_paths: bool,
) -> pa.Table:
    """Create a Arrow table with encoded bytes + dims + optional labels/paths + lineage."""
    images, labels, paths = _normalize_images_batch(data, meta)

    enc_bytes, heights, widths, chans = [], [], [], []
    for img in images:
        u8 = _ensure_uint8_rgb(img)
        enc = _encode_image(u8, fmt=encode, quality=jpeg_quality)
        enc_bytes.append(enc)
        h, w, c = u8.shape[:3]
        heights.append(h)
        widths.append(w)
        chans.append(c)

    # lineage
    m = meta or {}
    # support both direct and nested styles
    manifest_root_hash = m.get("manifest_root_hash") or m.get("identity", {}).get("root_hash")
    batch_hash = m.get("batch_hash")

    cols = {
        "img_bytes": pa.array(enc_bytes, type=pa.binary()),
        "height": pa.array(heights, type=pa.int32()),
        "width": pa.array(widths, type=pa.int32()),
        "channels": pa.array(chans, type=pa.int8()),
        "label": (pa.array(labels) if labels is not None
                  else pa.nulls(len(enc_bytes), type=pa.int32())),
        "manifest_root_hash": pa.array([manifest_root_hash] * len(enc_bytes), type=pa.string()),
        "batch_hash": pa.array([batch_hash] * len(enc_bytes), type=pa.string()),
    }

    if include_paths:
        if paths and len(paths) == len(enc_bytes):
            cols["path"] = pa.array(paths, type=pa.string())
        else:
            cols["path"] = pa.nulls(len(enc_bytes), type=pa.string())

    return pa.table(cols)

# ---- writer & conflict policy ----
def _write_parquet_table(
    table: pa.Table,
    out_path: Path,
    *,
    compression: Literal["zstd","snappy"],
    on_conflict: Literal["overwrite","fail","rename"],
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        if on_conflict == "fail":
            raise FileExistsError(f"Shard exists: {out_path}")
        elif on_conflict == "rename":
            out_path = _auto_rename(out_path)
        # "overwrite" -> do nothing

    pq.write_table(table, out_path, compression=compression)
    return out_path

def _auto_rename(p: Path) -> Path:
    i, stem, suf = 1, p.stem, p.suffix
    while True:
        cand = p.with_name(f"{stem}.{i}{suf}")
        if not cand.exists():
            return cand
        i += 1