from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import gzip


def _safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def head(
    path: Path,
    n: int = 8,
    *,
    compression: str = "GZIP",
    decode_images: bool = False,
    out_dir: Optional[Path] = None,
    max_side: int = 1024,
) -> List[Dict[str, Any]]:
    """
    Inspect first N TFRecord examples.
    If TensorFlow is not installed, falls back to JSONL+gzip preview (your fallback writer).
    If decode_images=True and bytes exist, persist preview PNGs to out_dir and return file paths.
    """
    try:
        import tensorflow as tf  # real TFRecord path

        comp = "GZIP" if compression == "GZIP" else ""
        ds = tf.data.TFRecordDataset([str(path)], compression_type=comp).take(n)
        out: List[Dict[str, Any]] = []

        if decode_images and out_dir is not None:
            _safe_mkdir(out_dir)

        for i, raw in enumerate(ds):
            ex = tf.train.Example()
            ex.ParseFromString(bytes(raw.numpy()))
            f = ex.features.feature
            has_bytes = False
            img_bytes = b""
            if "image/encoded" in f and f["image/encoded"].bytes_list.value:
                img_bytes = f["image/encoded"].bytes_list.value[0]
                has_bytes = len(img_bytes) > 0

            rec: Dict[str, Any] = {
                "example/id": f["example/id"].bytes_list.value[0].decode()
                if "example/id" in f
                else None,
                "H": int(f["image/height"].int64_list.value[0])
                if "image/height" in f
                else None,
                "W": int(f["image/width"].int64_list.value[0])
                if "image/width" in f
                else None,
                "has_bytes": bool(
                    "image/encoded" in f and f["image/encoded"].bytes_list.value
                ),
                "n_det": len(f["bbox/xmin"].float_list.value)
                if "bbox/xmin" in f
                else 0,
                "n_cls": len(f["cls/labels"].bytes_list.value)
                if "cls/labels" in f
                else 0,
            }

            saved = None
            if decode_images and has_bytes and out_dir is not None:
                img_bytes = f["image/encoded"].bytes_list.value[0]
                saved = (
                    _persist_png_tf(img_bytes, out_dir, rec["example/id"], max_side)
                    if out_dir
                    else None
                )
            rec["preview_path"] = str(saved) if saved else None
            out.append(rec)
        return out

    except ImportError:
        # Fallback: your JSONL writer output (cannot decode images; they were not serialized as bytes)
        opener = gzip.open if compression == "GZIP" else open
        out: List[Dict[str, Any]] = []
        with opener(path, "rt", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i >= n:
                    break
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    rec = {"_raw_line": line.strip()}
                # Normalize preview keys if possible
                out.append(
                    {
                        "example/id": rec.get("example/id") or rec.get("image_id"),
                        "H": rec.get("image/height") or rec.get("height"),
                        "W": rec.get("image/width") or rec.get("width"),
                        "has_bytes": bool(rec.get("image/encoded")),
                        "n_det": len(rec.get("bbox/xmin", [])),
                        "n_cls": len(rec.get("cls/labels", [])),
                        "preview_path": None,
                    }
                )
        return out


def _persist_png_tf(
    img_bytes: bytes, out_dir: Path, example_id: Optional[str], max_side: int
) -> Path:
    """Decode with TensorFlow, resize to <= max_side, and save PNG."""
    import tensorflow as tf

    # Decode to tensor HxWxC uint8
    img = tf.io.decode_image(img_bytes, channels=0, expand_animations=False)
    img.set_shape([None, None, None])
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]

    # Compute scale
    h_i = int(h.numpy())
    w_i = int(w.numpy())
    scale = (
        min(1.0, float(max_side) / float(max(h_i, w_i))) if max(h_i, w_i) > 0 else 1.0
    )
    if scale < 1.0:
        new_h = max(1, int(round(h_i * scale)))
        new_w = max(1, int(round(w_i * scale)))
        img = tf.image.resize(img, [new_h, new_w], method="bilinear")
        img = tf.cast(tf.round(img), tf.uint8)

    png_bytes = tf.io.encode_png(img).numpy()
    stem = example_id if example_id else "example"
    # sanitize filename
    stem = "".join(c for c in stem if c.isalnum() or c in ("-", "_"))[:128]
    fname = f"{stem}.preview.png"
    out_path = out_dir / fname
    with open(out_path, "wb") as f:
        f.write(png_bytes)
    return out_path
