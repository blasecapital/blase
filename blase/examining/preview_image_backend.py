from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import random
import io
from collections import Counter
import json

import numpy as np
from PIL import Image, ImageOps, ImageFile

from blase.utils.hashing import Hash
from blase.utils.fs import ensure_parent_dir
from blase.types import (
    SourceParams,
    SourceBuild,
    SourceAdapter,
    Population,
    PreviewItem,
)
from blase.extracting.image_backend import (
    scan_manifest_headers,
    ensure_item_content_hashes,
    compute_manifest_root_hash,
)
from blase.extracting.manifest_utils import (
    build_manifest_descriptor,
)

# ---------------- Registry ----------------

SOURCE_REGISTRY: Dict[str, SourceAdapter] = {}


def register_source(kind: str, adapter: SourceAdapter) -> None:
    if not kind or not isinstance(kind, str):
        raise ValueError("kind must be a non-empty string.")
    SOURCE_REGISTRY[kind] = adapter


def get_source(kind: str) -> SourceAdapter:
    try:
        return SOURCE_REGISTRY[kind]
    except KeyError:
        avail = (
            ", ".join(sorted(SOURCE_REGISTRY.keys()))
            or "(empty; register_source first)"
        )
        raise ValueError(f"Unknown source kind: {kind!r}. Available: {avail}") from None


def has_source(kind: str) -> bool:
    return kind in SOURCE_REGISTRY


# ---------------- Built-in adapters ----------------


def _todo_adapter(_: SourceParams) -> SourceBuild:
    raise NotImplementedError("Adapter not implemented for this source kind.")


def directory_adapter(p: SourceParams) -> SourceBuild:
    """
    Params:
      - source: str|Path
      - pattern: str = '**/*'
      - allowed_ext: list[str] | None  (None → default image list; [] → accept any image/* via MIME)
      - recursive: bool = True
    """
    root = Path(p["source"]).resolve()
    pattern = p.get("pattern", "**/*")
    recursive = bool(p.get("recursive", True))
    allowed_ext: Optional[List[str]] = p.get("allowed_ext")
    if allowed_ext is None:
        allowed_ext = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]

    it = root.rglob(pattern) if recursive else root.glob(pattern)

    import mimetypes

    paths: List[str] = []
    skipped_by_ext: Dict[str, int] = {}

    def _is_image_file(q: Path) -> bool:
        if allowed_ext == []:
            mt, _ = mimetypes.guess_type(q.name)
            return bool(mt and mt.startswith("image/"))
        ext = q.suffix.lower()
        if ext in allowed_ext:
            return True
        mt, _ = mimetypes.guess_type(q.name)
        return bool(mt and mt.startswith("image/"))

    for q in it:
        if not q.is_file():
            continue
        if _is_image_file(q):
            paths.append(str(q.resolve()))
        else:
            skipped_by_ext[q.suffix.lower()] = (
                skipped_by_ext.get(q.suffix.lower(), 0) + 1
            )

    paths.sort()

    class DirPop:
        def size(self) -> int:
            return len(paths)

        def path_at(self, idx: int) -> str:
            return paths[idx]

        def descriptor(self) -> Dict[str, Any]:
            return {
                "kind": "directory",
                "source": str(root),
                "pattern": pattern,
                "recursive": recursive,
                "allowed_ext": ([] if allowed_ext == [] else sorted(set(allowed_ext))),
                "count": len(paths),
            }

    meta = {"skipped_by_ext": skipped_by_ext, "root_exists": root.exists()}
    return DirPop(), meta


def parquet_adapter(p: SourceParams) -> SourceBuild:
    """
    Params:
      - source: str|Path (single parquet file)
      - path_col: str | None (column with image paths; None → guess)
      - path_root: str | None (join when paths are relative)
    """
    import pyarrow.parquet as pq

    src = str(Path(p["source"]).resolve())
    path_col = p.get("path_col")
    path_root = Path(p["path_root"]).resolve() if p.get("path_root") else None

    table = pq.read_table(src, columns=[path_col] if path_col else None)
    if path_col is None:
        cols = set(table.column_names)
        for guess in ("path", "image_path", "filepath"):
            if guess in cols:
                path_col = guess
                break
        if path_col is None:
            raise ValueError(
                "parquet_path_col not provided and no common path column found."
            )

    raw = table[path_col].to_pylist()
    col: List[str] = []
    for x in raw:
        if x is None:
            continue
        s = str(x).strip()
        if not s or s.lower() == "none":
            continue
        pth = Path(s)
        if path_root and not pth.is_absolute():
            pth = path_root / pth
        col.append(str(pth.resolve()))

    class ParqPop:
        def size(self) -> int:
            return len(col)

        def path_at(self, idx: int) -> str:
            return col[idx]

        def descriptor(self) -> Dict[str, Any]:
            return {
                "kind": "parquet",
                "source": src,
                "path_col": path_col,
                "rows": len(col),
                "path_root": str(path_root) if path_root else None,
            }

    meta = {
        "rows_total": len(raw),
        "rows_with_paths": len(col),
        "rows_null_paths": len(raw) - len(col),
        "path_root": str(path_root) if path_root else None,
    }
    return ParqPop(), meta


# Register built-ins
register_source("directory", directory_adapter)
register_source("parquet", parquet_adapter)

for _k in ("csv", "jsonl", "tfrecord", "npy", "video", "hf_datasets", "s3"):
    register_source(_k, _todo_adapter)

# ---------------- Sample ----------------


def parquet_bytes_available(path: str, bytes_col: str) -> bool:
    """
    Quick check: does <path> have a non-empty bytes column <bytes_col> suitable for image decode?
    - Avoids full-file reads. Scans up to a few row groups / rows.
    - Accepts Binary/LargeBinary, FixedSizeBinary, or List<UInt8> shapes.
    Returns True if at least one non-null, non-empty cell is found.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception:
        print("Cannot import pyarrow dependencies.")
        return False

    try:
        pf = pq.ParquetFile(path)
    except Exception:
        print(f"Cannot read {path} as a ParquetFile.")
        return False

    # 1) column exists?
    names = pf.schema_arrow.names
    if bytes_col not in names:
        print("bytes_col not available.")
        return False

    # 2) type check (best-effort)
    i = names.index(bytes_col)
    fld = pf.schema_arrow.field(i)
    t = fld.type

    def _is_bytes_type(tp: pa.DataType) -> bool:
        if (
            pa.types.is_binary(tp)
            or pa.types.is_large_binary(tp)
            or pa.types.is_fixed_size_binary(tp)
        ):
            return True
        if pa.types.is_list(tp) and pa.types.is_uint8(tp.value_type):
            return True
        if pa.types.is_large_list(tp) and pa.types.is_uint8(tp.value_type):
            return True
        print("Invalid parquet data format.")
        return False

    if not _is_bytes_type(t):
        # some files store as string; allow but treat as unlikely
        if not pa.types.is_string(t) and not pa.types.is_large_string(t):
            return False

    # 3) scan a few row groups for any non-empty cell
    max_row_groups_to_check = (
        min(3, pf.num_row_groups or 0) if pf.num_row_groups is not None else 1
    )
    if max_row_groups_to_check == 0:
        # single-RG files sometimes report 0; fall back to whole read with small slice
        max_row_groups_to_check = 1

    try:
        if pf.num_row_groups and pf.num_row_groups > 0:
            for rg in range(max_row_groups_to_check):
                tbl = pf.read_row_group(rg, columns=[bytes_col])
                if _has_any_nonempty_bytes(tbl[bytes_col]):
                    return True
            print("ParquetFile has empty bytes.")
            return False
        else:
            # Fallback: read a small slice
            tbl = pf.read(columns=[bytes_col])
            col = tbl[bytes_col]
            # Limit scan to first 256 rows
            n = min(len(col), 256)
            if n == 0:
                print("No rows in table to scan.")
                return False
            return _has_any_nonempty_bytes(col.slice(0, n))
    except Exception:
        import traceback

        traceback.print_exc()
        return False


def _has_any_nonempty_bytes(arr) -> bool:
    """
    True if any element is non-null and length>0.
    Supports Binary/LargeBinary/FixedSizeBinary, (Large)String, and (Large)List<UInt8>.
    Scans up to 256 elements per chunk.
    """
    import pyarrow as pa

    chunks = arr.chunks if hasattr(arr, "chunks") else [arr]
    for ch in chunks:
        n = len(ch)
        if n == 0 or (hasattr(ch, "null_count") and ch.null_count == n):
            continue

        ty = ch.type
        m = min(n, 256)

        # validity as a numpy mask
        try:
            valid = ch.is_valid().to_numpy(zero_copy_only=False)
        except Exception:
            # fallback: treat all as possibly valid
            valid = [True] * n

        if (
            pa.types.is_binary(ty)
            or pa.types.is_large_binary(ty)
            or pa.types.is_fixed_size_binary(ty)
        ):
            for i in range(m):
                if not valid[i]:
                    continue
                try:
                    if len(ch[i].as_buffer()) > 0:
                        return True
                except Exception:
                    v = ch[i].as_py()
                    if v:
                        return True

        elif pa.types.is_string(ty) or pa.types.is_large_string(ty):
            for i in range(m):
                if not valid[i]:
                    continue
                s = ch[i].as_py() or ""
                if len(s) > 0:
                    return True

        elif pa.types.is_list(ty) or pa.types.is_large_list(ty):
            # Expect List<UInt8>
            for i in range(m):
                if not valid[i]:
                    continue
                try:
                    v = ch[i].as_py()  # usually list[int]
                    if v and len(v) > 0:
                        return True
                except Exception:
                    # last-resort: treat as non-empty if to_pylist works
                    try:
                        if len(ch[i].to_pylist()) > 0:  # type: ignore[attr-defined]
                            return True
                    except Exception:
                        continue
        else:
            continue

    return False


def sample_parquet_bytes(
    *,
    source: str,
    bytes_col: str,
    k: int,
    seed: int,
    head_read_bytes: int,
    thumb_size: int,
    max_side: Optional[int],
) -> Tuple[List[int], List[PreviewItem]]:
    """
    Deterministically sample k rows from a Parquet file and decode thumbnails from a bytes column.
    Efficient: reads only the target column and only the row groups that contain sampled rows.

    Returns
    -------
    idx : list[int]
        Sorted global row indices sampled.
    items : list[PreviewItem]
        One PreviewItem per sampled index, same order as idx.
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(Path(source)))
    names = pf.schema_arrow.names
    if bytes_col not in names:
        return [], []

    # Row group sizes → total rows
    rg_sizes = [
        (pf.metadata.row_group(i).num_rows if pf.metadata is not None else 0)
        for i in range(pf.num_row_groups or 0)
    ]
    total = (
        sum(rg_sizes)
        if rg_sizes
        else (pf.metadata.num_rows if pf.metadata is not None else 0)
    )

    if total <= 0 or k <= 0:
        return [], []

    rng = random.Random(seed)
    k = min(k, total)
    idx = sorted(rng.sample(range(total), k))

    # Map sampled global indices → {rg: [local_indices]}
    # Build cumulative offsets to find rg for each global index
    items: List[PreviewItem] = []
    if rg_sizes:
        # Precompute cumulative starts
        cum = []
        s = 0
        for n in rg_sizes:
            cum.append(s)
            s += n

        by_rg: Dict[int, List[int]] = {}
        for g in idx:
            # binary search to find rg where cum[rg] <= g < cum[rg]+rg_sizes[rg]
            lo, hi = 0, len(rg_sizes) - 1
            rg = 0
            while lo <= hi:
                mid = (lo + hi) // 2
                start = cum[mid]
                end = start + rg_sizes[mid]
                if g < start:
                    hi = mid - 1
                elif g >= end:
                    lo = mid + 1
                else:
                    rg = mid
                    break
            local = g - cum[rg]
            by_rg.setdefault(rg, []).append(local)

        # Read only needed row groups and only the column
        tmp_results: Dict[int, Dict[int, PreviewItem]] = {}
        for rg, locals_ in by_rg.items():
            tbl = pf.read_row_group(rg, columns=[bytes_col])
            col = tbl[bytes_col]
            tmp_map: Dict[int, PreviewItem] = {}
            for li in sorted(set(locals_)):
                b = _extract_cell_bytes(col, li)
                rec = _probe_image_record_bytes(
                    b,
                    thumb_size=thumb_size,
                    head_read_bytes=head_read_bytes,
                    max_side=max_side,
                    pseudo_path=f"parquet:{Path(source).name}#rg={rg}:row={li}",
                )
                tmp_map[li] = rec
            tmp_results[rg] = tmp_map

        # Assemble in the order of idx
        for g in idx:
            # find rg, local again
            lo, hi = 0, len(rg_sizes) - 1
            rg = 0
            while lo <= hi:
                mid = (lo + hi) // 2
                start = cum[mid]
                end = start + rg_sizes[mid]
                if g < start:
                    hi = mid - 1
                elif g >= end:
                    lo = mid + 1
                else:
                    rg = mid
                    break
            local = g - cum[rg]
            items.append(tmp_results[rg][local])

    else:
        # No row groups info; fallback to whole-column read (still column-projected)
        tbl = pf.read(columns=[bytes_col])
        col = tbl[bytes_col]
        for i in idx:
            b = _extract_cell_bytes(col, i)
            rec = _probe_image_record_bytes(
                b,
                thumb_size=thumb_size,
                head_read_bytes=head_read_bytes,
                max_side=max_side,
                pseudo_path=f"parquet:{Path(source).name}#row={i}",
            )
            items.append(rec)

    return idx, items


def _extract_cell_bytes(col, i: int) -> Optional[bytes]:
    """
    Extract bytes from Arrow column at global index i.
    Supports Binary/LargeBinary/FixedSizeBinary, (Large)String, and (Large)List<UInt8>.
    Returns None for null or empty cells.
    """
    import pyarrow as pa
    import base64
    import binascii

    # Locate chunk and local position
    if isinstance(col, pa.ChunkedArray):
        pos = i
        ch = None
        for c in col.chunks:
            if pos < len(c):
                ch, li = c, pos
                break
            pos -= len(c)
        if ch is None:
            return None
    else:
        ch, li = col, i

    # Validity mask
    try:
        valid = ch.is_valid().to_numpy(zero_copy_only=False)
        if not valid[li]:
            return None
    except Exception:
        # Fallback: try scalar validity
        try:
            if ch[li].is_valid is False:  # type: ignore[attr-defined]
                return None
        except Exception:
            pass  # assume valid

    ty = ch.type
    cell = ch[li]

    # Binary-like
    if (
        pa.types.is_binary(ty)
        or pa.types.is_large_binary(ty)
        or pa.types.is_fixed_size_binary(ty)
    ):
        try:
            buf = cell.as_buffer()
            return buf.to_pybytes() if buf is not None else None
        except Exception:
            v = cell.as_py()
            return v if isinstance(v, (bytes, bytearray)) and len(v) > 0 else None

    # String-like (possibly base64)
    if pa.types.is_string(ty) or pa.types.is_large_string(ty):
        s = cell.as_py() or ""
        if not s:
            return None
        try:
            return base64.b64decode(s, validate=True)
        except binascii.Error:
            try:
                b = s.encode("utf-8")
                return b if b else None
            except Exception:
                return None

    # List<UInt8>-like
    if pa.types.is_list(ty) or pa.types.is_large_list(ty):
        try:
            vals = cell.values if hasattr(cell, "values") else None
            if isinstance(vals, pa.Array) and len(vals) > 0:
                try:
                    return bytes(vals.to_numpy(zero_copy_only=False).tolist())
                except Exception:
                    return bytes(vals.to_pylist())
        except Exception:
            try:
                py = cell.as_py()  # list[int]
                return bytes(py) if py else None
            except Exception:
                return None

    return None


def sample_population(
    pop: Population, *, k: int, seed: int
) -> Tuple[List[int], List[str]]:
    """
    Deterministically sample k items from a Population.
    Returns sorted global indices and corresponding paths.

    - Stable for a given (descriptor(), seed, k) assuming pop.size()/path_at are deterministic.
    - k is clamped to [0, pop.size()].
    """
    n = int(pop.size())
    if n < 0:
        raise ValueError("Population size must be >= 0.")
    k = max(0, min(int(k), n))
    if k == 0 or n == 0:
        return [], []

    rng = random.Random(int(seed))
    idx = sorted(rng.sample(range(n), k))
    paths = [pop.path_at(i) for i in idx]
    return idx, paths


# ---------------- Materialize ----------------
def maybe_materialize_grid_bytes(
    items: List[PreviewItem], to_path: Path, cols: int = 3
) -> Optional[str]:
    from PIL import Image

    ok = [it for it in items if it.ok and it.thumb_bytes]
    if not ok:
        return None
    thumbs = [Image.open(io.BytesIO(it.thumb_bytes)).convert("RGB") for it in ok]
    ts = max(t.width for t in thumbs)  # they’re uniform, but keep robust
    cols = max(1, min(cols, len(thumbs)))
    rows = (len(thumbs) + cols - 1) // cols
    grid = Image.new("RGB", (cols * ts, rows * ts))
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        grid.paste(t, (c * ts, r * ts))
    ensure_parent_dir(to_path)
    grid.save(to_path, "PNG", optimize=True)
    return str(to_path.resolve())


def maybe_materialize_thumbs_bytes(
    items: List[PreviewItem], out_dir: Path, seed: int
) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for i, it in enumerate(items):
        if not it.ok or not it.thumb_bytes:
            continue
        fn = out_dir / f"thumb_{seed}_{i}.png"
        fn.write_bytes(it.thumb_bytes)
        written.append(str(fn.resolve()))
    return written


def maybe_materialize_grid(
    items: List[PreviewItem],
    to_path: Path,
    fmt: str,
    jpeg_quality: int,
    thumb_size: int,
    cols: int = 3,
) -> Optional[str]:
    """
    Save a thumbnail grid for OK items that have real readable file paths.
    Returns the written absolute path, or None if nothing was written.
    Skips items from in-memory sources (e.g., pseudo paths like 'parquet:...').
    """
    ok_paths = []
    for it in items:
        # only use items that are ok and point to an existing file
        try:
            p = Path(it.path)
            if it.ok and p.is_file():
                ok_paths.append(p)
        except Exception:
            continue

    if not ok_paths:
        return None

    # Build thumbs
    thumbs: List[Image.Image] = []
    for p in ok_paths:
        try:
            with Image.open(p) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((thumb_size, thumb_size), Image.Resampling.BILINEAR)
                canvas = Image.new("RGB", (thumb_size, thumb_size))
                x = (thumb_size - im.width) // 2
                y = (thumb_size - im.height) // 2
                canvas.paste(im, (x, y))
                thumbs.append(canvas)
        except Exception:
            continue

    if not thumbs:
        return None

    cols = max(1, min(cols, len(thumbs)))
    rows = (len(thumbs) + cols - 1) // cols
    grid = Image.new("RGB", (cols * thumb_size, rows * thumb_size))
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        grid.paste(t, (c * thumb_size, r * thumb_size))

    ensure_parent_dir(to_path)
    ext = fmt.lower()
    if ext in ("jpg", "jpeg"):
        grid.save(to_path, "JPEG", quality=int(jpeg_quality), optimize=True)
    else:
        grid.save(to_path, "PNG", optimize=True)

    return str(to_path.resolve())


def maybe_materialize_thumbs(
    items: List[PreviewItem],
    out_dir: Path,
    fmt: str,
    jpeg_quality: int,
    thumb_size: int,
    seed: int,
) -> List[str]:
    """
    Save individual thumbnails for OK items that point to readable files.
    Skips pseudo-path items (e.g., 'parquet:...') and non-existent files.
    Returns a list of written absolute file paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    ext = "jpg" if fmt.lower() in ("jpg", "jpeg") else "png"

    for i, it in enumerate(items):
        p = Path(it.path)
        if not it.ok or not p.is_file():
            continue
        try:
            with Image.open(p) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((thumb_size, thumb_size), Image.Resampling.BILINEAR)
                fn = out_dir / f"thumb_{seed}_{i}.{ext}"
                if ext == "jpg":
                    im.save(fn, "JPEG", quality=int(jpeg_quality), optimize=True)
                else:
                    im.save(fn, "PNG", optimize=True)
                written.append(str(fn.resolve()))
        except Exception:
            # ignore unreadable paths
            continue

    return written


# ---------------- Meta ----------------
def build_meta(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a standardized meta dict for PreviewResult from a context payload.

    Expected ctx keys (best-effort; missing keys are tolerated):
      - mode: "paths" | "parquet_bytes"
      - source: str
      - source_kind: str
      - pattern: Optional[str]
      - parquet_path_col: Optional[str]
      - bytes_col: Optional[str]
      - path_col: Optional[str]
      - path_root: Optional[str]
      - sample_size_req: int
      - seed: int
      - thumb_size: int
      - items: List[PreviewItem]
      - written: List[str]
      - min_w, min_h, max_aspect: thresholds
      - sample_indices: List[int]
      - population_meta: Optional[Dict[str, Any]]
      - manifest_root_hash / preview_hash: Optional[str]  (when tracked)

    Returns a dict with rolled-up stats and passthrough context.
    """
    items: List[PreviewItem] = ctx.get("items", []) or []
    written: List[str] = ctx.get("written", []) or []

    ok_count = sum(1 for it in items if it.ok)
    bad_count = len(items) - ok_count
    bad_reasons = Counter(it.reason for it in items if not it.ok)
    dupe_map: Dict[str, List[str]] = {}
    for it in items:
        if it.sha256_head:
            dupe_map.setdefault(it.sha256_head, []).append(it.path)
    dupe_candidates = [v for v in dupe_map.values() if len(v) > 1]

    ws = [it.w for it in items if it.ok]
    hs = [it.h for it in items if it.ok]
    means = [it.mean for it in items if it.ok]
    stds = [it.std for it in items if it.ok]

    meta: Dict[str, Any] = {
        "mode": ctx.get("mode"),
        "source_kind": ctx.get("source_kind"),
        "source": ctx.get("source"),
        "pattern": ctx.get("pattern"),
        "parquet_path_col": ctx.get("parquet_path_col"),
        "bytes_col": ctx.get("bytes_col"),
        "path_col": ctx.get("path_col"),
        "path_root": ctx.get("path_root"),
        "sample_size_req": ctx.get("sample_size_req"),
        "sample_size_got": len(items),
        "seed": ctx.get("seed"),
        "thumb_size": ctx.get("thumb_size"),
        "ok_count": ok_count,
        "bad_count": bad_count,
        "bad_reasons": dict(bad_reasons),
        "dupe_candidates": dupe_candidates,
        "size_stats": _pct_stats(ws, hs),
        "intensity_stats": {
            "mean_p50": _percentile(means, 50),
            "std_p50": _percentile(stds, 50),
        },
        "written": written,
        "sample_indices": ctx.get("sample_indices") or [],
        "thresholds": {
            "min_w": ctx.get("min_w"),
            "min_h": ctx.get("min_h"),
            "max_aspect": ctx.get("max_aspect"),
        },
        # passthroughs if present
        "population_meta": ctx.get("population_meta"),
        "manifest_root_hash": ctx.get("manifest_root_hash"),
        "preview_hash": ctx.get("preview_hash"),
    }
    return meta


def _percentile(xs: List[float], p: int) -> float:
    if not xs:
        return 0.0
    if p <= 0:
        return float(min(xs))
    if p >= 100:
        return float(max(xs))
    xs_sorted = sorted(float(x) for x in xs)
    n = len(xs_sorted)
    k = (n - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, n - 1)
    if f == c:
        return xs_sorted[f]
    return xs_sorted[f] * (c - k) + xs_sorted[c] * (k - f)


def _pct_stats(ws: List[int], hs: List[int]) -> Dict[str, Any]:
    if not ws or not hs:
        return {
            "w_min": 0,
            "w_p50": 0.0,
            "w_p95": 0.0,
            "h_min": 0,
            "h_p50": 0.0,
            "h_p95": 0.0,
        }
    ws_f = [float(w) for w in ws]
    hs_f = [float(h) for h in hs]
    return {
        "w_min": int(min(ws_f)),
        "w_p50": _percentile(ws_f, 50),
        "w_p95": _percentile(ws_f, 95),
        "h_min": int(min(hs_f)),
        "h_p50": _percentile(hs_f, 50),
        "h_p95": _percentile(hs_f, 95),
    }


# ---------------- Probe ----------------
Image.MAX_IMAGE_PIXELS = 80_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = False


def _probe_image_record_bytes(
    data: Optional[bytes],
    *,
    thumb_size: int,
    head_read_bytes: int,
    max_side: Optional[int],
    pseudo_path: str,
) -> PreviewItem:
    """
    Inspect one image stored as bytes and return a PreviewItem with integrity stats.
    pseudo_path is a human-readable identifier like 'parquet:file.parquet#row=123'.
    """
    size_bytes = len(data) if data is not None else 0
    sha_head = Hash().hash_bytes((data or b"")[: int(head_read_bytes)]) if data else ""

    if not data:
        return PreviewItem(
            path=pseudo_path,
            ok=False,
            reason="no_bytes",
            w=0,
            h=0,
            ch=0,
            mode="",
            exif_orient=1,
            file_bytes=size_bytes,
            sha256_head=sha_head,
            mean=0.0,
            std=0.0,
            p0=0,
            p100=0,
        )

    try:
        with Image.open(io.BytesIO(data)) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            if max_side and max(im.size) > max_side:
                im.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)
            w, h = im.size

            t = im.copy()
            t.thumbnail((thumb_size, thumb_size), Image.Resampling.BILINEAR)
            arr = np.asarray(t, dtype=np.uint8)

            buf = io.BytesIO()
            t.save(buf, "PNG", optimize=True)
            thumb_png = buf.getvalue()

        mean, std, p0, p100 = _luma_stats(arr)
        return PreviewItem(
            path=pseudo_path,
            ok=True,
            reason="",
            w=w,
            h=h,
            ch=3,
            mode="RGB",
            exif_orient=1,
            file_bytes=size_bytes,
            sha256_head=sha_head,
            mean=float(mean),
            std=float(std),
            p0=int(p0),
            p100=int(p100),
            thumb_bytes=thumb_png,
        )
    except Exception:
        return PreviewItem(
            path=pseudo_path,
            ok=False,
            reason="decode_error_bytes",
            w=0,
            h=0,
            ch=0,
            mode="",
            exif_orient=1,
            file_bytes=size_bytes,
            sha256_head=sha_head,
            mean=0.0,
            std=0.0,
            p0=0,
            p100=0,
        )


def _luma_stats(arr: np.ndarray) -> tuple[float, float, int, int]:
    # RGB uint8 → luminance stats in [0,255]
    if arr.ndim == 3 and arr.shape[2] >= 3:
        r = arr[..., 0].astype(np.float32)
        g = arr[..., 1].astype(np.float32)
        b = arr[..., 2].astype(np.float32)
        y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    else:
        y = arr.astype(np.float32)
    if y.size == 0:
        return 0.0, 0.0, 0, 0
    return float(y.mean()), float(y.std()), int(y.min()), int(y.max())


def probe_paths(
    paths: List[str],
    *,
    thumb_size: int,
    head_read_bytes: int,
    max_side: Optional[int],
    max_total_decode_bytes: Optional[int],
) -> List[PreviewItem]:
    """
    Probe a list of filesystem paths into PreviewItem records.
    Applies a soft memory guard using max_total_decode_bytes ≈ Σ(w*h*3) over thumbs.
    Non-existent paths return decode_error without raising.
    """
    items: List[PreviewItem] = []
    decoded_budget = 0

    for p in paths:
        # cheap existence check to avoid noisy exceptions
        if not p or (not Path(p).exists()):
            items.append(
                PreviewItem(
                    path=str(p),
                    ok=False,
                    reason="missing_file",
                    w=0,
                    h=0,
                    ch=0,
                    mode="",
                    exif_orient=1,
                    file_bytes=0,
                    sha256_head="",
                    mean=0.0,
                    std=0.0,
                    p0=0,
                    p100=0,
                )
            )
            continue

        rec = _probe_image_record(
            path=p,
            thumb_size=thumb_size,
            return_thumb=False,  # stats only; no pixel retention
            head_read_bytes=head_read_bytes,
            max_side=max_side,
        )
        items.append(rec)

        # approximate decoded memory of the thumbnail just processed
        if rec.ok:
            decoded_budget += max(rec.w, 1) * max(rec.h, 1) * 3
            if max_total_decode_bytes and decoded_budget > int(max_total_decode_bytes):
                break

    return items


def _probe_image_record(
    path: str,
    *,
    thumb_size: int,
    return_thumb: bool,  # kept for API parity; thumbnails are not retained
    head_read_bytes: int,
    max_side: Optional[int],
) -> PreviewItem:
    """
    Inspect one filesystem image and return a PreviewItem with integrity stats.
    """
    p = Path(path)
    size_bytes = p.stat().st_size if p.exists() else 0

    # cheap duplicate signal
    sha_head = ""
    try:
        with open(p, "rb") as fh:
            sha_head = Hash().hash_bytes(fh.read(int(head_read_bytes)))
    except Exception:
        sha_head = ""

    if not p.exists() or not p.is_file():
        return PreviewItem(
            path=str(p),
            ok=False,
            reason="missing_file",
            w=0,
            h=0,
            ch=0,
            mode="",
            exif_orient=1,
            file_bytes=size_bytes,
            sha256_head=sha_head,
            mean=0.0,
            std=0.0,
            p0=0,
            p100=0,
        )

    try:
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            if max_side and max(im.size) > max_side:
                im.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)
            w, h = im.size

            t = im.copy()
            t.thumbnail((thumb_size, thumb_size), Image.Resampling.BILINEAR)
            arr = np.asarray(t, dtype=np.uint8)

        # luminance stats
        if arr.ndim == 3 and arr.shape[2] >= 3:
            r, g, b = (
                arr[..., 0].astype(np.float32),
                arr[..., 1].astype(np.float32),
                arr[..., 2].astype(np.float32),
            )
            y = 0.2126 * r + 0.7152 * g + 0.0722 * b
        else:
            y = arr.astype(np.float32)
        mean = float(y.mean()) if y.size else 0.0
        std = float(y.std()) if y.size else 0.0
        p0 = int(y.min()) if y.size else 0
        p100 = int(y.max()) if y.size else 0

        return PreviewItem(
            path=str(p),
            ok=True,
            reason="",
            w=w,
            h=h,
            ch=3,
            mode="RGB",
            exif_orient=1,
            file_bytes=size_bytes,
            sha256_head=sha_head,
            mean=mean,
            std=std,
            p0=p0,
            p100=p100,
        )
    except Exception:
        return PreviewItem(
            path=str(p),
            ok=False,
            reason="decode_error",
            w=0,
            h=0,
            ch=0,
            mode="",
            exif_orient=1,
            file_bytes=size_bytes,
            sha256_head=sha_head,
            mean=0.0,
            std=0.0,
            p0=0,
            p100=0,
        )


# ---------------- Manifest ----------------
def build_manifest_and_hash(
    *, kind: str, context: Dict[str, Any]
) -> Tuple[Dict[str, Any], str]:
    """
    Build a population manifest descriptor and a stable root hash for restore.

    kind:
      - "directory": content-hashed per-file manifest.
      - "parquet": schema+row-count hashed descriptor.
        * If context["parquet_image_bytes_col"] is set → bytes mode.
        * Else path mode (using a path column externally during sampling).

    Returns:
      (manifest_desc: dict, root_hash: str)
    """
    kind = str(kind)
    if kind == "directory":
        src = Path(context["source"]).resolve()
        pattern = context.get("pattern", "**/*")
        recursive = bool(context.get("recursive", True))

        manifest = scan_manifest_headers(
            directory=str(src),
            pattern=pattern,
            recursive=recursive,
            filename_filter=None,
        )
        # content hashes per item
        manifest = ensure_item_content_hashes(manifest)

        root_hash = compute_manifest_root_hash(
            manifest=manifest,
            directory=str(src),
            pattern=pattern,
            recursive=recursive,
            seed=42,  # seed only affects canonical ordering; fixed for identity
            hash_mode="content",
        )
        desc = build_manifest_descriptor(
            manifest=manifest,
            directory=str(src),
            pattern=pattern,
            recursive=recursive,
            seed=42,
            root_hash=root_hash,
            hash_mode="content",
        )
        # annotate population meta
        desc["population_meta"] = {
            "kind": "directory",
            "count": len(manifest),
            "pattern": pattern,
            "recursive": recursive,
        }
        return desc, root_hash

    if kind == "parquet":
        # We avoid full-file hashing. Use schema+row-count+mode as identity.
        try:
            import pyarrow.parquet as pq
        except Exception as e:
            raise RuntimeError("pyarrow is required for parquet manifests.") from e

        src = Path(context["source"]).resolve()
        bytes_col = context.get("parquet_image_bytes_col")
        path_col = context.get("parquet_path_col")
        path_root = context.get("path_root")

        pf = pq.ParquetFile(str(src))
        schema_str = pf.schema_arrow.to_string(show_field_metadata=True)
        nrows = pf.metadata.num_rows if pf.metadata is not None else None
        if nrows is None and pf.num_row_groups:
            nrows = sum(
                pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)
            )  # type: ignore[union-attr]
        nrows = int(nrows or 0)

        mode = "bytes" if bytes_col else "paths"
        ident = {
            "kind": "parquet",
            "source": str(src),
            "mode": mode,
            "schema": schema_str,
            "rows": nrows,
            "bytes_col": bytes_col,
            "path_col": path_col,
            "path_root": path_root,
        }
        blob = json.dumps(ident, ensure_ascii=False, sort_keys=True).encode("utf-8")
        root_hash = Hash().hash_bytes(blob)

        desc = {
            "kind": "table.manifest",
            "version": "1",
            "source": str(src),
            "schema_fp": Hash().hash_bytes(schema_str.encode("utf-8")),
            "rows": nrows,
            "mode": mode,
            "bytes_col": bytes_col,
            "path_col": path_col,
            "path_root": path_root,
            "root_hash": root_hash,
            "population_meta": {"kind": "parquet", "rows": nrows, "mode": mode},
        }
        return desc, root_hash

    # Future kinds: implement similarly with a stable identity
    raise ValueError(f"Unsupported kind for build_manifest_and_hash: {kind!r}")


# ---------------- Register ----------------
def register_tracked_population(stream, manifest_desc: Dict[str, Any]) -> str:
    """
    Store the population manifest in CAS and register it as a step *input*.
    Returns the CAS data_hash.

    Expects a dict like the output of build_manifest_and_hash(...):
      - For directories: kind typically "image.manifest"
      - For parquet:     kind typically "table.manifest"
    """
    kind = str(manifest_desc.get("kind") or "population.manifest")
    version = str(manifest_desc.get("version") or "1")
    payload = json.dumps(manifest_desc, ensure_ascii=False).encode("utf-8")

    manifest_hash = stream.step.register_data(  # type: ignore[attr-defined]
        kind=kind,
        version=version,
        path_or_bytes=payload,
        metadata=manifest_desc,
    )
    stream.step.add_input(manifest_hash, role="population")  # type: ignore[attr-defined]
    return manifest_hash


def register_tracked_preview_manifest(
    stream,
    items: List[PreviewItem],
    meta_core: Dict[str, Any],
) -> str:
    """
    Persist a deterministic preview manifest blob and register as a step output.
    Drops raw binary; records length and sha256 instead.
    """
    safe_items = []
    for it in items:
        d = it.to_dict()
        tb = d.pop("thumb_bytes", None)
        if tb:
            d["thumb_bytes_len"] = len(tb)
            d["thumb_bytes_sha256"] = Hash().hash_bytes(tb)
        safe_items.append(d)

    blob = {"items": safe_items, "meta": dict(meta_core)}
    payload = json.dumps(blob, ensure_ascii=False, sort_keys=True).encode("utf-8")

    data_hash = stream.step.register_data(  # type: ignore[attr-defined]
        kind="preview.manifest",
        version="1",
        path_or_bytes=payload,
        metadata={"count": len(items), **meta_core},
    )
    stream.step.add_output(data_hash, name="manifest")  # type: ignore[attr-defined]
    return data_hash


def register_tracked_materializations(
    stream,
    written_paths: List[str],
) -> None:
    """
    Register any written image files as outputs.
    """
    for p in written_paths:
        pp = Path(p)
        try:
            with pp.open("rb") as fh:
                bh = Hash().hash_bytes(fh.read())
        except Exception:
            bh = ""
        dh = stream.step.register_data(  # type: ignore[attr-defined]
            kind="image",
            version="1",
            path_or_bytes=pp,
            metadata={"sha256": bh, "producer": "blase.Examine.preview_images"},
        )
        stream.step.add_output(dh, name=f"materialization:{pp.name}")  # type: ignore[attr-defined]
