import glob
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple, Optional

import pyarrow as pa
import pyarrow.parquet as pq


def build_image_index(
    data_sources: Sequence[Dict[str, Any]],
    image_cfg: Dict[str, Any],
    scale_cfg: Dict[str, Any],
) -> Tuple[Mapping[str, Dict[str, Any]], Any, Any]:
    """
    Returns (kv_index, bloom_or_none, rowgroup_index_or_none).

    kv_index: image_id -> {
        "image_id": str,
        "path": str,
        "height": int,
        "width": int,
        "sha256": Optional[str],
        "shard_path": str,
        "rg_id": Optional[int],     # row-group id if determinable, else None
    }

    Notes
    -----
    - Minimal, streaming implementation. Parquet-only.
    - Respects batch size via scale_cfg["rows_per_chunk"] if provided.
    - Does not build a bloom filter yet (returns None).
    - Row-group index is not constructed here (returns None).
    """
    id_col = image_cfg.get("id_col", "image_id")
    path_col = image_cfg.get("path_col", "path")
    height_col = image_cfg.get("height_col", "height")
    width_col = image_cfg.get("width_col", "width")
    sha256_col = image_cfg.get("sha256_col")  # optional

    rows_per_chunk = int(scale_cfg.get("rows_per_chunk", 100_000))

    kv: Dict[str, Dict[str, Any]] = {}

    for ds in data_sources or ():
        fmt = ds.get("fmt", "parquet")
        if fmt != "parquet":
            raise ValueError(
                f"index_images: unsupported fmt={fmt!r}; only 'parquet' is implemented"
            )

        uri = str(ds["uri"])
        for shard in _expand_glob(uri):
            _ingest_parquet_shard(
                shard,
                kv,
                id_col=id_col,
                path_col=path_col,
                height_col=height_col,
                width_col=width_col,
                sha256_col=sha256_col,
                batch_rows=rows_per_chunk,
            )

    # No bloom or row-group index in this minimal pass
    bloom = None
    rowgroup_index = None
    return kv, bloom, rowgroup_index


# ---- helpers ----------------------------------------------------------------


def _expand_glob(pat: str) -> Iterable[str]:
    # Support both explicit file and glob pattern
    p = Path(pat)
    if p.exists():
        return [str(p)]
    return sorted(glob.glob(pat))


def _ingest_parquet_shard(
    shard_path: str,
    kv: Dict[str, Dict[str, Any]],
    *,
    id_col: str,
    path_col: str,
    height_col: str,
    width_col: str,
    sha256_col: Optional[str] = None,
    batch_rows: int,
) -> None:
    pf = pq.ParquetFile(shard_path)
    # Restrict columns to what we need
    cols = [c for c in (id_col, path_col, height_col, width_col, sha256_col) if c]
    for rg_i in range(pf.num_row_groups):
        # Iterate by row group to minimize memory
        batch_reader = pf.iter_batches(
            row_groups=[rg_i], batch_size=batch_rows, columns=cols
        )
        for rec_batch in batch_reader:
            # Convert to pyarrow.Table for easy column access if needed
            if isinstance(rec_batch, pa.RecordBatch):
                tbl = pa.Table.from_batches([rec_batch])
            else:
                tbl = rec_batch  # already a Table

            col_id = tbl.column(id_col)
            col_path = tbl.column(path_col)
            col_h = tbl.column(height_col)
            col_w = tbl.column(width_col)
            col_sha = tbl.column(sha256_col) if sha256_col else None

            n = tbl.num_rows
            # Fast path: use to_pylist on each column once
            ids = col_id.to_pylist()
            paths = col_path.to_pylist()
            hs = col_h.to_pylist()
            ws = col_w.to_pylist()
            shas = col_sha.to_pylist() if col_sha is not None else [None] * n

            for i in range(n):
                image_id = str(ids[i])
                if not image_id:
                    continue  # skip empty ids
                # First-write wins; duplicates can be handled later per JoinConfig if desired
                if image_id not in kv:
                    kv[image_id] = {
                        "image_id": image_id,
                        "path": paths[i],
                        "height": int(hs[i]) if hs[i] is not None else None,
                        "width": int(ws[i]) if ws[i] is not None else None,
                        "sha256": shas[i],
                        "shard_path": shard_path,
                        "rg_id": rg_i,
                    }
