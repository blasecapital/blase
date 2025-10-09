from typing import Iterable, Dict, Any, List
from pathlib import Path
import json

from blase.types import SinkResult, Artifact, Batch
from blase.preparing.interfaces import ManifestBatch, Splits
from blase.preparing.writers.tfrecord.features import make_example
from blase.preparing.writers.tfrecord.shard_plan import plan_shards
from blase.preparing.writers.tfrecord.io import open_writer


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _load_fs_bytes(row):
    p = row.get("path")
    if not p:
        return b""
    try:
        with open(p, "rb") as fh:
            return fh.read()
    except Exception:
        return b""


def _load_parquet_bytes(row, *, id_col, bytes_col):
    import pyarrow.parquet as pq

    shard = row.get("shard_path")
    if not shard:
        return b""
    pf = pq.ParquetFile(shard)
    rgs = [row["rg_id"]] if row.get("rg_id") is not None else range(pf.num_row_groups)
    want = row["image_id"].strip().lower()
    for rg in rgs:
        tbl = pf.read_row_group(rg, columns=[id_col, bytes_col])
        ids = tbl[id_col].to_pylist()
        bys = tbl[bytes_col].to_pylist()
        for iid, v in zip(ids, bys):
            if str(iid).strip().lower() == want:
                if hasattr(v, "as_buffer"):
                    return v.as_buffer().to_pybytes()
                if hasattr(v, "to_pybytes"):
                    return v.to_pybytes()
                return bytes(v) if v is not None else b""
    return b""


def write(
    manifest_iter: Iterable[ManifestBatch],
    splits: Splits,
    cfg: Dict[str, Any],
) -> SinkResult:
    out_dir = Path(cfg["out_dir"])
    _ensure_dir(out_dir)
    mode = cfg["mode"]
    compression = cfg["compression"]
    include_bytes = cfg["include_image_bytes"]
    order_key = cfg["order_key"]
    shard_size_mb = int(cfg["shard_size_mb"])
    write_alignment_index = cfg["write_alignment_index"]
    deterministic_order = cfg.get("deterministic_order", True)

    # Pass 1: collect rows in memory keyed by image_id (baseline; stream-opt later).
    rows: Dict[str, Dict[str, Any]] = {}
    for mb in manifest_iter:
        for r in mb.data:
            rows[r["image_id"]] = r

    # Plan shards deterministically.
    # Use a one-shot manifest from in-memory rows for planning.
    def _one_shot(manifest_iter):
        for mb in manifest_iter:
            # if you merge meta, do it here safely
            meta = mb.meta or {}
            yield Batch(data=mb.data, meta=meta, is_last=mb.is_last)

    shard_map = plan_shards(
        _one_shot(manifest_iter),
        splits,
        shard_size_mb=shard_size_mb,
        order_key=order_key,
    )

    artifacts: List[Artifact] = []
    alignment: Dict[
        str, Dict[str, int]
    ] = {}  # example_id -> {"images": idx, "labels": idx}

    # Write per split
    for sp, shard_rel_paths in shard_map.items():
        ids = [iid for iid in splits.get(sp, []) if iid in rows]
        # sort by order_key deterministically
        if deterministic_order:
            ids.sort(key=lambda iid: str(rows[iid].get(order_key) or iid))
        # chunk by number of shards
        if shard_rel_paths:
            per = max(1, (len(ids) + len(shard_rel_paths) - 1) // len(shard_rel_paths))
        else:
            per = len(ids)

        for si, shard_path_rel in enumerate(
            shard_rel_paths or [Path(f"{sp}-00000.tfrecord")]
        ):
            shard_ids = ids[si * per : (si + 1) * per]
            if not shard_ids:
                continue
            shard_path = out_dir / shard_path_rel
            _ensure_dir(shard_path.parent)
            w = open_writer(shard_path, compression=compression)

            # write examples
            rec_idx = 0
            for iid in shard_ids:
                row = rows[iid]
                if include_bytes and not row.get("_image_bytes"):
                    if cfg.get("read_bytes_from") == "parquet":
                        row["_image_bytes"] = _load_parquet_bytes(
                            row,
                            id_col=cfg.get("parquet_id_col", "image_id"),
                            bytes_col=cfg.get("parquet_bytes_col", "img_bytes"),
                        )
                    else:
                        try:
                            with open(row.get("path", ""), "rb") as fh:
                                row["_image_bytes"] = fh.read()
                        except Exception:
                            row["_image_bytes"] = b""

                    if not row["_image_bytes"]:
                        raise RuntimeError(
                            f"empty image bytes for {row['image_id']} "
                            f"(id_col={cfg.get('parquet_id_col')}, bytes_col={cfg.get('parquet_bytes_col')}, "
                            f"shard={row.get('shard_path')}, rg={row.get('rg_id')})"
                        )
                ex_bytes = make_example(
                    row, mode=mode, include_image_bytes=include_bytes
                )
                w.write(ex_bytes)  # your TF writer must accept dict or serialized bytes
                if write_alignment_index:
                    alignment.setdefault(iid, {})
                    alignment[iid][sp] = rec_idx
                rec_idx += 1
            w.close()
            artifacts.append(
                Artifact(path=str(shard_path), kind="tfrecord", data_hash=None)
            )

    # Optional alignment index for images_labels
    if mode == "images_labels" and write_alignment_index:
        idx_path = out_dir / "alignment_index.json"
        idx_path.write_text(json.dumps(alignment), encoding="utf-8")
        artifacts.append(
            Artifact(path=str(idx_path), kind="alignment_index", data_hash=None)
        )

    return SinkResult(
        is_last=True, meta={"mode": mode, "order_key": order_key}, artifacts=artifacts
    )
