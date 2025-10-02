from typing import Iterable, Dict, Any, List
from pathlib import Path
import json

from blase.types import SinkResult, Artifact
from blase.preparing.interfaces import ManifestBatch, Splits
from blase.preparing.writers.tfrecord.features import make_example
from blase.preparing.writers.tfrecord.shard_plan import plan_shards
from blase.preparing.writers.tfrecord.io import open_writer


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


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

    # Pass 1: collect rows in memory keyed by image_id (baseline; stream-opt later).
    rows: Dict[str, Dict[str, Any]] = {}
    for mb in manifest_iter:
        for r in mb.data:
            rows[r["image_id"]] = r

    # Plan shards deterministically.
    # Use a one-shot manifest from in-memory rows for planning.
    def _one_shot():
        yield type(mb)(
            data=list(rows.values()), meta={"count": len(rows)}, is_last=True
        )  # reusing Batch class

    shard_map = plan_shards(
        _one_shot(), splits, shard_size_mb=shard_size_mb, order_key=order_key
    )

    artifacts: List[Artifact] = []
    alignment: Dict[
        str, Dict[str, int]
    ] = {}  # example_id -> {"images": idx, "labels": idx}

    # Write per split
    for sp, shard_rel_paths in shard_map.items():
        ids = [iid for iid in splits.get(sp, []) if iid in rows]
        # sort by order_key deterministically
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
