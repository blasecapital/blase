from typing import Dict, Iterable, Mapping, Sequence, List, Tuple
from pathlib import Path
from blase.preparing.interfaces import ManifestBatch


def plan_shards(
    manifest_iter: Iterable[ManifestBatch],
    splits: Mapping[str, Sequence[str]],
    *,
    shard_size_mb: int,
    order_key: str,
) -> Dict[str, Sequence[Path]]:
    # First pass: collect lightweight (key, image_id) for ordering and sizing.
    order: Dict[str, Tuple[str, str]] = {}  # image_id -> (sort_key, split)
    for mb in manifest_iter:
        for r in mb.data:
            iid = r["image_id"]
            if iid not in order:
                split = (
                    "train"
                    if iid in set(splits.get("train", ()))
                    else "val"
                    if iid in set(splits.get("val", ()))
                    else "test"
                    if iid in set(splits.get("test", ()))
                    else None
                )
                if split is None:  # skip rows not in any split
                    continue
                key = str(r.get(order_key) or iid)
                order[iid] = (key, split)

    # Sort per split
    per_split: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    for sp in ("train", "val", "test"):
        ids = [iid for iid, (k, s) in order.items() if s == sp]
        ids.sort(key=lambda iid: order[iid][0])
        per_split[sp] = ids

    # Chunk into shard paths; simple count-based target (≈1MB/example with bytes, else ≈0.01MB).
    avg_mb = 1.0 if shard_size_mb >= 32 else 0.5  # crude heuristic; refine later
    examples_per_shard = max(1, int(shard_size_mb / max(0.01, avg_mb)))

    out: Dict[str, List[Path]] = {"train": [], "val": [], "test": []}
    for sp, ids in per_split.items():
        n = len(ids)
        if n == 0:
            continue
        num_shards = max(1, (n + examples_per_shard - 1) // examples_per_shard)
        for i in range(num_shards):
            out[sp].append(Path(f"{sp}-{i:05d}.tfrecord"))
    return out
