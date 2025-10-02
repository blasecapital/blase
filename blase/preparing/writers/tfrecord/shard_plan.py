from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence
from blase.preparing.interfaces import ManifestBatch


def plan_shards(
    manifest_iter: Iterable[ManifestBatch],
    splits: Mapping[str, Sequence[str]],
    *,
    shard_size_mb: int,
    order_key: str,
) -> Dict[str, Sequence[Path]]: ...
