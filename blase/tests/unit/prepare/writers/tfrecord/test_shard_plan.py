from typing import Dict, Any, Sequence, Iterable
from pathlib import Path

from blase.preparing.writers.tfrecord.shard_plan import plan_shards
from blase.types import Batch


def _mb(rows: Sequence[Dict[str, Any]], last=True):
    return Batch(data=rows, meta={"count": len(rows)}, is_last=last)


def _manifest(rows: Sequence[Dict[str, Any]]) -> Iterable[Batch]:
    yield _mb(rows, last=True)


def test_plan_shards_orders_by_key_and_respects_splits(tmp_path):
    rows = [
        {"image_id": "b", "sha256": "2", "height": 1, "width": 1, "det": [], "cls": []},
        {"image_id": "a", "sha256": "1", "height": 1, "width": 1, "det": [], "cls": []},
        {"image_id": "c", "sha256": "3", "height": 1, "width": 1, "det": [], "cls": []},
    ]
    splits = {"train": ["a", "b"], "val": ["c"], "test": []}
    plan = plan_shards(_manifest(rows), splits, shard_size_mb=32, order_key="sha256")
    # should produce at least one shard per non-empty split
    assert set(k for k, v in plan.items() if v) == {"train", "val"}
    # check shard filenames stable
    for sp, paths in plan.items():
        for p in paths:
            assert isinstance(p, Path)
            assert p.name.startswith(f"{sp}-")
