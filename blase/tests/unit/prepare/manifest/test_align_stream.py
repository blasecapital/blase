from itertools import chain
from typing import Iterable, Dict, Any

from blase.preparing.manifest.align_stream import align_stream
from blase.types import Batch


def _labels_many(n: int, unique_ids: int) -> Iterable[Dict[str, Any]]:
    # Stream n detection rows cycling across unique_ids image_ids
    for i in range(n):
        yield {
            "image_id": f"id{i % unique_ids}",
            "bbox": [0.0, 0.0, 1.0, 1.0],
            "class": "x",
            "iscrowd": 0,
        }


def test_align_stream_joins_labels_and_emits_batches():
    # minimal KV index
    kv = {
        "a": {
            "image_id": "a",
            "path": "a.jpg",
            "height": 2,
            "width": 4,
            "sha256": "aa",
        },
        "b": {
            "image_id": "b",
            "path": "b.jpg",
            "height": 4,
            "width": 8,
            "sha256": "bb",
        },
    }
    # two label iterators
    det_rows = [
        {"image_id": "a", "bbox": [0, 0, 2, 1], "class": "weed", "iscrowd": 0},
        {"image_id": "b", "bbox": [1, 1, 2, 2], "class": "radish", "iscrowd": 0},
    ]
    cls_rows = [{"image_id": "a", "class": "good"}, {"image_id": "b", "class": "bad"}]

    it = align_stream(
        label_iters=[iter(det_rows), iter(cls_rows)],
        kv_index=kv,
        rg_index=None,
        join_cfg={"drop_orphans": True, "keep_unlabeled_images": False},
        box_cfg={
            "coord_in": "xyxy_abs",
            "coord_out": "xyxy_rel",
            "clamp_boxes": True,
            "drop_oob_boxes": False,
        },
        class_cfg={"class_map": None, "normalize_names": True},
        scale_cfg={"batch_rows": 2},
    )

    batches = list(it)
    assert len(batches) >= 1
    # flatten rows
    rows = list(chain.from_iterable(b.data for b in batches))
    by_id = {r["image_id"]: r for r in rows}
    assert "det" in by_id["a"] and "cls" in by_id["a"]
    assert len(by_id["a"]["det"]) == 1


def test_align_stream_emits_bounded_batches_then_breaks_early():
    unique_ids = 50
    n_labels = 50_000  # large stream
    kv = {
        f"id{i}": {
            "image_id": f"id{i}",
            "path": f"{i}.jpg",
            "height": 2,
            "width": 2,
            "sha256": None,
        }
        for i in range(unique_ids)
    }

    it = align_stream(
        label_iters=[_labels_many(n_labels, unique_ids)],
        kv_index=kv,
        rg_index=None,
        join_cfg={"drop_orphans": True, "keep_unlabeled_images": False},
        box_cfg={
            "coord_in": "xywh_abs",
            "coord_out": "xyxy_rel",
            "clamp_boxes": True,
            "drop_oob_boxes": False,
        },
        class_cfg={"class_map": None, "normalize_names": True},
        scale_cfg={"batch_rows": 256},  # enforce small batches
    )

    # Consume only a few batches to simulate early stop and ensure bounded size
    consumed = 0
    for b in it:
        assert isinstance(b, Batch)
        assert len(b.data) <= 256
        consumed += 1
        if consumed >= 5:
            break

    # If we reached here without OOM or huge batches, streaming works
    assert consumed == 5
