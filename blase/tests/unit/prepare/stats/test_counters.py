from blase.types import Batch
from blase.prepare import Prepare
from typing import Dict, Any, Sequence


def _mb(rows: Sequence[Dict[str, Any]], last=False):
    return Batch(data=rows, meta={"count": len(rows)}, is_last=last)


def _manifest():
    rows1 = [
        {
            "image_id": "a",
            "height": 2,
            "width": 4,
            "det": [{"label": "weed"}],
            "cls": ["good"],
        },
        {
            "image_id": "b",
            "height": 4,
            "width": 8,
            "det": [{"label": "radish"}, {"label": "weed"}],
            "cls": [],
        },
    ]
    yield _mb(rows1, last=True)


def test_stats_class():
    prep = Prepare()
    b = prep.compute_stats(_manifest(), by="class")
    h = b.data["class_hist"]
    assert b.data["by"] == "class"
    assert b.data["n_images"] == 2
    assert h == {"good": 1, "radish": 1, "weed": 2}


def test_stats_image():
    prep = Prepare()
    b = prep.compute_stats(_manifest(), by="image")
    imgs = {r["image_id"]: r for r in b.data["images"]}
    assert imgs["a"]["n_det"] == 1 and imgs["a"]["n_cls"] == 1
    assert imgs["b"]["n_det"] == 2 and imgs["b"]["n_cls"] == 0


def test_stats_global():
    prep = Prepare()
    b = prep.compute_stats(_manifest(), by="global")
    assert b.data["n_images"] == 2
    assert b.data["n_det"] == 3
    assert b.data["n_cls"] == 1
    assert b.data["class_hist"]["weed"] == 2
