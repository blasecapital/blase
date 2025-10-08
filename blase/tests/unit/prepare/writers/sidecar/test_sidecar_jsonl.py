import json
from pathlib import Path

from blase.prepare import Prepare
from blase.types import Batch


def _manifest_rows():
    # sha256 chosen to test default order_key="sha256"
    return [
        {
            "image_id": "b",
            "path": "/p/b.jpg",
            "height": 2,
            "width": 2,
            "sha256": "2",
            "det": [
                {
                    "xmin": 0.1,
                    "ymin": 0.2,
                    "xmax": 0.3,
                    "ymax": 0.4,
                    "label": "x",
                    "iscrowd": 0,
                }
            ],
            "cls": ["bad"],
        },
        {
            "image_id": "a",
            "path": "/p/a.jpg",
            "height": 2,
            "width": 2,
            "sha256": "1",
            "det": [],
            "cls": ["good"],
        },
        {
            "image_id": "c",
            "path": "/p/c.jpg",
            "height": 2,
            "width": 2,
            "sha256": "3",
            "det": [],
            "cls": [],
        },
    ]


def _manifest_iter():
    yield Batch(data=_manifest_rows(), meta={"count": 3}, is_last=True)


def test_write_label_sidecars_jsonl_per_split_and_order(tmp_path):
    prep = Prepare()
    splits = {"train": ["a", "b"], "val": ["c"], "test": []}

    res = prep.write_label_sidecars(
        manifest=_manifest_iter(),
        splits=splits,
        out_dir=tmp_path,
        format="jsonl",
        deterministic_order=True,  # should sort by sha256 => a(1), b(2)
        order_key="sha256",
    )
    # artifacts include train and val jsonl files
    paths = [Path(a.path) for a in res.artifacts]
    train_path = next(p for p in paths if p.name == "train.labels.jsonl")
    val_path = next(p for p in paths if p.name == "val.labels.jsonl")
    assert train_path.exists() and val_path.exists()

    # check order in train
    with train_path.open("r", encoding="utf-8") as fh:
        lines = [json.loads(line) for line in fh if line.strip()]
    ids = [r["image_id"] for r in lines]
    assert ids == ["a", "b"]  # sorted by sha256 values 1 < 2

    # val contains c only
    with val_path.open("r", encoding="utf-8") as fh:
        lines = [json.loads(line) for line in fh if line.strip()]
    assert [r["image_id"] for r in lines] == ["c"]


def test_write_label_sidecars_jsonl_respects_input_order_when_nondeterministic(
    tmp_path,
):
    prep = Prepare()
    # split order is b then a; with deterministic_order=False we expect this preserved
    splits = {"train": ["b", "a"]}

    train_path = tmp_path / "train.labels.jsonl"
    _ = prep.write_label_sidecars(
        manifest=_manifest_iter(),
        splits=splits,
        out_dir=tmp_path,
        format="jsonl",
        deterministic_order=False,  # keep split order
        order_key="sha256",
    )
    assert train_path.exists()
    with train_path.open("r", encoding="utf-8") as fh:
        ids = [json.loads(line)["image_id"] for line in fh if line.strip()]
    assert ids == ["b", "a"]


def test_write_label_sidecars_jsonl_filters_missing_ids(tmp_path):
    prep = Prepare()
    # 'z' does not exist in manifest
    splits = {"train": ["z", "a"]}

    path = tmp_path / "train.labels.jsonl"
    prep.write_label_sidecars(
        manifest=_manifest_iter(),
        splits=splits,
        out_dir=tmp_path,
        format="jsonl",
    )
    with path.open("r", encoding="utf-8") as fh:
        ids = [json.loads(line)["image_id"] for line in fh if line.strip()]
    assert ids == ["a"]
