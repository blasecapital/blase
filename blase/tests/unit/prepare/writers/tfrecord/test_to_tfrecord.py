import json
import gzip
from pathlib import Path

import pytest

from blase.prepare import Prepare
from blase.types import Batch


# ---------- helpers ----------


def _manifest_rows():
    # two rows with stable sha256 ordering: "1" < "2"
    return [
        {
            "image_id": "b",
            "height": 2,
            "width": 2,
            "sha256": "2",
            "path": "/p/b.jpg",
            "det": [
                {
                    "xmin": 0.1,
                    "ymin": 0.1,
                    "xmax": 0.2,
                    "ymax": 0.2,
                    "label": "x",
                    "iscrowd": 0,
                }
            ],
            "cls": [],
        },
        {
            "image_id": "a",
            "height": 2,
            "width": 2,
            "sha256": "1",
            "path": "/p/a.jpg",
            "det": [],
            "cls": ["good"],
        },
    ]


def _manifest():
    yield Batch(data=_manifest_rows(), meta={"count": 2}, is_last=True)


# ---------- tests ----------


@pytest.mark.skipif(
    ("tensorflow" not in __import__("sys").modules)
    and (__import__("importlib").util.find_spec("tensorflow") is None),
    reason="TensorFlow not installed",
)
def test_prepare_to_tfrecord_tf_end_to_end_order(tmp_path):
    import tensorflow as tf

    prep = Prepare()
    splits = {"train": ["a", "b"], "val": [], "test": []}

    out = prep.to_tfrecord(
        manifest=_manifest(),
        splits=splits,
        out_dir=tmp_path,
        mode="combined",
        shard_size_mb=32,
        compression="NONE",
        include_image_bytes=False,
        deterministic_order=True,
        order_key="sha256",
    )

    # one train shard written
    train_files = [
        Path(a.path) for a in out.artifacts if Path(a.path).name.startswith("train-")
    ]
    assert train_files, "no train shard written"

    # read TFRecord and assert order by sha256: 'a' (1) then 'b' (2)
    ds = tf.data.TFRecordDataset([str(train_files[0])], compression_type="")
    ids = []
    for raw in ds:
        ex = tf.train.Example()
        ex.ParseFromString(bytes(raw.numpy()))
        ids.append(ex.features.feature["example/id"].bytes_list.value[0].decode())
    assert ids == ["a", "b"]


def test_prepare_to_tfrecord_fallback_jsonl_when_no_tf(tmp_path, monkeypatch):
    # force JSONL fallback
    monkeypatch.setitem(__import__("sys").modules, "tensorflow", None)

    prep = Prepare()
    splits = {"train": ["a", "b"], "val": [], "test": []}

    out = prep.to_tfrecord(
        manifest=_manifest(),
        splits=splits,
        out_dir=tmp_path,
        mode="combined",
        shard_size_mb=32,
        compression="GZIP",
        include_image_bytes=False,
        deterministic_order=True,
        order_key="sha256",
    )

    train_gz = [
        Path(a.path) for a in out.artifacts if Path(a.path).name.startswith("train-")
    ]
    assert train_gz
    with gzip.open(train_gz[0], "rt", encoding="utf-8") as fh:
        lines = [json.loads(line) for line in fh if line.strip()]
    # two examples written
    assert len(lines) == 2


def test_prepare_to_tfrecord_images_labels_alignment_index(tmp_path, monkeypatch):
    # fallback ok for this test
    monkeypatch.setitem(__import__("sys").modules, "tensorflow", None)

    prep = Prepare()
    splits = {"train": ["a", "b"], "val": [], "test": []}

    out = prep.to_tfrecord(
        manifest=_manifest(),
        splits=splits,
        out_dir=tmp_path,
        mode="images_labels",
        shard_size_mb=16,
        compression="GZIP",
        include_image_bytes=False,
        deterministic_order=True,
        order_key="sha256",
        write_alignment_index=True,
    )

    idx_path = Path(tmp_path) / "alignment_index.json"
    assert any(Path(a.path) == idx_path for a in out.artifacts), (
        "alignment_index.json not recorded"
    )
    data = json.loads(idx_path.read_text(encoding="utf-8"))
    # both ids present in index
    assert set(data.keys()) == {"a", "b"}
