import json
import gzip
from pathlib import Path

import pytest

from blase.prepare import Prepare
from blase.types import Batch


# helper manifest
def _manifest_rows():
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


@pytest.mark.skipif(
    "tensorflow" not in __import__("sys").modules
    and __import__("importlib").util.find_spec("tensorflow") is None,
    reason="TF not installed",
)
def test_writer_tf_end_to_end(tmp_path):
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
    # one artifact produced
    paths = [Path(a.path) for a in out.artifacts]
    train_files = [p for p in paths if p.name.startswith("train-")]
    assert train_files, "no train shard written"
    # read and assert order by sha256: 'a' (1) then 'b' (2)
    import tensorflow as tf

    ds = tf.data.TFRecordDataset([str(train_files[0])], compression_type="")
    ids = []
    for raw in ds:
        ex = tf.train.Example()
        ex.ParseFromString(bytes(raw.numpy()))
        ids.append(ex.features.feature["example/id"].bytes_list.value[0].decode())
    assert ids == ["a", "b"]


def test_writer_jsonl_fallback_when_no_tf(tmp_path, monkeypatch):
    # force fallback by masking tensorflow
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
    # read gzip jsonl fallback and confirm two lines
    with gzip.open(train_gz[0], "rt", encoding="utf-8") as fh:
        lines = [json.loads(line) for line in fh if line.strip()]
    assert len(lines) == 2
