import json
import gzip

import pytest
from blase.preparing.writers.tfrecord.io import open_writer


def test_open_writer_jsonl_fallback_when_no_tf(tmp_path, monkeypatch):
    # Force ImportError for tensorflow to trigger JSONL fallback
    monkeypatch.setitem(__import__("sys").modules, "tensorflow", None)
    p = tmp_path / "x.tfrecord"
    w = open_writer(p, compression="GZIP")
    w.write({"k": 1})
    w.write({"k": 2})
    w.close()
    assert p.exists()
    # Read gzip jsonl
    with gzip.open(p, "rt", encoding="utf-8") as fh:
        lines = [json.loads(line) for line in fh if line.strip()]
    assert [line["k"] for line in lines] == [1, 2]


@pytest.mark.skipif(
    "tensorflow" not in __import__("sys").modules
    and __import__("importlib").util.find_spec("tensorflow") is None,
    reason="TF not installed",
)
def test_open_writer_tf_backend(tmp_path):
    import tensorflow as tf

    p = tmp_path / "y.tfrecord"
    w = open_writer(p, compression="NONE")
    # write one minimal serialized Example
    ex = tf.train.Example()
    ex.features.feature["x"].int64_list.value.append(7)
    w.write(ex.SerializeToString())
    w.close()
    # read back
    ds = tf.data.TFRecordDataset([str(p)], compression_type="")
    got = []
    for raw in ds:
        e = tf.train.Example()
        e.ParseFromString(bytes(raw.numpy()))
        got.append(e.features.feature["x"].int64_list.value[0])
    assert got == [7]
