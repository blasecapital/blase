import json
import gzip
from pathlib import Path

import pytest

from blase.preparing.writers.tfrecord.inspect import head

# -------------------------
# Helpers
# -------------------------


def _write_jsonl_gz(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _make_tf_example_bytes():
    import tensorflow as tf

    def _b(x):
        return x if isinstance(x, (bytes, bytearray)) else str(x).encode()

    feats = {
        "example/id": tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"a"])),
        "image/height": tf.train.Feature(int64_list=tf.train.Int64List(value=[10])),
        "image/width": tf.train.Feature(int64_list=tf.train.Int64List(value=[20])),
        "bbox/xmin": tf.train.Feature(float_list=tf.train.FloatList(value=[0.1])),
        "bbox/ymin": tf.train.Feature(float_list=tf.train.FloatList(value=[0.2])),
        "bbox/xmax": tf.train.Feature(float_list=tf.train.FloatList(value=[0.3])),
        "bbox/ymax": tf.train.Feature(float_list=tf.train.FloatList(value=[0.4])),
        "cls/labels": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[b"flower"])
        ),
        # 1x1 PNG to keep tiny
        "image/encoded": tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[tf.io.encode_png(tf.ones([1, 1, 3], tf.uint8) * 255).numpy()]
            )
        ),
    }
    ex = tf.train.Example(features=tf.train.Features(feature=feats))
    return ex.SerializeToString()


# -------------------------
# Tests: TensorFlow backend
# -------------------------


@pytest.mark.skipif(
    (__import__("importlib").util.find_spec("tensorflow") is None),
    reason="TensorFlow not installed",
)
def test_head_tf_record_parses_and_optionally_decodes(tmp_path):
    import tensorflow as tf

    rec_path = tmp_path / "train-00000.tfrecord"
    # write one example
    with tf.io.TFRecordWriter(str(rec_path)) as w:
        w.write(_make_tf_example_bytes())

    # Summary without decode
    out = head(rec_path, n=1, compression="NONE", decode_images=False)
    assert len(out) == 1
    o = out[0]
    assert o["example/id"] == "a"
    assert o["H"] == 10 and o["W"] == 20
    assert o["has_bytes"] is True
    assert o["n_det"] == 1 and o["n_cls"] == 1
    assert o["preview_path"] is None

    # With decode + persist
    prev_dir = tmp_path / "previews"
    out2 = head(
        rec_path,
        n=1,
        compression="NONE",
        decode_images=True,
        out_dir=prev_dir,
        max_side=64,
    )
    assert len(out2) == 1
    p = out2[0]["preview_path"]
    assert p is not None and Path(p).exists()
    assert Path(p).suffix == ".png"


# -------------------------
# Tests: JSONL fallback (no TF)
# -------------------------


def test_head_jsonl_fallback_reads_gzip_summary(tmp_path, monkeypatch):
    # Force fallback path
    monkeypatch.setitem(__import__("sys").modules, "tensorflow", None)

    p = tmp_path / "train-00000.tfrecord"
    rows = [
        {
            "example/id": "x",
            "image/height": 5,
            "image/width": 7,
            "bbox/xmin": [0.1],
            "cls/labels": ["weed"],
            # no image/encoded in fallback
        },
        {"example/id": "y", "image/height": 9, "image/width": 11},
    ]
    _write_jsonl_gz(p, rows)

    out = head(p, n=1, compression="GZIP", decode_images=False)
    assert len(out) == 1
    o = out[0]
    assert o["example/id"] == "x"
    assert o["H"] == 5 and o["W"] == 7
    assert o["has_bytes"] is False
    assert o["n_det"] == 1
    assert o["n_cls"] == 1
    assert o["preview_path"] is None
