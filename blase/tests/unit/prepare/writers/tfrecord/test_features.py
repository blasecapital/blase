import pytest
from blase.preparing.writers.tfrecord.features import make_example

_tf = pytest.importorskip("tensorflow")


def test_make_example_serializes_and_roundtrips():
    row = {
        "image_id": "a",
        "height": 10,
        "width": 20,
        "sha256": "abc",
        "path": "/tmp/a.jpg",
        "det": [
            {
                "xmin": 0.1,
                "ymin": 0.2,
                "xmax": 0.3,
                "ymax": 0.4,
                "label": "flower",
                "iscrowd": 0,
            },
            {
                "xmin": 0.5,
                "ymin": 0.6,
                "xmax": 0.7,
                "ymax": 0.8,
                "label": "weed",
                "iscrowd": 1,
            },
        ],
        "cls": ["flower"],
        "_image_bytes": b"\x00\x01",  # tiny stub
    }
    b = make_example(row, mode="combined", include_image_bytes=True)
    ex = _tf.train.Example()
    ex.ParseFromString(b)
    f = ex.features.feature
    assert f["example/id"].bytes_list.value[0] == b"a"
    assert f["image/height"].int64_list.value[0] == 10
    assert f["image/width"].int64_list.value[0] == 20
    assert len(f["bbox/xmin"].float_list.value) == 2
    assert len(f["bbox/label"].bytes_list.value) == 2
    assert f["cls/labels"].bytes_list.value[0] == b"flower"
