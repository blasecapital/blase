from pathlib import Path

import pyarrow.parquet as pq

from blase.prepare import Prepare
from blase.types import Batch


def _manifest_rows():
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
            "cls": [b"bad"],
        },  # bytes label tests schema flexibility
        {
            "image_id": "a",
            "path": "/p/a.jpg",
            "height": 2,
            "width": 2,
            "sha256": "1",
            "det": [],
            "cls": [b"good"],
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


def test_write_label_sidecars_parquet_schema_and_order(tmp_path):
    prep = Prepare()
    splits = {"train": ["a", "b", "c"]}

    res = prep.write_label_sidecars(
        manifest=_manifest_iter(),
        splits=splits,
        out_dir=tmp_path,
        format="parquet",
        deterministic_order=True,
        order_key="sha256",
    )
    path = next(Path(a.path) for a in res.artifacts if a.kind == "sidecar_parquet")
    assert path.exists()

    tbl = pq.ParquetFile(path).read()
    cols = tbl.column_names
    assert set(
        ["image_id", "path", "sha256", "height", "width", "det", "cls"]
    ).issubset(cols)

    # order by sha256 => a, b, c
    ids = tbl.column("image_id").to_pylist()
    assert ids == ["a", "b", "c"]

    # det is list<struct>, first row has one box with fields
    det_col = tbl.column("det").to_pylist()
    idx_b = ids.index("b")
    assert isinstance(det_col[idx_b], list) and len(det_col[idx_b]) == 1
    box0 = det_col[idx_b][0]
    for k in ("xmin", "ymin", "xmax", "ymax", "label", "iscrowd"):
        assert k in box0


def test_write_label_sidecars_parquet_preserve_split_order_when_nondeterministic(
    tmp_path,
):
    prep = Prepare()
    splits = {"train": ["c", "b", "a"]}

    path = tmp_path / "train.labels.parquet"
    prep.write_label_sidecars(
        manifest=_manifest_iter(),
        splits=splits,
        out_dir=tmp_path,
        format="parquet",
        deterministic_order=False,
    )
    tbl = pq.ParquetFile(path).read()
    ids = tbl.column("image_id").to_pylist()
    assert ids == ["c", "b", "a"]
