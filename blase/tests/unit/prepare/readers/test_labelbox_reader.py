# tests/unit/preparing/readers/test_labelbox_reader.py

import json
import gzip
from pathlib import Path

import pytest

# Under test
from blase.preparing.readers import labelbox_jsonl as lb_reader


def _record(
    *,
    external_id="IMG_1302.jpg",
    row_data="https://storage.example.com/bucket/IMG_1302.jpg?sig=abc",
    top=1954.0,
    left=2714.0,
    height=392.0,
    width=413.0,
    cls="flower",
    project_id="proj_123",
    with_cls=False,
):
    base = {
        "data_row": {
            "id": "cmco2y8x61jzp0763eeai4zzm",
            "external_id": external_id,
            "row_data": row_data,
        },
        "media_attributes": {
            "height": 4284,
            "width": 4284,
            "asset_type": "image",
            "mime_type": "image/jpeg",
        },
        "metadata_fields": [],
        "projects": {
            project_id: {
                "name": "deep_flowers",
                "labels": [
                    {
                        "label_kind": "Default",
                        "version": "1.0.0",
                        "id": "label_1",
                        "annotations": {
                            "objects": [
                                {
                                    "feature_id": "feat_1",
                                    "name": cls,
                                    "value": cls,
                                    "annotation_kind": "ImageBoundingBox",
                                    "classifications": [],
                                    "bounding_box": {
                                        "top": top,
                                        "left": left,
                                        "height": height,
                                        "width": width,
                                    },
                                }
                            ],
                            "classifications": [],
                            "relationships": [],
                        },
                    }
                ],
                "project_tags": [],
            }
        },
    }
    if with_cls:
        base["projects"][project_id]["labels"][0]["annotations"]["classifications"] = [
            {"name": "species", "answer": {"value": "rose"}},
            {"name": "colors", "answers": [{"value": "red"}, {"value": "white"}]},
        ]
    return base


def test_labelbox_reader_json_single_record_external_stem(tmp_path: Path):
    p = tmp_path / "labels.json"
    rec = _record(external_id="IMG_1302.jpg")
    p.write_text(json.dumps(rec))

    rows = list(
        lb_reader.read(
            {
                "uri": str(p),
                "fmt": "labelbox",
                "options": {"id_from": "external_stem", "normalize_names": True},
            }
        )
    )

    assert len(rows) == 1
    r = rows[0]
    assert (
        r["image_id"] == "IMG_1302".lower()
        or r["image_id"] == "IMG_1302"
        or r["image_id"] == "img_1302"
    )
    assert r["bbox"] == [
        pytest.approx(2714.0),
        pytest.approx(1954.0),
        pytest.approx(413.0),
        pytest.approx(392.0),
    ]
    assert r["class"] in {"flower", "FLOWER", "Flower", "flower".lower()}
    assert r["iscrowd"] == 0


def test_labelbox_reader_jsonl_gz_row_data_basename_filter_project(tmp_path: Path):
    p = tmp_path / "labels.jsonl.gz"
    rec1 = _record(
        external_id=None,
        row_data="https://host/x/y/IMG_7777.jpg?token=z",
        project_id="keep_me",
        cls="Pink Rose",
    )
    rec2 = _record(
        external_id=None,
        row_data="https://host/x/y/IMG_9999.jpg?token=z",
        project_id="drop_me",
        cls="ShouldNotAppear",
    )

    with gzip.open(p, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps(rec1) + "\n")
        fh.write(json.dumps(rec2) + "\n")

    rows = list(
        lb_reader.read(
            {
                "uri": str(p),
                "fmt": "labelbox",
                "options": {
                    "id_from": "row_data_basename",
                    "project_id": "keep_me",
                    "normalize_names": True,
                },
            }
        )
    )

    # Only one project's annotations included
    assert len(rows) == 1
    r = rows[0]
    # Basename preserved (no stem)
    assert r["image_id"] == "IMG_7777.jpg"
    # Class normalized by default
    assert r["class"] == "pink_rose"
    # COCO-style xywh_abs preserved in reader
    assert len(r["bbox"]) == 4
    assert all(isinstance(v, float) for v in r["bbox"])


@pytest.mark.parametrize(
    "mode, want_det, want_cls",
    [("auto", True, True), ("detection", True, False), ("classification", False, True)],
)
def test_labelbox_reader_mode_filtering(
    tmp_path: Path, mode: str, want_det: bool, want_cls: bool
):
    # one record with one det object and three image-level classes
    p = tmp_path / "labels_mode.json"
    p.write_text(json.dumps(_record(external_id="IMG_42.jpg", with_cls=True)))

    rows = list(
        lb_reader.read(
            {
                "uri": str(p),
                "fmt": "labelbox",
                "options": {
                    "id_from": "external_stem",
                    "mode": mode,
                    "normalize_names": True,
                },
            }
        )
    )

    det_rows = [r for r in rows if "bbox" in r]
    cls_rows = [r for r in rows if "bbox" not in r]

    if want_det:
        assert len(det_rows) == 1
        assert det_rows[0]["class"] == "flower"
    else:
        assert len(det_rows) == 0

    if want_cls:
        assert sorted(r["class"] for r in cls_rows) == ["red", "rose", "white"]
    else:
        assert len(cls_rows) == 0


def test_labelbox_reader_id_from_variants(tmp_path: Path):
    rec = _record(external_id="IMG_7.jpeg", row_data="https://h/s/IMG_7.jpeg?x=1")
    f = tmp_path / "labels.jsonl"
    f.write_text(json.dumps(rec) + "\n")

    # external_id
    rows = list(
        lb_reader.read(
            {"uri": str(f), "fmt": "labelbox", "options": {"id_from": "external_id"}}
        )
    )
    assert all(r["image_id"] == "IMG_7.jpeg" for r in rows)

    # external_stem
    rows = list(
        lb_reader.read(
            {"uri": str(f), "fmt": "labelbox", "options": {"id_from": "external_stem"}}
        )
    )
    assert all(r["image_id"] == "IMG_7" for r in rows)

    # row_data_basename
    rows = list(
        lb_reader.read(
            {
                "uri": str(f),
                "fmt": "labelbox",
                "options": {"id_from": "row_data_basename"},
            }
        )
    )
    assert all(r["image_id"] == "IMG_7.jpeg" for r in rows)

    # row_data_stem
    rows = list(
        lb_reader.read(
            {"uri": str(f), "fmt": "labelbox", "options": {"id_from": "row_data_stem"}}
        )
    )
    assert all(r["image_id"] == "IMG_7" for r in rows)
