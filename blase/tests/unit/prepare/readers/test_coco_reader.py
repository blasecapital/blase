import json
from pathlib import Path


# Assumes: blase.preparing.readers.coco.read exists and yields dict rows
from blase.preparing.readers import coco as coco_reader


def _write_coco(p: Path, with_img_level=False):
    data = {
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 4, "height": 2},
            {"id": 2, "file_name": "b.jpg", "width": 8, "height": 4},
        ],
        "annotations": [
            {
                "id": 10,
                "image_id": 1,
                "bbox": [0, 0, 2, 1],
                "category_id": 5,
                "iscrowd": 0,
            },
            {
                "id": 11,
                "image_id": 2,
                "bbox": [2, 1, 2, 2],
                "category_id": 6,
                "iscrowd": 0,
            },
        ],
        "categories": [{"id": 5, "name": "weed"}, {"id": 6, "name": "radish"}],
    }
    if with_img_level:
        data["image_level_annotations"] = [
            {"image_id": 1, "category_id": 6},
            {"image_id": 2, "category_id": 5},
        ]
    p.write_text(json.dumps(data))


def test_coco_reader_detection_only(tmp_path: Path):
    f = tmp_path / "coco_det.json"
    _write_coco(f, with_img_level=True)

    rows = list(
        coco_reader.read(
            {
                "uri": str(f),
                "fmt": "coco",
                "options": {"id_from": "stem", "mode": "detection"},
            }
        )
    )
    assert all("bbox" in r for r in rows)
    assert {r["class"] for r in rows} == {"weed", "radish"}


def test_coco_reader_classification_only_from_image_level(tmp_path: Path):
    f = tmp_path / "coco_cls.json"
    _write_coco(f, with_img_level=True)

    rows = list(
        coco_reader.read(
            {
                "uri": str(f),
                "fmt": "coco",
                "options": {
                    "id_from": "stem",
                    "mode": "classification",
                    "image_level_key": "image_level_annotations",
                },
            }
        )
    )
    assert all("bbox" not in r for r in rows)
    # two image-level labels from the extra block
    assert len(rows) == 2
    assert {r["class"] for r in rows} == {"weed", "radish"}


def test_coco_reader_auto_emits_both_when_available(tmp_path: Path):
    f = tmp_path / "coco_both.json"
    _write_coco(f, with_img_level=True)

    rows = list(
        coco_reader.read(
            {
                "uri": str(f),
                "fmt": "coco",
                "options": {
                    "id_from": "filename",
                    "mode": "auto",
                    "image_level_key": "image_level_annotations",
                },
            }
        )
    )
    has_det = any("bbox" in r for r in rows)
    has_cls = any("bbox" not in r for r in rows)
    assert has_det and has_cls
    # image_id uses filename, not stem
    assert all(r["image_id"] in {"a.jpg", "b.jpg"} for r in rows)
