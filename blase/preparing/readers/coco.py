import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Union


def _norm(s: str) -> str:
    return s.strip().lower().replace(" ", "_")


def _image_key(img: Mapping[str, Any], *, id_from: str) -> str:
    if id_from == "coco_id":
        return str(img["id"])
    if id_from == "filename":
        return Path(img["file_name"]).name
    if id_from == "stem":
        return Path(img["file_name"]).stem
    raise ValueError(
        f"Unsupported id_from={id_from!r} (expected 'coco_id' | 'filename' | 'stem')"
    )


def read(src: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """
    src["uri"]: COCO json path
    src["options"]:
      id_from: "stem" | "filename" | "coco_id" = "stem"
      normalize_names: bool = True
      mode: "auto" | "detection" | "classification" = "auto"
      image_level_key: optional key name containing image-level cls annotations
                       (e.g., "image_level_annotations" or "image_labels")
                       Each entry must have {"image_id", "category_id"}.
    Yields detection rows with bbox (xywh_abs) and/or classification rows without bbox.
    """
    uri = Path(src["uri"])
    opts: Dict[str, Any] = dict(src.get("options") or {})
    id_from: str = opts.get("id_from", "stem")
    normalize_names: bool = bool(opts.get("normalize_names", True))
    mode: str = opts.get("mode", "auto")
    image_level_key: Union[str, None] = opts.get("image_level_key")

    want_det = mode in ("auto", "detection")
    want_cls = mode in ("auto", "classification")

    with uri.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # maps
    cat_map = {
        int(c["id"]): (_norm(c["name"]) if normalize_names else c["name"])
        for c in data.get("categories", [])
    }
    img_key = {
        int(img["id"]): _image_key(img, id_from=id_from)
        for img in data.get("images", [])
    }

    # detection annotations (standard COCO)
    if want_det:
        for ann in data.get("annotations", []):
            img_id = int(ann.get("image_id"))
            if img_id not in img_key:
                continue
            cid = int(ann.get("category_id"))
            cls_name = cat_map.get(cid)
            if not cls_name:
                continue
            bbox = ann.get("bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                yield {
                    "image_id": img_key[img_id],
                    "bbox": [
                        float(bbox[0]),
                        float(bbox[1]),
                        float(bbox[2]),
                        float(bbox[3]),
                    ],  # xywh_abs
                    "class": cls_name,
                    "iscrowd": int(ann.get("iscrowd", 0)),
                    **({"area": float(ann["area"])} if "area" in ann else {}),
                    **(
                        {"segmentation": ann["segmentation"]}
                        if "segmentation" in ann
                        else {}
                    ),
                }
            else:
                # If bbox missing but mode includes classification, treat as image-level cls
                if want_cls and cls_name:
                    yield {"image_id": img_key[img_id], "class": cls_name}

    # optional image-level classification block
    if want_cls and image_level_key and image_level_key in data:
        for a in data.get(image_level_key, []):
            img_id = int(a.get("image_id"))
            if img_id not in img_key:
                continue
            cid = int(a.get("category_id"))
            cls_name = cat_map.get(cid)
            if not cls_name:
                continue
            yield {"image_id": img_key[img_id], "class": cls_name}
