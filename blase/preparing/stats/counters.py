from typing import Dict, Any, Iterable, Sequence
from blase.preparing.interfaces import ManifestBatch, Stats


def _update_class_hist(
    class_hist: Dict[str, int], dets: Sequence[Dict[str, Any]], cls: Sequence[Any]
):
    for d in dets:
        k = str(d.get("label"))
        class_hist[k] = class_hist.get(k, 0) + 1
    for c in cls:
        k = str(c)
        class_hist[k] = class_hist.get(k, 0) + 1


def compute(manifest_iter: Iterable[ManifestBatch], cfg: Dict[str, Any]) -> Stats:
    by = cfg.get("by", "class")
    if by not in ("class", "image", "global"):
        raise ValueError(f"unsupported stats 'by'={by!r}")

    if by == "class":
        class_hist: Dict[str, int] = {}
        n_images = 0
        for mb in manifest_iter:
            for row in mb.data:
                n_images += 1
                _update_class_hist(class_hist, row.get("det", ()), row.get("cls", ()))
        return {
            "by": "class",
            "n_images": n_images,
            "class_hist": dict(sorted(class_hist.items())),
        }

    if by == "image":
        # per-image counts
        out = []
        for mb in manifest_iter:
            for row in mb.data:
                out.append(
                    {
                        "image_id": row.get("image_id"),
                        "n_det": len(row.get("det", ())),
                        "n_cls": len(row.get("cls", ())),
                        "height": row.get("height"),
                        "width": row.get("width"),
                    }
                )
        return {"by": "image", "images": out}

    # global
    n_images = 0
    n_det = 0
    n_cls = 0
    class_hist: Dict[str, int] = {}
    for mb in manifest_iter:
        for row in mb.data:
            n_images += 1
            dets = row.get("det", ())
            cls = row.get("cls", ())
            n_det += len(dets)
            n_cls += len(cls)
            _update_class_hist(class_hist, dets, cls)
    return {
        "by": "global",
        "n_images": n_images,
        "n_det": n_det,
        "n_cls": n_cls,
        "class_hist": dict(sorted(class_hist.items())),
    }
