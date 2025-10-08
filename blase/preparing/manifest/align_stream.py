from typing import Any, Dict, Iterable, Iterator, Mapping, Sequence, List
from blase.types import Batch
from blase.preparing.manifest.normalize_boxes import convert_box


def align_stream(
    label_iters: Sequence[Iterable[Dict[str, Any]]],
    kv_index: Mapping[str, Dict[str, Any]],
    rg_index: Any,
    *,
    join_cfg: Dict[str, Any],
    box_cfg: Dict[str, Any],
    class_cfg: Dict[str, Any],
    scale_cfg: Dict[str, Any],
) -> Iterator[Batch[Sequence[Dict[str, Any]], Dict[str, Any]]]:
    """
    Minimal join:
      - For each label row, resolve image by image_id in kv_index.
      - Build per-image dict with det and cls lists.
      - Convert det coords per config.
      - Yield in small batches to bound memory.
    """
    coord_in = box_cfg.get("coord_in", "xywh_abs")
    coord_out = box_cfg.get("coord_out", "xyxy_rel")
    clamp = bool(box_cfg.get("clamp_boxes", True))
    keep_unlabeled = bool(join_cfg.get("keep_unlabeled_images", True))
    batch_rows = int(scale_cfg.get("batch_rows", 2048))

    by_image: Dict[str, Dict[str, Any]] = {}

    # 1) seed from kv_index so negatives can be emitted
    if keep_unlabeled:
        for image_id, meta in kv_index.items():
            by_image[image_id] = {
                "image_id": image_id,
                "path": meta.get("path"),
                "height": meta.get("height"),
                "width": meta.get("width"),
                "sha256": meta.get("sha256"),
                "det": [],
                "cls": [],
                "shard_path": meta.get("shard_path"),
                "rg_id": meta.get("rg_id"),
            }

    # 2) stream labels and attach
    def _ensure(img_id: str) -> Dict[str, Any]:
        if img_id not in by_image:
            meta = kv_index.get(img_id)
            if not meta:
                return {}
            by_image[img_id] = {
                "image_id": img_id,
                "path": meta.get("path"),
                "height": meta.get("height"),
                "width": meta.get("width"),
                "sha256": meta.get("sha256"),
                "shard_path": meta.get("shard_path"),
                "rg_id": meta.get("rg_id"),
                "det": [],
                "cls": [],
            }
        return by_image[img_id]

    def _flush(final: bool) -> Iterator[Batch]:
        if not by_image:
            return
        chunk: List[Dict[str, Any]] = list(by_image.values())
        yield Batch(data=chunk, meta={"count": len(chunk)}, is_last=final)
        if final:
            by_image.clear()  # only clear at the very end

    since_flush = 0
    for it in label_iters:
        for row in it:
            img_id = row.get("image_id")
            if not img_id or img_id not in kv_index:
                # orphan -> drop if requested
                if join_cfg.get("drop_orphans", True):
                    continue
                else:
                    # no kv; skip for now
                    continue
            dst = _ensure(img_id)
            if not dst:  # kv missing
                continue
            W = max(int(dst.get("width") or 0), 1)
            H = max(int(dst.get("height") or 0), 1)

            is_det = "bbox" in row
            is_cls = ("class" in row) or ("classes" in row)  # noqa

            if is_det:
                ok, rel = convert_box(
                    row["bbox"],
                    W,
                    H,
                    coord_in=coord_in,
                    coord_out=coord_out,
                    clamp=clamp,
                    drop_oob=bool(box_cfg.get("drop_oob_boxes", False)),
                )
                if ok:
                    x1, y1, x2, y2 = rel
                    dst["det"].append(
                        {
                            "xmin": x1,
                            "ymin": y1,
                            "xmax": x2,
                            "ymax": y2,
                            "label": row.get("label") or row.get("class"),
                            "iscrowd": int(row.get("iscrowd", 0)),
                        }
                    )
            else:
                # classification
                if "classes" in row:
                    for c in row["classes"]:
                        dst["cls"].append(c)
                elif "class" in row:
                    dst["cls"].append(row["class"])

            # emit batches
            since_flush += 1
            if since_flush >= batch_rows:
                yield from _flush(final=False)
                since_flush = 0

    yield from _flush(final=True)
