from typing import Any, Dict, List, Union


def make_example(row: Dict[str, Any], *, mode: str, include_image_bytes: bool) -> Any:
    # Try TF first. If unavailable, return a dict for JSONL fallback.
    try:
        import tensorflow as tf  # type: ignore
    except Exception:
        # Fallback: dict shape used by JSONL writer
        ex: Dict[str, Any] = {
            "example/id": row["image_id"],
            "image/height": int(row["height"]),
            "image/width": int(row["width"]),
            "image/sha256": row.get("sha256", ""),
        }
        if include_image_bytes:
            ex["image/encoded"] = (
                row.get("_image_bytes", b"").decode("latin1")
                if isinstance(row.get("_image_bytes"), (bytes, bytearray))
                else ""
            )
        else:
            ex["image/uri"] = row.get("path", "")
        det = row.get("det") or []
        if det:
            ex["bbox/xmin"] = [float(d["xmin"]) for d in det]
            ex["bbox/ymin"] = [float(d["ymin"]) for d in det]
            ex["bbox/xmax"] = [float(d["xmax"]) for d in det]
            ex["bbox/ymax"] = [float(d["ymax"]) for d in det]
            ex["bbox/label"] = [str(d.get("label", "")) for d in det]
            ex["bbox/iscrowd"] = [int(d.get("iscrowd", 0)) for d in det]
        cls = row.get("cls") or []
        if cls:
            ex["cls/labels"] = [str(c) for c in cls]
        return ex

    # TF path: build serialized Example bytes
    def _b(x: Union[str, bytes]) -> bytes:
        return x if isinstance(x, (bytes, bytearray)) else str(x).encode("utf-8")

    def _bytes_feature(v: bytes):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))

    def _bytes_list(vs: List[bytes]):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=vs))

    def _int64_feature(v: int):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(v)]))

    def _int64_list(vs: List[int]):
        return tf.train.Feature(
            int64_list=tf.train.Int64List(value=[int(x) for x in vs])
        )

    def _float_list(vs: List[float]):
        return tf.train.Feature(
            float_list=tf.train.FloatList(value=[float(x) for x in vs])
        )

    feats: Dict[str, tf.train.Feature] = {
        "example/id": _bytes_feature(_b(row["image_id"])),
        "image/height": _int64_feature(int(row["height"])),
        "image/width": _int64_feature(int(row["width"])),
        "image/sha256": _bytes_feature(_b(row.get("sha256", ""))),
    }
    if include_image_bytes:
        feats["image/encoded"] = _bytes_feature(row.get("_image_bytes", b""))
    else:
        feats["image/uri"] = _bytes_feature(_b(row.get("path", "")))
    det = row.get("det") or []
    if det:
        feats["bbox/xmin"] = _float_list([d["xmin"] for d in det])
        feats["bbox/ymin"] = _float_list([d["ymin"] for d in det])
        feats["bbox/xmax"] = _float_list([d["xmax"] for d in det])
        feats["bbox/ymax"] = _float_list([d["ymax"] for d in det])
        feats["bbox/label"] = _bytes_list([_b(d.get("label", "")) for d in det])
        feats["bbox/iscrowd"] = _int64_list([int(d.get("iscrowd", 0)) for d in det])
    cls = row.get("cls") or []
    if cls:
        feats["cls/labels"] = _bytes_list([_b(c) for c in cls])
    ex = tf.train.Example(features=tf.train.Features(feature=feats))
    return ex.SerializeToString()
