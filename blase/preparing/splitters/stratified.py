from typing import Dict, Any, Iterable, DefaultDict, List
from collections import defaultdict
from random import Random
from blase.preparing.interfaces import ManifestBatch, Splits


def _primary_class(row: Dict[str, Any]) -> str:
    # choose first det label if present, else first cls, else "__neg__"
    det = row.get("det") or []
    if det:
        return str(det[0].get("label"))
    cls = row.get("cls") or []
    return str(cls[0]) if cls else "__neg__"


def split(manifest_iter: Iterable[ManifestBatch], cfg: Dict[str, Any]) -> Splits:
    r = Random(cfg.get("seed", 42))
    fr = cfg["fractions"]
    buckets: DefaultDict[str, List[str]] = defaultdict(list)
    for mb in manifest_iter:
        for row in mb.data:
            buckets[_primary_class(row)].append(row["image_id"])
    out = {"train": [], "val": [], "test": []}
    for _, ids in buckets.items():
        r.shuffle(ids)
        n = len(ids)
        n_train = int(round(fr["train"] * n))
        n_val = int(round(fr["val"] * n))
        out["train"].extend(ids[:n_train])
        out["val"].extend(ids[n_train : n_train + n_val])
        out["test"].extend(ids[n_train + n_val :])
    return out
