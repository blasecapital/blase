from typing import Dict, Any, Iterable, List
from random import Random
from blase.preparing.interfaces import ManifestBatch, Splits


def split(manifest_iter: Iterable[ManifestBatch], cfg: Dict[str, Any]) -> Splits:
    r = Random(cfg.get("seed", 42))
    fr = cfg["fractions"]
    ids: List[str] = []
    for mb in manifest_iter:
        ids.extend(row["image_id"] for row in mb.data)
    r.shuffle(ids)
    n = len(ids)
    n_train = int(round(fr["train"] * n))
    n_val = int(round(fr["val"] * n))
    train_ids = ids[:n_train]
    val_ids = ids[n_train : n_train + n_val]
    test_ids = ids[n_train + n_val :]
    return {"train": train_ids, "val": val_ids, "test": test_ids}
