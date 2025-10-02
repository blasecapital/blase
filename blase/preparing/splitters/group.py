from typing import Dict, Any, Iterable, DefaultDict, List
from collections import defaultdict
from random import Random
from blase.preparing.interfaces import ManifestBatch, Splits


def split(manifest_iter: Iterable[ManifestBatch], cfg: Dict[str, Any]) -> Splits:
    r = Random(cfg.get("seed", 42))
    key = cfg.get("group_key", "image_id")
    groups: DefaultDict[str, List[str]] = defaultdict(list)
    for mb in manifest_iter:
        for row in mb.data:
            groups[str(row.get(key, row["image_id"]))].append(row["image_id"])
    group_ids = list(groups.keys())
    r.shuffle(group_ids)
    # assign whole groups to buckets by running fill
    fr = cfg["fractions"]
    total = len(group_ids)
    target = {k: int(round(fr[k] * total)) for k in ("train", "val", "test")}
    counts = {"train": 0, "val": 0, "test": 0}
    out = {"train": [], "val": [], "test": []}
    order = ["train", "val", "test"]
    for g in group_ids:
        # pick the split with largest remaining need
        rem = {k: target[k] - counts[k] for k in order}
        split_name = max(order, key=lambda k: rem[k])
        out[split_name].extend(groups[g])
        counts[split_name] += 1
    return out
