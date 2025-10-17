from typing import Dict, Any, Iterable, List, Tuple
from blase.preparing.interfaces import ManifestBatch, Splits


def split(manifest_iter: Iterable[ManifestBatch], cfg: Dict[str, Any]) -> Splits:
    # expects per-row timestamp under row["timestamp"] (float or int epoch or sortable)
    rows: List[Tuple[str, float]] = []
    for mb in manifest_iter:
        for row in mb.data:
            ts = row.get("timestamp")
            if ts is None:
                ts = 0.0
            rows.append((row["image_id"], float(ts)))
    rows.sort(key=lambda x: x[1])
    n = len(rows)
    fr = cfg["fractions"]
    n_train = int(round(fr["train"] * n))
    n_val = int(round(fr["val"] * n))
    ids = [rid for rid, _ in rows]
    return {
        "train": ids[:n_train],
        "val": ids[n_train : n_train + n_val],
        "test": ids[n_train + n_val :],
    }
