from typing import Any, Dict, Iterable, Iterator, Mapping, Sequence, Tuple, Set
from collections import OrderedDict


def _iter_labels(
    label_iters: Sequence[Iterable[Dict[str, Any]]],
) -> Iterator[Dict[str, Any]]:
    for it in label_iters:
        for row in it:
            yield row


def _norm(name: str, normalize: bool) -> str:
    if not normalize:
        return name
    return " ".join(name.strip().lower().split())


def build_or_validate_class_map(
    label_iters: Sequence[Iterable[Dict[str, Any]]],
    class_cfg: Dict[str, Any],
) -> Tuple[Mapping[str, int], Dict[str, Any]]:
    """
    Contract:
      - If class_cfg['class_map'] is None -> build map from observed class names.
      - If provided -> validate that all observed classes exist in map.
      - Always return (map, meta) where meta includes counts and ordered names.
      - Deterministic ids: background=0 reserved, classes assigned by sorted normalized names.
    """
    provided = class_cfg.get("class_map")
    normalize = bool(class_cfg.get("normalize_names", True))

    seen: "OrderedDict[str, int]" = OrderedDict()
    counts: Dict[str, int] = {}

    # One pass to collect unique labels and counts
    for row in _iter_labels(label_iters):
        # detection rows: may carry "label" or "class"
        # classification rows: "class" or "classes" (list)
        names: Sequence[str] = []
        if "classes" in row and isinstance(row["classes"], (list, tuple)):
            names = [str(x) for x in row["classes"]]
        elif "class" in row:
            names = [str(row["class"])]
        elif "label" in row:
            names = [str(row["label"])]
        else:
            continue

        for name in names:
            n = _norm(name, normalize)
            counts[n] = counts.get(n, 0) + 1
            if n not in seen:
                seen[n] = 0  # placeholder

    if provided is None:
        # Build new map: background=0, then 1..N by sorted class names
        names_sorted = sorted(seen.keys())
        class_map = {"__background__": 0}
        for i, name in enumerate(names_sorted, start=1):
            class_map[name] = i
        meta = {
            "normalize_names": normalize,
            "class_names": names_sorted,
            "counts": counts,
        }
        return class_map, meta

    # Validate provided
    provided_norm = {
        (_norm(k, normalize)): v for k, v in provided.items() if k != "__background__"
    }
    missing: Set[str] = set(seen.keys()) - set(provided_norm.keys())
    if missing:
        raise ValueError(f"class_map missing classes: {sorted(missing)}")

    # Keep provided as-is, but report meta in normalized space
    names_sorted = sorted(seen.keys())
    meta = {"normalize_names": normalize, "class_names": names_sorted, "counts": counts}
    return provided, meta
