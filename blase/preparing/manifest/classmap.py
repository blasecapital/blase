from typing import Any, Dict, Iterable, Iterator, Mapping, Sequence, Tuple, Set
from collections import OrderedDict
import re


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


# ----------
# class_map_io
# ----------

_NAME_CLEAN_RE = re.compile(r"\s+")


def _normalize_name(s: str) -> str:
    s = s.strip().lower()
    s = _NAME_CLEAN_RE.sub("_", s)
    return s


def canonicalize_class_map(
    cm: Mapping[str, int],
    *,
    normalize_names: bool = True,
) -> Dict[str, int]:
    """
    Returns a deterministic class map:
      - optionally normalizes keys,
      - validates uniqueness and contiguous non-negative ids,
      - sorts by id and returns a plain dict.
    """
    if not cm:
        return {}

    # 1) normalize keys if requested
    items = []
    seen_keys: Set[str] = set()
    for k, v in cm.items():
        if not isinstance(v, int) or v < 0:
            raise ValueError(f"class id must be non-negative int; got {k!r}:{v!r}")
        kk = _normalize_name(k) if normalize_names else k
        if kk in seen_keys:
            raise ValueError(f"duplicate class after normalization: {kk!r}")
        seen_keys.add(kk)
        items.append((kk, v))

    # 2) ensure ids are unique
    ids = [v for _, v in items]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate class ids detected")

    # 3) sort deterministically by id
    items.sort(key=lambda kv: kv[1])

    # 4) return as dict preserving order
    return dict(items)
