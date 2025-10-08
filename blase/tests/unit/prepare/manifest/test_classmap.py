import pytest
from blase.preparing.manifest.classmap import (
    build_or_validate_class_map,
    _normalize_name,
    canonicalize_class_map,
)


def _labels(rows):
    # simple iterable wrapper
    return rows


def test_build_map_deterministic_sorted():
    its = [
        _labels([{"class": "Weed"}, {"class": "Radish"}]),
        _labels([{"classes": ["Weed", "Soil"]}]),
    ]
    cmap, meta = build_or_validate_class_map(
        its, {"class_map": None, "normalize_names": True}
    )
    assert cmap["__background__"] == 0
    # names sorted: ["radish","soil","weed"] -> ids 1..3
    assert [cmap[k] for k in ["radish", "soil", "weed"]] == [1, 2, 3]
    assert meta["class_names"] == ["radish", "soil", "weed"]
    assert meta["counts"]["weed"] == 2


def test_validate_map_ok():
    its = [_labels([{"class": "weed"}])]
    provided = {"__background__": 0, "weed": 1, "radish": 2}
    cmap, meta = build_or_validate_class_map(
        its, {"class_map": provided, "normalize_names": True}
    )
    assert cmap is provided
    assert "weed" in meta["class_names"]


def test_validate_map_missing_raises():
    its = [_labels([{"class": "unknown"}])]
    provided = {"__background__": 0, "weed": 1}
    with pytest.raises(ValueError):
        build_or_validate_class_map(
            its, {"class_map": provided, "normalize_names": True}
        )


def test_no_normalization_keeps_distinct():
    its = [_labels([{"class": "Weed"}, {"class": "weed"}])]
    cmap, _ = build_or_validate_class_map(
        its, {"class_map": None, "normalize_names": False}
    )
    # both appear, alphabetical: "Weed", "weed"
    assert set(k for k in cmap if k not in {"__background__"}) == {"Weed", "weed"}


# ----------
# class_map_io
# ----------
# -------- _normalize_name --------


def test_normalize_name_basic():
    assert _normalize_name(" Flower ") == "flower"
    assert _normalize_name("RED ROSE") == "red_rose"
    assert _normalize_name("blue\tiris\n") == "blue_iris"
    assert _normalize_name("  many   spaces  ") == "many_spaces"


# -------- canonicalize_class_map (happy paths) --------


def test_canonicalize_empty():
    assert canonicalize_class_map({}) == {}


def test_canonicalize_sorted_by_id_and_normalized_keys():
    cm = {"Flower": 2, " red  rose ": 0, "BLUE\tIRIS": 1}
    out = canonicalize_class_map(cm, normalize_names=True)
    # sorted by id: 0 -> red_rose, 1 -> blue_iris, 2 -> flower
    assert list(out.items()) == [("red_rose", 0), ("blue_iris", 1), ("flower", 2)]


def test_canonicalize_without_normalization_preserves_keys():
    cm = {"Red Rose": 0, "BLUE IRIS": 1}
    out = canonicalize_class_map(cm, normalize_names=False)
    assert list(out.items()) == [("Red Rose", 0), ("BLUE IRIS", 1)]


# -------- canonicalize_class_map (error cases) --------


def test_canonicalize_rejects_negative_id():
    with pytest.raises(ValueError, match="non-negative"):
        canonicalize_class_map({"bad": -1})


def test_canonicalize_rejects_non_int_id():
    with pytest.raises(ValueError, match="non-negative int"):
        canonicalize_class_map({"bad": "0"})  # type: ignore


def test_canonicalize_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="duplicate class ids"):
        canonicalize_class_map({"a": 0, "b": 0})


def test_canonicalize_rejects_duplicate_after_normalization():
    # "Red Rose" and " red   rose " normalize to same key
    with pytest.raises(ValueError, match="duplicate class after normalization"):
        canonicalize_class_map({"Red Rose": 1, " red   rose ": 2}, normalize_names=True)
