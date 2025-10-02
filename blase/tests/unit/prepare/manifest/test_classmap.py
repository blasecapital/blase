import pytest
from blase.preparing.manifest.classmap import build_or_validate_class_map


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
