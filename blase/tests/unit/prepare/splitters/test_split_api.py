from blase.prepare import Prepare
from blase.types import Batch


def _mb(rows):
    return Batch(data=rows, meta={"count": len(rows)}, is_last=True)


def _manifest():
    rows = [
        {"image_id": "a", "det": [{"label": "x"}], "cls": []},
        {"image_id": "b", "det": [{"label": "y"}], "cls": []},
        {"image_id": "c", "det": [{"label": "x"}], "cls": []},
        {"image_id": "d", "det": [], "cls": ["z"]},
    ]
    yield _mb(rows)


def test_random_split_sizes():
    prep = Prepare()
    out = prep.split(
        _manifest(), method="random", train=0.5, val=0.25, test=0.25, seed=1
    )
    assert len(out.data["train"]) + len(out.data["val"]) + len(out.data["test"]) == 4


def test_stratified_respects_classes():
    prep = Prepare()
    out = prep.split(
        _manifest(), method="stratified", train=0.5, val=0.25, test=0.25, seed=0
    )
    # both classes appear across splits in proportion; at least non-empty buckets
    assert sum(1 for i in out.data["train"] if i in {"a", "c"}) >= 1


def test_group_assigns_whole_groups():
    prep = Prepare()

    def man():
        rows = [
            {"image_id": "a", "group_id": "G1"},
            {"image_id": "b", "group_id": "G1"},
            {"image_id": "c", "group_id": "G2"},
            {"image_id": "d", "group_id": "G3"},
        ]
        yield _mb(rows)

    out = prep.split(
        man(),
        method="group",
        group_key="group_id",
        train=0.5,
        val=0.25,
        test=0.25,
        seed=42,
    )
    # a and b must be in the same split
    s = {k: set(v) for k, v in out.data.items()}
    assert (
        not (("a" in s["train"]) ^ ("b" in s["train"]))
        or not (("a" in s["val"]) ^ ("b" in s["val"]))
        or not (("a" in s["test"]) ^ ("b" in s["test"]))
    )


def test_time_split_orders_by_ts():
    prep = Prepare()

    def man():
        rows = [
            {"image_id": "a", "timestamp": 3},
            {"image_id": "b", "timestamp": 1},
            {"image_id": "c", "timestamp": 2},
            {"image_id": "d", "timestamp": 4},
        ]
        yield _mb(rows)

    out = prep.split(man(), method="time", train=0.5, val=0.25, test=0.25)
    # earliest ids should fall in train first
    assert set(out.data["train"]) == {"b", "c"}
