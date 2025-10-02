import pytest
from itertools import chain
from typing import Any, Dict, Iterable, Iterator

from blase.prepare import (
    Prepare,
    ImageConfig,
    JoinConfig,
    BoxConfig,
    ClassConfig,
    ScaleConfig,
    DataSource,
    LabelSource,
)
import blase.prepare as prep_mod

# --- fakes --------------------------------------------------------------------


class FakeReader:
    def __init__(self, rows: list[Dict[str, Any]]):
        self._rows = rows

    def read(self, _cfg: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        def gen() -> Iterator[Dict[str, Any]]:
            for r in self._rows:
                yield dict(r)  # fresh each call

        return gen()


class FakeRegistry:
    def __init__(self, readers: Dict[str, Any]):
        self.label_readers = readers


def _kv(a=True, b=True):
    kv = {}
    if a:
        kv["a"] = {
            "image_id": "a",
            "path": "a.jpg",
            "height": 2,
            "width": 4,
            "sha256": "aa",
        }
    if b:
        kv["b"] = {
            "image_id": "b",
            "path": "b.jpg",
            "height": 4,
            "width": 8,
            "sha256": "bb",
        }
    return kv, None, None


# ---------- fixtures ----------
@pytest.fixture
def fake_kv():
    def _make(ids):
        kv = {
            i: {
                "image_id": i,
                "path": f"{i}.jpg",
                "height": 4,
                "width": 8,
                "sha256": None,
            }
            for i in ids
        }
        # Prepare expects indexer to return (kv_index, bloom, rg_index)
        return kv, None, None

    return _make


@pytest.fixture
def fake_reader_det():
    def _make(rows):
        return FakeReader(rows)

    return _make


@pytest.fixture
def fake_reader_cls():
    def _make(rows):
        return FakeReader(rows)

    return _make


@pytest.fixture
def fake_registry():
    def _make(readers):
        return FakeRegistry(readers)

    return _make


# --- tests --------------------------------------------------------------------


def test_build_manifest_happy_path(monkeypatch):
    # 1) mock indexer
    monkeypatch.setattr(
        prep_mod._idx,
        "build_image_index",
        lambda **kwargs: _kv(True, True),
        raising=False,
    )
    # 2) registry with one det reader + one cls reader
    det = FakeReader(
        [
            {"image_id": "a", "bbox": [0, 0, 1, 1], "class": "weed", "iscrowd": 0},
            {"image_id": "b", "bbox": [1, 1, 2, 2], "class": "radish", "iscrowd": 0},
        ]
    )
    cls = FakeReader(
        [
            {"image_id": "a", "class": "good"},
            {"image_id": "b", "class": "bad"},
        ]
    )
    prep = Prepare(registry=FakeRegistry({"fake_det": det, "fake_cls": cls}))

    ds = [DataSource(uri="mem://shard.parquet", fmt="parquet", options={})]
    ls = [
        LabelSource(
            kind="detection", uri="mem://det.ndjson", fmt="fake_det", options={}
        ),
        LabelSource(
            kind="classification", uri="mem://cls.ndjson", fmt="fake_cls", options={}
        ),
    ]

    it = prep.build_manifest(
        data_sources=ds,
        label_sources=ls,
        image_cfg=ImageConfig(
            id_col="image_id", path_col="path", height_col="height", width_col="width"
        ),
        join_cfg=JoinConfig(drop_orphans=True, keep_unlabeled_images=False),
        box_cfg=BoxConfig(coord_in="xyxy_abs", coord_out="xyxy_rel", clamp_boxes=True),
        class_cfg=ClassConfig(),
        scale_cfg=ScaleConfig(batch_rows=16),
    )

    batches = list(it)
    assert len(batches) >= 1
    # flatten to latest snapshot
    rows = list(chain.from_iterable(b.data for b in batches))
    by_id = {r["image_id"]: r for r in rows}
    assert set(by_id.keys()) == {"a", "b"}
    assert len(by_id["a"]["det"]) == 1 and by_id["a"]["cls"] == ["good"]
    # last batch flagged
    assert any(b.is_last for b in batches)


def test_build_manifest_keeps_unlabeled_images(monkeypatch):
    monkeypatch.setattr(
        prep_mod._idx,
        "build_image_index",
        lambda **_: _kv(a=True, b=True),
        raising=False,
    )
    det = FakeReader(
        [{"image_id": "a", "bbox": [0, 0, 1, 1], "class": "x", "iscrowd": 0}]
    )
    prep = Prepare(registry=FakeRegistry({"fake_det": det}))
    ds = [DataSource(uri="mem://x.parquet", fmt="parquet", options={})]
    ls = [LabelSource(kind="detection", uri="mem://det", fmt="fake_det", options={})]

    it = prep.build_manifest(
        data_sources=ds,
        label_sources=ls,
        join_cfg=JoinConfig(drop_orphans=True, keep_unlabeled_images=True),
        box_cfg=BoxConfig(coord_in="xyxy_abs", coord_out="xyxy_rel", clamp_boxes=True),
    )
    rows = list(chain.from_iterable(b.data for b in it))
    by_id = {r["image_id"]: r for r in rows}
    assert "b" in by_id and by_id["b"]["det"] == [] and by_id["b"]["cls"] == []


def test_build_manifest_raises_on_id_mismatch(monkeypatch):
    monkeypatch.setattr(
        prep_mod._idx,
        "build_image_index",
        lambda **_: _kv(a=True, b=False),
        raising=False,
    )
    # labels reference only "b" → mismatch with kv_ids {"a"}
    det = FakeReader(
        [{"image_id": "b", "bbox": [0, 0, 1, 1], "class": "x", "iscrowd": 0}]
    )
    prep = Prepare(registry=FakeRegistry({"fake_det": det}))
    with pytest.raises(RuntimeError, match="no label image_ids match"):
        _ = list(
            prep.build_manifest(
                data_sources=[
                    DataSource(uri="mem://x.parquet", fmt="parquet", options={})
                ],
                label_sources=[
                    LabelSource(
                        kind="detection", uri="mem://det", fmt="fake_det", options={}
                    )
                ],
                join_cfg=JoinConfig(drop_orphans=True, keep_unlabeled_images=False),
            )
        )


def test_build_manifest_batches_and_final_flag(monkeypatch):
    monkeypatch.setattr(
        prep_mod._idx,
        "build_image_index",
        lambda **_: (
            (
                {
                    f"id{i}": {
                        "image_id": f"id{i}",
                        "path": f"{i}.jpg",
                        "height": 2,
                        "width": 2,
                        "sha256": None,
                    }
                    for i in range(100)
                }
            ),
            None,
            None,
        ),
    )
    many = [
        {"image_id": f"id{i % 100}", "bbox": [0, 0, 1, 1], "class": "x", "iscrowd": 0}
        for i in range(10_000)
    ]
    prep = Prepare(registry=FakeRegistry({"fake_det": FakeReader(many)}))
    it = prep.build_manifest(
        data_sources=[DataSource(uri="mem://x.parquet", fmt="parquet", options={})],
        label_sources=[
            LabelSource(kind="detection", uri="mem://det", fmt="fake_det", options={})
        ],
        box_cfg=BoxConfig(coord_in="xywh_abs", coord_out="xyxy_rel", clamp_boxes=True),
        scale_cfg=ScaleConfig(batch_rows=256),
        join_cfg=JoinConfig(drop_orphans=True, keep_unlabeled_images=False),
    )
    seen = 0
    last_seen = False
    for b in it:
        assert len(b.data) >= 1
        assert isinstance(b.meta, dict) and "count" in b.meta
        last_seen = b.is_last
        seen += 1
        if seen == 3:
            break
    assert seen == 3
    assert last_seen is False  # not the final batch yet


def test_unlabeled_behavior(monkeypatch, fake_kv, fake_reader_det, fake_registry):
    # mock indexer
    monkeypatch.setattr(
        prep_mod._idx,
        "build_image_index",
        lambda **_: fake_kv(["a", "b"]),
        raising=False,
    )

    det = fake_reader_det(
        [{"image_id": "a", "bbox": [0, 0, 1, 1], "class": "x", "iscrowd": 0}]
    )
    prep = Prepare(registry=fake_registry({"fake": det}))

    base_kwargs = dict(
        data_sources=[DataSource(uri="mem://imgs.parquet", fmt="parquet", options={})],
        label_sources=[
            LabelSource(kind="detection", uri="mem://det", fmt="fake", options={})
        ],
        box_cfg=BoxConfig(coord_in="xywh_abs", coord_out="xyxy_rel", clamp_boxes=True),
        image_cfg=ImageConfig(
            id_col="image_id", path_col="path", height_col="height", width_col="width"
        ),
        scale_cfg=ScaleConfig(batch_rows=16),
    )

    # keep_unlabeled=True → include "b" with empty det/cls
    rows = list(
        chain.from_iterable(
            b.data
            for b in prep.build_manifest(
                **base_kwargs,
                join_cfg=JoinConfig(drop_orphans=True, keep_unlabeled_images=True),
            )
        )
    )
    by_id = {r["image_id"]: r for r in rows}
    assert (
        "a" in by_id
        and "b" in by_id
        and by_id["b"]["det"] == []
        and by_id["b"]["cls"] == []
    )

    # keep_unlabeled=False → exclude "b"
    rows2 = list(
        chain.from_iterable(
            b.data
            for b in prep.build_manifest(
                **base_kwargs,
                join_cfg=JoinConfig(drop_orphans=True, keep_unlabeled_images=False),
            )
        )
    )
    ids2 = {r["image_id"] for r in rows2}
    assert ids2 == {"a"}


def test_box_conversion(monkeypatch, fake_kv, fake_reader_det, fake_registry):
    monkeypatch.setattr(
        prep_mod._idx,
        "build_image_index",
        lambda **_: fake_kv(["a"]),
        raising=False,
    )

    # xywh_abs
    det_xywh = fake_reader_det(
        [{"image_id": "a", "bbox": [0, 0, 2, 1], "class": "x", "iscrowd": 0}]
    )
    prep = Prepare(registry=fake_registry({"fake": det_xywh}))
    rows = list(
        chain.from_iterable(
            b.data
            for b in prep.build_manifest(
                data_sources=[
                    DataSource(uri="mem://p.parquet", fmt="parquet", options={})
                ],
                label_sources=[
                    LabelSource(kind="detection", uri="mem://l", fmt="fake", options={})
                ],
                box_cfg=BoxConfig(
                    coord_in="xywh_abs", coord_out="xyxy_rel", clamp_boxes=True
                ),
                join_cfg=JoinConfig(drop_orphans=True, keep_unlabeled_images=False),
                image_cfg=ImageConfig(
                    id_col="image_id",
                    path_col="path",
                    height_col="height",
                    width_col="width",
                ),
                scale_cfg=ScaleConfig(batch_rows=8),
            )
        )
    )
    d = rows[0]["det"][0]
    assert 0 <= d["xmin"] < d["xmax"] <= 1 and 0 <= d["ymin"] < d["ymax"] <= 1

    # xyxy_abs
    det_xyxy = fake_reader_det(
        [{"image_id": "a", "bbox": [0, 0, 2, 1], "class": "x", "iscrowd": 0}]
    )
    prep2 = Prepare(registry=fake_registry({"fake2": det_xyxy}))
    rows = list(
        chain.from_iterable(
            b.data
            for b in prep2.build_manifest(
                data_sources=[
                    DataSource(uri="mem://p.parquet", fmt="parquet", options={})
                ],
                label_sources=[
                    LabelSource(
                        kind="detection", uri="mem://l", fmt="fake2", options={}
                    )
                ],
                box_cfg=BoxConfig(
                    coord_in="xyxy_abs", coord_out="xyxy_rel", clamp_boxes=True
                ),
                join_cfg=JoinConfig(drop_orphans=True, keep_unlabeled_images=False),
                image_cfg=ImageConfig(
                    id_col="image_id",
                    path_col="path",
                    height_col="height",
                    width_col="width",
                ),
                scale_cfg=ScaleConfig(batch_rows=8),
            )
        )
    )
    d2 = rows[0]["det"][0]
    assert 0 <= d2["xmin"] < d2["xmax"] <= 1 and 0 <= d2["ymin"] < d2["ymax"] <= 1


def test_classmap_provided_and_missing(
    monkeypatch, fake_kv, fake_reader_det, fake_registry
):
    monkeypatch.setattr(
        prep_mod._idx,
        "build_image_index",
        lambda **_: fake_kv(["a"]),
        raising=False,
    )
    det = fake_reader_det([{"image_id": "a", "bbox": [0, 0, 1, 1], "class": "weed"}])

    # OK with provided map
    prep_ok = Prepare(registry=fake_registry({"fake": det}))
    list(
        prep_ok.build_manifest(
            data_sources=[DataSource(uri="mem://p.parquet", fmt="parquet", options={})],
            label_sources=[
                LabelSource(kind="detection", uri="mem://l", fmt="fake", options={})
            ],
            class_cfg=ClassConfig(class_map={"__background__": 0, "weed": 1}),
            join_cfg=JoinConfig(drop_orphans=True, keep_unlabeled_images=False),
            box_cfg=BoxConfig(coord_in="xywh_abs", coord_out="xyxy_rel"),
            image_cfg=ImageConfig(
                id_col="image_id",
                path_col="path",
                height_col="height",
                width_col="width",
            ),
            scale_cfg=ScaleConfig(batch_rows=8),
        )
    )

    # Missing class raises
    prep_bad = Prepare(registry=fake_registry({"fake": det}))
    with pytest.raises(ValueError):
        list(
            prep_bad.build_manifest(
                data_sources=[
                    DataSource(uri="mem://p.parquet", fmt="parquet", options={})
                ],
                label_sources=[
                    LabelSource(kind="detection", uri="mem://l", fmt="fake", options={})
                ],
                class_cfg=ClassConfig(class_map={"__background__": 0, "radish": 1}),
                join_cfg=JoinConfig(drop_orphans=True, keep_unlabeled_images=False),
                box_cfg=BoxConfig(coord_in="xywh_abs", coord_out="xyxy_rel"),
                image_cfg=ImageConfig(
                    id_col="image_id",
                    path_col="path",
                    height_col="height",
                    width_col="width",
                ),
                scale_cfg=ScaleConfig(batch_rows=8),
            )
        )
