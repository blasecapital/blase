import pytest
from itertools import chain
from typing import Any, Dict, Iterable, Iterator, Mapping, Tuple, Sequence
from types import SimpleNamespace

from blase.types import Batch
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
from blase.preparing.registry import PrepareRegistry

# --- fakes --------------------------------------------------------------------


class FakeReader:
    def __init__(self, rows: list[Dict[str, Any]]):
        self._rows = rows

    def read(self, _cfg: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        def gen() -> Iterator[Dict[str, Any]]:
            for r in self._rows:
                yield dict(r)

        return gen()


# KV index builder
def kv_index(*ids: str) -> Tuple[Mapping[str, Dict[str, Any]], Any, Any]:
    kv = {
        i: {"image_id": i, "path": f"{i}.jpg", "height": 4, "width": 8, "sha256": None}
        for i in ids
    }
    return kv, None, None


# Class map builder: validate if map provided; otherwise infer classes present
def _classmap_build_or_validate(
    label_iters: Sequence[Iterable[Dict[str, Any]]], cfg: Dict[str, Any]
):
    provided = (cfg or {}).get("class_map")
    seen: set[str] = set()
    for it in label_iters:
        for r in it:
            c = r.get("class")
            if isinstance(c, str):
                seen.add(c)
    if provided is not None:
        missing = seen - set(provided.keys())
        if missing:
            raise ValueError(f"missing classes in provided map: {sorted(missing)}")
        return dict(provided), {"source": "provided"}
    # default: infer with background first
    cm = {"__background__": 0}
    for i, name in enumerate(sorted(seen), start=1):
        cm[name] = i
    return cm, {"source": "inferred"}


# Box conversion util (supports xywh_abs, xyxy_abs -> xyxy_rel)
def _to_xyxy_rel(b, img_w, img_h, coord_in):
    if coord_in == "xywh_abs":
        x, y, w, h = b
        xmin, ymin, xmax, ymax = x, y, x + w, y + h
    elif coord_in == "xyxy_abs":
        xmin, ymin, xmax, ymax = b
    else:
        # already relative or unknown: passthrough then clamp
        xmin, ymin, xmax, ymax = b
    xmin = max(0.0, min(1.0, xmin / img_w))
    xmax = max(0.0, min(1.0, xmax / img_w))
    ymin = max(0.0, min(1.0, ymin / img_h))
    ymax = max(0.0, min(1.0, ymax / img_h))
    if xmin > xmax:
        xmin, xmax = xmax, xmin
    if ymin > ymax:
        ymin, ymax = ymax, ymin
    return xmin, ymin, xmax, ymax


# Tolerant aligner: merges det + cls, applies unlabeled policy and box conversion
def _align_stream(
    *, label_iters, kv_index, rg_index, join_cfg, box_cfg, class_cfg, scale_cfg
):
    keep_unlabeled = bool((join_cfg or {}).get("keep_unlabeled_images", False))
    coord_in = (box_cfg or {}).get("coord_in", "xyxy_abs")
    coord_out = (box_cfg or {}).get("coord_out", "xyxy_rel")

    iters = list(label_iters)
    det_rows = list(iters[0]) if len(iters) >= 1 else []
    cls_rows = list(iters[1]) if len(iters) >= 2 else []

    det_by_id: Dict[str, list] = {}
    for r in det_rows:
        iid = r.get("image_id")
        if iid in kv_index:
            if coord_out == "xyxy_rel":
                W = kv_index[iid]["width"]
                H = kv_index[iid]["height"]
                xmin, ymin, xmax, ymax = _to_xyxy_rel(r["bbox"], W, H, coord_in)
                r = {**r, "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}
            det_by_id.setdefault(iid, []).append(r)

    cls_by_id: Dict[str, list[str]] = {}
    for r in cls_rows:
        iid = r.get("image_id")
        if iid in kv_index:
            cls_by_id.setdefault(iid, []).append(r.get("class"))

    rows = []
    for iid, meta in kv_index.items():
        has_any = iid in det_by_id or iid in cls_by_id
        if has_any or keep_unlabeled:
            rows.append(
                {
                    "image_id": iid,
                    "det": det_by_id.get(iid, []),
                    "cls": cls_by_id.get(iid, []),
                }
            )

    yield Batch(data=rows, meta={"count": len(rows)}, is_last=True)


def _chunked_align(
    *, label_iters, kv_index, rg_index, join_cfg, box_cfg, class_cfg, scale_cfg
):
    # build det index
    det_rows = list(label_iters[0]) if len(label_iters) >= 1 else []
    det_by_id = {}
    for r in det_rows:
        iid = r["image_id"]
        if iid in kv_index:
            W = kv_index[iid]["width"]
            H = kv_index[iid]["height"]
            x, y, w, h = r["bbox"]
            xmin, ymin, xmax, ymax = x / W, y / H, (x + w) / W, (y + h) / H
            det_by_id.setdefault(iid, []).append(
                {**r, "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}
            )

    keep_unlabeled = bool((join_cfg or {}).get("keep_unlabeled_images", False))
    rows = []
    for iid in kv_index.keys():
        has = iid in det_by_id
        if has or keep_unlabeled:
            rows.append({"image_id": iid, "det": det_by_id.get(iid, []), "cls": []})

    bs = int(scale_cfg.get("batch_rows", 256))
    n = len(rows)
    from blase.types import Batch

    for i in range(0, n, bs):
        chunk = rows[i : i + bs]
        yield Batch(data=chunk, meta={"count": len(chunk)}, is_last=(i + bs >= n))


# Build a ready PrepareRegistry with simple fakes
def make_registry(
    *, readers: Mapping[str, Any], kv_ids: Sequence[str], aligner_fn=_align_stream
):
    # adapters with required method names
    image_indexer = SimpleNamespace(build_index=lambda sources, cfg: kv_index(*kv_ids))
    classmap = SimpleNamespace(build_or_validate=_classmap_build_or_validate)
    aligner = SimpleNamespace(align_stream=aligner_fn)
    # trivial split/stats/writers
    splitters = {
        "stratified": SimpleNamespace(
            split=lambda it, cfg: {"train": [], "val": [], "test": []}
        )
    }
    stats = SimpleNamespace(compute=lambda it, cfg: {})
    tfr_writer = SimpleNamespace(
        write=lambda it, splits, cfg: Batch(data=[], meta={}, is_last=True)
    )
    sidecar_writers = {
        "jsonl": SimpleNamespace(
            write=lambda it, splits, cfg: Batch(data=[], meta={}, is_last=True)
        ),
        "parquet": SimpleNamespace(
            write=lambda it, splits, cfg: Batch(data=[], meta={}, is_last=True)
        ),
    }
    return PrepareRegistry(
        image_indexer=image_indexer,
        label_readers=readers,
        classmap=classmap,
        aligner=aligner,
        splitters=splitters,
        stats=stats,
        tfr_writer=tfr_writer,
        sidecar_writers=sidecar_writers,
    )


# --- tests --------------------------------------------------------------------


def test_build_manifest_happy_path(monkeypatch):
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
    prep = Prepare(
        registry=make_registry(
            readers={"fake_det": det, "fake_cls": cls}, kv_ids=["a", "b"]
        )
    )
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
    det = FakeReader(
        [{"image_id": "a", "bbox": [0, 0, 1, 1], "class": "x", "iscrowd": 0}]
    )
    prep = Prepare(registry=make_registry(readers={"fake_det": det}, kv_ids=["a", "b"]))
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
    # labels reference only "b" → mismatch with kv_ids {"a"}
    det = FakeReader(
        [{"image_id": "b", "bbox": [0, 0, 1, 1], "class": "x", "iscrowd": 0}]
    )
    prep = Prepare(registry=make_registry(readers={"fake_det": det}, kv_ids=["a"]))
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


def test_build_manifest_batches_and_final_flag():
    ids = [f"id{i}" for i in range(100)]
    many = [
        {"image_id": f"id{i % 100}", "bbox": [0, 0, 1, 1], "class": "x", "iscrowd": 0}
        for i in range(10_000)
    ]
    prep = Prepare(
        registry=make_registry(
            readers={"fake_det": FakeReader(many)},
            kv_ids=ids,
            # override default aligner with chunked batching
            aligner_fn=_chunked_align,
        )
    )

    it = prep.build_manifest(
        data_sources=[DataSource(uri="mem://x.parquet", fmt="parquet", options={})],
        label_sources=[
            LabelSource(kind="detection", uri="mem://det", fmt="fake_det", options={})
        ],
        box_cfg=BoxConfig(coord_in="xywh_abs", coord_out="xyxy_rel", clamp_boxes=True),
        scale_cfg=ScaleConfig(batch_rows=32),
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
    assert last_seen is False


def test_unlabeled_behavior():
    det = FakeReader(
        [{"image_id": "a", "bbox": [0, 0, 1, 1], "class": "x", "iscrowd": 0}]
    )
    prep = Prepare(registry=make_registry(readers={"fake": det}, kv_ids=["a", "b"]))

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
    assert "a" in by_id and "b" in by_id
    assert by_id["b"]["det"] == [] and by_id["b"]["cls"] == []

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
    assert {r["image_id"] for r in rows2} == {"a"}


def test_box_conversion():
    # registry with one image id "a"
    prep = Prepare(
        registry=make_registry(
            readers={
                "fake": FakeReader(
                    [
                        {
                            "image_id": "a",
                            "bbox": [0, 0, 2, 1],
                            "class": "x",
                            "iscrowd": 0,
                        }
                    ]
                )
            },
            kv_ids=["a"],
        )
    )
    # xywh_abs → xyxy_rel
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

    # xyxy_abs → xyxy_rel
    prep2 = Prepare(
        registry=make_registry(
            readers={
                "fake2": FakeReader(
                    [
                        {
                            "image_id": "a",
                            "bbox": [0, 0, 2, 1],
                            "class": "x",
                            "iscrowd": 0,
                        }
                    ]
                )
            },
            kv_ids=["a"],
        )
    )
    rows2 = list(
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
    d2 = rows2[0]["det"][0]
    assert 0 <= d2["xmin"] < d2["xmax"] <= 1 and 0 <= d2["ymin"] < d2["ymax"] <= 1


def test_classmap_provided_and_missing():
    det = FakeReader([{"image_id": "a", "bbox": [0, 0, 1, 1], "class": "weed"}])
    prep_ok = Prepare(registry=make_registry(readers={"fake": det}, kv_ids=["a"]))

    # OK with provided map
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
    prep_bad = Prepare(registry=make_registry(readers={"fake": det}, kv_ids=["a"]))
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
