import pytest
import warnings
import types
from pathlib import Path

import pandas as pd
import polars as pl

from blase.extract import Extract
import blase.extract as mod

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Temporary CSV content for testing
TEST_CSV_CONTENT = """id,value
1,100
2,200
3,300
4,400
5,500
"""


@pytest.fixture
def temp_csv_file(tmp_path):
    file_path = tmp_path / "test.csv"
    file_path.write_text(TEST_CSV_CONTENT)
    return str(file_path)


class FakeStreamStep:
    def __init__(self):
        self.outputs = []
        self.registered = []
        self.db = "/tmp/nodes.db"
        self.step_hash = "fake_step_hash"

    def register_data(self, kind, version, path_or_bytes, metadata):
        # Return a deterministic 12-char hex-like token per registration.
        token = f"{kind}:{metadata.get('ordinal', 'root')}"
        self.registered.append((kind, version, path_or_bytes, metadata))
        return token

    def add_output(self, data_id, name=None):
        self.outputs.append((name, data_id))


class FakeStream:
    def __init__(self):
        self.step = FakeStreamStep()
        self._closed_ok = False
        self._closed_err = None

    def emit(self, last_batch: bool, meta: dict):
        # Return meta the way the implementation expects.
        out = dict(meta)
        out.setdefault("producer_step", "fake_step_hash")
        return out

    def close_ok(self):
        self._closed_ok = True

    def close_error(self, etype, exc, tb):
        self._closed_err = (etype, str(exc))


class FakeTracker:
    def stream(self, fqn, params, code_fn):
        return FakeStream()


def _fake_manifest(n=3):
    # Minimal items: abs_path, hash_content
    items = [
        {
            "abs_path": f"/img/{i}.jpg",
            "rel_path": f"{i}.jpg",
            "hash_content": f"h{i:02d}",
        }
        for i in range(n)
    ]
    return items


def _fake_plan(items, batch_sizes):
    batches = []
    idx = 0
    for k in batch_sizes:
        chunk = items[idx : idx + k]
        idx += k
        batches.append(
            {
                "items": chunk,
                "est_decoded_bytes": 1234 * max(1, k),
                "is_last": idx >= len(items),
            }
        )
    return batches


@pytest.fixture
def isolate_deps(monkeypatch):
    import blase.extracting.image_backend as ib

    # backend + manifest + plan stubs
    monkeypatch.setattr(mod, "resolve_backend_images", lambda b: b)
    monkeypatch.setattr(mod, "scan_manifest_headers", lambda **kw: _fake_manifest(5))
    monkeypatch.setattr(mod, "shuffle_manifest", lambda m, s: list(reversed(m)))
    monkeypatch.setattr(mod, "ensure_item_content_hashes", lambda m: m)
    monkeypatch.setattr(mod, "compute_manifest_root_hash", lambda **kw: "rootHASH")

    def _plan(**kw):
        items = _fake_manifest(5)
        return _fake_plan(items, [2, 2, 1])

    monkeypatch.setattr(mod, "plan_image_batches", _plan)
    monkeypatch.setattr(mod, "iter_batches_from_plan", lambda plan: iter(plan))

    # decode + batch hash
    monkeypatch.setattr(
        mod,
        "decode_batch_pil",
        lambda items, rt, color, ms: [f"im:{x['abs_path']}" for x in items],
    )
    monkeypatch.setattr(
        mod,
        "decode_batch_cv2",
        lambda items, rt, color, ms: [f"im:{x['abs_path']}" for x in items],
    )
    monkeypatch.setattr(
        mod, "compute_batch_hash", lambda root, items: "bh_" + str(len(items))
    )

    # CAS + dataset stubs
    monkeypatch.setattr(
        mod, "build_manifest_descriptor", lambda **kw: {"root_hash": "rootHASH"}
    )
    monkeypatch.setattr(mod, "ensure_dataset_for_manifest", lambda *a, **k: "ds1")

    # Return a concrete budget so read_images never calls image_backend
    monkeypatch.setattr(
        mod, "auto_target_bytes_from_system", lambda rt: 256 * 1024 * 1024
    )
    # Belt-and-suspenders if other code calls the module function directly
    monkeypatch.setattr(
        ib, "auto_target_bytes_from_system", lambda rt: 256 * 1024 * 1024
    )
    yield


# ----- Extract().read_csv -----


def test_read_csv_batches(temp_csv_file):
    extractor = Extract()
    batches = [
        batch
        for batch, _, _ in extractor.read_csv(
            file_path=temp_csv_file,
            mode="manual",
            batch_size=2,
            backend="pandas",
            track=False,
        )
    ]

    # Should split into 3 batches (2+2+1)
    assert len(batches) == 3
    assert isinstance(batches[0], pd.DataFrame)
    assert batches[0].shape[0] == 2
    assert batches[1].shape[0] == 2
    assert batches[2].shape[0] == 1
    assert batches[0].iloc[0]["id"] == 1


def test_read_csv_batches_polars(temp_csv_file):
    extractor = Extract()
    batches = [
        batch
        for batch, _, _ in extractor.read_csv(
            file_path=temp_csv_file,
            mode="manual",
            batch_size=2,
            backend="polars",
            track=False,
        )
    ]

    # Should split into 3 batches (2+2+1)
    assert len(batches) == 3
    assert isinstance(batches[0], pl.DataFrame)
    assert batches[0].shape[0] == 2
    assert batches[1].shape[0] == 2
    assert batches[2].shape[0] == 1

    # Polars uses `batches[0][col][row]` for indexing
    assert batches[0]["id"][0] == 1


# ----- Extract().read_images test -----
def test_untracked_auto_mode_yields_dict_batches(isolate_deps, monkeypatch, tmp_path):
    # No active tracker
    class _NoTrack:
        @staticmethod
        def get(flag):  # flag is True by default
            return None

    monkeypatch.setattr(mod, "Track", _NoTrack)

    ex = types.SimpleNamespace()
    # Bind the method function under test
    gen = mod.Extract.read_images.__get__(ex, mod.Extract)(
        directory=str(tmp_path),
        backend="pil",
        return_type="np",
        mode="auto",
        shuffle=True,  # exercise seeding default path
        track=False,  # also forces untracked path since Track.get returns None
    )

    outs = list(gen)
    assert len(outs) == 3
    for i, batch in enumerate(outs, 1):
        assert isinstance(batch, dict)
        assert set(batch.keys()) == {"paths", "images", "meta"}
        assert all(p.startswith("/img/") for p in batch["paths"])
        assert len(batch["paths"]) == len(batch["images"])
        assert batch["meta"]["ordinal"] == i
        assert batch["meta"]["batch_hash"].startswith("bh_")
        assert batch["meta"]["manifest_root_hash"] == "rootHASH"
        assert batch["meta"]["reader_backend"] == "pil"
        assert batch["meta"]["return_type"] == "np"


def test_tracked_path_emits_tuple_and_logs(isolate_deps, monkeypatch, tmp_path):
    # Active tracker
    class _Track:
        @staticmethod
        def get(flag):
            return FakeTracker()

    monkeypatch.setattr(mod, "Track", _Track)

    ex = types.SimpleNamespace()
    gen = mod.Extract.read_images.__get__(ex, mod.Extract)(
        directory=str(tmp_path),
        backend="cv2",
        return_type="np",
        mode="manual",
        batch_size=2,
        shuffle=False,
        track=True,
    )

    outs = list(gen)
    assert len(outs) == 3

    # Each yield is (paths, is_last, images, meta)
    for i, tup in enumerate(outs, 1):
        assert isinstance(tup, (list, tuple)) and len(tup) == 4
        paths, is_last, images, meta = tup
        assert all(p.startswith("/img/") for p in paths)
        assert isinstance(is_last, bool)
        assert len(paths) == len(images)
        assert meta["batch_hash"].startswith("bh_")
        assert meta["manifest_root_hash"] == "rootHASH"
        assert meta["producer_step"]  # added in FakeStream.emit

    # Last tuple should have is_last True
    assert outs[-1][1] is True


def test_invalid_mode_raises(monkeypatch, isolate_deps, tmp_path):
    class _NoTrack:
        @staticmethod
        def get(flag):
            return None

    monkeypatch.setattr(mod, "Track", _NoTrack)

    ex = types.SimpleNamespace()
    with pytest.raises(ValueError):
        gen = mod.Extract.read_images.__get__(ex, mod.Extract)(
            directory=str(tmp_path), mode="weird"
        )
        next(gen)

    with pytest.raises(ValueError):
        gen = mod.Extract.read_images.__get__(ex, mod.Extract)(
            directory=str(tmp_path), mode="manual"
        )
        next(gen)


def test_seed_default_applied_when_shuffle_true(isolate_deps, monkeypatch, tmp_path):
    # Intercept shuffle_manifest to assert seed value when None
    called = {}

    def _shuffle(manifest, seed):
        called["seed"] = seed
        return manifest[::-1]

    monkeypatch.setattr(mod, "shuffle_manifest", _shuffle)

    class _NoTrack:
        @staticmethod
        def get(flag):
            return None

    monkeypatch.setattr(mod, "Track", _NoTrack)

    ex = types.SimpleNamespace()
    list(
        mod.Extract.read_images.__get__(ex, mod.Extract)(
            directory=str(tmp_path),
            shuffle=True,
            seed=None,
            track=False,
        )
    )
    # The implementation sets seed to 42 when shuffle=True and seed is None
    assert called["seed"] == 42


# ---------------- read_parquet tests ----------------


def test_read_parquet_param_validation_errors(monkeypatch):
    from blase.extract import Extract
    import blase.extract as mod

    # Minimal stubs to avoid touching FS if anything runs
    monkeypatch.setattr(
        mod, "_normalize_source_list", lambda *a, **k: [Path("/p/a.parquet")]
    )
    monkeypatch.setattr(
        mod,
        "_scan_parquet_manifest_stub",
        lambda **kw: {"schema_fp": "S", "kind": "parquet"},
    )
    monkeypatch.setattr(
        mod, "_plan_parquet_batches_stub", lambda **kw: [{"count": 1, "is_last": True}]
    )

    ex = Extract()

    with pytest.raises(ValueError):
        for _ in ex.read_parquet("/p", format="bad"):  # noqa: B018
            pass

    with pytest.raises(ValueError):
        for _ in ex.read_parquet("/p", return_type="bad"):  # noqa: B018
            pass

    with pytest.raises(ValueError):
        for _ in ex.read_parquet("/p", mode="bad"):  # noqa: B018
            pass

    with pytest.raises(ValueError):
        for _ in ex.read_parquet("/p", decode="bad"):  # noqa: B018
            pass

    with pytest.raises(ValueError):
        for _ in ex.read_parquet("/p", images_return="bad"):  # noqa: B018
            pass


def test_read_parquet_untracked_table_mode(monkeypatch):
    """
    Track disabled (or Track.get -> None).
    Expect simple dict batches: {"data": <table_like>, "meta": {...}}.
    """
    from blase.extract import Extract
    import blase.extract as mod

    # Force untracked path
    monkeypatch.setattr(mod, "Track", types.SimpleNamespace(get=lambda _: None))

    # Stubs
    monkeypatch.setattr(
        mod,
        "_normalize_source_list",
        lambda *a, **k: [Path("/p/a.parquet"), Path("/p/b.parquet")],
    )
    monkeypatch.setattr(
        mod,
        "_scan_parquet_manifest_stub",
        lambda **kw: {"schema_fp": "SFP", "kind": "dataset"},
    )
    monkeypatch.setattr(
        mod,
        "_plan_parquet_batches_stub",
        lambda **kw: [{"count": 2, "is_last": False}, {"count": 1, "is_last": True}],
    )
    monkeypatch.setattr(
        mod,
        "_read_parquet_table_stub",
        lambda manifest,
        plan,
        return_type,
        columns,
        use_threads,
        to_pandas_kwargs=None,
        mode=None: f"T{plan['count']}",
    )

    ex = Extract()
    it = ex.read_parquet(
        source="/data/parquet_root",
        mode="table",
        return_type="arrow",
        track=False,  # belt-and-suspenders; Track.get also returns None
    )

    batches = list(it)
    assert len(batches) == 2

    d0 = batches[0]
    assert set(d0.keys()) == {"data", "meta"}
    assert d0["data"] == "T2"
    m0 = d0["meta"]
    assert m0["ordinal"] == 1
    assert m0["count"] == 2
    assert m0["mode"] == "table"
    assert m0["schema_fp"] == "SFP"
    assert m0["source_kind"] == "dataset"
    assert m0["batch_policy"]["mode"] == "rows"

    d1 = batches[1]
    assert d1["data"] == "T1"
    assert d1["meta"]["ordinal"] == 2
    assert d1["meta"]["count"] == 1
    assert d1["meta"]["mode"] == "table"


def test_read_parquet_tracked_images_mode(monkeypatch):
    """
    Track enabled. images mode.
    Expect per-batch tuples: ((imgs, labels, paths), is_last, meta)
    with upstream references and items count.
    """
    from blase.extract import Extract
    import blase.extract as mod

    # Use provided FakeTracker/FakeStream
    t = FakeTracker()
    monkeypatch.setattr(mod, "Track", types.SimpleNamespace(get=lambda _: t))

    # Manifest + hashing + dataset
    monkeypatch.setattr(
        mod, "_normalize_source_list", lambda *a, **k: [Path("/p/a.parquet")]
    )
    monkeypatch.setattr(
        mod,
        "_scan_parquet_manifest_stub",
        lambda **kw: {"schema_fp": "SFP", "kind": "dataset"},
    )
    monkeypatch.setattr(
        mod, "_ensure_parquet_content_hashes", lambda manifest: manifest
    )
    monkeypatch.setattr(
        mod, "_compute_manifest_root_hash_stub", lambda manifest: "rootHASH"
    )
    monkeypatch.setattr(
        mod,
        "build_manifest_descriptor",
        lambda **kw: {"root_hash": "rootHASH", "n": 2},
    )
    monkeypatch.setattr(mod, "ensure_dataset_for_manifest", lambda **kw: "ds1")

    # Plan: two batches, counts 2 then 1
    monkeypatch.setattr(
        mod,
        "_plan_parquet_batches_stub",
        lambda **kw: [
            {"count": 2, "is_last": False, "fragment": 0, "row_group": 0, "offset": 0},
            {"count": 1, "is_last": True, "fragment": 0, "row_group": 1, "offset": 2},
        ],
    )
    monkeypatch.setattr(
        mod, "_compute_batch_hash_stub", lambda root, plan: f"bh_{plan['count']}"
    )

    # Table read is irrelevant for images but keep a stub
    monkeypatch.setattr(
        mod,
        "_read_parquet_table_stub",
        lambda manifest,
        plan,
        return_type,
        columns,
        mode=None: f"ArrowTable(count={plan['count']})",
    )

    # Images path helpers
    monkeypatch.setattr(
        mod,
        "_auto_detect_image_cols",
        lambda table_like, bytes_col, dims_cols, label_col, path_col: (
            bytes_col,
            dims_cols,
            label_col,
            path_col,
        ),
    )
    monkeypatch.setattr(mod, "_validate_image_schema", lambda **kw: None)

    def _iter_images_from_table(**kw):
        cnt = kw["table_like"]
        # parse "ArrowTable(count=N)" -> N
        n = int(str(cnt).split("=")[1].rstrip(")"))
        imgs = [f"img{i}" for i in range(n)]
        labels = [f"lbl{i}" for i in range(n)] if kw.get("label_col") else None
        paths = [f"/p/{i}.jpg" for i in range(n)]
        yield imgs, labels, paths

    monkeypatch.setattr(
        mod, "_iter_images_from_table", lambda **kw: _iter_images_from_table(**kw)
    )

    ex = Extract()
    gen = ex.read_parquet(
        source="/data/parquet_root",
        mode="images",
        return_type="arrow",
        decode="np",
        images_return="np",
        track=True,
    )

    out = list(gen)
    assert len(out) == 2

    # Batch 1
    (imgs1, lbls1, paths1), last1, meta1 = out[0]
    assert last1 is False
    assert imgs1 == ["img0", "img1"]
    assert isinstance(lbls1, list) and len(lbls1) == 2
    assert len(paths1) == 2
    assert meta1["ordinal"] == 1
    assert meta1["count"] == 2
    assert meta1["manifest_root_hash"] == "rootHASH"
    assert meta1["producer_step"] == "fake_step_hash"
    up1 = meta1["upstream"]
    roles1 = {u["role"] for u in up1}
    assert roles1 == {"manifest", "batch_desc", "batch"}

    # Batch 2
    (imgs2, lbls2, paths2), last2, meta2 = out[1]
    assert last2 is True
    assert imgs2 == ["img0"]
    assert isinstance(lbls2, list) and len(lbls2) == 1
    assert len(paths2) == 1
    assert meta2["ordinal"] == 2
    assert meta2["count"] == 1

    # Stream registered outputs: manifest + batch_desc_* tokens exist
    names = [
        name for (name, _id) in t.stream("x", {}, None).step.outputs
    ]  # new stream; check shape only
    assert isinstance(names, list)  # sanity of FakeStream API
