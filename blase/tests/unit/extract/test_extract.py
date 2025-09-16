import pytest
import warnings
import types

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
