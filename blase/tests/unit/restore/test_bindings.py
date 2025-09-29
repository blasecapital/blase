from types import SimpleNamespace, ModuleType
from pathlib import Path
import sys
import pytest
import json
import sqlite3

from blase.types import Batch, SinkResult, Artifact
import blase.restoring.bindings as b


def _fake_manifest(n=5):
    return [
        {
            "rel_path": f"{i}.jpg",
            "relpath": f"{i}.jpg",  # some code paths read this key
            "path": f"/img/{i}.jpg",
            "abs_path": f"/img/{i}.jpg",
            "bytes": 1234,
            "width": 10,
            "height": 10,
            "mode": "RGB",
            "channels": 3,
            "is_corrupt": False,
            "hash_path_mtime": 0,
            "hash_content": f"h{i:02d}",
        }
        for i in range(n)
    ]


def _fake_plan(items, sizes):
    out, i = [], 0
    for k in sizes:
        chunk = items[i : i + k]
        i += k
        out.append(
            {"items": chunk, "est_decoded_bytes": 4096 * k, "is_last": i >= len(items)}
        )
    return out


def _write_blob(base: Path, kind: str, h: str, data: dict, reversed_order=False):
    """Write JSON to one of the candidate CAS layouts."""
    a, b2 = h[:2], h[2:4]
    if reversed_order:
        p = base / "cas" / "sha256" / kind / b2 / a / h
    else:
        p = base / "cas" / "sha256" / kind / a / b2 / h
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data))


def _init_db(base: Path):
    db_dir = base / "nodes"
    db_dir.mkdir()
    db_path = db_dir / "nodes.db"
    con = sqlite3.connect(db_path)
    con.execute(
        "CREATE TABLE data (data_hash TEXT PRIMARY KEY, kind TEXT, metadata_json TEXT)"
    )
    con.commit()
    return db_path


# ---------- fixture to stub dependencies in the bindings namespace ----------


@pytest.fixture
def isolate_restore(monkeypatch):
    # deterministic hashing and planning
    monkeypatch.setattr(b, "scan_manifest_headers", lambda **kw: _fake_manifest(5))
    monkeypatch.setattr(b, "shuffle_manifest", lambda m, s: list(reversed(m)))
    monkeypatch.setattr(b, "ensure_item_content_hashes", lambda m: m)
    monkeypatch.setattr(b, "compute_manifest_root_hash", lambda **kw: "rootHASH")
    monkeypatch.setattr(b, "compute_batch_hash", lambda root, items: f"BH{len(items)}")

    # batching
    def _plan(**kw):
        items = _fake_manifest(5)
        return _fake_plan(items, [2, 2, 1])

    monkeypatch.setattr(b, "plan_image_batches", _plan)
    monkeypatch.setattr(b, "iter_batches_from_plan", lambda plan: iter(plan))

    # decoders
    monkeypatch.setattr(
        b,
        "decode_batch_pil",
        lambda items, rt, color, ms: [f"im:{x['rel_path']}" for x in items],
    )
    monkeypatch.setattr(
        b,
        "decode_batch_cv2",
        lambda items, rt, color, ms: [f"im:{x['rel_path']}" for x in items],
    )

    # CAS loader
    def _load_cas_json(run_path, h, kind):
        if kind == "image.manifest":
            return {"root_hash": "rootHASH"}
        if kind == "image.batch.meta":
            # ordinal is not used here; entry i is requested positionally
            idx = int(h.split("_")[-1])  # e.g., "desc_1"
            count = [2, 2, 1][idx - 1]
            return {
                "manifest_hash": "manifest1",
                "ordinal": idx,
                "count": count,
                "batch_hash": f"BH{count}",
                "batch_desc_hash": h,
            }
        if kind == "image.batch":
            return {
                "manifest_hash": "manifest1",
                "batch_hash": "BH2",
                "ordinal": 1,
                "count": 2,
            }
        raise FileNotFoundError(h)

    monkeypatch.setattr(b, "_load_cas_json", _load_cas_json)

    yield


@pytest.fixture
def isolate_save(monkeypatch, tmp_path):
    # ---- fake Hash ----
    class FakeHash:
        def hash_bytes(self, x):
            return "hb_" + str(len(x))

        def hash_file(self, p):
            return "hf_" + p.name

    monkeypatch.setattr(b, "Hash", FakeHash)

    # ---- fake buffer with Arrow-like API ----
    class FakeBuffer:
        def to_pybytes(self):
            return b"xx"

    # ---- fake table ----
    class FakeTable:
        def __init__(self, n=3):
            self._rows = n
            self._schema = ["img_bytes", "path"]

        @property
        def num_rows(self):
            return self._rows

        @property
        def num_columns(self):
            return len(self._schema)

        @property
        def schema(self):
            return SimpleNamespace(names=self._schema)

        def column(self, name):
            if name == "img_bytes":
                # combine_chunks() returns an object exposing buffers()
                return SimpleNamespace(
                    to_pylist=lambda: [b"x"] * self._rows,
                    combine_chunks=lambda: SimpleNamespace(
                        buffers=lambda: [FakeBuffer()]
                    ),
                )
            if name == "path":
                return SimpleNamespace(to_pylist=lambda: ["/a.jpg"] * self._rows)
            return SimpleNamespace(to_pylist=lambda: [])

    # ---- stub helpers used by the function ----
    monkeypatch.setitem(
        b.__dict__, "_build_parquet_table_from_images", lambda **kw: FakeTable()
    )
    monkeypatch.setitem(
        b.__dict__,
        "_deterministic_shard_path",
        lambda td, prefix, ord: (td / f"{prefix}_{ord}.parquet"),
    )
    monkeypatch.setitem(
        b.__dict__, "_write_parquet_table", lambda table, path, **kw: path
    )  # no real I/O
    monkeypatch.setitem(
        b.__dict__, "resolve_conflict_path", lambda p, policy, suffix: p
    )
    monkeypatch.setitem(
        b.__dict__,
        "store",
        SimpleNamespace(record_materialization=lambda *a, **k: None),
    )
    monkeypatch.setitem(b.__dict__, "RECORDED_WRITER_CFG", {"engine": "pyarrow"})

    # ---- install fake pyarrow + pyarrow.parquet into sys.modules ----
    fake_pa = ModuleType("pyarrow")
    fake_pa.__version__ = "fake"

    def _md():
        # add num_columns to satisfy code: for c in range(md.num_columns)
        return SimpleNamespace(
            num_row_groups=1,
            num_rows=3,
            num_columns=2,
            created_by="fake",
            row_group=lambda i: SimpleNamespace(
                num_rows=3,
                column=lambda j: SimpleNamespace(
                    compression="none",
                    encodings=["PLAIN"],
                    dictionary_page_offset=None,
                    has_dictionary_page=None,
                ),
            ),
            metadata=None,
        )

    fake_pq = ModuleType("pyarrow.parquet")
    fake_pq.ParquetFile = lambda path: _md()
    fake_pq.read_metadata = lambda path: _md()
    fake_pq.read_table = lambda path: FakeTable()

    monkeypatch.setitem(sys.modules, "pyarrow", fake_pa)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", fake_pq)

    yield tmp_path


# ---------------- apply_registry ----------------


def test_apply_registry_maps_roles_to_kwargs():
    kwargs = {}
    realized = {"source": "/tmp/data.csv"}
    b.apply_registry("blase.Extract.read_csv", realized, kwargs)
    assert kwargs["file_path"] == "/tmp/data.csv"


def test_apply_registry_no_mapping_noop():
    kwargs = {}
    b.apply_registry("blase.Unknown.step", {"x": 1}, kwargs)
    assert kwargs == {}


# --------------- _pick_replay_target ---------------


def test__pick_replay_target_default_and_conflicts(tmp_path, monkeypatch):
    # deterministic timestamp for rename branch
    monkeypatch.setattr(
        b,
        "datetime",
        SimpleNamespace(
            now=lambda: SimpleNamespace(strftime=lambda fmt: "20250101-120000")
        ),
    )

    # 1) no target in params → fallback path, created
    p = b._pick_replay_target(
        params={}, target_override=None, on_conflict="overwrite", allow_append=False
    )
    assert p.name == "output.csv"
    assert p.parent.name == "replayed"  # data/working/replayed/output.csv
    assert p.parent.exists()

    # 2) target exists + allow_append -> returns same path
    t = tmp_path / "out.csv"
    t.write_text("old\n")
    p2 = b._pick_replay_target(
        {"target": str(t)}, None, on_conflict="fail", allow_append=True
    )
    assert p2 == t

    # 3) exists + fail -> raises
    with pytest.raises(FileExistsError):
        b._pick_replay_target(
            {"target": str(t)}, None, on_conflict="fail", allow_append=False
        )

    # 4) exists + overwrite -> returns the same path
    p4 = b._pick_replay_target(
        {"target": str(t)}, None, on_conflict="overwrite", allow_append=False
    )
    assert p4 == t

    # 5) exists + rename -> suffixed with timestamp
    p5 = b._pick_replay_target(
        {"target": str(t)}, None, on_conflict="rename", allow_append=False
    )
    assert p5.name.startswith("out_replayed_20250101-120000")


# --------------- run_read_csv_restore ---------------


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_run_read_csv_restore_backend_and_hash_check(tmp_path, monkeypatch, backend):
    # prepare a tiny file
    f = tmp_path / "src.csv"
    f.write_text("a\n")

    # mock the readers to yield exactly one batch
    called = {}

    def fake_reader(path, batch_size, use_cols, filter_by):
        called["used"] = "pandas" if backend == "pandas" else "polars"
        assert str(path) == str(f)
        yield (["row"], True)

    if backend == "pandas":
        monkeypatch.setattr(b, "read_batches_pandas", fake_reader)
    else:
        monkeypatch.setattr(b, "read_batches_polars", fake_reader)

    # hash verification: make it match
    monkeypatch.setattr(b.Hash, "hash_file", lambda self, p: "H")
    gen = b.run_read_csv_restore(
        run_path=tmp_path,
        params={"backend": backend},
        realized={"source": str(f), "__expected_source_hash__": "H"},
        transform_fn=None,
    )
    batch = next(gen)
    assert batch.data == ["row"] and batch.is_last is True
    assert called["used"] == backend


def test_run_read_csv_restore_hash_mismatch_raises(tmp_path, monkeypatch):
    f = tmp_path / "src.csv"
    f.write_text("x\n")
    monkeypatch.setattr(b.Hash, "hash_file", lambda self, p: "BAD")
    # reader stub (won't be reached due to error)
    monkeypatch.setattr(b, "read_batches_pandas", lambda *a, **k: iter([]))
    with pytest.raises(RuntimeError):
        list(
            b.run_read_csv_restore(
                run_path=tmp_path,
                params={"backend": "pandas"},
                realized={"source": str(f), "__expected_source_hash__": "EXPECTED"},
                transform_fn=None,
            )
        )


# --------------- run_apply_function_restore ---------------


def test_run_apply_function_restore_happy(tmp_path, monkeypatch):
    f = tmp_path / "src.csv"
    f.write_text("x\n")

    # make the batch reader yield two batches
    def reader(path, batch_size, use_cols, filter_by):
        yield Batch(data=["r1"], labels=None, paths=None, is_last=False, meta={})
        yield Batch(data=["r2"], labels=None, paths=None, is_last=True, meta={})

    monkeypatch.setattr(b, "read_batches_pandas", reader)

    # transform works on data (or make it Batch-aware if your impl prefers)
    def fn(data):
        return [x.upper() for x in data]

    out = list(
        b.run_apply_function_restore(
            run_path=tmp_path,
            params={"backend": "pandas"},
            realized={"source": str(f)},
            transform_fn=fn,
        )
    )

    assert out[0].data == ["R1"]
    assert out[0].is_last is False
    assert out[1].data == ["R2"]
    assert out[1].is_last is True


def test_run_apply_function_restore_missing_source_raises(tmp_path):
    with pytest.raises(RuntimeError):
        list(
            b.run_apply_function_restore(
                run_path=tmp_path,
                params={},
                realized={},  # no "source"
                transform_fn=lambda x: x,
            )
        )


# --------------- run_save_to_csv_replay ---------------


def _fake_saver(records):
    """Return a Load.save_to_csv replacement that appends 'records' to the tmp file."""

    def _save_to_csv(
        *, batch: Batch, path: str, backend=None, track=None, use_blase_path=None
    ) -> SinkResult:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if p.exists() else "w"
        with p.open(mode) as fh:
            for r in batch.data:
                fh.write(f"{r}\n")
        return SinkResult(
            artifacts=[Artifact(path=str(p), kind="csv")],
            is_last=batch.is_last,
            meta=dict(batch.meta or {}),
        )

    return _save_to_csv


def test_run_save_to_csv_replay_writes_and_overwrites(tmp_path, monkeypatch):
    target = tmp_path / "final.csv"

    def as_batches(seq):
        for data, last in seq:
            yield Batch(data=data, labels=None, paths=None, is_last=last, meta={})

    upstream = as_batches([(["a"], False), (["b"], True)])

    # saver uses new API
    monkeypatch.setattr(b.Load, "save_to_csv", staticmethod(_fake_saver(["a", "b"])))

    def fake_hash(self, p):
        text = Path(p).read_text()
        return "OK" if text.strip().splitlines() == ["a", "b"] else "BAD"

    monkeypatch.setattr(b.Hash, "hash_file", fake_hash)

    recorded = {}

    def record_materialization(run_path, data_hash, path):
        recorded["ok"] = data_hash == "OK" and Path(path) == target

    monkeypatch.setattr(b.store, "record_materialization", record_materialization)

    res = b.run_save_to_csv_replay(
        run_path=tmp_path,
        params={"target": str(target)},
        upstream_gen=upstream,
        on_conflict="overwrite",
        expected_out_hash="OK",
    )

    assert Path(res.artifacts[0].path) == target
    assert target.read_text().strip().splitlines() == ["a", "b"]
    assert recorded.get("ok") is True


def test_run_save_to_csv_replay_fail_without_upstream(tmp_path):
    with pytest.raises(RuntimeError):
        b.run_save_to_csv_replay(
            run_path=tmp_path,
            params={"target": str(tmp_path / "x.csv")},
            upstream_gen=None,
        )


def test_run_save_to_csv_replay_hash_mismatch_raises(tmp_path, monkeypatch):
    target = tmp_path / "x.csv"
    upstream = iter([Batch(data=(["one"], True), is_last=True, meta={})])
    monkeypatch.setattr(b.Load, "save_to_csv", staticmethod(_fake_saver(["one"])))
    monkeypatch.setattr(b.Hash, "hash_file", lambda self, p: "WRONG")
    with pytest.raises(RuntimeError):
        b.run_save_to_csv_replay(
            run_path=tmp_path,
            params={"target": str(target)},
            upstream_gen=upstream,
            expected_out_hash="EXPECTED",
        )


def test_run_save_to_csv_replay_rename_conflict(tmp_path, monkeypatch):
    # deterministic timestamp for rename
    monkeypatch.setattr(
        b,
        "datetime",
        SimpleNamespace(
            now=lambda: SimpleNamespace(strftime=lambda fmt: "20250101-120000")
        ),
    )
    # existing target
    target = tmp_path / "out.csv"
    target.write_text("old\n")

    upstream = iter([Batch(data=(["x"], True), is_last=True, meta={})])
    monkeypatch.setattr(b.Load, "save_to_csv", staticmethod(_fake_saver(["x"])))
    # any computed hash is fine since we don't pass expected_out_hash
    monkeypatch.setattr(b.Hash, "hash_file", lambda self, p: "H")

    res = b.run_save_to_csv_replay(
        run_path=tmp_path,
        params={"target": str(target)},
        upstream_gen=upstream,
        on_conflict="rename",
    )
    assert Path(res.artifacts[0].path) != target
    assert Path(res.artifacts[0].path).name == "out_replayed_20250101-120000.csv"
    assert Path(res.artifacts[0].path).exists()
    # original remains
    assert target.read_text() == "old\n"


def test_run_save_to_csv_replay_preseed_appends(tmp_path, monkeypatch):
    target = tmp_path / "out.csv"
    pre = tmp_path / "seed.csv"
    pre.write_text("seed\n")
    upstream = iter(
        [
            Batch(data=["n1"], labels=None, paths=None, is_last=False, meta={}),
            Batch(data=["n2"], labels=None, paths=None, is_last=True, meta={}),
        ]
    )

    monkeypatch.setattr(b.Load, "save_to_csv", staticmethod(_fake_saver(["n1", "n2"])))
    monkeypatch.setattr(b.Hash, "hash_file", lambda self, p: "HASH")

    res = b.run_save_to_csv_replay(
        run_path=tmp_path,
        params={"target": str(target)},
        upstream_gen=upstream,
        preseed_path=pre,
        on_conflict="overwrite",
    )

    lines = Path(res.artifacts[0].path).read_text().splitlines()
    assert lines == ["seed", "n1", "n2"]


# ----- run_read_images_restore tests -----
def test_restore_happy_path_with_manifest_and_batch_descs(isolate_restore, tmp_path):
    run_path = tmp_path
    params = {
        "backend": "pil",
        "return_type": "np",
        "color": "rgb",
        "mode": "auto",
        "pattern": "**/*.jpg",
        "recursive": True,
        "shuffle": False,
        "hash_mode": "content",
    }
    realized = {
        "source": "/data/images",
        "manifest": "manifest1",
        "batch_descs": ["desc_1", "desc_2", "desc_3"],
    }

    gen = b.run_read_images_restore(run_path=run_path, params=params, realized=realized)
    outs = list(gen)
    assert len(outs) == 3
    for i, batch in enumerate(outs, 1):
        assert isinstance(batch.data, list) and batch.data
        assert batch.meta["manifest_root_hash"] == "rootHASH"
        assert batch.meta["batch_hash"] == f"BH{2 if i < 3 else 1}"
        assert batch.meta["manifest"] == "manifest1"
        assert isinstance(batch.is_last, bool)
        # upstream should include manifest and batch entries
        roles = [u["role"] for u in batch.meta["upstream"] if u]
        assert "manifest" in roles and "batch" in roles


def test_restore_requires_source_or_directory(isolate_restore, tmp_path):
    params = {}
    realized = {}  # no source, no params['directory']
    with pytest.raises(RuntimeError):
        next(
            b.run_read_images_restore(
                run_path=tmp_path, params=params, realized=realized
            )
        )


def test_restore_shuffle_without_seed_raises(isolate_restore, tmp_path):
    params = {"shuffle": True}  # seed missing
    realized = {"source": "/x"}
    with pytest.raises(RuntimeError):
        next(
            b.run_read_images_restore(
                run_path=tmp_path, params=params, realized=realized
            )
        )


def test_restore_root_hash_mismatch_raises(isolate_restore, monkeypatch, tmp_path):
    # Make manifest CAS report a different root hash
    monkeypatch.setattr(
        b,
        "_load_cas_json",
        lambda rp, h, kind: {"root_hash": "DIFF"} if kind == "image.manifest" else {},
    )
    params = {"shuffle": False}
    realized = {"source": "/x", "manifest": "manifest1"}
    with pytest.raises(RuntimeError):
        next(
            b.run_read_images_restore(
                run_path=tmp_path, params=params, realized=realized
            )
        )


def test_restore_validates_batch_count_and_hash(isolate_restore, monkeypatch, tmp_path):
    # Use correct count but wrong hash for batch 2
    def _load_cas_json_bad_hash(run_path, h, kind):
        if kind == "image.manifest":
            return {"root_hash": "rootHASH"}
        if kind == "image.batch.meta":
            idx = int(h.split("_")[-1])
            count = [2, 2, 1][idx - 1]
            bad = "BAD" if idx == 2 else f"BH{count}"
            return {
                "manifest_hash": "manifest1",
                "ordinal": idx,
                "count": count,
                "batch_hash": bad,
            }
        raise FileNotFoundError

    monkeypatch.setattr(b, "_load_cas_json", _load_cas_json_bad_hash)

    params = {"shuffle": False}
    realized = {
        "source": "/x",
        "manifest": "manifest1",
        "batch_descs": ["desc_1", "desc_2", "desc_3"],
    }
    # Should fail on second batch due to hash mismatch
    gen = b.run_read_images_restore(run_path=tmp_path, params=params, realized=realized)
    # consume first ok batch
    next(gen)
    with pytest.raises(RuntimeError):
        next(gen)


# ----- _load_cas_json tests -----
def test_load_from_canonical(tmp_path):
    h = "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"
    payload = {"ok": True}
    p = b.cas.path_for(tmp_path, "image.manifest", h)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload))
    out = b._load_cas_json(tmp_path, h, kind="image.manifest")
    assert out == payload


def test_load_from_legacy_flat(tmp_path):
    h = "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"
    payload = {"legacy": True}
    p = tmp_path / "cas" / "sha256" / h[:2] / h[2:4] / h
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload))
    out = b._load_cas_json(tmp_path, h, kind="image.manifest")
    assert out == payload


def test_load_from_legacy_typo_meta(tmp_path):
    h = "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"
    payload = {"typo": True}
    # kind 'image.batch.meta' -> directory 'image.batch.met'
    p = tmp_path / "cas" / "sha256" / "image.batch.met" / h[:2] / h[2:4] / h
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload))
    out = b._load_cas_json(tmp_path, h, kind="image.batch.meta")
    assert out == payload


def test_load_from_reversed_order(tmp_path):
    h = "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"
    payload = {"rev": True}
    _write_blob(tmp_path, "image.manifest", h, payload, reversed_order=True)
    out = b._load_cas_json(tmp_path, h, kind="image.manifest")
    assert out == payload


def test_kind_required(tmp_path):
    h = "deadbeef" * 8
    with pytest.raises(ValueError):
        b._load_cas_json(tmp_path, h, kind=None)


def test_not_found_raises(tmp_path):
    h = "deadbeef" * 8
    with pytest.raises(FileNotFoundError):
        b._load_cas_json(tmp_path, h, kind="image.manifest")


# ----- run_save_images_to_parquet_replay tests -----
def test_basic_replay(isolate_save):
    run_path = isolate_save
    params = {"target_dir": str(run_path), "shard_prefix": "b"}

    # upstream generator yields two batches
    def upstream():
        yield Batch(
            data=["im1"], labels=None, paths=None, is_last=False, meta={"ordinal": 1}
        )
        yield Batch(
            data=["im2"], labels=None, paths=None, is_last=True, meta={"ordinal": 2}
        )

    res: SinkResult = b.run_save_images_to_parquet_replay(
        run_path=run_path,
        params=params,
        realized={},
        upstream_gen=upstream(),
    )

    # one sink result containing two parquet artifacts
    assert isinstance(res, SinkResult)
    assert len(res.artifacts) == 2
    paths = [a.path for a in res.artifacts]
    assert all(str(p).endswith(".parquet") for p in paths)


def test_expected_hash_mismatch_raises(isolate_save):
    run_path = isolate_save
    params = {"target_dir": str(run_path), "shard_prefix": "b"}

    # one batch
    def upstream():
        yield Batch(data="im1", meta={"ordinal": 1}, is_last=True)

    # expected hash deliberately mismatched
    with pytest.raises(IOError):
        b.run_save_images_to_parquet_replay(
            run_path=run_path,
            params=params,
            realized={},
            upstream_gen=upstream(),
            expected_out_hashes=["different_hash"],
        )


def test_target_override_and_on_conflict(isolate_save):
    run_path = isolate_save
    override_dir = run_path / "custom"
    params = {"shard_prefix": "c"}

    def upstream():
        yield Batch(data="im1", meta={"ordinal": 1}, is_last=True)

    res = b.run_save_images_to_parquet_replay(
        run_path=run_path,
        params=params,
        realized={},
        upstream_gen=upstream(),
        target_override=str(override_dir),
        on_conflict="overwrite",
    )
    # directory override respected
    assert all(Path(a.path).parent == override_dir for a in res.artifacts)
