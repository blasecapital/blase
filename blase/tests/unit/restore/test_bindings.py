from types import SimpleNamespace
from pathlib import Path
import shutil
import pytest
from _pytest.monkeypatch import MonkeyPatch

import blase.restoring.bindings as b

@pytest.fixture(autouse=True)
def _bindings_sandbox(tmp_path, monkeypatch):
    # Force all relative paths (like 'data/...') under a temp dir
    monkeypatch.chdir(tmp_path)
    yield
    # Clean anything this module created
    shutil.rmtree(tmp_path / "data", ignore_errors=True)
    shutil.rmtree(tmp_path / "runs", ignore_errors=True)
    shutil.rmtree(tmp_path / "restore", ignore_errors=True)

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
    monkeypatch.setattr(b, "datetime", SimpleNamespace(
        now=lambda: SimpleNamespace(strftime=lambda fmt: "20250101-120000")
    ))

    # 1) no target in params → fallback path, created
    p = b._pick_replay_target(params={}, target_override=None, on_conflict="overwrite", allow_append=False)
    assert p.name == "output.csv"
    assert p.parent.name == "replayed"  # data/working/replayed/output.csv
    assert p.parent.exists()

    # 2) target exists + allow_append -> returns same path
    t = tmp_path / "out.csv"
    t.write_text("old\n")
    p2 = b._pick_replay_target({"target": str(t)}, None, on_conflict="fail", allow_append=True)
    assert p2 == t

    # 3) exists + fail -> raises
    with pytest.raises(FileExistsError):
        b._pick_replay_target({"target": str(t)}, None, on_conflict="fail", allow_append=False)

    # 4) exists + overwrite -> returns the same path
    p4 = b._pick_replay_target({"target": str(t)}, None, on_conflict="overwrite", allow_append=False)
    assert p4 == t

    # 5) exists + rename -> suffixed with timestamp
    p5 = b._pick_replay_target({"target": str(t)}, None, on_conflict="rename", allow_append=False)
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
        called["used"] = ("pandas" if backend == "pandas" else "polars")
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
    batch, last = next(gen)
    assert batch == ["row"] and last is True
    assert called["used"] == backend

def test_run_read_csv_restore_hash_mismatch_raises(tmp_path, monkeypatch):
    f = tmp_path / "src.csv"; f.write_text("x\n")
    monkeypatch.setattr(b.Hash, "hash_file", lambda self, p: "BAD")
    # reader stub (won't be reached due to error)
    monkeypatch.setattr(b, "read_batches_pandas", lambda *a, **k: iter([]))
    with pytest.raises(RuntimeError):
        list(b.run_read_csv_restore(
            run_path=tmp_path,
            params={"backend": "pandas"},
            realized={"source": str(f), "__expected_source_hash__": "EXPECTED"},
            transform_fn=None,
        ))

# --------------- run_apply_function_restore ---------------

def test_run_apply_function_restore_happy(tmp_path, monkeypatch):
    f = tmp_path / "src.csv"; f.write_text("x\n")

    # make the batch reader yield two batches
    def reader(path, batch_size, use_cols, filter_by):
        yield (["r1"], False)
        yield (["r2"], True)
    monkeypatch.setattr(b, "read_batches_pandas", reader)

    fn = lambda batch: [x.upper() for x in batch]
    out = list(b.run_apply_function_restore(
        run_path=tmp_path,
        params={"backend": "pandas"},
        realized={"source": str(f)},
        transform_fn=fn,
    ))
    assert out == [(["R1"], False), (["R2"], True)]

def test_run_apply_function_restore_missing_source_raises(tmp_path):
    with pytest.raises(RuntimeError):
        list(b.run_apply_function_restore(
            run_path=tmp_path,
            params={},
            realized={},          # no "source"
            transform_fn=lambda x: x,
        ))


# --------------- run_save_to_csv_replay ---------------

def _fake_saver(records):
    """Return a Load.save_to_csv replacement that appends 'records' to the tmp file."""
    def _save_to_csv(*, data, last_batch, meta, path, backend, track, use_blase_path):
        # append one line per element in 'data'
        p = Path(path)
        mode = "a" if p.exists() else "w"
        with p.open(mode) as fh:
            for r in data:
                fh.write(str(r) + "\n")
    return _save_to_csv

def test_run_save_to_csv_replay_writes_and_overwrites(tmp_path, monkeypatch):
    target = tmp_path / "final.csv"
    # upstream yields two batches
    upstream = iter([(["a"], False), (["b"], True)])
    # make Load.save_to_csv write to the tmp path
    monkeypatch.setattr(b.Load, "save_to_csv", staticmethod(_fake_saver(["a","b"])))
    # hash matches after write
    def fake_hash(self, p):
        # temp file should contain two lines "a\nb\n"
        text = Path(p).read_text()
        return "OK" if text.strip().splitlines() == ["a", "b"] else "BAD"
    monkeypatch.setattr(b.Hash, "hash_file", fake_hash)

    # make sure record_materialization is called
    recorded = {}
    def record_materialization(run_path, data_hash, path):
        recorded["ok"] = (data_hash == "OK" and Path(path) == target)
    monkeypatch.setattr(b.store, "record_materialization", record_materialization)

    out = b.run_save_to_csv_replay(
        run_path=tmp_path,
        params={"target": str(target)},
        upstream_gen=upstream,
        on_conflict="overwrite",
        expected_out_hash="OK",
    )
    assert Path(out) == target
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
    upstream = iter([(["one"], True)])
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
    monkeypatch.setattr(b, "datetime", SimpleNamespace(
        now=lambda: SimpleNamespace(strftime=lambda fmt: "20250101-120000")
    ))
    # existing target
    target = tmp_path / "out.csv"
    target.write_text("old\n")

    upstream = iter([(["x"], True)])
    monkeypatch.setattr(b.Load, "save_to_csv", staticmethod(_fake_saver(["x"])))
    # any computed hash is fine since we don't pass expected_out_hash
    monkeypatch.setattr(b.Hash, "hash_file", lambda self, p: "H")

    out = b.run_save_to_csv_replay(
        run_path=tmp_path,
        params={"target": str(target)},
        upstream_gen=upstream,
        on_conflict="rename",
    )
    assert Path(out) != target
    assert Path(out).name == "out_replayed_20250101-120000.csv"
    assert Path(out).exists()
    # original remains
    assert target.read_text() == "old\n"

def test_run_save_to_csv_replay_preseed_appends(tmp_path, monkeypatch):
    target = tmp_path / "out.csv"
    pre = tmp_path / "seed.csv"
    pre.write_text("seed\n")
    upstream = iter([(["n1"], False), (["n2"], True)])
    monkeypatch.setattr(b.Load, "save_to_csv", staticmethod(_fake_saver(["n1","n2"])))
    # hash returns something, not enforcing a particular value
    monkeypatch.setattr(b.Hash, "hash_file", lambda self, p: "HASH")

    out = b.run_save_to_csv_replay(
        run_path=tmp_path,
        params={"target": str(target)},
        upstream_gen=upstream,
        preseed_path=pre,
        on_conflict="overwrite",
    )
    lines = Path(out).read_text().splitlines()
    # should include seed + new lines (order preserved)
    assert lines == ["seed", "n1", "n2"]
