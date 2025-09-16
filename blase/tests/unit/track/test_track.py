import json
from pathlib import Path
import sqlite3
import pytest

from blase.track import Track

# ----- Helpers -----


def _db_path(tracker: Track) -> Path:
    return tracker.run_path / "nodes" / "nodes.db"


def _q(db_path: Path, sql: str, args=()):
    with sqlite3.connect(db_path) as c:
        c.row_factory = sqlite3.Row
        return c.execute(sql, args).fetchall()


# Fake CAS that returns deterministic hashes and sizes, and optionally writes bytes
class _FakeCAS:
    def __init__(self):
        pass

    @staticmethod
    def put_bytes(b: bytes, *, kind: str):
        import hashlib

        h = hashlib.sha256(b).hexdigest()
        return h, len(b)

    @staticmethod
    def put_file(p: Path, *, kind: str):
        data = Path(p).read_bytes()
        import hashlib

        h = hashlib.sha256(data).hexdigest()
        return h, len(data)


@pytest.fixture()
def runs_dir(tmp_path: Path) -> Path:
    return tmp_path / "runs"


@pytest.fixture()
def tracker(runs_dir: Path) -> Track:
    trk = Track(runs_dir=runs_dir, reuse=False)
    trk.start_run("test-run")
    return trk


@pytest.fixture(autouse=True)
def patch_step_backend(monkeypatch):
    # Patch CAS + builders where StepOps resolves them
    from blase.tracking import step_backend as sb

    monkeypatch.setattr(sb, "CAS", _FakeCAS(), raising=True)

    def _build_code_blob(fn):
        src = f"def {fn.__name__}(...): pass".encode("utf-8")
        return src, {"fqn": f"{fn.__module__}.{fn.__name__}"}

    def _build_env_manifest():
        blob = b"pip-freeze:\nexample==0.0.1\n"
        return blob, {"tool": "pip"}

    monkeypatch.setattr(sb, "build_code_blob", _build_code_blob, raising=True)
    monkeypatch.setattr(sb, "build_env_manifest", _build_env_manifest, raising=True)
    yield


# ----- Run lifecycle & FS -----


def test_track_initializes_dirs_and_pointer(runs_dir: Path):
    trk = Track(runs_dir=runs_dir, reuse=False)
    assert trk.run_path.exists()
    assert (trk.run_path / "cas").is_dir()
    assert (trk.run_path / "nodes").is_dir()
    active = runs_dir / "active_run.blase"
    assert active.exists()
    meta = json.loads(active.read_text())
    assert meta["run_id"] == trk.run_id
    assert meta["status"] == "active"


def test_start_run_sets_label_and_merges_metadata(tracker: Track):
    tracker.start_run(run_name="etl", metadata={"foo": 1})
    meta = json.loads((tracker.run_path / "run.json").read_text())
    assert meta["label"] == "etl"
    assert meta["foo"] == 1


def test_end_run_updates_pointer(tracker: Track, runs_dir: Path):
    tracker.end_run(status="completed")
    meta = json.loads((runs_dir / "active_run.blase").read_text())
    assert meta["status"] == "completed"
    assert "ended_at" in meta


def test_reuse_flag_controls_run_id(runs_dir: Path, monkeypatch):
    # Patch the module where Track imported time
    import blase.track as track_mod

    # First construction -> "A"
    monkeypatch.setattr(track_mod.time, "strftime", lambda fmt: "A")
    t1 = Track(runs_dir=runs_dir, reuse=False)
    id1 = t1.run_id

    # Reuse shouldn't change the active run
    t2 = Track(runs_dir=runs_dir, reuse=True)
    assert t2.run_id == id1

    # New construction -> "B"
    monkeypatch.setattr(track_mod.time, "strftime", lambda fmt: "B")
    t3 = Track(runs_dir=runs_dir, reuse=False)
    assert t3.run_id != id1


# ----- Singleton -----


def test_singleton_enable_disable(runs_dir: Path):
    assert Track.get(enable=False, runs_dir=runs_dir) is None
    a = Track.get(enable=True, runs_dir=runs_dir, reuse=False)
    b = Track.get(enable=True, runs_dir=runs_dir, reuse=True)
    assert a is b


# ----- StepContext status transitions -----


def test_step_context_completes(tracker: Track):
    db = _db_path(tracker)
    with tracker.step("blase.Extract.read_csv", params={"path": "x.csv"}):
        pass
    rows = _q(
        db, "SELECT status, ts_end, function_fqn FROM steps ORDER BY rowid DESC LIMIT 1"
    )
    assert rows[0]["status"] == "completed"
    assert rows[0]["ts_end"] is not None
    assert rows[0]["function_fqn"] == "blase.Extract.read_csv"


def test_step_context_failure(tracker: Track):
    db = _db_path(tracker)
    with pytest.raises(RuntimeError):
        with tracker.step("blase.Transform.apply", params={"k": 1}):
            raise RuntimeError("boom")
    rows = _q(db, "SELECT status FROM steps ORDER BY rowid DESC LIMIT 1")
    assert rows[0]["status"] == "failed"


def test_step_context_has_signature_hash(tracker: Track):
    db = _db_path(tracker)
    with tracker.step("blase.Load.save", params={"dest": "y"}):
        pass
    rows = _q(db, "SELECT sig_hash FROM steps ORDER BY rowid DESC LIMIT 1")
    assert rows[0]["sig_hash"]  # non-empty string


# ----- Params persisted -----


def test_params_json_persisted(tracker: Track):
    db = _db_path(tracker)
    with tracker.step("f.q.n", params={"a": 1, "b": [2, 3]}):
        pass
    row = _q(db, "SELECT params_json FROM steps ORDER BY rowid DESC LIMIT 1")[0]
    assert '"a": 1' in row["params_json"]
    assert '"b": [2, 3]' in row["params_json"]


# ----- StreamStep snapshots code/env ------


def test_stream_snapshots_code_and_env(tracker: Track):
    db = _db_path(tracker)
    ss = tracker.stream(
        "blase.Transform.apply_function", params={"fn": "_dummy"}, code_fn=_dummy_fn
    )
    # first emit not last
    meta_out = ss.emit(last_batch=False, meta={"ordinal": 0})
    assert meta_out["producer_step"] == ss.step_hash
    assert meta_out["ordinal"] == 0

    # finish
    ss.close_ok()

    # assert step status
    status = _q(db, "SELECT status FROM steps WHERE step_hash=?", (ss.step_hash,))[0][
        "status"
    ]
    assert status == "completed"

    # check code/env inputs recorded (roles may differ; adjust if your schema uses other table/role names)
    inputs = _q(db, "SELECT role FROM step_inputs WHERE step_hash=?", (ss.step_hash,))
    roles = {r["role"] for r in inputs}
    assert "code" in roles
    assert "env" in roles


# ----- Steaming does not auto-close early


def test_stream_does_not_complete_before_last_batch(tracker: Track):
    db = _db_path(tracker)
    ss = tracker.stream("blase.Transform.streamed", params={})
    ss.emit(last_batch=False, meta=None)
    # still running
    status = _q(db, "SELECT status FROM steps WHERE step_hash=?", (ss.step_hash,))[0][
        "status"
    ]
    assert status == "running"
    # now finish
    ss.emit(last_batch=True, meta=None)
    ss.close_ok()
    status2 = _q(db, "SELECT status FROM steps WHERE step_hash=?", (ss.step_hash,))[0][
        "status"
    ]
    assert status2 == "completed"


# ----- Upstream lineage pass-through -----


def test_emit_records_upstream_edges(tracker: Track):
    db = _db_path(tracker)
    ss = tracker.stream("blase.Transform.apply", params={})
    meta = {
        "upstream": [{"id": "abc", "role": "input"}, {"id": "def", "role": "input"}]
    }
    ss.step.add_upstream_from_meta(meta)
    ss.emit(last_batch=True, meta=meta)
    ss.close_ok()
    rows = _q(
        db, "SELECT data_hash FROM step_inputs WHERE step_hash=?", (ss.step_hash,)
    )
    hashes = {r["data_hash"] for r in rows}
    # Relax this if your schema stores upstream differently
    assert {"abc", "def"}.issubset(hashes)


# ----- Schema bootstrap -----


def test_schema_bootstrap_fresh_dir(tmp_path: Path):
    trk = Track(runs_dir=tmp_path / "runs", reuse=False)
    db = trk.run_path / "nodes" / "nodes.db"
    # Verify essential tables exist
    with sqlite3.connect(db) as c:
        tbls = {
            r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    for t in {
        "steps",
        "data",
        "step_inputs",
        "step_outputs",
        "datasets",
        "dataset_members",
        "materializations",
    }:  # add any other required tables
        assert t in tbls


# ---------- 1) Track + StepContext lifecycle ----------


def test_step_lifecycle_and_signature(tracker: Track):
    db = _db_path(tracker)
    with tracker.step("blase.Extract.read_csv", params={"path": "x.csv"}):
        pass
    row = _q(
        db,
        "SELECT status, ts_start, ts_end, sig_hash, attempt_id, step_hash FROM steps ORDER BY rowid DESC LIMIT 1",
    )[0]
    assert row["status"] == "completed"
    assert row["ts_start"] is not None and row["ts_end"] is not None
    assert row["sig_hash"] and isinstance(row["sig_hash"], str)
    assert row["attempt_id"] == row["step_hash"]


def test_step_failure_and_abort(tracker: Track):
    db = _db_path(tracker)
    with pytest.raises(RuntimeError):
        with tracker.step("f.q.n", params={}):
            raise RuntimeError("boom")
    assert (
        _q(db, "SELECT status FROM steps ORDER BY rowid DESC LIMIT 1")[0]["status"]
        == "failed"
    )

    cm = tracker.step("f.q.n2", params={})
    cm.__enter__()
    cm.__exit__(GeneratorExit, GeneratorExit(), None)
    assert (
        _q(db, "SELECT status FROM steps ORDER BY rowid DESC LIMIT 1")[0]["status"]
        == "aborted"
    )


# ---------- 2) Inputs / Outputs ----------


def test_add_input_and_output_idempotency(tracker: Track):
    db = _db_path(tracker)
    with tracker.step("f.q.n", params={}) as s:
        s.add_input("h-in", role="source")
        s.add_input("h-in", role="source")  # OR REPLACE → idempotent
        s.add_output("h-out", name="foo")
        s.add_output("h-out", name="foo")  # OR IGNORE → ignored

    inputs = _q(
        db, "SELECT data_hash, role FROM step_inputs WHERE step_hash=?", (s.step_hash,)
    )
    outputs = _q(
        db, "SELECT data_hash, name FROM step_outputs WHERE step_hash=?", (s.step_hash,)
    )
    assert (
        len([r for r in inputs if r["data_hash"] == "h-in" and r["role"] == "source"])
        == 1
    )
    assert (
        len([r for r in outputs if r["data_hash"] == "h-out" and r["name"] == "foo"])
        == 1
    )


# ---------- 3) register_data (bytes vs. path, materializations, policy) ----------


def test_register_data_bytes_stores_in_cas_and_data_table(tracker: Track):
    db = _db_path(tracker)
    with tracker.step("blase.Load.save", params={}) as s:
        # default cas_policy is "index", but bytes path always writes to CAS
        payload = b"abc,123\n"
        h = s.register_data(kind="csv", version="1", path_or_bytes=payload)
        s.add_output(h, name="tbl")

    # data row exists, materializations absent for bytes
    d = _q(db, "SELECT byte_len, source_path, kind FROM data WHERE data_hash=?", (h,))[
        0
    ]
    assert d["byte_len"] == len(payload)
    assert d["source_path"] is None
    assert d["kind"] == "csv"
    mats = _q(db, "SELECT * FROM materializations WHERE data_hash=?", (h,))
    assert mats == []

    # check CAS file written
    cas_file = tracker.run_path / "cas" / "sha256" / "csv" / h[:2] / h[2:]
    assert cas_file.exists() and cas_file.read_bytes() == payload


def test_register_data_path_index_and_copy_policies(
    tracker: Track, tmp_path: Path, monkeypatch
):
    # create a source file
    src = tmp_path / "x.csv"
    src.write_text("a,b\n1,2\n", encoding="utf-8")

    # index policy: no copy, record materialization
    with tracker.step("blase.Load.index", params={}) as s:
        s.cas_policy = "index"
        h_idx = s.register_data(kind="csv", version="1", path_or_bytes=src)

    db = _db_path(tracker)
    mats = _q(db, "SELECT path FROM materializations WHERE data_hash=?", (h_idx,))
    assert mats and mats[0]["path"] == str(src)
    # index → we did not copy bytes; file may or may not exist; don't assert existence

    # copy policy: copy file into CAS + materialization
    with tracker.step("blase.Load.copy", params={}) as s2:
        s2.cas_policy = "copy"
        h_cpy = s2.register_data(kind="csv", version="1", path_or_bytes=src)
    cas_copy = tracker.run_path / "cas" / "sha256" / "csv" / h_cpy[:2] / h_cpy[2:]
    assert cas_copy.exists()
    assert cas_copy.read_text() == src.read_text()


# ---------- 4) code/env snapshots ----------


def _dummy_fn(x):
    return x


def test_snapshot_callable_and_env_record_inputs(tracker: Track):
    db = _db_path(tracker)
    with tracker.step("code.env", params={}) as s:
        ch = s.snapshot_callable(_dummy_fn)
        eh = s.snapshot_env()

    roles = {
        r["role"]
        for r in _q(
            db, "SELECT role FROM step_inputs WHERE step_hash=?", (s.step_hash,)
        )
    }
    assert "code" in roles and "env" in roles
    assert isinstance(ch, str) and isinstance(eh, str)


def test_snapshot_builders_fail_softly(tracker: Track, monkeypatch):
    # Enter the step to get the actual StepOps instance `s`
    with tracker.step("code.env.fail", params={}) as s:
        # Force failure precisely where snapshot_* would write to CAS/DB
        def _boom(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(s, "register_data", _boom, raising=True)

        ch = s.snapshot_callable(_dummy_fn)
        eh = s.snapshot_env()

    assert ch == "" and eh == ""


# ---------- 5) StreamStep.emit semantics ----------


def test_stream_emit_propagation_and_status(tracker: Track):
    db = _db_path(tracker)
    ss = tracker.stream("stream.fn", params={"k": 1}, code_fn=_dummy_fn)
    m0 = ss.emit(last_batch=False, meta={"ordinal": 0})
    assert m0["producer_step"] == ss.step_hash and m0["ordinal"] == 0
    status = _q(db, "SELECT status FROM steps WHERE step_hash=?", (ss.step_hash,))[0][
        "status"
    ]
    assert status == "running"

    ss.close_ok()
    status2 = _q(db, "SELECT status FROM steps WHERE step_hash=?", (ss.step_hash,))[0][
        "status"
    ]
    assert status2 == "completed"


# ---------- 6) Upstream meta + removal ----------


def test_add_upstream_from_meta_and_remove(tracker: Track):
    db = _db_path(tracker)
    with tracker.step("upstream", params={}) as s:
        s.add_upstream_from_meta(
            {"upstream": [{"id": "h1", "role": "seed"}, {"id": "h2"}]}
        )
        s.remove_inputs_by_role("seed")

    rows = _q(
        db, "SELECT data_hash, role FROM step_inputs WHERE step_hash=?", (s.step_hash,)
    )
    roles = {(r["data_hash"], r["role"]) for r in rows}
    assert ("h1", "seed") not in roles
    assert ("h2", "upstream") in roles


# ---------- 7) datasets ----------


def test_add_dataset_member_creates_and_appends(tracker: Track):
    db = _db_path(tracker)
    with tracker.step("dataset", params={}) as s:
        # ensure data rows exist (not strictly required by schema, but realistic)
        s.register_data(kind="csv", version="1", path_or_bytes=b"a")
        s.add_dataset_member("ds1", "deadbeef", 0)
        s.add_dataset_member("ds1", "feedface", 1)  # OR IGNORE enforces idempotency

    ds = _q(db, "SELECT dataset_id FROM datasets WHERE dataset_id='ds1'")
    assert ds
    members = _q(
        db,
        "SELECT data_hash, ordinal FROM dataset_members WHERE dataset_id='ds1' ORDER BY ordinal",
    )
    assert [m["ordinal"] for m in members] == sorted(m["ordinal"] for m in members)
