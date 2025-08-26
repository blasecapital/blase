import json, sqlite3, shutil
from types import SimpleNamespace
from pathlib import Path
import pytest

from blase.cli.restore import restore_cli as rc

# ----- Fixtures and helpers -----

# Minimal schema (only what restore_cli queries)
DDL = [
    """CREATE TABLE IF NOT EXISTS steps (
         step_hash TEXT PRIMARY KEY, run_id TEXT, step_id TEXT,
         function_fqn TEXT, params_json TEXT, seeds_json TEXT,
         ts_start TEXT, ts_end TEXT, attempt_id TEXT, sig_hash TEXT,
         status TEXT NOT NULL
       )""",
    """CREATE TABLE IF NOT EXISTS step_inputs (
         step_hash TEXT, data_hash TEXT, role TEXT, arg_name TEXT,
         PRIMARY KEY (step_hash, data_hash, role)
       )""",
    """CREATE TABLE IF NOT EXISTS step_outputs (
         step_hash TEXT, data_hash TEXT, name TEXT,
         PRIMARY KEY (step_hash, name), UNIQUE (data_hash)
       )""",
    """CREATE TABLE IF NOT EXISTS data (
         data_hash TEXT PRIMARY KEY, kind TEXT, version TEXT,
         byte_len INTEGER, source_path TEXT, metadata_json TEXT
       )""",
    """CREATE TABLE IF NOT EXISTS materializations (
         data_hash TEXT, path TEXT, ts TEXT, PRIMARY KEY (data_hash, path)
       )""",
]


@pytest.fixture(autouse=True)
def _isolate_and_redirect(tmp_path, monkeypatch):
    # isolate cwd
    monkeypatch.chdir(tmp_path)

    # force runs root under tmp
    import blase.cli.restore.restore_cli as rc
    runs_root = tmp_path / "runs"
    monkeypatch.setattr(rc, "_runs_root", lambda cwd=None: runs_root)

    # redirect default restore dir in BOTH places:
    # 1) the config module (future imports)
    from blase.utils import config
    restore_dir = tmp_path / "restore"
    monkeypatch.setattr(config, "RESTORE_DEFAULT_DIR", restore_dir, raising=False)
    # 2) the symbol already imported into restore_cli at import time
    monkeypatch.setattr(rc, "RESTORE_DEFAULT_DIR", restore_dir, raising=False)

    yield

    # teardown: clean anything created
    shutil.rmtree(runs_root, ignore_errors=True)
    shutil.rmtree(restore_dir, ignore_errors=True)
    # also nuke the default "runs/restored" in case some path still used it
    shutil.rmtree(tmp_path / "runs" / "restored", ignore_errors=True)

@pytest.fixture
def run_dir(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    run_id = "2025-01-02T03-04-05"
    run_path = runs / run_id
    (run_path / "nodes").mkdir(parents=True, exist_ok=True)
    (run_path / "cas" / "sha256" / "csv").mkdir(parents=True, exist_ok=True)
    # active pointer
    (runs / "active_run.blase").write_text(json.dumps({"run_id": run_id, "status": "active"}, indent=2))
    # minimal nodes.db
    db = run_path / "nodes" / "nodes.db"
    with sqlite3.connect(db) as c:
        for stmt in DDL: c.execute(stmt)
        c.commit()
    return run_path

def args(**kw):
    # argparse.Namespace-like
    return SimpleNamespace(**kw)

# ----- Run/DB discovery + guards -----

def test__resolve_run_path_active_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs").mkdir()  # no active_run.blase
    import pytest
    with pytest.raises(SystemExit):
        rc._resolve_run_path(None)

def test__resolve_run_path_by_id_and_by_path(run_dir, monkeypatch):
    # cwd must let _runs_root() find the runs dir
    monkeypatch.chdir(run_dir.parent)  # the "runs" directory
    # by id
    p = rc._resolve_run_path(run_dir.name)
    assert p.resolve() == run_dir.resolve()
    # by path
    p2 = rc._resolve_run_path(str(run_dir))
    assert p2.resolve() == run_dir.resolve()

def test__open_db_missing_raises(tmp_path):
    with pytest.raises(SystemExit):
        rc._open_db(tmp_path / "runs" / "nope")

def test__assert_path_matches_hash_ok_and_mismatch(tmp_path):
    from blase.utils.hashing import Hash
    f = tmp_path / "x.csv"
    f.write_text("a,b\n1,2\n", encoding="utf-8")
    h = Hash().hash_file(f)
    rc._assert_path_matches_hash(f, h, "csv")
    with pytest.raises(SystemExit):
        rc._assert_path_matches_hash(f, "deadbeef", "csv")

# ----- Source/seed resolution (no-copy / ephemeral) -----

def test__resolve_source_no_copy_happy(run_dir, monkeypatch, tmp_path):
    # store.load_data_source_path -> existing path with matching hash
    src = tmp_path / "src.csv"; src.write_text("x\n", encoding="utf-8")
    from blase.utils.hashing import Hash; h = Hash().hash_file(src)

    class Store:
        @staticmethod
        def load_data_source_path(run_path, data_hash): return str(src)
        @staticmethod
        def get_data_kind(run_path, h): return "csv"
    monkeypatch.setattr(rc.store, "load_data_source_path", Store.load_data_source_path)
    monkeypatch.setattr(rc.store, "get_data_kind", Store.get_data_kind)

    path, created = rc._resolve_source_no_copy(run_dir, h, seen_steps=set(), created_paths=[])
    assert path == src and created is False

def test__resolve_source_no_copy_fallback_replay(run_dir, monkeypatch, tmp_path):
    # No original; ensure_local via helper
    called = {}
    def ensure(run_path, data_hash, kind, **_): 
        called["ok"]=True; p = tmp_path / f"{data_hash}.csv"; p.write_text("y\n"); return p
    monkeypatch.setattr(
        rc,
        "_ensure_data_local_or_replay",
        lambda *a, **k: (tmp_path / "h.csv", True),
    )
    monkeypatch.setattr(rc.store, "load_data_source_path", lambda *a, **k: None)
    monkeypatch.setattr(rc.store, "get_data_kind", lambda *a, **k: "csv")

    created_paths = []
    p, created = rc._resolve_source_no_copy(run_dir, "h", created_paths=created_paths)
    assert created and p in created_paths

def test__resolve_seed_no_copy_or_ephemeral_order(run_dir, monkeypatch, tmp_path):
    # 1) materialized path ok
    mat = tmp_path / "m.csv"; mat.write_text("z\n")
    h = rc.Hash().hash_file(mat)
    monkeypatch.setattr(rc.store, "get_materialized_path", lambda *_: mat)
    monkeypatch.setattr(rc.Hash(), "hash_file", lambda *_: h)  # shortcut not strictly needed
    p, created = rc._resolve_seed_no_copy_or_ephemeral(run_dir, h)
    assert p == mat and not created

def test__resolve_seed_ephemeral_replay(run_dir, monkeypatch, tmp_path):
    # no mat, no recorded source, but has producer → write tmp and return True
    monkeypatch.setattr(rc.store, "get_materialized_path", lambda *a, **k: None)
    monkeypatch.setattr(rc.store, "get_recorded_source_path", lambda *a, **k: None)
    monkeypatch.setattr(rc.store, "producer_step_for_data", lambda *a, **k: "stepX")

    # Write the expected temp file inside _exec_plan_for_step
    def fake_exec(run_path, prod, to_path, **_):
        p = Path(to_path); p.write_text("seed\n")
        return {}
    monkeypatch.setattr(rc, "_exec_plan_for_step", fake_exec)

    # compute expected hash from what fake_exec writes
    tmp = tmp_path / "probe.csv"; tmp.write_text("seed\n")
    h = rc.Hash().hash_file(tmp)

    created_paths = []
    p, created = rc._resolve_seed_no_copy_or_ephemeral(run_dir, h, created_paths=created_paths)
    assert created and p.exists() and p in created_paths
    assert rc.Hash().hash_file(p) == h

def test__resolve_seed_no_producer_raises(run_dir, monkeypatch):
    monkeypatch.setattr(rc.store, "get_materialized_path", lambda *a, **k: None)
    monkeypatch.setattr(rc.store, "get_recorded_source_path", lambda *a, **k: None)
    monkeypatch.setattr(rc.store, "producer_step_for_data", lambda *a, **k: None)
    with pytest.raises(SystemExit):
        rc._resolve_seed_no_copy_or_ephemeral(run_dir, "deadbeef")

# ----- Plan scanning / upstream builder -----

def test__upstream_gen_for_sink_prefers_transform(run_dir, monkeypatch):
    # Make the plan contain the sink and a transform right before it
    monkeypatch.setattr(
        rc.planner, "plan_for_step",
        lambda *_: [
            {"step_hash": "A",    "function_fqn": "blase.Extract.read_csv"},
            {"step_hash": "B",    "function_fqn": "blase.Transform.apply_function"},
            {"step_hash": "SINK", "function_fqn": "blase.Load.save_to_csv"},
        ]
    )

    # Patch the actual function that _upstream_gen_for_sink calls:
    # it imports `from blase import restore as restore_mod` and uses restore_mod.step(...)
    import blase.restore as restore_mod
    called = {}
    def fake_restore_step(run_path, step_hash, kind):
        called["hash"] = step_hash
        # return a trivial generator
        return iter([([], True)])

    monkeypatch.setattr(restore_mod, "step", fake_restore_step)

    # Run
    gen = rc._upstream_gen_for_sink(run_dir, "SINK")
    next(gen)  # consume one item

    # Assert it picked the transform ("B"), not the extract
    assert called["hash"] == "B"

def test__upstream_gen_for_sink_no_upstream_raises(run_dir, monkeypatch):
    monkeypatch.setattr(rc.planner, "plan_for_step", lambda *_: [{"step_hash":"SINK","function_fqn":"blase.Load.save_to_csv"}])
    with pytest.raises(SystemExit):
        rc._upstream_gen_for_sink(run_dir, "SINK")

# ----- _normalize_plan_nodes branches (dict/tuple/str variants) -----

def test__normalize_plan_nodes_all_forms(run_dir, monkeypatch):
    def load_step(_, h):
        return {"function_fqn": f"fqn.{h}", "params": {}}
    monkeypatch.setattr(rc.store, "load_step", load_step)

    nodes = rc._normalize_plan_nodes(run_dir, [
        {"step_hash":"A","function_fqn":"fqn.A"},
        {"hash":"B"},
        {"id":"C"},
        ("D", "extra"),
        "E",
    ])
    got = [(n["step_hash"], n["function_fqn"]) for n in nodes]
    assert got == [("A","fqn.A"), ("B","fqn.B"), ("C","fqn.C"), ("D","fqn.D"), ("E","fqn.E")]

# ----- _exec_plan_for_step: end-to-end over fqn cases -----

def test__exec_plan_for_step_runs_all_kinds(run_dir, monkeypatch, tmp_path):
    # plan: Extract -> Transform -> Load
    monkeypatch.setattr(rc.planner, "plan_for_step", lambda *_: ["EX", "TR", "SNK"])

    steps = {
        "EX": {"function_fqn":"blase.Extract.read_csv", "params": {"file":"x.csv"}},
        "TR": {"function_fqn":"blase.Transform.apply_function", "params": {"fn":"f"}},
        "SNK":{"function_fqn":"blase.Load.save_to_csv",  "params": {"target":"out.csv"}},
    }
    monkeypatch.setattr(rc.store, "load_step", lambda _, h: steps[h])
    monkeypatch.setattr(rc.store, "load_step_inputs",  lambda *_: [{"data_hash":"SRC","role":"source"}])
    monkeypatch.setattr(rc.store, "load_step_outputs", lambda *_: [{"data_hash":"OUT","name":"out"}])
    monkeypatch.setattr(rc.store, "get_data_kind", lambda *_: "csv")
    monkeypatch.setattr(rc.cas, "path_for", lambda *a, **k: run_dir / "cas" / "sha256" / "csv" / "xx" / "yy")
    monkeypatch.setattr(rc.store, "pick_code_hash", lambda *_: "CODE")
    monkeypatch.setattr(rc.code, "load_callable_from_blob", lambda *_: (lambda x: x))

    # source path
    src = tmp_path / "src.csv"; src.write_text("a\n")
    monkeypatch.setattr(rc.store, "load_data_source_path", lambda *_: str(src))

    # make the hash check pass for the source
    monkeypatch.setattr(rc.Hash, "hash_file",
                        lambda self, p: "SRC" if Path(p) == src else "OUT")
    
    def save_stub(**k):
        out_file.write_text("ok\n")
        return str(out_file)

    # bindings runners
    out_file = tmp_path / "final.csv"
    monkeypatch.setattr(rc.bindings, "run_read_csv_restore",     lambda **k: iter([(["row"], True)]))
    monkeypatch.setattr(rc.bindings, "run_apply_function_restore", lambda **k: iter([(["row2"], True)]))
    monkeypatch.setattr(rc.bindings, "run_save_to_csv_replay", save_stub)

    produced = rc._exec_plan_for_step(run_dir, "SNK", to_path=None, backend_override=None)
    assert "OUT" in produced and produced["OUT"] == out_file

# ----- _ensure_data_local_or_replay paths -----

def test__ensure_data_local_or_replay_fast(run_dir, monkeypatch, tmp_path):
    f = tmp_path / "x.csv"; f.write_text("ok\n")
    h = rc.Hash().hash_file(f)
    monkeypatch.setattr(rc.materialize, "ensure_local", lambda *a, **k: f)
    p, created = rc._ensure_data_local_or_replay(run_dir, h, "csv")
    assert p == f and not created

def test__ensure_data_local_or_replay_replay(run_dir, monkeypatch, tmp_path):
    class Need(rc.NeedReplay): pass
    def ensure_fail(*a, **k): raise Need()
    monkeypatch.setattr(rc.materialize, "ensure_local", ensure_fail)
    monkeypatch.setattr(rc.store, "producer_step_for_data", lambda *_: "S")
    # _exec_plan_for_step returns mapping with our hash
    out = tmp_path / "out.csv"; out.write_text("ok\n")
    h = rc.Hash().hash_file(out)
    monkeypatch.setattr(rc, "_exec_plan_for_step", lambda *a, **k: {h: out})
    p, created = rc._ensure_data_local_or_replay(run_dir, h, "csv")
    assert created and p == out

# ----- CLI commands: cmd_list, cmd_show, cmd_plan, cmd_run -----

def write_step(run_dir, *, step_hash, fqn, status="completed", params=None):
    db = run_dir / "nodes" / "nodes.db"
    with sqlite3.connect(db) as c:
        c.execute("INSERT OR REPLACE INTO steps(step_hash,run_id,function_fqn,params_json,ts_start,status) VALUES(?,?,?,?,datetime('now'),?)",
                  (step_hash, run_dir.name, fqn, json.dumps(params or {}), status))
        c.commit()

def test_cmd_list_and_show(run_dir, capsys, monkeypatch):
    write_step(run_dir, step_hash="S1", fqn="blase.Transform.apply_function")
    ns = type("A",(object,),{"run":None,"limit":10,"like_fqn":None})
    monkeypatch.chdir(run_dir.parent)
    rc.cmd_list(ns)
    out = capsys.readouterr().out
    assert "S1" in out

    ns2 = type("A",(object,),{"run":None,"step":"S1"})
    rc.cmd_show(ns2)
    out2 = capsys.readouterr().out
    assert "step: S1" in out2 and "fqn=" in out2

def test_cmd_plan_marks_states(run_dir, capsys, monkeypatch, tmp_path):
    # Create both steps in their own short-lived connections
    write_step(run_dir, step_hash="S", fqn="blase.Load.save_to_csv")
    write_step(run_dir, step_hash="P", fqn="blase.Transform.apply_function")

    db = run_dir / "nodes" / "nodes.db"
    # Now open ONE connection to add edges + data rows
    import sqlite3
    with sqlite3.connect(db) as c:
        # S consumes I and produces O
        c.execute(
            "INSERT OR REPLACE INTO step_inputs(step_hash,data_hash,role,arg_name) VALUES(?,?,?,?)",
            ("S", "I", "source", None),
        )
        c.execute(
            "INSERT OR REPLACE INTO step_outputs(step_hash,data_hash,name) VALUES(?,?,?)",
            ("S", "O", "out"),
        )
        c.execute(
            "INSERT OR REPLACE INTO data(data_hash,kind,version) VALUES(?,?,?)",
            ("O", "csv", "1"),
        )
        # P produced I → enables the “need replay” path
        c.execute(
            "INSERT OR REPLACE INTO step_outputs(step_hash,data_hash,name) VALUES(?,?,?)",
            ("P", "I", "prod_i"),
        )
        c.commit()

    # Ensure the active run can be resolved
    monkeypatch.chdir(run_dir.parent)

    ns = type("A", (object,), {"run": None, "step": "S"})
    rc.cmd_plan(ns)
    out = capsys.readouterr().out
    # Now we should see the replay hint and outputs section
    assert "need replay" in out and "outputs:" in out

def test_cmd_run_data_fast_materialize(run_dir, monkeypatch, tmp_path, capsys):
    # ensure active run is discoverable
    monkeypatch.chdir(run_dir.parent)  # the "runs" directory with active_run.blase

    f = tmp_path / "o.csv"
    f.write_text("z\n")
    h = rc.Hash().hash_file(f)

    # fast-path materialize returns our file path
    monkeypatch.setattr(rc.materialize, "ensure_local", lambda *a, **k: str(f))

    ns = SimpleNamespace(
        run=None,
        data=h,
        mode="materialize",
        to=None,
        on_conflict="overwrite",
        keep_intermediates=False,
        backend=None,
        limit_batches=None,
    )

    code = rc.cmd_run(ns)
    out = capsys.readouterr().out
    assert code == 0
    assert "Materialized:" in out


def test_cmd_run_step_materialize_needs_replay_hint(run_dir, monkeypatch, capsys):
    # step with no available output locally → code 3 + hint
    write_step(run_dir, step_hash="S", fqn="blase.Load.save_to_csv", params={"target": "out.csv"})
    db = run_dir / "nodes" / "nodes.db"
    with sqlite3.connect(db) as c:
        c.execute(
            "INSERT OR REPLACE INTO step_outputs(step_hash,data_hash,name) VALUES(?,?,?)",
            ("S", "O", "out"),
        )
        c.execute(
            "INSERT OR REPLACE INTO data(data_hash,kind,version) VALUES(?,?,?)",
            ("O", "csv", "1"),
        )
        c.commit()

    # force materialize to raise NeedReplay
    class Need(rc.materialize.NeedReplay): ...
    monkeypatch.setattr(rc.materialize, "ensure_local", lambda *a, **k: (_ for _ in ()).throw(Need()))

    # ensure active run is discoverable
    monkeypatch.chdir(run_dir.parent)

    ns = SimpleNamespace(
        run=None,
        step="S",
        mode="materialize",
        to=str(run_dir / "restored.csv"),
        on_conflict="overwrite",
        # unused in this path but harmless:
        data=None,
        keep_intermediates=False,
        backend=None,
        limit_batches=None,
    )

    code = rc.cmd_run(ns)
    out = capsys.readouterr().out
    assert code == 3 and "blase restore plan" in out and "blase restore run" in out