import json
import sqlite3
from types import SimpleNamespace
from pathlib import Path
import pytest
import sys

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


class Calls:
    def __init__(self):
        self.args = None
        self.kw = None
        self.count = 0


def _mk_db(run_path: Path, ddl):
    db = run_path / "nodes" / "nodes.db"
    with sqlite3.connect(db) as c:
        for stmt in ddl:
            c.execute(stmt)
        c.commit()


def _mk_planner_single(sh):
    # planner.plan_for_step -> [sh]
    # planner._db / _step_row used by manifest path
    if sh:
        return SimpleNamespace(
            plan_for_step=lambda run_path, tip: [sh],
            _db=lambda run_path: "DB",
            _step_row=lambda _db, sh2: {"ts_start": "2025-01-01T00:00:00"},
        )
    return SimpleNamespace(
        _db=lambda run_path: "DB",
        _step_row=lambda _db, sh: {"ts_start": "2025-01-01T00:00:00"},
    )


def _dummy_fn(*a, **k):  # loaded user function
    return None


@pytest.fixture
def run_path(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    rid = "2025-01-02T03-04-05"
    rp = runs / rid
    (rp / "nodes").mkdir(parents=True, exist_ok=True)
    (rp / "cas" / "sha256").mkdir(parents=True, exist_ok=True)
    (runs / "active_run.blase").write_text(
        json.dumps({"run_id": rid, "status": "active"})
    )
    # minimal schema
    DDL = [
        """CREATE TABLE IF NOT EXISTS steps (step_hash TEXT PRIMARY KEY, function_fqn TEXT, params_json TEXT)""",
        """CREATE TABLE IF NOT EXISTS step_inputs (step_hash TEXT, data_hash TEXT, role TEXT, PRIMARY KEY (step_hash, data_hash, role))""",
        """CREATE TABLE IF NOT EXISTS step_outputs (step_hash TEXT, data_hash TEXT, name TEXT, PRIMARY KEY (step_hash, name), UNIQUE (data_hash))""",
        """CREATE TABLE IF NOT EXISTS data (data_hash TEXT PRIMARY KEY, kind TEXT, version TEXT, byte_len INTEGER, source_path TEXT, metadata_json TEXT)""",
    ]
    _mk_db(rp, DDL)
    return rp


@pytest.fixture
def run_dir(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    run_id = "2025-01-02T03-04-05"
    run_path = runs / run_id
    (run_path / "nodes").mkdir(parents=True, exist_ok=True)
    (run_path / "cas" / "sha256" / "csv").mkdir(parents=True, exist_ok=True)
    # active pointer
    (runs / "active_run.blase").write_text(
        json.dumps({"run_id": run_id, "status": "active"}, indent=2)
    )
    # minimal nodes.db
    db = run_path / "nodes" / "nodes.db"
    with sqlite3.connect(db) as c:
        for stmt in DDL:
            c.execute(stmt)
        c.commit()
    return run_path


def _args(**kw):
    # default args for --data path; caller can override
    base = dict(
        run=None,
        mode="replay",
        data="DHASH",
        step=None,
        to=None,
        on_conflict="overwrite",
        keep_intermediates=False,
        limit_batches=None,
        backend=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture
def setup_env(monkeypatch, tmp_path):
    monkeypatch.setattr(rc, "_resolve_run_path", lambda run: tmp_path)

    restore_dir = tmp_path / "restore"
    restore_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(rc, "RESTORE_DEFAULT_DIR", restore_dir, raising=False)

    class _NeedReplay(Exception):
        pass

    # make both names resolve in cmd_run: `materialize.NeedReplay` and bare `NeedReplay`
    monkeypatch.setattr(
        rc,
        "materialize",
        SimpleNamespace(
            ensure_local=lambda *a, **k: (_ for _ in ()).throw(_NeedReplay()),
            NeedReplay=_NeedReplay,
        ),
    )
    monkeypatch.setattr(rc, "NeedReplay", _NeedReplay)

    monkeypatch.setattr(rc, "_assert_path_matches_hash", lambda path, h, ctx: None)

    yield tmp_path, restore_dir


@pytest.fixture
def isolate_cmd_run(monkeypatch, tmp_path):
    # active run + restore dir
    monkeypatch.setattr(rc, "_resolve_run_path", lambda run: tmp_path)
    rd = tmp_path / "restore"
    rd.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(rc, "RESTORE_DEFAULT_DIR", rd, raising=False)

    # prevent any fast-path materialize calls from hitting DB
    class _NeedReplay(Exception): ...

    monkeypatch.setattr(rc, "NeedReplay", _NeedReplay)
    monkeypatch.setattr(
        rc,
        "materialize",
        SimpleNamespace(
            ensure_local=lambda *a, **k: (_ for _ in ()).throw(_NeedReplay()),
            NeedReplay=_NeedReplay,
        ),
    )

    # stub DB-backed store calls that materialize uses
    fake_store_extra = dict(
        load_materializations=lambda *a, **k: [],
        get_data_kind=lambda *a, **k: "data",
        producer_step_for_data=lambda *a, **k: None,
        load_step_outputs=lambda *a, **k: [],
        load_step_inputs=lambda *a, **k: [],
    )
    return tmp_path, rd, fake_store_extra


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
    src = tmp_path / "src.csv"
    src.write_text("x\n", encoding="utf-8")
    from blase.utils.hashing import Hash

    h = Hash().hash_file(src)

    class Store:
        @staticmethod
        def load_data_source_path(run_path, data_hash):
            return str(src)

        @staticmethod
        def get_data_kind(run_path, h):
            return "csv"

    monkeypatch.setattr(rc.store, "load_data_source_path", Store.load_data_source_path)
    monkeypatch.setattr(rc.store, "get_data_kind", Store.get_data_kind)

    path, created = rc._resolve_source_no_copy(
        run_dir, h, seen_steps=set(), created_paths=[]
    )
    assert path == src and created is False


def test__resolve_source_no_copy_fallback_replay(run_dir, monkeypatch, tmp_path):
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
    mat = tmp_path / "m.csv"
    mat.write_text("z\n")
    h = rc.Hash().hash_file(mat)
    monkeypatch.setattr(rc.store, "get_materialized_path", lambda *_: mat)
    monkeypatch.setattr(
        rc.Hash(), "hash_file", lambda *_: h
    )  # shortcut not strictly needed
    p, created = rc._resolve_seed_no_copy_or_ephemeral(run_dir, h)
    assert p == mat and not created


def test__resolve_seed_ephemeral_replay(run_dir, monkeypatch, tmp_path):
    # no mat, no recorded source, but has producer → write tmp and return True
    monkeypatch.setattr(rc.store, "get_materialized_path", lambda *a, **k: None)
    monkeypatch.setattr(rc.store, "get_recorded_source_path", lambda *a, **k: None)
    monkeypatch.setattr(rc.store, "producer_step_for_data", lambda *a, **k: "stepX")

    # Write the expected temp file inside _exec_plan_for_step
    def fake_exec(run_path, prod, to_path, **_):
        p = Path(to_path)
        p.write_text("seed\n")
        return {}

    monkeypatch.setattr(rc, "_exec_plan_for_step", fake_exec)

    # compute expected hash from what fake_exec writes
    tmp = tmp_path / "probe.csv"
    tmp.write_text("seed\n")
    h = rc.Hash().hash_file(tmp)

    created_paths = []
    p, created = rc._resolve_seed_no_copy_or_ephemeral(
        run_dir, h, created_paths=created_paths
    )
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
    # Plan: Extract -> Transform -> SINK
    monkeypatch.setattr(
        rc.planner,
        "plan_for_step",
        lambda *_: [
            {"step_hash": "A", "function_fqn": "blase.Extract.read_csv"},
            {"step_hash": "B", "function_fqn": "blase.Transform.apply_function"},
            {"step_hash": "SINK", "function_fqn": "blase.Load.save_to_csv"},
        ],
    )

    # Patch the *same store module object used inside restore_cli*
    monkeypatch.setattr(
        rc.store,
        "load_step",
        lambda run_path, step_hash: {
            "function_fqn": "blase.Load.save_to_csv",
            "params": {},
            "status": "completed",
        }
        if step_hash == "SINK"
        else {
            "function_fqn": "blase.Transform.apply_function",
            "params": {},
            "status": "completed",
        },
    )

    # Intercept which upstream step hash is chosen
    called = {}

    def fake_build(run_path, step_hash):
        called["hash"] = step_hash
        return iter([([], True)])

    monkeypatch.setattr(rc, "_build_stream_for_step", fake_build)

    gen = rc._upstream_gen_for_sink(run_dir, "SINK")
    list(gen)  # exhaust generator

    assert called["hash"] == "B"


def test__upstream_gen_for_sink_no_upstream_raises(run_dir, monkeypatch):
    # Only the sink in the plan
    monkeypatch.setattr(
        rc.planner,
        "plan_for_step",
        lambda *_: [{"step_hash": "SINK", "function_fqn": "blase.Load.save_to_csv"}],
    )

    # Patch the store reference used by restore_cli
    monkeypatch.setattr(
        rc.store,
        "load_step",
        lambda run_path, step_hash: {
            "function_fqn": "blase.Load.save_to_csv",
            "params": {},
            "status": "completed",
        },
    )

    with pytest.raises(SystemExit):
        rc._upstream_gen_for_sink(run_dir, "SINK")


# ----- _normalize_plan_nodes branches (dict/tuple/str variants) -----


def test__normalize_plan_nodes_all_forms(run_dir, monkeypatch):
    def load_step(_, h):
        return {"function_fqn": f"fqn.{h}", "params": {}}

    monkeypatch.setattr(rc.store, "load_step", load_step)

    nodes = rc._normalize_plan_nodes(
        run_dir,
        [
            {"step_hash": "A", "function_fqn": "fqn.A"},
            {"hash": "B"},
            {"id": "C"},
            ("D", "extra"),
            "E",
        ],
    )
    got = [(n["step_hash"], n["function_fqn"]) for n in nodes]
    assert got == [
        ("A", "fqn.A"),
        ("B", "fqn.B"),
        ("C", "fqn.C"),
        ("D", "fqn.D"),
        ("E", "fqn.E"),
    ]


# ----- _exec_plan_for_step: end-to-end over fqn cases -----


def test__exec_plan_for_step_runs_all_kinds(run_dir, monkeypatch, tmp_path):
    # plan: Extract -> Transform -> Load
    monkeypatch.setattr(rc.planner, "plan_for_step", lambda *_: ["EX", "TR", "SNK"])

    steps = {
        "EX": {"function_fqn": "blase.Extract.read_csv", "params": {"file": "x.csv"}},
        "TR": {"function_fqn": "blase.Transform.apply_function", "params": {"fn": "f"}},
        "SNK": {
            "function_fqn": "blase.Load.save_to_csv",
            "params": {"target": "out.csv"},
        },
    }
    monkeypatch.setattr(rc.store, "load_step", lambda _, h: steps[h])
    monkeypatch.setattr(
        rc.store,
        "load_step_inputs",
        lambda *_: [{"data_hash": "SRC", "role": "source"}],
    )
    monkeypatch.setattr(
        rc.store, "load_step_outputs", lambda *_: [{"data_hash": "OUT", "name": "out"}]
    )
    monkeypatch.setattr(rc.store, "get_data_kind", lambda *_: "csv")
    monkeypatch.setattr(
        rc.cas,
        "path_for",
        lambda *a, **k: run_dir / "cas" / "sha256" / "csv" / "xx" / "yy",
    )
    monkeypatch.setattr(rc.store, "pick_code_hash", lambda *_: "CODE")
    monkeypatch.setattr(rc.code, "load_callable_from_blob", lambda *_: (lambda x: x))

    # source path
    src = tmp_path / "src.csv"
    src.write_text("a\n")
    monkeypatch.setattr(rc.store, "load_data_source_path", lambda *_: str(src))

    # make the hash check pass for the source
    monkeypatch.setattr(
        rc.Hash, "hash_file", lambda self, p: "SRC" if Path(p) == src else "OUT"
    )

    def save_stub(**k):
        out_file.write_text("ok\n")
        return str(out_file)

    # bindings runners
    out_file = tmp_path / "final.csv"
    monkeypatch.setattr(
        rc.bindings, "run_read_csv_restore", lambda **k: iter([(["row"], True)])
    )
    monkeypatch.setattr(
        rc.bindings, "run_apply_function_restore", lambda **k: iter([(["row2"], True)])
    )
    monkeypatch.setattr(rc.bindings, "run_save_to_csv_replay", save_stub)

    produced = rc._exec_plan_for_step(
        run_dir, "SNK", to_path=None, backend_override=None
    )
    assert "OUT" in produced and produced["OUT"] == out_file


def test_exec_plan_read_parquet_uses_sources_list_and_sorts_batch_descs(
    monkeypatch, run_path, tmp_path
):
    sh = "step_read_parquet"

    # One-step plan
    monkeypatch.setattr(
        rc, "planner", SimpleNamespace(plan_for_step=lambda rp, tip: [sh])
    )

    # Store with params['sources'], manifest, and unordered batch_descs
    params = {"sources": ["/p/a", "/p/b"], "mode": "images"}
    st_row = {"function_fqn": "blase.Extract.read_parquet", "params": params}
    ins = [
        {"data_hash": "CODE123", "role": "code"},
        {"data_hash": "ENV999", "role": "env"},
    ]
    outs = [
        {"name": "batch_desc_10", "data_hash": "BD10"},
        {"name": "manifest", "data_hash": "MANI"},
        {"name": "batch_desc_2", "data_hash": "BD2"},
        {"name": "weird", "data_hash": "IGN"},
    ]
    store_fake = SimpleNamespace(
        load_step=lambda rp, s: st_row,
        load_step_inputs=lambda rp, s: ins,
        load_step_outputs=lambda rp, s: outs,
        pick_code_hash=lambda ins_list: next(
            i["data_hash"] for i in ins_list if i["role"] == "code"
        ),
    )
    monkeypatch.setattr(rc, "store", store_fake)

    calls = {}

    def _restore(**kw):
        calls["kw"] = kw

        def _gen():
            yield (["dummy"], {"ordinal": 1}, False)

        return _gen()

    monkeypatch.setattr(
        rc, "bindings", SimpleNamespace(run_read_parquet_restore=_restore)
    )

    out = rc._exec_plan_for_step(run_path, sh, None, None)
    assert out == {}

    kw = calls["kw"]
    assert kw["run_path"] == run_path
    assert kw["params"] is params
    realized = kw["realized"]
    assert "sources" in realized and realized["sources"] == ["/p/a", "/p/b"]
    assert "source" not in realized
    assert realized["manifest"] == "MANI"
    assert realized["batch_descs"] == ["BD2", "BD10"]  # numeric sort


def test_exec_plan_read_parquet_falls_back_to_single_source(
    monkeypatch, run_path, tmp_path
):
    sh = "step_read_parquet_single"

    monkeypatch.setattr(
        rc, "planner", SimpleNamespace(plan_for_step=lambda rp, tip: [sh])
    )

    params = {"source": "/only/one"}
    st_row = {"function_fqn": "blase.Extract.read_parquet", "params": params}
    ins = []
    outs = [
        {"name": "batch_desc_1", "data_hash": "BD1"},
        # manifest intentionally absent
    ]
    store_fake = SimpleNamespace(
        load_step=lambda rp, s: st_row,
        load_step_inputs=lambda rp, s: ins,
        load_step_outputs=lambda rp, s: outs,
        pick_code_hash=lambda *_: None,
    )
    monkeypatch.setattr(rc, "store", store_fake)

    calls = {}

    def _restore(**kw):
        calls["kw"] = kw

        def _gen():
            yield (["dummy"], {"ordinal": 1}, False)

        return _gen()

    monkeypatch.setattr(
        rc, "bindings", SimpleNamespace(run_read_parquet_restore=_restore)
    )

    out = rc._exec_plan_for_step(run_path, sh, None, None)
    assert out == {}

    kw = calls["kw"]
    assert kw["params"] is params
    realized = kw["realized"]
    assert realized.get("source") == "/only/one"
    assert "sources" not in realized
    assert "manifest" not in realized  # none recorded
    assert realized["batch_descs"] == ["BD1"]


def test_exec_plan_read_images_builds_realized_and_calls_restore(
    monkeypatch, run_path, tmp_path
):
    sh = "step_read_images"
    # Fake planner: one-step plan
    monkeypatch.setattr(
        rc, "planner", SimpleNamespace(plan_for_step=lambda rp, tip: [sh])
    )
    # Fake store
    params = {"directory": "/data/images", "pattern": "**/*.jpg", "mode": "auto"}
    st_row = {"function_fqn": "blase.Extract.read_images", "params": params}
    ins = [
        {"data_hash": "CODE123", "role": "code"},
        {"data_hash": "ENV999", "role": "env"},
    ]
    outs = [
        {"name": "batch_desc_10", "data_hash": "BD10"},
        {"name": "manifest", "data_hash": "MANI"},
        {"name": "batch_desc_2", "data_hash": "BD2"},
    ]
    store_fake = SimpleNamespace(
        load_step=lambda rp, s: st_row,
        load_step_inputs=lambda rp, s: ins,
        load_step_outputs=lambda rp, s: outs,
        pick_code_hash=lambda ins_list: next(
            i["data_hash"] for i in ins_list if i["role"] == "code"
        ),
    )
    monkeypatch.setattr(rc, "store", store_fake)

    # Capture bindings call
    calls = Calls()

    def _restore(**kw):
        calls.args, calls.kw, calls.count = (), kw, calls.count + 1

        # return a simple generator
        def _gen():
            yield (["im1"], {"ordinal": 1}, False)

        return _gen()

    monkeypatch.setattr(
        rc, "bindings", SimpleNamespace(run_read_images_restore=_restore)
    )

    out = rc._exec_plan_for_step(run_path, sh, None, None)
    assert out == {}  # no sinks in this branch

    # Verify call
    assert calls.count == 1
    assert calls.kw["run_path"] == run_path
    assert calls.kw["params"] is params
    realized = calls.kw["realized"]
    # includes source, manifest, and numerically sorted batch_descs
    assert realized["source"] == "/data/images"
    assert realized["manifest"] == "MANI"
    assert realized["batch_descs"] == ["BD2", "BD10"]


def test_apply_function_manifest_uses_parquet_and_calls_apply_restore(
    monkeypatch, run_path, tmp_path
):
    sh = "step_apply_fn"
    monkeypatch.setattr(rc, "planner", _mk_planner_single(sh))

    params = {"module": "m", "function": "f"}
    ins = [
        {"data_hash": "B2", "role": "batch"},
        {"data_hash": "B1", "role": "batchmeta"},
        {"data_hash": "MANI", "role": "manifest"},
    ]
    code_hash = "CODEHASH"

    store_fake = SimpleNamespace(
        load_step=lambda rp, s: {
            "function_fqn": "blase.Transform.apply_function",
            "params": params,
        },
        load_step_inputs=lambda rp, s: ins,
        load_step_outputs=lambda *a, **k: [],
        read_parquet_params_for_manifest=lambda run_path, man_hash, ts_before=None: {
            "ok": True
        },
        pick_code_hash=lambda *_: code_hash,
    )
    monkeypatch.setattr(rc, "store", store_fake)
    monkeypatch.setattr(
        rc, "_load_user_function", lambda *a, **k: _dummy_fn, raising=False
    )

    # create CAS blob for code hash using run_path fixture
    code_blob = run_path / "cas" / "sha256" / "code" / code_hash[:2] / code_hash[2:]
    code_blob.parent.mkdir(parents=True, exist_ok=True)
    code_blob.write_text(
        json.dumps(
            {
                "entry": {"qualname": "f", "module": "m"},
                "module_source": "",
                "function_source": "def f(*args, **kwargs):\n    return None\n",
            },
            ensure_ascii=False,
        )
    )

    calls = {"gen_called": 0, "apply_called": 0, "apply_kw": None}

    def _rgfm(run_path_, manifest_hash_, step_hash_, store_, bindings_):
        calls["gen_called"] += 1
        assert manifest_hash_ == "MANI"
        assert step_hash_ == sh

        def _gen():
            yield (["img"], {"ordinal": 1}, False)

        return _gen()

    def _apply_restore(*, run_path, params: dict, realized: dict, transform_fn):
        calls["apply_called"] += 1
        calls["apply_kw"] = dict(
            run_path=run_path,
            params=params,
            realized=realized,
            transform_fn=transform_fn,
        )

        def _up():
            yield (["out"], {"ordinal": 1}, False)

        return _up()

    monkeypatch.setattr(rc, "_restore_gen_for_manifest", _rgfm)
    monkeypatch.setattr(
        rc, "bindings", SimpleNamespace(run_apply_function_restore=_apply_restore)
    )

    out = rc._exec_plan_for_step(run_path, sh, None, None)
    assert out == {}
    assert calls["gen_called"] == 1
    assert calls["apply_called"] == 1
    assert "source_gen" in calls["apply_kw"]["realized"]


def test_apply_function_manifest_missing_manifest_raises(
    monkeypatch, run_path, tmp_path
):
    sh = "step_apply_fn_nomani"
    monkeypatch.setattr(rc, "planner", _mk_planner_single(sh))

    params = {"module": "m", "function": "f"}
    ins_no_manifest = [
        {"data_hash": "B2", "role": "batch"},
        {"data_hash": "B1", "role": "batchmeta"},
    ]
    code_hash = "CODEHASH"

    store_fake = SimpleNamespace(
        load_step=lambda rp, s: {
            "function_fqn": "blase.Transform.apply_function",
            "params": params,
        },
        load_step_inputs=lambda rp, s: ins_no_manifest,
        load_step_outputs=lambda *a, **k: [],
        read_parquet_params_for_manifest=lambda run_path, man_hash, ts_before=None: {
            "ok": True
        },
        pick_code_hash=lambda *_: code_hash,
    )
    monkeypatch.setattr(rc, "store", store_fake)
    monkeypatch.setattr(
        rc, "_load_user_function", lambda *a, **k: _dummy_fn, raising=False
    )

    # create CAS blob for code hash using run_path fixture
    code_blob = run_path / "cas" / "sha256" / "code" / code_hash[:2] / code_hash[2:]
    code_blob.parent.mkdir(parents=True, exist_ok=True)
    code_blob.write_text(
        json.dumps(
            {
                "entry": {"qualname": "f", "module": "m"},
                "module_source": "",
                "function_source": "def f(*args, **kwargs):\n    return None\n",
            },
            ensure_ascii=False,
        )
    )

    with pytest.raises(SystemExit) as ei:
        rc._exec_plan_for_step(run_path, sh, None, None)

    msg = str(ei.value)
    assert (
        "restore: cannot resolve manifest for upstream stream" in msg
        or "Transform.apply_function missing anchor" in msg
    )


def test_exec_plan_read_images_without_manifest_or_batch_descs(monkeypatch, run_path):
    sh = "step_read_images"
    monkeypatch.setattr(
        rc, "planner", SimpleNamespace(plan_for_step=lambda rp, tip: [sh])
    )
    params = {"directory": "/x"}
    st_row = {"function_fqn": "blase.Extract.read_images", "params": params}
    store_fake = SimpleNamespace(
        load_step=lambda rp, s: st_row,
        load_step_inputs=lambda rp, s: [],
        load_step_outputs=lambda rp, s: [],  # no manifest, no batch_desc_*
        pick_code_hash=lambda ins_list: None,
    )
    monkeypatch.setattr(rc, "store", store_fake)

    captured = {}

    def _restore(**kw):
        captured.update(kw)

        def _gen():
            if False:
                yield None

        return _gen()

    monkeypatch.setattr(
        rc, "bindings", SimpleNamespace(run_read_images_restore=_restore)
    )

    rc._exec_plan_for_step(run_path, sh, None, None)
    assert captured["realized"] == {"source": "/x"}


def test_exec_plan_read_images_missing_directory_omits_source(monkeypatch, run_path):
    sh = "step_read_images"
    monkeypatch.setattr(
        rc, "planner", SimpleNamespace(plan_for_step=lambda rp, tip: [sh])
    )
    st_row = {
        "function_fqn": "blase.Extract.read_images",
        "params": {"pattern": "*.jpg"},
    }
    outs = [{"name": "manifest", "data_hash": "M"}]
    store_fake = SimpleNamespace(
        load_step=lambda rp, s: st_row,
        load_step_inputs=lambda rp, s: [],
        load_step_outputs=lambda rp, s: outs,
        pick_code_hash=lambda ins_list: None,
    )
    monkeypatch.setattr(rc, "store", store_fake)

    captured = {}

    def _restore(**kw):
        captured.update(kw)

        def _gen():
            if False:
                yield None

        return _gen()

    monkeypatch.setattr(
        rc, "bindings", SimpleNamespace(run_read_images_restore=_restore)
    )

    rc._exec_plan_for_step(run_path, sh, None, None)
    assert "source" not in captured["realized"]
    assert captured["realized"]["manifest"] == "M"


def test_apply_function_with_source_anchor(monkeypatch, run_path):
    sh = "step_apply"
    monkeypatch.setattr(rc, "planner", _mk_planner_single(sh))

    # store: inputs have source anchor + code hash
    ins = [
        {"role": "code", "data_hash": "CODE123"},
        {"role": "env", "data_hash": "ENV1"},
        {"role": "source", "data_hash": "SRC123"},
    ]
    st_row = {"function_fqn": "blase.Transform.apply_function", "params": {"p": 1}}
    store_fake = SimpleNamespace(
        load_step=lambda rp, s: st_row,
        load_step_inputs=lambda rp, s: ins,
        load_step_outputs=lambda rp, s: [],
        pick_code_hash=lambda ins_list: "CODE123",
    )
    monkeypatch.setattr(rc, "store", store_fake)

    # cas + code loader for callable
    monkeypatch.setattr(
        rc, "cas", SimpleNamespace(path_for=lambda rp, kind, h: run_path / "cas" / h)
    )
    monkeypatch.setattr(
        rc, "code", SimpleNamespace(load_callable_from_blob=lambda p: _dummy_fn)
    )

    # resolve source path
    monkeypatch.setattr(
        rc,
        "_resolve_source_no_copy",
        lambda rp, h, seen_steps, created_paths: ("/data/src.csv", False),
    )

    # capture apply_function_restore call
    called = {}
    monkeypatch.setattr(
        rc,
        "bindings",
        SimpleNamespace(
            run_apply_function_restore=lambda **kw: called.setdefault("kw", kw)
            or (x for x in ()),
        ),
    )

    out = rc._exec_plan_for_step(run_path, sh, None, None)
    assert out == {}
    assert called["kw"]["realized"] == {"source": "/data/src.csv"}
    assert callable(called["kw"]["transform_fn"])


def test_apply_function_with_manifest_anchor(monkeypatch, run_path):
    sh = "step_apply"
    monkeypatch.setattr(rc, "planner", _mk_planner_single(sh))

    # Inputs include manifest anchor plus batch_desc + batch
    ins_first = [
        {"role": "code", "data_hash": "CODEX"},
        {"role": "manifest", "data_hash": "MANI123"},
        {"role": "batch_desc", "data_hash": "BD1"},
        {"role": "batch", "data_hash": "B2"},
    ]
    st_row = {"function_fqn": "blase.Transform.apply_function", "params": {"q": 2}}
    store_fake = SimpleNamespace(
        load_step=lambda rp, s: st_row,
        load_step_inputs=lambda rp, s: ins_first,
        load_step_outputs=lambda rp, s: [],
        pick_code_hash=lambda ins_list: "CODEX",
        read_images_params_for_manifest=lambda rp, mh, ts_before: {
            "directory": "/imgs",
            "pattern": "**/*.jpg",
        },
    )
    monkeypatch.setattr(rc, "store", store_fake)

    monkeypatch.setattr(
        rc, "cas", SimpleNamespace(path_for=lambda rp, kind, h: run_path / "cas" / h)
    )
    monkeypatch.setattr(
        rc, "code", SimpleNamespace(load_callable_from_blob=lambda p: _dummy_fn)
    )

    # run_read_images_restore returns a generator placeholder
    def _gen():
        if False:  # empty generator
            yield None

    calls = {}

    def _run_read_images_restore(**kw):
        calls["ri"] = kw
        return _gen()

    def _run_apply_function_restore(**kw):
        calls["af"] = kw
        return _gen()

    monkeypatch.setattr(
        rc,
        "bindings",
        SimpleNamespace(
            run_read_images_restore=_run_read_images_restore,
            run_apply_function_restore=_run_apply_function_restore,
        ),
    )

    out = rc._exec_plan_for_step(run_path, sh, None, None)
    assert out == {}
    # read_images_restore received manifest + batch lists and source directory
    assert calls["ri"]["realized"]["manifest"] == "MANI123"
    # normalize presence of either key is accepted by reader; here we pass 'batch'
    assert set(calls["ri"]["realized"].keys()) >= {"source", "manifest", "batch"}
    # apply_function_restore got source_gen
    assert "source_gen" in calls["af"]["realized"]


def test_apply_function_missing_anchor_raises(monkeypatch, run_path):
    sh = "step_apply"
    monkeypatch.setattr(rc, "planner", _mk_planner_single(sh))

    # No source/manifest/dataset/manifest_root roles
    ins = [{"role": "code", "data_hash": "C"}]
    st_row = {"function_fqn": "blase.Transform.apply_function", "params": {}}
    store_fake = SimpleNamespace(
        load_step=lambda rp, s: st_row,
        load_step_inputs=lambda rp, s: ins,
        load_step_outputs=lambda rp, s: [],
        pick_code_hash=lambda ins_list: "C",
    )
    monkeypatch.setattr(rc, "store", store_fake)
    monkeypatch.setattr(rc, "cas", SimpleNamespace(path_for=lambda *a, **k: Path("/x")))
    monkeypatch.setattr(
        rc, "code", SimpleNamespace(load_callable_from_blob=lambda p: _dummy_fn)
    )

    with pytest.raises(SystemExit):
        rc._exec_plan_for_step(run_path, sh, None, None)


def test_apply_function_manifest_anchor_missing_params_raises(monkeypatch, run_path):
    sh = "step_apply"
    monkeypatch.setattr(rc, "planner", _mk_planner_single(sh))

    ins = [{"role": "code", "data_hash": "C"}, {"role": "manifest", "data_hash": "M"}]
    st_row = {"function_fqn": "blase.Transform.apply_function", "params": {}}
    store_fake = SimpleNamespace(
        load_step=lambda rp, s: st_row,
        load_step_inputs=lambda rp, s: ins,
        load_step_outputs=lambda rp, s: [],
        pick_code_hash=lambda ins_list: "C",
        read_images_params_for_manifest=lambda rp,
        mh,
        ts_before: None,  # critical: missing
    )
    monkeypatch.setattr(rc, "store", store_fake)
    monkeypatch.setattr(rc, "cas", SimpleNamespace(path_for=lambda *a, **k: Path("/x")))
    monkeypatch.setattr(
        rc, "code", SimpleNamespace(load_callable_from_blob=lambda p: _dummy_fn)
    )

    with pytest.raises(SystemExit):
        rc._exec_plan_for_step(run_path, sh, None, None)


def test_parquet_with_recorded_outs_and_override(monkeypatch, tmp_path):
    sh = "step_parquet"
    monkeypatch.setattr(rc, "planner", _mk_planner_single(sh))

    # store: recorded outs include unsorted shard names
    outs = [
        {"name": "parquet_shard_10", "data_hash": "H10"},
        {"name": "parquet_shard_2", "data_hash": "H2"},
    ]
    st_row = {
        "function_fqn": "blase.Load.save_images_to_parquet",
        "params": {"shard_prefix": "b"},
    }
    store_fake = SimpleNamespace(
        load_step=lambda rp, s: st_row,
        load_step_inputs=lambda rp, s: [],
        load_step_outputs=lambda rp, s: outs,
    )
    monkeypatch.setattr(rc, "store", store_fake)

    # upstream_gen should come from _upstream_gen_for_sink when upstream is None
    made_gen = object()
    called_upstream = {}
    monkeypatch.setattr(
        rc,
        "_upstream_gen_for_sink",
        lambda rp, s: called_upstream.setdefault("gen", made_gen),
    )

    # handler in RESTORE_HANDLERS
    recorded = {}

    def fake_handler(**kw):
        recorded.update(kw)
        # return two shard paths in the same order as expected_out_hashes after sort: H2, H10
        td = Path(kw["target_override"])
        return [td / "b_1.parquet", td / "b_2.parquet"]

    monkeypatch.setattr(
        rc,
        "bindings",
        SimpleNamespace(
            RESTORE_HANDLERS={"blase.Load.save_images_to_parquet": fake_handler}
        ),
    )

    # override target dir
    override_dir = str(tmp_path / "outdir")
    created_paths = []
    produced = rc._exec_plan_for_step(
        run_path=tmp_path,
        tip_step_hash=sh,
        to_path=override_dir,
        backend_override=None,
        seen_steps=None,
        created_paths=created_paths,
        ephemeral_only=False,
    )

    # handler call args
    assert recorded["run_path"] == tmp_path
    assert recorded["params"] is st_row["params"]
    assert recorded["target_override"] == override_dir
    assert recorded["on_conflict"] == "overwrite"
    assert recorded["expected_out_hashes"] == ["H2", "H10"]  # numerically sorted
    assert recorded["record_materialization"] is True
    assert recorded["upstream_gen"] is made_gen  # came from _upstream_gen_for_sink

    # produced mapping ties recorded data_hash -> returned path
    assert recorded["expected_out_hashes"] == ["H2", "H10"]
    assert produced == {
        "H2": Path(override_dir) / "b_1.parquet",
        "H10": Path(override_dir) / "b_2.parquet",
    }
    # created_paths appended in order
    assert created_paths == [
        Path(override_dir) / "b_1.parquet",
        Path(override_dir) / "b_2.parquet",
    ]


def test_parquet_without_recorded_outs_uses_default_dir_and_hashes(
    monkeypatch, tmp_path
):
    sh = "step_parquet"
    monkeypatch.setattr(rc, "planner", _mk_planner_single(sh))

    st_row = {
        "function_fqn": "blase.Load.save_images_to_parquet",
        "params": {"shard_prefix": "c"},
    }
    store_fake = SimpleNamespace(
        load_step=lambda rp, s: st_row,
        load_step_inputs=lambda rp, s: [],
        load_step_outputs=lambda rp, s: [],  # no recorded outs
    )
    monkeypatch.setattr(rc, "store", store_fake)

    # ensure default dir exists in rc (autouse fixture in your suite should already set this)
    rc.RESTORE_DEFAULT_DIR.mkdir(parents=True, exist_ok=True)

    # no upstream present -> generator sourced via helper
    made_gen = object()
    monkeypatch.setattr(rc, "_upstream_gen_for_sink", lambda rp, s: made_gen)

    # fake handler returns three paths
    def fake_handler(**kw):
        td = Path(kw["target_override"])
        return [td / f"c_{i}.parquet" for i in (1, 2, 3)]

    monkeypatch.setattr(
        rc,
        "bindings",
        SimpleNamespace(
            RESTORE_HANDLERS={"blase.Load.save_images_to_parquet": fake_handler}
        ),
    )

    # stub Hash().hash_file to compute fake hashes from filename
    class FakeHash:
        def hash_file(self, p):
            return f"HASH::{Path(p).name}"

    monkeypatch.setattr(rc, "Hash", FakeHash)

    created_paths = []
    produced = rc._exec_plan_for_step(
        run_path=tmp_path,
        tip_step_hash=sh,
        to_path=None,
        backend_override=None,
        seen_steps=None,
        created_paths=created_paths,
        ephemeral_only=False,
    )

    # target_override should be default restore dir
    # produced keys are computed hashes since outs was empty
    expected_paths = [rc.RESTORE_DEFAULT_DIR / f"c_{i}.parquet" for i in (1, 2, 3)]
    expected_hashes = [f"HASH::{p.name}" for p in expected_paths]
    assert set(produced.keys()) == set(expected_hashes)
    assert set(produced.values()) == set(expected_paths)
    assert created_paths == expected_paths


# ----- _ensure_data_local_or_replay paths -----


def test__ensure_data_local_or_replay_fast(run_dir, monkeypatch, tmp_path):
    f = tmp_path / "x.csv"
    f.write_text("ok\n")
    h = rc.Hash().hash_file(f)
    monkeypatch.setattr(rc.materialize, "ensure_local", lambda *a, **k: f)
    p, created = rc._ensure_data_local_or_replay(run_dir, h, "csv")
    assert p == f and not created


def test__ensure_data_local_or_replay_replay(run_dir, monkeypatch, tmp_path):
    class Need(rc.NeedReplay):
        pass

    def ensure_fail(*a, **k):
        raise Need()

    monkeypatch.setattr(rc.materialize, "ensure_local", ensure_fail)
    monkeypatch.setattr(rc.store, "producer_step_for_data", lambda *_: "S")
    # _exec_plan_for_step returns mapping with our hash
    out = tmp_path / "out.csv"
    out.write_text("ok\n")
    h = rc.Hash().hash_file(out)
    monkeypatch.setattr(rc, "_exec_plan_for_step", lambda *a, **k: {h: out})
    p, created = rc._ensure_data_local_or_replay(run_dir, h, "csv")
    assert created and p == out


# ----- _build_stream_for_step tests -----


def test_read_csv_branch(monkeypatch, tmp_path):
    sh = "S1"
    st_row = {"function_fqn": "pkg.Extract.read_csv", "params": {"p": 1}}
    ins = [{"role": "source", "data_hash": "SRC123"}]

    monkeypatch.setattr(
        rc,
        "store",
        SimpleNamespace(
            load_step=lambda rp, s: st_row,
            load_step_inputs=lambda rp, s: ins,
        ),
    )

    sentinel_gen = object()

    def run_read_csv_restore(**kw):
        # assert realized includes expected source hash
        assert kw["realized"]["__expected_source_hash__"] == "SRC123"
        return sentinel_gen

    monkeypatch.setattr(
        rc, "bindings", SimpleNamespace(run_read_csv_restore=run_read_csv_restore)
    )

    got = rc._build_stream_for_step(tmp_path, sh)
    assert got is sentinel_gen


def test_read_images_branch(monkeypatch, tmp_path):
    sh = "S2"
    st_row = {
        "function_fqn": "pkg.Extract.read_images",
        "params": {"directory": "/imgs"},
    }
    ins = [
        {"role": "manifest", "data_hash": "MANI"},
        {"role": "batch", "data_hash": "B1"},
        {"role": "batch", "data_hash": "B2"},
    ]

    monkeypatch.setattr(
        rc,
        "store",
        SimpleNamespace(
            load_step=lambda rp, s: st_row,
            load_step_inputs=lambda rp, s: ins,
        ),
    )

    captured = {}

    def run_read_images_restore(**kw):
        captured.update(kw)
        return iter(())  # empty generator

    monkeypatch.setattr(
        rc, "bindings", SimpleNamespace(run_read_images_restore=run_read_images_restore)
    )

    rc._build_stream_for_step(tmp_path, sh)
    assert captured["params"] is st_row["params"]
    assert captured["realized"]["source"] == "/imgs"
    assert captured["realized"]["manifest"] == "MANI"
    assert captured["realized"]["batch"] == ["B1", "B2"]


def test_transform_branch_uses_plan_upstream(monkeypatch, tmp_path):
    t_sh = "T1"
    up_sh = "U_csv"
    # plan returns upstream stream node before transform
    nodes = [
        {"step_hash": up_sh, "function_fqn": "pkg.Extract.read_csv"},
        {"step_hash": t_sh, "function_fqn": "pkg.Transform.apply_function"},
    ]
    monkeypatch.setattr(
        rc,
        "planner",
        SimpleNamespace(
            plan_for_step=lambda rp, s: [n["step_hash"] for n in nodes],
            _db=lambda rp: None,
            _step_row=lambda *_: {"ts_start": "X"},
        ),
    )
    monkeypatch.setattr(rc, "_normalize_plan_nodes", lambda rp, raw: nodes)
    monkeypatch.setattr(rc, "_is_stream_fqn", lambda fqn: fqn.endswith("read_csv"))

    # upstream read_csv inputs
    csv_st = {"function_fqn": "pkg.Extract.read_csv", "params": {}}
    tr_st = {"function_fqn": "pkg.Transform.apply_function", "params": {"alpha": 1}}
    ins_csv = [{"role": "source", "data_hash": "SRC"}]
    ins_tr = [{"role": "code", "data_hash": "CODEHASH"}]

    def load_step(rp, sh):
        return csv_st if sh == up_sh else tr_st

    def load_step_inputs(rp, sh):
        return ins_csv if sh == up_sh else ins_tr

    monkeypatch.setattr(
        rc,
        "store",
        SimpleNamespace(
            load_step=load_step,
            load_step_inputs=load_step_inputs,
            pick_code_hash=lambda ins: "CODEHASH",
        ),
    )

    # code loader and cas
    monkeypatch.setattr(
        rc, "cas", SimpleNamespace(path_for=lambda rp, kind, h: tmp_path / "code" / h)
    )
    monkeypatch.setattr(
        rc, "code", SimpleNamespace(load_callable_from_blob=lambda p: lambda x: x)
    )

    # bindings
    up_gen = object()

    def run_read_csv_restore(**kw):
        return up_gen

    af_called = {}

    def run_apply_function_restore(**kw):
        af_called.update(kw)
        return "AF_GEN"

    monkeypatch.setattr(
        rc,
        "bindings",
        SimpleNamespace(
            run_read_csv_restore=run_read_csv_restore,
            run_apply_function_restore=run_apply_function_restore,
        ),
    )

    got = rc._build_stream_for_step(tmp_path, t_sh)
    assert got == "AF_GEN"
    assert af_called["realized"]["source"] is up_gen
    assert callable(af_called["transform_fn"])


def test_transform_branch_lineage_manifest(monkeypatch, tmp_path):
    t_sh = "T2"
    ex_sh = "E_images"
    # plan has only the transform; no explicit stream node
    monkeypatch.setattr(
        rc,
        "planner",
        SimpleNamespace(
            plan_for_step=lambda rp, s: [t_sh],
            _db=lambda rp: None,
            _step_row=lambda *_: {"ts_start": "Y"},
        ),
    )
    monkeypatch.setattr(
        rc,
        "_normalize_plan_nodes",
        lambda rp, raw: [
            {"step_hash": t_sh, "function_fqn": "pkg.Transform.apply_function"}
        ],
    )
    monkeypatch.setattr(rc, "_is_stream_fqn", lambda fqn: False)

    # lineage: manifest → producer step is an Extract.read_images
    tr_st = {"function_fqn": "pkg.Transform.apply_function", "params": {"beta": 2}}
    ex_st = {
        "function_fqn": "pkg.Extract.read_images",
        "params": {"directory": "/imgs"},
    }
    ins_tr = [
        {"role": "code", "data_hash": "CODE2"},
        {"role": "manifest", "data_hash": "MANI2"},
        {"role": "batch", "data_hash": "B"},
    ]

    def load_step(rp, sh):
        return ex_st if sh == ex_sh else tr_st

    monkeypatch.setattr(
        rc,
        "store",
        SimpleNamespace(
            load_step=load_step,
            load_step_inputs=lambda rp, s: ins_tr,
            pick_code_hash=lambda ins: "CODE2",
            producer_step_for_data=lambda rp, h: ex_sh,
        ),
    )

    # cas/code
    monkeypatch.setattr(
        rc, "cas", SimpleNamespace(path_for=lambda rp, kind, h: tmp_path / "code" / h)
    )
    monkeypatch.setattr(
        rc, "code", SimpleNamespace(load_callable_from_blob=lambda p: lambda y: y)
    )

    # bindings for images + apply
    def run_read_images_restore(**kw):
        return "IMG_GEN"

    captured = {}

    def run_apply_function_restore(**kw):
        captured.update(kw)
        return "AF2"

    monkeypatch.setattr(
        rc,
        "bindings",
        SimpleNamespace(
            run_read_images_restore=run_read_images_restore,
            run_apply_function_restore=run_apply_function_restore,
        ),
    )

    got = rc._build_stream_for_step(tmp_path, t_sh)
    assert got == "AF2"
    assert captured["realized"]["source"] == "IMG_GEN"  # chained to images stream


def test_transform_branch_missing_anchor_raises(monkeypatch, tmp_path):
    t_sh = "T3"
    monkeypatch.setattr(
        rc,
        "planner",
        SimpleNamespace(
            plan_for_step=lambda rp, s: [t_sh],
            _db=lambda rp: None,
            _step_row=lambda *_: {"ts_start": "Z"},
        ),
    )
    monkeypatch.setattr(
        rc,
        "_normalize_plan_nodes",
        lambda rp, raw: [
            {"step_hash": t_sh, "function_fqn": "pkg.Transform.apply_function"}
        ],
    )
    monkeypatch.setattr(rc, "_is_stream_fqn", lambda fqn: False)

    tr_st = {"function_fqn": "pkg.Transform.apply_function", "params": {}}
    ins_tr = [{"role": "code", "data_hash": "C"}]

    monkeypatch.setattr(
        rc,
        "store",
        SimpleNamespace(
            load_step=lambda rp, s: tr_st,
            load_step_inputs=lambda rp, s: ins_tr,
            pick_code_hash=lambda ins: "C",
            producer_step_for_data=lambda rp, h: None,
            get_materialized_path=lambda *a, **k: None,
            get_recorded_source_path=lambda *a, **k: None,
            get_data_kind=lambda *a, **k: "csv",
        ),
    )

    # materialize.ensure_local will not be reached because there is no 'source' role
    with pytest.raises(SystemExit):
        rc._build_stream_for_step(tmp_path, t_sh)


def test_non_stream_fallback(monkeypatch, tmp_path):
    sh = "S3"
    st_row = {"function_fqn": "pkg.Other.non_stream", "params": {}}
    monkeypatch.setattr(rc, "store", SimpleNamespace(load_step=lambda rp, s: st_row))

    # Fake package + submodule: blase and blase.restore
    import types

    fake_restore = types.ModuleType("blase.restore")
    fake_restore.step = lambda run_path, step_hash, kind="data": "RESTORE_RESULT"

    fake_blase = types.ModuleType("blase")
    fake_blase.restore = fake_restore

    sys.modules["blase"] = fake_blase
    sys.modules["blase.restore"] = fake_restore

    got = rc._build_stream_for_step(tmp_path, sh)
    assert got == "RESTORE_RESULT"


# ----- CLI commands: cmd_list, cmd_show, cmd_plan, cmd_run -----


def write_step(run_dir, *, step_hash, fqn, status="completed", params=None):
    db = run_dir / "nodes" / "nodes.db"
    with sqlite3.connect(db) as c:
        c.execute(
            "INSERT OR REPLACE INTO steps(step_hash,run_id,function_fqn,params_json,ts_start,status) VALUES(?,?,?,?,datetime('now'),?)",
            (step_hash, run_dir.name, fqn, json.dumps(params or {}), status),
        )
        c.commit()


def test_cmd_list_and_show(run_dir, capsys, monkeypatch):
    write_step(run_dir, step_hash="S1", fqn="blase.Transform.apply_function")
    ns = type("A", (object,), {"run": None, "limit": 10, "like_fqn": None})
    monkeypatch.chdir(run_dir.parent)
    rc.cmd_list(ns)
    out = capsys.readouterr().out
    assert "S1" in out

    ns2 = type("A", (object,), {"run": None, "step": "S1"})
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


def test_restore_gen_prefers_parquet_and_returns_binding(monkeypatch):
    monkeypatch.setattr(rc, "planner", _mk_planner_single(None))

    # inputs include only roles the helper collects
    ins = [
        {"data_hash": "X0", "role": "ignore_me"},
        {"data_hash": "B1", "role": "batch"},
        {"data_hash": "B2", "role": "batchmeta"},
        {"data_hash": "B3", "role": "table.batch"},
    ]

    class Store:
        def load_step_inputs(self, run_path, step_hash):
            return ins

        def read_parquet_params_for_manifest(self, run_path, manifest_hash, ts_before):
            # assert ts_before propagated
            assert ts_before == "2025-01-01T00:00:00"
            return {"sources": ["/a", "/b"], "source": "/a"}

        def read_images_params_for_manifest(self, *a, **k):
            pytest.fail("images path should not be used when parquet succeeds")

    SENTINEL = object()

    class Bindings:
        def run_read_parquet_restore(self, *, run_path, params, realized, transform_fn):
            assert params == {"sources": ["/a", "/b"], "source": "/a"}
            assert realized["sources"] == ["/a", "/b"]
            assert realized["source"] == "/a"
            assert realized["manifest"] == "MH"
            assert realized["batch_descs"] == ["B1", "B2", "B3"]
            assert transform_fn is None
            return SENTINEL

    out = rc._restore_gen_for_manifest(
        run_path=Path("/tmp/whatever"),
        manifest_hash="MH",
        step_hash="SH",
        store=Store(),
        bindings=Bindings(),
    )
    assert out is SENTINEL


def test_restore_gen_falls_back_to_images_on_parquet_error(monkeypatch):
    monkeypatch.setattr(rc, "planner", _mk_planner_single(None))

    class Store:
        def load_step_inputs(self, run_path, step_hash):
            return [{"data_hash": "B9", "role": "batch_desc"}]

        def read_parquet_params_for_manifest(self, run_path, manifest_hash, ts_before):
            return {"directory": "/parq_src"}  # valid, but binding will raise

        def read_images_params_for_manifest(self, run_path, manifest_hash, ts_before):
            assert ts_before == "2025-01-01T00:00:00"
            return {"directory": "/img_src"}

    SENTINEL = object()

    class Bindings:
        def run_read_parquet_restore(self, **kw):
            raise RuntimeError("simulated parquet path failure")

        def run_read_images_restore(self, *, run_path, params, realized, transform_fn):
            assert params == {"directory": "/img_src"}
            assert realized["source"] == "/img_src"
            assert realized["manifest"] == "MH"
            assert realized["batch_descs"] == ["B9"]
            assert transform_fn is None
            return SENTINEL

    out = rc._restore_gen_for_manifest(
        run_path=Path("/any"),
        manifest_hash="MH",
        step_hash="SH",
        store=Store(),
        bindings=Bindings(),
    )
    assert out is SENTINEL


def test_restore_gen_raises_when_no_params_available(monkeypatch):
    monkeypatch.setattr(rc, "planner", _mk_planner_single(None))

    class Store:
        def load_step_inputs(self, run_path, step_hash):
            return []  # empty is fine; focus is on missing params

        def read_parquet_params_for_manifest(self, *a, **k):
            return None

        def read_images_params_for_manifest(self, *a, **k):
            return None

    class Bindings:
        def run_read_parquet_restore(self, **kw):  # pragma: no cover
            raise AssertionError("should not be called")

        def run_read_images_restore(self, **kw):  # pragma: no cover
            raise AssertionError("should not be called")

    with pytest.raises(SystemExit) as ei:
        rc._restore_gen_for_manifest(
            run_path=Path("/any"),
            manifest_hash="MH",
            step_hash="SH",
            store=Store(),
            bindings=Bindings(),
        )
    assert "cannot find read_parquet or read_images params for manifest" in str(
        ei.value
    )


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
    write_step(
        run_dir,
        step_hash="S",
        fqn="blase.Load.save_to_csv",
        params={"target": "out.csv"},
    )
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

    monkeypatch.setattr(
        rc.materialize, "ensure_local", lambda *a, **k: (_ for _ in ()).throw(Need())
    )

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


def test_replay_missing_producer_raises(monkeypatch, setup_env):
    tmp_path, _ = setup_env
    # store: no producer found
    monkeypatch.setattr(
        rc,
        "store",
        SimpleNamespace(
            get_data_kind=lambda rp, h: "csv",
            producer_step_for_data=lambda rp, h: None,
        ),
    )
    with pytest.raises(SystemExit, match=r"no producer step recorded"):
        rc.cmd_run(_args(data="X"))


def test_replay_prefers_produced_mapping_and_cleans_others(monkeypatch, setup_env):
    tmp_path, restore_dir = setup_env
    data_hash = "H1"

    # producer step resolved
    monkeypatch.setattr(
        rc,
        "store",
        SimpleNamespace(
            get_data_kind=lambda rp, h: "csv",
            producer_step_for_data=lambda rp, h: "STEP1",
        ),
    )

    # produced mapping includes our hash; created_paths has extra files
    final = restore_dir / "final.csv"
    extra1 = restore_dir / "tmp1.csv"
    extra2 = restore_dir / "tmp2.csv"
    for p in (final, extra1, extra2):
        p.write_text("x")

    def _exec(run_path, prod, to_path, backend_override, seen_steps, created_paths):
        created_paths.extend([final, extra1, extra2])
        return {data_hash: final}

    monkeypatch.setattr(rc, "_exec_plan_for_step", _exec)

    # run
    code = rc.cmd_run(_args(data=data_hash))
    assert code == 0
    assert final.exists()
    # extras should be deleted
    assert not extra1.exists()
    assert not extra2.exists()


def test_replay_fallback_to_to_path_when_not_in_mapping(monkeypatch, setup_env):
    tmp_path, _ = setup_env
    data_hash = "H2"
    out = tmp_path / "custom.csv"
    out.write_text("x")

    monkeypatch.setattr(
        rc,
        "store",
        SimpleNamespace(
            get_data_kind=lambda rp, h: "csv",
            producer_step_for_data=lambda rp, h: "STEP2",
        ),
    )
    # produced has no entry for our hash
    monkeypatch.setattr(
        rc, "_exec_plan_for_step", lambda *a, **k: {"OTHER": Path("irrelevant.csv")}
    )

    code = rc.cmd_run(_args(data=data_hash, to=str(out)))
    assert code == 0  # verifies and returns 0


def test_replay_fallback_to_restore_default_name(monkeypatch, setup_env):
    tmp_path, restore_dir = setup_env
    data_hash = "H3"
    # expected default file name comes from sink params['target'] if present
    default_name = "sink_out.csv"
    expected_path = restore_dir / default_name
    expected_path.write_text("x")

    def _producer(rp, h):
        return "SINKSTEP"

    def _load_step(rp, sh):
        return {"params": {"target": default_name}}

    monkeypatch.setattr(
        rc,
        "store",
        SimpleNamespace(
            get_data_kind=lambda rp, h: "csv",
            producer_step_for_data=_producer,
            load_step=_load_step,
        ),
    )
    # produced has no mapping for data_hash → triggers fallback
    monkeypatch.setattr(rc, "_exec_plan_for_step", lambda *a, **k: {})

    code = rc.cmd_run(_args(data=data_hash))
    assert code == 0


def test_replay_verify_branch_with_non_sink_stream(monkeypatch, setup_env):
    tmp_path, restore_dir = setup_env
    data_hash = "H4"

    args = _args(data=data_hash, keep_intermediates=True, mode="verify")

    # Producer is a non-sink but stream FQN
    def _producer(rp, h):
        return "STEPX"

    # MUST include params.target so fallback default path can be computed
    default_name = "fallback.csv"

    def _load_step(rp, sh):
        return {
            "function_fqn": "blase.Extract.read_csv",
            "params": {"target": default_name},
        }

    # Create the fallback file so the existence check passes
    (restore_dir / default_name).write_text("x")

    monkeypatch.setattr(
        rc,
        "store",
        SimpleNamespace(
            get_data_kind=lambda rp, h: "csv",
            producer_step_for_data=_producer,
            load_step=_load_step,
        ),
    )
    # produced has no mapping, forces fallback path logic
    monkeypatch.setattr(rc, "_exec_plan_for_step", lambda *a, **k: {})

    # stream detection and builder
    monkeypatch.setattr(rc, "_is_stream_fqn", lambda fqn: True)

    def _gen():
        yield ([1, 2], False)
        yield ([3], True)

    monkeypatch.setattr(rc, "_build_stream_for_step", lambda rp, sh: _gen())
    monkeypatch.setattr(rc, "restore_step", lambda *a, **k: (_ for _ in ()))

    code = rc.cmd_run(args)
    assert code == 0


def test_verify_sink_uses_upstream_gen_and_coercion(monkeypatch, isolate_cmd_run):
    tmp_path, _rd, store_extras = isolate_cmd_run
    st = {"function_fqn": "blase.Load.save_to_csv", "params": {}}
    fake_store = SimpleNamespace(load_step=lambda rp, sh: st, **store_extras)
    monkeypatch.setattr(rc, "store", fake_store)

    def upstream():
        yield ([1, 2, 3], {"meta": 1}, False)  # 3-tuple
        yield ([4], True)  # 2-tuple
        yield ([5, 6])  # 1-tuple -> last=False
        yield "XYZ"  # non-tuple -> last=False

    monkeypatch.setattr(rc, "_upstream_gen_for_sink", lambda rp, sh: upstream())

    assert rc.cmd_run(_args(mode="verify", step="S", data=None, limit_baches=10)) == 0


def test_verify_stream_step_uses_build_stream(monkeypatch, isolate_cmd_run):
    tmp_path, _rd, store_extras = isolate_cmd_run
    st = {"function_fqn": "blase.Extract.read_images", "params": {}}
    fake_store = SimpleNamespace(load_step=lambda rp, sh: st, **store_extras)
    monkeypatch.setattr(rc, "store", fake_store)
    monkeypatch.setattr(rc, "_is_stream_fqn", lambda fqn: True)

    def gen():
        yield ([0, 1], False)
        yield ([2], True)

    monkeypatch.setattr(rc, "_build_stream_for_step", lambda rp, sh: gen())

    assert rc.cmd_run(_args(mode="verify", step="S", data=None, limit_baches=5)) == 0


def test_verify_non_stream_uses_restore_step(monkeypatch, isolate_cmd_run):
    tmp_path, _rd, store_extras = isolate_cmd_run
    st = {"function_fqn": "blase.Other.non_stream", "params": {}}
    fake_store = SimpleNamespace(load_step=lambda rp, sh: st, **store_extras)
    monkeypatch.setattr(rc, "store", fake_store)
    monkeypatch.setattr(rc, "_is_stream_fqn", lambda fqn: False)

    def gen():
        yield ([1, 2], False)
        yield ([3], True)

    monkeypatch.setattr(rc, "restore_step", lambda rp, sh, kind="data": gen())

    assert rc.cmd_run(_args(mode="verify", step="S", data=None, limit_baches=1)) == 0


def test_replay_sink_calls_handler(monkeypatch, isolate_cmd_run):
    tmp_path, _rd, store_extras = isolate_cmd_run
    st = {
        "function_fqn": "blase.Load.save_images_to_parquet",
        "params": {"shard_prefix": "x"},
    }
    fake_store = SimpleNamespace(load_step=lambda rp, sh: st, **store_extras)
    monkeypatch.setattr(rc, "store", fake_store)

    upstream_obj = object()
    monkeypatch.setattr(rc, "_upstream_gen_for_sink", lambda rp, sh: upstream_obj)

    recorded = {}

    def handler(**kw):
        recorded.update(kw)
        return tmp_path / "dir" / "x_1.parquet"

    monkeypatch.setattr(
        rc,
        "bindings",
        SimpleNamespace(
            RESTORE_HANDLERS={"blase.Load.save_images_to_parquet": handler}
        ),
    )

    args = _args(mode="replay", step="S", data=None, to=str(tmp_path / "dir"))
    assert rc.cmd_run(args) == 0
    assert recorded["run_path"] == tmp_path
    assert recorded["params"] is st["params"]
    assert recorded["upstream_gen"] is upstream_obj
    assert recorded["target_override"] == str(tmp_path / "dir")


def test_replay_non_stream_streams_batches(monkeypatch, isolate_cmd_run):
    tmp_path, _rd, store_extras = isolate_cmd_run

    st = {"function_fqn": "blase.Transform.apply_function", "params": {}}
    fake_store = SimpleNamespace(load_step=lambda rp, sh: st, **store_extras)
    monkeypatch.setattr(rc, "store", fake_store)

    monkeypatch.setattr(rc, "_is_stream_fqn", lambda fqn: True)

    def gen():
        yield ([1, 2, 3], False)
        yield ([4], True)

    monkeypatch.setattr(rc, "_build_stream_for_step", lambda rp, sh: gen())

    # ensure step-centric, not data-centric
    args = _args(mode="replay", step="S", data=None, limit_batches=1)
    assert rc.cmd_run(args) == 0
