import os
import sys
import json
import sqlite3
import subprocess
import importlib
from types import ModuleType
from pathlib import Path
import pytest

# ---------- helpers ----------

def import_blase_fresh(project_root: Path) -> ModuleType:
    os.environ["BLASE_HOME"] = str(project_root)
    cwd = os.getcwd()
    os.chdir(project_root)
    try:
        for m in [m for m in list(sys.modules) if m == "blase" or m.startswith("blase.")]:
            sys.modules.pop(m, None)
        return importlib.import_module("blase")
    finally:
        os.chdir(cwd)

def workspace(tmp_path, monkeypatch):
    project_root = tmp_path / "proj_classmap_restore"
    project_root.mkdir(parents=True, exist_ok=True)
    (project_root / "runs").mkdir(parents=True, exist_ok=True)
    (project_root / "data" / "working").mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(project_root)
    monkeypatch.setenv("BLASE_HOME", str(project_root))
    return project_root

def _latest_run_path(project_root: Path) -> Path:
    runs = project_root / "runs"
    cands = []
    for d in runs.iterdir():
        if not d.is_dir():
            continue
        db = d / "nodes" / "nodes.db"
        if db.exists():
            cands.append((db.stat().st_mtime, d))
    assert cands, "no runs found"
    cands.sort(reverse=True)
    return cands[0][1]

def _con(project_root: Path) -> sqlite3.Connection:
    rp = _latest_run_path(project_root)
    db = rp / "nodes" / "nodes.db"
    c = sqlite3.connect(db)
    c.row_factory = sqlite3.Row
    return c

def latest_step(project_root: Path, fqn: str) -> str:
    c = _con(project_root)
    try:
        row = c.execute(
            """
            SELECT step_hash
            FROM steps
            WHERE function_fqn=? AND status IN ('completed','complete','done','success','ok')
            ORDER BY ts_start DESC
            LIMIT 1
            """,
            (fqn,),
        ).fetchone()
        assert row, f"no completed {fqn} step recorded"
        return row["step_hash"]
    finally:
        c.close()

def outputs_for_step(project_root: Path, step_hash: str):
    c = _con(project_root)
    try:
        rows = c.execute(
            "SELECT data_hash, name FROM step_outputs WHERE step_hash=? ORDER BY name ASC",
            (step_hash,),
        ).fetchall()
        return [{"data_hash": r["data_hash"], "name": r["name"]} for r in rows]
    finally:
        c.close()

def cli_restore_step_to_file(project_root: Path, step_hash: str, to_file: Path):
    run_path = _latest_run_path(project_root)
    cmd = [
        sys.executable, "-m", "blase.cli.cli", "restore", "run",
        "--run", str(run_path),
        "--step", step_hash,
        "--mode", "replay",
        "--to", str(to_file),
        "--on-conflict", "overwrite",
    ]
    return subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)

def cli_restore_data_to(project_root: Path, data_hash: str, to_path: Path):
    run_path = _latest_run_path(project_root)
    cmd = [
        sys.executable, "-m", "blase.cli.cli", "restore", "run",
        "--run", str(run_path),
        "--data", data_hash,
        "--mode", "replay",
        "--to", str(to_path),
        "--on-conflict", "overwrite",
    ]
    return subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)

# ---------- test ----------

@pytest.mark.integration
def test_class_map_io_step_and_data_restore(tmp_path, monkeypatch):
    project_root = workspace(tmp_path, monkeypatch)
    blase = import_blase_fresh(project_root)
    Prepare = blase.Prepare

    prep = Prepare()

    # Create and save a simple class map (tracked)
    cm = {"weed": 0, "radish": 1, "other": 2}
    orig_path = project_root / "data" / "working" / "classmap_orig" / "class_map.json"
    orig_path.parent.mkdir(parents=True, exist_ok=True)
    _ = prep.class_map_io(map=cm, save=orig_path, normalize=True, track=True)

    # Validate written file
    assert orig_path.exists() and orig_path.stat().st_size > 0
    payload = json.loads(orig_path.read_text(encoding="utf-8"))
    # basic sanity: keys present and ids are ints
    assert set(payload.keys()) == set(cm.keys())
    assert all(isinstance(v, int) for v in payload.values())

    # Fetch the recorded step + output hash
    step_hash = latest_step(project_root, "blase.Prepare.class_map_io")
    outs = outputs_for_step(project_root, step_hash)
    assert outs, "class_map_io step recorded no outputs"
    # use the first (only) output
    out_hash = outs[0]["data_hash"]

    # ---- STEP RESTORE → write to a new file via --to
    replay_file = project_root / "data" / "working" / "classmap_replay_step.json"
    if replay_file.exists():
        replay_file.unlink()
    rc = cli_restore_step_to_file(project_root, step_hash, replay_file)
    assert rc.returncode == 0, f"step replay failed\nSTDOUT:\n{rc.stdout}\nSTDERR:\n{rc.stderr}"
    assert replay_file.exists() and replay_file.stat().st_size > 0
    payload2 = json.loads(replay_file.read_text(encoding="utf-8"))
    assert set(payload2.keys()) == set(cm.keys())

    # ---- DATA RESTORE → materialize that specific output hash to a target path
    data_target = project_root / "data" / "working" / f"{out_hash}.json"
    if data_target.exists():
        data_target.unlink()
    rc2 = cli_restore_data_to(project_root, out_hash, data_target)
    assert rc2.returncode == 0, f"data restore failed\nSTDOUT:\n{rc2.stdout}\nSTDERR:\n{rc2.stderr}"
    assert data_target.exists() and data_target.stat().st_size > 0
    payload3 = json.loads(data_target.read_text(encoding="utf-8"))
    assert set(payload3.keys()) == set(cm.keys())