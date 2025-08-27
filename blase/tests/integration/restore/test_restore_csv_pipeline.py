import csv
from types import ModuleType
import sqlite3
import sys
import os
import importlib
import subprocess
from pathlib import Path

import pytest


# ----- Helpers -----

def import_blase_fresh(project_root: Path) -> ModuleType:
    os.environ["BLASE_HOME"] = str(project_root)
    os.chdir(project_root)  # match what your lib expects

    # Purge blase modules to avoid cached globals (runs root, config, bindings, etc.)
    to_delete = [m for m in list(sys.modules) if m == "blase" or m.startswith("blase.")]
    for m in to_delete:
        sys.modules.pop(m, None)

    return importlib.import_module("blase")


def workspace(tmp_path, monkeypatch):
    """Create isolated project and pin BLASE_HOME so runs land here."""
    project_root = tmp_path / "proj"
    project_root.mkdir()
    monkeypatch.chdir(project_root)
    monkeypatch.setenv("BLASE_HOME", str(project_root))

    # proactively create runs/ so the lib won't pick project_root.parent/runs
    (project_root / "runs").mkdir(parents=True, exist_ok=True)

    data_root = project_root / "data" / "source_of_truth"
    data_root.mkdir(parents=True, exist_ok=True)
    return project_root, data_root


def gen_data(data_root: Path) -> Path:
    src = data_root / "House_Rent_Dataset.csv"
    with src.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Posted On", "BHK", "Rent", "Size", "City"])
        for i in range(3001):
            w.writerow([
                f"2024-01-{(i % 28) + 1:02d}",
                (i % 4) + 1,
                1000 + i,
                500 + (i % 100),
                f"City{(i % 5)}",
            ])
    return src


def run_pipeline_once(src: Path, blase_mod=None) -> Path:
    blase = blase_mod or import_blase_fresh(src.parents[3]) 
    ex, tr, ld = blase.Extract(), blase.Transform(), blase.Load()

    def func_pandas(df):
        df = df.copy()
        if "Rent_per_sqft" not in df.columns:
            df["Rent_per_sqft"] = df["Rent"] / df["Size"].replace({0: 1})
        return df

    gen = ex.read_csv(str(src), backend="pandas", mode="manual", batch_size=2500, track=True)
    final_path = None
    try:
        for i, (b, last, meta) in enumerate(gen, 1):
            tb, tl, tm = tr.apply_function(
                data=b,
                transform_func=func_pandas,
                last_batch=last,
                meta=meta,
                track=True,
            )
            out, ol, om = ld.save_to_csv(
                data=tb,
                last_batch=tl,
                meta=tm,
                file_name="house_rent_transformed.csv",
                subdir="csv_data",
                backend="pandas",
                track=True,
                use_blase_path=True,
            )
            final_path = Path(out)
            if i >= 2:
                break
        return final_path
    finally:
        try:
            gen.close()
        except Exception:
            pass


def _latest_run_path(project_root: Path) -> Path:
    """
    Find the most recent runs/<id> with nodes/nodes.db under:
      - project_root
      - project_root.parent
      - $BLASE_HOME (if set)
      - parent($BLASE_HOME)
    """
    roots = {project_root, project_root.parent}
    blase_home = Path(os.environ.get("BLASE_HOME", str(project_root)))
    roots.add(blase_home)
    roots.add(blase_home.parent)

    candidates = []
    for root in roots:
        rr = root / "runs"
        if not rr.exists():
            continue
        for d in rr.iterdir():
            if not d.is_dir():
                continue
            db = d / "nodes" / "nodes.db"
            if db.exists():
                try:
                    mtime = db.stat().st_mtime
                except Exception:
                    mtime = d.stat().st_mtime
                candidates.append((mtime, d))

    if not candidates:
        # helpful debug if this ever trips again
        details = []
        for root in roots:
            rr = root / "runs"
            try:
                items = list(rr.glob("**/*"))
            except Exception:
                items = []
            details.append(f"{rr} -> {len(items)} entries")
        raise AssertionError("No runs found with nodes/nodes.db. Scanned:\n" + "\n".join(details))

    candidates.sort(reverse=True)
    return candidates[0][1]


def con(project_root: Path) -> sqlite3.Connection:
    run_path = _latest_run_path(project_root)
    db_path = run_path / "nodes" / "nodes.db"
    c = sqlite3.connect(db_path)
    c.row_factory = sqlite3.Row
    return c


def sink_hash_for_latest(project_root: Path) -> str:
    c = con(project_root)
    try:
        row = c.execute("""
            SELECT step_hash
            FROM steps
            WHERE function_fqn='blase.Load.save_to_csv'
              AND status IN ('completed','complete','done','success','ok')
            ORDER BY ts_start DESC LIMIT 1
        """).fetchone()
        assert row, "No completed Load.save_to_csv step recorded"
        return row["step_hash"]
    finally:
        c.close()


def data_hash_for_step(project_root: Path, step_hash: str) -> str:
    c = con(project_root)
    try:
        row = c.execute("SELECT data_hash FROM step_outputs WHERE step_hash=?", (step_hash,)).fetchone()
        assert row, "No output recorded for sink step"
        return row["data_hash"]
    finally:
        c.close()


def step_replay(project_root: Path, step_hash: str, final_path: Path) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable, "-m", "blase.cli.cli",
        "restore", "run",
        "--step", step_hash,
        "--mode", "replay",
        "--to", str(final_path),
        "--on-conflict", "overwrite",
    ]
    return subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)


def data_replay(project_root: Path, data_hash: str, final_path: Path):
    run_path = _latest_run_path(project_root)
    cmd = [
        sys.executable, "-m", "blase.cli.cli",
        "restore", "run",
        "--run", str(run_path),
        "--data", data_hash,
        "--mode", "replay",
        "--to", str(final_path),
        "--on-conflict", "overwrite",
    ]
    return subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)


def data_materialize(project_root: Path, data_hash: str, final_path: Path):
    run_path = _latest_run_path(project_root)
    cmd = [
        sys.executable, "-m", "blase.cli.cli",
        "restore", "run",
        "--run", str(run_path),
        "--data", data_hash,
        "--mode", "materialize",
        "--to", str(final_path),
        "--on-conflict", "overwrite",
    ]
    return subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)


# ----- Tests -----

@pytest.mark.integration
def test_restore_cli_step_replay(tmp_path, monkeypatch):
    from blase.utils.hashing import Hash
    
    project_root, data_root = workspace(tmp_path, monkeypatch)
    blase = import_blase_fresh(project_root)
    src = gen_data(data_root)

    final_path = run_pipeline_once(src, blase_mod=blase)
    assert final_path and final_path.exists()
    orig_hash = Hash().hash_file(final_path)

    # Discover sink step + data hash from DB
    shash = sink_hash_for_latest(project_root)
    dhash = data_hash_for_step(project_root, shash)

    # Delete to force restore
    final_path.unlink()
    assert not final_path.exists()

    # Step replay (note: may not enforce byte identity depending on your handler)
    res = step_replay(project_root, shash, final_path)
    assert res.returncode == 0, f"CLI failed: {res.stdout}\n{res.stderr}"
    assert final_path.exists()

    # If your step replay path doesn’t enforce expected hash, this assert can fail.
    # If it does, great — keep it. Otherwise, switch this to a smoke check.
    restored = Hash().hash_file(final_path)
    assert restored == orig_hash, f"Mismatch: {restored} != {orig_hash}"

    # Clean up
    final_path.unlink(missing_ok=True)

    # Data materialize (guaranteed to enforce the original bytes)
    res = data_materialize(project_root, dhash, final_path)
    assert res.returncode == 0, f"CLI failed: {res.stdout}\n{res.stderr}"
    assert final_path.exists()
    assert Hash().hash_file(final_path) == orig_hash

    final_path.unlink(missing_ok=True)

@pytest.mark.integration
def test_restore_cli_duplicate_append_then_restore(tmp_path, monkeypatch):
    """
    Run the ETL twice with identical params/target so the sink appends duplicate rows.
    Restore by DATA hash and verify we get the post-append bytes (via replay & materialize).
    """
    from blase.utils.hashing import Hash

    project_root, data_root = workspace(tmp_path, monkeypatch)
    blase = import_blase_fresh(project_root)
    src = gen_data(data_root)

    # run twice → append duplicate
    final_path = run_pipeline_once(src, blase_mod=blase)
    hash1 = Hash().hash_file(final_path)
    final_path2 = run_pipeline_once(src, blase_mod=blase)
    assert final_path2 == final_path
    hash2 = Hash().hash_file(final_path)
    assert hash2 != hash1

    # sanity: doubled rows
    total_rows = sum(1 for _ in csv.reader(final_path.open()) ) - 1
    assert total_rows == 3001 * 2

    # ---- Prefer materialize FIRST (fast-path copy) WITHOUT deleting the source
    restored_copy = final_path.with_name(final_path.stem + "_restored.csv")
    if restored_copy.exists():
        restored_copy.unlink()
    res = data_materialize(project_root, hash2, restored_copy)
    assert res.returncode == 0, f"CLI materialize failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert restored_copy.exists()
    assert Hash().hash_file(restored_copy) == hash2
    restored_copy.unlink(missing_ok=True)

    # ---- If you want to keep a replay check, make it xfail until CLI/planner is fixed
    final_path.unlink(missing_ok=True)  # remove the original now so replay must generate
    res = data_replay(project_root, hash2, final_path)
    assert Hash().hash_file(final_path) == hash2
    final_path.unlink(missing_ok=True)