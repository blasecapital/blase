from __future__ import annotations
from typing import Dict, Any, List, Optional
from pathlib import Path
import sqlite3, json

def _conn(db: Path) -> sqlite3.Connection:
    c = sqlite3.connect(db)
    c.execute("PRAGMA journal_mode=WAL;")
    c.execute("PRAGMA synchronous=NORMAL;")
    return c

def load_step(run_path: Path, step_hash: str) -> Dict[str, Any]:
    db = run_path / "nodes" / "nodes.db"
    with _conn(db) as c:
        row = c.execute(
            "SELECT function_fqn, params_json, status FROM steps WHERE step_hash=?", (step_hash,)
        ).fetchone()
    if not row:
        raise KeyError(f"step not found: {step_hash}")
    function_fqn, params_json, status = row
    return {"function_fqn": function_fqn, "params": json.loads(params_json or "{}"), "status": status}

def load_step_inputs(run_path: Path, step_hash: str) -> List[Dict[str, Any]]:
    db = run_path / "nodes" / "nodes.db"
    with _conn(db) as c:
        rows = c.execute(
            "SELECT data_hash, role, COALESCE(arg_name,'') FROM step_inputs WHERE step_hash=?",
            (step_hash,),
        ).fetchall()
    return [{"data_hash": h, "role": r, "arg_name": a or None} for (h, r, a) in rows]

def load_data_kinds(run_path: Path, inputs):
    db = run_path / "nodes" / "nodes.db"
    hashes = tuple({i["data_hash"] for i in inputs if i["role"] not in ("code","env")})
    sql = f"SELECT data_hash, kind FROM data WHERE data_hash IN ({','.join(['?']*len(hashes))})"
    with _conn(db) as c:
        rows = c.execute(sql, hashes).fetchall()
    return {h: k for (h, k) in rows}

def pick_code_hash(inputs: List[Dict[str, Any]]) -> str:
    for i in inputs:
        if i["role"] == "code":
            return i["data_hash"]
    raise KeyError("no code blob recorded for step")

def first_input_by_role(inputs: List[Dict[str, Any]], role: str) -> str:
    for i in inputs:
        if i["role"] == role:
            return i["data_hash"]
    raise KeyError(f"no input with role='{role}'")

def lookup_source_path(run_path: Path, data_hash: str) -> Optional[str]:
    db = run_path / "nodes" / "nodes.db"
    with _conn(db) as c:
        row = c.execute("SELECT source_path FROM data WHERE data_hash=?", (data_hash,)).fetchone()
    if not row:
        return None
    return row[0] or None

def load_materializations(run_path: Path, data_hash: str) -> List[str]:
    db = run_path / "nodes" / "nodes.db"
    with _conn(db) as c:
        rows = c.execute("SELECT path FROM materializations WHERE data_hash=?", (data_hash,)).fetchall()
    return [r[0] for r in rows]

def drop_materialization(run_path: Path, data_hash: str, path: str) -> None:
    db = run_path / "nodes" / "nodes.db"
    with _conn(db) as c:
        c.execute("DELETE FROM materializations WHERE data_hash=? AND path=?", (data_hash, path))
        c.commit()

def load_data_source_path(run_path: Path, data_hash: str) -> Optional[str]:
    db = run_path / "nodes" / "nodes.db"
    with _conn(db) as c:
        row = c.execute("SELECT source_path FROM data WHERE data_hash=?", (data_hash,)).fetchone()
    return row[0] if row and row[0] else None

def get_data_kind(run_path: Path, data_hash: str) -> Optional[str]:
    db = run_path / "nodes" / "nodes.db"
    with _conn(db) as c:
        row = c.execute("SELECT kind FROM data WHERE data_hash=?", (data_hash,)).fetchone()
    return row[0] if row else None

def load_step_outputs(run_path: Path, step_hash: str) -> List[Dict[str, Any]]:
    db = run_path / "nodes" / "nodes.db"
    with _conn(db) as c:
        rows = c.execute("SELECT data_hash, name FROM step_outputs WHERE step_hash=?", (step_hash,)).fetchall()
    return [{"data_hash": r[0], "name": r[1]} for r in rows]

def record_materialization(run_path: Path, data_hash: str, path: str) -> None:
    from time import strftime
    db = run_path / "nodes" / "nodes.db"
    with _conn(db) as c:
        c.execute("""INSERT OR IGNORE INTO materializations(data_hash, path, ts)
                     VALUES (?,?,?)""", (data_hash, path, strftime("%Y-%m-%dT%H:%M:%S")))
        c.commit()

def get_materialized_path(run_path: Path, data_hash: str) -> Optional[Path]:
    db = run_path / "nodes" / "nodes.db"
    with _conn(db) as c:
        row = c.execute(
            "SELECT path FROM materializations WHERE data_hash=? ORDER BY ts DESC LIMIT 1",
            (data_hash,)
        ).fetchone()
    return Path(row[0]) if row and row[0] else None

def get_recorded_source_path(run_path: Path, data_hash: str) -> Optional[Path]:
    db = run_path / "nodes" / "nodes.db"
    with _conn(db) as c:
        row = c.execute(
            "SELECT source_path FROM data WHERE data_hash=?",
            (data_hash,)
        ).fetchone()
    return Path(row[0]) if row and row[0] else None