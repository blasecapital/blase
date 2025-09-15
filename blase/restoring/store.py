from __future__ import annotations
from typing import Dict, Any, List, Optional
from pathlib import Path
import sqlite3
import json
from time import strftime


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

def kind_for_hash(run_path: Path, data_hash: str) -> str:
    db = run_path / "nodes" / "nodes.db"
    with _conn(db) as c:
        r = c.execute("SELECT kind FROM data WHERE data_hash=?", (data_hash,)).fetchone()
    return r[0] if r else None

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

def producer_step_for_data(run_path: Path, data_hash: str) -> Optional[str]:
    db = run_path / "nodes" / "nodes.db"
    with _conn(db) as c:
        row = c.execute("SELECT step_hash FROM step_outputs WHERE data_hash=?", (data_hash,)).fetchone()
    return row[0] if row else None

def read_images_params_for_manifest(run_path: Path, manifest_hash: str, ts_before: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Return the params_json dict from the Extract.read_images step that produced `manifest_hash`.
    If ts_before is given, pick the latest at-or-before that timestamp.
    """
    db = run_path / "nodes" / "nodes.db"
    sql = """
      SELECT s.params_json
      FROM steps s
      JOIN step_outputs o ON o.step_hash = s.step_hash
      WHERE s.function_fqn = 'blase.Extract.read_images'
        AND o.data_hash = ?
    """
    args = [manifest_hash]
    if ts_before:
        sql += " AND s.ts_start <= ?"
        args.append(ts_before)
    sql += " ORDER BY s.ts_start DESC LIMIT 1"
    with _conn(db) as c:
        row = c.execute(sql, tuple(args)).fetchone()
    if not row:
        return None
    try:
        params = json.loads(row[0])
    except Exception:
        return None

    # Provide sane defaults for keys used by restore.
    params.setdefault("backend", "pil")
    params.setdefault("return_type", "np")
    params.setdefault("color", "rgb")
    params.setdefault("pattern", "**/*.jpg")
    params.setdefault("recursive", True)
    params.setdefault("shuffle", False)
    params.setdefault("safety_margin", 0.15)
    params.setdefault("hash_mode", "content")
    return params
