from __future__ import annotations
from typing import Dict, Any, List, Optional
from pathlib import Path
import sqlite3, json
from collections import deque, defaultdict
from time import strftime

from blase.restoring.materialize import _valid_local_candidates

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
    """Return the step_hash that produced data_hash (or None if unknown)."""
    db = run_path / "nodes" / "nodes.db"
    with _conn(db) as c:
        row = c.execute("SELECT step_hash FROM step_outputs WHERE data_hash=?", (data_hash,)).fetchone()
    return row[0] if row else None

def find_materializations(run_path: Path, data_hash: str) -> list[Dict[str, Any]]:
    """Return all known materializations for a data_hash, newest first."""
    db = run_path / "nodes" / "nodes.db"
    with _conn(db) as c:
        rows = c.execute(
            "SELECT path, ts FROM materializations WHERE data_hash=? ORDER BY ts DESC",
            (data_hash,)
        ).fetchall()
    return [{"path": r[0], "ts": r[1]} for r in rows]

def is_anchor_data(run_path: Path, data_hash: str) -> bool:
    """An anchor is something we don't need to replay: code/env, valid materialization, or live source_path that hash-matches."""
    kind = get_data_kind(run_path, data_hash) or "data"
    if kind in ("code","env"):
        return True
    return bool(_valid_local_candidates(run_path, data_hash))

def inputs_for_step(run_path: Path, step_hash: str) -> list[dict]:
    """Return [{'data_hash':..., 'role':..., 'arg_name':...}, ...]"""
    return load_step_inputs(run_path, step_hash)

def producer_for_data(run_path: Path, data_hash: str) -> str | None:
    return producer_step_for_data(run_path, data_hash)  # you added this helper earlier

def outputs_for_step(run_path: Path, step_hash: str) -> list[str]:
    outs = load_step_outputs(run_path, step_hash)
    return [o["data_hash"] for o in outs]

def build_dependency_graph(run_path: Path, target_data_hash: str) -> dict[str, set[str]]:
    """
    Return a graph: step_hash -> set(upstream_step_hashes)
    Stops recursion at anchors (no upstream step needed).
    """
    graph: dict[str, set[str]] = {}
    visited_data: set[str] = set()
    stack: list[str] = [target_data_hash]

    while stack:
        dh = stack.pop()
        if dh in visited_data:
            continue
        visited_data.add(dh)

        if is_anchor_data(run_path, dh):
            continue

        prod = producer_for_data(run_path, dh)
        if not prod:
            # Unknown producer and not an anchor -> dead end; leave it to error at replay
            continue

        # Ensure node exists
        graph.setdefault(prod, set())

        # For this producer step, traverse each non-anchor input back to its producer
        for i in inputs_for_step(run_path, prod):
            ih = i["data_hash"]
            if is_anchor_data(run_path, ih):
                continue
            up = producer_for_data(run_path, ih)
            if up:
                graph.setdefault(up, set())
                graph[prod].add(up)
                stack.append(ih)
            else:
                # No producer; if not anchor, we'll error later
                pass

    return graph

def toposort_steps(graph: dict[str, set[str]]) -> list[str]:
    """Return a forward order of steps (producers before dependents)."""
    indeg = defaultdict(int)
    for n in graph:
        indeg.setdefault(n, 0)
    for n, ups in graph.items():
        for u in ups:
            indeg[n] += 1
            indeg.setdefault(u, 0)
    q = deque([n for n, d in indeg.items() if d == 0])
    order: list[str] = []
    # Build reverse edges for decrement
    rev = {n: set() for n in indeg}
    for n, ups in graph.items():
        for u in ups:
            rev[u].add(n)
    while q:
        u = q.popleft()
        order.append(u)
        for v in rev[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(order) != len(indeg):
        raise RuntimeError("Cycle detected in restore graph (should not happen).")
    return order