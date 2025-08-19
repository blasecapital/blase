from __future__ import annotations
from pathlib import Path
import sqlite3
from typing import Dict, List, Optional, Literal, Tuple
from . import store

def _db(run_path: Path) -> Path:
    return (run_path / "nodes" / "nodes.db").resolve()

def _fetchone(db: Path, sql: str, args: Tuple) -> Optional[sqlite3.Row]:
    con = sqlite3.connect(db)
    try:
        con.row_factory = sqlite3.Row
        cur = con.execute(sql, args)
        return cur.fetchone()
    finally:
        con.close()

def _fetchall(db: Path, sql: str, args: Tuple) -> list[sqlite3.Row]:
    con = sqlite3.connect(db)
    try:
        con.row_factory = sqlite3.Row
        cur = con.execute(sql, args)
        return cur.fetchall()
    finally:
        con.close()

def _step_row(db: Path, step_hash: str) -> sqlite3.Row:
    row = _fetchone(db, "SELECT * FROM steps WHERE step_hash=?", (step_hash,))
    if not row:
        raise KeyError(f"step not found: {step_hash}")
    return row

def _source_hash_for_step(db: Path, step_hash: str) -> Optional[str]:
    r = _fetchone(db, "SELECT data_hash FROM step_inputs WHERE step_hash=? AND role='source'", (step_hash,))
    return r["data_hash"] if r else None

def _latest_with_source_before(db: Path, fqn_like: str, source_hash: str, ts_end: str) -> Optional[str]:
    r = _fetchone(
        db,
        """
        SELECT s.step_hash
        FROM steps s
        JOIN step_inputs i ON i.step_hash = s.step_hash
        WHERE s.function_fqn LIKE ?
          AND i.role='source' AND i.data_hash=?
          AND s.ts_start <= ?
        ORDER BY s.ts_start DESC
        LIMIT 1
        """,
        (fqn_like, source_hash, ts_end),
    )
    return r["step_hash"] if r else None

def plan_for_step(run_path: Path, tip_step_hash: str) -> List[str]:
    """
    Returns an ordered list of step_hashes to replay, ending with tip_step_hash.
    Heuristic: chain by the same 'source' blob and timestamp ordering.
    """
    db = _db(run_path)
    tip = _step_row(db, tip_step_hash)
    fqn = tip["function_fqn"]
    ts = tip["ts_start"]
    src = _source_hash_for_step(db, tip_step_hash)

    if fqn.endswith("Load.save_to_csv"):
        plan: List[str] = []
        if src:
            tr = _latest_with_source_before(db, "%blase.Transform.apply_function%", src, ts)
            if tr:
                ex = _latest_with_source_before(db, "%blase.Extract.read_csv%", src, _step_row(db, tr)["ts_start"])
                if ex: plan.append(ex)
                plan.append(tr)
        plan.append(tip_step_hash)
        return plan

    if fqn.endswith("Transform.apply_function"):
        plan = []
        if src:
            ex = _latest_with_source_before(db, "%blase.Extract.read_csv%", src, ts)
            if ex: plan.append(ex)
        plan.append(tip_step_hash)
        return plan

    if fqn.endswith("Extract.read_csv"):
        return [tip_step_hash]

    # Fallback: just the tip
    return [tip_step_hash]

Action = Dict[str, str]  # keep it simple: {"type": "...", ...}

def plan_data(run_path: Path, target_data_hash: str, policy: Literal["reuse","replay"]="replay") -> List[Action]:
    # If we can reuse, do it
    mats = store.find_materializations(run_path, target_data_hash)
    if mats and policy == "reuse":
        return [{"type": "reuse", "data_hash": target_data_hash, "path": mats[0]["path"]}]

    # Build back-edge graph and compute forward order
    graph = store.build_dependency_graph(run_path, target_data_hash)  # implement as in prior guidance
    steps_in_order = store.toposort_steps(graph)

    actions: List[Action] = []
    for sh in steps_in_order:
        st = store.load_step(run_path, sh)
        actions.append({"type": "replay_step", "step_hash": sh, "function_fqn": st["function_fqn"]})

    prod = store.producer_step_for_data(run_path, target_data_hash)
    if prod and (not steps_in_order or prod != steps_in_order[-1]):
        st = store.load_step(run_path, prod)
        actions.append({"type": "replay_step", "step_hash": prod, "function_fqn": st["function_fqn"]})

    actions.append({"type": "verify", "data_hash": target_data_hash})
    return actions