from __future__ import annotations
from pathlib import Path
import sqlite3
from typing import List, Optional, Tuple

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

def _step_row(db: Path, step_hash: str) -> sqlite3.Row:
    row = _fetchone(db, "SELECT * FROM steps WHERE step_hash=?", (step_hash,))
    if not row:
        raise KeyError(f"step not found: {step_hash}")
    return row


def _input_hash_for_role(db, step_hash: str, role: str) -> Optional[str]:
    r = _fetchone(
        db,
        "SELECT data_hash FROM step_inputs WHERE step_hash = ? AND role = ? LIMIT 1",
        (step_hash, role),
    )
    return r["data_hash"] if r else None

def _latest_with_input_before(db, fqn_like: str, *, role: str, data_hash: str, ts_before: str) -> Optional[str]:
    r = _fetchone(
        db,
        """
        SELECT s.step_hash
        FROM steps s
        JOIN step_inputs i ON i.step_hash = s.step_hash
        WHERE s.function_fqn LIKE ?
          AND i.role = ?
          AND i.data_hash = ?
          AND s.ts_start <= ?
        ORDER BY s.ts_start DESC
        LIMIT 1
        """,
        (fqn_like, role, data_hash, ts_before),
    )
    return r["step_hash"] if r else None

# Back-compat wrappers (CSV-style "source")
def _source_hash_for_step(db, step_hash: str) -> Optional[str]:
    return _input_hash_for_role(db, step_hash, "source")

def _latest_with_source_before(db, fqn_like: str, data_hash: str, ts_before: str) -> Optional[str]:
    return _latest_with_input_before(db, fqn_like, role="source", data_hash=data_hash, ts_before=ts_before)

def _latest_producer_of_output_before(db, fqn_like: str, *, data_hash: str, ts_before: str) -> Optional[str]:
    r = _fetchone(
        db,
        """
        SELECT s.step_hash
        FROM steps s
        JOIN step_outputs o ON o.step_hash = s.step_hash
        WHERE s.function_fqn LIKE ?
          AND o.data_hash = ?
          AND s.ts_start <= ?
        ORDER BY s.ts_start DESC
        LIMIT 1
        """,
        (fqn_like, data_hash, ts_before),
    )
    return r["step_hash"] if r else None

def plan_for_step(run_path: Path, tip_step_hash: str) -> List[str]:
    """
    Construct a replay plan for a given pipeline step.

    This function inspects the run database and builds a minimal,
    ordered list of step hashes that should be replayed to reproduce
    the requested tip step. The plan ends with the specified
    ``tip_step_hash`` and may include prerequisite steps (e.g.,
    Extract or Transform stages) based on heuristics that trace back
    through the same data source and execution timestamps.

    Parameters
    ----------
    run_path : Path
        Root path of the run containing the step database.
    tip_step_hash : str
        Hash identifier of the target step to replay.

    Returns
    -------
    List[str]
        Ordered list of step hashes to replay, ending with
        ``tip_step_hash``.

    Notes
    -----
    - The heuristic relies on:
      
      * **Function FQN (fully-qualified name)** of the step
        (e.g., ``Load.save_to_csv``).
      * **Source blob hash**, indicating the originating data.
      * **Timestamps**, to ensure causal ordering.

    - Special cases handled:
      
      * ``Load.save_to_csv``: Plan includes the latest Transform
        (``Transform.apply_function``) on the same source before the tip,
        and optionally the Extract (``Extract.read_csv``) that fed it.
      * ``Transform.apply_function``: Plan includes the corresponding
        Extract step before the tip.
      * ``Extract.read_csv``: Plan consists only of the tip.
      * All other steps: Plan defaults to just the tip.

    - This logic ensures replay sequences are minimal but sufficient
      for reconstructing derived data artifacts.

    Examples
    --------
    >>> from pathlib import Path
    >>> plan_for_step(Path("runs/2024-08-01T12-00-00"), "step123")
    ['step045', 'step078', 'step123']

    For an Extract step:
    >>> plan_for_step(Path("runs/2024-08-01T12-00-00"), "extract_hash")
    ['extract_hash']

    For an unknown step type:
    >>> plan_for_step(Path("runs/2024-08-01T12-00-00"), "misc_hash")
    ['misc_hash']
    """
    db  = _db(run_path)
    tip = _step_row(db, tip_step_hash)
    fqn = tip["function_fqn"]
    ts  = tip["ts_start"]
    # CSV-style source if present (back-compat)
    src = _source_hash_for_step(db, tip_step_hash)
    plan: List[str] = []

    # ---------------------------
    # Load → CSV (existing path)
    # ---------------------------
    if fqn.endswith("Load.save_to_csv"):
        if src:
            tr = _latest_with_source_before(db, "%blase.Transform.apply_function%", src, ts)
            if tr:
                # prefer image lineage if transform had a manifest
                man = _input_hash_for_role(db, tr, "manifest")
                if man:
                    ex_img = _latest_with_input_before(
                        db, "%blase.Extract.read_images%",
                        role="manifest", data_hash=man, ts_before=_step_row(db, tr)["ts_start"]
                    )
                    if ex_img:
                        plan.append(ex_img)
                else:
                    ex_csv = _latest_with_source_before(
                        db, "%blase.Extract.read_csv%", src, _step_row(db, tr)["ts_start"]
                    )
                    if ex_csv:
                        plan.append(ex_csv)
                plan.append(tr)
        plan.append(tip_step_hash)
        return plan

    # ---------------------------------------------
    # Load → images→parquet (prefer manifest)
    # ---------------------------------------------
    elif fqn.endswith("Load.save_images_to_parquet"):
        man = _input_hash_for_role(db, tip_step_hash, "manifest")
        tr = None
        if man:
            tr = _latest_with_input_before(
                db, "%blase.Transform.apply_function%",
                role="manifest", data_hash=man, ts_before=ts
            )
        if tr is None and src:
            tr = _latest_with_source_before(db, "%blase.Transform.apply_function%", src, ts)

        if tr:
            tr_ts  = _step_row(db, tr)["ts_start"]
            man_tr = _input_hash_for_role(db, tr, "manifest")

            ex_img = None
            if man_tr:
                ex_img = _latest_producer_of_output_before(
                    db, "%blase.Extract.read_images%", data_hash=man_tr, ts_before=tr_ts
                )
                if ex_img is None:
                    ex_img = _latest_with_input_before(
                        db, "%blase.Extract.read_images%",
                        role="manifest", data_hash=man_tr, ts_before=tr_ts
                    )

            if ex_img:
                plan.append(ex_img)
            else:
                src_tr = _source_hash_for_step(db, tr)
                if src_tr:
                    ex_csv = _latest_with_source_before(db, "%blase.Extract.read_csv%", src_tr, tr_ts)
                    if ex_csv:
                        plan.append(ex_csv)
            plan.append(tr)
        else:
            # fallback to CSV/source lineage
            src_tr = _source_hash_for_step(db, tr)
            if src_tr:
                ex_csv = _latest_with_source_before(db, "%blase.Extract.read_csv%", src_tr, tr_ts)
                if ex_csv:
                    plan.append(ex_csv)

            plan.append(tr)

        plan.append(tip_step_hash)
        return plan

    # ---------------------------
    # Transform (images or csv)
    # ---------------------------
    elif fqn.endswith("Transform.apply_function"):
        man = _input_hash_for_role(db, tip_step_hash, "manifest")
        if man:
            ex_img = _latest_producer_of_output_before(
                db, "%blase.Extract.read_images%", data_hash=man, ts_before=ts
            )
            if ex_img:
                plan.append(ex_img)
        elif src:
            ex_csv = _latest_with_source_before(db, "%blase.Extract.read_csv%", src, ts)
            if ex_csv:
                plan.append(ex_csv)
        plan.append(tip_step_hash)
        return plan

    # ---------------------------
    # Extracts
    # ---------------------------
    elif fqn.endswith("Extract.read_csv"):
        return [tip_step_hash]

    elif fqn.endswith("Extract.read_images"):
        return [tip_step_hash]

    # ---------------------------
    # Fallback
    # ---------------------------
    return [tip_step_hash]
