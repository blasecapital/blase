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


def _latest_with_input_before(
    db, fqn_like: str, *, role: str, data_hash: str, ts_before: str
) -> Optional[str]:
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


def _latest_with_source_before(
    db, fqn_like: str, data_hash: str, ts_before: str
) -> Optional[str]:
    return _latest_with_input_before(
        db, fqn_like, role="source", data_hash=data_hash, ts_before=ts_before
    )


def _latest_producer_of_output_before(
    db, fqn_like: str, *, data_hash: str, ts_before: str
) -> Optional[str]:
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


def _append_if(plan: List[str], step_hash: Optional[str]) -> None:
    if step_hash:
        plan.append(step_hash)


def _prefer_parquet_then_images_for_manifest(
    db, manifest_hash: str, ts_before: int
) -> Optional[str]:
    """Return latest extract step for a manifest, preferring parquet then images."""
    ex_tbl = _latest_producer_of_output_before(
        db, "%blase.Extract.read_parquet%", data_hash=manifest_hash, ts_before=ts_before
    ) or _latest_with_input_before(
        db,
        "%blase.Extract.read_parquet%",
        role="manifest",
        data_hash=manifest_hash,
        ts_before=ts_before,
    )
    if ex_tbl:
        return ex_tbl

    ex_img = _latest_producer_of_output_before(
        db, "%blase.Extract.read_images%", data_hash=manifest_hash, ts_before=ts_before
    ) or _latest_with_input_before(
        db,
        "%blase.Extract.read_images%",
        role="manifest",
        data_hash=manifest_hash,
        ts_before=ts_before,
    )
    return ex_img


def _latest_extract_for_source(db, src_hash: str, ts_before: int) -> Optional[str]:
    """CSV lineage fallback when no manifest path exists."""
    return _latest_with_source_before(
        db, "%blase.Extract.read_csv%", src_hash, ts_before
    )


def _latest_transform_for_manifest_or_source(
    db, tip_ts: int, manifest_hash: Optional[str], src_hash: Optional[str]
) -> Optional[str]:
    """
    Find the latest Transform.apply_function that is causally before tip_ts,
    anchored by manifest when available, else by source.
    """
    if manifest_hash:
        tr = _latest_with_input_before(
            db,
            "%blase.Transform.apply_function%",
            role="manifest",
            data_hash=manifest_hash,
            ts_before=tip_ts,
        )
        if tr:
            return tr

    if src_hash:
        return _latest_with_source_before(
            db, "%blase.Transform.apply_function%", src_hash, tip_ts
        )

    return None


# ---------------------------
# Main planner
# ---------------------------


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
    db = _db(run_path)
    tip = _step_row(db, tip_step_hash)
    fqn = tip["function_fqn"]
    ts = tip["ts_start"]

    # Back-compat CSV source (may be absent on image/table flows)
    src = _source_hash_for_step(db, tip_step_hash)
    plan: List[str] = []

    # ---- Load.save_to_csv ----
    if fqn.endswith("Load.save_to_csv"):
        if src:
            tr = _latest_with_source_before(
                db, "%blase.Transform.apply_function%", src, ts
            )
            if tr:
                tr_ts = _step_row(db, tr)["ts_start"]
                man = _input_hash_for_role(db, tr, "manifest")

                if man:
                    _append_if(
                        plan, _prefer_parquet_then_images_for_manifest(db, man, tr_ts)
                    )
                else:
                    _append_if(plan, _latest_extract_for_source(db, src, tr_ts))

                plan.append(tr)

        plan.append(tip_step_hash)
        return plan

    # ---- Load.save_images_to_parquet ----
    if fqn.endswith("Load.save_images_to_parquet"):
        man_tip = _input_hash_for_role(db, tip_step_hash, "manifest")
        tr = _latest_transform_for_manifest_or_source(db, ts, man_tip, src)

        if tr:
            tr_ts = _step_row(db, tr)["ts_start"]
            man_tr = _input_hash_for_role(db, tr, "manifest")

            if man_tr:
                _append_if(
                    plan, _prefer_parquet_then_images_for_manifest(db, man_tr, tr_ts)
                )
            else:
                src_tr = _source_hash_for_step(db, tr)
                _append_if(
                    plan,
                    _latest_extract_for_source(db, src_tr, tr_ts) if src_tr else None,
                )

            plan.append(tr)
        else:
            # pure CSV/source fallback (very old runs)
            if src:
                _append_if(plan, _latest_extract_for_source(db, src, ts))

        plan.append(tip_step_hash)
        return plan

    # ---- Transform.apply_function ----
    if fqn.endswith("Transform.apply_function"):
        man = _input_hash_for_role(db, tip_step_hash, "manifest")
        if man:
            _append_if(plan, _prefer_parquet_then_images_for_manifest(db, man, ts))
        elif src:
            _append_if(plan, _latest_extract_for_source(db, src, ts))

        plan.append(tip_step_hash)
        return plan

    # ---- Extracts ----
    if (
        fqn.endswith("Extract.read_csv")
        or fqn.endswith("Extract.read_images")
        or fqn.endswith("Extract.read_parquet")
    ):
        return [tip_step_hash]

    # ---- Fallback ----
    return [tip_step_hash]
