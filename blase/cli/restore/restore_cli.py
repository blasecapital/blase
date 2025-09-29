import json
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any

from blase.cli.restore.dispatcher import Policy, run_data, run_step
from blase.cli.restore.pretty import _print_rows, _print_step
from blase.restoring import cas
from blase.utils.config import RESTORE_CONFLICT as DEFAULT_CONFLICT


# --------- run discovery helpers ---------


def _runs_root(cwd: Optional[Path] = None) -> Path:
    cwd = cwd or Path.cwd()
    for p in (cwd, cwd.parent):
        r = p / "runs"
        if r.exists():
            return r
    r = cwd / "runs"
    r.mkdir(parents=True, exist_ok=True)
    return r


def _active_run_id(run_root: Path) -> Optional[str]:
    cfg = run_root / "active_run.blase"
    if not cfg.exists():
        return None
    try:
        return (json.loads(cfg.read_text()) or {}).get("run_id")
    except Exception:
        return None


def _resolve_run_path(run_arg: Optional[str]) -> Path:
    """
    Resolve a run argument to a filesystem path.

    Parameters
    ----------
    run_arg : str or None
        Either a run ID, a direct path to a run directory, or ``None``.
        If ``None``, the currently active run is used.

    Returns
    -------
    Path
        Filesystem path to the resolved run directory.

    Raises
    ------
    SystemExit
        If the run argument does not correspond to an existing run
        or if no active run is found when `run_arg` is None.
    """
    rr = _runs_root()
    if run_arg:
        p = Path(run_arg)
        if p.exists():
            return p
        rp = rr / run_arg
        if rp.exists():
            return rp
        raise SystemExit(f"[restore] run not found: {run_arg}")
    # default to active
    rid = _active_run_id(rr)
    if not rid:
        raise SystemExit("[restore] no active run; pass --run <ID|PATH>")
    return rr / rid


def _open_db(run_path: Path) -> sqlite3.Connection:
    db = run_path / "nodes" / "nodes.db"
    if not db.exists():
        raise SystemExit(f"[restore] nodes db missing: {db}")
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    return con


# --------- list/show/plan ---------


def cmd_list(args):
    run_path = _resolve_run_path(args.run)
    con = _open_db(run_path)
    rows = con.execute(
        """
        SELECT step_hash, function_fqn, status, ts_start, ts_end
        FROM steps
        ORDER BY ts_start DESC
        LIMIT ?
    """,
        (args.limit or 20,),
    ).fetchall()
    if args.like_fqn:
        rows = [r for r in rows if args.like_fqn in r["function_fqn"]]
    if not rows:
        print("[restore] no steps found")
        return
    _print_rows(
        [
            [r["step_hash"], r["function_fqn"], r["status"], r["ts_start"], r["ts_end"]]
            for r in rows
        ]
    )


def cmd_show(args):
    run_path = _resolve_run_path(args.run)
    con = _open_db(run_path)
    _print_step(con, args.step)


def _data_present(con, run_path: Path, data_hash: str) -> Dict[str, Any]:
    """Check CAS/materialization availability for a data hash."""
    # infer kind
    row = con.execute(
        "SELECT kind FROM data WHERE data_hash=?", (data_hash,)
    ).fetchone()
    kind = row["kind"] if row else "data"
    cas_path = cas.path_for(run_path, kind=kind, data_hash=data_hash)
    have_cas = cas_path.exists()
    mat = con.execute(
        """
        SELECT path, ts FROM materializations WHERE data_hash=? ORDER BY ts DESC LIMIT 1
    """,
        (data_hash,),
    ).fetchone()
    return {
        "kind": kind,
        "cas": have_cas,
        "cas_path": str(cas_path),
        "materialized": bool(mat),
        "materialized_path": (mat["path"] if mat else None),
    }


def cmd_plan(args):
    """
    Display a restore plan for a recorded step.

    This command inspects the run database and prints the execution plan for
    a given step hash. The plan shows which inputs are already available
    (in CAS or materialized locally), which require replay from a producer
    step, and which are missing entirely. It also enumerates the outputs of
    the step and their availability status.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI options. Expected attributes:

        run : str or None
            Run identifier or path. If omitted, the active run is resolved.
        step : str
            Step hash to inspect.

    Returns
    -------
    None
        The plan is printed to stdout. No explicit return value.

    Raises
    ------
    SystemExit
        If the step is not found in the run database, or if the run cannot be
        resolved/opened.

    Notes
    -----
    **Inputs**

    * If an input data hash is present in CAS, it is marked with a ✓ and
      labeled ``CAS``.
    * If an input data hash is materialized locally, it is marked with a ✓
      and labeled ``MAT``.
    * If the input is not present but has a known producer step, it is marked
      with a ↻ indicating that replay is required.
    * If the input is missing entirely (no CAS, no materialization, and no
      known producer), it is marked with ✗.

    **Outputs**

    * Outputs are listed under an ``outputs:`` block.
    * Each output data hash is annotated as one of:
      - ``CAS`` if stored in content-addressable storage,
      - ``MAT`` if materialized locally,
      - ``MISSING`` if neither.

    Examples
    --------
    Print the plan for a step::

        blase restore plan --step deadbeef1234

    Example output::

        Plan for step deadbeef1234 (blase.Transform.apply_fn):
          ✓ input data      abc123...  (CAS)
          ↻ need replay for input labels   def456...  (producer step cafebabe...)
          ✗ missing input metadata  7890ab...  (no CAS/materialization; unknown producer)
          outputs:
            - out.csv   13579df...  (MAT)
            - log.json  24680ac...  (MISSING)
    """
    run_path = _resolve_run_path(args.run)
    con = _open_db(run_path)
    s = con.execute(
        "SELECT function_fqn FROM steps WHERE step_hash=?", (args.step,)
    ).fetchone()
    if not s:
        raise SystemExit(f"[restore] step not found: {args.step}")
    print(f"Plan for step {args.step} ({s['function_fqn']}):")
    ins = con.execute(
        "SELECT data_hash, role FROM step_inputs WHERE step_hash=?", (args.step,)
    ).fetchall()
    for r in ins:
        st = _data_present(con, run_path, r["data_hash"])
        if st["cas"] or st["materialized"]:
            src = "CAS" if st["cas"] else "MAT"
            print(f"  ✓ input {r['role']:<8s} {r['data_hash']}  ({src})")
        else:
            # try to find producer step
            prod = con.execute(
                "SELECT step_hash FROM step_outputs WHERE data_hash=?",
                (r["data_hash"],),
            ).fetchone()
            if prod:
                print(
                    f"  ↻ need replay for input {r['role']:<8s} {r['data_hash']}  (producer step {prod['step_hash']})"
                )
            else:
                print(
                    f"  ✗ missing input {r['role']:<8s} {r['data_hash']}  (no CAS/materialization; unknown producer)"
                )
    outs = con.execute(
        "SELECT data_hash, name FROM step_outputs WHERE step_hash=?", (args.step,)
    ).fetchall()
    if outs:
        print("  outputs:")
        for r in outs:
            st = _data_present(con, run_path, r["data_hash"])
            src = "CAS" if st["cas"] else ("MAT" if st["materialized"] else "MISSING")
            print(f"    - {r['name']:<8s} {r['data_hash']}  ({src})")


# --------- run (verify/materialize/replay) ---------
def cmd_run(args):
    """
    Run the `blase restore run` CLI for a data hash or a step hash.

    This dispatches to materialize, verify, or replay workflows using the
    recorded run metadata and CAS, wiring upstream generators for sink steps.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed options.

        General
            run : str or None
                Run ID or path. If None, the active run is used.
            mode : {"materialize", "verify", "replay"}
                Execution mode.
            to : str or None
                Destination path (file or directory) for outputs.
            on_conflict : {"fail", "rename", "overwrite"} or None
                Conflict policy for writing outputs. If None, defaults apply.
            keep_intermediates : bool
                If True, do not delete intermediate files after replay.
            limit_batches : int or None
                Max batches to consume when verifying streams.
            backend : {"pandas", "polars"} or None
                Optional backend override for sink replays.

        Target (mutually exclusive)
            data : str or None
                Target data hash for data-centric workflows.
            step : str or None
                Target step hash for step-centric workflows.

    Returns
    -------
    int
        Exit code:
        - 0 on success,
        - 2 when a step has no recorded outputs in materialize mode,
        - 3 when fast materialization is not possible and replay is required.

    Raises
    ------
    SystemExit
        On usage/state errors, including:
        - No active run and no `--run` provided.
        - Missing or unreadable `nodes.db`.
        - Neither or both of `--data` and `--step`.
        - No producer step for a data hash that requires replay.
        - Final output missing or content-hash mismatch.
        - Unknown `--mode`.

    Notes
    -----
    Data-centric (`--data`):
      1. Try fast-path materialization via CAS; verify bytes.
      2. If unavailable, replay the producer plan, verify final bytes,
         and clean intermediates unless `--keep_intermediates`.

    Step-centric (`--step`):
      - materialize: bring back recorded outputs only (no replay).
      - verify: stream results without writing; wires an upstream generator
        for sink steps.
      - replay: wire upstream and write sink outputs to `--to` or defaults.

    Examples
    --------
    Materialize a data hash to a file::

        blase restore run --data 0123abcd... --mode materialize --to out.csv

    Replay a sink step to a path::

        blase restore run --step deadbeef... --mode replay --to restored.csv

    Verify a transform step with a batch cap::

        blase restore run --step cafe... --mode verify --limit-batches 5
    """
    run_path = _resolve_run_path(args.run)
    if not getattr(args, "on_conflict", None):
        args.on_conflict = DEFAULT_CONFLICT

    has_step = bool(getattr(args, "step", None))
    has_data = bool(getattr(args, "data", None))
    mode = getattr(args, "mode", None)
    if mode is None:
        if has_step:
            mode = "replay"
        elif has_data:
            mode = "materialize"
        else:
            raise SystemExit(
                "restore: either --step or --data is required (mutually exclusive)."
            )

    pol = Policy(
        on_conflict=args.on_conflict,
        keep_intermediates=getattr(args, "keep_intermediates", False),
        limit_batches=getattr(args, "limit_batches", None),
        backend=getattr(args, "backend", None),
        mode=mode,
        to=getattr(args, "to", None),
    )

    if has_step:
        return run_step(run_path, args.step, mode, pol.to, pol)
    return run_data(run_path, args.data, mode, pol.to, pol)
