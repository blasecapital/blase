from __future__ import annotations
import argparse, json, sqlite3, sys, os
from pathlib import Path
from typing import Optional, Dict, Any

from blase.restore import step as restore_step
from blase.restoring import materialize, store, cas, bindings, planner, code
from blase.utils.config import RESTORE_CONFLICT as DEFAULT_CONFLICT, RESTORE_DEFAULT_DIR

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
    rr = _runs_root()
    if run_arg:
        p = Path(run_arg)
        if p.exists():  # treat as path
            return p
        # treat as run id under runs/
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

# --------- pretty helpers ---------

def _print_rows(rows):
    for r in rows:
        print(" | ".join(str(v) for v in r))

def _print_step(con, step_hash: str):
    s = con.execute("SELECT * FROM steps WHERE step_hash=?", (step_hash,)).fetchone()
    if not s:
        print(f"[restore] step not found: {step_hash}")
        return
    print(f"step: {s['step_hash']}  fqn={s['function_fqn']}  status={s['status']}")
    print(f"  run_id={s['run_id']}  ts_start={s['ts_start']}  ts_end={s['ts_end']}")
    print(f"  params={s['params_json']}")
    ins = con.execute("SELECT data_hash, role, arg_name FROM step_inputs WHERE step_hash=?", (step_hash,)).fetchall()
    print("  inputs:")
    for r in ins:
        print(f"    - {r['role']:8s} {r['data_hash']} arg={r['arg_name']}")
    outs = con.execute("SELECT data_hash, name FROM step_outputs WHERE step_hash=?", (step_hash,)).fetchall()
    print("  outputs:")
    for r in outs:
        print(f"    - {r['name']:8s} {r['data_hash']}")

def _latest_completed_step(con, like_fqn: str) -> Optional[str]:
    row = con.execute("""
        SELECT step_hash
        FROM steps
        WHERE function_fqn LIKE ?
          AND status IN ('completed','complete','done','success','ok')
        ORDER BY ts_start DESC LIMIT 1
    """, (like_fqn,)).fetchone()
    return row["step_hash"] if row else None

# --------- list/show/plan ---------

def cmd_list(args):
    run_path = _resolve_run_path(args.run)
    con = _open_db(run_path)
    rows = con.execute("""
        SELECT step_hash, function_fqn, status, ts_start, ts_end
        FROM steps
        ORDER BY ts_start DESC
        LIMIT ?
    """, (args.limit or 20,)).fetchall()
    if args.like_fqn:
        rows = [r for r in rows if args.like_fqn in r["function_fqn"]]
    if not rows:
        print("[restore] no steps found")
        return
    _print_rows([[r["step_hash"], r["function_fqn"], r["status"], r["ts_start"], r["ts_end"]] for r in rows])

def cmd_show(args):
    run_path = _resolve_run_path(args.run)
    con = _open_db(run_path)
    _print_step(con, args.step)

def _data_present(con, run_path: Path, data_hash: str) -> Dict[str, Any]:
    """Check CAS/materialization availability for a data hash."""
    # infer kind
    row = con.execute("SELECT kind FROM data WHERE data_hash=?", (data_hash,)).fetchone()
    kind = row["kind"] if row else "data"
    cas_path = cas.path_for(run_path, kind=kind, data_hash=data_hash)
    have_cas = cas_path.exists()
    mat = con.execute("""
        SELECT path, ts FROM materializations WHERE data_hash=? ORDER BY ts DESC LIMIT 1
    """, (data_hash,)).fetchone()
    return {
        "kind": kind,
        "cas": have_cas,
        "cas_path": str(cas_path),
        "materialized": bool(mat),
        "materialized_path": (mat["path"] if mat else None),
    }

def cmd_plan(args):
    run_path = _resolve_run_path(args.run)
    con = _open_db(run_path)
    s = con.execute("SELECT function_fqn FROM steps WHERE step_hash=?", (args.step,)).fetchone()
    if not s:
        raise SystemExit(f"[restore] step not found: {args.step}")
    print(f"Plan for step {args.step} ({s['function_fqn']}):")
    ins = con.execute("SELECT data_hash, role FROM step_inputs WHERE step_hash=?", (args.step,)).fetchall()
    for r in ins:
        st = _data_present(con, run_path, r["data_hash"])
        if st["cas"] or st["materialized"]:
            src = "CAS" if st["cas"] else "MAT"
            print(f"  ✓ input {r['role']:<8s} {r['data_hash']}  ({src})")
        else:
            # try to find producer step
            prod = con.execute("SELECT step_hash FROM step_outputs WHERE data_hash=?", (r["data_hash"],)).fetchone()
            if prod:
                print(f"  ↻ need replay for input {r['role']:<8s} {r['data_hash']}  (producer step {prod['step_hash']})")
            else:
                print(f"  ✗ missing input {r['role']:<8s} {r['data_hash']}  (no CAS/materialization; unknown producer)")
    outs = con.execute("SELECT data_hash, name FROM step_outputs WHERE step_hash=?", (args.step,)).fetchall()
    if outs:
        print("  outputs:")
        for r in outs:
            st = _data_present(con, run_path, r["data_hash"])
            src = "CAS" if st["cas"] else ("MAT" if st["materialized"] else "MISSING")
            print(f"    - {r['name']:<8s} {r['data_hash']}  ({src})")

# --------- run (verify/materialize/replay) ---------

def _resolve_to_path(to: Optional[str], default_name: str) -> Path:
    if to:
        p = Path(to)
        if p.is_dir():
            p = p / default_name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p.resolve()
    # default: config default dir + name
    RESTORE_DEFAULT_DIR.mkdir(parents=True, exist_ok=True)
    return (RESTORE_DEFAULT_DIR / default_name).resolve()

def _exec_replay(run_path: Path, tip_step_hash: str, to_path: Optional[str], backend_override: Optional[str]):
    plan = planner.plan_for_step(run_path, tip_step_hash)
    upstream = None

    for sh in plan:
        st = store.load_step(run_path, sh)
        fqn = st["function_fqn"]
        handler = bindings.RESTORE_HANDLERS.get(fqn)

        # Load callable snapshot when needed (for Transform)
        fn = None
        if fqn == "blase.Transform.apply_function":
            ins = store.load_step_inputs(run_path, sh)
            from blase.restoring import cas
            fn = code.load_callable_from_blob(cas.path_for(run_path, "code", store.pick_code_hash(ins)))

        if fqn == "blase.Extract.read_csv":
            upstream = bindings.run_read_csv_restore(run_path=run_path, params=st["params"], realized={}, transform_fn=None)

        elif fqn == "blase.Transform.apply_function":
            upstream = bindings.run_apply_function_restore(run_path=run_path, params=st["params"], realized={}, transform_fn=fn, upstream_gen=upstream)

        elif fqn == "blase.Load.save_to_csv":
            out = bindings.run_save_to_csv_replay(
                run_path=run_path, params=st["params"], realized={}, transform_fn=None,
                upstream_gen=upstream, target_override=to_path, backend_override=backend_override
            )
            print(f"Replayed: {out}")
            upstream = None

        else:
            # Fallback for simple, non-stream steps
            from blase import restore
            restore.step(run_path, sh)

def _normalize_plan_nodes(run_path: Path, plan):
    """
    Accept nodes as strings (step_hash), tuples (step_hash, ...),
    or dicts; return a list of dicts with at least:
      {"step_hash": <hash>, "function_fqn": <fqn>}
    """
    out = []
    for node in plan:
        if isinstance(node, dict):
            if "step_hash" in node and "function_fqn" in node:
                out.append({"step_hash": node["step_hash"], "function_fqn": node["function_fqn"]})
            elif "hash" in node:
                st = store.load_step(run_path, node["hash"])
                out.append({"step_hash": node["hash"], "function_fqn": st["function_fqn"]})
            else:
                # Last resort: try to find something that looks like a hash
                h = node.get("id") or node.get("step") or node.get("node") or node.get("sha")
                if not h:
                    continue
                st = store.load_step(run_path, h)
                out.append({"step_hash": h, "function_fqn": st["function_fqn"]})
        elif isinstance(node, (list, tuple)) and node:
            h = node[0]
            st = store.load_step(run_path, h)
            out.append({"step_hash": h, "function_fqn": st["function_fqn"]})
        elif isinstance(node, str):
            h = node
            st = store.load_step(run_path, h)
            out.append({"step_hash": h, "function_fqn": st["function_fqn"]})
    return out

def cmd_run(args):
    run_path = _resolve_run_path(args.run)
    step_hash = args.step

    # Load step metadata up front (so we can decide sink vs non-sink)
    st = store.load_step(run_path, step_hash)   # {'function_fqn','params','status'}
    fqn = st["function_fqn"]

    # Wrapper for "verify" and "replay"
    limit_batches = getattr(args, "limit_baches", None)

    # ---- MATERIALIZE: just bring recorded outputs back (no replay) ----
    if args.mode == "materialize":
        outs = store.load_step_outputs(run_path, step_hash)
        if not outs:
            print("No outputs recorded for this step.")
            return 2

        # Take the first output (extend to loop if you shard)
        out_hash = outs[0]["data_hash"]
        kind = store.get_data_kind(run_path, out_hash) or "data"

        # Resolve target path pieces
        to_path = Path(args.to).resolve()
        to_dir = to_path.parent
        target_name = to_path.name

        try:
            path = materialize.ensure_local(
                run_path, out_hash, kind=kind,
                to_dir=to_dir, target_name=target_name,
                on_conflict=args.on_conflict
            )
            print(f"Materialized: {path}")
            return 0
        except materialize.NeedReplay:
            print("Output not available locally. Use:")
            print(f"  blase restore plan --step {step_hash}")
            print(f"  blase restore run  --step {step_hash} --mode replay")
            return 3

    # Helper: pick nearest upstream compute step (prefer Transform, else Extract)
    def _upstream_for_sink(target_step_hash: str):
        raw_plan = planner.plan_for_step(run_path, target_step_hash)
        nodes = _normalize_plan_nodes(run_path, raw_plan)

        # find index of the sink in plan
        try:
            idx = next(i for i, n in enumerate(nodes) if n["step_hash"] == target_step_hash)
        except StopIteration:
            raise SystemExit("restore: target step not found in plan")

        # scan backward to find nearest Transform
        upstream = None
        for j in range(idx - 1, -1, -1):
            if nodes[j]["function_fqn"].endswith("Transform.apply_function"):
                upstream = nodes[j]
                break
        # fallback: nearest Extract
        if upstream is None:
            for j in range(idx - 1, -1, -1):
                if nodes[j]["function_fqn"].endswith("Extract.read_csv"):
                    upstream = nodes[j]
                    break
        if upstream is None:
            raise SystemExit("No upstream compute step found to feed Load.save_to_csv replay.")
        return restore_step(run_path, upstream["step_hash"], kind="csv")

    # ---- VERIFY: stream results without writing ----
    if args.mode == "verify":
        if fqn == "blase.Load.save_to_csv":
            # For a sink, verify by consuming its upstream (don’t write)
            gen = _upstream_for_sink(step_hash)
        else:
            # For producers/transforms, restore directly
            gen = restore_step(run_path, step_hash, kind="csv")

        total = 0
        for i, (b, last) in enumerate(gen, 1):
            try:
                n = len(b)
            except Exception:
                n = "?"
            print(f"[verify] batch {i}: {n} rows  last={last}")
            if isinstance(n, int):
                total += n
            if limit_batches and i >= limit_batches:
                break
        print(f"[verify] total rows (best-effort): {total}")
        return 0

    # ---- REPLAY: for sinks, wire upstream; for non-sinks, just restore ----
    if args.mode == "replay":
        backend_override = getattr(args, "backend", None)
        target_override = getattr(args, "to", None)

        if fqn == "blase.Load.save_to_csv":
            upstream_gen = _upstream_for_sink(step_hash)
            handler = bindings.RESTORE_HANDLERS[fqn]
            out_path = handler(
                run_path=run_path,
                params=st["params"],
                realized={},                   # not needed for replay write
                upstream_gen=upstream_gen,
                target_override=target_override,       # optional --to
                backend_override=backend_override  # optional --backend
            )
            print(out_path)
            return 0
        else:
            # Non-sinks just stream the restored batches (caller can pipe elsewhere)
            gen = restore_step(run_path, step_hash, kind="csv")
            for i, (b, last) in enumerate(gen, 1):
                print(f"batch {i}: {len(b)} rows, last={last}")
                if limit_batches and i >= limit_batches:
                    break
            return 0

    raise SystemExit(f"[restore] unknown mode: {args.mode}")