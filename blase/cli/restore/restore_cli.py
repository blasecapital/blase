from __future__ import annotations
import json
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Set, List
import tempfile

from blase.restore import step as restore_step
from blase.restoring import materialize, store, cas, bindings, planner, code
from blase.restoring.materialize import NeedReplay
from blase.utils.config import RESTORE_CONFLICT as DEFAULT_CONFLICT, RESTORE_DEFAULT_DIR
from blase.utils.hashing import Hash

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


def _assert_path_matches_hash(path: Path, expected_hash: str, what: str) -> None:
    got = Hash().hash_file(path)
    if got != expected_hash:
        raise SystemExit(
            f"restore: {what} hash mismatch: got {got}, expected {expected_hash}"
        )


def _resolve_source_no_copy(
    run_path: Path,
    source_hash: str,
    *,
    seen_steps: Optional[Set[str]] = None,
    created_paths: Optional[List[Path]] = None,
) -> Tuple[Path, bool]:
    """
    Return a path to the recorded source without copying when possible.
    If the original source_path exists and matches the recorded hash, use it.
    Else replay/materialize (this will copy into /restored); we return that path and mark created=True.
    """
    # Try original source_of_truth path first (no copy)
    src = store.load_data_source_path(run_path, source_hash)
    if src:
        p = Path(src)
        if p.exists() and Hash().hash_file(p) == source_hash:
            return p, False  # not created

    # Fallback: ensure locally (may trigger replay of the producer; will copy/link into /restored)
    p, created = _ensure_data_local_or_replay(
        run_path,
        source_hash,
        store.get_data_kind(run_path, source_hash) or "csv",
        seen_steps=seen_steps,
        created_paths=created_paths,
    )
    # Track for cleanup if we had to copy
    if created and created_paths is not None:
        created_paths.append(p)
    return p, created


def _resolve_seed_no_copy_or_ephemeral(
    run_path: Path,
    seed_hash: str,
    *,
    seen_steps: Optional[Set[str]] = None,
    created_paths: Optional[List[Path]] = None,
) -> Tuple[Path, bool]:
    """
    Return a path to the recorded seed (run-1 output) without persisting it in /restored.
    - If a valid existing path already exists (materialization or original), use it (no copy).
    - Otherwise, replay the seed's producer into a *temporary* file (not /restored) and
      do NOT record a materialization. Mark created=True and remember for cleanup.
    """
    # 1) Prefer an existing materialized path (no copy)
    p = store.get_materialized_path(run_path, seed_hash)
    if p and p.exists() and Hash().hash_file(p) == seed_hash:
        return p, False

    # 2) Try recorded source_path (the original file), if any
    p2 = store.get_recorded_source_path(run_path, seed_hash)
    if p2 and p2.exists() and Hash().hash_file(p2) == seed_hash:
        return p2, False

    # 3) No local copy → replay the seed into a temporary file (EPHEMERAL)
    prod = store.producer_step_for_data(run_path, seed_hash)
    if not prod:
        raise SystemExit(
            f"restore: seed {seed_hash} missing and no producer step recorded"
        )

    tmpdir = Path(tempfile.mkdtemp(prefix="blase-seed-"))
    tmpfile = tmpdir / f"{seed_hash}.csv"

    # Execute only the seed’s plan, writing to tmpfile; prevent materialization records.
    _exec_plan_for_step(
        run_path,
        prod,
        to_path=str(tmpfile),
        backend_override=None,
        seen_steps=seen_steps,
        created_paths=None,
        ephemeral_only=True,
    )

    if not tmpfile.exists() or Hash().hash_file(tmpfile) != seed_hash:
        raise SystemExit("restore: ephemeral seed replay produced wrong bytes")

    if created_paths is not None:
        created_paths.append(tmpfile)

    return tmpfile, True


def _upstream_gen_for_sink(run_path: Path, target_step_hash: str):
    """
    Build an upstream (batch,last) generator for a Load.save_to_csv step by scanning
    its plan backward to the nearest Transform.apply_function (preferred) or Extract.read_csv.
    """
    # figure out which sink we’re wiring
    tip = store.load_step(run_path, target_step_hash)
    tip_fqn = tip["function_fqn"]

    # what counts as a streaming producer
    def _is_extract_images(fqn: str) -> bool:
        return fqn.endswith("Extract.read_images")

    def _is_extract_csv(fqn: str) -> bool:
        return fqn.endswith("Extract.read_csv")

    def _is_transform(fqn: str) -> bool:
        return fqn.endswith("Transform.apply_function")

    # ask the planner for a minimal chain and normalize it
    raw_plan = planner.plan_for_step(run_path, target_step_hash) or []
    nodes = _normalize_plan_nodes(run_path, raw_plan)

    try:
        idx = next(i for i, n in enumerate(nodes) if n["step_hash"] == target_step_hash)
    except StopIteration:
        raise SystemExit("restore: target sink step not found in plan")

    # scan backwards to find a producer
    upstream_node = None

    # 1) nearest Transform.apply_function
    for j in range(idx - 1, -1, -1):
        if _is_transform(nodes[j]["function_fqn"]):
            upstream_node = nodes[j]
            break

    # 2) if none, pick the right Extract fallback by sink type
    if upstream_node is None:
        if tip_fqn.endswith("Load.save_images_to_parquet"):
            for j in range(idx - 1, -1, -1):
                if _is_extract_images(nodes[j]["function_fqn"]):
                    upstream_node = nodes[j]
                    break
        else:
            # default/legacy CSV path
            for j in range(idx - 1, -1, -1):
                if _is_extract_csv(nodes[j]["function_fqn"]):
                    upstream_node = nodes[j]
                    break

    if upstream_node is None:
        # helpful debug: show what we actually saw
        dbg = " → ".join(f"{n['function_fqn']}[{n['step_hash'][:8]}]" for n in nodes)
        raise SystemExit(
            f"No upstream compute step found for sink replay.\n[debug] plan: {dbg}"
        )

    # delegate to the generic streaming builder for whatever node we found
    return _build_stream_for_step(run_path, upstream_node["step_hash"])


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
    ins = con.execute(
        "SELECT data_hash, role, arg_name FROM step_inputs WHERE step_hash=?",
        (step_hash,),
    ).fetchall()
    print("  inputs:")
    for r in ins:
        print(f"    - {r['role']:8s} {r['data_hash']} arg={r['arg_name']}")
    outs = con.execute(
        "SELECT data_hash, name FROM step_outputs WHERE step_hash=?", (step_hash,)
    ).fetchall()
    print("  outputs:")
    for r in outs:
        print(f"    - {r['name']:8s} {r['data_hash']}")


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


def _exec_plan_for_step(
    run_path: Path,
    tip_step_hash: str,
    to_path: Optional[str],
    backend_override: Optional[str],
    *,
    seen_steps: Optional[Set[str]] = None,
    created_paths: Optional[List[Path]] = None,
    ephemeral_only: bool = False,
) -> Dict[str, Path]:
    """
    Replay a DAG of recorded steps forward from any upstream roots to `tip_step_hash`.

    This is the core engine used by `blase restore run --data <hash>` and similar
    commands. It walks the stored execution graph, resolves upstream inputs,
    and replays each step’s recorded function in dependency order, producing
    exactly the same materialized files.

    Parameters
    ----------
    run_path : Path
        Root directory of the recorded run (contains `nodes/` and `cas/` stores).
    tip_step_hash : str
        Hash of the final step to replay. The function backtracks from this hash
        to discover and execute all required upstream steps.
    to_path : str or None
        Optional override path for the final sink’s output. If supplied,
        the final output (CSV file or Parquet directory) is written there
        instead of the recorded default.
    backend_override : str or None
        Optional override for downstream I/O backends (e.g., a different
        Parquet writer). Passed to sink replay functions if supported.
    seen_steps : set of str, optional
        Set of step hashes already executed. Steps in this set are skipped,
        allowing incremental or multi-branch replays without duplication.
    created_paths : list of Path, optional
        Collects every on-disk path created during replay for caller inspection
        or later cleanup.
    ephemeral_only : bool, default=False
        If True, write outputs but do not record them back into the CAS
        materialization index. Useful for temporary previews or dry runs.

    Returns
    -------
    dict
        Mapping of `{produced_data_hash: Path(actual_output_file)}` for every
        sink materialized during replay (e.g., CSV files, Parquet shards).

    Notes
    -----
    - The replay algorithm performs a topological walk from each required
      upstream step to the requested tip. Each known function type is handled
      specifically:
        * `blase.Extract.read_csv` and `blase.Extract.read_images` recreate
          upstream batch generators.
        * `blase.Transform.apply_function` re-applies stored user functions.
        * `blase.Load.save_to_csv` and `blase.Load.save_images_to_parquet`
          write final sink files and validate their hashes.
    - Steps that have already been replayed (present in `seen_steps`) are
      skipped. Outputs already present with matching CAS hashes are not
      duplicated.
    - If `ephemeral_only` is True, on-disk artifacts are created but not
      registered as permanent CAS data, so subsequent runs will not treat
      them as cached outputs.
    - Raises `SystemExit` if required upstream anchors (e.g., manifest or
      source data) are missing or inconsistent with recorded metadata.
    """
    plan = planner.plan_for_step(run_path, tip_step_hash) or []
    upstream = None
    produced: Dict[str, Path] = {}
    produced_hashes: Set[str] = set()
    seen_steps = seen_steps or set()
    created_paths = created_paths if created_paths is not None else []

    for sh in plan:
        if sh in seen_steps:
            continue
        seen_steps.add(sh)

        st = store.load_step(run_path, sh)
        fqn = st["function_fqn"]

        if fqn == "blase.Extract.read_csv":
            ins = store.load_step_inputs(run_path, sh)
            src_hash = next(
                (i["data_hash"] for i in ins if i["role"] == "source"), None
            )
            realized = {}
            if src_hash:
                realized["__expected_source_hash__"] = src_hash
            upstream = bindings.run_read_csv_restore(
                run_path=run_path,
                params=st["params"],
                realized=realized,
                transform_fn=None,
            )
            continue

        if fqn == "blase.Extract.read_images":
            ins = store.load_step_inputs(run_path, sh)  # code/env only
            outs = store.load_step_outputs(run_path, sh)  # manifest + batch_desc_*

            # recorded artifacts come from outputs
            manifest_hash = next(
                (o["data_hash"] for o in outs if o["name"] == "manifest"), None
            )

            def _ord(o):
                n = o["name"]
                try:
                    return int(n.rsplit("_", 1)[-1])
                except Exception:
                    return 0

            batch_descs = [
                o["data_hash"]
                for o in sorted(outs, key=_ord)
                if o["name"].startswith("batch_desc_")
            ]

            realized = {}
            if "directory" in st["params"]:
                realized["source"] = st["params"]["directory"]
            if manifest_hash:
                realized["manifest"] = manifest_hash
            if batch_descs:
                realized["batch_descs"] = batch_descs  # your reader should accept this

            upstream = bindings.run_read_images_restore(
                run_path=run_path,
                params=st["params"],
                realized=realized,
                transform_fn=None,
            )
            continue

        if fqn == "blase.Transform.apply_function":
            ins = store.load_step_inputs(run_path, sh)
            fn = code.load_callable_from_blob(
                cas.path_for(run_path, "code", store.pick_code_hash(ins))
            )

            # If an upstream generator already exists, consume it.
            if upstream is not None:
                upstream = bindings.run_apply_function_restore(
                    run_path=run_path,
                    params=st["params"],
                    realized={"source_gen": upstream},
                    transform_fn=fn,
                )
                continue

            # Otherwise, anchor by manifest/source and build the upstream now.
            def _pick_anchor(roles=("source", "manifest", "dataset", "manifest_root")):
                for r in roles:
                    h = next((i["data_hash"] for i in ins if i["role"] == r), None)
                    if h:
                        return r, h
                return None, None

            role, anchor_hash = _pick_anchor()
            if not anchor_hash:
                raise SystemExit(
                    "replay: Transform.apply_function missing anchor (source|manifest|dataset)"
                )

            if role == "source":
                src_path, _ = _resolve_source_no_copy(
                    run_path,
                    anchor_hash,
                    seen_steps=seen_steps,
                    created_paths=created_paths,
                )
                upstream = bindings.run_apply_function_restore(
                    run_path=run_path,
                    params=st["params"],
                    realized={"source": src_path},
                    transform_fn=fn,
                )
            else:
                # treat anchor as the manifest
                man_hash = anchor_hash
                ts_before = planner._step_row(planner._db(run_path), sh)["ts_start"]
                ri_params = store.read_images_params_for_manifest(
                    run_path, man_hash, ts_before=ts_before
                )
                if not ri_params:
                    raise SystemExit(
                        "replay: cannot find read_images params for manifest"
                    )

                ins_tr = store.load_step_inputs(run_path, sh)
                batch_descs = [
                    i["data_hash"]
                    for i in ins_tr
                    if i["role"] in ("batch_desc", "batch")
                ]
                gen = bindings.run_read_images_restore(
                    run_path=run_path,
                    params=ri_params,
                    realized={
                        "source": ri_params.get("directory"),
                        "manifest": man_hash,
                        "batch": batch_descs,
                    },  # your reader should normalize 'batch'/'batch_descs'
                )
                upstream = bindings.run_apply_function_restore(
                    run_path=run_path,
                    params=st["params"],
                    realized={"source_gen": gen},
                    transform_fn=fn,
                )
            continue

        if fqn == "blase.Load.save_to_csv":
            ins = store.load_step_inputs(run_path, sh)
            outs = store.load_step_outputs(run_path, sh)
            expected_out_hash = outs[0]["data_hash"] if outs else None

            # Skip if we've already produced this exact hash
            if expected_out_hash and expected_out_hash in produced_hashes:
                continue

            # Pre-seed (append semantics)
            preseed_path = None
            seed_hash = _pick_viable_seed(run_path, ins)
            if seed_hash:
                preseed_path, created_seed = _resolve_seed_no_copy_or_ephemeral(
                    run_path,
                    seed_hash,
                    seen_steps=seen_steps,
                    created_paths=created_paths,
                )
                if created_seed and created_paths is not None:
                    created_paths.append(preseed_path)

            # Deterministic final target
            if to_path:
                target_override_this_sink = to_path
                conflict_policy = "overwrite"
            elif expected_out_hash:
                target_override_this_sink = str(
                    (RESTORE_DEFAULT_DIR / f"{expected_out_hash}.csv").resolve()
                )
                conflict_policy = "overwrite"
            else:
                default_name = Path(st["params"]["target"]).name
                target_override_this_sink = str(
                    (RESTORE_DEFAULT_DIR / default_name).resolve()
                )
                conflict_policy = "overwrite"

            upstream_gen = upstream
            if upstream_gen is None:
                upstream_gen = _upstream_gen_for_sink(run_path, sh)

            out = bindings.run_save_to_csv_replay(
                run_path=run_path,
                params=st["params"],
                realized={},
                upstream_gen=upstream_gen,
                target_override=target_override_this_sink,
                backend_override=backend_override,
                preseed_path=preseed_path,
                expected_out_hash=expected_out_hash,
                on_conflict=conflict_policy,
                record_materialization=(not ephemeral_only),
            )
            print(f"Replayed: {out}")

            if expected_out_hash:
                outp = Path(out)
                produced[expected_out_hash] = outp
                created_paths.append(outp)
                produced_hashes.add(expected_out_hash)

            upstream = None
            continue

        if fqn == "blase.Load.save_images_to_parquet":
            ins = store.load_step_inputs(run_path, sh)
            outs = store.load_step_outputs(run_path, sh)
            st = store.load_step(run_path, sh)

            def _ord(o):
                n = o["name"]
                return int(n.split("_")[-1]) if n.startswith("parquet_shard_") else 0

            outs.sort(key=_ord)

            # Collect expected hashes (one per shard) if they were recorded.
            expected_out_hashes = [
                o["data_hash"] for o in outs if o["name"].startswith("parquet_shard")
            ]

            # Destination: a directory
            if to_path:
                target_dir_override = to_path
                conflict_policy = "overwrite"
            else:
                # default: deterministic directory under RESTORE_DEFAULT_DIR
                # using the first expected hash as a hint to make the path unique.
                RESTORE_DEFAULT_DIR.mkdir(parents=True, exist_ok=True)
                target_dir_override = str((RESTORE_DEFAULT_DIR).resolve())
                conflict_policy = "overwrite"

            upstream_gen = upstream or _upstream_gen_for_sink(run_path, sh)
            handler = bindings.RESTORE_HANDLERS[fqn]
            out_paths = handler(
                run_path=run_path,
                params=st["params"],
                realized={},
                upstream_gen=upstream_gen,
                target_override=target_dir_override,
                on_conflict=conflict_policy,
                expected_out_hashes=expected_out_hashes or None,
                record_materialization=(not ephemeral_only),
            )

            print("Replayed shards:")
            for p in out_paths:
                print(f"  {p}")

            # Track produced hashes → paths
            # If outputs were recorded, map them 1:1; else compute here.
            if outs:
                for i, o in enumerate(outs):
                    dh = o["data_hash"]
                    if i < len(out_paths):
                        produced[dh] = Path(out_paths[i])
                        created_paths.append(Path(out_paths[i]))
                        produced_hashes.add(dh)
            else:
                # No recorded outputs? compute and index
                for p in out_paths:
                    dh = Hash().hash_file(p)
                    produced[dh] = Path(p)
                    created_paths.append(Path(p))
                    produced_hashes.add(dh)

            upstream = None
            continue

        # Fallback for non-stream steps, if any
        from blase import restore as _restore

        _restore.step(run_path, sh)

    return produced


def _ensure_data_local_or_replay(
    run_path: Path,
    data_hash: str,
    kind: str,
    *,
    seen_steps: Optional[Set[str]] = None,
    created_paths: Optional[List[Path]] = None,
) -> Tuple[Path, bool]:
    """
    Ensure a data hash is locally materialized. If not, replay its producer step
    (using shared seen_steps/created_paths), then materialize it.
    Returns (path, created_now).
    """
    if not kind:
        kind = store.kind_for_hash(run_path, data_hash)
    try:
        ext = {
            "parquet": ".parquet",
            "parquet.shard": ".parquet",
            "image.manifest": ".json",
            "image.batch.meta": ".json",
            "code": ".py",
            "env": ".json",
        }.get(kind, ".bin")
        target_name = f"{data_hash}{ext}"
        p = materialize.ensure_local(
            run_path,
            data_hash,
            kind=kind,
            policy="reuse",
            to_dir=None,
            target_name=target_name,
        )
        _assert_path_matches_hash(p, data_hash, f"{kind} materialization")
        return p, False
    except NeedReplay:
        prod = store.producer_step_for_data(run_path, data_hash)
        if not prod:
            raise SystemExit(
                f"restore: missing local copy and no producer for {data_hash}"
            )

        produced = _exec_plan_for_step(
            run_path,
            prod,
            to_path=None,
            backend_override=None,
            seen_steps=seen_steps,
            created_paths=created_paths,
        )

        p = produced.get(data_hash) if produced else None
        if p is None:
            # fallback: now that replay happened, ensure_local should succeed
            p = materialize.ensure_local(
                run_path, data_hash, kind=kind, policy="reuse", to_dir=None
            )

        _assert_path_matches_hash(p, data_hash, f"{kind} materialization after replay")
        return p, True


def _pick_viable_seed(run_path: Path, ins: list[dict]) -> Optional[str]:
    """
    From a list of inputs, pick a seed data hash that is either replayable (has a producer)
    or already materialized. Skip dead seeds.
    """
    seeds = [i["data_hash"] for i in ins if i.get("role") == "seed"]
    if not seeds:
        return None
    for dh in seeds:
        if store.producer_step_for_data(run_path, dh):
            return dh
    for dh in seeds:
        try:
            kind = store.get_data_kind(run_path, dh) or "csv"
            materialize.ensure_local(
                run_path, dh, kind=kind, policy="reuse", to_dir=None
            )
            return dh
        except Exception:
            pass
    return None


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
                out.append(
                    {
                        "step_hash": node["step_hash"],
                        "function_fqn": node["function_fqn"],
                    }
                )
            elif "hash" in node:
                st = store.load_step(run_path, node["hash"])
                out.append(
                    {"step_hash": node["hash"], "function_fqn": st["function_fqn"]}
                )
            else:
                # Last resort: try to find something that looks like a hash
                h = (
                    node.get("id")
                    or node.get("step")
                    or node.get("node")
                    or node.get("sha")
                )
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


STREAM_FQNS = (
    "blase.Extract.read_csv",
    "blase.Extract.read_images",
    "blase.Transform.apply_function",
)


def _is_stream_fqn(fqn: str) -> bool:
    return any(fqn.endswith(s) for s in STREAM_FQNS)


def _build_stream_for_step(run_path: Path, step_hash: str):
    """
    Construct a streaming generator for a recorded step.

    Given a `step_hash`, this inspects the recorded function type and returns
    an upstream generator compatible with replay bindings:
    - `Extract.read_csv` → `bindings.run_read_csv_restore(...)`
    - `Extract.read_images` → `bindings.run_read_images_restore(...)`
    - `Transform.apply_function` → chains to the nearest upstream stream
      (from the plan or via lineage), then returns
      `bindings.run_apply_function_restore(...)` seeded by that stream.
    Non-stream steps delegate to `blase.restore.step(..., kind="data")`.

    Parameters
    ----------
    run_path : Path
        Root directory of the recorded run.
    step_hash : str
        Hash of the step to materialize as a stream.

    Returns
    -------
    iterator
        A generator yielding `(batch, is_last)` tuples, or a binding-specific
        generator that internally yields `(batch, meta, is_last)` for image
        pipelines and is consumed by downstream replay code. For non-stream
        steps, the return value of `blase.restore.step(..., kind="data")` is
        forwarded.

    Notes
    -----
    - For `Extract.read_csv`, the realized inputs include
      `__expected_source_hash__` when a recorded source hash exists.
    - For `Extract.read_images`, realized inputs may include `manifest`
      and a list of `batch` hashes when present in recorded inputs.
    - For `Transform.apply_function`, the function:
        1) Attempts to find the nearest upstream streaming node from the
           topological plan. If none is found, it falls back to lineage:
           manifest → batch_desc → batch producer step.
        2) Loads the exact recorded callable via `code.load_callable_from_blob`
           using the recorded code hash.
        3) Returns the binding `run_apply_function_restore` wired to the
           upstream stream (or a materialized path for older CSV runs).
    - If no suitable upstream can be located for a transform, a
      `SystemExit` is raised by the caller block that invokes this helper.

    See Also
    --------
    run_read_csv_restore
    run_read_images_restore
    run_apply_function_restore
    blase.restore.step
    """
    st = store.load_step(run_path, step_hash)
    fqn = st["function_fqn"]

    # ---- Extract.read_csv ----
    if fqn.endswith("Extract.read_csv"):
        ins = store.load_step_inputs(run_path, step_hash)
        realized = {}
        src_hash = next((i["data_hash"] for i in ins if i["role"] == "source"), None)
        if src_hash:
            realized["__expected_source_hash__"] = src_hash
        return bindings.run_read_csv_restore(
            run_path=run_path, params=st["params"], realized=realized, transform_fn=None
        )

    # ---- Extract.read_images ----
    if fqn.endswith("Extract.read_images"):
        ins = store.load_step_inputs(run_path, step_hash)
        manifest_hash = next(
            (i["data_hash"] for i in ins if i["role"] == "manifest"), None
        )
        batch_hashes = [i["data_hash"] for i in ins if i["role"] == "batch"]
        realized = {"source": st["params"].get("directory")}
        if manifest_hash:
            realized["manifest"] = manifest_hash
        if batch_hashes:
            realized["batch"] = batch_hashes
        return bindings.run_read_images_restore(
            run_path=run_path, params=st["params"], realized=realized, transform_fn=None
        )

    # ---- Transform.apply_function (chain to its immediate upstream) ----
    if fqn.endswith("Transform.apply_function"):
        raw_plan = planner.plan_for_step(run_path, step_hash) or []
        nodes = _normalize_plan_nodes(run_path, raw_plan)
        try:
            idx = next(i for i, n in enumerate(nodes) if n["step_hash"] == step_hash)
        except StopIteration:
            raise SystemExit("restore: transform step not found in plan")
        upstream_node = None
        for j in range(idx - 1, -1, -1):
            if _is_stream_fqn(nodes[j]["function_fqn"]):
                upstream_node = nodes[j]
                break
        if upstream_node is None:
            # ---- Resolve by lineage hashes (manifest / batch_desc / batch) ----
            ins = store.load_step_inputs(run_path, step_hash)

            # prefer a manifest hash if present
            man = next((i["data_hash"] for i in ins if i["role"] == "manifest"), None)
            if man:
                ex = store.producer_step_for_data(run_path, man)
                if ex:
                    upstream_node = {
                        "step_hash": ex,
                        "function_fqn": store.load_step(run_path, ex)["function_fqn"],
                    }

            # else try batch_desc, then batch
            if upstream_node is None:
                bdesc = next(
                    (
                        i["data_hash"]
                        for i in ins
                        if i["role"]
                        in (
                            "batch_desc",
                            "batch_meta",
                            "batchmeta",
                            "batch_desc_1",
                            "batch_desc_2",
                        )
                    ),
                    None,
                )
                if bdesc:
                    ex = store.producer_step_for_data(run_path, bdesc)
                    if ex:
                        upstream_node = {
                            "step_hash": ex,
                            "function_fqn": store.load_step(run_path, ex)[
                                "function_fqn"
                            ],
                        }

            if upstream_node is None:
                b = next(
                    (
                        i["data_hash"]
                        for i in ins
                        if i["role"] in ("batch", "image.batch")
                    ),
                    None,
                )
                if b:
                    ex = store.producer_step_for_data(run_path, b)
                    if ex:
                        upstream_node = {
                            "step_hash": ex,
                            "function_fqn": store.load_step(run_path, ex)[
                                "function_fqn"
                            ],
                        }
        if upstream_node is None:
            # Fallback: use recorded CSV source path if present (older runs)
            ins = store.load_step_inputs(run_path, step_hash)
            src = next((i["data_hash"] for i in ins if i["role"] == "source"), None)
            if not src:  # nothing to chain
                raise SystemExit(
                    "restore: cannot locate upstream producer for Transform.apply_function"
                )
            p = store.get_materialized_path(
                run_path, src
            ) or store.get_recorded_source_path(run_path, src)
            if not p or not p.exists():
                p = materialize.ensure_local(
                    run_path, src, kind=store.get_data_kind(run_path, src)
                )
            upstream_gen = p.as_posix()  # path-like; binding will read CSV
        else:
            upstream_gen = _build_stream_for_step(run_path, upstream_node["step_hash"])

        # load the exact user callable
        ins = store.load_step_inputs(run_path, step_hash)
        code_hash = store.pick_code_hash(ins)
        fn = code.load_callable_from_blob(cas.path_for(run_path, "code", code_hash))

        return bindings.run_apply_function_restore(
            run_path=run_path,
            params=st["params"],
            realized={"source": upstream_gen},
            transform_fn=fn,
        )

    # Non-stream fallback
    from blase import restore as restore_mod

    return restore_mod.step(run_path, step_hash, kind="data")


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

    # ---------------------------
    # DATA-CENTRIC RESTORE PATH
    # ---------------------------
    if getattr(args, "data", None):
        data_hash = args.data
        kind = store.get_data_kind(run_path, data_hash) or "data"

        # try fast materialize first
        try:
            if args.to:
                to_path = Path(args.to).resolve()
                out = materialize.ensure_local(
                    run_path,
                    data_hash,
                    kind=kind,
                    to_dir=to_path.parent,
                    target_name=to_path.name,
                    on_conflict=args.on_conflict,
                )
            else:
                RESTORE_DEFAULT_DIR.mkdir(parents=True, exist_ok=True)
                ext = "csv" if kind == "csv" else "bin"
                out = materialize.ensure_local(
                    run_path,
                    data_hash,
                    kind=kind,
                    to_dir=RESTORE_DEFAULT_DIR,
                    target_name=f"{data_hash}.{ext}",
                    on_conflict=args.on_conflict,
                )
            _assert_path_matches_hash(Path(out), data_hash, "fast-path materialize")
            print(f"Materialized: {out}")
            return 0
        except NeedReplay:
            pass  # fall through to replay

        # Replay path
        seen_steps: Set[str] = set()
        created_paths: List[Path] = []

        # find the producer step and execute its forward plan
        prod = store.producer_step_for_data(run_path, data_hash)
        if not prod:
            raise SystemExit(
                f"[restore] no producer step recorded for data {data_hash}"
            )

        produced = _exec_plan_for_step(
            run_path,
            prod,
            to_path=args.to,
            backend_override=getattr(args, "backend", None),
            seen_steps=seen_steps,
            created_paths=created_paths,
        )

        # ---- VERIFY FINAL ARTIFACT MATCHES REQUESTED HASH ----
        # 1) Prefer the actual path that produced THIS hash (robust against renames)
        final_path = produced.get(data_hash)

        # 2) Fallbacks if mapping isn’t available (shouldn’t happen, but be safe)
        if final_path is None:
            if args.to:
                final_path = Path(args.to).resolve()
            else:
                # fallback to sink target basename under RESTORE_DEFAULT_DIR
                sink_step = store.producer_step_for_data(run_path, data_hash)
                if not sink_step:
                    raise SystemExit(
                        "restore: cannot locate producing sink for final verification"
                    )
                sink_meta = store.load_step(run_path, sink_step)
                default_name = Path(
                    sink_meta["params"].get("target", f"{data_hash}.csv")
                ).name
                final_path = (RESTORE_DEFAULT_DIR / default_name).resolve()

        if not final_path.exists():
            raise SystemExit(
                f"restore: expected final output does not exist: {final_path}"
            )

        _assert_path_matches_hash(final_path, data_hash, "final output")
        print(f"Verified: {final_path} == {data_hash[:16]}…")

        # --- Cleanup intermediates unless user asked to keep them ---
        if not getattr(args, "keep_intermediates", False):
            final_real = final_path.resolve()

            # De-dupe by resolved path and remove everything except the final
            seen: set[Path] = set()
            for p in created_paths:
                try:
                    r = p.resolve()
                except Exception:
                    continue
                if r in seen:
                    continue
                seen.add(r)
                if r == final_real:
                    continue
                try:
                    Path(r).unlink(missing_ok=True)
                except Exception:
                    # best-effort: ignore failures (e.g., already deleted)
                    pass

            return 0

        # Non-sink producer: stream verify only for now
        if args.mode in ("verify",):
            gen = (
                _build_stream_for_step(run_path, prod)
                if _is_stream_fqn(store.load_step(run_path, prod)["function_fqn"])
                else restore_step(run_path, prod, kind="data")
            )
            total = 0
            for i, (b, last) in enumerate(gen, 1):
                n = len(b) if hasattr(b, "__len__") else "?"
                print(f"[verify] batch {i}: {n} rows  last={last}")
                if isinstance(n, int):
                    total += n
                if getattr(args, "limit_batches", None) and i >= args.limit_batches:
                    break
            print(f"[verify] total rows (best-effort): {total}")
            return 0

        raise SystemExit(
            "restore: replay for non-sink data is not implemented yet in --data mode. "
            "Use `--step <producer_step>` or `--mode verify`."
        )

    # ---------------------------
    # STEP-CENTRIC RESTORE PATH
    # ---------------------------
    if not getattr(args, "step", None):
        raise SystemExit(
            "restore: either --step or --data is required (mutually exclusive)."
        )

    step_hash = args.step

    # Load step metadata up front (so we can decide sink vs non-sink)
    st = store.load_step(run_path, step_hash)  # {'function_fqn','params','status'}
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
                run_path,
                out_hash,
                kind=kind,
                to_dir=to_dir,
                target_name=target_name,
                on_conflict=args.on_conflict,
            )
            print(f"Materialized: {path}")
            return 0
        except materialize.NeedReplay:
            print("Output not available locally. Use:")
            print(f"  blase restore plan --step {step_hash}")
            print(f"  blase restore run  --step {step_hash} --mode replay")
            return 3

    # ---- VERIFY: stream results without writing ----
    if args.mode == "verify":
        if fqn in ("blase.Load.save_to_csv", "blase.Load.save_images_to_parquet"):
            upstream_gen = _upstream_gen_for_sink(run_path, step_hash)
            gen = upstream_gen
        elif _is_stream_fqn(fqn):
            gen = _build_stream_for_step(run_path, step_hash)
        else:
            gen = restore_step(run_path, step_hash, kind="data")

        def _coerce_stream_2(gen):
            for item in gen:
                if isinstance(item, tuple):
                    if len(item) == 3:
                        b, _meta, last = item
                    elif len(item) == 2:
                        b, last = item
                    else:
                        b, last = item, False
                else:
                    b, last = item, False
                yield b, last

        total = 0
        for i, (b, last) in enumerate(_coerce_stream_2(gen), 1):
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
        if fqn in ("blase.Load.save_to_csv", "blase.Load.save_images_to_parquet"):
            upstream_gen = _upstream_gen_for_sink(run_path, step_hash)
            handler = bindings.RESTORE_HANDLERS[fqn]  # handler must exist for both FQNs
            out_path = handler(
                run_path=run_path,
                params=st["params"],
                realized={},  # sinks don’t need extra realized inputs
                upstream_gen=upstream_gen,  # (batch, is_last) generator
                target_override=getattr(args, "to", None),
                backend_override=getattr(args, "backend", None),
            )
            print(out_path)
            return 0

        # non-sinks: as before
        gen = (
            _build_stream_for_step(run_path, step_hash)
            if _is_stream_fqn(fqn)
            else restore_step(run_path, step_hash, kind="data")
        )
        for i, (b, last) in enumerate(gen, 1):
            n = len(b) if hasattr(b, "__len__") else "?"
            print(f"batch {i}: {n} items, last={last}")
            if getattr(args, "limit_batches", None) and i >= args.limit_batches:
                break
        return 0

    raise SystemExit(f"[restore] unknown mode: {args.mode}")
