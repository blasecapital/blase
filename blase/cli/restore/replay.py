from pathlib import Path
from typing import Optional, Tuple, Set, List, Dict, Optional

from blase.cli.restore.replay_cas import _assert_path_matches_hash
from blase.cli.restore.replay_exec import (
    _exec_extract_csv,
    _exec_extract_images,
    _exec_extract_parquet,
    _exec_transform_apply_function,
    _exec_load_save_to_csv,
    _exec_load_save_images_to_parquet,
)
from blase.restoring import store, planner, code, bindings, materialize, cas
from blase.restoring.materialize import NeedReplay


# ============
# helpers
# ============
STREAM_FQNS = {
    "blase.Extract.read_csv",
    "blase.Extract.read_images",
    "blase.Extract.read_parquet",
    "blase.Transform.apply_function",
}
EXTRACT_FQNS = {
    "blase.Extract.read_csv",
    "blase.Extract.read_images",
    "blase.Extract.read_parquet",
}


def _is_stream_fqn(fqn: str) -> bool:
    return any(fqn.endswith(s) for s in STREAM_FQNS)


def _is_extract(fqn: str) -> bool:
    return any(fqn.endswith(s) for s in EXTRACT_FQNS)


def _is_transform(fqn: str) -> bool:
    return fqn == "blase.Transform.apply_function"


def _realized_for_read_csv(run_path: Path, step_hash: str) -> Dict:
    ins = store.load_step_inputs(run_path, step_hash)
    src = next((i["data_hash"] for i in ins if i["role"] == "source"), None)
    return {"__expected_source_hash__": src} if src else {}

def _realized_for_read_images(run_path: Path, step_hash: str, params: Dict) -> Dict:
    ins = store.load_step_inputs(run_path, step_hash)
    man = next((i["data_hash"] for i in ins if i["role"] == "manifest"), None)
    batches = [i["data_hash"] for i in ins if i["role"] == "batch"]
    r = {}
    if "directory" in params:
        r["source"] = params["directory"]
    if man:
        r["manifest"] = man
    if batches:
        r["batch"] = batches
    return r

def _realized_for_read_parquet(run_path: Path, step_hash: str, params: Dict) -> Dict:
    ins = store.load_step_inputs(run_path, step_hash)
    man = next((i["data_hash"] for i in ins if i["role"] == "manifest"), None)
    batches = [i["data_hash"] for i in ins if i["role"] == "batch"]
    r: Dict = {}
    if "sources" in params:
        r["sources"] = params["sources"]
    elif "source" in params:
        r["source"] = params["source"]
    if man:
        r["manifest"] = man
    if batches:
        r["batch"] = batches
    return r


def _find_upstream_stream_node(run_path: Path, step_hash: str) -> Optional[Dict]:
    """Prefer nearest stream in the planner’s order; else try lineage anchors."""
    plan = _normalize_plan_nodes(run_path, planner.plan_for_step(run_path, step_hash))
    try:
        idx = next(i for i, n in enumerate(plan) if n["step_hash"] == step_hash)
    except StopIteration:
        raise SystemExit("restore: transform step not found in plan")

    # nearest previous stream
    for j in range(idx - 1, -1, -1):
        if _is_stream_fqn(plan[j]["function_fqn"]):
            return plan[j]

    # lineage-based fallbacks
    ins = store.load_step_inputs(run_path, step_hash)

    man = next((i["data_hash"] for i in ins if i["role"] == "manifest"), None)
    if man:
        ex = store.producer_step_for_data(run_path, man)
        if ex:
            return {"step_hash": ex, "function_fqn": store.load_step(run_path, ex)["function_fqn"]}

    bdesc = next(
        (i["data_hash"] for i in ins if i["role"] in {
            "batch_desc","batch_meta","batchmeta","batch_desc_1","batch_desc_2","table.batch.meta"
        }), None
    )
    if bdesc:
        ex = store.producer_step_for_data(run_path, bdesc)
        if ex:
            return {"step_hash": ex, "function_fqn": store.load_step(run_path, ex)["function_fqn"]}

    b = next((i["data_hash"] for i in ins if i["role"] in {"batch","image.batch","table.batch"}), None)
    if b:
        ex = store.producer_step_for_data(run_path, b)
        if ex:
            return {"step_hash": ex, "function_fqn": store.load_step(run_path, ex)["function_fqn"]}

    return None

def _fallback_csv_path_as_stream(run_path: Path, step_hash: str) -> str:
    ins = store.load_step_inputs(run_path, step_hash)
    src = next((i["data_hash"] for i in ins if i["role"] == "source"), None)
    if not src:
        raise SystemExit("restore: cannot locate upstream producer for Transform.apply_function")
    p = store.get_materialized_path(run_path, src) or store.get_recorded_source_path(run_path, src)
    if not p or not p.exists():
        p = materialize.ensure_local(run_path, src, kind=store.get_data_kind(run_path, src))
    return p.as_posix()


# ============
# stream
# ============
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

    if fqn.endswith("Extract.read_csv"):
        realized = _realized_for_read_csv(run_path, step_hash)
        return bindings.run_read_csv_restore(run_path=run_path, params=st["params"], realized=realized, transform_fn=None)

    if fqn.endswith("Extract.read_images"):
        realized = _realized_for_read_images(run_path, step_hash, st["params"])
        return bindings.run_read_images_restore(run_path=run_path, params=st["params"], realized=realized, transform_fn=None)

    if fqn.endswith("Extract.read_parquet"):
        realized = _realized_for_read_parquet(run_path, step_hash, st["params"])
        return bindings.run_read_parquet_restore(run_path=run_path, params=st["params"], realized=realized, transform_fn=None)

    if fqn.endswith("Transform.apply_function"):
        upstream_node = _find_upstream_stream_node(run_path, step_hash)
        upstream_gen = (
            _build_stream_for_step(run_path, upstream_node["step_hash"])
            if upstream_node is not None else
            _fallback_csv_path_as_stream(run_path, step_hash)
        )
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


def _upstream_gen_for_sink(run_path: Path, target_step_hash: str):
    """
    Build an upstream (batch,last) generator for a Load.save_to_csv step by scanning
    its plan backward to the nearest Transform.apply_function (preferred) or Extract.read_csv.
    """
    # figure out which sink we’re wiring
    tip = store.load_step(run_path, target_step_hash)
    tip_fqn = tip["function_fqn"]

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
                if _is_extract(nodes[j]["function_fqn"]):
                    upstream_node = nodes[j]
                    break
                elif _is_extract(nodes[j]["function_fqn"]):
                    upstream_node = nodes[j]
                    break
        else:
            # default/legacy CSV path
            for j in range(idx - 1, -1, -1):
                if _is_extract(nodes[j]["function_fqn"]):
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


# ============
# execute plan
# ============
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
    Replay recorded steps up to ``tip_step_hash`` and materialize sinks.

    Walks the stored plan in dependency order, reconstructs upstream generators
    for extract steps, re-applies recorded transforms, and replays load steps to
    produce the same on-disk artifacts (CSV files or Parquet shards). Supports
    both images and table/parquet flows. For ``Transform.apply_function``,
    prefers a Parquet-backed upstream when a manifest anchor is present, and
    falls back to images if needed. When an upstream generator already exists,
    it is piped directly into the next transform without re-anchoring.

    Parameters
    ----------
    run_path : Path
        Run root containing ``nodes/`` and ``cas/``.
    tip_step_hash : str
        Step hash to reach; all required upstream steps are replayed.
    to_path : str or None
        Optional final target override for the sink. When set, the last sink
        writes here regardless of the recorded location.
    backend_override : str or None
        Optional sink backend override passed to replay handlers.
    seen_steps : set of str, optional
        Steps already executed in this process; they are skipped.
    created_paths : list of Path, optional
        Collector for all paths created during this replay.
    ephemeral_only : bool, default False
        If True, do not record materializations back to CAS.

    Returns
    -------
    dict
        Mapping ``{produced_data_hash: Path(output_file)}`` for each sink.

    Raises
    ------
    SystemExit
        If required anchors (e.g., manifest/source) are missing or if neither
        Parquet nor images params can be resolved for a manifest-anchored
        transform.

    Notes
    -----
    Handles:
      * ``Extract.read_csv`` → CSV stream restore.
      * ``Extract.read_images`` → image batches via recorded manifest + descs.
      * ``Extract.read_parquet`` → table/image-compatible batches with recorded
        ``manifest`` and ``batch_desc_*``.
      * ``Transform.apply_function`` → uses an existing upstream generator when
        available; otherwise anchors by ``source`` or ``manifest``. Tries
        ``read_parquet`` params first, then images.
      * ``Load.save_to_csv`` and ``Load.save_images_to_parquet`` → write outputs,
        validate expected hashes, and optionally record materializations.
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
            upstream = _exec_extract_csv(run_path, sh, st)
            continue

        if fqn == "blase.Extract.read_images":
            upstream = _exec_extract_images(run_path, sh, st)
            continue

        if fqn == "blase.Extract.read_parquet":
            upstream = _exec_extract_parquet(run_path, sh, st)
            continue

        if fqn == "blase.Transform.apply_function":
            upstream = _exec_transform_apply_function(
                run_path, upstream, sh, st, seen_steps, created_paths
            )
            continue

        if fqn == "blase.Load.save_to_csv":
            upstream = _exec_load_save_to_csv(
                run_path,
                sh,
                produced_hashes,
                seen_steps,
                created_paths,
                to_path,
                st,
                backend_override,
                ephemeral_only,
                produced,
                upstream,
            )
            continue

        if fqn == "blase.Load.save_images_to_parquet":
            upstream = _exec_load_save_images_to_parquet(
                run_path,
                sh,
                to_path,
                upstream,
                fqn,
                ephemeral_only,
                produced,
                created_paths,
                produced_hashes,
            )
            continue

        # Fallback for non-stream steps, if any
        from blase import restore as _restore

        _restore.step(run_path, sh)

    return produced
