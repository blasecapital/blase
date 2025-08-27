from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any
import inspect

from blase.restoring import store, cas, code, materialize, bindings

def step(run_path: Path, step_hash: str, kind: str = "data", policy: str = "reuse",
         materialize_to: Optional[Path] = None):
    """
    Restore and execute a tracked pipeline step by replaying or materializing its inputs.

    This function re-creates the environment for a given step in a tracked run,
    resolving its inputs (code, environment, and data) from the content-addressable
    store (CAS), from recorded source paths, or by materializing them locally
    if necessary. It then reconstructs the function call associated with the step,
    either by dispatching to a registered restore handler or by invoking the
    original user-defined callable.

    Parameters
    ----------
    run_path : Path
        Filesystem path to the run directory containing the step metadata
        (e.g., ``runs/<run_id>``).
    step_hash : str
        Unique identifier of the step to restore. Used to look up step metadata,
        inputs, and associated code.
    kind : {"data", "code", "env"}, optional
        Kind of artifact to materialize if a CAS lookup fails. Defaults to "data".
    policy : {"reuse", "copy", "link"}, optional
        Materialization policy to apply when restoring missing inputs.
        - ``"reuse"``: reuse existing local files if possible.
        - ``"copy"``: create a copy of the artifact under ``materialize_to``.
        - ``"link"``: create a filesystem link when possible.
        Defaults to "reuse".
    materialize_to : Path or None, optional
        Directory in which to materialize inputs if required. If ``None``,
        the default CAS storage or recorded source path is used.

    Returns
    -------
    Any
        The result of invoking the restored function. This is the same output
        that would have been produced during the original tracked execution.

    Raises
    ------
    FileNotFoundError
        If an input cannot be found in CAS, no recorded source path exists,
        and replay of upstream steps is required.
    RuntimeError
        If attempting to directly restore steps that cannot be re-executed
        (e.g., ``blase.Load.save_to_csv`` must be restored via planner replay).
    SystemExit
        If run metadata or step information cannot be resolved from the given
        ``run_path`` and ``step_hash``.

    Notes
    -----
    - Inputs are resolved in the following order of preference:
      1. CAS path for the given input hash.
      2. Previously recorded source path, if still available on disk.
      3. Materialization via the configured policy.
    - Special-case handling is applied for code and environment inputs,
      which are always resolved directly from CAS.
    - If a registered restore handler exists for the step's function FQN,
      it will be used instead of directly calling the underlying function.
    - For steps without a handler, arguments are reconstructed from stored
      parameters and resolved inputs, with filtering against the function
      signature to avoid passing unexpected parameters.
    """
    st  = store.load_step(run_path, step_hash)
    ins = store.load_step_inputs(run_path, step_hash)
    kinds = store.load_data_kinds(run_path, ins)

    realized: Dict[str, Any] = {}
    for i in ins:
        role = i["role"]
        dh   = i["data_hash"]
        if role in ("code", "env"):
            realized[role] = cas.path_for(run_path, kind=role, data_hash=dh)
            continue

        k = kinds[dh]
        try:
            p = store.get_materialized_path(run_path, dh) or store.get_recorded_source_path(run_path, dh)
            if p and p.exists():
                realized[role] = p
            else:
                # only as a fallback (e.g., explicit `materialize` CLI), not during `replay`:
                realized[role] = materialize.ensure_local(
                    run_path, dh, kind=k, policy=policy, to_dir=materialize_to
                )
        except FileNotFoundError:
            # CAS miss then fall back to original source_path if present
            src = store.lookup_source_path(run_path, dh)
            if src and Path(src).exists():
                # Use the original file in place (no copy). Good for read_csv restore.
                realized[role] = Path(src)
            else:
                # Nothing in CAS and no live source file then planner replay needed.
                raise FileNotFoundError(
                    f"Input {dh} is not in CAS and no valid source_path is available; "
                    f"replay required."
                )

    fn = code.load_callable_from_blob(
        cas.path_for(run_path, "code", store.pick_code_hash(ins))
    )

    fqn = st["function_fqn"]
    handler = bindings.RESTORE_HANDLERS.get(fqn)
    if handler is not None:
        return handler(run_path=run_path, params=st["params"], realized=realized, transform_fn=fn)
    
    if fqn == "blase.Load.save_to_csv":
        raise RuntimeError(
            "Direct restore of Load.save_to_csv is not supported. "
            "Use the planner via CLI: `blase restore run --mode replay --step <id>` "
            "or `--mode materialize` to link/copy an existing artifact."
        )

    kwargs = dict(st["params"])
    for i in ins:
        if i["arg_name"]:
            kwargs[i["arg_name"]] = str(realized.get(i["role"], kwargs.get(i["arg_name"])))
    bindings.apply_registry(fqn, realized, kwargs)

    try:
        sig = inspect.signature(fn)
        allowed = set(sig.parameters.keys())
        kwargs = {k: v for k, v in kwargs.items() if k in allowed}
    except (ValueError, TypeError):
        # builtins / C-extensions may not have a usable signature
        pass

    return fn(**kwargs)
