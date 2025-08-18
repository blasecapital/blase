from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any
import inspect
from pathlib import Path

from blase.restoring import store, cas, code, materialize, bindings

def step(run_path: Path, step_hash: str, kind: str = "data", policy: str = "reuse",
         materialize_to: Optional[Path] = None):
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
            # CAS miss → fall back to original source_path if present
            src = store.lookup_source_path(run_path, dh)
            if src and Path(src).exists():
                # Use the original file in place (no copy). Good for read_csv restore.
                realized[role] = Path(src)
            else:
                # Nothing in CAS and no live source file → planner replay needed.
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
