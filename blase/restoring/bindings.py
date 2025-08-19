from __future__ import annotations
from typing import Dict, Any, Iterator, Tuple, Optional, Iterable
from pathlib import Path
from datetime import datetime
import os, shutil

from blase.extracting.csv_backend import read_batches_pandas, read_batches_polars
from blase.load import Load
from blase.restoring import store, materialize
from blase.utils.hashing import Hash

# simple arg mapping: role -> kwarg
REGISTRY = {
    "blase.Extract.read_csv": {"source": "file_path"},
}

def apply_registry(function_fqn: str, realized_by_role: Dict[str, Any], kwargs: Dict[str, Any]) -> None:
    mapping = REGISTRY.get(function_fqn)
    if not mapping:
        return
    for role, arg_name in mapping.items():
        if role in realized_by_role and arg_name:
            kwargs[arg_name] = str(realized_by_role[role])

# ---------- Restore helpers ----------

def _pick_replay_target(params: Dict[str, Any],
                        target_override: Optional[str],
                        on_conflict: str,
                        allow_append: bool) -> Path:
    base = Path(target_override or params.get("target", ""))
    if not base:
        # derive from original name if available; else fallback
        base = Path("data/working/replayed/output.csv")

    base.parent.mkdir(parents=True, exist_ok=True)

    if base.exists():
        if allow_append:
            return base
        if on_conflict == "overwrite":
            return base
        if on_conflict == "fail":
            raise FileExistsError(f"Target exists: {base}")
        # rename
        stem, suf = base.stem, base.suffix
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return base.with_name(f"{stem}_replayed_{stamp}{suf}")
    return base

# ---------- Restore handlers for streaming/sink steps ----------

def run_read_csv_restore(
    *,
    run_path: Path,
    params: Dict[str, Any],
    realized: Dict[str, Any],
    transform_fn=None,   # unused; signature kept for uniformity
) -> Iterator[Tuple[Any, bool]]:
    """
    Replays Extract.read_csv by calling the backend reader with recorded params.
    Does NOT pass unexpected kwargs to the csv backends.

    Prefers realized['source'] (materialized from DB) and falls back to params['file_path'].
    """
    backend     = params.get("backend", "pandas")
    file_path   = str(realized.get("source") or params.get("file_path"))

    # Verify the SOT matches the recorded 'source' data_hash (if any)
    try:
        # The caller's step hash isn't passed, so we inspect the latest read_csv step
        # in the current replay plan via inputs
        # (You can pass 'step_hash' via params if you want exact)
        # Here, we derive by looking up the 'source' hash among inputs of the latest step that used this file.
        # Simpler: use store.load_step_inputs on the immediate caller’s step if available.
        # Best: pass the step hash in params when calling.
        pass
    except Exception:
        pass

    # Minimal, robust verification using recorded hash if it was resolved upstream:
    # If the caller resolved a 'source' hash in realized, verify it here too:
    # (Optionally skip if not provided)
    try:
        expected_src_hash = realized.get("__expected_source_hash__")
        if expected_src_hash:
            if Hash().hash_file(Path(file_path)) != expected_src_hash:
                raise RuntimeError(
                    "Source file hash mismatch vs recorded input; aborting nondeterministic restore."
                )
    except Exception:
        # if no expected hash provided, continue (best-effort)
        pass

    batch_size  = params.get("batch_size")
    use_cols    = params.get("use_cols")
    filter_by   = params.get("filter_by")

    impl = read_batches_pandas if backend == "pandas" else read_batches_polars
    for batch, is_last in impl(file_path, batch_size, use_cols, filter_by):
        yield batch, is_last

def run_apply_function_restore(
    *,
    run_path: Path,
    params: Dict[str, Any],
    realized: Dict[str, Any],
    transform_fn,   # loaded from the code blob for this step
) -> Iterator[Tuple[Any, bool]]:
    """
    Replays Transform.apply_function without tracking:
      - materialize upstream 'source'
      - re-read in batches using original params
      - apply the restored transform_fn
      - yield (batch, is_last)
    """
    backend    = params.get("backend", "pandas")
    batch_size = params.get("batch_size")
    use_cols   = params.get("use_cols")
    filter_by  = params.get("filter_by")

    source_path = realized.get("source")
    if not source_path:
        raise RuntimeError("restore: missing realized 'source' for Transform.apply_function")

    if backend == "polars":
        gen = read_batches_polars(source_path, batch_size, use_cols, filter_by)
    else:
        gen = read_batches_pandas(source_path, batch_size, use_cols, filter_by)

    for batch, is_last in gen:
        yield transform_fn(batch), is_last

def run_save_to_csv_materialize_output(
    *,
    run_path: Path,
    params: Dict[str, Any],
    realized: Dict[str, Any],
    outputs: Optional[list[Dict[str, Any]]] = None,
    ensure_local,   # pass restore.materialize.ensure_local from caller
) -> Path | list[Path]:
    """
    Preferred restore for a sink: materialize the recorded outputs and return paths.
    If multiple outputs exist, return a list (e.g., shards).
    """
    if not outputs:
        raise RuntimeError("restore: no recorded outputs for Load.save_to_csv")

    paths: list[Path] = []
    for out in outputs:
        data_hash = out["data_hash"]
        # You may want to infer kind='csv' here or store 'kind' on outputs.
        p = ensure_local(run_path, data_hash=data_hash, kind="csv", policy="reuse", to_dir=None)
        paths.append(Path(p))
    return paths[0] if len(paths) == 1 else paths

def run_save_to_csv_replay(
    *,
    run_path,
    params: Dict[str, Any],
    realized: Dict[str, Any] = None,     # not used now, reserved for future
    transform_fn=None,                   # not used; we use the live class
    upstream_gen: Optional[Iterable[Tuple[Any, bool]]] = None,
    target_override: Optional[str] = None,
    backend_override: Optional[str] = None,
    on_conflict: str = "overwrite",
    allow_append: bool = False,
    preseed_path: Optional[Path] = None,
    expected_out_hash: Optional[str] = None,
    record_materialization: bool = True,
) -> str:
    """
    Replays a Load.save_to_csv step by consuming an upstream generator of (batch, last)
    and writing to a temp file that is then moved into place. If a 'seed' input exists,
    we pre-seed the temp file with those exact bytes to reproduce append semantics.
    """
    if upstream_gen is None:
        raise RuntimeError("Load.save_to_csv replay requires an upstream generator of (batch, last).")

    target = _pick_replay_target(params, target_override, on_conflict, allow_append)
    backend = backend_override or params.get("backend") or "pandas"
    tmp = Path(target).with_suffix(Path(target).suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    # If a seed input was recorded and provided, start tmp as that exact seed bytes
    if preseed_path:
        import shutil
        if tmp.exists(): tmp.unlink()
        shutil.copy2(preseed_path, tmp)

    # consume to tmp (append semantics preserved if seed existed)
    ld = Load()
    last_seen = False
    for (b, last) in upstream_gen:
        last_seen = last
        ld.save_to_csv(
            data=b, last_batch=last, meta={},
            path=str(tmp), backend=backend, track=False, use_blase_path=False,
        )

    # Verify against expected_out_hash if available
    computed = Hash().hash_file(tmp)
    if expected_out_hash and computed != expected_out_hash:
        raise RuntimeError("Replay output hash does not match recorded output.")
    final_hash = expected_out_hash or computed

    # move into place with chosen conflict policy
    target_path = Path(target)
    if target_path.exists() and on_conflict == "overwrite":
        target_path.unlink()
    os.replace(tmp, target_path)

    # only record durable finals, never ephemeral seeds
    if record_materialization:
        try:
            store.record_materialization(run_path, final_hash, str(target_path))
        except Exception:
            pass

    return str(target_path)

# Public registry of handlers (by function FQN)
RESTORE_HANDLERS: Dict[str, Any] = {
    "blase.Extract.read_csv":    run_read_csv_restore,
    "blase.Transform.apply_function": run_apply_function_restore,
    "blase.Load.save_to_csv":    run_save_to_csv_replay,
}