from __future__ import annotations
from typing import Dict, Any, Iterator, Tuple, Optional, Iterable
from pathlib import Path
from datetime import datetime
import os

from blase.extracting.csv_backend import read_batches_pandas, read_batches_polars
from blase.load import Load
from blase.restoring import store
from blase.utils.hashing import Hash

# simple arg mapping: role -> kwarg
REGISTRY = {
    "blase.Extract.read_csv": {"source": "file_path"},
}

def apply_registry(function_fqn: str, realized_by_role: Dict[str, Any], kwargs: Dict[str, Any]) -> None:
    """
    Apply argument mapping from REGISTRY to kwargs.

    Parameters
    ----------
    function_fqn : str
        Fully-qualified function name (e.g., ``"blase.Extract.read_csv"``).
    realized_by_role : dict of str -> Any
        Mapping of realized role names to values.
    kwargs : dict
        Keyword arguments to be updated in-place.
    """
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
    """
    Choose an output file path for replay with conflict handling.

    Parameters
    ----------
    params : dict
        Original recorded parameters, may include ``"target"``.
    target_override : str, optional
        Override target path, if provided.
    on_conflict : {"overwrite", "fail", "rename"}
        Policy if target already exists.
    allow_append : bool
        Whether appending to existing output is permitted.

    Returns
    -------
    Path
        Resolved replay output path.
    """
    raw = target_override or params.get("target")
    base = Path(raw) if raw else Path("data/working/replayed/output.csv")
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
    Replay an ``Extract.read_csv`` step from a prior run.

    This function replays a CSV extraction by invoking the recorded backend
    (`pandas` or `polars`) with parameters captured during the original run.  
    It verifies source hashes when available, ensuring deterministic restores.

    Parameters
    ----------
    run_path : Path
        Path to the recorded run directory.
    params : dict
        Recorded parameters (e.g., ``backend``, ``batch_size``, ``use_cols``,
        ``filter_by``).
    realized : dict
        Realized inputs/materializations, including ``"source"``.
    transform_fn : callable, optional
        Unused. Present for uniformity across replay signatures.

    Yields
    ------
    tuple
        Two-element tuple ``(batch, is_last)`` where ``batch`` is a DataFrame-like
        object from the backend and ``is_last`` is a bool indicating the final batch.

    Raises
    ------
    RuntimeError
        If a provided expected source hash mismatches the current file contents.
    """
    backend   = params.get("backend", "pandas")
    file_path = str(realized.get("source") or params.get("file_path"))

    # Strict verification if the caller provides the expected source hash.
    expected_src_hash = realized.get("__expected_source_hash__")
    if expected_src_hash is not None:
        actual = Hash().hash_file(Path(file_path))
        if actual != expected_src_hash:
            raise RuntimeError(
                "Source file hash mismatch vs recorded input; aborting nondeterministic restore."
            )

    batch_size = params.get("batch_size")
    use_cols   = params.get("use_cols")
    filter_by  = params.get("filter_by")

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
    Replay a ``Transform.apply_function`` step without tracking.

    Loads upstream materialized data, applies the restored transformation
    function batch-by-batch, and yields transformed outputs.

    Parameters
    ----------
    run_path : Path
        Path to the run directory.
    params : dict
        Recorded parameters controlling extraction (backend, batch_size, etc.).
    realized : dict
        Realized upstream inputs. Must contain ``"source"`` path.
    transform_fn : callable
        User-defined transform function restored from recorded code.

    Yields
    ------
    tuple
        Two-element tuple ``(batch, is_last)`` where ``batch`` is the transformed
        batch object and ``is_last`` indicates end of stream.

    Raises
    ------
    RuntimeError
        If ``"source"`` is missing from realized inputs.
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
    Replay a ``Load.save_to_csv`` step.

    Consumes an upstream generator of data batches, writes output to a temp file,
    verifies against expected output hashes, and moves the result into place with
    conflict policies. Supports pre-seeding for append semantics.

    Parameters
    ----------
    run_path : Path
        Path to the run directory.
    params : dict
        Recorded step parameters (backend, etc.).
    realized : dict, optional
        Reserved for future use.
    transform_fn : callable, optional
        Not used. Present for uniformity across replay signatures.
    upstream_gen : iterable of (Any, bool), optional
        Upstream generator yielding (batch, is_last). Required.
    target_override : str, optional
        Explicit path for output file.
    backend_override : str, optional
        Backend to force ("pandas" or "polars"), overrides params.
    on_conflict : {"overwrite", "fail", "rename"}
        Policy if target exists.
    allow_append : bool
        Whether appending is allowed.
    preseed_path : Path, optional
        File to seed the temporary output (used to reproduce append semantics).
    expected_out_hash : str, optional
        Expected output hash for verification.
    record_materialization : bool
        If True, record final materialization in store.

    Returns
    -------
    str
        Path to the final replayed CSV file.

    Raises
    ------
    RuntimeError
        If no upstream generator is provided or output hash mismatches.
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
        if tmp.exists(): 
            tmp.unlink()
        shutil.copy2(preseed_path, tmp)

    # consume to tmp (append semantics preserved if seed existed)
    ld = Load()
    for (b, last) in upstream_gen:
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