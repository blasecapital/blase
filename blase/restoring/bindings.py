from __future__ import annotations
from typing import Dict, Any, Iterator, Tuple, Optional, Iterable, Union, List
from pathlib import Path
from datetime import datetime
import os
import json

from blase.extracting.csv_backend import read_batches_pandas, read_batches_polars
from blase.extracting.image_backend import (
    scan_manifest_headers, 
    shuffle_manifest,
    ensure_item_content_hashes,
    compute_manifest_root_hash,
    plan_image_batches,
    decode_batch_pil,
    decode_batch_cv2,
    iter_batches_from_plan,
    compute_batch_hash
)
from blase.load import Load
from blase.restoring import store, cas
from blase.utils.hashing import Hash

# simple arg mapping: role -> kwarg
REGISTRY = {
    "blase.Extract.read_csv": {"source": "file_path"},
    "blase.Extract.read_images": {
        "source": "directory",
        "manifest": "_manifest_hash",
        "batch": "_batch_desc",
    },
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
        if role not in realized_by_role or not arg_name:
            continue
        val = realized_by_role[role]
        if isinstance(val, (str, os.PathLike)):
            kwargs[arg_name] = str(val)
        else:
            kwargs[arg_name] = val

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

def _load_cas_json(run_path: Path, sha256_hex: str, *, kind: Optional[str]) -> Dict[str, Any]:
    """
    Load a CAS JSON blob by hash and kind, tolerant of legacy layouts.
    """
    if not kind:
        raise ValueError("CAS kind is required to resolve namespaced layout.")

    # 1) Canonical path (namespaced by kind, bucketed aa/bb/hash)
    cand = [cas.path_for(run_path, kind, sha256_hex)]

    # 2) Legacy fallback: flat (no kind)
    cand.append(run_path / "cas" / "sha256" / sha256_hex[:2] / sha256_hex[2:4] / sha256_hex)

    # 3) Legacy typo fallback for image.batch.meta -> image.batch.met
    if kind.endswith(".meta"):
        legacy_kind = kind[:-1]  # drop the trailing 'a'
        cand.append(run_path / "cas" / "sha256" / legacy_kind / sha256_hex[:2] / sha256_hex[2:4] / sha256_hex)

    # 4) Reversed bucket order fallback (seen in older runs)
    cand.append(run_path / "cas" / "sha256" / kind / sha256_hex[2:4] / sha256_hex[:2] / sha256_hex)

    for p in cand:
        try:
            with open(p, "rb") as f:
                return json.loads(f.read().decode("utf-8"))
        except FileNotFoundError:
            continue

    raise FileNotFoundError(f"CAS blob not found for kind={kind} hash={sha256_hex}")

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
    backend   = params.get("backend", "polars")
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


def run_read_images_restore(
    *,
    run_path: Path,
    params: Dict[str, Any],
    realized: Dict[str, Any],
    transform_fn=None,   # unused; signature kept for uniformity
) -> Iterator[Tuple[Any, bool]]:
    """
    Replay an Extract.read_images step deterministically.

    Strategy:
      1) Resolve directory + recorded params.
      2) If a manifest hash is present, load it and enforce the same root_hash.
      3) Re-scan headers + content hashes; compute root_hash and compare.
      4) Rebuild the batch plan (deterministic), optionally validate against recorded batch metas.
      5) Decode with the recorded backend and yield (batch, is_last).
    """
    # ---------- 1) Resolve core params ----------
    backend   = params.get("backend", "pil")
    return_tp = params.get("return_type", "np")
    color     = params.get("color", "rgb")
    max_side  = params.get("max_side")
    mode      = params.get("mode", "auto")
    safety    = params.get("safety_margin", 0.15)
    max_item_decoded_bytes = params.get("max_item_decoded_bytes")
    batch_size = params.get("batch_size")
    target_bytes = params.get("target_batch_bytes")
    pattern   = params.get("pattern", "**/*.jpg")
    recursive = bool(params.get("recursive", True))
    shuffle   = bool(params.get("shuffle", False))
    seed      = params.get("seed", None)
    hash_mode = params.get("hash_mode", "content")

    directory = str(realized.get("source") or params.get("directory"))
    if not directory:
        raise RuntimeError("read_images restore requires 'source' (directory) or params['directory'].")

    if shuffle and seed is None:
        raise RuntimeError("Recorded run used shuffle=True but no seed; cannot restore deterministically.")

    # Recorded manifest/batch (optional but recommended)
    manifest_hash = realized.get("manifest")
    if manifest_hash:
        recorded_manifest_desc = _load_cas_json(run_path, manifest_hash, kind="image.manifest")
        expected_root_hash = (recorded_manifest_desc.get("identity", {}) or {}).get("root_hash") \
                            or recorded_manifest_desc.get("root_hash")
    recorded_batches = realized.get("batch") or []
    recorded_batch_descs = []
    for b in (recorded_batches if isinstance(recorded_batches, list) else [recorded_batches]):
        if isinstance(b, str):
            try:
                desc = _load_cas_json(run_path, b, kind="image.batch.meta")
                recorded_batch_descs.append(desc)
            except FileNotFoundError:
                recorded_batch_descs.append({})
        else:
            recorded_batch_descs.append(b)

    # ---------- 2) Load recorded manifest (if provided) ----------
    expected_root_hash = None
    recorded_manifest_desc = None
    if manifest_hash:
        recorded_manifest_desc = _load_cas_json(run_path, manifest_hash, kind="image.manifest")
        expected_root_hash = (
            recorded_manifest_desc.get("identity", {}).get("root_hash")
            or recorded_manifest_desc.get("root_hash")  # tolerate older schema
        )

    # ---------- 3) Rebuild manifest deterministically ----------
    # Header scan (no pixels)
    manifest = scan_manifest_headers(
        directory=directory,
        pattern=pattern,
        recursive=recursive,
        filename_filter=None,
    )
    # (Optional) shuffle — must use recorded seed if used during the original run
    if shuffle:
        manifest = shuffle_manifest(manifest, seed)

    # Ensure content hashes (use cached fast path if available)
    manifest = ensure_item_content_hashes(manifest)

    # Compute dataset identity root hash the same way as during the run
    root_hash = compute_manifest_root_hash(
        manifest=manifest,
        directory=directory,
        pattern=pattern,
        recursive=recursive,
        seed=seed,
        hash_mode=hash_mode,
    )

    # Enforce identity if we have a recorded manifest
    if expected_root_hash and root_hash != expected_root_hash:
        raise RuntimeError(
            f"Restore aborted: current manifest root_hash ({root_hash[:12]}) "
            f"does not match recorded ({expected_root_hash[:12]}). Source changed."
        )

    # ---------- 4) Rebuild batch plan (deterministic) ----------
    # If target_bytes wasn't recorded, prefer a stable default (avoid RAM-based inference)
    if mode == "auto" and target_bytes is None:
        target_bytes = 256 * 1024 * 1024  # 256 MiB default

    batch_plan = plan_image_batches(
        manifest=manifest,
        mode=mode,
        batch_size=batch_size,
        target_batch_bytes=target_bytes,
        safety_margin=safety,
        max_item_decoded_bytes=max_item_decoded_bytes,
        max_side=max_side,
    )

    # Optional validation against recorded batch metas
    recorded_batch_descs: List[Dict[str, Any]] = []
    if recorded_batches:
        # Normalize to list of dicts
        rec_hashes = recorded_batches if isinstance(recorded_batches, list) else [recorded_batches]
        for b in rec_hashes:
            desc = _load_cas_json(run_path, b, kind="image.batch.meta") if isinstance(b, str) else b
            recorded_batch_descs.append(desc)

    impl = decode_batch_pil if backend == "pil" else decode_batch_cv2

    # ---------- 5) Decode and yield ----------
    for i, planned in enumerate(iter_batches_from_plan(batch_plan), 1):
        items = planned["items"]
        # Recompute the same batch_hash we logged originally
        batch_hash = compute_batch_hash(root_hash, items)

        # Cross-check vs recorded batch meta when available (count + optional hash)
        if recorded_batch_descs:
            if i-1 >= len(recorded_batch_descs):
                raise RuntimeError("Recorded batch metas shorter than planned batches; cannot restore.")
            rec = recorded_batch_descs[i-1]
            rec_count = rec.get("count")
            if rec_count is not None and rec_count != len(items):
                raise RuntimeError(
                    f"Batch #{i} size mismatch: recorded={rec_count}, planned={len(items)}."
                )
            rec_hash = rec.get("batch_hash")
            if rec_hash and rec_hash != batch_hash:
                raise RuntimeError(
                    f"Batch #{i} hash mismatch: recorded={rec_hash[:12]}, planned={batch_hash[:12]}."
                )

        decoded = impl(items, return_tp, color, max_side)
        is_last = bool(planned.get("is_last", False))
        yield decoded, is_last

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
    upstream = realized.get("source") or realized.get("source_gen")
    if upstream is None:
        raise RuntimeError("restore: missing realized 'source' for Transform.apply_function")

    # If upstream is already a generator of (batch, is_last), use it.
    if not isinstance(upstream, (str, os.PathLike)):
        gen = upstream
    else:
        # Legacy CSV path
        source_path = str(upstream)
        backend    = params.get("backend", "polars")
        batch_size = params.get("batch_size")
        use_cols   = params.get("use_cols")
        filter_by  = params.get("filter_by")
        gen = read_batches_polars(source_path, batch_size, use_cols, filter_by) \
              if backend == "polars" else \
              read_batches_pandas(source_path, batch_size, use_cols, filter_by)

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
    "blase.Extract.read_csv": run_read_csv_restore,
    "blase.Extract.read_images": run_read_images_restore,
    "blase.Transform.apply_function": run_apply_function_restore,
    "blase.Load.save_to_csv": run_save_to_csv_replay
}