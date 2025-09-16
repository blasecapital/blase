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
from blase.restoring.io_safety import resolve_conflict_path
from blase.loading.img_parquet_backend import (
    _build_parquet_table_from_images,
    _write_parquet_table,
    RECORDED_WRITER_CFG
)

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

        Parameters
    ----------
    run_path : Path
        Root directory of the run containing the `cas/sha256` hierarchy.
    sha256_hex : str
        SHA-256 hex digest of the desired CAS object.
    kind : str
        Logical CAS kind (e.g., "image.manifest"). Required.

    Returns
    -------
    dict
        Parsed JSON content of the CAS blob.

    Raises
    ------
    ValueError
        If `kind` is None or empty.
    FileNotFoundError
        If no candidate path contains the requested blob.

    Notes
    -----
    Search order:
      1. Canonical path: `cas/sha256/<kind>/<aa>/<bb>/<hash>`
      2. Legacy flat path: `cas/sha256/<aa>/<bb>/<hash>`
      3. Legacy typo for `*.meta`: drop final "a" in kind directory.
      4. Reversed bucket order: `cas/sha256/<kind>/<bb>/<aa>/<hash>`
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

def _lookup_data_row(run_path: Path, h: str):
    """Return {'kind', 'metadata'} for a CAS data row or None if missing."""
    db = run_path / "nodes" / "nodes.db"
    import sqlite3, json as _json
    con = sqlite3.connect(db)
    try:
        con.row_factory = sqlite3.Row
        row = con.execute("SELECT kind, metadata_json FROM data WHERE data_hash=?", (h,)).fetchone()
        if not row:
            return None
        md = None
        try:
            md = _json.loads(row["metadata_json"]) if row["metadata_json"] else None
        except Exception:
            md = None
        return {"kind": row["kind"], "metadata": md}
    finally:
        con.close()

def _try_load_batch_desc(run_path: Path, h: str) -> dict:
    """Return a batch descriptor dict by trying CAS meta, data row, or batch blob in order."""
    # 1) First try the canonical batch-meta blob
    try:
        return _load_cas_json(run_path, h, kind="image.batch.meta")
    except FileNotFoundError:
        pass
    except Exception:
        # tolerate older runs with odd contents
        pass

    # 2) Fallback: check data row kind + metadata_json
    row = _lookup_data_row(run_path, h)
    if row:
        k = (row.get("kind") or "").lower()
        if k in ("image.batch.meta", "image.batch"):
            if isinstance(row.get("metadata"), dict):
                return row["metadata"]

    # 3) Last resort: try loading the CAS blob as JSON without enforcing kind
    try:
        return _load_cas_json(run_path, h, kind="image.batch")
    except Exception:
        # give a benign empty descriptor so restore can continue
        return {}

def run_read_images_restore(
    *,
    run_path: Path,
    params: Dict[str, Any],
    realized: Dict[str, Any],
    transform_fn=None,   # unused; signature kept for uniformity
) -> Iterator[Tuple[Any, Dict, bool]]:
    """
    Restore a prior `Extract.read_images` step deterministically.

    This replays decoding and batching using recorded parameters and CAS
    descriptors from a tracked run.

    Parameters
    ----------
    run_path : Path
        Root directory of the recorded run containing CAS blobs.
    params : dict
        Original step parameters captured at record time. Expected keys include
        `backend`, `return_type`, `color`, `max_side`, `mode`, `safety_margin`,
        `max_item_decoded_bytes`, `batch_size`, `target_batch_bytes`,
        `pattern`, `recursive`, `shuffle`, `seed`, and `hash_mode`.
    realized : dict
        Realized outputs from the recorded step. Expected keys:
        - "manifest": str CAS id for the manifest descriptor (optional).
        - "batch_descs" or "batch": list[str] CAS ids for per-batch descriptors.
        - "source": str path to the image directory (preferred).
        If "source" is missing, `params["directory"]` is used.
    transform_fn : callable, optional
        Unused. Present for signature uniformity across restore bindings.

    Yields
    ------
    tuple
        `(images, meta, is_last)` where:
        - `images`: decoded images (type per `params["return_type"]`).
        - `meta`: dict containing `ordinal`, `manifest_root_hash`, `batch_hash`,
          `manifest` (CAS id if available), and `upstream` lineage entries.
        - `is_last`: bool indicating final batch.

    Notes
    -----
    - If a manifest CAS id is provided, the function loads its descriptor and
      enforces that the recomputed root hash matches. Mismatches raise
      `RuntimeError`.
    - If recorded batch descriptors are available, batch count and batch hash
      are validated per batch. Mismatches raise `RuntimeError`.

    Raises
    ------
    RuntimeError
        If neither `realized["source"]` nor `params["directory"]` is provided,
        or if `shuffle=True` without a recorded `seed`, or if manifest or batch
        validation fails.
    """

    # ---------- helpers ----------
    def _load_manifest_desc(run_path: Path, h: str) -> Dict[str, Any]:
        desc = _load_cas_json(run_path, h, kind="image.manifest")
        return desc or {}

    def _try_load_batch_desc(run_path: Path, h: str) -> Dict[str, Any]:
        """
        Accept either kind:
          - image.batch.meta (preferred; full descriptor)
          - image.batch      (fallback; synthesize minimal descriptor)
        """
        try:
            return _load_cas_json(run_path, h, kind="image.batch.meta")
        except FileNotFoundError:
            pass
        # Fallback to image.batch
        try:
            b = _load_cas_json(run_path, h, kind="image.batch")
            # synthesize minimal structure the validator uses
            return {
                "manifest_hash": b.get("manifest_hash"),
                "ordinal": b.get("ordinal"),
                "count": b.get("count"),          # may be absent; validator tolerates None
                "batch_hash": b.get("batch_hash"),
            }
        except FileNotFoundError:
            # Surface a clear error so users know which CAS entry is missing
            raise FileNotFoundError(
                f"CAS blob not found as image.batch.meta or image.batch for hash={h}"
            )

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
    rec_ids = realized.get("batch_descs") or realized.get("batch") or []
    recorded_batch_descs = []
    for h in rec_ids:
        try:
            recorded_batch_descs.append(_try_load_batch_desc(run_path, h))
        except FileNotFoundError:
            pass
    if not recorded_batch_descs:
        recorded_batch_descs = None

    directory = str(realized.get("source") or params.get("directory") or "")
    if not directory:
        raise RuntimeError("read_images restore requires 'source' (directory) or params['directory'].")

    if shuffle and seed is None:
        raise RuntimeError("Recorded run used shuffle=True but no seed; cannot restore deterministically.")

    # ---------- 2) Manifest enforcement (if provided) ----------
    manifest_hash = realized.get("manifest")
    expected_root_hash: Optional[str] = None
    if manifest_hash:
        manifest_desc = _load_manifest_desc(run_path, manifest_hash)
        # support both new (identity.root_hash) and older (root_hash) shapes
        expected_root_hash = (
            (manifest_desc.get("identity") or {}).get("root_hash")
            or manifest_desc.get("root_hash")
        )

    # ---------- 3) Rebuild manifest deterministically ----------
    manifest = scan_manifest_headers(
        directory=directory,
        pattern=pattern,
        recursive=recursive,
        filename_filter=None,
    )
    if shuffle:
        manifest = shuffle_manifest(manifest, seed)

    manifest = ensure_item_content_hashes(manifest)

    root_hash = compute_manifest_root_hash(
        manifest=manifest,
        directory=directory,
        pattern=pattern,
        recursive=recursive,
        seed=seed,
        hash_mode=hash_mode,
    )
    if expected_root_hash and root_hash != expected_root_hash:
        raise RuntimeError(
            f"Restore aborted: current manifest root_hash ({root_hash[:12]}) "
            f"does not match recorded ({expected_root_hash[:12]}). Source changed."
        )

    # ---------- 4) Rebuild batch plan ----------
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

    impl = decode_batch_pil if backend == "pil" else decode_batch_cv2

    # ---------- 5) Decode and yield ----------
    for i, planned in enumerate(iter_batches_from_plan(batch_plan), 1):
        items = planned["items"]
        batch_hash = compute_batch_hash(root_hash, items)

        if items:
            k0 = list(items[0].keys())
            print(f"[READ.ITEMS] n={len(items)} keys0={k0}")
        paths_items = [rec.get("rel_path") for rec in items]
        print(f"[READ.BATCH] #{i} is_last={planned.get('is_last')} "
            f"planned_hash={batch_hash[:12]} count={len(items)} "
            f"items_paths[:3]={paths_items[:3]}")

        rec = None
        if recorded_batch_descs:
            if i - 1 >= len(recorded_batch_descs):
                raise RuntimeError("Recorded batch metas shorter than planned batches; cannot restore.")
            rec = recorded_batch_descs[i - 1]
            rec_count = rec.get("count")
            if isinstance(rec_count, int) and rec_count != len(items):
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

        paths = [it.get("relpath") or it.get("path") for it in items]
        print(f"[READ.BATCH] #{i} is_last={is_last} "
            f"planned_hash={batch_hash[:12]} count={len(items)} "
            f"paths[:3]={paths[:3]}")
        items_rel_paths = [rec.get("rel_path") for rec in items]
        meta = {
            "ordinal": i,
            "manifest": manifest_hash,
            "manifest_root_hash": root_hash,
            "batch_hash": batch_hash,
            "recorded": rec,             # include fields you used at write-time
            #"paths": paths,
            "upstream": [
                {"id": manifest_hash, "role": "manifest"},
                *([{"id": rec.get("batch_desc_hash"), "role": "batch_desc"}] if rec else []),
                {"id": batch_hash, "role": "batch"},
            ],
            #"items_rel_paths": items_rel_paths,
        }
        #print(f"[READ.META]  #{i} meta_paths[:3]={meta['items_rel_paths']} len={len(meta['items_rel_paths'])}")

        yield decoded, meta, is_last

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
    upstream = realized.get("source_gen") or realized.get("source")
    if upstream is None:
        raise RuntimeError("restore: missing realized 'source' or 'source_gen' for Transform.apply_function")

    if isinstance(upstream, (str, os.PathLike)):
        # legacy CSV path → build a 2-tuple generator
        source_path = str(upstream)
        backend    = params.get("backend", "polars")
        batch_size = params.get("batch_size")
        use_cols   = params.get("use_cols")
        filter_by  = params.get("filter_by")
        gen = read_batches_polars(source_path, batch_size, use_cols, filter_by) \
              if backend == "polars" else \
              read_batches_pandas(source_path, batch_size, use_cols, filter_by)
    else:
        gen = upstream  # passthrough

    for item in gen:
        if isinstance(item, tuple):
            if len(item) == 3:
                batch, meta, is_last = item
                out = transform_fn(batch)
                try: in_n=len(batch)
                except: in_n="?"
                try: out_n=len(out)
                except: out_n="?"
                ord_ = (meta or {}).get("ordinal") if (isinstance(item, tuple) and len(item)==3) else "?"
                mpaths = (meta or {}).get("items_rel_paths") if (isinstance(item, tuple) and len(item)==3) else None
                print(f"[XFORM.BATCH] ord={ord_} in={in_n} out={out_n}")
                if mpaths:
                    print(f"[XFORM.PATHS] ord={ord_} first3={mpaths[:3]} len={len(mpaths)}")
                yield transform_fn(batch), meta, is_last
            elif len(item) == 2:
                batch, is_last = item
                yield transform_fn(batch), is_last
            else:
                batch = item[0]
                yield transform_fn(batch), False
        else:
            yield transform_fn(item), False

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
    def _coerce2(gen):
        for item in gen:
            if isinstance(item, tuple):
                if len(item) == 3:
                    b, _meta, last = item
                    yield b, last
                elif len(item) == 2:
                    b, last = item
                    yield b, last
                else:
                    yield item[0], False
            else:
                yield item, False
    ld = Load()
    for (b, last) in _coerce2(upstream_gen):
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

def _deterministic_shard_path(
    target_dir: Path,
    shard_prefix: str,
    ordinal: int,
) -> Path:
    # Matches your runtime shard naming scheme (prefix-000001.parquet)
    return (target_dir / f"{shard_prefix}-{ordinal:06d}.parquet").resolve()

def run_save_images_to_parquet_replay(
    *,
    run_path: Path,
    params: Dict[str, Any],
    realized: Dict[str, Any],           # not used currently, kept for parity
    backend_override: Optional[str] = None,
    upstream_gen: Iterator[Tuple[Any, bool]],
    target_override: Optional[str] = None,   # if provided, use this directory
    on_conflict: str = "overwrite",
    expected_out_hashes: Optional[List[str]] = None,   # optional strict check
    record_materialization: bool = True,
) -> List[Path]:
    """
    Re-run a recorded `Load.save_images_to_parquet` step.

    This consumes an upstream generator of image batches and writes one
    deterministic Parquet shard per batch, verifying optional recorded hashes.

    Parameters
    ----------
    run_path : Path
        Base path of the recorded run.
    params : dict
        Original step parameters captured at record time. Relevant keys include
        `target_dir`, `shard_prefix`, `encode`, `jpeg_quality`, `compression`,
        `include_paths`, and optional `writer_cfg`.
    realized : dict
        Not currently used. Present for API parity with other replay bindings.
    backend_override : str, optional
        Reserved for future backend selection; currently ignored.
    upstream_gen : iterator of tuple
        Upstream generator yielding either:
        - (batch, meta, is_last) or
        - (batch, is_last)
        where `batch` is image data and `meta` is optional batch metadata.
    target_override : str, optional
        Directory to write shards. Overrides both `params["target_dir"]`
        and default restore directory if provided.
    on_conflict : {"overwrite", "rename", "fail"}, default="overwrite"
        File conflict policy applied when writing each shard.
    expected_out_hashes : list of str, optional
        SHA-256 digests of expected shards. If provided, each written shard is
        re-hashed and compared for exact byte equality.
    record_materialization : bool, default=True
        If True, record each written shard in CAS so later restores can skip
        re-materializing identical outputs.

    Returns
    -------
    list of Path
        Absolute paths of all written Parquet shard files, in order.

    Raises
    ------
    IOError
        If a shard's computed hash differs from its recorded expected hash.
    """
    # --------- 1) Resolve sink parameters ----------
    target_dir = (
        Path(target_override).resolve()
        if target_override else
        Path(params.get("target_dir") or RESTORE_DEFAULT_DIR).resolve()
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    shard_prefix  = params.get("shard_prefix", "batch")
    encode        = params.get("encode", "jpeg")
    jpeg_quality  = int(params.get("jpeg_quality", 95))
    compression   = params.get("compression", "zstd")
    include_paths = bool(params.get("include_paths", True))

    # optional: if the recorded step logged batch_desc inputs we can map ordinals
    # to tighten determinism of lineage meta (ordinal, hashes, etc.)
    # We only *read* them to reconstruct meta columns; not required to run.
    # Fetch via step_inputs if you need them here, or let the caller pass them.
    # (Keeping it simple: we recompute ordinals 1..N; _build_parquet_table_from_images
    # uses meta for lineage; if you want strict byte-for-byte equivalence with
    # recorded runs that include lineage cols, enrich 'meta' below from batch_desc.)
    out_paths: List[Path] = []
    hasher = Hash()

    ordinal = 0
    # --------- 2) Stream batches and write shards ----------
    writer_cfg = params.get("writer_cfg") or RECORDED_WRITER_CFG
    import pyarrow as pa, pyarrow.parquet as pq, json, os
    print(f"[ARROW] pyarrow={pa.__version__}")
    print(f"[WRITER.CFG] {json.dumps(writer_cfg, sort_keys=True)}")
    for item in upstream_gen:
        if len(item) == 3:
            batch, meta, is_last = item
            ordinal = meta.get("ordinal") or (ordinal + 1)
        else:
            batch, is_last = item
            ordinal += 1
            meta = {"ordinal": ordinal}
            ordinal += 1

        # Build table exactly like at runtime
        table = _build_parquet_table_from_images(
            data=batch,
            meta=meta,
            encode=encode,
            jpeg_quality=jpeg_quality,
            include_paths=include_paths,
        )

        shard_path = _deterministic_shard_path(target_dir, shard_prefix, ordinal)
        shard_path = _write_parquet_table(table, shard_path, writer_cfg=writer_cfg, on_conflict="overwrite")
        # resolve conflict policy (overwrite/rename/fail) on per-file basis
        shard_path = resolve_conflict_path(
            shard_path,
            policy=on_conflict,
            suffix=".restore",
        )

        # Sanity on meta / ordering
        print(f"[SINK.BATCH] ord={ordinal} rows={len(meta.get('items_rel_paths') or [])}"
            f" writer_cfg={json.dumps(writer_cfg, sort_keys=True)}")

        # Peek at the to-be-written table to confirm content & row order
        print(f"[SINK.TABLE] rows={table.num_rows} cols={table.num_columns} schema={table.schema.names}")
        imgs  = table.column("img_bytes").to_pylist()
        paths = table.column("path").to_pylist() if "path" in table.schema.names else ["<none>"]*table.num_rows
        row_fp = [(hasher.hash_bytes(imgs[k]), paths[k]) for k in range(min(3, len(imgs)))]
        print(f"[SINK.TABLE] ord={ordinal} rows={table.num_rows} cols={table.schema.names}")
        print(f"[SINK.ROWS]  ord={ordinal} first3_fps+paths={row_fp}")
        print(f"[SINK.TABLE] ord={ordinal} rows={table.num_rows} cols={table.schema.names}")

        _write_parquet_table(
            table,
            shard_path,
            writer_cfg=writer_cfg,
            on_conflict="overwrite",  # we already resolved name conflicts above
        )
        import pyarrow.parquet as pq
        h = hasher.hash_file(shard_path)
        pf = pq.ParquetFile(shard_path)

        print(f"[SINK.WRITE] ord={ordinal} out={shard_path.name} hash={h[:16]} "
            f"expected={(expected_out_hashes[ordinal-1][:16] if expected_out_hashes and ordinal-1 < len(expected_out_hashes) else 'None')}")

        md = pq.read_metadata(shard_path)
        print(f"[PARQUET.META] rg_counts={[md.row_group(i).num_rows for i in range(md.num_row_groups)]} created_by={md.created_by}")
        for c in range(md.num_columns):
            cc = md.row_group(0).column(c)
            print(f"[PARQUET.CHUNK] col={table.schema.names[c]} codec={cc.compression} encodings={tuple(cc.encodings)} has_dict={cc.dictionary_page_offset is not None}")
        rgs = [md.row_group(i).num_rows for i in range(md.num_row_groups)]
        print(f"[PARQUET.META] ord={ordinal} rows={md.num_rows} rgs={rgs} created_by={md.created_by}")

        try:
            kv = dict(md.metadata) if md.metadata is not None else {}
            print(f"[PARQUET.KV] {kv}")
        except Exception as e:
            print(f"[PARQUET.KV] <err {e!r}>")

        try:
            for i, name in enumerate(table.schema.names):
                col = md.row_group(0).column(i)
                enc = getattr(col, "encodings", None)
                has_dict = getattr(col, "has_dictionary_page", None)
                print(f"[PARQUET.CHUNK] col={name} codec={col.compression} encodings={enc} has_dict={has_dict}")
        except Exception as e:
            print(f"[PARQUET.CHUNK] <err {e!r}>")

        # Read-back verification (does disk contain exactly what table had?)
        try:
            rb = pq.read_table(shard_path)
            print(f"[READBACK.SCHEMA] {rb.schema}")
            # buffer-level fp for first column
            col0 = rb.column(0).combine_chunks()
            bufs = [b.to_pybytes() for b in col0.buffers() if b is not None]
            print(f"[READBACK.BUFFERS] col0 nbuf={len(bufs)} fps={[Hash().hash_bytes(b)[:12] for b in bufs]}")
            rb_imgs  = rb.column("img_bytes").to_pylist()
            rb_paths = rb.column("path").to_pylist() if "path" in rb.schema.names else ["<none>"]*rb.num_rows
            rb_fp = [(hasher.hash_bytes(rb_imgs[k]), rb_paths[k]) for k in range(min(3, len(rb_imgs)))]
            print(f"[PARQUET.READBACK] ord={ordinal} first3_fps+paths={rb_fp} rows={len(rb_imgs)}")
            if len(rb_imgs) == len(imgs):
                mismatch_ix = next((k for k in range(len(imgs))
                                    if hasher.hash_bytes(imgs[k]) != hasher.hash_bytes(rb_imgs[k]) or paths[k] != rb_paths[k]), None)
                print(f"[PARQUET.CHECK] ord={ordinal} row_mismatch_at={mismatch_ix}")
        except Exception as e:
            print(f"[PARQUET.READBACK] ord={ordinal} failed: {e!r}")

        if expected_out_hashes:
            # If the step recorded N outputs (one per batch), this ensures exact bytes
            h = hasher.hash_file(shard_path)
            # Guard against mismatched counts OR content drift
            if ordinal - 1 < len(expected_out_hashes):
                exp = expected_out_hashes[ordinal - 1]
                if exp and exp != h:
                    raise IOError(
                        f"[restore] parquet shard #{ordinal} hash mismatch: "
                        f"expected={exp[:16]}… got={h[:16]}…"
                    )

        if record_materialization:
            # index materialization so future restores can fast-path
            try:
                store.record_materialization(run_path, hasher.hash_file(shard_path), str(shard_path))
            except Exception:
                pass

        out_paths.append(shard_path)

    return out_paths

# Public registry of handlers (by function FQN)
RESTORE_HANDLERS: Dict[str, Any] = {
    "blase.Extract.read_csv": run_read_csv_restore,
    "blase.Extract.read_images": run_read_images_restore,
    "blase.Transform.apply_function": run_apply_function_restore,
    "blase.Load.save_to_csv": run_save_to_csv_replay,
    "blase.Load.save_images_to_parquet": run_save_images_to_parquet_replay
}