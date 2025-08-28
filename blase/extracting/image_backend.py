import importlib.util
from pathlib import Path
from typing import Callable, List, Optional, Dict, Any, Literal, Iterator
import random

from blase.utils.hashing import Hash

def auto_target_bytes_from_system(return_type: str) -> Optional[int]:
    """
    Infer a safe default target batch size (in decoded bytes) based on
    available system memory (GPU if tensor + CUDA/TF-GPU; otherwise CPU).

    Parameters
    ----------
    return_type : {"np", "pil", "tensor"}
        Desired decoded output:
        - "np" or "pil": plan against CPU memory.
        - "tensor": try GPU first (PyTorch CUDA, then TensorFlow GPU); otherwise CPU.

    Returns
    -------
    target_bytes : int or None
        Suggested soft upper bound (decoded bytes) for each batch.
        Returns None if memory cannot be probed (caller should fall back to a
        conservative default, e.g., 256 MiB).

    Heuristics
    ----------
    GPU (PyTorch):
        - Uses torch.cuda.mem_get_info() → (free, total) bytes
        - target = free * 0.70

    GPU (TensorFlow), preference order:
        1) pynvml (if installed): nvmlDeviceGetMemoryInfo(handle).free
           - target = free * 0.70
        2) TF virtual device memory limit (if configured) minus current allocation:
           - total = tf.config.experimental.get_virtual_device_configuration(gpu)[0].memory_limit (MiB)
           - current = tf.config.experimental.get_memory_info("GPU:0")["current"] (bytes)
           - free ≈ max(0, total_bytes - current)
           - target = free * 0.70

        If neither is available, fall back to CPU path.

    CPU:
        - Requires psutil
        - target = psutil.virtual_memory().available * 0.50

    Clamps
    ------
    The returned value is clamped to [128 MiB, 2 GiB] to avoid degenerate plans.

    Notes
    -----
    - The value is intended as a *planning* budget. You should still apply a
      safety margin when packing (e.g., 10–20%) and validate actual decoded
      memory at runtime.
    - TensorFlow's get_memory_info reports TF allocator usage, not raw device
      free memory; using a TF virtual device memory limit makes the estimate
      reasonable. For raw device memory without PyTorch, consider installing
      `pynvml` (works for both TF and non-ML workloads).
    """
    min_cap = 128 * 1024 * 1024      # 128 MiB
    max_cap = 2 * 1024 * 1024 * 1024 # 2 GiB

    def clamp(v: int) -> int:
        return max(min_cap, min(v, max_cap))

    # ---------------- GPU path preferred when returning tensors ----------------
    if return_type == "tensor":
        # --------- PyTorch CUDA ---------
        if importlib.util.find_spec("torch") is not None:
            import torch
            if torch.cuda.is_available():
                try:
                    free, _total = torch.cuda.mem_get_info()  # bytes
                    return clamp(int(free * 0.70))
                except Exception:
                    # fall through to TF / CPU
                    pass

        # --------- TensorFlow GPU ---------
        if importlib.util.find_spec("tensorflow") is not None:
            import tensorflow as tf
            try:
                gpus = tf.config.list_physical_devices("GPU")
            except Exception:
                gpus = []

            if gpus:
                # 1) Try pynvml for true device free memory
                if importlib.util.find_spec("pynvml") is not None:
                    try:
                        import pynvml  # provided by nvidia-ml-py3
                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        free = info.free  # bytes
                        pynvml.nvmlShutdown()
                        return clamp(int(free * 0.70))
                    except Exception:
                        pass

                # 2) TF-only estimate: virtual device memory limit - current allocation
                try:
                    # Note: virtual device configuration is present only if user set a limit.
                    vcfg = tf.config.experimental.get_virtual_device_configuration(gpus[0])
                    if vcfg and len(vcfg) > 0 and getattr(vcfg[0], "memory_limit", None):
                        total_bytes = int(vcfg[0].memory_limit) * 1024 * 1024  # MiB -> bytes
                        # current allocated per TF allocator (bytes)
                        get_mem = getattr(tf.config.experimental, "get_memory_info", None)
                        current = 0
                        if callable(get_mem):
                            try:
                                meminfo = get_mem("GPU:0")  # {'current':..., 'peak':...}
                                current = int(meminfo.get("current", 0))
                            except Exception:
                                current = 0
                        free_est = max(0, total_bytes - current)
                        return clamp(int(free_est * 0.70))
                except Exception:
                    # Fall through to CPU
                    pass

    # ---------------- CPU path ----------------
    if importlib.util.find_spec("psutil") is not None:
        try:
            import psutil
            avail = int(psutil.virtual_memory().available)  # bytes
            return clamp(int(avail * 0.50))
        except Exception:
            pass

    # ---------------- fallback ----------------
    return None

def scan_manifest_headers(
    directory: str,
    pattern: str,
    recursive: bool,
    filename_filter: Optional[Callable[[str], bool]],
) -> List[Dict[str, Any]]:
    """
    Scan a directory for images and collect **header-only** metadata for each file.

    This function builds the per-item records that power memory-aware batching,
    integrity checks, and deterministic replay. It **does not** load pixel data.

    Parameters
    ----------
    directory : str
        Root directory to scan.
    pattern : str
        Glob pattern for matching files (e.g., ``"**/*.jpg"``, ``"*.png"``).
        If ``recursive`` is False and the pattern includes ``"**"``, the double-star
        segments are stripped to avoid unintended recursion.
    recursive : bool
        Whether to recurse into subdirectories. If False, only the top-level
        directory is scanned.
    filename_filter : callable, optional
        Optional predicate ``f(path: str) -> bool`` applied **after** globbing.
        If provided and it returns False, the file is skipped (useful to exclude
        thumbnails, hidden files, etc.).

    Returns
    -------
    items : list of dict
        One dict per discovered file, sorted deterministically by ``rel_path``.
        Each dict contains:

        - ``abs_path`` : str  
          Absolute filesystem path.
        - ``rel_path`` : str  
          Path relative to ``directory`` for cross-machine stability.
        - ``ext`` : str  
          Lowercased extension without the dot (e.g., ``"jpg"``).
        - ``bytes`` : int  
          File size in bytes (from ``stat``), or ``None`` if unavailable.
        - ``mtime`` : int  
          Last modification time (seconds since epoch, int), or ``None`` if unavailable.
        - ``width`` : int or None  
          Image width from header; ``None`` if unreadable.
        - ``height`` : int or None  
          Image height from header; ``None`` if unreadable.
        - ``mode`` : str or None  
          Pillow mode (``"RGB"``, ``"L"``, ``"RGBA"``, etc.), or ``None``.
        - ``channels`` : int or None  
          Inferred from mode (``L→1``, ``RGB→3``, ``RGBA→4``), else ``None``.
        - ``exif_orientation`` : int or None  
          EXIF orientation (1..8) if present; ``1`` commonly means “upright”.
        - ``is_corrupt`` : bool  
          True if header open failed or file unreadable as an image.
        - ``estimated_decoded_bytes`` : int  
          Heuristic decoded footprint in bytes = ``width * height * channels``
          (assuming ``uint8``); uses 0 for unknown fields.
        - ``hash_path_mtime`` : str  
          **Fast** change detector: SHA-256 of ``rel_path|mtime|bytes`` (hex).

    Notes
    -----
    - Requires Pillow. Only headers are read: ``Image.open(...).size`` and ``mode``
      do not trigger full pixel decoding.
    - Files are **sorted by rel_path** for deterministic ordering across runs/machines.
    - Corrupt/unreadable files are not dropped; they’re included with
      ``is_corrupt=True`` and missing header fields, so downstream health checks
      can report them deterministically.
    - For **strong identity** later, compute ``hash_content`` (full-file SHA-256)
      in a separate step and augment these records (or replace the fast hash).

    Examples
    --------
    >>> items = scan_manifest_headers("data/radish", "**/*.jpg", recursive=True, filename_filter=None)
    >>> items[0]["rel_path"], items[0]["width"], items[0]["is_corrupt"]
    ('img/000001.jpg', 1024, False)
    """
    # --- dependency check (Pillow) ---
    if importlib.util.find_spec("PIL") is None:
        raise RuntimeError("Pillow is required for header scans. Install with: pip install pillow")
    from PIL import Image, UnidentifiedImageError  # imported after check

    # --- normalize inputs ---
    root = Path(directory).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory!r}")

    patt = pattern or "*"
    if not recursive and "**" in patt:
        # Remove recursive segments to honor the flag
        patt = patt.replace("**/", "").replace("**", "")

    # --- enumerate files deterministically ---
    def _iter_paths() -> List[Path]:
        if recursive:
            # rglob handles "**" naturally, but also works with simple patterns
            return sorted([p for p in root.rglob(patt) if p.is_file()])
        else:
            return sorted([p for p in root.glob(patt) if p.is_file()])

    paths: List[Path] = _iter_paths()
    items: List[Dict[str, Any]] = []

    # --- per-file scan (header only) ---
    def _fast_hash(rel_path: str, mtime: Optional[int], size: Optional[int]) -> str:
        """
        Generate a fast, deterministic hash for change detection
        based on path, mtime, and size.
        """
        payload = f"{rel_path}|{mtime or -1}|{size or -1}".encode("utf-8")
        hasher = Hash()
        return hasher.hash_bytes(payload)

    # map Pillow modes to channel counts
    MODE_TO_CHANNELS = {"1": 1, "L": 1, "P": 1, "RGB": 3, "RGBA": 4, "CMYK": 4, "YCbCr": 3}

    for p in paths:
        # filename filter (post-glob)
        if filename_filter is not None and not filename_filter(str(p)):
            continue

        try:
            st = p.stat()
            size = int(st.st_size)
            mtime = int(st.st_mtime)
        except OSError:
            size, mtime = None, None

        rec: Dict[str, Any] = {
            "abs_path": str(p),
            "rel_path": str(p.relative_to(root)) if p.is_relative_to(root) else p.as_posix(),
            "ext": p.suffix.lower().lstrip("."),
            "bytes": size,
            "mtime": mtime,
            "width": None,
            "height": None,
            "mode": None,
            "channels": None,
            "exif_orientation": None,
            "is_corrupt": False,
            "estimated_decoded_bytes": 0,
            "hash_path_mtime": None,
        }

        # header read (without full decode)
        try:
            with Image.open(p) as im:
                w, h = im.size  # header
                mode = im.mode
                # EXIF orientation (0x0112). Use 1 if absent to denote "upright".
                exif_orientation = None
                try:
                    exif = getattr(im, "getexif", lambda: {})()
                    if exif:
                        exif_orientation = int(exif.get(0x0112, 1))
                except Exception:
                    exif_orientation = None

                rec["width"] = int(w)
                rec["height"] = int(h)
                rec["mode"] = mode
                rec["channels"] = MODE_TO_CHANNELS.get(mode, None)
                rec["exif_orientation"] = exif_orientation
                rec["is_corrupt"] = False

        except (UnidentifiedImageError, OSError):
            # unreadable / corrupt
            rec["is_corrupt"] = True

        # estimated decoded footprint (uint8 assumption)
        w = rec["width"] or 0
        h = rec["height"] or 0
        c = rec["channels"] or 3  # assume RGB if unknown
        rec["estimated_decoded_bytes"] = int(w * h * c)

        # fast change detector
        rec["hash_path_mtime"] = _fast_hash(rec["rel_path"], rec["mtime"], rec["bytes"])

        items.append(rec)

    # deterministic order
    items.sort(key=lambda r: r["rel_path"])
    return items

def shuffle_manifest(manifest: List[Dict[str, Any]], seed: Optional[int]) -> List[Dict[str, Any]]:
    """
    Shuffle a manifest deterministically if a seed is provided.

    This function reorders the list of manifest records (as produced by
    ``scan_manifest_headers``) in a reproducible way, based on the given
    seed. If ``seed`` is None, the manifest is returned unchanged.

    Parameters
    ----------
    manifest : list of dict
        List of item records (each a dict with at least a "rel_path" key).
        Typically produced by ``scan_manifest_headers``.
    seed : int or None
        RNG seed. If provided, the manifest is shuffled deterministically
        using the Python stdlib RNG seeded with this value. If None, the
        manifest is left as-is.

    Returns
    -------
    shuffled : list of dict
        New list of manifest records. Same contents as input, but possibly
        reordered.

    Notes
    -----
    - The input manifest is **not modified in place**; a shallow copy is made.
    - Determinism: given the same manifest and seed, the output order is
      identical across runs and machines (uses Python's ``random`` not NumPy).
    - If you need stronger guarantees (e.g., cross-language reproducibility),
      use ``hash_object`` on the manifest list and store the seed in the
      manifest descriptor.
    - For resume/replay, the chosen ``seed`` should be logged alongside the
      manifest root hash in step metadata.

    Examples
    --------
    >>> manifest = [{"rel_path": "a.jpg"}, {"rel_path": "b.jpg"}, {"rel_path": "c.jpg"}]
    >>> shuffle_manifest(manifest, seed=42)
    [{'rel_path': 'b.jpg'}, {'rel_path': 'c.jpg'}, {'rel_path': 'a.jpg'}]
    """
    if seed is None:
        return manifest  # leave order intact

    rng = random.Random(seed)
    shuffled = manifest.copy()
    rng.shuffle(shuffled)
    return shuffled

def ensure_item_content_hashes(manifest: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ensure each manifest item has a strong content hash.

    This function iterates over header-scanned manifest records (as produced by
    ``scan_manifest_headers``) and fills the ``hash_content`` field by hashing the
    **entire file bytes**. Existing non-empty ``hash_content`` values are left
    unchanged to allow upstream caching.

    Parameters
    ----------
    manifest : list of dict
        Each dict is expected to contain at least:
        - ``abs_path`` : absolute path to the file (str)
        - ``rel_path`` : path relative to the scanned root (str)
        - ``is_corrupt`` : bool (from header scan; corrupt files can still be hashed)
        - ``hash_path_mtime`` : fast change detector (str), optional

        Other fields (width, height, mode, etc.) are ignored here.

    Returns
    -------
    updated : list of dict
        The same list object (mutated in place) is also returned for convenience.
        For each item:
        - ``hash_content`` : str
            Hex digest computed from the full file bytes, if hashing succeeds.
            Left as-is if already present and non-empty.
            Set to ``None`` if hashing fails due to I/O errors.

    Notes
    -----
    - Corrupt images (unreadable by Pillow) can still be hashed at the byte level;
      this helps you separate "corrupt for decoding" from "file bytes changed".
    - If you maintain a cache keyed by ``hash_path_mtime``, you can pre-populate
      ``hash_content`` before calling this function to avoid rehashing unchanged files.

    Examples
    --------
    >>> items = scan_manifest_headers("data/imgs", "**/*.jpg", True, None)
    >>> ensure_item_content_hashes(items)  # fills item["hash_content"]
    >>> items[0]["hash_content"][:8]
    '4f9c1a2b'
    """
    for rec in manifest:
        # Skip if already present and non-empty
        if rec.get("hash_content"):
            continue

        abs_path = rec.get("abs_path")
        if not abs_path:
            rec["hash_content"] = None
            continue

        p = Path(abs_path)
        try:
            # Read whole file into memory and hash with the library helper
            # (If you add a streaming hasher later, swap this out.)
            hasher = Hash()
            rec["hash_content"] = hasher.hash_file(p)
        except Exception:
            # I/O issues (missing permissions, removed file, etc.)
            rec["hash_content"] = None

    return manifest

def compute_manifest_root_hash(
    manifest: List[Dict[str, Any]],
    directory: str,
    pattern: str,
    recursive: bool,
    seed: Optional[int],
    hash_mode: str = "content",
) -> str:
    """
    Compute a deterministic root hash for an image manifest.

    The root hash acts as a single identifier for the *entire* dataset state.
    It is computed from:
      1) scan parameters (directory, pattern, recursive, seed, hash_mode), and
      2) the ordered list of per-item identities:
         - in ``hash_mode="content"``: (rel_path, hash_content)
         - in ``hash_mode="path_mtime_size"``: (rel_path, hash_path_mtime)

    Ordering is by ``rel_path`` to ensure stability across machines.

    Parameters
    ----------
    manifest : list of dict
        Records produced by the header scan (and possibly augmented with
        content hashes). Each item must contain:
          - ``rel_path`` : str
          - For ``hash_mode="content"``: ``hash_content`` : str (hex digest)
          - For ``hash_mode="path_mtime_size"``: ``hash_path_mtime`` : str
        Other fields are ignored here.
    directory : str
        Root directory that was scanned (used for identity).
    pattern : str
        Glob pattern used during the scan (part of identity).
    recursive : bool
        Whether recursion was enabled (part of identity).
    seed : int or None
        Shuffle seed applied to the manifest order before batching (logged
        as part of identity; ``None`` is treated distinctly from any integer).
    hash_mode : {"content", "path_mtime_size"}, default "content"
        Identity mode for items. Use:
          - "content" for strong identity (SHA-256 of file bytes).
          - "path_mtime_size" for a fast, weaker identity.

    Returns
    -------
    root_hash : str
        Hex digest string representing the entire dataset state.

    Raises
    ------
    ValueError
        If required fields are missing for the requested ``hash_mode`` or if
        an unrecognized ``hash_mode`` is provided.

    Notes
    -----
    - The manifest list is not modified. It is *read* and canonicalized into
      a small, pickle-serializable structure which is then hashed via your
      centralized ``hash_object`` helper, ensuring consistent hashing across
      the codebase.
    - Include **all** items, even corrupt ones. Corruption affects decoding,
      not the byte identity; in strong mode you can still hash the bytes.
    - Sorting by ``rel_path`` guarantees stability regardless of how the OS
      yields files. If your upstream already sorted, this remains stable.

    Examples
    --------
    >>> items = [
    ...   {"rel_path": "a.jpg", "hash_content": "aa..."},
    ...   {"rel_path": "b.jpg", "hash_content": "bb..."},
    ... ]
    >>> compute_manifest_root_hash(items, "data/imgs", "**/*.jpg", True, 42, "content")[:8]
    '9c41f2d0'
    """
    mode = (hash_mode or "content").strip().lower()
    if mode not in ("content", "path_mtime_size"):
        raise ValueError(
            f"Unsupported hash_mode: {hash_mode!r}. "
            "Use 'content' or 'path_mtime_size'."
        )

    # Canonicalize scan params (strings, bools, and simple types only).
    root = str(Path(directory).expanduser().resolve())  # stable absolute path string
    patt = pattern or ""

    # Build the ordered item identity list.
    # Each entry is a 2-tuple: (rel_path, item_hash)
    try:
        items = []
        # Sort deterministically by rel_path (POSIX-like)
        for rec in sorted(manifest, key=lambda r: str(r.get("rel_path", ""))):
            rel = str(rec.get("rel_path", ""))
            if mode == "content":
                h = rec.get("hash_content")
                if not h:
                    raise ValueError(
                        "compute_manifest_root_hash(hash_mode='content') requires "
                        "each manifest item to have 'hash_content'. Missing for: "
                        f"{rel!r}"
                    )
            else:  # path_mtime_size
                h = rec.get("hash_path_mtime")
                if not h:
                    raise ValueError(
                        "compute_manifest_root_hash(hash_mode='path_mtime_size') requires "
                        "each manifest item to have 'hash_path_mtime'. Missing for: "
                        f"{rel!r}"
                    )
            items.append((rel, str(h)))
    except KeyError as e:
        raise ValueError(f"Manifest record missing required key: {e!s}") from e

    # Construct a compact, versioned identity object.
    # Keep it simple and pickle-serializable for your Hash.hash_object().
    identity_obj = {
        "type": "image.manifest:v1",
        "scan": {
            "directory": root,
            "pattern": patt,
            "recursive": bool(recursive),
            "seed": int(seed) if isinstance(seed, int) else None,
            "hash_mode": mode,
        },
        "items": items,  # ordered list of (rel_path, item_hash)
    }

    # Centralized hashing
    hasher = Hash()
    root_hash = hasher.hash_object(identity_obj)
    return root_hash

def plan_image_batches(
    manifest: List[Dict[str, Any]],
    mode: Literal["auto", "manual"],
    batch_size: Optional[int],
    target_batch_bytes: Optional[int],
    safety_margin: float,
    max_item_decoded_bytes: Optional[int],
    max_side: Optional[int],
) -> List[Dict[str, Any]]:
    """
    Plan image batches from a manifest for memory-aware decoding.

    Parameters
    ----------
    manifest : list of dict
        Records from `scan_manifest_headers(...)` (optionally augmented) in the
        *final* iteration order (already shuffled if applicable). Each record
        should include at least:
          - "width", "height" (ints or None)
          - "channels" (int or None)
          - "estimated_decoded_bytes" (int; optional, will be recomputed if missing)
          - "abs_path" / "rel_path" (for reference; not used in sizing)
        Corrupt items are not excluded here; they’ll have zero/unknown sizing and
        should be handled downstream at decode time (skip or raise).

    mode : {"auto","manual"}
        - "auto": pack items until decoded-bytes budget is reached.
        - "manual": fixed item count per batch.
        In both modes, if `batch_size` is provided, it acts as a *hard ceiling*
        on the number of items in a batch.

    batch_size : int, optional
        Maximum number of items per batch. Required when `mode="manual"`. In
        `mode="auto"`, this is an optional hard ceiling.

    target_batch_bytes : int, optional
        Soft budget (decoded bytes) per batch. Required when `mode="auto"`
        (the caller should have inferred a default if None). Ignored in pure
        `mode="manual"` packing.

    safety_margin : float
        Fraction (e.g., 0.15) of the target budget to hold as headroom. The
        effective planning cap is:
            cap = int(target_batch_bytes * (1 - safety_margin))

    max_item_decoded_bytes : int, optional
        Guardrail for single-item size. If an item's estimated decoded bytes
        exceeds this value:
          - if `max_side` is provided and would downscale the item, that scaled
            estimate is used instead;
          - otherwise the item is planned as a single-item batch (so progress is
            guaranteed). Decode-time logic can still decide to downscale/skip.

    max_side : int, optional
        If provided, the estimator accounts for downscaling the longer side to
        `max_side` (aspect-preserving) when computing per-item decoded bytes.

    Returns
    -------
    plan : list of dict
        Each element represents a batch:
          - "items": list[dict]  (subset of `manifest` records, in order)
          - "est_decoded_bytes": int  (sum of per-item estimates post-scaling)
          - "is_last": bool  (True only for the final batch)

    Notes
    -----
    - Greedy first-fit packing in input order for determinism.
    - Per-item estimate uses:
         est = width * height * channels * 1  (uint8 bytes per pixel),
      adjusted for `max_side` downscale when specified. If fields are missing,
      falls back to `record.get("estimated_decoded_bytes", 0)` or 0.
    - This function *plans* batches; decode-time verification should still
      enforce caps and log any corrective actions (e.g., downscale, isolate).
    """
    if mode not in ("auto", "manual"):
        raise ValueError(f"Unsupported mode: {mode!r}")

    if mode == "manual":
        if not batch_size or batch_size <= 0:
            raise ValueError("When mode='manual', batch_size must be a positive integer.")
    else:  # auto
        if not target_batch_bytes or target_batch_bytes <= 0:
            raise ValueError("When mode='auto', target_batch_bytes must be a positive integer.")
        if not (0.0 <= safety_margin < 1.0):
            raise ValueError("safety_margin must be in [0.0, 1.0).")

    def _scaled_estimate(rec: Dict[str, Any]) -> int:
        """Estimate decoded bytes, considering optional max_side downscale."""
        w = rec.get("width") or 0
        h = rec.get("height") or 0
        c = rec.get("channels") or 3
        if w <= 0 or h <= 0 or c <= 0:
            # fallback to any precomputed estimate; else 0
            return int(rec.get("estimated_decoded_bytes") or 0)

        if isinstance(max_side, int) and max_side > 0:
            m = max(w, h)
            if m > max_side:
                scale = max_side / float(m)
                w = int(w * scale)
                h = int(h * scale)

        est = int(w) * int(h) * int(c)
        # Apply item-level guardrail
        if isinstance(max_item_decoded_bytes, int) and max_item_decoded_bytes > 0:
            if est > max_item_decoded_bytes:
                # Cap estimate; decode step should still enforce/adjust
                est = max_item_decoded_bytes
        return int(est)

    n = len(manifest)
    if n == 0:
        return []

    plan: List[Dict[str, Any]] = []

    if mode == "manual":
        # Fixed-size chunks by count; enforce hard ceiling if batch_size provided.
        bs = batch_size  # already validated
        for start in range(0, n, bs):
            chunk = manifest[start : start + bs]
            total = sum(_scaled_estimate(rec) for rec in chunk)
            plan.append({
                "items": chunk,
                "est_decoded_bytes": int(total),
                "is_last": (start + bs) >= n,
            })
        return plan

    # mode == "auto" (bytes-aware greedy packing)
    cap = int(target_batch_bytes * (1.0 - safety_margin))
    if cap <= 0:
        raise ValueError("Effective planning cap (after safety_margin) must be positive.")

    cur_items: List[Dict[str, Any]] = []
    cur_total = 0

    def _flush(is_last_flag: bool = False):
        nonlocal cur_items, cur_total
        if cur_items:
            plan.append({
                "items": cur_items,
                "est_decoded_bytes": int(cur_total),
                "is_last": is_last_flag,
            })
            cur_items, cur_total = [], 0

    for idx, rec in enumerate(manifest, 1):
        est = _scaled_estimate(rec)

        # If we already have items and adding this one would exceed cap,
        # or we'd exceed the optional hard count ceiling, flush first.
        would_exceed_bytes = (cur_total + est) > cap if cap > 0 else False
        would_exceed_count = (batch_size is not None and len(cur_items) >= batch_size)

        if cur_items and (would_exceed_bytes or would_exceed_count):
            _flush(is_last_flag=False)

        # Ensure progress for a single huge item:
        # if est > cap and batch is empty, accept it as a single-item batch.
        if not cur_items and (est > cap) and (batch_size is None or batch_size > 0):
            # Single-item batch for this oversized record
            plan.append({
                "items": [rec],
                "est_decoded_bytes": int(est),
                "is_last": False,  # may be updated below if it happens to be last
            })
            continue

        # Add to current batch
        cur_items.append(rec)
        cur_total += est

        # If hard ceiling reached by count exactly, flush immediately
        if batch_size is not None and len(cur_items) >= batch_size:
            _flush(is_last_flag=False)

    # Flush any remainder and mark the last batch
    if cur_items:
        # mark last batch; we’ll correct previous "is_last" flags next
        plan.append({
            "items": cur_items,
            "est_decoded_bytes": int(cur_total),
            "is_last": True,
        })

    # Ensure only the final batch has is_last=True
    if plan:
        for b in plan[:-1]:
            b["is_last"] = False
        plan[-1]["is_last"] = True

    return plan

def decode_batch_pil(
    items: List[Dict[str, Any]],
    return_type: Literal["np", "pil", "tensor"] = "np",
    color: Literal["rgb", "gray"] = "rgb",
    max_side: Optional[int] = None,
):
    """
    Decode a batch of images using Pillow (header-then-decode), with optional
    EXIF orientation fix, color conversion, and size capping.

    Parameters
    ----------
    items : list of dict
        Subset of manifest records for this batch. Each item must contain:
        - "abs_path" : str (absolute path to the file)
        Other keys (width/height/mode/is_corrupt, etc.) are ignored here.
    return_type : {"np", "pil", "tensor"}, default="np"
        Decoded output container:
        - "np"     : list of numpy arrays (dtype=uint8), shape HxWxC (RGB) or HxW (gray)
        - "pil"    : list of PIL.Image objects
        - "tensor" : list of torch tensors (dtype=uint8), shape like numpy case
                     (torch is imported lazily; if unavailable, raises RuntimeError)
        Note: images in a batch may have different spatial sizes; a single stacked
        array/tensor is not returned.
    color : {"rgb", "gray"}, default="rgb"
        Target color space. Converts via Pillow:
        - "rgb"  → mode "RGB"
        - "gray" → mode "L"
    max_side : int, optional
        If provided and the image's longer side exceeds this value, downscale
        proportionally so that max(width, height) == max_side (aspect preserved).
        Uses high-quality resampling.

    Returns
    -------
    decoded : list
        List of decoded images in the requested representation. Decoding failures
        (e.g., corrupt files) yield `None` at the corresponding position so the
        caller can decide to skip or fail fast.

    Notes
    -----
    - This function performs **pixel decoding** (unlike the header scan). It does
      not perform any CAS/DB logging and is designed to be pure/deterministic.
    - EXIF orientation is normalized via `ImageOps.exif_transpose`.
    - Resampling uses LANCZOS where available for downscaling.
    - For `"tensor"`, tensors are `uint8` to mirror the numpy path and conserve
      memory; downstream transforms can convert to float/normalize as needed.

    Examples
    --------
    >>> decoded = decode_batch_pil(batch_items, return_type="np", color="rgb", max_side=1024)
    >>> type(decoded[0]).__name__
    'ndarray'
    """
    # Lazy imports so the module doesn't hard-require heavy deps
    import importlib.util
    if importlib.util.find_spec("PIL") is None:
        raise RuntimeError("Pillow is required to decode images with backend 'pil'. Install with: pip install pillow")
    from PIL import Image, ImageOps

    to_numpy = (return_type == "np")
    to_pil    = (return_type == "pil")
    to_tensor = (return_type == "tensor")

    if to_tensor:
        if importlib.util.find_spec("torch") is None:
            raise RuntimeError("return_type='tensor' requires PyTorch. Install with: pip install torch")
        import torch

    if to_numpy:
        if importlib.util.find_spec("numpy") is None:
            raise RuntimeError("return_type='np' requires NumPy. Install with: pip install numpy")
        import numpy as np  # type: ignore

    out: List[Any] = []
    target_mode = "RGB" if color == "rgb" else "L"

    # Choose a high-quality resampler if available (LANCZOS best for downscale)
    try:
        resample = Image.Resampling.LANCZOS  # Pillow ≥9.1
    except Exception:
        resample = Image.LANCZOS  # older Pillow fallback

    for rec in items:
        path = rec.get("abs_path")
        if not path:
            out.append(None)
            continue

        try:
            with Image.open(path) as im:
                # Normalize EXIF orientation (no-op if absent)
                im = ImageOps.exif_transpose(im)

                # Color conversion
                if im.mode != target_mode:
                    im = im.convert(target_mode)

                # Optional downscale to cap the longer side
                if isinstance(max_side, int) and max_side > 0:
                    w, h = im.size
                    m = max(w, h)
                    if m > max_side:
                        scale = max_side / float(m)
                        new_w = max(1, int(round(w * scale)))
                        new_h = max(1, int(round(h * scale)))
                        im = im.resize((new_w, new_h), resample=resample)

                # Materialize to desired output type
                if to_pil:
                    out.append(im.copy())
                elif to_numpy:
                    # np array in HxWxC (RGB) or HxW (L); dtype=uint8
                    arr = np.array(im, dtype=np.uint8, copy=True)
                    out.append(arr)
                else:  # to_tensor
                    # uint8 tensor; HxWxC or HxW
                    t = torch.from_numpy(np.array(im, dtype=np.uint8, copy=True))
                    out.append(t)

        except Exception:
            # Unreadable/corrupt file or I/O error: append None placeholder
            out.append(None)

    return out

def decode_batch_cv2(
    items: List[Dict[str, Any]],
    return_type: Literal["np", "pil", "tensor"] = "np",
    color: Literal["rgb", "gray"] = "rgb",
    max_side: Optional[int] = None,
):
    """
    Decode a batch of images using OpenCV (cv2) with optional color conversion
    and size capping.

    Parameters
    ----------
    items : list of dict
        Subset of manifest records for this batch. Each item must contain:
        - "abs_path" : str (absolute path to the file)
        Other keys (width/height/mode/is_corrupt, etc.) are ignored here.
    return_type : {"np", "pil", "tensor"}, default="np"
        Decoded output container:
        - "np"     : list of numpy arrays (dtype=uint8), shape HxWxC (RGB) or HxW (gray)
        - "pil"    : list of PIL.Image objects (requires Pillow)
        - "tensor" : list of torch tensors (dtype=uint8), shape as above (requires PyTorch)
        Note: images in a batch may have different spatial sizes; a single stacked
        array/tensor is not returned.
    color : {"rgb", "gray"}, default="rgb"
        Target color space:
        - "rgb"  → outputs 3-channel RGB arrays
        - "gray" → outputs single-channel grayscale arrays
    max_side : int, optional
        If provided and the image's longer side exceeds this value, downscale
        proportionally so that max(width, height) == max_side (aspect preserved).
        Uses `INTER_AREA` resampling for downscale.

    Returns
    -------
    decoded : list
        List of decoded images in the requested representation. Decoding failures
        (e.g., unreadable/corrupt files) yield `None` at the corresponding position.

    Notes
    -----
    - OpenCV loads images as **BGR** by default and does **not** honor EXIF
      orientation. If you require EXIF normalization, prefer the PIL backend.
    - Downscaling uses `cv2.INTER_AREA`, generally best for shrinking.
    - For `"tensor"`, tensors are `uint8` to match the numpy path; downstream
      transforms can convert to float/normalize as needed.
    - This function performs pixel decoding only; no CAS/DB side effects.

    Examples
    --------
    >>> decoded = decode_batch_cv2(batch_items, return_type="np", color="rgb", max_side=1024)
    >>> decoded[0].dtype, decoded[0].shape
    (dtype('uint8'), (720, 1280, 3))
    """
    # Required deps (lazy)
    if importlib.util.find_spec("cv2") is None:
        raise RuntimeError("OpenCV is required for decode_batch_cv2. Install with: pip install opencv-python")
    import cv2  # type: ignore

    to_numpy = (return_type == "np")
    to_pil    = (return_type == "pil")
    to_tensor = (return_type == "tensor")

    if to_pil:
        if importlib.util.find_spec("PIL") is None:
            raise RuntimeError("return_type='pil' requires Pillow. Install with: pip install pillow")
        from PIL import Image  # type: ignore

    if to_tensor:
        if importlib.util.find_spec("torch") is None:
            raise RuntimeError("return_type='tensor' requires PyTorch. Install with: pip install torch")
        import torch  # type: ignore

    if to_numpy or to_tensor or to_pil:
        if importlib.util.find_spec("numpy") is None:
            raise RuntimeError("NumPy is required for this decoder path. Install with: pip install numpy")
        import numpy as np  # type: ignore

    out: List[Any] = []

    for rec in items:
        path = rec.get("abs_path")
        if not path:
            out.append(None)
            continue

        try:
            # Read image (as-is, color conversion handled below)
            if color == "gray":
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # HxW
                if img is None:
                    out.append(None)
                    continue
                # Downscale if needed
                if isinstance(max_side, int) and max_side > 0:
                    h, w = img.shape[:2]
                    m = max(h, w)
                    if m > max_side:
                        scale = max_side / float(m)
                        new_w = max(1, int(round(w * scale)))
                        new_h = max(1, int(round(h * scale)))
                        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                # Convert to requested return_type
                if to_pil:
                    out.append(Image.fromarray(img, mode="L"))
                elif to_tensor:
                    out.append(torch.from_numpy(img.copy()))
                else:
                    out.append(img.copy())

            else:  # color == "rgb"
                img = cv2.imread(path, cv2.IMREAD_COLOR)  # HxWx3, BGR
                if img is None:
                    out.append(None)
                    continue
                # Convert BGR -> RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Downscale if needed
                if isinstance(max_side, int) and max_side > 0:
                    h, w = img.shape[:2]
                    m = max(h, w)
                    if m > max_side:
                        scale = max_side / float(m)
                        new_w = max(1, int(round(w * scale)))
                        new_h = max(1, int(round(h * scale)))
                        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                # Convert to requested return_type
                if to_pil:
                    out.append(Image.fromarray(img, mode="RGB"))
                elif to_tensor:
                    out.append(torch.from_numpy(img.copy()))
                else:
                    out.append(img.copy())

        except Exception:
            out.append(None)

    return out

def iter_batches_from_plan(batch_plan: List[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
    """
    Iterate over a batch plan produced by `plan_image_batches`.

    This generator yields each batch dict in order and enforces a clean contract:
    - Each yielded element has the keys: {"items", "est_decoded_bytes", "is_last"}.
    - Exactly one batch (the final one) has `is_last=True`. If the incoming plan
      marks multiple batches as last (or none), this function corrects it.

    Parameters
    ----------
    batch_plan : list of dict
        Output from `plan_image_batches(...)`. Each element should be a dict with:
          - "items" : list[dict]     (subset of manifest records)
          - "est_decoded_bytes" : int
          - "is_last" : bool

    Yields
    ------
    batch : dict
        The (possibly normalized) batch dict.

    Raises
    ------
    ValueError
        If a batch is missing required keys or has invalid types.

    Notes
    -----
    - This function does not mutate `batch_plan` in place; it yields shallow copies.
    - If `batch_plan` is empty, nothing is yielded.
    - Normalization ensures downstream consumers can rely on a single terminal `is_last=True`.

    Examples
    --------
    >>> plan = [
    ...   {"items": [1,2], "est_decoded_bytes": 123, "is_last": False},
    ...   {"items": [3],   "est_decoded_bytes": 45,  "is_last": True},
    ... ]
    >>> list(iter_batches_from_plan(plan))[-1]["is_last"]
    True
    """
    if not batch_plan:
        return
        yield  # pragma: no cover (keep as generator even on early return)

    # Find index of the last non-empty entry to mark as final.
    last_idx = len(batch_plan) - 1

    for i, b in enumerate(batch_plan):
        if not isinstance(b, dict):
            raise ValueError(f"Batch plan entry {i} must be a dict, got {type(b).__name__}.")

        # Validate required keys
        for k in ("items", "est_decoded_bytes", "is_last"):
            if k not in b:
                raise ValueError(f"Batch plan entry {i} missing required key: {k!r}.")

        items = b["items"]
        est = b["est_decoded_bytes"]
        is_last = bool(b["is_last"])

        if not isinstance(items, list):
            raise ValueError(f"Batch plan entry {i} 'items' must be a list, got {type(items).__name__}.")
        if not isinstance(est, int):
            # allow ints that arrive as bools? no—force int to catch bugs early
            raise ValueError(f"Batch plan entry {i} 'est_decoded_bytes' must be int, got {type(est).__name__}.")

        # Normalize final flag: only the last entry should be True
        norm_is_last = (i == last_idx)

        yield {
            "items": items,
            "est_decoded_bytes": est,
            "is_last": norm_is_last if is_last != norm_is_last else is_last,
        }

def compute_batch_hash(root_hash: str, items: List[Dict[str, Any]]) -> str:
    """
    Compute a deterministic hash for a batch of items within a dataset.

    The batch hash ties a particular batch to its parent dataset by combining:
      - The dataset `root_hash` (already computed from the full manifest).
      - The ordered list of per-item identities from this batch.

    Parameters
    ----------
    root_hash : str
        Strong hash string identifying the entire dataset (from
        `compute_manifest_root_hash`).
    items : list of dict
        Manifest item records for this batch, in the exact order they
        were planned. Each must contain:
          - "rel_path" : str
          - "hash_content" : str (strong content digest)

    Returns
    -------
    batch_hash : str
        Hex digest string uniquely identifying this batch of items.

    Notes
    -----
    - Uses the library's central `hash_object` helper to ensure consistent
      hashing policy across the codebase.
    - The batch hash is *order-sensitive*: the same items in a different order
      yield a different batch hash. This makes resume/replay deterministic.
    - If any item is missing `hash_content`, this function raises a ValueError.
    - Typically called inside `read_images` (or `read_csv` equivalent) when
      emitting each batch.

    Examples
    --------
    >>> root = "abc123..."
    >>> items = [
    ...   {"rel_path": "img/001.jpg", "hash_content": "aa..."},
    ...   {"rel_path": "img/002.jpg", "hash_content": "bb..."},
    ... ]
    >>> compute_batch_hash(root, items)[:8]
    '9fbc21a1'
    """
    if not isinstance(root_hash, str) or not root_hash:
        raise ValueError("root_hash must be a non-empty string")

    ordered: List[tuple[str, str]] = []
    for rec in items:
        rel = str(rec.get("rel_path", ""))
        hc = rec.get("hash_content")
        if not hc:
            raise ValueError(f"Item {rel!r} missing required 'hash_content'")
        ordered.append((rel, str(hc)))

    identity = {
        "type": "image.batch:v1",
        "root_hash": root_hash,
        "items": ordered,
    }

    return Hash().hash_object(identity)