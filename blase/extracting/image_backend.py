import importlib.util
from pathlib import Path
from typing import Callable, List, Optional, Dict, Any, Literal, Iterator
import random

from blase.utils.hashing import Hash


def auto_target_bytes_from_system(return_type: str) -> Optional[int]:
    """
    Estimate a safe per-batch decoded-bytes budget using available system memory.

    Parameters
    ----------
    return_type : {'np', 'pil', 'tensor'}
        Decoded output type.
        - 'np' or 'pil' → use CPU memory.
        - 'tensor'     → prefer GPU (PyTorch CUDA, then TensorFlow GPU), else CPU.

    Returns
    -------
    int or None
        Suggested soft upper bound in bytes. Clamped to [128 MiB, 2 GiB].
        Returns None if memory cannot be probed.

    Algorithm
    ---------
    tensor + PyTorch
        free = torch.cuda.mem_get_info()[0]            # bytes
        target = 0.70 * free
    tensor + TensorFlow
        Prefer NVML (pynvml): free = nvmlDeviceGetMemoryInfo(0).free
        Else, if TF virtual device has a memory_limit:
            total = memory_limit * 2**20                # MiB → bytes
            current = tf.config.experimental.get_memory_info('GPU:0')['current']
            free ≈ max(0, total - current)
        target = 0.70 * free
    CPU (np/pil or fallback)
        avail = psutil.virtual_memory().available
        target = 0.50 * avail

    Notes
    -----
    The value is for planning. Apply an additional safety margin during packing,
    and verify decoded memory at runtime.
    """
    min_cap = 128 * 1024 * 1024  # 128 MiB
    max_cap = 2 * 1024 * 1024 * 1024  # 2 GiB

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
                    vcfg = tf.config.experimental.get_virtual_device_configuration(
                        gpus[0]
                    )
                    if (
                        vcfg
                        and len(vcfg) > 0
                        and getattr(vcfg[0], "memory_limit", None)
                    ):
                        total_bytes = (
                            int(vcfg[0].memory_limit) * 1024 * 1024
                        )  # MiB -> bytes
                        # current allocated per TF allocator (bytes)
                        get_mem = getattr(
                            tf.config.experimental, "get_memory_info", None
                        )
                        current = 0
                        if callable(get_mem):
                            try:
                                meminfo = get_mem(
                                    "GPU:0"
                                )  # {'current':..., 'peak':...}
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
    Scan a directory for images and collect header-only metadata per file.

    This builds per-item records for batching, health checks, and deterministic
    replay. It never decodes pixel data.

    Parameters
    ----------
    directory : str
        Root directory to scan. Must exist.
    pattern : str
        Glob pattern (e.g., "**/*.jpg", "*.png"). If `recursive` is False and
        the pattern contains "**", those segments are stripped.
    recursive : bool
        Recurse into subdirectories when True. Otherwise scan only `directory`.
    filename_filter : callable or None
        Optional predicate `f(path: str) -> bool` applied after globbing.
        Files for which the predicate returns False are skipped.

    Returns
    -------
    list of dict
        One record per discovered file, sorted by `rel_path`. Each record has:
        - `abs_path` : str
        - `rel_path` : str           # path relative to `directory`
        - `ext`      : str           # lowercased extension without dot
        - `bytes`    : int or None
        - `mtime`    : int or None   # seconds since epoch
        - `width`    : int or None
        - `height`   : int or None
        - `mode`     : str or None   # Pillow mode ("RGB", "L", ...)
        - `channels` : int or None   # inferred from mode
        - `exif_orientation` : int or None  # 1..8 when present
        - `is_corrupt` : bool
        - `estimated_decoded_bytes` : int   # width * height * channels
        - `hash_path_mtime` : str    # fast SHA-256 over "rel|mtime|bytes"

    Notes
    -----
    - Requires Pillow; only image headers are read.
    - Corrupt/unreadable files are included with `is_corrupt=True`.
    - For strong identity, compute full-content hashes separately and merge.
    """
    # --- dependency check (Pillow) ---
    if importlib.util.find_spec("PIL") is None:
        raise RuntimeError(
            "Pillow is required for header scans. Install with: pip install pillow"
        )
    from PIL import Image, UnidentifiedImageError

    # --- normalize inputs ---
    root = Path(directory).expanduser()
    if not root.exists() or not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory!r}")
    root = root.resolve()

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
    MODE_TO_CHANNELS = {
        "1": 1,
        "L": 1,
        "P": 1,
        "RGB": 3,
        "RGBA": 4,
        "CMYK": 4,
        "YCbCr": 3,
    }

    for p in paths:
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
            "rel_path": p.resolve().relative_to(root).as_posix(),
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


def shuffle_manifest(
    manifest: List[Dict[str, Any]], seed: Optional[int]
) -> List[Dict[str, Any]]:
    """
    Deterministically shuffle a manifest when a seed is provided.

    Parameters
    ----------
    manifest : list of dict
        Records as produced by `scan_manifest_headers` (must be indexable).
    seed : int or None
        If an int, use Python's `random.Random(seed).shuffle` for a stable order.
        If None, return the input list unchanged (same object).

    Returns
    -------
    list of dict
        A reordered shallow copy when `seed` is not None; otherwise the original
        list object.

    Notes
    -----
    - No in-place modification when `seed` is not None (shallow copy). Dict items
      are not copied; only order changes.
    - With `seed is None`, the function returns the *same* list object; callers
      should not mutate it if they need the original preserved.
    - Given the same `manifest` and `seed`, order is reproducible across machines
      (uses Mersenne Twister via `random.Random`).
    """
    if seed is None:
        return manifest

    rng = random.Random(seed)
    shuffled = manifest.copy()
    rng.shuffle(shuffled)
    return shuffled


def ensure_item_content_hashes(manifest: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Populate `hash_content` for each manifest item using a full-file hash.

    Parameters
    ----------
    manifest : list of dict
        Records from `scan_manifest_headers`. Each item should include:
        - `abs_path` : str      # absolute file path
        - `rel_path` : str      # relative path (unused here)
        - `is_corrupt` : bool   # may be True; bytes can still be hashed
        Optional:
        - `hash_content` : str  # preserved if already non-empty

    Returns
    -------
    list of dict
        The same list object, mutated in place. For each item:
        - `hash_content` : str or None
            Hex digest of file bytes when hashing succeeds.
            Unchanged if already set and non-empty.
            None on I/O or hashing errors, or if `abs_path` missing.

    Notes
    -----
    - Byte-level hashing is independent of image decodability.
    - For large files, consider a streaming hasher; this implementation reads
      via the project helper `Hash().hash_file(path)`.
    """
    for rec in manifest:
        if rec.get("hash_content"):
            continue

        abs_path = rec.get("abs_path")
        if not abs_path:
            rec["hash_content"] = None
            continue

        p = Path(abs_path)
        try:
            # Read whole file into memory and hash with the library helper
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

    The root hash identifies the dataset state from:
      (1) scan parameters (directory, pattern, recursive, seed, hash_mode), and
      (2) the ordered per-item identity:
          - if `hash_mode == "content"`          → (rel_path, hash_content)
          - if `hash_mode == "path_mtime_size"`  → (rel_path, hash_path_mtime)

    Items are sorted by `rel_path` for stability.

    Parameters
    ----------
    manifest : list of dict
        Records from header scan (optionally augmented). Each item must have:
        - `rel_path` : str
        - if `hash_mode=="content"`         : `hash_content` : str
        - if `hash_mode=="path_mtime_size"` : `hash_path_mtime` : str
    directory : str
        Scanned root directory. Included in identity.
    pattern : str
        Glob pattern used for scanning. Included in identity.
    recursive : bool
        Whether recursion was enabled. Included in identity.
    seed : int or None
        Shuffle seed recorded for planning. `None` is distinct from any int.
    hash_mode : {'content','path_mtime_size'}, default 'content'
        Per-item identity mode.

    Returns
    -------
    str
        Hex digest representing the entire dataset state.

    Raises
    ------
    ValueError
        On unsupported `hash_mode` or missing required fields.

    Notes
    -----
    - The function does not mutate `manifest`.
    - Hashing is centralized via `Hash().hash_object(identity_obj)` to keep
      consistency with the rest of the system.
    - Include corrupt items; decodability is orthogonal to byte identity.
    """
    mode = (hash_mode or "content").strip().lower()
    if mode not in ("content", "path_mtime_size"):
        raise ValueError(
            f"Unsupported hash_mode: {hash_mode!r}. Use 'content' or 'path_mtime_size'."
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
    Plan image batches for memory-aware decoding.

    Parameters
    ----------
    manifest : list of dict
        Final-ordered records from `scan_manifest_headers(...)`. Each item should
        include `width`, `height`, `channels` (ints or None). If any are missing,
        `estimated_decoded_bytes` is used when present, else 0.
    mode : {'auto', 'manual'}
        'auto' packs greedily by decoded-bytes budget. 'manual' uses fixed count.
    batch_size : int or None
        Hard ceiling on items per batch. Required when `mode='manual'`.
    target_batch_bytes : int or None
        Soft budget per batch in bytes. Required when `mode='auto'`.
    safety_margin : float
        Headroom fraction. Effective cap = `target_batch_bytes * (1 - safety_margin)`.
        Must satisfy 0.0 ≤ safety_margin < 1.0.
    max_item_decoded_bytes : int or None
        Per-item guardrail. If an item's estimate exceeds this, cap the estimate
        (decode-time can still downscale or isolate).
    max_side : int or None
        If set, estimate uses an aspect-preserving downscale so the longer side
        equals `max_side` before computing bytes.

    Returns
    -------
    list of dict
        One dict per batch:
          - 'items' : list[dict]   # subset of input records, in order
          - 'est_decoded_bytes' : int
          - 'is_last' : bool       # True only on the final batch

    Notes
    -----
    - Deterministic greedy first-fit in input order.
    - Oversized single items (> cap) form a single-item batch to guarantee progress.
    - Only planning is done here; enforce caps again at decode time.
    """
    if mode not in ("auto", "manual"):
        raise ValueError(f"Unsupported mode: {mode!r}")

    if mode == "manual":
        if not batch_size or batch_size <= 0:
            raise ValueError(
                "When mode='manual', batch_size must be a positive integer."
            )
    else:  # auto
        if not target_batch_bytes or target_batch_bytes <= 0:
            raise ValueError(
                "When mode='auto', target_batch_bytes must be a positive integer."
            )
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
            plan.append(
                {
                    "items": chunk,
                    "est_decoded_bytes": int(total),
                    "is_last": (start + bs) >= n,
                }
            )
        return plan

    # mode == "auto" (bytes-aware greedy packing)
    cap = int(target_batch_bytes * (1.0 - safety_margin))
    if cap <= 0:
        raise ValueError(
            "Effective planning cap (after safety_margin) must be positive."
        )

    cur_items: List[Dict[str, Any]] = []
    cur_total = 0

    def _flush(is_last_flag: bool = False):
        nonlocal cur_items, cur_total
        if cur_items:
            plan.append(
                {
                    "items": cur_items,
                    "est_decoded_bytes": int(cur_total),
                    "is_last": is_last_flag,
                }
            )
            cur_items, cur_total = [], 0

    for idx, rec in enumerate(manifest, 1):
        est = _scaled_estimate(rec)

        would_exceed_bytes = (cur_total + est) > cap if cap > 0 else False
        would_exceed_count = batch_size is not None and len(cur_items) >= batch_size

        if cur_items and (would_exceed_bytes or would_exceed_count):
            _flush(is_last_flag=False)

        # Ensure progress for a single huge item:
        # if est > cap and batch is empty, accept it as a single-item batch.
        if not cur_items and (est > cap) and (batch_size is None or batch_size > 0):
            # Single-item batch for this oversized record
            plan.append(
                {
                    "items": [rec],
                    "est_decoded_bytes": int(est),
                    "is_last": False,  # may be updated below if it happens to be last
                }
            )
            continue

        cur_items.append(rec)
        cur_total += est

        if batch_size is not None and len(cur_items) >= batch_size:
            _flush(is_last_flag=False)

    # Flush any remainder and mark the last batch
    if cur_items:
        plan.append(
            {
                "items": cur_items,
                "est_decoded_bytes": int(cur_total),
                "is_last": True,
            }
        )

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
    Decode a batch of images with Pillow, with EXIF orientation, color conversion,
    and optional size cap.

    Parameters
    ----------
    items : list of dict
        Each item must include:
        - 'abs_path' : str  Absolute file path.
    return_type : {'np', 'pil', 'tensor'}, default 'np'
        'np'     → list of numpy arrays (uint8), HxWxC (RGB) or HxW (gray).
        'pil'    → list of PIL.Image objects.
        'tensor' → list of torch tensors (uint8), HxWxC or HxW.
    color : {'rgb', 'gray'}, default 'rgb'
        Target color space ('RGB' or 'L').
    max_side : int or None
        If set and max(width,height) > max_side, downscale with LANCZOS to fit.

    Returns
    -------
    list
        Decoded items in requested representation. Failures yield None in-place.

    Notes
    -----
    - Lazy-imports dependencies (Pillow, NumPy, PyTorch).
    - Uses `ImageOps.exif_transpose` to normalize orientation.
    - Only decodes pixels. No side effects or DB writes.
    """
    import importlib.util

    if importlib.util.find_spec("PIL") is None:
        raise RuntimeError(
            "Pillow is required to decode images with backend 'pil'. Install with: pip install pillow"
        )
    from PIL import Image, ImageOps

    # ensure codecs are registered for both save and open
    try:
        import PIL.JpegImagePlugin  # noqa: F401
        import PIL.PngImagePlugin  # noqa: F401
        import PIL.BmpImagePlugin  # noqa: F401
    except Exception:
        pass
    Image.init()

    to_numpy = return_type == "np"
    to_pil = return_type == "pil"
    to_tensor = return_type == "tensor"

    if to_tensor:
        if importlib.util.find_spec("torch") is None:
            raise RuntimeError(
                "return_type='tensor' requires PyTorch. Install with: pip install torch"
            )
        import torch

    if to_numpy:
        if importlib.util.find_spec("numpy") is None:
            raise RuntimeError(
                "return_type='np' requires NumPy. Install with: pip install numpy"
            )
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
    Decode a batch of images with OpenCV, with color conversion and optional size cap.

    Parameters
    ----------
    items : list of dict
        Each item must include:
        - 'abs_path' : str  Absolute file path.
    return_type : {'np', 'pil', 'tensor'}, default 'np'
        'np'     → list of numpy arrays (uint8), HxWxC (RGB) or HxW (gray).
        'pil'    → list of PIL.Image objects (requires Pillow).
        'tensor' → list of torch tensors (uint8), HxWxC or HxW (requires PyTorch).
    color : {'rgb', 'gray'}, default 'rgb'
        Target output color. BGR→RGB conversion is applied for 'rgb'.
    max_side : int or None
        If set and max(width,height) > max_side, downscale with INTER_AREA to fit.

    Returns
    -------
    list
        Decoded items in requested representation. Failures yield None in-place.

    Notes
    -----
    - OpenCV ignores EXIF orientation. Use the PIL backend if EXIF normalization is required.
    - Only decodes pixels. No CAS/DB side effects.
    """
    if importlib.util.find_spec("cv2") is None:
        raise RuntimeError(
            "OpenCV is required for decode_batch_cv2. Install with: pip install opencv-python"
        )
    import cv2  # type: ignore

    to_numpy = return_type == "np"
    to_pil = return_type == "pil"
    to_tensor = return_type == "tensor"

    if to_pil:
        if importlib.util.find_spec("PIL") is None:
            raise RuntimeError(
                "return_type='pil' requires Pillow. Install with: pip install pillow"
            )
        from PIL import Image  # type: ignore

    if to_tensor:
        if importlib.util.find_spec("torch") is None:
            raise RuntimeError(
                "return_type='tensor' requires PyTorch. Install with: pip install torch"
            )
        import torch  # type: ignore

    if to_numpy or to_tensor or to_pil:
        if importlib.util.find_spec("numpy") is None:
            raise RuntimeError(
                "NumPy is required for this decoder path. Install with: pip install numpy"
            )

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
                        img = cv2.resize(
                            img, (new_w, new_h), interpolation=cv2.INTER_AREA
                        )
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
                        img = cv2.resize(
                            img, (new_w, new_h), interpolation=cv2.INTER_AREA
                        )
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


def iter_batches_from_plan(
    batch_plan: List[Dict[str, Any]],
) -> Iterator[Dict[str, Any]]:
    """
    Iterate over a batch plan from `plan_image_batches` and normalize flags.

    Ensures every yielded batch dict has keys:
    {"items", "est_decoded_bytes", "is_last"}.
    Exactly one batch (the final entry) has `is_last=True`. If the incoming
    plan marks multiple batches as last (or none), this function normalizes it
    so only the final entry is True.

    Parameters
    ----------
    batch_plan : list of dict
        Each element should be:
        - "items" : list[dict]
        - "est_decoded_bytes" : int
        - "is_last" : bool

    Yields
    ------
    dict
        Shallow-copied batch dict with normalized `is_last`.

    Raises
    ------
    ValueError
        If an entry is not a dict, is missing required keys, or has wrong types.

    Notes
    -----
    - Empty input yields nothing.
    - Shallow copy: the returned dict is a new object, but its "items" list is the same object.
    """
    if not batch_plan:
        return
        yield  # pragma: no cover (keep as generator even on early return)

    # Find index of the last non-empty entry to mark as final.
    last_idx = len(batch_plan) - 1

    for i, b in enumerate(batch_plan):
        if not isinstance(b, dict):
            raise ValueError(
                f"Batch plan entry {i} must be a dict, got {type(b).__name__}."
            )

        # Validate required keys
        for k in ("items", "est_decoded_bytes", "is_last"):
            if k not in b:
                raise ValueError(f"Batch plan entry {i} missing required key: {k!r}.")

        items = b["items"]
        est = b["est_decoded_bytes"]
        is_last = bool(b["is_last"])

        if not isinstance(items, list):
            raise ValueError(
                f"Batch plan entry {i} 'items' must be a list, got {type(items).__name__}."
            )
        if not isinstance(est, int):
            # allow ints that arrive as bools? no—force int to catch bugs early
            raise ValueError(
                f"Batch plan entry {i} 'est_decoded_bytes' must be int, got {type(est).__name__}."
            )

        # Normalize final flag: only the last entry should be True
        norm_is_last = i == last_idx

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
