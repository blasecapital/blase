from __future__ import annotations
from typing import Callable, Iterable, Dict, Any, Optional, List, Literal, Iterator
import json

import numpy as np

from blase.extracting.csv_backend import memory_aware_batcher, read_batches_pandas, read_batches_polars
from blase.extracting.image_backend import (
    auto_target_bytes_from_system, 
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
from blase.extracting.manifest_utils import build_manifest_descriptor, ensure_dataset_for_manifest
from blase.utils.backends import resolve_backend_csv, resolve_backend_images
from blase.track import Track


class Extract:
    """
    Handles batch extraction of raw data from various local sources, with optional extension to cloud storage.
    It support two primary workflows:
        - Full-dataset iteration: Stream the entire dataset in memory-safe batches using a simple loop.
        - Targeted batch reading: For structured or file-based sources (e.g., databases, directories), 
          users can specify filters, WHERE clauses, or custom logic to read specific subsets of data.

    This module enables memory-efficient reading of structured and unstructured data by **processing in batches**.
    It dynamically adjusts batch sizes based on available system memory to prevent memory overflow.
    It is designed to be **local-first** but supports extension to cloud-based resources.

    Supported Local Sources:
    - **Tabular Data**: CSV, Parquet, SQLite, DuckDB, PostgreSQL.
    - **Unstructured Data**: Images, text files, JSON, log files.
    - **APIs**: Allows batch retrieval from HTTP-based data sources.

    Extendability:
    -------------
    Users can **extend this module** to support cloud-based storage (e.g., AWS S3, Google Cloud Storage, Azure Blob)
    by implementing custom extractors using the provided interface.

    The extraction process:
    1. Determines available system memory.
    2. Dynamically adjusts batch size based on memory constraints.
    3. Reads data in **configurable batch sizes** to prevent memory overflow.
    4. Streams data **directly from the source** without full in-memory reading.
    5. Converts data into a **standard format** (e.g., DataFrame, NumPy, or raw text).

    Methods:
    --------
    read_csv(file_path: str, batch_size: int) -> Iterable[pd.DataFrame]
        Reads CSV in chunks and yields DataFrame batches.

    read_sql(query: str, connection, batch_size: int) -> Iterable[pd.DataFrame]
        Streams SQL query results in batches.

    read_images(directory: str, batch_size: int) -> Iterable[List[np.ndarray]]
        reads images from a directory in batches.

    read_json(file_path: str, batch_size: int) -> Iterable[List[dict]]
        Parses JSON files in chunks.

    Example:
    --------
    >>> extractor = Extract()
    >>> for batch, flag, parent_hash in extractor.read_csv("data.csv", batch_size=1000):
    >>>     process_batch(batch, parent_hash)  # Handle each batch separately

    Extending to Cloud:
    -------------------
    Users can implement cloud extractors by subclassing `Extract`:
    >>> class S3Extract(Extract):
    >>>     def read_s3(self, bucket_name, file_key, batch_size):
    >>>         pass  # Implement cloud-specific extraction logic

    Notes:
    ------
    - If batch size is not specified, the module defaults to an optimized chunk size.
    - This module will read the entire dataset if it's size is below 10% of available memory.
    """

    def __init__(self, track: bool = True) -> None:
        """
        Initialize the Extract module.

        Args:
            track (bool): Whether to enable automatic logging for extraction steps.
                        Can be overridden per method call. Defaults to True.
        """
        self.track = track

    def read_csv(
        self,
        file_path: str,
        mode: str = "auto",
        batch_size: Optional[int] = None,
        use_cols: Optional[List[str]] = None,
        filter_by: Optional[list] = None,
        backend: str = "polars",
        track: bool = True
    ) -> Iterable[Any]:
        """
        Stream a CSV in memory-safe batches with optional lineage tracking.

        This generator reads a CSV file incrementally using either the Polars or
        Pandas backend and yields `(batch, is_last, meta)` tuples. When `track=True`,
        the read is recorded as a tracked step (via `Track`), the source file is
        registered (CAS/DB), and the emitted `meta` includes upstream lineage
        (e.g., the source data hash and producer step). When `track=False`, no
        tracking side-effects occur and `meta` is `None`.

        Parameters
        ----------
        file_path : str
            Path to the CSV file on disk.

        mode : {"auto", "manual"}, default "auto"
            Batch-size selection mode.
            - ``"auto"``: Determine a batch size via a backend-specific memory
            heuristic.
            - ``"manual"``: Use the provided ``batch_size`` (must be a positive int).

        batch_size : int, optional
            Number of rows per batch when ``mode="manual"``. Ignored if
            ``mode="auto"``.

        use_cols : list of str, optional
            Subset of column names to read. If ``None``, all columns are read.

        filter_by : list[dict] or dict, optional
            Row-level filters to apply per batch *after* reading. Each filter is a
            dict (implementation-dependent) such as
            ``{"col": "City", "value": "SF"}``. A single dict is also accepted and
            treated as a one-element list. Order is preserved.

        backend : {"pandas", "polars"}, default "polars"
            Backend to use for batch reading/parsing.

        track : bool, default True
            If ``True``, record a tracked step:
            - Registers the source CSV in CAS/metadata and binds it as input
            (role ``"source"``).
            - Emits lineage-rich ``meta`` containing ``upstream`` (source data hash),
            ``producer_step``, ``ordinal``, and other hints.
            If ``False``, yields `(batch, is_last, None)` without any side-effects.

        Yields
        ------
        tuple
            A 3-tuple ``(batch, is_last, meta)``:
            - ``batch`` : DataFrame-like batch (``pandas.DataFrame`` or
            ``polars.DataFrame`` depending on backend).
            - ``is_last`` : bool indicating the final batch for this stream.
            - ``meta`` : dict with lineage/step info when ``track=True``; ``None``
            when ``track=False``. When tracked, ``meta`` includes (at minimum):
            ``{"upstream": [{"id": <data_hash>, "role": "source"}],
                "producer_step": <step_hash>, "ordinal": <int>, "chunk_size": <int>,
                "reader_backend": <str>, "use_cols": <list|None>, "filter_by": <list|None>}``.

        Raises
        ------
        ValueError
            If ``mode`` is not one of {"auto", "manual"} or if ``mode="manual"`` and
            ``batch_size`` is missing/invalid.
        FileNotFoundError
            If ``file_path`` does not exist or is not readable.
        Exception
            Any backend-specific parsing errors are propagated.

        Notes
        -----
        - In tracked mode, the function opens a `StreamStep`, snapshots code/env
        once, records the source file as input (role ``"source"``), and seals the
        step on the final batch (or marks aborted on error).
        - Filtering semantics are post-read and backend-dependent; they operate on
        each batch independently.
        - ``mode="auto"`` delegates to a memory-aware batch estimator; exact
        heuristic may vary by backend.
        - This function **does not** load the entire dataset into memory.

        Examples
        --------
        Basic (untracked) streaming:

        >>> ex = Extract()
        >>> for batch, is_last, meta in ex.read_csv("data.csv", backend="pandas", track=False):
        ...     do_something(batch)

        Tracked run with manual batching and column subset:

        >>> ex = Extract()
        >>> for batch, is_last, meta in ex.read_csv(
        ...         "data.csv", mode="manual", batch_size=10_000,
        ...         use_cols=["colA", "colB"], backend="polars", track=True):
        ...     # meta contains source data hash and step metadata
        ...     consume(batch)
        """

        # dependency handling
        backend = resolve_backend_csv(backend)

        # argument checks
        if mode not in ("auto", "manual"):
            raise ValueError("Unsupported mode: %r. Must be 'auto' or 'manual'." % mode)
        if mode == "manual" and not batch_size:
            raise ValueError("When mode='manual', batch_size must be specified.")
        if mode == "auto":
            batch_size = memory_aware_batcher(file_path, backend)

        if isinstance(filter_by, dict):  # allow single dict
            filter_by = [filter_by]

        tracker = Track.get(track)

        # untracked path (no DAG/CAS)
        if tracker is None:
            if backend == "polars":
                yield from ((b, last, None) for (b, last) in read_batches_polars(file_path, batch_size, use_cols, filter_by))
            else:
                yield from ((b, last, None) for (b, last) in read_batches_pandas(file_path, batch_size, use_cols, filter_by))
            return

        # tracked path
        params = {"file_path": str(file_path), "batch_size": batch_size, "backend": backend}
        impl = read_batches_pandas if backend == "pandas" else read_batches_polars

        stream = tracker.stream("blase.Extract.read_csv", params, code_fn=impl)

        # register the source file as data + input binding
        src_hash = stream.step.register_data(kind="csv", version="1", path_or_bytes=file_path, metadata={})
        stream.step.add_input(src_hash, role="source", arg_name="file_path")

        try:
            for i, (batch, is_last) in enumerate(impl(file_path, batch_size, use_cols, filter_by), 1):
                meta = {
                    "upstream": [{"id": src_hash, "role": "source"}],
                    "producer_step": stream.step.step_hash,  # <-- fix
                    "ordinal": i,
                    "chunk_size": batch_size,
                    "reader_backend": backend,
                    "use_cols": use_cols,
                    "filter_by": filter_by,
                }
                new_meta = stream.emit(last_batch=is_last, meta=meta)
                yield (batch, is_last, new_meta)
            stream.close_ok()
        except Exception as e:
            stream.close_error(type(e), e, e.__traceback__)
            raise

    def read_images(
        self,
        directory: str,
        pattern: str = "**/*.jpg",
        backend: Literal["pil", "cv2"] = "pil",
        return_type: Literal["np", "pil", "tensor"] = "np",
        mode: Literal["auto", "manual"]="auto",
        safety_margin: float = 0.15, 
        max_item_decoded_bytes: Optional[int]=None,
        batch_size: Optional[int] = None,
        target_batch_bytes: Optional[int] = None,
        filename_filter: Optional[Callable[[str], bool]] = None,
        shuffle: bool = False,
        seed: Optional[int] = None,
        recursive: bool = True,
        color: Literal["rgb", "gray"] = "rgb",
        max_side: Optional[int] = None,
        track: bool = True
    ) -> Iterator[dict]:
        """
        Yield memory-aware batches of images from a directory.

        Parameters
        ----------
        directory : str
            Root directory containing image files.
        pattern : str, default="**/*.jpg"
            Glob pattern to match files (respects `recursive`).
        backend : {"pil", "cv2"}, default="pil"
            Image decoding library.
        return_type : {"np", "pil", "tensor"}, default="np"
            Format of returned images.
        mode : {"auto", "manual"}, default="auto"
            Batching policy:
            - "auto": choose batch boundaries by decoded-bytes target (see `target_batch_bytes`).
            - "manual": fixed item count per batch (see `batch_size`).
            In both modes `batch_size` can act as a hard ceiling if provided.
        safety_margin : float, default=0.15
            Fractional headroom reserved when packing by bytes to avoid OOM (e.g., 0.15 = 15%).
        max_item_decoded_bytes : int, optional
            Hard per-item upper bound on decoded size. Items exceeding this will be downscaled
            (if `max_side` allows) or isolated into a single-item batch; if neither is possible
            and `strict` is desired in the future, raise.
        batch_size : int, optional
            Max number of images per batch (required when `mode="manual"`).
        target_batch_bytes : int, optional
            Approximate decoded-bytes budget per batch (required when `mode="auto"` unless
            determinable via system memory; recorded in step metadata).
        filename_filter : callable, optional
            Predicate `f(path:str) -> bool` to include/exclude files after globbing.
        shuffle : bool, default=False
            Shuffle file order before batching.
        seed : int, optional
            RNG seed for reproducible shuffling. If `shuffle=True` and `seed is None`, a default is chosen.
        recursive : bool, default=True
            Recurse into subdirectories for the glob.
        color : {"rgb", "gray"}, default="rgb"
            Output color space; conversion happens in the backend adapter.
        max_side : int, optional
            Downscale longer side to this length at decode time (preserves aspect ratio).
        track : bool, default=True
            If True, record DAG/CAS events; if False (or no active Track), run untracked.

        Yields
        ------
        dict
            A dictionary per batch with:
            - "paths": list[str]
            - "images": list/array/tensor of decoded images (per `return_type`)
            - "meta": dict with lineage and batch planning details (or None if untracked)

        Notes
        -----
        Dataset identity and replay:
        - Computes a strong **manifest root hash** (Merkle over `(rel_path, content_hash)` in
        deterministic order) and stores it in the manifest descriptor.
        - Computes a **batch hash** per emitted batch from the manifest root + ordered item hashes.
        - Both hashes are logged in tracked mode to support idempotent resume/replay.
        """
        # ---------------- argument + dependency checks ----------------
        print("Resolving backend...")
        backend = resolve_backend_images(backend)  # validates and lazy-import checks

        if mode not in ("auto", "manual"):
            raise ValueError(f"Unsupported mode: {mode!r}. Must be 'auto' or 'manual'.")

        if mode == "manual" and not batch_size:
            raise ValueError("When mode='manual', batch_size must be specified.")

        if mode == "auto" and target_batch_bytes is None:
            # Try to infer from system memory; fall back to a conservative default
            target_batch_bytes = auto_target_bytes_from_system(return_type) or (256 * 1024 * 1024)

        if shuffle and seed is None:
            seed = 42  # deterministic default

        # ---------------- manifest scan (header-only; no pixels) ----------------
        print("Scanning manifest headers...")
        manifest = scan_manifest_headers(
            directory=directory,
            pattern=pattern,
            recursive=recursive,
            filename_filter=filename_filter,
        )
        if shuffle:
            manifest = shuffle_manifest(manifest, seed)

        # Compute per-item content hashes (or incremental updates) and manifest Merkle root.
        # Helpers should fill/return item["hash_content"] (hex) and the ordered list used.
        print("Ensuring item content hashes...")
        manifest = ensure_item_content_hashes(manifest)  # may reuse cached hashes when (mtime,size) unchanged
        print("Computing manifest root hash...")
        root_hash = compute_manifest_root_hash(
            manifest=manifest,
            directory=directory,
            pattern=pattern,
            recursive=recursive,
            seed=seed,
            hash_mode="content",  # recorded in descriptor
        )

        # ---------------- batching plan ----------------
        # Pack by bytes in auto mode; allow batch_size to act as a hard ceiling when provided.
        print("Planning image extracting batches...")
        batch_plan = plan_image_batches(
            manifest=manifest,
            mode=mode,
            batch_size=batch_size,
            target_batch_bytes=target_batch_bytes,
            safety_margin=safety_margin,
            max_item_decoded_bytes=max_item_decoded_bytes,
            max_side=max_side,  # for better decoded-size estimates if downscaling
        )

        # ---------------- untracked path ----------------
        tracker = Track.get(track)
        if tracker is None:
            impl_decode = decode_batch_pil if backend == "pil" else decode_batch_cv2
            for i, batch in enumerate(iter_batches_from_plan(batch_plan), 1):
                batch_hash = compute_batch_hash(root_hash, batch["items"])  # uses ordered item content hashes
                decoded = impl_decode(batch["items"], return_type, color, max_side)
                meta = {
                    "ordinal": i,
                    "count": len(batch["items"]),
                    "estimated_decoded_bytes": batch["est_decoded_bytes"],
                    "batch_hash": batch_hash,
                    "manifest_root_hash": root_hash,
                    "batch_policy": {
                        "mode": mode,
                        "batch_size": batch_size,
                        "target_batch_bytes": target_batch_bytes,
                        "safety_margin": safety_margin,
                        "max_item_decoded_bytes": max_item_decoded_bytes,
                    },
                    "reader_backend": backend,
                    "return_type": return_type,
                    "color": color,
                    "max_side": max_side,
                    "shuffle": shuffle,
                    "seed": seed,
                }
                yield {"paths": [it["abs_path"] for it in batch["items"]],
                    "images": decoded,
                    "meta": meta}
            return

        # ---------------- tracked path ----------------
        params = {
            "directory": str(directory),
            "pattern": pattern,
            "backend": backend,
            "return_type": return_type,
            "mode": mode,
            "safety_margin": safety_margin,
            "max_item_decoded_bytes": max_item_decoded_bytes,
            "batch_size": batch_size,
            "target_batch_bytes": target_batch_bytes,
            "filename_filter": bool(filename_filter),
            "shuffle": shuffle,
            "seed": seed,
            "recursive": recursive,
            "color": color,
            "max_side": max_side,
            "hash_mode": "content",
        }
        impl = decode_batch_pil if backend == "pil" else decode_batch_cv2
        stream = tracker.stream("blase.Extract.read_images", params, code_fn=impl)

        try:
            # Register the manifest descriptor (CAS) and mark as input.
            manifest_desc = build_manifest_descriptor(
                manifest=manifest,
                directory=directory,
                pattern=pattern,
                recursive=recursive,
                seed=seed,
                root_hash=root_hash,
                hash_mode="content",
            )
            manifest_hash = stream.step.register_data(
                kind="image.manifest", 
                version="1",
                path_or_bytes=json.dumps(manifest_desc, ensure_ascii=False).encode("utf-8"),
                metadata=manifest_desc,
            )
            stream.step.add_input(manifest_hash, role="manifest", arg_name="manifest")

            # Record dataset + members for Inspect/CLI (ordered; cached fields optional).
            dataset_id = ensure_dataset_for_manifest(
                manifest_hash=manifest_hash,
                manifest=manifest,
                tracker=stream.step,
                cache_member_fields=True,
                root_hash=root_hash,
            )

            # Emit batches
            for i, batch in enumerate(iter_batches_from_plan(batch_plan), 1):
                batch_hash = compute_batch_hash(root_hash, batch["items"])

                # Register tiny batch descriptor (CAS) so resume can key off it.
                batch_meta_desc = {
                    "manifest_hash": manifest_hash,
                    "dataset_id": dataset_id,
                    "ordinal": i,
                    "count": len(batch["items"]),
                    "estimated_decoded_bytes": batch["est_decoded_bytes"],
                    "color": color,
                    "max_side": max_side,
                    "batch_hash": batch_hash,
                    "manifest_root_hash": root_hash,
                }
                batch_desc_hash = stream.step.register_data(
                    kind="image.batch.meta", 
                    version="1",
                    path_or_bytes=json.dumps(batch_meta_desc, ensure_ascii=False).encode("utf-8"),
                    metadata=batch_meta_desc,
                )
                stream.step.add_input(batch_desc_hash, role="batch", arg_name="batch_desc")

                # Decode (no CAS for pixels).
                decoded = impl(batch["items"], return_type, color, max_side)

                # Emit step event with lineage + policy.
                meta = {
                    "upstream": [
                        {"id": manifest_hash, "role": "manifest"},
                        {"id": batch_desc_hash, "role": "batch_desc"},
                    ],
                    "producer_step": stream.step.step_hash,
                    "ordinal": i,
                    "count": len(batch["items"]),
                    "estimated_decoded_bytes": batch["est_decoded_bytes"],
                    "batch_hash": batch_hash,
                    "manifest_root_hash": root_hash,
                    "batch_policy": {
                        "mode": mode,
                        "batch_size": batch_size,
                        "target_batch_bytes": target_batch_bytes,
                        "safety_margin": safety_margin,
                        "max_item_decoded_bytes": max_item_decoded_bytes,
                    },
                    "reader_backend": backend,
                    "return_type": return_type,
                    "color": color,
                    "max_side": max_side,
                    "shuffle": shuffle,
                    "seed": seed,
                }
                new_meta = stream.emit(last_batch=batch["is_last"], meta=meta)

                yield {
                    "paths": [it["abs_path"] for it in batch["items"]],
                    "images": decoded,
                    "meta": new_meta,
                }

            stream.close_ok()

        except Exception as e:
            stream.close_error(type(e), e, e.__traceback__)
            raise

    def read_json(
        self,
        file_path: str,
        mode: str = "auto",
        batch_size: Optional[int] = None,
        backend: str = "polars",
        track: bool = True
    ) -> Iterable[Any]:
        """
        Read a JSON file in memory-safe batches using the specified backend.

        This method streams data in chunks, allowing for efficient processing of large JSON files
        that may not fit into memory. It supports auto-calculated or manually specified batch sizes,
        and filters can be applied to restrict which rows are read.

        Parameters:
        -----------
        file_path : str
            Path to the JSON file.

        mode : {"auto", "manual"}, default="auto"
            Determines how batch size is handled:
            - "auto": Uses a backend memory estimator to determine optimal batch size.
            - "manual": Uses the provided `batch_size` parameter (required in this mode).

        batch_size : int, optional
            Number of lines per batch (required if mode is "manual").

        backend : {"pandas", "polars"}, default="polars"
            The data handling library to use for reading and parsing.
            - "polars": Optimized for speed and memory efficiency.
            - "pandas": Standard and robust.
            
        track : bool, default=True
            Whether to log the operation via the `Track` system (if initialized).

        Returns:
        --------
        Iterable[Any]
            An iterator over data chunks, where each chunk is a DataFrame-like object
            (either `pandas.DataFrame` or `polars.DataFrame`, depending on backend).

        Example:
        --------
        >>> extractor = Extract()
        >>> for batch in extractor.read_json(file_path="data/large.json", mode="auto", backend="pandas"):
        >>>     process_batch(batch)

        Notes:
        ------
        - This function does not read the entire dataset into memory.
        - Backends must be installed separately (e.g., `pip install polars`).
        """

        backend = resolve_backend_csv(backend)
    
    def read_parquet(self, file_path: str, batch_size: int = None) -> Iterable[Any]: pass
    def read_hdf5(self, file_path: str, batch_size: int = None) -> Iterable[np.ndarray]: pass
    def read_orc(self, file_path: str, batch_size: int = None) -> Iterable[Any]: pass
    def read_excel(self, file_path: str, sheet_name: str = None, batch_size: int = None) -> Iterable[Any]: pass
    def read_tsv(self, file_path: str, batch_size: int = None) -> Iterable[Any]: pass
    def read_xml(self, file_path: str, batch_size: int = None) -> Iterable[dict]: pass
    def read_audio(self, file_path: str, batch_size: int = None) -> Iterable[np.ndarray]: pass
    def read_npy(self, file_path: str, batch_size: int) -> Iterable[np.ndarray]: pass
    def read_sql(self, query: str, connection, batch_size: int) -> Iterable[Any]: pass
    def read_api(
        self, 
        endpoint: str, 
        params: Dict[str, Any] = None, 
        headers: Dict[str, str] = None, 
        batch_size: int = 100, 
        max_pages: int = None, 
        rate_limit: float = 1.0
    ) -> Iterable[Any]: pass
