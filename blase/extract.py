from __future__ import annotations
from typing import (
    Callable,
    Iterable,
    Dict,
    Any,
    Optional,
    List,
    Literal,
    Iterator,
    Sequence,
    Union,
    Tuple,
    TYPE_CHECKING,
)
from pathlib import Path
import json

import numpy as np

if TYPE_CHECKING:
    import pyarrow as pa

from blase.extracting.csv_backend import (
    memory_aware_batcher,
    read_batches_pandas,
    read_batches_polars,
)
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
    compute_batch_hash,
)
from blase.extracting.parquet_backend import (
    _normalize_source_list,
    _scan_parquet_manifest_stub,
    _ensure_parquet_content_hashes,
    _compute_manifest_root_hash_stub,
    _plan_parquet_batches_stub,
    _read_parquet_table_stub,
    _compute_batch_hash_stub,
    _auto_detect_image_cols,
    _validate_image_schema,
    _iter_images_from_table,
)
from blase.extracting.manifest_utils import (
    build_manifest_descriptor,
    ensure_dataset_for_manifest,
)
from blase.types import Batch
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
    >>> for batch in extractor.read_csv("data.csv", batch_size=1000):
    >>>     process_batch(batch)  # Handle each batch separately

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
        track: bool = True,
    ) -> Iterable[Batch]:
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
        Batch
            - batch.data: DataFrame-like batch (``pandas.DataFrame`` or
                ``polars.DataFrame`` depending on backend).
            - batch.is_last: bool indicating the final batch for this stream.
            - batch.meta: dict with lineage/step info when ``track=True``; ``None``
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
        >>> for batch in ex.read_csv("data.csv", backend="pandas", track=False):
        ...     do_something(batch.data)

        Tracked run with manual batching and column subset:

        >>> ex = Extract()
        >>> for batch in ex.read_csv(
        ...         "data.csv", mode="manual", batch_size=10_000,
        ...         use_cols=["colA", "colB"], backend="polars", track=True):
        ...     # meta contains source data hash and step metadata
        ...     consume(batch.data)
        """

        backend = resolve_backend_csv(backend)

        if mode not in ("auto", "manual"):
            raise ValueError("Unsupported mode: %r. Must be 'auto' or 'manual'." % mode)
        if mode == "manual" and not batch_size:
            raise ValueError("When mode='manual', batch_size must be specified.")
        if mode == "auto":
            batch_size = memory_aware_batcher(file_path, backend)

        if isinstance(filter_by, dict):
            filter_by = [filter_by]

        tracker = Track.get(track)

        # ----- untracked path -----
        if tracker is None:
            if backend == "polars":
                for b, last in read_batches_polars(
                    file_path, batch_size, use_cols, filter_by
                ):
                    yield Batch(data=b, labels=None, paths=None, is_last=last, meta={})
            else:
                for b, last in read_batches_pandas(
                    file_path, batch_size, use_cols, filter_by
                ):
                    yield Batch(data=b, labels=None, paths=None, is_last=last, meta={})
            return

        # ----- tracked path -----
        params = {
            "file_path": str(file_path),
            "batch_size": batch_size,
            "backend": backend,
        }
        impl = read_batches_pandas if backend == "pandas" else read_batches_polars

        stream = tracker.stream("blase.Extract.read_csv", params, code_fn=impl)

        src_hash = stream.step.register_data(
            kind="csv", version="1", path_or_bytes=file_path, metadata={}
        )
        stream.step.add_input(src_hash, role="source", arg_name="file_path")

        try:
            for i, (batch, is_last) in enumerate(
                impl(file_path, batch_size, use_cols, filter_by), 1
            ):
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
                yield Batch(
                    data=batch, labels=None, paths=None, is_last=is_last, meta=new_meta
                )
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
        mode: Literal["auto", "manual"] = "auto",
        safety_margin: float = 0.15,
        max_item_decoded_bytes: Optional[int] = None,
        batch_size: Optional[int] = None,
        target_batch_bytes: Optional[int] = None,
        filename_filter: Optional[Callable[[str], bool]] = None,
        shuffle: bool = False,
        seed: Optional[int] = None,
        recursive: bool = True,
        color: Literal["rgb", "gray"] = "rgb",
        max_side: Optional[int] = None,
        track: bool = True,
    ) -> Iterator[Batch]:
        """
        Yield memory-aware batches of decoded images from a directory.

        Parameters
        ----------
        directory : str
            Root directory containing image files.
        pattern : str, default="**/*.jpg"
            Glob pattern to match files. Honors `recursive`.
        backend : {"pil", "cv2"}, default="pil"
            Image decoding library selector.
        return_type : {"np", "pil", "tensor"}, default="np"
            Type of decoded images.
        mode : {"auto", "manual"}, default="auto"
            Batching policy.
            - "auto": target approx decoded bytes per batch (see `target_batch_bytes`).
            - "manual": fixed item count per batch (see `batch_size`).
            In both modes, `batch_size` acts as a hard ceiling when provided.
        safety_margin : float, default=0.15
            Fractional headroom reserved when packing by bytes to reduce OOM risk.
        max_item_decoded_bytes : int, optional
            Per-item decoded-size cap. Oversized items are downscaled if `max_side`
            allows, otherwise isolated into 1-item batches.
        batch_size : int, optional
            Max images per batch. Required when `mode="manual"`.
        target_batch_bytes : int, optional
            Approx decoded-bytes budget per batch. If `mode="auto"` and omitted,
            a default is inferred from system memory or 256 MiB fallback.
        filename_filter : callable, optional
            Predicate `f(path: str) -> bool` applied after globbing.
        shuffle : bool, default=False
            Shuffle file order before batching.
        seed : int, optional
            RNG seed for reproducible shuffling. If `shuffle=True` and `seed is None`,
            a deterministic default is chosen.
        recursive : bool, default=True
            Recurse into subdirectories for the glob.
        color : {"rgb", "gray"}, default="rgb"
            Output color space.
        max_side : int, optional
            Downscale longer side to this length at decode time (keeps aspect ratio).
        track : bool, default=True
            If True and an active tracker exists, emit DAG/CAS events. Otherwise run untracked.

        Yields
        ------
        Batch
            **Untracked mode** (`Track.get(track)` is None):
                - batch.paths: list[str]
                - batch.data: decoded images (list/array/tensor per `return_type`)
                - batch.meta: dict with batch planning details
            **Tracked mode** (active tracker):
                - batch.paths: list[str]
                - batch.is_last: bool
                - batch.data: decoded images (per `return_type`)
                - batch.meta: dict including upstream lineage, batch and manifest hashes

        Notes
        -----
        Dataset identity and replay:
        - Computes a manifest **Merkle root** over `(rel_path, content_hash)` in
        deterministic order and records it.
        - Computes a **batch hash** from the manifest root and ordered item hashes.
        - In tracked mode, both are logged to enable idempotent resume/replay.

        Raises
        ------
        ValueError
            If `mode` is invalid or `mode="manual"` without `batch_size`.
        """
        print("Resolving backend...")
        backend = resolve_backend_images(backend)

        if mode not in ("auto", "manual"):
            raise ValueError(f"Unsupported mode: {mode!r}. Must be 'auto' or 'manual'.")

        if mode == "manual" and not batch_size:
            raise ValueError("When mode='manual', batch_size must be specified.")

        if mode == "auto" and target_batch_bytes is None:
            target_batch_bytes = auto_target_bytes_from_system(return_type) or (
                256 * 1024 * 1024
            )

        if shuffle and seed is None:
            seed = 42

        print("Scanning manifest headers...")
        manifest = scan_manifest_headers(
            directory=directory,
            pattern=pattern,
            recursive=recursive,
            filename_filter=filename_filter,
        )
        if shuffle:
            manifest = shuffle_manifest(manifest, seed)

        print("Ensuring item content hashes...")
        manifest = ensure_item_content_hashes(manifest)
        print("Computing manifest root hash...")
        root_hash = compute_manifest_root_hash(
            manifest=manifest,
            directory=directory,
            pattern=pattern,
            recursive=recursive,
            seed=seed,
            hash_mode="content",
        )

        print("Planning image extracting batches...")
        batch_plan = plan_image_batches(
            manifest=manifest,
            mode=mode,
            batch_size=batch_size,
            target_batch_bytes=target_batch_bytes,
            safety_margin=safety_margin,
            max_item_decoded_bytes=max_item_decoded_bytes,
            max_side=max_side,
        )

        # ---------------- untracked path ----------------
        tracker = Track.get(track)
        if tracker is None:
            impl_decode = decode_batch_pil if backend == "pil" else decode_batch_cv2
            for i, batch in enumerate(iter_batches_from_plan(batch_plan), 1):
                batch_hash = compute_batch_hash(root_hash, batch["items"])
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
                yield Batch(
                    data=decoded,
                    labels=None,
                    paths=[it["abs_path"] for it in batch["items"]],
                    is_last=bool(batch.get("is_last", False)),
                    meta=meta,
                )
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
                path_or_bytes=json.dumps(manifest_desc, ensure_ascii=False).encode(
                    "utf-8"
                ),
                metadata=manifest_desc,
            )
            stream.step.add_output(manifest_hash, name="manifest")

            dataset_id = ensure_dataset_for_manifest(
                manifest_hash=manifest_hash,
                manifest=manifest,
                tracker=stream.step,
                cache_member_fields=True,
                root_hash=root_hash,
            )

            for i, batch in enumerate(iter_batches_from_plan(batch_plan), 1):
                batch_hash = compute_batch_hash(root_hash, batch["items"])

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
                    path_or_bytes=json.dumps(
                        batch_meta_desc, ensure_ascii=False
                    ).encode("utf-8"),
                    metadata=batch_meta_desc,
                )
                stream.step.add_output(batch_desc_hash, name=f"batch_desc_{i}")
                batch_token_hash = stream.step.register_data(
                    kind="image.batch",
                    version="1",
                    path_or_bytes=batch_hash.encode("utf-8"),
                    metadata={
                        "batch_hash": batch_hash,
                        "manifest_hash": manifest_hash,
                        "ordinal": i,
                    },
                )
                stream.step.add_output(batch_token_hash, name=f"batch_{i}")

                # Decode (no CAS for pixels).
                decoded = impl(batch["items"], return_type, color, max_side)

                meta = {
                    "upstream": [
                        {"id": manifest_hash, "role": "manifest"},
                        {"id": batch_desc_hash, "role": "batch_desc"},
                        {"id": batch_token_hash, "role": "batch"},
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
                # Do not record inputs for this method but pass upstream dict to downstream steps
                meta_no_up = dict(meta)
                meta_no_up.pop("upstream", None)
                new_meta = stream.emit(last_batch=batch["is_last"], meta=meta_no_up)
                new_meta["upstream"] = meta["upstream"]

                yield Batch(
                    data=decoded,
                    labels=None,
                    paths=[it["abs_path"] for it in batch["items"]],
                    is_last=batch["is_last"],
                    meta=new_meta,
                )

            stream.close_ok()

        except Exception as e:
            stream.close_error(type(e), e, e.__traceback__)
            raise

    def read_parquet(
        self,
        source: Union[str, Path, Sequence[Union[str, Path]]],
        *,
        pattern: str = "**/*.parquet",
        recursive: bool = True,
        format: Literal["auto", "parquet", "dataset", "feather"] = "auto",
        return_type: Literal["arrow", "pandas"] = "arrow",
        columns: Optional[Sequence[str]] = None,
        filters: Any = None,
        batch_size: Optional[int] = None,
        use_threads: bool = True,
        memory_map: bool = True,
        deterministic: bool = True,
        max_rows: Optional[int] = None,
        limit_files: Optional[int] = None,
        schema: "pa.schema | None" = None,
        to_pandas_kwargs: Optional[dict] = None,
        # images mode additions
        mode: Literal["table", "images"] = "table",
        bytes_col: str = "img_bytes",
        dims_cols: Tuple[str, str, str] = ("height", "width", "channels"),
        label_col: Optional[str] = "label",
        path_col: Optional[str] = "path",
        decode: Optional[Literal["np", "pil", "cv2", "tensor"]] = None,
        images_return: Literal["bytes", "np", "pil", "tensor"] = "bytes",
        # tracking
        track: bool = True,
    ) -> Iterator[Batch]:
        """
        Stream Parquet/Feather data as batched tables or decoded images.

        The reader builds a manifest from one or more sources, plans row-based
        batches, and yields results either untracked (simple iterator) or tracked
        (with CAS bookkeeping). In ``mode='images'`` it decodes image bytes into
        arrays, PIL images, or tensors.

        Parameters
        ----------
        source : str or Path or sequence of (str | Path)
            File(s) or directory(ies). Directories are globbed using *pattern*.
        pattern : str, default="**/*.parquet"
            Glob used when a directory is provided.
        recursive : bool, default=True
            Use recursive globbing for directories.
        format : {"auto","parquet","dataset","feather"}, default="auto"
            Parsing mode hint. ``"dataset"`` treats directories as a dataset.
        return_type : {"arrow","pandas"}, default="arrow"
            Table representation for non-image batches and internal reads.
        columns : sequence of str, optional
            Projected columns. ``None`` reads all columns.
        filters : Any, optional
            Predicate pushdown (shape depends on backend).
        batch_size : int, optional
            Target rows per batch. If ``None``, a bytes-aware packer is used.
        use_threads : bool, default=True
            Enable multi-threaded Parquet reads when supported.
        memory_map : bool, default=True
            Use memory-mapped IO when supported.
        deterministic : bool, default=True
            Stabilize file ordering for reproducible manifests.
        max_rows : int, optional
            Upper bound on total rows to read.
        limit_files : int, optional
            Read at most this many files after sorting deterministically.
        schema : pyarrow.Schema, optional
            Optional schema hint.
        to_pandas_kwargs : dict, optional
            Passed to ``Table.to_pandas`` when ``return_type='pandas'``.
        mode : {"table","images"}, default="table"
            Table mode yields tables. Images mode yields decoded image items.
        bytes_col : str, default="img_bytes"
            Column containing encoded image bytes (images mode).
        dims_cols : tuple(str, str, str), default=("height","width","channels")
            Column names for image dimensions (images mode).
        label_col : str or None, default="label"
            Optional label column name (images mode).
        path_col : str or None, default="path"
            Optional path column name (images mode).
        decode : {"np","pil","cv2","tensor"} or None, default=None
            Force a decode target. If ``None``, controlled by *images_return*.
        images_return : {"bytes","np","pil","tensor"}, default="bytes"
            Output type for images mode.
        track : bool, default=True
            If True, record manifest and batch metadata to CAS and emit tracked
            batches. If False, yield simple untracked batches.

        Yields
        ------
        Batch
            **Untracked mode** (`Track.get(track)` is None):
                - batch.data: where *data* is a
                    ``pyarrow.Table`` or ``pandas.DataFrame`` per *return_type*.
                - batch.meta: {...}
            **Tracked mode** (`Track.get(track)` is True):
                - batch.data: images where *images*
                    are ``bytes``/``np``/``PIL.Image``/tensor per *images_return*.
                - batch.is_last
                - batch.meta

        Raises
        ------
        ValueError
            On unsupported ``format``, ``return_type``, ``mode``, ``decode``,
            or ``images_return`` selections.

        Notes
        -----
        - In tracked table mode (``track=True`` and ``mode='table'``), this
        function records batches and metadata but does not yield table batches.
        - Images mode automatically resolves and validates schema for required
        columns, then iterates one image per row.

        Examples
        --------
        Untracked tables:

        >>> it = ex.read_parquet("data/", return_type="arrow", track=False)
        >>> first = next(it); table, meta = first.data, first.meta

        Tracked images as NumPy arrays:

        >>> it = ex.read_parquet("imgs/", mode="images", decode="np", images_return="np")
        >>> batch = next(it)
        """
        if format not in ("auto", "parquet", "dataset", "feather"):
            raise ValueError(
                f"Unsupported format: {format!r}. Must be 'auto', 'parquet', 'dataset', 'feather'."
            )

        if return_type not in ("arrow", "pandas"):
            raise ValueError(
                f"Unsupported return_type: {return_type!r}. Must be 'arrow' or 'pandas'."
            )

        if mode not in ("table", "images"):
            raise ValueError(
                f"Unsupported mode: {mode!r}. Must be 'table' or 'images'."
            )

        if decode is not None and decode not in ("np", "pil", "cv2", "tensor"):
            raise ValueError(
                f"Unsupported decode: {decode!r}. Must be 'np', 'pil', 'cv2', 'tensor'."
            )

        if images_return not in ("bytes", "np", "pil", "tensor"):
            raise ValueError(
                f"Unsupported images_return: {images_return!r}. Must be 'bytes', 'np', 'pil', 'tensor'."
            )

        # ---------- 1) Resolve + normalize source(s) ----------
        src_list = _normalize_source_list(
            source, pattern, recursive, deterministic, limit_files
        )

        # ---------- 2) Build manifest + identity ----------
        manifest = _scan_parquet_manifest_stub(
            sources=src_list,
            fmt=format,
            columns=columns,
            filters=filters,
            schema=schema,
        )

        # ---------- 3) Plan batches (row-based) ----------
        batch_plan = _plan_parquet_batches_stub(
            manifest=manifest,
            mode=mode,
            batch_size=batch_size,
            max_rows=max_rows,
        )

        # ---------- 4) Untracked path ----------
        tracker = Track.get(track)
        if tracker is None:
            for i, plan in enumerate(batch_plan, 1):
                table_like = _read_parquet_table_stub(
                    manifest,
                    plan,
                    return_type,
                    columns,
                    use_threads,
                    to_pandas_kwargs,
                    mode=mode,
                )
                meta = {
                    "ordinal": i,
                    "count": plan["count"],
                    "batch_hash": "",
                    "manifest_root_hash": "",
                    "schema_fp": manifest.get("schema_fp"),
                    "source_kind": manifest.get("kind"),
                    "batch_policy": {"mode": "rows", "batch_size": batch_size},
                    "mode": mode,
                    "bytes_col": bytes_col,
                    "dims_cols": list(dims_cols) if dims_cols else None,
                    "label_col": label_col,
                    "path_col": path_col,
                    "decode": decode,
                    "images_return": images_return,
                }
                if mode == "images":
                    # 1) resolve columns
                    eff_bytes, eff_dims, eff_label, eff_path = _auto_detect_image_cols(
                        table_like=table_like,
                        bytes_col=bytes_col,
                        dims_cols=dims_cols,
                        label_col=label_col,
                        path_col=path_col,
                    )
                    # 2) validate schema
                    _validate_image_schema(
                        table_like=table_like,
                        bytes_col=eff_bytes,
                        dims_cols=eff_dims,
                        label_col=eff_label,
                        path_col=eff_path,
                    )
                    # 3) iterate rows → per-image items
                    imgs: List[Any] = []
                    lbls: Optional[List[Any]] = [] if eff_label else None
                    pths: List[Optional[str]] = []

                    for one_imgs, one_labels, one_paths in _iter_images_from_table(
                        table_like=table_like,
                        return_type=return_type,
                        bytes_col=eff_bytes,
                        dims_cols=eff_dims,
                        label_col=eff_label,
                        path_col=eff_path,
                        decode=decode,
                        images_return=images_return,
                    ):
                        imgs.extend(one_imgs)
                        if lbls is not None:
                            lbls.extend(one_labels or [None] * len(one_imgs))
                        pths.extend(one_paths or [None] * len(one_imgs))

                    meta["items"] = len(imgs)
                yield Batch(
                    data=table_like if mode == "table" else imgs,
                    labels=None if mode == "table" else lbls,
                    paths=None if mode == "table" else pths,
                    is_last=plan["is_last"],
                    meta=meta,  # dict
                )
            return

        # ---------- 5) Tracked path ----------
        manifest = _ensure_parquet_content_hashes(manifest=manifest)
        root_hash = _compute_manifest_root_hash_stub(manifest)
        params = {
            "sources": [str(p) for p in src_list],
            "format": format,
            "return_type": return_type,
            "columns": list(columns) if columns else None,
            "filters": bool(filters),
            "batch_size": batch_size,
            "use_threads": use_threads,
            "memory_map": memory_map,
            "deterministic": deterministic,
            "max_rows": max_rows,
            "limit_files": limit_files,
            "schema": bool(schema),
            "mode": mode,
            "bytes_col": bytes_col,
            "dims_cols": list(dims_cols) if dims_cols else None,
            "label_col": label_col,
            "path_col": path_col,
            "decode": decode,
            "images_return": images_return,
        }

        stream = tracker.stream("blase.Extract.read_parquet", params, code_fn=None)  # type: ignore[attr-defined]

        try:
            manifest_desc = build_manifest_descriptor(
                manifest=manifest,
                deterministic=deterministic,
                root_hash=root_hash,
            )
            manifest_hash = stream.step.register_data(  # type: ignore[attr-defined]
                kind="table.manifest",
                version="1",
                path_or_bytes=json.dumps(manifest_desc, ensure_ascii=False).encode(
                    "utf-8"
                ),
                metadata=manifest_desc,
            )
            stream.step.add_output(manifest_hash, name="manifest")  # type: ignore[attr-defined]
            dataset_id = ensure_dataset_for_manifest(
                manifest_hash=manifest_hash,
                manifest=manifest,
                tracker=stream.step,
                cache_member_fields=True,
                root_hash=root_hash,
            )

            for i, plan in enumerate(batch_plan, 1):
                batch_hash = _compute_batch_hash_stub(root_hash, plan)

                batch_meta_desc = {
                    "manifest_hash": manifest_hash,
                    "dataset_id": dataset_id,
                    "ordinal": i,
                    "fragment": plan.get("fragment"),
                    "row_group": plan.get("row_group"),
                    "offset": plan.get("offset", 0),
                    "count": plan["count"],
                    "batch_hash": batch_hash,
                    "manifest_root_hash": root_hash,
                    "schema_fp": manifest.get("schema_fp"),
                }
                bmeta_hash = stream.step.register_data(  # type: ignore[attr-defined]
                    kind="table.batch.meta",
                    version="1",
                    path_or_bytes=json.dumps(
                        batch_meta_desc, ensure_ascii=False
                    ).encode("utf-8"),
                    metadata=batch_meta_desc,
                )
                stream.step.add_output(bmeta_hash, name=f"batch_desc_{i}")  # type: ignore[attr-defined]
                btoken_hash = stream.step.register_data(  # type: ignore[attr-defined]
                    kind="table.batch",
                    version="1",
                    path_or_bytes=batch_hash.encode("utf-8"),
                    metadata={
                        "batch_hash": batch_hash,
                        "manifest_hash": manifest_hash,
                        "ordinal": i,
                    },
                )
                stream.step.add_output(btoken_hash, name=f"batch_{i}")  # type: ignore[attr-defined]

                table_like = _read_parquet_table_stub(
                    manifest=manifest,
                    plan=plan,
                    return_type=return_type,
                    columns=columns,
                    use_threads=use_threads,
                    to_pandas_kwargs=to_pandas_kwargs,
                    mode=mode,
                )

                meta = {
                    "upstream": [
                        {"id": manifest_hash, "role": "manifest"},
                        {"id": bmeta_hash, "role": "batch_desc"},
                        {"id": btoken_hash, "role": "batch"},
                    ],
                    "producer_step": getattr(stream.step, "step_hash", None),  # type: ignore[attr-defined]
                    "ordinal": i,
                    "count": plan["count"],
                    "batch_hash": batch_hash,
                    "manifest_root_hash": root_hash,
                    "schema_fp": manifest.get("schema_fp"),
                    "batch_policy": {"mode": "rows", "batch_size": batch_size},
                }
                meta_no_up = dict(meta)
                meta_no_up.pop("upstream", None)
                new_meta = stream.emit(last_batch=plan["is_last"], meta=meta_no_up)  # type: ignore[attr-defined]
                new_meta["upstream"] = meta["upstream"]

                if mode == "table":
                    yield Batch(
                        data=table_like,
                        labels=None,
                        paths=None,
                        is_last=plan["is_last"],
                        meta=new_meta,
                    )

                elif mode == "images":
                    # 1) resolve columns
                    eff_bytes, eff_dims, eff_label, eff_path = _auto_detect_image_cols(
                        table_like=table_like,
                        bytes_col=bytes_col,
                        dims_cols=dims_cols,
                        label_col=label_col,
                        path_col=path_col,
                    )
                    # 2) validate schema
                    _validate_image_schema(
                        table_like=table_like,
                        bytes_col=eff_bytes,
                        dims_cols=eff_dims,
                        label_col=eff_label,
                        path_col=eff_path,
                    )
                    # 3) iterate rows → per-image items
                    imgs: List[Any] = []
                    lbls: Optional[List[Any]] = [] if eff_label else None
                    pths: List[Optional[str]] = []

                    for one_imgs, one_labels, one_paths in _iter_images_from_table(
                        table_like=table_like,
                        return_type=return_type,
                        bytes_col=eff_bytes,
                        dims_cols=eff_dims,
                        label_col=eff_label,
                        path_col=eff_path,
                        decode=decode,
                        images_return=images_return,
                    ):
                        imgs.extend(one_imgs)
                        if lbls is not None:
                            lbls.extend(one_labels or [None] * len(one_imgs))
                        pths.extend(one_paths or [None] * len(one_imgs))

                    new_meta["items"] = len(imgs)
                    yield Batch(
                        data=imgs,
                        labels=lbls,
                        paths=pths,
                        is_last=plan["is_last"],
                        meta=new_meta,
                    )

            stream.close_ok()  # type: ignore[attr-defined]

        except Exception as e:
            stream.close_error(type(e), e, e.__traceback__)  # type: ignore[attr-defined]
            raise
        pass

    def read_json(
        self,
        file_path: str,
        mode: str = "auto",
        batch_size: Optional[int] = None,
        backend: str = "polars",
        track: bool = True,
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

    def read_hdf5(self, file_path: str, batch_size: int = None) -> Iterable[np.ndarray]:
        pass

    def read_orc(self, file_path: str, batch_size: int = None) -> Iterable[Any]:
        pass

    def read_excel(
        self, file_path: str, sheet_name: str = None, batch_size: int = None
    ) -> Iterable[Any]:
        pass

    def read_tsv(self, file_path: str, batch_size: int = None) -> Iterable[Any]:
        pass

    def read_xml(self, file_path: str, batch_size: int = None) -> Iterable[dict]:
        pass

    def read_audio(
        self, file_path: str, batch_size: int = None
    ) -> Iterable[np.ndarray]:
        pass

    def read_npy(self, file_path: str, batch_size: int) -> Iterable[np.ndarray]:
        pass

    def read_sql(self, query: str, connection, batch_size: int) -> Iterable[Any]:
        pass

    def read_api(
        self,
        endpoint: str,
        params: Dict[str, Any] = None,
        headers: Dict[str, str] = None,
        batch_size: int = 100,
        max_pages: int = None,
        rate_limit: float = 1.0,
    ) -> Iterable[Any]:
        pass
