from typing import Union, Literal, Optional, List
from pathlib import Path

import numpy as np
import pandas as pd

from blase.track import Track
from blase.types import PreviewResult
from blase.utils.fs import ensure_parent_dir
from blase.examining.preview_image_backend import (
    parquet_bytes_available,
    sample_parquet_bytes,
    get_source,
    maybe_materialize_grid,
    maybe_materialize_grid_bytes,
    maybe_materialize_thumbs,
    maybe_materialize_thumbs_bytes,
    build_meta,
    sample_population,
    probe_paths,
    build_manifest_and_hash,
    register_tracked_population,
    register_tracked_preview_manifest,
    register_tracked_materializations,
    _pct_stats,
    _percentile,
)


class Examine:
    """
    Provides tools for inspecting and visualizing batched datasets.

    The `Examine` class enables users to analyze data quality, distributions, and patterns
    during the preprocessing pipeline. It is designed to work with batched datasets and can
    either report insights on a per-batch basis or aggregate information across multiple batches
    for comprehensive analysis.

    Core Features:
    --------------
    - Generate batch-wise and cumulative summary statistics
    - Visualize distributions, correlations, outliers, and missing values
    - Support numerical, categorical, text, image, and audio data
    - Easily integrate with data extracted using the `Extract` class
    - Uses Matplotlib for clear and customizable plots

    Usage:
    ------
    >>> examine = Examine(aggregate_batches=True)
    >>> for batch in extractor.load_csv("data.csv", batch_size=1000):
    >>>     examine.process_batch(batch)
    >>> examine.generate_summary()

    Parameters:
    -----------
    aggregate_batches (bool): Whether to track and report stats across all processed batches

    Notes:
    ------
    - When `aggregate_batches=True`, cumulative metrics will be computed over all batches
    - Visualizations can be called per batch or after full dataset processing
    - For large datasets, automatic sampling may be used in plots
    """

    def __init__(self, thumb_size: int = 256):
        self.thumb_size = int(thumb_size)

    def process_batch(self, data: Union[pd.DataFrame, np.ndarray]):
        """Process a batch of data to compute statistics and update aggregations."""
        pass

    def generate_summary(self):
        """Generate and print summary statistics from all processed batches."""
        pass

    def plot_feature_distribution(self, feature: str, bins: int = 20):
        """Plot histogram of a numerical feature."""
        pass

    def correlation_matrix(self):
        """Plot correlation heatmap for numerical features."""
        pass

    def missing_values_report(self):
        """Show missing value counts and visualize as a heatmap."""
        pass

    def detect_outliers(self, feature: str, method: str = "iqr"):
        """Detect outliers using IQR or Z-score method."""
        pass

    def categorical_summary(self, feature: str):
        """Display bar chart of category frequencies."""
        pass

    def text_summary(self, column: str):
        """Report basic text statistics like length distribution and common tokens."""
        pass

    def preview_images(
        self,
        source: str,
        *,
        source_kind: Literal[
            "directory",
            "parquet",
            "csv",
            "jsonl",
            "tfrecord",
            "npy",
            "video",
            "hf_datasets",
            "s3",
        ] = "directory",
        pattern: str = "**/*",
        allowed_ext: Optional[List[str]] = None,
        parquet_path_col: Optional[str] = None,
        parquet_image_bytes_col: Optional[str] = None,
        path_root: Optional[str] = None,
        recursive: bool = True,
        sample_size: int = 9,
        seed: Optional[int] = 42,
        backend: Literal["pil"] = "pil",
        return_thumbs: bool = False,
        head_read_bytes: int = 128 * 1024,
        max_side: Optional[int] = None,
        max_total_decode_bytes: Optional[
            int
        ] = None,
        save: bool = False,
        to: Optional[str] = None,
        save_individual: bool = False,
        out_dir: Optional[str] = None,
        format: Literal["png", "jpg"] = "png",
        jpeg_quality: int = 90,
        min_w: int = 64,
        min_h: int = 64,
        max_aspect: float = 5.0,
        track: bool = True,
    ) -> PreviewResult:
        """
        Preview a sample of images from a source and compute lightweight diagnostics.

        This method supports two acquisition modes:

        1. **Path-based mode** (default): Build a population of file paths from a mixed
        directory or a path-bearing dataset (e.g., Parquet with a path column),
        deterministically sample `sample_size` entries, then open each path and
        probe a thumbnail plus basic stats.

        2. **Parquet-bytes mode**: If `source_kind == "parquet"` and
        `parquet_image_bytes_col` is provided and found to contain non-empty image
        bytes (via a fast check), read only that column and decode thumbnails
        directly from in-file bytes without touching the filesystem paths.

        Both modes are designed to be **headless** and **deterministic** with `seed`.
        Optional tracking hooks write a minimal “preview manifest” for reproducibility
        and future restore.

        Parameters
        ----------
        source : str
            Input source. For `source_kind="directory"`, this is a directory path.
            For `source_kind="parquet"`, this is a Parquet file path. Other kinds are
            reserved for future extensions.
        source_kind : {'directory', 'parquet', 'csv', 'jsonl', 'tfrecord', 'npy', 'video', 'hf_datasets', 's3'}, optional
            Interpretation of `source`. Defaults to ``'directory'``. Currently,
            path-based mode is implemented for ``'directory'`` and ``'parquet'`` with a path column.
        pattern : str, optional
            Glob for directory enumeration. Ignored unless `source_kind='directory'`.
            Default is ``'**/*'``.
        allowed_ext : list of str, optional
            File extensions to include in directory mode (e.g., ``['.jpg', '.png']``).
            Case-insensitive. If ``None``, a standard image extension set is used.
        parquet_path_col : str, optional
            Column name containing on-disk image paths for Parquet path-based mode.
            If omitted, a common name is guessed: ``'path'``, ``'image_path'``,
            or ``'filepath'``.
        parquet_image_bytes_col : str, optional
            Column name containing image bytes for Parquet-bytes mode. If provided,
            the function will prefer bytes mode when the column is present and has
            any non-empty cells.
        path_root : str, optional
            Base directory to join with relative paths from a path column.
        recursive : bool, optional
            Recurse into subdirectories when in directory mode. Default ``True``.
        sample_size : int, optional
            Number of samples to return. Must be > 0. Default ``9``.
        seed : int, optional
            RNG seed for deterministic sampling. If ``None``, defaults to ``42``.
        backend : {'pil'}, optional
            Image decode backend. Only ``'pil'`` is supported. Default ``'pil'``.
        return_thumbs : bool, optional
            Reserved. Thumbnails are computed internally for stats, but this method
            returns only metadata objects; thumbnails are written only via `save` or
            `save_individual`. Default ``False``.
        head_read_bytes : int, optional
            Maximum header bytes to read for format probing in bytes mode. Default
            ``128 * 1024``.
        max_side : int, optional
            Optional max side length for thumbnail decode. If ``None``, a sensible
            default is used by probing helpers.
        max_total_decode_bytes : int, optional
            Upper bound on aggregate decoded byte budget across the sampled set;
            decoding may be skipped once the budget is exceeded. Enforced in probe
            helpers. If ``None``, no additional guard is applied beyond defaults.
        save : bool, optional
            If ``True`` and any items are OK, write a contact-sheet grid to `to`.
        to : str, optional
            Output filepath for grid image. Used only when `save=True`.
        save_individual : bool, optional
            If ``True``, write per-sample thumbnails to `out_dir`.
        out_dir : str, optional
            Output directory for individual thumbnails. Used when `save_individual=True`.
        format : {'png', 'jpg'}, optional
            Output format for path-based thumbnail writing. Default ``'png'``.
        jpeg_quality : int, optional
            JPEG quality when `format='jpg'`. Default ``90``.
        min_w : int, optional
            Minimum image width accepted as OK. Default ``64``.
        min_h : int, optional
            Minimum image height accepted as OK. Default ``64``.
        max_aspect : float, optional
            Maximum aspect ratio (long_side / short_side) accepted as OK. Default ``5.0``.
        track : bool, optional
            If ``True``, record a tracked preview step, including a population
            manifest hash and a preview manifest for restore/replay. Default ``True``.

        Returns
        -------
        PreviewResult
            Dataclass-like object with:
            - ``items`` : list of PreviewItem
                One per sampled image. Each item includes fields such as:
                ``ok`` (bool), ``reason`` (str if not ok), dimensions (``w``, ``h``),
                intensity stats (``mean``, ``std``), thumbnail byte size, and a
                ``path`` or pseudo-path (e.g., ``'parquet:<file>#rg=<i>:row=<j>'``).
            - ``meta`` : dict
                Context and summary statistics:
                - ``mode`` : ``'paths'`` or ``'parquet_bytes'``.
                - ``source``, ``source_kind``, and relevant column names.
                - ``sample_size_req`` and ``sample_indices`` (global row indices or
                population indices).
                - ``thumb_size`` (implementation default), ``min_w``, ``min_h``,
                ``max_aspect``.
                - ``ok_count``, ``bad_count``, and ``bad_reasons``.
                - ``size_stats`` and ``intensity_stats`` percentiles.
                - Optional tracking keys when `track=True`:
                ``manifest_root_hash``, ``manifest_hash``, ``preview_hash``,
                and any recorded materializations in ``written``.

        Raises
        ------
        ValueError
            If ``backend`` is not ``'pil'`` or ``sample_size <= 0``.
        RuntimeError
            Propagated errors from tracking stream if an unexpected exception occurs
            during a tracked run.

        Notes
        -----
        - **Bytes-first heuristic**: When `source_kind='parquet'` and
        `parquet_image_bytes_col` is set, a constant-time scan checks for any
        non-empty bytes in that column. If present, the function enters
        parquet-bytes mode and decodes directly from column data.
        - **Determinism**: Sampling uses a local RNG seeded by `seed` and is stable
        for the same population and inputs.
        - **Side effects**: No files are written unless `save=True` and/or
        `save_individual=True`. When tracking is enabled, preview manifests and
        materialization references are registered for restore.
        - **Performance**: Parquet-bytes mode column-projects and reads only the row
        groups required by the sampled indices. Path-based mode opens only the
        sampled files.

        Examples
        --------
        Preview from a directory and write a grid:

        >>> res = examiner.preview_images(
        ...     "/data/images",
        ...     source_kind="directory",
        ...     sample_size=12,
        ...     seed=123,
        ...     save=True,
        ...     to="/tmp/preview_grid.png",
        ...     save_individual=True,
        ...     out_dir="/tmp/preview_sample"
        ... )

        Preview from a Parquet file with an image-bytes column:

        >>> res = examiner.preview_images(
        ...     "dataset.parquet",
        ...     source_kind="parquet",
        ...     parquet_image_bytes_col="image_bytes",
        ...     sample_size=9,
        ...     seed=42,
        ... )

        See Also
        --------
        sample_population : Deterministic sampling over an abstract population adapter.
        sample_parquet_bytes : Efficient sampler/decoder for Parquet image-bytes columns.
        probe_paths : Path-based probe producing per-item stats and thumbnails.
        """
        if backend != "pil":
            raise ValueError("Only 'pil' supported.")
        if sample_size <= 0:
            raise ValueError("sample_size must be > 0.")
        if seed is None:
            seed = 42

        src_path = Path(source).resolve()

        # ======================
        # UNTRACKED
        # ======================
        tracker = Track.get(track)
        if tracker is None:
            if (
                source_kind == "parquet"
                and parquet_image_bytes_col
                and parquet_bytes_available(str(src_path), parquet_image_bytes_col)
            ):
                # Fast path: bytes inside parquet
                idx, items = sample_parquet_bytes(
                    source=str(src_path),
                    bytes_col=parquet_image_bytes_col,
                    k=sample_size,
                    seed=seed,
                    head_read_bytes=head_read_bytes,
                    thumb_size=self.thumb_size,
                    max_side=max_side,
                )
                written: List[str] = []
                if save and to and any(it.ok for it in items):
                    wp = maybe_materialize_grid_bytes(items, Path(to))
                    if wp:
                        written.append(wp)
                if save_individual and out_dir:
                    written.extend(
                        maybe_materialize_thumbs_bytes(items, Path(out_dir), seed)
                    )
                meta = build_meta(
                    {
                        "mode": "parquet_bytes",
                        "source": str(src_path),
                        "source_kind": "parquet",
                        "bytes_col": parquet_image_bytes_col,
                        "path_col": None,
                        "pattern": None,
                        "parquet_path_col": None,
                        "sample_size_req": sample_size,
                        "seed": seed,
                        "thumb_size": self.thumb_size,
                        "items": items,
                        "written": written,
                        "min_w": min_w,
                        "min_h": min_h,
                        "max_aspect": max_aspect,
                        "sample_indices": idx,
                    }
                )
                return PreviewResult(items=items, meta=meta)

            # Generic population path: directory or parquet paths or future kinds
            src_params = {
                "source": str(src_path),
                "pattern": pattern,
                "allowed_ext": allowed_ext,
                "recursive": recursive,
                "path_col": parquet_path_col,
                "path_root": path_root,
            }
            adapter = get_source(source_kind)
            pop, pop_meta = adapter(src_params)

            idx, paths = sample_population(pop, k=sample_size, seed=seed)
            items = probe_paths(
                paths,
                thumb_size=self.thumb_size,
                head_read_bytes=head_read_bytes,
                max_side=max_side,
                max_total_decode_bytes=max_total_decode_bytes,
            )
            written: List[str] = []
            if save and to and any(it.ok for it in items):
                wp = maybe_materialize_grid_bytes(items, Path(to))
                if wp:
                    written.append(wp)
            if save_individual and out_dir:
                written.extend(
                    maybe_materialize_thumbs_bytes(items, Path(out_dir), seed)
                )

            meta = build_meta(
                {
                    "mode": "paths",
                    "source": str(src_path),
                    "source_kind": source_kind,
                    "pattern": pattern if source_kind == "directory" else None,
                    "parquet_path_col": parquet_path_col
                    if source_kind == "parquet"
                    else None,
                    "bytes_col": None,
                    "path_root": path_root,
                    "sample_size_req": sample_size,
                    "seed": seed,
                    "thumb_size": self.thumb_size,
                    "items": items,
                    "written": written,
                    "min_w": min_w,
                    "min_h": min_h,
                    "max_aspect": max_aspect,
                    "sample_indices": idx,
                    "population_meta": pop_meta,
                }
            )
            return PreviewResult(items=items, meta=meta)

        # ======================
        # TRACKED
        # ======================
        is_bytes_mode = (
            source_kind == "parquet"
            and parquet_image_bytes_col
            and parquet_bytes_available(str(src_path), parquet_image_bytes_col)
        )
        # Common params recorded on the step
        step_params = {
            "source": str(src_path),
            "source_kind": source_kind,
            "pattern": pattern if source_kind == "directory" else None,
            "mode": "parquet_bytes"
            if is_bytes_mode
            else "paths"
            if (source_kind == "parquet" and parquet_image_bytes_col)
            else "paths",
            "parquet_path_col": parquet_path_col if source_kind == "parquet" else None,
            "parquet_image_bytes_col": parquet_image_bytes_col,
            "path_root": path_root,
            "sample_size": sample_size,
            "seed": seed,
            "thumb_size": self.thumb_size,
            "save": save,
            "to": to,
            "save_individual": save_individual,
            "out_dir": out_dir,
            "format": format,
            "jpeg_quality": jpeg_quality,
            "head_read_bytes": head_read_bytes,
            "max_side": max_side,
            "max_total_decode_bytes": max_total_decode_bytes,
            "min_w": min_w,
            "min_h": min_h,
            "max_aspect": max_aspect,
            "allowed_ext": allowed_ext,
            "recursive": recursive,
        }
        stream = tracker.stream(
            "blase.Examine.preview_images", step_params, code_fn=None
        )

        try:
            # A) Build population + manifest hash for replay identity
            pop_ctx = {
                "source_kind": source_kind,
                "source": str(src_path),
                "pattern": pattern,
                "recursive": recursive,
                "parquet_path_col": parquet_path_col,
                "parquet_image_bytes_col": parquet_image_bytes_col,
                "path_root": path_root,
                "head_read_bytes": head_read_bytes,
            }
            manifest_desc, root_hash = build_manifest_and_hash(
                kind=source_kind, context=pop_ctx
            )
            manifest_hash = register_tracked_population(stream, manifest_desc)

            # B) Collect items via bytes or paths path
            if is_bytes_mode:
                idx, items = sample_parquet_bytes(
                    source=str(src_path),
                    bytes_col=parquet_image_bytes_col,
                    k=sample_size,
                    seed=seed,
                    head_read_bytes=head_read_bytes,
                    thumb_size=self.thumb_size,
                    max_side=max_side,
                )
                pop_meta = manifest_desc.get("population_meta")
            else:
                adapter = get_source(source_kind)
                pop, pop_meta = adapter(
                    {
                        "source": str(src_path),
                        "pattern": pattern,
                        "allowed_ext": allowed_ext,
                        "recursive": recursive,
                        "path_col": parquet_path_col,
                        "path_root": path_root,
                    }
                )
                idx, paths = sample_population(pop, k=sample_size, seed=seed)
                items = probe_paths(
                    paths,
                    thumb_size=self.thumb_size,
                    head_read_bytes=head_read_bytes,
                    max_side=max_side,
                    max_total_decode_bytes=max_total_decode_bytes,
                )

            # empty guard
            if not items:
                stream.emit(
                    last_batch=True,
                    meta={
                        "manifest_root_hash": root_hash,
                        "ok_count": 0,
                        "bad_count": 0,
                    },
                )
                stream.close_ok()
                return PreviewResult(
                    items=[],
                    meta=build_meta(
                        {
                            "mode": "parquet_bytes" if is_bytes_mode else "paths",
                            "source": str(src_path),
                            "source_kind": source_kind,
                            "sample_size_req": sample_size,
                            "seed": seed,
                            "thumb_size": self.thumb_size,
                            "items": [],
                            "written": [],
                            "sample_indices": idx,
                            "manifest_root_hash": root_hash,
                            "population_meta": pop_meta,
                        }
                    ),
                )

            # C) Optional materializations
            written: List[str] = []
            if save and to and any(it.ok for it in items):
                tp = Path(to)
                ensure_parent_dir(tp)
                wp = (
                    maybe_materialize_grid_bytes(items, tp)
                    if is_bytes_mode
                    else maybe_materialize_grid(
                        items, tp, format, jpeg_quality, self.thumb_size
                    )
                )
                if wp:
                    written.append(wp)

            if save_individual and out_dir:
                if is_bytes_mode:
                    written.extend(
                        maybe_materialize_thumbs_bytes(items, Path(out_dir), seed)
                    )  # bytes thumbs
                else:
                    written.extend(
                        maybe_materialize_thumbs(
                            items,
                            Path(out_dir),
                            format,
                            jpeg_quality,
                            self.thumb_size,
                            seed,
                        )
                    )

            # D) Preview manifest (diagnostic) + emit
            meta_core = {
                "manifest_root_hash": root_hash,
                "sample_indices": idx,
                "seed": seed,
                "thumb_size": self.thumb_size,
                "source_kind": source_kind,
                "bytes_col": parquet_image_bytes_col if is_bytes_mode else None,
                "path_col": parquet_path_col if not is_bytes_mode else None,
                "path_root": path_root,
                "schema_fp": manifest_desc.get("schema_fp"),
                "min_w": min_w,
                "min_h": min_h,
                "max_aspect": max_aspect,
            }
            preview_hash = register_tracked_preview_manifest(stream, items, meta_core)
            register_tracked_materializations(stream, written)

            ok = sum(1 for it in items if it.ok)
            bad = len(items) - ok
            reasons = {}
            for it in items:
                if not it.ok:
                    reasons[it.reason] = reasons.get(it.reason, 0) + 1

            ws = [it.w for it in items if it.ok]
            hs = [it.h for it in items if it.ok]
            means = [it.mean for it in items if it.ok]
            stds = [it.std for it in items if it.ok]

            new_meta = {
                "manifest_root_hash": root_hash,
                "manifest_hash": manifest_hash,
                "preview_hash": preview_hash,
                "written": written,
                "ok_count": ok,
                "bad_count": bad,
                "bad_reasons": reasons,
                "sample_size_got": len(items),
                "sample_indices": idx,
                "size_stats": _pct_stats(ws, hs),
                "intensity_stats": {
                    "mean_p50": _percentile(means, 50),
                    "std_p50": _percentile(stds, 50),
                },
                "bytes_col": parquet_image_bytes_col if is_bytes_mode else None,
                "path_col": parquet_path_col if not is_bytes_mode else None,
                "schema_fp": manifest_desc.get("schema_fp"),
            }

            stream.emit(last_batch=True, meta=new_meta)
            stream.close_ok()

            # E) Return result
            meta = build_meta(
                {
                    "mode": "parquet_bytes" if is_bytes_mode else "paths",
                    "source": str(src_path),
                    "source_kind": source_kind,
                    "pattern": pattern if source_kind == "directory" else None,
                    "parquet_path_col": parquet_path_col
                    if source_kind == "parquet"
                    else None,
                    "bytes_col": parquet_image_bytes_col
                    if source_kind == "parquet"
                    else None,
                    "path_root": path_root,
                    "sample_size_req": sample_size,
                    "seed": seed,
                    "thumb_size": self.thumb_size,
                    "items": items,
                    "written": written,
                    "min_w": min_w,
                    "min_h": min_h,
                    "max_aspect": max_aspect,
                    "sample_indices": idx,
                    "manifest_root_hash": root_hash,
                    "preview_hash": preview_hash,
                    "manifest_hash": manifest_hash,
                    "population_meta": pop_meta,
                }
            )
            return PreviewResult(items=items, meta=meta)

        except Exception as e:
            stream.close_error(type(e), e, e.__traceback__)
            raise

    def plot_waveform(self, audio_file: str):
        """Plot waveform of an audio file."""
        pass

    # Save feature/target distributions as .json files for monitoring/prior storage
