from typing import Union, Literal, Optional, List, Dict, Any
import random
from pathlib import Path
import json

import numpy as np
import pandas as pd

from blase.track import Track
from blase.types import PreviewResult
from blase.examining.preview_image_backend import (
    parquet_bytes_available,
    sample_parquet_bytes,
    get_source,
    maybe_materialize_grid_bytes,
    maybe_materialize_thumbs_bytes,
    build_meta,
    sample_population,
    probe_paths,
    build_manifest_and_hash,
    register_tracked_population,
    register_tracked_preview_manifest,
    register_tracked_materializations,
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
        allowed_ext: Optional[List[str]] = None,  # directory only
        parquet_path_col: Optional[str] = None,  # parquet: on-disk paths
        parquet_image_bytes_col: Optional[str] = None,  # parquet: in-file bytes
        path_root: Optional[str] = None,  # parquet: base dir for relative paths
        recursive: bool = True,
        sample_size: int = 9,
        seed: Optional[int] = 42,
        backend: Literal["pil"] = "pil",
        return_thumbs: bool = False,  # reserved (we do stats-only by default)
        head_read_bytes: int = 128 * 1024,
        max_side: Optional[int] = None,
        max_total_decode_bytes: Optional[
            int
        ] = None,  # guard; enforced in probe helpers
        save: bool = False,
        to: Optional[str] = None,  # grid image path
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
        Headless preview with deterministic sampling, mixed-type directory handling, and restore wiring.
        Returns stats and optional thumbnails; writes only if requested.
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
                     written.extend(maybe_materialize_thumbs_bytes(items, Path(out_dir), seed))
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
                written.extend(maybe_materialize_thumbs_bytes(items, Path(out_dir), seed))

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
        # Common params recorded on the step
        step_params = {
            "source": str(src_path),
            "source_kind": source_kind,
            "pattern": pattern if source_kind == "directory" else None,
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
            if (
                source_kind == "parquet"
                and parquet_image_bytes_col
                and parquet_bytes_available(str(src_path), parquet_image_bytes_col)
            ):
                idx, items = sample_parquet_bytes(
                    source=str(src_path),
                    bytes_col=parquet_image_bytes_col,
                    k=sample_size,
                    seed=seed,
                    head_read_bytes=head_read_bytes,
                    thumb_size=self.thumb_size,
                    max_side=max_side,
                )
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

            # C) Optional materializations
            written: List[str] = []
            if save and to and any(it.ok for it in items):
                wp = maybe_materialize_grid_bytes(items, Path(to))
                if wp:
                    written.append(wp)
            if save_individual and out_dir:
                written.extend(maybe_materialize_thumbs_bytes(items, Path(out_dir), seed))

            # D) Preview manifest (diagnostic) + emit
            meta_core = {
                "manifest_root_hash": root_hash,
                "sample_indices": idx,
                "seed": seed,
                "thumb_size": self.thumb_size,
            }
            preview_hash = register_tracked_preview_manifest(stream, items, meta_core)
            register_tracked_materializations(stream, written)

            new_meta = {
                "manifest_root_hash": root_hash,
                "preview_hash": preview_hash,
                "written": written,
            }
            stream.emit(last_batch=True, meta=new_meta)
            stream.close_ok()

            # E) Return result
            meta = build_meta(
                {
                    "mode": "parquet_bytes"
                    if (source_kind == "parquet" and parquet_image_bytes_col)
                    else "paths",
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
                    "population_meta": manifest_desc.get("population_meta"),
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
