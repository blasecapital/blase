from __future__ import annotations
from typing import Callable, Iterable, Any, Optional, Dict

from blase.track import Track

class Transform:
    """
    A utility class for applying user-defined and built-in transformations to batched datasets.

    The Transform class standardizes how transformations are applied to data during preprocessing
    and feature engineering. It is designed to work with batched data to support memory-efficient
    workflows and reproducible pipelines.

    Primary Use Cases:
    ------------------
    - Apply a user-defined function to each data batch
    - Dynamically load and apply a transformation from an external Python module
    - Use built-in transformation utilities for common cleaning and engineering tasks
    - Log applied transformations for traceability (optional)

    Methods:
    --------
    apply_function(data: Iterable, transform_func: Callable) -> Iterable:
        Applies a user-defined function to batched data.

    apply_from_module(data: Iterable, module_path: str, function_name: str) -> Iterable:
        Dynamically loads a function from a specified module and applies it to batched data.

    apply_standard_transformation(data, transformation: str, columns: list):
        Applies a built-in transformation such as standard scaling, min-max scaling, or one-hot encoding.

    Logging:
    --------
    - Set `enable_logging=True` when instantiating Transform to enable logging.
    - Logging outputs which transformation was applied to each batch.
    - Users can also choose to log to a file using `log_to_file=True` and specify a path with `log_file`.

    Example:
    --------
    >>> def custom_feature_engineering(batch):
    >>>     batch["new_feature"] = batch["feature1"] * batch["feature2"]
    >>>     return batch
    >>>
    >>> transformer = Transform(enable_logging=True)
    >>> transformed_data = transformer.apply_function(extracted_batches, custom_feature_engineering)
    >>> for batch in transformed_data:
    >>>     process(batch)

    Example Using External Module:
    ------------------------------
    >>> transformer = Transform()
    >>> transformed_data = transformer.apply_from_module(extracted_batches, "feature_engineering", "scale_features")
    >>> for batch in transformed_data:
    >>>     process(batch)

    Notes:
    ------
    - Works as an **intermediate step** between data extraction and model training.
    - Provides flexibility to use **either inline functions, classes, or external scripts** for transformations.
    """
    def __init__(self):
        self._stream = None
        self._fn_tag = None

    def apply_function(
        self,
        *,
        data,
        transform_func,
        last_batch,
        meta=None,
        track=True
    ):
        """
        Apply a user-supplied transform to a stream of batches.
        Records enough info in the step params so restore can faithfully replay.
        """
        tracker = Track.get(track)

        # Untracked path: just run the function and propagate meta.
        if tracker is None:
            out = transform_func(data)
            return out, last_batch, (meta or {})

        # Identify the callable for stream grouping (one step per function per stream)
        fn_qual = getattr(transform_func, "__qualname__", getattr(transform_func, "__name__", "callable"))

        # If new stream or function changed, (re)open a StreamStep and stash replay hints
        if getattr(self, "_stream", None) is None or getattr(self, "_fn_tag", None) != fn_qual:
            # Close any prior open step cleanly
            if getattr(self, "_stream", None) is not None:
                self._stream.close_ok()

            # Pull replay hints from meta (provided by Extract.read_csv) — all optional
            m = meta or {}
            params = {
                "fn_qualname": fn_qual,
                # pull from Extract meta; ok if None, restore will still have a fallback
                "batch_size": m.get("chunk_size"),
                "reader_backend": m.get("reader_backend"),
                "use_cols": m.get("use_cols"),
                "filter_by": m.get("filter_by"),
            }

            # Open one tracked step for the whole stream; snapshot code/env once
            self._stream = tracker.stream("blase.Transform.apply_function", params, code_fn=transform_func)
            self._fn_tag = fn_qual

        # Execute user code
        try:
            out = transform_func(data)
        except Exception as e:
            # mark failed/aborted and reset stream state
            self._stream.close_error(type(e), e, e.__traceback__)
            self._stream = None
            self._fn_tag = None
            raise

        # Record upstream lineage for this batch; seal on last
        new_meta = self._stream.emit(last_batch=last_batch, meta=meta)

        if last_batch:
            self._stream.close_ok()
            self._stream = None
            self._fn_tag = None

        return out, last_batch, new_meta

    def apply_from_module(self, data: Iterable, module_path: str, function_name: str) -> Iterable: pass
    def apply_standard_transformation(self, data, transformation: str, columns: list): pass
    # Maybe include a fill missing values explicitly here or include it in standard transformation
