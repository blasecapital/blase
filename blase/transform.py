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
        Apply a user transform to one batch in a tracked stream, propagating lineage.

        This method is designed to be called once per incoming batch in an Extract→Transform→Load
        pipeline. It applies ``transform_func`` to ``data`` and returns a triple
        ``(out_batch, last_batch, new_meta)``. When ``track=True``, it opens (or reuses) a single
        tracked step for the entire stream **per unique transform function**, snapshots the
        function's code and environment once, records upstream lineage for each batch, and seals
        the step on the final batch. When ``track=False``, no tracking side-effects occur and the
        input ``meta`` is passed through (or `{}` if ``None``).

        Parameters
        ----------
        data : Any
            The batch payload to transform (e.g., a pandas or polars DataFrame). The type is
            whatever the upstream Extract yielded.
        transform_func : callable
            User-supplied function that accepts ``data`` and returns the transformed batch.
            The function's qualified name (``__qualname__``) is used to group all batches
            of a stream into a single tracked step.
        last_batch : bool
            ``True`` if this is the final batch for the stream. Triggers sealing the tracked
            step (when tracking is enabled) and resets internal stream state.
        meta : dict or None, optional
            Per-batch metadata propagated from upstream (e.g., the Extract step). Common
            fields include:
            ``{"upstream": [{"id": <source_hash>, "role": "source"}],
            "producer_step": <extract_step_hash>,
            "ordinal": <int>,
            "chunk_size": <int>,
            "reader_backend": <"pandas"|"polars">,
            "use_cols": <list|None>,
            "filter_by": <list|None>}``.
            These hints are copied into the Transform step's params on the first batch to
            facilitate deterministic restore/replay. May be ``None``.
        track : bool, default True
            If ``True``, use the tracking system to record a single Transform step for the
            whole stream (per unique ``transform_func``), snapshotting code/env once and
            recording lineage on each emit. If ``False``, no tracking is performed.

        Returns
        -------
        tuple
            A 3-tuple ``(out, last_batch, new_meta)``:
            - ``out`` : Any
            The result of ``transform_func(data)``.
            - ``last_batch`` : bool
            Echo of the input flag to enable downstream control flow.
            - ``new_meta`` : dict
            When tracking, the metadata returned by the stream step's ``emit`` (includes
            lineage and this transform's producer step). When not tracking, ``meta`` is
            passed through or replaced with an empty dict.

        Notes
        -----
        - **Stream grouping:** One tracked step is opened per stream *and* per unique
        ``transform_func`` (based on ``__qualname__``). If the function object changes
        mid-stream, the previous step is cleanly closed and a new step is opened.
        - **Code/env snapshot:** On the first batch for a given function, the function code
        and environment are snapshotted exactly once and associated to the step to enable
        faithful replay.
        - **Lineage propagation:** On every batch, upstream lineage from ``meta`` is recorded.
        On ``last_batch=True``, the step is sealed and internal state is reset.
        - **Untracked mode:** When ``track=False``, this method is a thin wrapper around
        calling ``transform_func`` and returning metadata unchanged (or ``{}``).

        Raises
        ------
        Exception
            Any exception raised by ``transform_func`` is propagated. In tracked mode, the
            step is marked failed/aborted and internal stream state is reset before re-raising.

        Examples
        --------
        Basic usage with tracking:

        >>> tr = Transform()
        >>> def normalize(df):
        ...     df = df.copy()
        ...     df["x_norm"] = df["x"] / df["x"].abs().max()
        ...     return df
        >>> out, last, meta = tr.apply_function(
        ...     data=batch, transform_func=normalize, last_batch=is_last, meta=up_meta, track=True
        ... )
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
