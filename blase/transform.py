from __future__ import annotations
from typing import Iterable

from blase.track import Track
from blase.types import Batch


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

    Example:
    --------
    >>> def custom_feature_engineering(batch):
    >>>     batch["new_feature"] = batch["feature1"] * batch["feature2"]
    >>>     return batch
    >>>
    >>> transformer = Transform()
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
        self, *, batch: Batch, transform_func, track=True, pass_batch: bool = False
    ) -> Batch:
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
        batch: Batch
            batch.data: Any
                The batch payload to transform (e.g., a pandas or polars DataFrame). The type is
                whatever the upstream Extract yielded.
            batch.is_last: bool
                ``True`` if this is the final batch for the stream. Triggers sealing the tracked
                step (when tracking is enabled) and resets internal stream state.
            batch.meta: dict or None, optional
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
        transform_func : callable
            User-supplied function that accepts ``data`` and returns the transformed batch.
            The function's qualified name (``__qualname__``) is used to group all batches
            of a stream into a single tracked step.
        track : bool, default True
            If ``True``, use the tracking system to record a single Transform step for the
            whole stream (per unique ``transform_func``), snapshotting code/env once and
            recording lineage on each emit. If ``False``, no tracking is performed.

        Returns
        -------
        Batch
            - batch.data : Any
            The result of ``transform_func(data)``.
            - batch.is_last : bool
            Echo of the input flag to enable downstream control flow.
            - batch.meta : dict
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
        >>> batch = tr.apply_function(
        ...     batch=batch, transform_func=normalize, track=True
        ... )
        """
        tracker = Track.get(track)
        arg = batch if pass_batch else batch.data

        if tracker is None:
            out = transform_func(arg)
            # allow Batch return for label/path edits
            if isinstance(out, Batch):
                return type(batch)(
                    data=out.data,
                    labels=out.labels if out.labels is not None else batch.labels,
                    paths=out.paths if out.paths is not None else batch.paths,
                    is_last=batch.is_last,
                    meta=dict(batch.meta or {}, **(out.meta or {})),
                )
            return Batch(
                data=out,
                labels=batch.labels,
                paths=batch.paths,
                is_last=batch.is_last,
                meta=dict(batch.meta or {}),
            )

        fn_qual = getattr(
            transform_func,
            "__qualname__",
            getattr(transform_func, "__name__", "callable"),
        )

        # If new stream or function changed, (re)open a StreamStep and stash replay hints
        if (
            getattr(self, "_stream", None) is None
            or getattr(self, "_fn_tag", None) != fn_qual
        ):
            # Close any prior open step cleanly
            if getattr(self, "_stream", None) is not None:
                self._stream.close_ok()

            m = batch.meta or {}
            params = {
                "fn_qualname": fn_qual,
                "pass_batch": bool(pass_batch),
            }

            bs = m.get("chunk_size")
            if bs is None:
                bs = m.get("batch_size")
            if bs is not None:
                params["batch_size"] = int(bs)

            # backend aliasing (old runs used 'reader_backend')
            be = m.get("reader_backend") or m.get("backend")
            if be is not None:
                params["backend"] = be

            # keep existing CSV hints
            for k in ("use_cols", "filter_by"):
                if m.get(k) is not None:
                    params[k] = m[k]

            # image-ish hints
            for k in ("return_type", "color", "max_side", "mode", "target_batch_bytes"):
                if m.get(k) is not None:
                    params[k] = m[k]

            self._stream = tracker.stream(
                "blase.Transform.apply_function", params, code_fn=transform_func
            )
            m = batch.meta or {}
            up = m.get("upstream") or []
            for u in up:
                rid = u.get("id")
                role = u.get("role")
                if rid and role in {"manifest", "batch_desc", "batch"}:
                    try:
                        self._stream.step.add_input(rid, role=role, arg_name=None)
                    except Exception:
                        pass  # tolerate replays/duplicates

            # you can also store manifest root for convenience (optional)
            root = m.get("manifest_root_hash") or m.get("root_hash")
            if root:
                try:
                    self._stream.step.add_input(
                        root, role="manifest_root", arg_name=None
                    )
                except Exception:
                    pass
            self._fn_tag = fn_qual

        try:
            out = transform_func(arg)
        except Exception as e:
            self._stream.close_error(type(e), e, e.__traceback__)
            self._stream = None
            self._fn_tag = None
            raise

        self._stream.step.add_upstream_from_meta(batch.meta)
        new_meta = self._stream.emit(last_batch=batch.is_last, meta=batch.meta)

        if batch.is_last:
            self._stream.close_ok()
            self._stream = None
            self._fn_tag = None

        # accept Batch return
        if isinstance(out, Batch):
            return type(batch)(
                data=out.data,
                labels=out.labels if out.labels is not None else batch.labels,
                paths=out.paths if out.paths is not None else batch.paths,
                is_last=batch.is_last,
                meta={**(new_meta or {}), **(out.meta or {})},
            )

        # legacy: data-only return
        return Batch(
            data=out,
            labels=batch.labels,
            paths=batch.paths,
            is_last=batch.is_last,
            meta=new_meta,
        )

    def apply_from_module(
        self, data: Iterable, module_path: str, function_name: str
    ) -> Iterable:
        pass

    def apply_standard_transformation(self, data, transformation: str, columns: list):
        pass

    # Maybe include a fill missing values explicitly here or include it in standard transformation
