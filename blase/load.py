from __future__ import annotations
from typing import Any, Optional, Dict, Tuple, Callable
from pathlib import Path

import numpy as np

from blase.loading.csv_backend import save_batch_pandas, save_batch_polars
from blase.utils.backends import resolve_backend
from blase.track import Track

class Load:
    """
    Handles storage of processed data in big-data-friendly formats and locations.

    This module is designed to store structured and unstructured data efficiently by detecting the data type
    and choosing an ideal format by default. Users can override storage preferences, validate schemas,
    and track save events via optional metadata logging.

    Core Responsibilities:
    ----------------------
    - Auto-detect data type (e.g., DataFrame, ndarray) and choose a default storage strategy.
    - Save structured data to local or remote databases (SQLite, DuckDB, PostgreSQL, etc.).
    - Save unstructured or semi-structured data to JSON, Parquet, CSV, or binary formats like NPY.
    - Enable users to pass a custom function to route or save files (useful for file-based storage).
    - Optionally log metadata (timestamp, schema, hash, etc.) with each save for reproducibility.
    - Provide optional schema validation to ensure structure consistency across saves.

    Supported Storage Types:
    ------------------------
    - Tabular formats: CSV, Parquet, SQLite, DuckDB, PostgreSQL
    - Binary formats: NPY, HDF5
    - Semi-structured: JSON
    - Unstructured: File system-based image/text/audio storage
    - Extensible to cloud: S3, GCS, Azure Blob (via user extension)

    Parameters:
    -----------
    - data (Any): The processed batch or dataset to store.
    - destination (str): File path, table name, or custom route.
    - data_type (str): Type of data ("auto", "csv", "sqlite", etc.).
    - custom_file_saver (Callable): Optional user function to handle save logic for files.
    - validate_schema (bool): Whether to validate the schema against a saved reference.
    - schema_path (str): Path to schema file for validation.
    - metadata (dict): Additional metadata to store with the save event.

    Example:
    --------
    >>> loader = Load()
    >>> loader.save(data=df, destination="my_data", data_type="auto")
    >>> loader.save_to_sqlite(df, db_path="my.db", table="my_table", mode="append")
    >>> loader.save(data=batch, custom_file_saver=my_custom_routing_function)

    Notes:
    ------
    - By default, the module infers the best storage format from the input data.
    - Schema validation and metadata logging are optional but enhance pipeline safety and traceability.
    """
    def __init__(self):
        self._stream = None
        self._target_path: Optional[Path] = None
        self._wrote_header: bool = False
        self._backend: Optional[str] = None

    def _resolve_target_path(
        self,
        *,
        tracker: Track,
        use_blase_path: bool,
        path: Optional[str],
        file_name: Optional[str],
        subdir: Optional[str],
    ) -> Path:
        if use_blase_path:
            # project working dir, not runs/*
            base = tracker.paths["data_working"]
            if subdir:
                base = base / subdir
            base.mkdir(parents=True, exist_ok=True)
            if not file_name:
                raise ValueError("file_name must be provided when use_blase_path=True")
            return (base / file_name).resolve()
        else:
            if not path:
                raise ValueError("path is required when use_blase_path=False")
            p = Path(path).resolve()
            p.parent.mkdir(parents=True, exist_ok=True)
            return p
    
    # --- Main API ---

    def save(self, 
             data: Any, 
             destination: str, 
             data_type: str = "auto", 
             custom_file_saver: Callable = None, 
             validate_schema: bool = False, 
             schema_path: str = None, 
             metadata: dict = None):
        pass

    def save_to_csv(
        self,
        *,
        data: Any,
        last_batch: bool,
        meta: Optional[Dict[str, Any]] = None,   # pass upstream lineage from Extract/Transform
        path: Optional[str] = None,
        file_name: Optional[str] = None,
        subdir: Optional[str] = "csv_data",
        backend: str = "pandas",
        track: bool = True,
        use_blase_path: bool = True,
    ) -> Tuple[str, bool, Dict[str, Any]]:
        """
        Write one batch of data to a CSV target with optional tracking and lineage.

        This method appends ``data`` to a target CSV file in either *tracked* or
        *untracked* mode. In tracked mode, a single `Load.save_to_csv` stream is opened
        per unique (target, backend) combination, recording upstream lineage, seed
        inputs (if the file already exists), and registering the final materialized file
        in CAS on the last batch. In untracked mode, the file is simply written using
        backend I/O and no lineage is recorded.

        Parameters
        ----------
        data : Any
            A batch of records to append to the CSV file. Typically a `pandas.DataFrame`
            or `polars.DataFrame`, depending on the backend chosen.
        last_batch : bool
            Whether this batch is the final batch in the stream. If ``True``, the stream
            is sealed and the completed file is registered as an output.
        meta : dict, optional
            Metadata propagated from upstream steps (e.g., from Extract or Transform).
            This information is recorded per batch when tracking is enabled.
        path : str, optional
            Absolute or relative path to the output file. Mutually exclusive with
            ``file_name``/``subdir`` if ``use_blase_path=True``.
        file_name : str, optional
            Optional explicit file name to use under ``subdir`` when resolving the target.
        subdir : str, default="csv_data"
            Subdirectory under the run’s output root to place the CSV target. Ignored if
            ``use_blase_path=False`` and ``path`` is explicitly provided.
        backend : {"pandas", "polars"}, default="pandas"
            Backend library to use for writing CSV shards. Controls both the file-writing
            logic and the identity of the stream step.
        track : bool, default=True
            Whether to engage the Blase tracking system. If ``False``, no DAG lineage or
            CAS registration is performed.
        use_blase_path : bool, default=True
            If ``True``, resolve the target path relative to the Blase run output root
            using ``subdir``/``file_name``. If ``False``, use ``path`` directly.

        Returns
        -------
        tuple of (str, bool, dict)
            A 3-tuple containing:
            - ``target_path`` : str  
            Filesystem path of the CSV file being written to.
            - ``last_batch`` : bool  
            Echo of the input flag, enabling downstream flow control.
            - ``new_meta`` : dict  
            Metadata returned by the tracking system’s emit call if tracking is enabled,
            otherwise the input ``meta`` or an empty dict.

        Notes
        -----
        - **Stream semantics:** One `Load.save_to_csv` stream is maintained per unique
        (target, backend). If the target path or backend changes mid-run, the previous
        stream is cleanly closed and a new stream is opened.
        - **Seed inputs:** If the target file already exists when the stream is opened,
        its contents are snapshotted and registered as a "seed" input. This allows
        append semantics to be faithfully restored during replay.
        - **Lineage:** Each batch inherits upstream metadata via ``meta`` and is logged
        through the stream step’s emit call.
        - **Finalization:** On ``last_batch=True``, the completed file is registered in CAS
        as an output artifact of kind "csv", then the step is sealed.
        - **Untracked path:** When ``track=False``, this method only writes the data to
        the target file using the chosen backend. No tracking, CAS registration, or
        lineage propagation occurs.

        Raises
        ------
        Exception
            Any exception raised during CSV writing propagates upward. In tracked mode,
            the current stream is marked failed/aborted and internal state is reset.

        Examples
        --------
        >>> loader = Load()
        >>> for batch, is_last, meta in pipeline:
        ...     target, last, out_meta = loader.save_to_csv(
        ...         data=batch, last_batch=is_last, meta=meta,
        ...         file_name="output.csv", backend="pandas"
        ...     )
        >>> print("Final output:", target)
        """
        backend = resolve_backend(backend)
        tracker = Track.get(track)

        # ---------- Untracked fast-path ----------
        if tracker is None:
            target = self._resolve_target_path(
                tracker=Track.get(True),  # for path resolution only
                use_blase_path=use_blase_path,
                path=path, file_name=file_name, subdir=subdir,
            )
            file_exists = target.exists()
            (save_batch_pandas if backend == "pandas" else save_batch_polars)(data, target, file_exists)
            return str(target), last_batch, (meta or {})

        # ---------- Tracked path ----------
        target = self._resolve_target_path(
            tracker=tracker,
            use_blase_path=use_blase_path,
            path=path,
            file_name=file_name,
            subdir=subdir,
        )

        # One StreamStep per (target, backend)
        new_target = (self._target_path is None) or (self._target_path != target)
        new_backend = (self._backend is None) or (self._backend != backend)

        if self._stream is None or new_target or new_backend:
            # close any prior stream cleanly
            if self._stream is not None:
                self._stream.close_ok()

            params = {
                "target": str(target),
                "backend": backend,
                "subdir": subdir,
                "use_blase_path": bool(use_blase_path),
            }
            # snapshot the callable/env once per stream
            self._stream = tracker.stream("blase.Load.save_to_csv", params, code_fn=self.save_to_csv)
            self._target_path = target
            self._backend = backend

            # If target exists, snapshot its current bytes as a "seed" input
            self._stream.step.remove_inputs_by_role("seed")
            if target.exists():
                seed_hash = self._stream.step.register_data(
                    kind="csv", version="1", path_or_bytes=target,
                    metadata={"role": "seed", "target": str(target)}
                )
                self._stream.step.add_input(seed_hash, role="seed", arg_name=None)

        # 1) write the shard via your csv_backend functions (exactly like original)
        file_exists = target.exists()
        try:
            if backend == "pandas":
                save_batch_pandas(data, target, file_exists)
            else:
                save_batch_polars(data, target, file_exists)
        except Exception as e:
            # record failure and close context
            if self._stream is not None:
                self._stream.close_error(type(e), e, e.__traceback__)
                self._stream = None
                self._target_path = None
                self._backend = None
            raise

        # 2) per-batch lineage + seal on last batch
        new_meta = self._stream.emit(last_batch=last_batch, meta=meta)

        # 3) on final batch, register final file and close the stream
        if last_batch:
            out_hash = self._stream.step.register_data(
                kind="csv", version="1", path_or_bytes=target, metadata={"target": str(target)}
            )
            self._stream.step.add_output(out_hash, name="csv")

            self._stream.close_ok()
            self._stream = None
            self._target_path = None
            self._backend = None

        return str(target), last_batch, new_meta

    def save_to_parquet(self, data: Any, path: str, mode: str = "overwrite"):
        pass

    def save_to_npy(self, data: np.ndarray, path: str):
        pass

    def save_to_json(self, data: Any, path: str):
        pass

    def save_to_sqlite(self, data: Any, db_path: str, table: str, mode: str = "append"):
        pass

    def save_to_duckdb(self, data: Any, db_path: str, table: str, mode: str = "append"):
        pass

    def save_to_postgresql(self, data: Any, conn_params: dict, table: str, mode: str = "append"):
        pass

    def save_to_filesystem(self, data: Any, directory: str, filename_fn: Callable):
        pass

    def save_metadata(self, metadata: dict, destination: str):
        pass

    def validate_schema(self, data: Any, schema_path: str):
        pass
