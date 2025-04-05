from typing import Any, Callable
from pathlib import Path

import numpy as np

from blase.track import Track
from blase.utils.hashing import Hash
from blase.utils.backends import resolve_backend
from blase.loading.csv_backend import save_batch_pandas, save_batch_polars


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
            data: Any, 
            path: str,
            backend: str = "polars",
            track: bool = True
    ) -> None:
        """
        Saves a Pandas or Polars DataFrame to a CSV file.

        If the file exists, appends the data without duplicating headers.
        If the file does not exist, creates it with headers.

        Args:
            data (Any): A pandas or polars DataFrame.
            path (str): Path to save the CSV.
            backend : {"pandas", "polars"}, default="polars"
            track (bool): Whether to track this save operation.
        """

        path_obj = Path(path)
        file_exists = path_obj.exists()
        backend = resolve_backend(backend)

        # Tracking
        if track:
            track_dir = Track._validate_run_directory(track=track)
            run_path = Track._validate_active_run(track_dir=track_dir)

            hasher = Hash()
            pre_save_hash = hasher.hash_file(path_obj) if file_exists else None

            inputs = {
                "file_path": str(path_obj),
                "pre_save_hash": pre_save_hash,
                "file_exists": file_exists
            }

            step_id, step_file_path = Track._start_step(
                run_path=run_path,
                function="save_to_csv",
                params={
                    "backend": backend,
                    "file_path": str(path_obj)
                },
                inputs=inputs
            )

        # Write logic
        try:
            if backend == "polars":
                save_path = save_batch_polars(data, path_obj, file_exists)
            elif backend == "pandas":
                save_path = save_batch_pandas(data, path_obj, file_exists)

            if track:
                post_save_hash = hasher.hash_file(save_path)

                Track._end_step(
                    step_file_path=step_file_path,
                    status="completed",
                    outputs={
                        "backend": backend,
                        "post_save_hash": post_save_hash,
                        "file_path": str(save_path),
                        "file_size": save_path.stat().st_size
                    }
                )
        except Exception as e:
            if track:
                Track._end_step(
                    step_file_path=step_file_path,
                    status="failed",
                    outputs={"error": str(e)}
                )
            raise

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
