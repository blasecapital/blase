from typing import Callable, Iterable, Dict, Any, Optional, List
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from blase.track import Track
from blase.utils.hashing import Hash

from blase.extracting.csv_backend import memory_aware_batcher, read_batches_pandas, read_batches_polars
from blase.utils.backends import resolve_backend


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
        track: bool = True
    ) -> Iterable[Any]:
        """
        Read a CSV file in memory-safe batches using the specified backend.

        This method streams data in chunks, allowing for efficient processing of large CSV files
        that may not fit into memory. It supports auto-calculated or manually specified batch sizes,
        and filters can be applied to restrict which rows are read.

        Parameters:
        -----------
        file_path : str
            Path to the CSV file.

        mode : {"auto", "manual"}, default="auto"
            Determines how batch size is handled:
            - "auto": Uses a backend memory estimator to determine optimal batch size.
            - "manual": Uses the provided `batch_size` parameter (required in this mode).

        batch_size : int, optional
            Number of rows per batch (required if mode is "manual").

        use_cols : ["col1", "col2"], optional
            A list of column name strings.
            Filters what columns to read.

        filter_by : list, optional
            A list of column-level filters (e.g., [{"col": "country", "value": "USA"}]).
            Applied after each chunk is read. May be ignored by some backends.

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
        >>> for batch in extractor.read_csv(file_path="data/large.csv", mode="auto", backend="pandas"):
        >>>     process_batch(batch)

        Notes:
        ------
        - This function does not read the entire dataset into memory.
        - Users can apply filters post-read in custom transformation steps if needed.
        - filter_by filters the data in the order of the dicts
        - Backends must be installed separately (e.g., `pip install polars`).
        """

        # tracking
        if track:
            track_dir = Track._validate_run_directory(track=track)
            run_path = Track._validate_active_run(track_dir=track_dir)

            # Hash file in background
            hasher = Hash()
            executor = ThreadPoolExecutor(max_workers=1)
            file_hash_future = executor.submit(hasher.hash_file, file_path)
            
            step_id, step_hash = Track._start_step(
                run_path=run_path,
                function="read_csv",
                params={
                    "file_path": str(file_path),
                    "mode": mode,
                    "batch_size": batch_size,
                    "use_cols": use_cols,
                    "filter_by": filter_by,
                    "backend": backend,
                    "track": track
                }
            )

        # dependency handling
        backend = resolve_backend(backend)

        # error handling
        if mode not in ("auto", "manual"):
            raise ValueError(f"Unsupported mode: {mode}. Must be 'auto' or 'manual'.")
        if mode == "manual" and not batch_size:
            raise ValueError("When mode is 'manual', batch_size must be specified.")
        if mode == "auto":
            batch_size = memory_aware_batcher(file_path, backend)

        if isinstance(filter_by, dict):
            filter_by = [filter_by]

        # main logic
        if track:
            try:
                if backend == "polars":
                    batch_gen = read_batches_polars(file_path, batch_size, use_cols, filter_by)
                    wrapped_gen = ((batch, flag, step_hash) for batch, flag in batch_gen)
                    yield from wrapped_gen

                elif backend == "pandas":
                    batch_gen = read_batches_pandas(file_path, batch_size, use_cols, filter_by)
                    wrapped_gen = ((batch, flag, step_hash) for batch, flag in batch_gen)
                    yield from wrapped_gen
                
                file_hash = file_hash_future.result()
                executor.shutdown()
                parent_dict = {
                    'parent': None,
                    'source_path': file_path,
                    'logged_by': step_id
                }
                Track._log_data(
                    run_path=run_path,
                    file_hash=file_hash,
                    parent_dict=parent_dict
                )

                Track._end_step(
                    step_hash=step_hash,
                    run_path=run_path,
                    status="completed",
                    outputs={},
                    file_hash=file_hash
                )

            except Exception as e:
                Track._end_step(
                    step_hash=step_hash,
                    run_path=run_path,
                    status="failed",
                    outputs={"error": str(e)}
                )
                raise
        else:
            if backend == "polars":
                yield from read_batches_polars(file_path, batch_size, use_cols, filter_by)

            elif backend == "pandas":
                yield from read_batches_pandas(file_path, batch_size, use_cols, filter_by)

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

        backend = resolve_backend(backend)

    def read_parquet(self, file_path: str, batch_size: int = None) -> Iterable[Any]: pass
    def read_hdf5(self, file_path: str, batch_size: int = None) -> Iterable[np.ndarray]: pass
    def read_orc(self, file_path: str, batch_size: int = None) -> Iterable[Any]: pass
    def read_excel(self, file_path: str, sheet_name: str = None, batch_size: int = None) -> Iterable[Any]: pass
    def read_tsv(self, file_path: str, batch_size: int = None) -> Iterable[Any]: pass
    def read_xml(self, file_path: str, batch_size: int = None) -> Iterable[dict]: pass
    def read_audio(self, file_path: str, batch_size: int = None) -> Iterable[np.ndarray]: pass
    def read_npy(self, file_path: str, batch_size: int) -> Iterable[np.ndarray]: pass
    def read_sql(self, query: str, connection, batch_size: int) -> Iterable[Any]: pass
    def read_images(self, directory: str, batch_size: int = None, filename_filter: Callable[[str], bool] = None): pass
    def read_api(
        self, 
        endpoint: str, 
        params: Dict[str, Any] = None, 
        headers: Dict[str, str] = None, 
        batch_size: int = 100, 
        max_pages: int = None, 
        rate_limit: float = 1.0
    ) -> Iterable[Any]: pass
