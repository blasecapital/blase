from typing import Callable, Iterable, Dict, Any, Optional, List

import numpy as np

from blase.extracting.csv_backend import memory_aware_batcher, load_batches_pandas, load_batches_polars
from blase.utils.backends import resolve_backend


class Extract:
    """
    Handles batch extraction of raw data from various local sources, with optional extension to cloud storage.
    It support two primary workflows:
        - Full-dataset iteration: Stream the entire dataset in memory-safe batches using a simple loop.
        - Targeted batch loading: For structured or file-based sources (e.g., databases, directories), 
          users can specify filters, WHERE clauses, or custom logic to load specific subsets of data.

    This module enables memory-efficient loading of structured and unstructured data by **processing in batches**.
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
    3. Loads data in **configurable batch sizes** to prevent memory overflow.
    4. Streams data **directly from the source** without full in-memory loading.
    5. Converts data into a **standard format** (e.g., DataFrame, NumPy, or raw text).

    Methods:
    --------
    load_csv(file_path: str, batch_size: int) -> Iterable[pd.DataFrame]
        Reads CSV in chunks and yields DataFrame batches.

    load_sql(query: str, connection, batch_size: int) -> Iterable[pd.DataFrame]
        Streams SQL query results in batches.

    load_images(directory: str, batch_size: int) -> Iterable[List[np.ndarray]]
        Loads images from a directory in batches.

    load_json(file_path: str, batch_size: int) -> Iterable[List[dict]]
        Parses JSON files in chunks.

    Example:
    --------
    >>> extractor = Extract()
    >>> for batch in extractor.load_csv("data.csv", batch_size=1000):
    >>>     process_batch(batch)  # Handle each batch separately

    Extending to Cloud:
    -------------------
    Users can implement cloud extractors by subclassing `Extract`:
    >>> class S3Extract(Extract):
    >>>     def load_s3(self, bucket_name, file_key, batch_size):
    >>>         pass  # Implement cloud-specific extraction logic

    Notes:
    ------
    - If batch size is not specified, the module defaults to an optimized chunk size.
    - This module will load the entire dataset if it's size is below 10% of available memory.
    """

    def __init__(self, track: bool = True) -> None:
        """
        Initialize the Extract module.

        Args:
            track (bool): Whether to enable automatic logging for extraction steps.
                        Can be overridden per method call. Defaults to True.
        """
        self.track = track

    def load_csv(
        self,
        file_path: str,
        mode: str = "auto",
        batch_size: Optional[int] = None,
        use_cols: Optional[List[str]] = None,
        filter_by: Optional[list] = None,
        backend: str = "pandas",
        track: bool = True
    ) -> Iterable[Any]:
        """
        Load a CSV file in memory-safe batches using the specified backend.

        This method streams data in chunks, allowing for efficient processing of large CSV files
        that may not fit into memory. It supports auto-calculated or manually specified batch sizes,
        and filters can be applied to restrict which rows are loaded.

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
            Filters what columns to load.

        filter_by : list, optional
            A list of column-level filters (e.g., [{"col": "country", "value": "USA"}]).
            Applied after each chunk is loaded. May be ignored by some backends.

        backend : {"pandas", "polars"}, default="pandas"
            The data handling library to use for loading and parsing.
            - "pandas": Standard and robust, most compatible.
            - "polars": Optimized for speed and memory efficiency (if installed).

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
        >>> for batch in extractor.load_csv("data/large.csv", mode="auto", backend="polars"):
        >>>     process_batch(batch)

        Notes:
        ------
        - This function does not load the entire dataset into memory.
        - Users can apply filters post-load in custom transformation steps if needed.
        - filter_by filters the data in the order of the dicts
        - Backends must be installed separately (e.g., `pip install polars`).
        """

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
        if backend == "polars":
            yield from load_batches_polars(file_path, batch_size, use_cols, filter_by)

        elif backend == "pandas":
            yield from load_batches_pandas(file_path, batch_size, use_cols, filter_by)

    def load_parquet(self, file_path: str, batch_size: int = None) -> Iterable[Any]: pass
    def load_hdf5(self, file_path: str, batch_size: int = None) -> Iterable[np.ndarray]: pass
    def load_orc(self, file_path: str, batch_size: int = None) -> Iterable[Any]: pass
    def load_excel(self, file_path: str, sheet_name: str = None, batch_size: int = None) -> Iterable[Any]: pass
    def load_tsv(self, file_path: str, batch_size: int = None) -> Iterable[Any]: pass
    def load_xml(self, file_path: str, batch_size: int = None) -> Iterable[dict]: pass
    def load_audio(self, file_path: str, batch_size: int = None) -> Iterable[np.ndarray]: pass
    def load_npy(self, file_path: str, batch_size: int) -> Iterable[np.ndarray]: pass
    def load_sql(self, query: str, connection, batch_size: int) -> Iterable[Any]: pass
    def load_images(self, directory: str, batch_size: int = None, filename_filter: Callable[[str], bool] = None): pass
    def load_api(
        self, 
        endpoint: str, 
        params: Dict[str, Any] = None, 
        headers: Dict[str, str] = None, 
        batch_size: int = 100, 
        max_pages: int = None, 
        rate_limit: float = 1.0
    ) -> Iterable[Any]: pass
