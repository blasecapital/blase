from typing import Callable, Iterable, Dict, Any

import numpy as np
import pandas as pd


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
    
    def load_csv(self, file_path: str, batch_size: int = None) -> Iterable[pd.DataFrame]: pass
    def load_parquet(self, file_path: str, batch_size: int = None) -> Iterable[pd.DataFrame]: pass
    def load_hdf5(self, file_path: str, batch_size: int = None) -> Iterable[np.ndarray]: pass
    def load_orc(self, file_path: str, batch_size: int = None) -> Iterable[pd.DataFrame]: pass
    def load_excel(self, file_path: str, sheet_name: str = None, batch_size: int = None) -> Iterable[pd.DataFrame]: pass
    def load_tsv(self, file_path: str, batch_size: int = None) -> Iterable[pd.DataFrame]: pass
    def load_xml(self, file_path: str, batch_size: int = None) -> Iterable[dict]: pass
    def load_audio(self, file_path: str, batch_size: int = None) -> Iterable[np.ndarray]: pass
    def load_npy(self, file_path: str, batch_size: int) -> Iterable[np.ndarray]: pass
    def load_sql(self, query: str, connection, batch_size: int) -> Iterable[pd.DataFrame]: pass
    def load_images(self, directory: str, batch_size: int = None, filename_filter: Callable[[str], bool] = None): pass
    def load_api(
        self, 
        endpoint: str, 
        params: Dict[str, Any] = None, 
        headers: Dict[str, str] = None, 
        batch_size: int = 100, 
        max_pages: int = None, 
        rate_limit: float = 1.0
    ) -> Iterable[pd.DataFrame]: pass
