from typing import Optional, Iterable, List, Dict, Any
import os

try:
    import psutil
except ImportError:
    psutil = None


def memory_aware_batcher(
    file_path: str,
    backend: str = "pandas",
    safety_factor: float = 0.2,
    verbose: bool = False,
    fallback_memory_limit_mb: int = 512
) -> int:
    """
    Estimate an optimal batch size (number of rows) for loading a CSV file,
    based on system memory and sampling average row size.

    Parameters:
    -----------
    file_path : str
        Path to the CSV file.
    backend : {"pandas", "polars"}
        Backend to use for memory estimation and sampling.
    safety_factor : float, default=0.2
        Proportion of available memory to use. E.g., 0.2 = use 20% of RAM.
    verbose : bool, default=False
        Whether to print debug information.
    fallback_memory_limit_mb : int, default=512
        If psutil is not available, assume this much memory (in MB) is usable.

    Returns:
    --------
    int
        Recommended batch size (in number of rows).
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Determine available memory
    if psutil is not None:
        available_memory = psutil.virtual_memory().available * safety_factor
    else:
        available_memory = fallback_memory_limit_mb * 1024 * 1024  # convert MB to bytes
        if verbose:
            print(f"[WARN] psutil not installed. Assuming {fallback_memory_limit_mb}MB available.")

    sample_rows = 1000
    if backend == "pandas":
        import pandas as pd
        sample_df = pd.read_csv(file_path, nrows=sample_rows)
        mem_usage = sample_df.memory_usage(deep=True).sum()
    elif backend == "polars":
        import polars as pl
        sample_df = pl.read_csv(file_path, n_rows=sample_rows)
        mem_usage = sample_df.estimated_size()
    else:
        raise ValueError(f"Unsupported backend: '{backend}'")

    memory_per_row = mem_usage / sample_rows
    estimated_batch_size = int(available_memory / memory_per_row)
    batch_size = max(estimated_batch_size, 1)

    if verbose:
        print(f"[INFO] Estimated memory per row: {memory_per_row:.2f} bytes")
        print(f"[INFO] Available memory used: {available_memory / (1024**2):.2f} MB")
        print(f"[INFO] Recommended batch size: {batch_size} rows")

    return batch_size


def load_batches_pandas(
    file_path: str,
    batch_size: int,
    use_cols: Optional[List[str]] = None,
    filter_by: Optional[List[Dict[str, Any]]] = None,
) -> Iterable[Any]:
    """
    Load a CSV in memory-safe batches using pandas, with optional filtering on each chunk.

    Parameters:
    -----------
    file_path : str
        Path to the CSV file to load.
    batch_size : int
        Number of rows per batch.
    use_cols : list of str, optional
        Specific columns to load from the file. If None, all columns are loaded.
    filter_by : list of dict, optional
        List of filtering conditions to apply after each chunk is loaded.
        Example: [{"col": "country", "value": "USA"}, {"col": "year", "value": 2020}]

    Yields:
    -------
    pd.DataFrame
        A filtered or unfiltered batch of rows.
    """

    import pandas as pd

    try:
        chunk_iter = pd.read_csv(file_path, chunksize=batch_size, usecols=use_cols)

        for chunk in chunk_iter:
            if filter_by:
                for condition in filter_by:

                    col = condition.get("col")
                    val = condition.get("value")

                    if col not in chunk.columns:
                        raise ValueError(f"[Pandas Backend] Column '{col}' not found in loaded columns.")
                    
                    if isinstance(val, list):
                        chunk = chunk[chunk[col].isin(val)]
                    else:
                        chunk = chunk[chunk[col] == val]

            if not chunk.empty:
                yield chunk

    except Exception as e:
        raise RuntimeError(f"[Pandas Backend] Failed to load CSV in batches: {e}")
    
    
def load_batches_polars(
        file_path: str, 
        batch_size: int, 
        use_cols: Optional[List[str]] = None,
        filter_by: Optional[List[Dict[str, Any]]] = None
) -> Iterable[Any]:
    """
    Stream batches from a CSV using Polars with optional column filters.

    Parameters:
    -----------
    file_path : str
        Path to the CSV file.
    use_cols : list of str, optional
        Specific columns to load from the file. If None, all columns are loaded.
    batch_size : int
        Number of rows per batch.
    filter_by : list of dicts, optional
        Each dict should have "col" and "value" keys for filtering.
        Example: [{"col": "country", "value": "USA"}]

    Yields:
    -------
    pl.DataFrame
        A Polars DataFrame for each batch.
    """

    import polars as pl

    try:
        scan_csv = pl.read_csv(file_path, infer_schema_length=1000, columns=use_cols)
        total_rows = scan_csv.height

        for i in range(0, total_rows, batch_size):
            batch = scan_csv.slice(i, batch_size)

            if filter_by:
                for f in filter_by:
                    col = f["col"]
                    val = f["value"]

                    if col not in batch.columns:
                        raise ValueError(f"Column '{col}' not found in Polars DataFrame.")
                    
                    if isinstance(val, list):
                        batch = batch.filter(pl.col(col).is_in(val))
                    else:
                        batch = batch.filter(pl.col(col) == val)

            # Ensure batch is still a DataFrame (not a Series)
            if isinstance(batch, pl.Series):
                batch = batch.to_frame()

            if batch.is_empty():
                continue

            yield batch

    except Exception as e:
        raise RuntimeError(f"[Polars Backend] Failed to load batches from CSV: {e}")
