from pathlib import Path
from typing import Any

def save_batch_pandas(df: Any, path: Path, file_exists: bool):
    """
    Save a batch of data to a CSV file using Pandas.

    If the file exists, the batch is appended without duplicating the header.
    If the file does not exist, it is created with a header.

    Args:
        df (Any): The data batch to save.
        path (Path): The file path to write to.
        file_exists (bool): Whether the file already exists.
    """
    df.to_csv(
        path,
        mode="a" if file_exists else "w",
        header=not file_exists,
        index=False
    )
    return path

def save_batch_polars(df: Any, base_path: Path, file_exists: bool):
    """
    Save a batch of data to a uniquely numbered CSV file using Polars.

    Since Polars doesn't support true appending to CSV, each batch is saved
    as a new file with an incrementing suffix (e.g., `data_001.csv`, `data_002.csv`, etc.).

    Args:
        df (Any): The data batch to save.
        base_path (Path): Base file path (e.g., 'data/output.csv').
        file_exists (bool): Flag indicating whether any previous file has been saved.
    """
    base_dir = base_path.parent
    base_name = base_path.stem
    base_ext = base_path.suffix or ".csv"

    base_dir.mkdir(parents=True, exist_ok=True)

    # Determine next available index
    existing_batches = list(base_dir.glob(f"{base_name}_*.csv"))
    next_id = len(existing_batches) + 1
    batch_path = base_dir / f"{base_name}_{next_id:03d}{base_ext}"

    df.write_csv(batch_path)
    return batch_path
