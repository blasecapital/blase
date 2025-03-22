from typing import Callable, List, Optional, Any


class Clean:
    """
    Identifies and tracks undesirable data for removal or exclusion across multiple datasets,
    ensuring consistent alignment of samples throughout the ML pipeline.

    The `Clean` module enables users to define custom logic for detecting invalid, irrelevant,
    or misaligned data across structured and unstructured formats. Rather than directly modifying
    the data, `Clean` focuses on flagging and recording unwanted sample keys—such as row indices,
    file names, or IDs—so that alignment is preserved and reproducibility is maintained.

    It is especially useful when working with disconnected datasets (e.g., features, targets, metadata),
    where sample-level alignment must be preserved even if invalid samples are identified in only
    one of the datasets.

    Key Responsibilities:
    ---------------------
    - Accept custom filtering functions to detect samples to exclude.
    - Track and persist identifiers (keys) of unwanted samples.
    - Ensure consistent removal of unwanted keys across all data sources.
    - Support soft filtering (ignore at later stages) or hard filtering (drop immediately).
    - Interface with `Prepare` to exclude invalid samples before training-ready data is created.

    This module does not modify the content of the data. Tasks like missing value imputation,
    image resizing, or feature correction should be handled in the `Transform` module.

    Methods:
    --------
    apply_filter(batch: Any, filter_fn: Callable) -> None
        Applies a user-defined filtering function to a batch and adds matching sample keys to the exclusion list.

    remove_keys(batch: Any) -> Any
        Removes previously flagged keys from a batch to maintain alignment across datasets.

    validate_keys(batch: Any, key_fn: Callable) -> List[str]
        Extracts keys from a batch and checks which ones are marked for exclusion.

    save_keys(tag: str, path: Optional[str] = None) -> None
        Saves the current set of keys to be excluded under a given tag for reuse in later stages.

    load_keys(tag: str, path: Optional[str] = None) -> None
        Loads a previously saved exclusion list into memory.

    clear_keys() -> None
        Clears the currently tracked keys (e.g., when starting a new cleaning session).

    Example:
    --------
    >>> cleaner = Clean()

    >>> def filter_nan_rows(df):
    >>>     return df[df.isnull().any(axis=1)].index.tolist()

    >>> for batch in extractor.load_csv("features.csv", batch_size=1000):
    >>>     cleaner.apply_filter(batch, filter_nan_rows)

    >>> for batch in extractor.load_csv("targets.csv", batch_size=1000):
    >>>     aligned_batch = cleaner.remove_keys(batch)
    >>>     save(aligned_batch)

    >>> cleaner.save_keys("cleaned_v1")

    Notes:
    ------
    - Keys may represent row indices, filenames, sample IDs, or other unique identifiers.
    - This module does not alter the data itself—it only tracks and removes flagged samples.
    - Repairing or transforming problematic data should be handled in the `Transform` module.
    - For detecting schema inconsistencies or statistical anomalies, use the `Examine` module.

    """

    def apply_filter(self, batch: Any, filter_fn: Callable) -> None: pass
    def remove_keys(self, batch: Any) -> Any: pass
    def validate_keys(self, batch: Any, key_fn: Callable) -> List[str]: pass
    def save_keys(self, tag: str, path: Optional[str] = None) -> None: pass
    def load_keys(self, tag: str, path: Optional[str] = None) -> None: pass
    def clear_keys(self) -> None: pass
