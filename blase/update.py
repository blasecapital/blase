from typing import Callable, Optional, Dict


class Update:
    """
    Coordinates the reproducible and memory-efficient update of a previously trained model
    by integrating new data and retraining or fine-tuning as needed.

    The `Update` class acts as an orchestration layer across the `Extract`, `Transform`,
    `Prepare`, `Train`, and optionally `Evaluate` modules. It allows users to retrain models 
    on newly available data while minimizing redundant computation and storage. 

    It uses logged metadata and data hashing to determine what has changed since the last
    training run, making updates both **efficient** and **traceable**. Whether performing 
    full retraining or light fine-tuning, `Update` enables production-ready iteration with
    minimal setup.

    Features:
    ---------
    - Automatically detects and integrates new raw data.
    - Supports update modes like: 
        * 'append_only' – fine-tune using only new data
        * 'full_retrain' – retrain on all data (old + new)
        * 'rolling' – keep only the latest `n` batches
    - Supports hash-based caching and smart recomputation of intermediate datasets.
    - Reuses logged transformation functions and training configs.
    - Tracks version lineage and logs new model metadata.
    - Optionally evaluates updated models and stores predictions and metrics.
    - Offers a `dry_run()` option to simulate an update for verification.
    
    Parameters for `run()`:
    -----------------------
    base_model_path : str
        Path to the previously trained model directory or checkpoint.
    
    new_data_path : str
        Path to the new raw data to integrate into the update.

    update_mode : str
        One of: 'append_only', 'full_retrain', 'rolling'. Determines the retraining logic.

    output_path : str, optional
        Where the updated model, logs, and metadata should be saved. Defaults to versioned path.

    transform_fn : str or Callable, optional
        Optional path to a transformation function for preparing the new data.

    evaluate_after : bool, optional
        If True, the updated model will be evaluated automatically after training.

    cache_depth : int, optional
        Controls how much intermediate data is preserved (e.g., 0 = only raw, 1 = +prepped).

    Example:
    --------
    >>> updater = Update()
    >>> updater.run(
    >>>     base_model_path='models/v1/',
    >>>     new_data_path='raw_data/',
    >>>     update_mode='append_only',
    >>>     transform_fn='transforming.clean_features',
    >>>     evaluate_after=True
    >>> )

    Notes:
    ------
    - This class does not assume that all prior data still exists on disk. If needed, it can
      recreate previous intermediate files using logs and hashes.
    - All updates are automatically logged with version hashes and training configs.
    - The class prioritizes efficient memory use and minimal I/O to remain local-first.

    """

    def run(
        self,
        base_model_path: str,
        new_data_path: str,
        update_mode: str = "append_only",
        output_path: Optional[str] = None,
        transform_fn: Optional[Callable] = None,
        evaluate_after: bool = False,
        cache_depth: int = 1,
    ) -> None:
        pass

    def set_base_model(self, path: str) -> None:
        pass

    def set_new_data(self, path: str) -> None:
        pass

    def configure_update_mode(self, mode: str) -> None:
        pass

    def evaluate_after_update(self, flag: bool = True) -> None:
        pass

    def set_output_path(self, path: str) -> None:
        pass

    def register_filter_function(self, fn: Callable) -> None:
        pass

    def register_transform_function(self, fn: Callable) -> None:
        pass

    def set_cache_depth(self, depth: int) -> None:
        pass

    def dry_run(self) -> None:
        pass

    def version_history(self) -> None:
        pass

    def rollback(self) -> None:
        pass
