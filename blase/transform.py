import importlib
import logging
import os
from typing import Callable, Iterable, Any, Optional
from pathlib import Path

from blase.track import Track
from blase.utils.hashing import Hash

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

    Logging:
    --------
    - Set `enable_logging=True` when instantiating Transform to enable logging.
    - Logging outputs which transformation was applied to each batch.
    - Users can also choose to log to a file using `log_to_file=True` and specify a path with `log_file`.

    Example:
    --------
    >>> def custom_feature_engineering(batch):
    >>>     batch["new_feature"] = batch["feature1"] * batch["feature2"]
    >>>     return batch
    >>>
    >>> transformer = Transform(enable_logging=True)
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
        self.tracked = False
        self.run_path = None
        self.step_hash = None
        self.func_hash = None
        self.parent_hash = None
        self.parent_type = None

    def apply_function(
        self,
        data: Any,
        parent: tuple[str, str],
        transform_func: Callable,
        last_batch: bool,
        track: bool = True
    ) -> Iterable:
        """
        Applies a user-defined transformation function to batched data.

        Args:
            data (Any): A DataFrames object.
            parent (tuple[str, str]): Hash and type of parent.
            transform_func (Callable): A function that transforms a batch.
            last_batch (bool): Flag from Extract to end tracking step.
            track (bool): Whether to log this transformation step.

        Yields:
            Iterable: Transformed batches.
        """
        # tracking
        if track and not self.tracked:
            hasher = Hash()
            self.func_hash = hasher.hash_function(transform_func)
            self.parent_hash, self.parent_type = parent

            track_dir = Track._validate_run_directory(track=track)
            self.run_path = Track._validate_active_run(track_dir=track_dir)
            step_id, self.step_hash = Track._start_step(
                run_path=self.run_path,
                function=transform_func.__name__,
                params={
                    "parent_hash": self.parent_hash,
                    "transform_func": transform_func.__name__,
                    "track": track
                },
                parent=self.parent_hash
            )
            self.tracked = True

        # main logic
        result = transform_func(data)

        if track and last_batch:
            Track._end_step(
                step_hash=self.step_hash,
                run_path=self.run_path,
                outputs={
                    "function_hash": self.func_hash
                },
                parent=(self.parent_hash, self.parent_type)
            )

        return result, (self.step_hash, "step")

    def apply_from_module(self, data: Iterable, module_path: str, function_name: str) -> Iterable: pass
    def apply_standard_transformation(self, data, transformation: str, columns: list): pass
    # Maybe include a fill missing values explicitly here or include it in standard transformation
