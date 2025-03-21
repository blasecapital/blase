import importlib
import logging
import os
from typing import Callable, Iterable, Any

from transforming import transform_utils


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

    def __init__(self, enable_logging=True, log_to_file=False, log_file="transform.log"):
        self.enable_logging = enable_logging
        if self.enable_logging:
            if log_to_file:
                log_dir = os.path.dirname(log_file) or "."
                os.makedirs(log_dir, exist_ok=True)
                
                logging.basicConfig(
                    filename=log_file,
                    filemode="a",
                    level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                )
            else:
                logging.basicConfig(
                    level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                )

    def apply_function(self, data: Iterable, transform_func: Callable) -> Iterable: pass
    def apply_from_module(self, data: Iterable, module_path: str, function_name: str) -> Iterable: pass
    def apply_standard_transformation(self, data, transformation: str, columns: list): pass
