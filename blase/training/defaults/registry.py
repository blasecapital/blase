from typing import (
    Any,
    Optional,
    Literal,
)

from blase.training.protocols import (
    StrategyAdapter,
    PrecisionPolicy,
    Tracker,
    DatasetBuilder,
    Compiler,
    Fitter,
    CallbackFactory,
    Evaluator,
    Exporter,
)

Backend = Literal["tensorflow", "torch"]

# ---- Public API -------------------------------------------------------------


def strategy(backend: Backend) -> StrategyAdapter:
    if backend == "tensorflow":
        from .tf_adapters import TfStrategy

        return TfStrategy()
    raise NotImplementedError(f"strategy: backend {backend}")


def precision(backend: Backend) -> PrecisionPolicy:
    if backend == "tensorflow":
        from .tf_adapters import TfPrecision

        return TfPrecision()
    raise NotImplementedError(f"precision: backend {backend}")


def tracker(run_dir: Optional[str]) -> Tracker:
    from .track_adapter import DefaultTracker

    return DefaultTracker(run_dir)


def compiler(backend: Backend) -> Compiler:
    if backend == "tensorflow":
        from .tf_adapters import TfCompiler

        return TfCompiler()
    raise NotImplementedError(f"compiler: backend {backend}")


def fitter(backend: Backend) -> Fitter:
    if backend == "tensorflow":
        from .tf_adapters import TfFitter

        return TfFitter()
    raise NotImplementedError(f"fitter: backend {backend}")


def evaluator(backend: Backend) -> Evaluator:
    if backend == "tensorflow":
        from .tf_adapters import TfEvaluator

        return TfEvaluator()
    raise NotImplementedError(f"evaluator: backend {backend}")


def exporter(backend: Backend) -> Exporter:
    if backend == "tensorflow":
        from .tf_adapters import TfExporter

        return TfExporter()
    raise NotImplementedError(f"exporter: backend {backend}")


def callbacks(backend: Backend, **cfg: Any) -> CallbackFactory:
    if backend == "tensorflow":
        from .tf_adapters import TfCallbackFactory

        return TfCallbackFactory(**cfg)
    raise NotImplementedError(f"callbacks: backend {backend}")


def tf_dataset_builder(**cfg: Any) -> DatasetBuilder:
    from .tf_input import TfDatasetBuilder

    return TfDatasetBuilder(**cfg)


def wrap_existing_dataset(
    train_ds: Any, val_ds: Any = None, test_ds: Any = None, assume_batched: bool = True
) -> DatasetBuilder:
    from .tf_input import ExistingDatasetBuilder

    return ExistingDatasetBuilder(train_ds, val_ds, test_ds, assume_batched)
