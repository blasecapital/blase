from typing import (
    Any,
    Dict,
    Iterable,
    Optional,
    Protocol,
    runtime_checkable,
    Tuple,
    Sequence,
    Mapping,
)
from blase.types import Batch


# Data
@runtime_checkable
class DatasetBuilder(Protocol):
    def build(
        self,
    ) -> Tuple[
        Iterable[Batch], Optional[Iterable[Batch]], Optional[Iterable[Batch]]
    ]: ...


@runtime_checkable
class Parser(Protocol):
    def __call__(self, example: Any) -> Any: ...


@runtime_checkable
class Augmenter(Protocol):
    def __call__(self, sample: Any) -> Any: ...


# Strategy / precision
@runtime_checkable
class StrategyAdapter(Protocol):
    def scope(self): ...  # context manager
    def shard_dataset(self, ds: Iterable[Batch], is_train: bool) -> Iterable[Batch]: ...


@runtime_checkable
class PrecisionPolicy(Protocol):
    def apply(self) -> None: ...


# Model
@runtime_checkable
class ModelFactory(Protocol):
    def __call__(self, **kwargs: Any) -> Any: ...


@runtime_checkable
class Compiler(Protocol):
    def compile(
        self,
        model: Any,
        optimizer: Any,
        loss: Any,
        metrics: Sequence[Any],
        lr_schedule: Optional[Any],
    ) -> Any: ...


# Train loop
@runtime_checkable
class Fitter(Protocol):
    def fit(
        self,
        model: Any,
        train: Iterable[Batch],
        val: Optional[Iterable[Batch]],
        epochs: int,
        steps_per_epoch: Optional[int],
        validation_steps: Optional[int],
        class_weight: Optional[Mapping[Any, float]],
        callbacks: Sequence[Any],
    ) -> Dict[str, Any]: ...


@runtime_checkable
class Evaluator(Protocol):
    def evaluate(
        self, model: Any, ds: Iterable[Batch], steps: Optional[int]
    ) -> Dict[str, float]: ...


# Export
@runtime_checkable
class Exporter(Protocol):
    def export(
        self,
        model: Any,
        kind: str,
        path: Optional[str],
        extra: Optional[Dict[str, Any]],
    ) -> str: ...


# Callbacks
@runtime_checkable
class CallbackFactory(Protocol):
    def build(self) -> Sequence[Any]: ...


# Tracking (injected)
@runtime_checkable
class Tracker(Protocol):
    def start_step(self, fqn: str, params: Dict[str, Any]): ...
    def log_metric(self, name: str, value: Any): ...
    def log_artifact(self, name: str, path: str): ...
    def end_step(self, status: str = "ok"): ...
