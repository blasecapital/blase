from dataclasses import dataclass
from typing import Generic, Optional, Sequence, Any, Mapping, Literal, TypeVar

D = TypeVar("D")
M = TypeVar("M", bound=Mapping[str, Any])

__all__ = [
    "StepMsg",
    "Batch",
    "Artifact",
    "SinkResult",
    "Metrics",
    "Predictions",
    "Checkpoint",
    "Progress",
    "Alert",
]


@dataclass(frozen=True)
class StepMsg(Generic[M]):
    is_last: bool
    meta: M  # base


@dataclass(frozen=True)
class Batch(StepMsg[M], Generic[D, M]):
    data: D
    labels: Optional[Sequence[Any]] = None
    paths: Optional[Sequence[str]] = None
    role: str = "data"  # multi-input disambiguation


@dataclass(frozen=True)
class Artifact:
    path: str
    kind: str  # "parquet","tfrecord","onnx","ckpt","db","png"
    data_hash: Optional[str] = None


@dataclass(frozen=True)
class SinkResult(StepMsg[M]):
    artifacts: Sequence[Artifact]


@dataclass(frozen=True)
class Metrics(StepMsg[M]):
    values: Mapping[str, float]  # loss, acc, f1, drift scores, etc.


@dataclass(frozen=True)
class Predictions(StepMsg[M]):
    # for Evaluate/Monitor streaming previews before persisted artifacts land
    inputs_ref: Artifact  # dataset shard ref
    outputs_ref: Optional[Artifact]  # e.g., temp preds file


@dataclass(frozen=True)
class Checkpoint(StepMsg[M]):
    artifact: Artifact  # model.ckpt or similar
    step: int  # global step/epoch


@dataclass(frozen=True)
class Progress(StepMsg[M]):
    step: int
    total: Optional[int] = None  # enables UI progress bars


@dataclass(frozen=True)
class Alert(StepMsg[M]):
    code: str  # "DRIFT_HIGH","DATA_QUALITY_DROP"
    severity: Literal["info", "warn", "crit"]
    details: Mapping[str, Any]
