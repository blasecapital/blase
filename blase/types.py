from dataclasses import dataclass, field, asdict
from typing import (
    Generic,
    Optional,
    Sequence,
    Any,
    Mapping,
    Literal,
    TypeVar,
    Dict,
    Tuple,
    Callable,
    List,
    Protocol,
)

D = TypeVar("D")
M = TypeVar("M", bound=Mapping[str, Any])

__all__ = [
    "StepMsg",
    "Batch",
    "Artifact",
    "SinkResult",
    "PreviewItem",
    "PreviewResult",
    "Population",
    "SourceParams",
    "SourceBuild",
    "SourceAdapter",
    "Metrics",
    "Predictions",
    "Checkpoint",
    "Progress",
    "Alert",
]


# ----------------------------
# Core step I/O datatypes
# ----------------------------


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


# ----------------------------
# Examine preview datatypes
# ----------------------------


@dataclass
class PreviewItem:
    path: str
    ok: bool
    reason: str  # "" if ok else e.g. "decode_error"
    w: int  # width if ok else 0
    h: int  # height if ok else 0
    ch: int  # channels (3 for RGB etc.)
    mode: str  # "RGB","L", etc.
    exif_orient: int  # 1 if none
    file_bytes: int
    sha256_head: str  # hash of first N KiB for dupe detection
    mean: float  # 0..255 luminance proxy
    std: float  # 0..255
    p0: int  # min pixel
    p100: int  # max pixel
    thumb_bytes: Optional[bytes] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PreviewResult:
    items: List[PreviewItem]
    meta: Dict[str, Any] = field(default_factory=dict)


# ----------------------------
# Source registry types (for future sources)
# ----------------------------


class Population(Protocol):
    """
    Abstract population over which samples are drawn.
    Implementations must be deterministic for a given descriptor().
    """

    def size(self) -> int: ...
    def path_at(self, idx: int) -> str: ...
    def descriptor(
        self,
    ) -> Dict[str, Any]: ...  # stable identity used in replay hashing


SourceParams = Dict[str, Any]
SourceBuild = Tuple[Population, Dict[str, Any]]  # (population, population_meta)
SourceAdapter = Callable[[SourceParams], SourceBuild]


# ----------------------------
# TODO
# ----------------------------


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
