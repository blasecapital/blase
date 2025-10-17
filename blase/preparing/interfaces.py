from typing import Any, Dict, Iterable, Mapping, Protocol, Sequence, Tuple

from blase.types import Batch

ManifestRow = Dict[str, Any]
ManifestMeta = Dict[str, Any]
ManifestBatch = Batch[Sequence[ManifestRow], ManifestMeta]
Splits = Mapping[str, Sequence[str]]
Stats = Dict[str, Any]


class ImageIndexProvider(Protocol):
    def build_image_index(
        self, sources: Sequence[Dict[str, Any]], cfg: Dict[str, Any]
    ) -> Tuple[Mapping[str, Dict[str, Any]], Any, Any]: ...


class LabelReader(Protocol):
    def read(self, src: Dict[str, Any]) -> Iterable[Dict[str, Any]]: ...


class ClassMapBuilder(Protocol):
    def build_or_validate_class_map(
        self, label_iters: Sequence[Iterable[Dict[str, Any]]], cfg: Dict[str, Any]
    ) -> Tuple[Mapping[str, int], Dict[str, Any]]: ...


class Aligner(Protocol):
    def align_stream(
        self,
        *,
        label_iters: Sequence[Iterable[Dict[str, Any]]],
        kv_index: Mapping[str, Dict[str, Any]],
        rg_index: Any,
        join_cfg: Dict[str, Any],
        box_cfg: Dict[str, Any],
        class_cfg: Dict[str, Any],
        scale_cfg: Dict[str, Any],
    ) -> Iterable[ManifestBatch]: ...


class Splitter(Protocol):
    def split(
        self, manifest_iter: Iterable[ManifestBatch], cfg: Dict[str, Any]
    ) -> Splits: ...


class StatsComputer(Protocol):
    def compute(
        self, manifest_iter: Iterable[ManifestBatch], cfg: Dict[str, Any]
    ) -> Stats: ...


class TFRecordWriter(Protocol):
    def write(
        self,
        manifest_iter: Iterable[ManifestBatch],
        splits: Splits,
        cfg: Dict[str, Any],
    ) -> Batch[Sequence[Dict[str, Any]], ManifestMeta]: ...


class SidecarWriter(Protocol):
    def write(
        self,
        manifest_iter: Iterable[ManifestBatch],
        splits: Splits,
        cfg: Dict[str, Any],
    ) -> Batch[Sequence[Dict[str, Any]], ManifestMeta]: ...
