from typing import Any, Dict, Iterable, Mapping, Protocol, Sequence

from blase.types import Batch

ManifestRow = Dict[str, Any]
ManifestMeta = Dict[str, Any]
ManifestBatch = Batch[Sequence[ManifestRow], ManifestMeta]
Splits = Mapping[str, Sequence[str]]
Stats = Dict[str, Any]


class ImageIndexProvider(Protocol):
    def build_index(
        self, sources: Sequence[Dict[str, Any]], cfg: Dict[str, Any]
    ) -> Mapping[str, Dict[str, Any]]: ...


class LabelReader(Protocol):
    def read(self, src: Dict[str, Any]) -> Iterable[Dict[str, Any]]: ...


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
