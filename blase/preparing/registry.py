from dataclasses import dataclass
from typing import Mapping

from .interfaces import (
    ImageIndexProvider,
    LabelReader,
    Splitter,
    StatsComputer,
    TFRecordWriter,
    SidecarWriter,
)


@dataclass
class PrepareRegistry:
    image_indexer: ImageIndexProvider
    label_readers: Mapping[str, LabelReader]  # fmt -> reader
    splitters: Mapping[str, Splitter]  # name -> splitter
    stats: StatsComputer
    tfr_writer: TFRecordWriter
    sidecar_writers: Mapping[str, SidecarWriter]  # "jsonl","parquet" -> writer
