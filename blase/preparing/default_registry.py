from .registry import PrepareRegistry

# capability backends
from .manifest import index_images as _idx
from .readers import (
    coco as _coco,
    jsonl_boxes as _jsonl,
    csv_cls as _csv,
    parquet_det as _pdet,
    parquet_cls as _pcls,
    labelbox_jsonl as _lb,
)
from .splitters import (
    stratified as _strat,
    random as _rand,
    group as _group,
    time as _time,
)
from .stats import counters as _stats
from .writers.tfrecord import writer as _tfr
from .writers.sidecar import jsonl as _sidecar_jsonl, parquet as _sidecar_parquet


def default_registry() -> PrepareRegistry:
    return PrepareRegistry(
        image_indexer=_idx,  # module with build_index(...)
        label_readers={
            "coco": _coco,
            "jsonl_boxes": _jsonl,
            "csv": _csv,
            "parquet": _pdet,  # detection via parquet
            "parquet_cls": _pcls,  # classification via parquet
            "labelbox": _lb,
        },
        splitters={
            "stratified": _strat,
            "random": _rand,
            "group": _group,
            "time": _time,
        },
        stats=_stats,
        tfr_writer=_tfr,
        sidecar_writers={
            "jsonl": _sidecar_jsonl,
            "parquet": _sidecar_parquet,
        },
    )
