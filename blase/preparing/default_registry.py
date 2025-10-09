from types import SimpleNamespace

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
from .manifest.classmap import build_or_validate_class_map as _classmap
from .manifest.align_stream import align_stream as _align
from .writers.tfrecord import writer as _tfr
from .writers.sidecar import jsonl as _sidecar_jsonl, parquet as _sidecar_parquet


image_indexer = SimpleNamespace(
    build_index=lambda sources, cfg: _idx.build_image_index(
        data_sources=sources,
        image_cfg={
            "id_col": cfg.get("id_col"),
            "path_col": cfg.get("path_col"),
            "height_col": cfg.get("height_col"),
            "width_col": cfg.get("width_col"),
            "sha256_col": cfg.get("sha256_col"),
        },
        scale_cfg={
            "rows_per_chunk": cfg.get("rows_per_chunk"),
            "index_backend": cfg.get("index_backend"),
        },
    )
)
classmap = SimpleNamespace(build_or_validate=_classmap)
aligner = SimpleNamespace(align_stream=_align)


def default_registry() -> PrepareRegistry:
    return PrepareRegistry(
        image_indexer=image_indexer,  # module with build_index(...)
        label_readers={
            "coco": _coco,
            "jsonl_boxes": _jsonl,
            "csv": _csv,
            "parquet": _pdet,  # detection via parquet
            "parquet_cls": _pcls,  # classification via parquet
            "labelbox": _lb,
        },
        classmap=classmap,
        aligner=aligner,
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
