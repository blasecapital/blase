import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from blase.prepare import (
    Prepare,
    ImageConfig,
    JoinConfig,
    BoxConfig,
    ClassConfig,
    ScaleConfig,
    images_parquet,
    coco,
)
from blase.preparing.registry import PrepareRegistry

from blase.preparing.manifest import index_images as _idx
from blase.preparing.manifest.classmap import build_or_validate_class_map as _classmap
from blase.preparing.manifest.align_stream import align_stream as _align
from blase.preparing.readers import coco as _coco
from blase.preparing.splitters import stratified as _strat
from blase.preparing.stats import counters as _stats
from blase.preparing.writers.tfrecord import writer as _tfr
from blase.preparing.writers.sidecar import jsonl as _sidecar


def _write_parquet_images(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            {"image_id": "a", "path": "a.jpg", "height": 2, "width": 4, "sha256": "aa"},
            {"image_id": "b", "path": "b.jpg", "height": 4, "width": 8, "sha256": "bb"},
            {"image_id": "c", "path": "c.jpg", "height": 4, "width": 8, "sha256": "cc"},
        ]
    )
    pq.write_table(pa.Table.from_pandas(df), out_dir / "images_000.parquet")


def _write_tiny_coco(p: Path):
    data = {
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 4, "height": 2},
            {"id": 2, "file_name": "b.jpg", "width": 8, "height": 4},
            {"id": 3, "file_name": "c.jpg", "width": 8, "height": 4},
        ],
        "annotations": [
            {
                "id": 10,
                "image_id": 1,
                "bbox": [0, 0, 2, 1],
                "category_id": 5,
                "iscrowd": 0,
            },
            {
                "id": 11,
                "image_id": 2,
                "bbox": [2, 1, 2, 2],
                "category_id": 6,
                "iscrowd": 0,
            },
        ],
        "categories": [{"id": 5, "name": "weed"}, {"id": 6, "name": "radish"}],
    }
    p.write_text(json.dumps(data))


def _default_registry():
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

    return PrepareRegistry(
        image_indexer=image_indexer,  # now has .build_index(...)
        label_readers={"coco": _coco},
        classmap=classmap,  # exposes .build_or_validate(...)
        aligner=aligner,  # exposes .align_stream(...)
        splitters={"stratified": _strat, "random": _strat},
        stats=_stats,
        tfr_writer=_tfr,
        sidecar_writers={"jsonl": _sidecar},
    )


def test_manifest_then_split(tmp_path: Path):
    pq_dir = tmp_path / "parquet"
    coco_json = tmp_path / "labels.json"
    _write_parquet_images(pq_dir)
    _write_tiny_coco(coco_json)

    prep = Prepare(registry=_default_registry())
    manifest = prep.build_manifest(
        data_sources=[images_parquet(str(pq_dir / "images_*.parquet"))],
        label_sources=[coco(str(coco_json))],
        image_cfg=ImageConfig(bytes_col=None, sha256_col="sha256"),
        join_cfg=JoinConfig(image_id_resolver="provided", keep_unlabeled_images=True),
        box_cfg=BoxConfig(coord_in="xyxy_abs", coord_out="xyxy_rel", clamp_boxes=True),
        class_cfg=ClassConfig(),
        scale_cfg=ScaleConfig(join_strategy="duckdb_temp", batch_rows=2, seed=123),
    )

    stats_b = prep.compute_stats(manifest, by="class")
    assert isinstance(stats_b.data, dict) and "class_hist" in stats_b.data

    splits_b = prep.split(
        manifest, method="stratified", train=0.67, val=0.33, test=0.0, seed=123
    )
    assert "train" in splits_b.data and "val" in splits_b.data
    # image ids are from parquet
    all_ids = set().union(*splits_b.data.values())
    assert all_ids.issubset({"a", "b", "c"})
