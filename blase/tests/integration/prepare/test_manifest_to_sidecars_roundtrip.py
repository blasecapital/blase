from pathlib import Path
from types import SimpleNamespace

from blase.prepare import Prepare, images_parquet, coco
from blase.preparing.registry import PrepareRegistry

from blase.preparing.manifest import index_images as _idx
from blase.preparing.manifest.classmap import build_or_validate_class_map as _classmap
from blase.preparing.manifest.align_stream import align_stream as _align
from blase.preparing.readers import coco as _coco
from blase.preparing.splitters import stratified as _strat
from blase.preparing.stats import counters as _stats
from blase.preparing.writers.sidecar import jsonl as _sidecar

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json


def _write_parquet_images(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            {"image_id": "a", "path": "a.jpg", "height": 2, "width": 4, "sha256": "aa"},
            {"image_id": "b", "path": "b.jpg", "height": 4, "width": 8, "sha256": "bb"},
        ]
    )
    pq.write_table(pa.Table.from_pandas(df), out_dir / "images_000.parquet")


def _write_tiny_coco(p: Path):
    data = {
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 4, "height": 2},
            {"id": 2, "file_name": "b.jpg", "width": 8, "height": 4},
        ],
        "annotations": [
            {
                "id": 10,
                "image_id": 1,
                "bbox": [0, 0, 2, 1],
                "category_id": 5,
                "iscrowd": 0,
            },
        ],
        "categories": [{"id": 5, "name": "weed"}],
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

    # dummy writer to satisfy the field; not used in this test
    tfr_writer = SimpleNamespace(write=lambda *a, **k: None)

    return PrepareRegistry(
        image_indexer=image_indexer,  # has .build_index(...)
        label_readers={"coco": _coco},  # each exposes .read(...)
        classmap=classmap,  # has .build_or_validate(...)
        aligner=aligner,  # has .align_stream(...)
        splitters={"stratified": _strat},  # exposes .split(...)
        stats=_stats,  # exposes .compute(...)
        tfr_writer=tfr_writer,  # unused here
        sidecar_writers={"jsonl": _sidecar},  # exposes .write(...)
    )


def test_write_sidecars(tmp_path: Path):
    pq_dir = tmp_path / "parquet"
    coco_json = tmp_path / "labels.json"
    out_dir = tmp_path / "out"
    _write_parquet_images(pq_dir)
    _write_tiny_coco(coco_json)

    prep = Prepare(registry=_default_registry())
    manifest = prep.build_manifest(
        data_sources=[images_parquet(str(pq_dir / "images_*.parquet"))],
        label_sources=[coco(str(coco_json))],
    )
    splits_b = prep.split(
        manifest, method="stratified", train=1.0, val=0.0, test=0.0, seed=7
    )

    res = prep.write_label_sidecars(
        manifest, splits_b.data, out_dir=out_dir, format="jsonl"
    )
    # at least one file artifact returned
    assert res.artifacts and any(a.path.endswith(".jsonl") for a in res.artifacts)
