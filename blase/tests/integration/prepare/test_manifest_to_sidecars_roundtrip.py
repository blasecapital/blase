from pathlib import Path

from blase.prepare import Prepare, images_parquet, coco
from blase.preparing.registry import PrepareRegistry

# Reuse same concrete modules as previous test
from blase.preparing.manifest import index_images as _idx
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
    return PrepareRegistry(
        image_indexer=_idx,
        label_readers={"coco": _coco},
        splitters={"stratified": _strat},
        stats=_stats,
        tfr_writer=None,  # not used here
        sidecar_writers={"jsonl": _sidecar},
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
