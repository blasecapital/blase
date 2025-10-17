from pathlib import Path
import pytest

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from blase.preparing.manifest import index_images as idx


def _write(df, path: Path):
    pq.write_table(pa.Table.from_pandas(df), path)


def _write_parquet_images(glob_dir: Path):
    glob_dir.mkdir(parents=True, exist_ok=True)
    # two shards
    df1 = pd.DataFrame(
        [
            {"image_id": "a", "path": "a.jpg", "height": 2, "width": 4, "sha256": "aa"},
            {"image_id": "b", "path": "b.jpg", "height": 4, "width": 8, "sha256": "bb"},
        ]
    )
    df2 = pd.DataFrame(
        [{"image_id": "c", "path": "c.jpg", "height": 1, "width": 1, "sha256": "cc"}]
    )
    pq.write_table(pa.Table.from_pandas(df1), glob_dir / "images_000.parquet")
    pq.write_table(pa.Table.from_pandas(df2), glob_dir / "images_001.parquet")


def test_build_image_index_maps_ids_to_meta(tmp_path: Path):
    p = tmp_path / "parquet"
    _write_parquet_images(p)

    kv, bloom, rg = idx.build_image_index(
        data_sources=[{"uri": str(p / "images_*.parquet"), "fmt": "parquet"}],
        image_cfg={
            "id_col": "image_id",
            "path_col": "path",
            "height_col": "height",
            "width_col": "width",
            "sha256_col": "sha256",
        },
        scale_cfg={"index_backend": "duckdb", "rows_per_chunk": 100_000},
    )
    assert set(kv.keys()) == {"a", "b", "c"}
    assert all("path" in kv[k] and "height" in kv[k] and "width" in kv[k] for k in kv)


def test_handles_rows_per_chunk_below_rg_size(tmp_path: Path):
    p = tmp_path / "pq"
    p.mkdir()
    df = pd.DataFrame(
        [
            {"image_id": f"id{i}", "path": f"{i}.jpg", "height": 1, "width": 2}
            for i in range(10)
        ]
    )
    _write(df, p / "shard.parquet")

    kv, bloom, rg = idx.build_image_index(
        data_sources=[{"uri": str(p / "shard.parquet"), "fmt": "parquet"}],
        image_cfg={
            "id_col": "image_id",
            "path_col": "path",
            "height_col": "height",
            "width_col": "width",
            "sha256_col": None,
        },
        scale_cfg={"rows_per_chunk": 3},
    )
    assert len(kv) == 10
    assert set(kv) == {f"id{i}" for i in range(10)}
    assert all(kv[k]["rg_id"] == 0 for k in kv)  # single-RG file


def test_duplicate_ids_first_write_wins(tmp_path: Path):
    p = tmp_path / "pq"
    p.mkdir()
    df = pd.DataFrame(
        [
            {"image_id": "dup", "path": "a.jpg", "height": 1, "width": 1},
            {"image_id": "dup", "path": "b.jpg", "height": 2, "width": 2},
        ]
    )
    _write(df, p / "shard.parquet")

    kv, *_ = idx.build_image_index(
        data_sources=[{"uri": str(p / "shard.parquet"), "fmt": "parquet"}],
        image_cfg={
            "id_col": "image_id",
            "path_col": "path",
            "height_col": "height",
            "width_col": "width",
            "sha256_col": None,
        },
        scale_cfg={"rows_per_chunk": 1},
    )
    assert kv["dup"]["path"] == "a.jpg"  # first row kept


def test_missing_optional_sha256_ok(tmp_path: Path):
    p = tmp_path / "pq"
    p.mkdir()
    df = pd.DataFrame([{"image_id": "x", "path": "x.jpg", "height": 3, "width": 4}])
    _write(df, p / "shard.parquet")

    kv, *_ = idx.build_image_index(
        data_sources=[{"uri": str(p / "shard.parquet"), "fmt": "parquet"}],
        image_cfg={
            "id_col": "image_id",
            "path_col": "path",
            "height_col": "height",
            "width_col": "width",
            "sha256_col": None,
        },
        scale_cfg={},
    )
    assert "sha256" in kv["x"] and kv["x"]["sha256"] is None


def test_invalid_fmt_raises(tmp_path: Path):
    with pytest.raises(ValueError):
        idx.build_image_index(
            data_sources=[{"uri": "whatever.arrow", "fmt": "arrow"}],
            image_cfg={
                "id_col": "image_id",
                "path_col": "path",
                "height_col": "height",
                "width_col": "width",
                "sha256_col": None,
            },
            scale_cfg={},
        )


def test_build_image_index_streams_large_file_with_small_chunks(tmp_path: Path):
    """
    Stress the batch path without exploding memory:
    - Write ~200k rows but only 100 unique ids (duplicates → KV stays small).
    - Use rows_per_chunk=257 to force many iter_batches calls.
    Expect:
      - Function completes.
      - KV size equals number of unique ids (100).
    """
    p = tmp_path / "pq"
    p.mkdir()
    n_rows = 200_000
    n_unique = 100
    ids = [f"id{i % n_unique}" for i in range(n_rows)]
    df = pd.DataFrame(
        {
            "image_id": ids,
            "path": [f"{ids[i]}.jpg" for i in range(n_rows)],
            "height": [3] * n_rows,
            "width": [4] * n_rows,
        }
    )
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), p / "shard.parquet")

    kv, bloom, rg = idx.build_image_index(
        data_sources=[{"uri": str(p / "shard.parquet"), "fmt": "parquet"}],
        image_cfg={
            "id_col": "image_id",
            "path_col": "path",
            "height_col": "height",
            "width_col": "width",
            "sha256_col": None,
        },
        scale_cfg={"rows_per_chunk": 257},
    )

    assert len(kv) == n_unique
    # spot check entries present
    for i in range(0, n_unique, 17):
        k = f"id{i}"
        assert k in kv and kv[k]["path"].startswith(k)
