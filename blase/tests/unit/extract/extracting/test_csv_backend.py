import os
import pytest
from tempfile import NamedTemporaryFile

import pandas as pd
import polars as pl

from blase.extracting.csv_backend import (
    memory_aware_batcher, 
    load_batches_pandas,
    load_batches_polars
)

# Create temporary CSV files for testing
@pytest.fixture(scope="module")
def temp_csv_file(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("data")
    file_path = os.path.join(tmp_dir, "sample.csv")
    df = pd.DataFrame({
        "id": range(10),
        "value": [f"text_{i}" for i in range(10)]
    })
    df.to_csv(file_path, index=False)
    return file_path

@pytest.fixture
def sample_csv_file():
    data = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "country": ["USA", "Canada", "USA", "Mexico", "USA"],
        "year": [2020, 2021, 2020, 2022, 2020]
    })

    with NamedTemporaryFile(delete=False, mode='w', suffix=".csv") as f:
        data.to_csv(f.name, index=False)
        yield f.name
        os.unlink(f.name)

def test_memory_aware_batcher_pandas(temp_csv_file):
    batch_size = memory_aware_batcher(
        file_path=temp_csv_file,
        backend="pandas",
        verbose=True
    )
    assert isinstance(batch_size, int)
    assert batch_size > 0

def test_memory_aware_batcher_polars(temp_csv_file):
    try:
        batch_size = memory_aware_batcher(
            file_path=temp_csv_file,
            backend="polars",
            verbose=True
        )
        assert isinstance(batch_size, int)
        assert batch_size > 0
    except ImportError:
        pytest.skip("polars not installed")

def test_memory_aware_batcher_missing_file():
    with pytest.raises(FileNotFoundError):
        memory_aware_batcher(
            file_path="nonexistent.csv",
            backend="pandas",
            verbose=True
        )

def test_memory_aware_batcher_fallback_memory(monkeypatch, temp_csv_file):
    # Simulate psutil not being installed
    import blase.extracting.csv_backend as csv_backend
    monkeypatch.setattr(csv_backend, "psutil", None)

    batch_size = memory_aware_batcher(
        file_path=temp_csv_file,
        backend="pandas",
        verbose=True,
        fallback_memory_limit_mb=256
    )
    assert isinstance(batch_size, int)
    assert batch_size > 0


def test_load_batches_pandas_with_use_cols_only(sample_csv_file):
    batches = list(load_batches_pandas(
        sample_csv_file,
        batch_size=2,
        use_cols=["id", "country"],
        filter_by=None
    ))

    assert len(batches) == 3  # 2 + 2 + 1
    assert isinstance(batches[0], pd.DataFrame)
    assert "id" in batches[0].columns
    assert "country" in batches[0].columns
    assert "year" not in batches[0].columns  # ensure it's excluded
    assert batches[0].iloc[0]["id"] == 1


def test_load_batches_pandas_with_use_cols_subset(sample_csv_file):
    batches = list(load_batches_pandas(
        sample_csv_file,
        batch_size=3,
        use_cols=["country"],
        filter_by=None
    ))

    all_rows = pd.concat(batches, ignore_index=True)
    assert list(all_rows.columns) == ["country"]
    assert all_rows.shape[0] == 5


def test_load_batches_pandas_no_filter(sample_csv_file):
    batches = list(load_batches_pandas(sample_csv_file, batch_size=2, use_cols=None, filter_by=None))

    assert len(batches) == 3  # 2 + 2 + 1
    assert isinstance(batches[0], pd.DataFrame)
    assert batches[0].iloc[0]["id"] == 1
    assert batches[-1].shape[0] == 1


def test_load_batches_pandas_with_filter(sample_csv_file):
    batches = list(load_batches_pandas(
        sample_csv_file,
        batch_size=2,
        use_cols=None,
        filter_by=[{"col": "country", "value": "USA"}]
    ))

    all_rows = pd.concat(batches, ignore_index=True)
    assert all(all_rows["country"] == "USA")
    assert len(all_rows) == 3


def test_load_batches_pandas_multiple_filters(sample_csv_file):
    filters = [
        {"col": "country", "value": "USA"},
        {"col": "year", "value": 2020}
    ]
    
    batches = list(load_batches_pandas(sample_csv_file, batch_size=2, use_cols=None, filter_by=filters))
    all_rows = pd.concat(batches)

    assert all(all_rows["country"] == "USA")
    assert all(all_rows["year"] == 2020)
    assert all_rows.shape[0] == 3


def test_load_batches_pandas_column_missing(sample_csv_file):
    with pytest.raises(RuntimeError) as exc_info:
        list(load_batches_pandas(
            sample_csv_file,
            batch_size=2,
            use_cols=None,
            filter_by=[{"col": "nonexistent", "value": "X"}]
        ))

    assert "Column 'nonexistent' not found" in str(exc_info.value)


def test_load_batches_polars_basic(sample_csv_file):
    batches = list(load_batches_polars(sample_csv_file, batch_size=2, use_cols=None, filter_by=None))
    
    assert len(batches) == 3
    assert isinstance(batches[0], pl.DataFrame)
    assert batches[0].shape[0] == 2
    assert batches[2].shape[0] == 1
    assert batches[0]["id"][0] == 1


def test_load_batches_polars_with_use_cols_only(sample_csv_file):
    batches = list(load_batches_polars(
        sample_csv_file,
        batch_size=2,
        use_cols=["id", "country"],
        filter_by=None
    ))

    assert len(batches) == 3  # 2 + 2 + 1
    assert isinstance(batches[0], pl.DataFrame)
    assert "id" in batches[0].columns
    assert "country" in batches[0].columns
    assert "year" not in batches[0].columns  # ensure it's excluded
    assert batches[0][0, "id"] == 1 


def test_load_batches_polars_with_use_cols_subset(sample_csv_file):
    batches = list(load_batches_polars(
        sample_csv_file,
        batch_size=3,
        use_cols=["country"],
        filter_by=[{"col": "country", "value": "USA"}]
    ))

    all_rows = pl.concat(batches)
    assert list(all_rows.columns) == ["country"]
    assert all_rows.shape[0] == 3


def test_load_batches_polars_with_filter(sample_csv_file):
    filters = [{"col": "country", "value": "USA"}]
    batches = list(load_batches_polars(sample_csv_file, batch_size=2, use_cols=None, filter_by=filters))
    
    all_rows = pl.concat(batches)
    assert all(all_rows["country"] == "USA")
    assert all_rows.shape[0] == 3  # 3 USA rows


def test_load_batches_polars_with_no_matching_filter(sample_csv_file):
    filters = [{"col": "country", "value": "Brazil"}]
    batches = list(load_batches_polars(sample_csv_file, batch_size=2, use_cols=None, filter_by=filters))
    
    assert len(batches) == 0


def test_load_batches_polars_multiple_filters(sample_csv_file):
    filters = [
        {"col": "country", "value": "USA"},
        {"col": "year", "value": 2020}
    ]
    batches = list(load_batches_polars(sample_csv_file, batch_size=2, use_cols=None, filter_by=filters))
    
    all_rows = pl.concat(batches)
    assert all(all_rows["country"] == "USA")
    assert all(all_rows["year"] == 2020)
    assert all_rows.shape[0] == 3


def test_load_batches_polars_column_missing(sample_csv_file):
    with pytest.raises(RuntimeError) as excinfo:
        list(load_batches_polars(
            sample_csv_file,
            batch_size=2,
            use_cols=None, 
            filter_by=[{"col": "nonexistent", "value": "X"}]
        ))
    assert "Column 'nonexistent'" in str(excinfo.value)
