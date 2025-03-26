import pytest
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
import polars as pl

from blase.extract import Extract

# Temporary CSV content for testing
TEST_CSV_CONTENT = """id,value
1,100
2,200
3,300
4,400
5,500
"""

@pytest.fixture
def temp_csv_file(tmp_path):
    file_path = tmp_path / "test.csv"
    file_path.write_text(TEST_CSV_CONTENT)
    return str(file_path)

def test_load_csv_batches(temp_csv_file):
    extractor = Extract()
    batches = list(extractor.load_csv(file_path=temp_csv_file, mode="manual", batch_size=2, backend="pandas"))
    
    # Should split into 3 batches (2+2+1)
    assert len(batches) == 3
    assert isinstance(batches[0], pd.DataFrame)
    assert batches[0].shape[0] == 2
    assert batches[1].shape[0] == 2
    assert batches[2].shape[0] == 1
    assert batches[0].iloc[0]["id"] == 1

def test_load_csv_batches_polars(temp_csv_file):
    extractor = Extract()
    batches = list(extractor.load_csv(file_path=temp_csv_file, mode="manual", batch_size=2, backend="polars"))

    # Should split into 3 batches (2+2+1)
    assert len(batches) == 3
    assert isinstance(batches[0], pl.DataFrame)
    assert batches[0].shape[0] == 2
    assert batches[1].shape[0] == 2
    assert batches[2].shape[0] == 1

    # Polars uses `batches[0][col][row]` for indexing
    assert batches[0]["id"][0] == 1
