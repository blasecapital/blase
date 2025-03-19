# test_clean_raw_data.py


import unittest
from unittest.mock import patch, MagicMock, mock_open
import sqlite3
import tempfile
import os

import pandas as pd
import numpy as np
from concurrent.futures import Future

from training.components.clean_raw_data import CleanRawData


class TestCleanRawData(unittest.TestCase):

    @patch('training.components.clean_raw_data.EnvLoader')
    def setUp(self, MockEnvLoader):
        self.mock_env_loader = MockEnvLoader.return_value
        self.mock_env_loader.get.side_effect = lambda key: ":memory:" if key in ["DATABASE_1", "DATABASE_2"] else None
        self.mock_env_loader.load_config_module.return_value = {
            "source_query": {
                "table1" : ("DATABASE_1", 
                """
                SELECT * FROM base_table
                ORDER BY date, pair
                """),
                "table2" : ("DATABASE_2", 
                """
                SELECT * FROM base_table
                """)
            },
            "primary_key": ["date", "pair"],
            "clean_function": "clean_fn"
        }

        # Create an in-memory SQLite database
        self.mock_db = sqlite3.connect(":memory:")
        self.create_mock_tables()

        # Patch sqlite3.connect to return our in-memory database
        patcher = patch("sqlite3.connect", return_value=self.mock_db)
        self.addCleanup(patcher.stop)
        self.mock_connect = patcher.start()

        self.df = pd.read_sql_query("SELECT * FROM base_table", self.mock_db)

        self.clean_raw_data = CleanRawData()

    def create_mock_tables(self):
        """Create mock tables in the in-memory database."""
        cursor = self.mock_db.cursor()

        # Create a sample table
        cursor.execute("""
        CREATE TABLE base_table (
            date INTEGER PRIMARY KEY,
            pair TEXT NOT NULL,
            value INT NOT NULL
        )
        """)

        # Insert mock data
        cursor.executemany("INSERT INTO base_table (pair, value) VALUES (?, ?)", [
            ("EURUSD", 4),
            ("GBPUSD", 9),
            ("AUDUSD", 14),
            ("USDJPY", 19),
            ("EURUSD", 24),
            ("GBPUSD", 29),
            ("AUDUSD", 34),
            ("USDJPY", 39),
            ("EURUSD", 44),
            ("GBPUSD", 49),
        ])

        self.mock_db.commit()

    def test_calculate_percentiles(self):
        """Test perentiles are calculated correctly."""
        total_count = 100
        bin_edges = [0, 10, 20, 30, 40, 50]
        bin_frequencies = {
            "underflow": 0, 
            "overflow": 0, 
            0: 10, 
            1: 20, 
            2: 40, 
            3: 20, 
            4: 10}
        expected_result = {"25%": 7.5, "50%": 15.0, "75%": 22.5}
        result = self.clean_raw_data._calculate_percentiles(
            bin_edges=bin_edges,
            bin_frequencies=bin_frequencies,
            total_count=total_count
        )
        self.assertEqual(result, expected_result)

    @patch("training.components.clean_raw_data.tqdm")
    @patch("training.components.clean_raw_data.ThreadPoolExecutor")
    def test_initialize_bins_from_sql(self, mock_executor, mock_tqdm):
        """Test bins are initialized correctly from SQL query without threading issues."""
        mock_tqdm.return_value.__enter__.return_value = MagicMock()

        mock_executor_instance = MagicMock()
        mock_executor.return_value.__enter__.return_value = mock_executor_instance

        def sync_submit(fn, *args, **kwargs):
            """Runs the function synchronously and returns a valid Future object."""
            future = Future()
            result = fn(*args, **kwargs)
            future.set_result(result)
            return future

        mock_executor_instance.submit.side_effect = sync_submit

        expected_result = {"value": np.linspace(4, 49, num=101)}
        result = self.clean_raw_data._initialize_bins_from_sql(col_list=["value"], key="table1")
        np.testing.assert_array_almost_equal(result["value"], expected_result["value"])

    def test_collect_cols(self):
        """Test columns are collected correctly."""
        # Note: _collect_cols currently only supports queries with FROM and SELECT keywords
        result = self.clean_raw_data._collect_cols("table2")
        self.assertEqual(result, ["value"])

    def test_remove_primary_key(self):
        """Test primary key is removed from columns."""
        expected_df = self.df.drop(columns=["date", "pair"])
        result = self.clean_raw_data._remove_primary_key(self.df)
        pd.testing.assert_frame_equal(result, expected_df)

    def test_convert_to_numeric(self):
        """Test columns are converted to numeric."""
        df_copy = self.df.copy()
        result = self.clean_raw_data._convert_to_numeric(self.df)
        pd.testing.assert_frame_equal(result, df_copy)

    def test_import_function(self):
        """Test that `_import_function` correctly loads a function from a dynamically imported module."""
        function_code = """
def test_function():
    return "Hello, World!"
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            module_path = os.path.join(temp_dir, "test_module.py")
            
            # Write test function to a temporary module file
            with open(module_path, "w") as f:
                f.write(function_code)

            # Set module_path in PrepData
            self.clean_raw_data.module_path = module_path  

            # Call `_import_function` and check if it correctly loads `test_function`
            imported_function = self.clean_raw_data._import_function("test_function")
            self.assertEqual(imported_function(), "Hello, World!")

    @patch("builtins.open", new_callable=mock_open, read_data="('EURUSD', '2024-03-12')\n('GBPUSD', '2024-03-10')")
    @patch("os.listdir", return_value=["mock_bad_keys_bad_keys.txt"])
    def test_create_bad_key_set(self, mock_listdir, mock_open_file):
        """Test extracting bad keys from a simulated bad_keys.txt file."""
        mock_directory = "/mock/directory"
        expected_result = {("EURUSD", "2024-03-12"), ("GBPUSD", "2024-03-10")}
        result = self.clean_raw_data._create_bad_key_set(mock_directory)
        self.assertEqual(result, expected_result)

    def test_remove_bad_keys(self):
        """Test removing bad keys from the dataset."""
        bad_keys = {("EURUSD", 9), ("GBPUSD", 10)}
        expected_df = self.df[~self.df.apply(lambda row: (row["pair"], row["date"]) in bad_keys, axis=1)].reset_index(drop=True)
        result = self.clean_raw_data._remove_bad_keys(self.df, bad_keys)
        pd.testing.assert_frame_equal(result, expected_df)


if __name__ == '__main__':
    unittest.main()
    