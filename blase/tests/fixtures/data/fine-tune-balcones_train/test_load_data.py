# test_load_data.py


import sqlite3
import unittest
from unittest.mock import patch

import pandas as pd

from data_preparation.components.load_data import LoadData


class TestLoadData(unittest.TestCase):

    @patch("data_preparation.components.load_data.EnvLoader")
    def setUp(self, MockEnvLoader):
        """Set up an in-memory database and mock environment loader."""
        self.mock_env_loader = MockEnvLoader.return_value
        self.mock_env_loader.get.side_effect = lambda key: "mock_db_path" if key in ["base_source", "feature_source", "target_source"] else None

        self.mock_env_loader.load_config_module.return_value = {
            "base_query": "SELECT * FROM base_table WHERE pair IN ({placeholders})",
            "feature_query": "SELECT * FROM feature_table WHERE pair IN ({placeholders})",
            "target_query": "SELECT * FROM target_table WHERE pair IN ({placeholders})",
        }

        # Create an in-memory SQLite database
        self.mock_db = sqlite3.connect(":memory:")
        self.create_mock_tables()

        # Patch sqlite3.connect to return our in-memory database
        patcher = patch("sqlite3.connect", return_value=self.mock_db)
        self.addCleanup(patcher.stop)
        self.mock_connect = patcher.start()

        self.load_data = LoadData()

    def create_mock_tables(self):
        """Create mock tables in the in-memory database."""
        cursor = self.mock_db.cursor()

        # Create a sample table
        cursor.execute("""
        CREATE TABLE base_table (
            id INTEGER PRIMARY KEY,
            pair TEXT NOT NULL,
            value TEXT NOT NULL
        )
        """)

        # Insert mock data
        cursor.executemany("INSERT INTO base_table (pair, value) VALUES (?, ?)", [
            ("EURUSD", "data1"),
            ("GBPUSD", "data2"),
            ("AUDUSD", "data3"),
        ])

        self.mock_db.commit()

    def test_load_data_single_pair(self):
        """Test loading a single pair from the in-memory database."""
        df = self.load_data.load_data("base", pair="EURUSD")

        expected_df = pd.DataFrame({"id": [1], "pair": ["EURUSD"], "value": ["data1"]})
        pd.testing.assert_frame_equal(df, expected_df)

    def test_load_data_batch(self):
        """Test loading data in batches from the in-memory database."""
        df = self.load_data.load_data("base", batch=["EURUSD", "GBPUSD"])

        expected_df = pd.DataFrame({"id": [1, 2], "pair": ["EURUSD", "GBPUSD"], "value": ["data1", "data2"]})
        pd.testing.assert_frame_equal(df, expected_df)

    def test_load_data_full_dataset(self):
        """Test loading all data from the in-memory database."""
        df = self.load_data.load_data("base")

        expected_df = pd.DataFrame({"id": [1, 2, 3], "pair": ["EURUSD", "GBPUSD", "AUDUSD"], "value": ["data1", "data2", "data3"]})
        pd.testing.assert_frame_equal(df, expected_df)

    def tearDown(self):
        """Close the in-memory database after tests complete."""
        self.mock_db.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
