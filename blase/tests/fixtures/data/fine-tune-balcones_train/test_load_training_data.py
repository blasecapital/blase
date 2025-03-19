# test_load_training_data.py


import unittest
from unittest.mock import patch
import sqlite3

import pandas as pd

from training.components.load_training_data import LoadTrainingData


class TestLoadTrainingData(unittest.TestCase):

    @patch('training.components.load_training_data.EnvLoader')
    def setUp(self, MockEnvLoader):
        self.mock_env_loader = MockEnvLoader.return_value
        self.mock_env_loader.get.side_effect = lambda key: ":memory:" if key in ["DATABASE_1", "DATABASE_2"] else None
        self.mock_env_loader.load_config_module.return_value = {
            "source_query": {
                "table1" : ("DATABASE_1", 
                """
                SELECT * FROM base_table
                ORDER BY date, pair
                """)
            },
            "primary_key": ["date", "pair"],
        }

        # Create an in-memory SQLite database
        self.mock_db = sqlite3.connect(":memory:")
        self.create_mock_tables()

        # Patch sqlite3.connect to return our in-memory database
        patcher = patch("sqlite3.connect", return_value=self.mock_db)
        self.addCleanup(patcher.stop)
        self.mock_connect = patcher.start()

        self.load_training_data = LoadTrainingData()
        self.load_training_data.max_chunk_size = 10000

    def create_mock_tables(self):
        """Create mock tables in the in-memory database."""
        cursor = self.mock_db.cursor()

        # Create a sample table
        cursor.execute("""
        CREATE TABLE base_table (
            date INTEGER PRIMARY KEY,
            pair TEXT NOT NULL,
            value TEXT NOT NULL
        )
        """)

        # Insert mock data
        cursor.executemany("INSERT INTO base_table (pair, value) VALUES (?, ?)", [
            ("EURUSD", "data1"),
            ("GBPUSD", "data2"),
            ("AUDUSD", "data3"),
            ("USDJPY", "data4"),
            ("EURUSD", "data5"),
            ("GBPUSD", "data6"),
            ("AUDUSD", "data7"),
            ("USDJPY", "data8"),
            ("EURUSD", "data9"),
            ("GBPUSD", "data10"),
        ])

        self.mock_db.commit()

    def test_get_row_count(self):
        """Test getting the number of rows in a query result."""
        queries = ["SELECT * FROM base_table", "SELECT * FROM base_table WHERE pair='EURUSD'",
                   "SELECT * FROM base_table WHERE pair='GBPUSD' ORDER BY 'date'"]
        expected_count = [10, 3, 3]
        for query, expected_count in zip(queries, expected_count):
            result = self.load_training_data._get_row_count("mock_db_path", query)
            self.assertEqual(result, expected_count)

    def test_get_column_count(self):
        """Test getting the number of comumns in the query result."""
        queries = ["SELECT * FROM base_table", "SELECT * FROM base_table WHERE pair='EURUSD'",
                   "SELECT * FROM base_table WHERE pair='GBPUSD' ORDER BY 'date'"]
        expected_count = [3, 3, 3]
        for query, expected_count in zip(queries, expected_count):
            results = self.load_training_data._get_column_count("mock_db_path", query)
            self.assertEqual(results, expected_count)

    def test_chunk_keys(self):
        """Test chunk_keys returns sets of primary keys."""
        expected_result = [
            ((1, "EURUSD"), (2, "GBPUSD")),
            ((3, "AUDUSD"), (4, "USDJPY")),
            ((5, "EURUSD"), (6, "GBPUSD")),
            ((7, "AUDUSD"), (8, "USDJPY")),
            ((9, "EURUSD"), (10, "GBPUSD"))
        ]
        for test_mode in ['config', 'manual']:
            if test_mode == 'manual':
                result = self.load_training_data.chunk_keys(mode='manual', db_path="mock_db_path", query="SELECT * FROM base_table")
            elif test_mode == 'config':
                result = self.load_training_data.chunk_keys(mode='config', source="source_query", key="table1")
            self.assertEqual(result, expected_result)

    def test_chunk_keys_invalid_mode(self):
        """Test chunk_keys raises error for invalid mode."""
        with self.assertRaises(ValueError) as context:
            self.load_training_data.chunk_keys(mode="invalid")
        self.assertEqual(str(context.exception), "Invalid mode. Choose either 'config' or 'manual'.")

    def test_chunk_keys_missing_parameters(self):
        """Test chunk_keys raises errors for missing parameters."""
        with self.assertRaises(ValueError) as context:
            self.load_training_data.chunk_keys(mode="config", source=None, key="table1")
        self.assertEqual(str(context.exception), "When using 'config' mode, 'source' and 'key' must be provided.")

        with self.assertRaises(ValueError) as context:
            self.load_training_data.chunk_keys(mode="manual", db_path=None, query="SELECT * FROM base_table")
        self.assertEqual(str(context.exception), "When using 'manual' mode, 'db_path' and 'query' must be provided.")

    def test_load_chunk(self):
        """Test loading a chunk given a key and chunk key."""
        expected_result = pd.DataFrame({
            "date": [1, 2],
            "pair": ["EURUSD", "GBPUSD"],
            "value": ["data1", "data2"]
        })

        for test_mode in ['config', 'manual']:
            if test_mode == 'manual':
                result = self.load_training_data.load_chunk(
                    mode='manual', 
                    db_path="mock_db_path", 
                    query="SELECT * FROM base_table",
                    chunk_key=((1, "EURUSD"), (2, "GBPUSD")))
            elif test_mode == 'config':
                result = self.load_training_data.load_chunk(
                    mode='config', 
                    source="source_query", 
                    key="table1",
                    chunk_key=((1, "EURUSD"), (2, "GBPUSD")))
            pd.testing.assert_frame_equal(result, expected_result)


if __name__ == '__main__':
    unittest.main(verbosity=2)
