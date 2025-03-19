# test_create_features.py


import unittest
from unittest.mock import patch, MagicMock
import tempfile
import os

import pandas as pd

from data_preparation.components.create_features import CreateFeatures


class TestCreateFeatures(unittest.TestCase):

    @patch("data_preparation.components.create_features.EnvLoader")
    def setUp(self, MockEnvLoader):
        """Set up an in-memory database and mock environment loader."""
        self.mock_env_loader = MockEnvLoader.return_value
        self.mock_env_loader.get.side_effect = lambda key: "mock_db_path" if key in [
            "base_source", "feature_source", "target_source", "BASE_DATABASE",
            "FEATURE_DATASE", "TARGET_DATABASE"] else None
        self.mock_env_loader.get.return_value = "/mock/config.py"

        self.mock_env_loader.load_config_module.return_value = {
            "main_feature_module": "test_function",
            "feature_storage_map": "storage_map",
            "primary_key": ['id', 'obj'],
            "pair_query": {"BASE_DATABASE": "SELECT pair FROM pairs"}
        }

        self.mock_df = pd.DataFrame({
            'id': [1, 2, 3],
            'obj': ['a', 'b', 'c'],
            'value': ['data1', 'data2', 'data3']
        })

        self.create_features = CreateFeatures()

    def test_import_feature_module(self):
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
            self.create_features.module_path = module_path  

            # Call `_import_function` and check if it correctly loads `test_function`
            imported_function = self.create_features._import_feature_module()
            self.assertEqual(imported_function(), "Hello, World!")

    def test_import_storage_map(self):
        """Test that _import_storage_map imports the storage_map from the feature script."""
        storage_map = """
storage_map = {
    "feature1": "data1",
    "feature2": "data2"
}
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            module_path = os.path.join(temp_dir, "test_module.py")
            
            # Write test function to a temporary module file
            with open(module_path, "w") as f:
                f.write(storage_map)

            # Set module_path in PrepData
            self.create_features.module_path = module_path  

            imported_map = self.create_features._import_storage_map()
            self.assertEqual(imported_map, {"feature1": "data1", "feature2": "data2"})

    def test_store_original_columns(self):
        """Test that _store_original_columns removes primary keys from DataFrame columns."""
        expected_columns = ['value']
        result_columns  = self.create_features._store_original_columns(self.mock_df)
        self.assertEqual(expected_columns, result_columns )

    @patch("data_preparation.components.create_features.pd.read_sql_query")
    @patch("data_preparation.components.create_features.sqlite3.connect")
    def test_create_pairs_list(self, mock_sqlite_connect, mock_read_sql):
        """Test _create_pairs_list to ensure it correctly queries and returns pairs."""
        mock_conn = MagicMock()
        mock_sqlite_connect.return_value.__enter__.return_value = mock_conn

        mock_df = pd.DataFrame({"pair": ["EURUSD", "GBPUSD", "AUDUSD"]})
        mock_read_sql.return_value = mock_df

        pair_query = self.create_features.config["pair_query"]
        result = self.create_features._create_pairs_list(pair_query)

        expected_pairs = ["EURUSD", "GBPUSD", "AUDUSD"]
        self.assertEqual(result, expected_pairs) 

        mock_sqlite_connect.assert_called_once_with("mock_db_path")
        mock_read_sql.assert_called_once_with("SELECT pair FROM pairs", mock_conn)

    def test_create_batches_even_split(self):
        """Test _create_batches with an evenly divisible list."""
        pairs_list = ["EURUSD", "GBPUSD", "AUDUSD", "USDJPY", "USDCAD", "EURGBP"]
        batch_size = 2

        result = list(self.create_features._create_batches(pairs_list, batch_size))
        expected = [["EURUSD", "GBPUSD"], ["AUDUSD", "USDJPY"], ["USDCAD", "EURGBP"]]

        self.assertEqual(result, expected)

    def test_create_batches_uneven_split(self):
        """Test _create_batches with an unevenly divisible list."""
        pairs_list = ["EURUSD", "GBPUSD", "AUDUSD", "USDJPY", "USDCAD"]
        batch_size = 2

        result = list(self.create_features._create_batches(pairs_list, batch_size))
        expected = [["EURUSD", "GBPUSD"], ["AUDUSD", "USDJPY"], ["USDCAD"]]

        self.assertEqual(result, expected)

    def test_create_batches_single_batch(self):
        """Test _create_batches where batch_size is larger than list."""
        pairs_list = ["EURUSD", "GBPUSD"]
        batch_size = 5  # Larger than the list

        result = list(self.create_features._create_batches(pairs_list, batch_size))
        expected = [["EURUSD", "GBPUSD"]]

        self.assertEqual(result, expected)

    def test_create_batches_empty_list(self):
        """Test _create_batches with an empty list."""
        pairs_list = []
        batch_size = 3

        result = list(self.create_features._create_batches(pairs_list, batch_size))
        expected = []

        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()