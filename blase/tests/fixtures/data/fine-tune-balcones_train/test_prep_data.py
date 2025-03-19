# test_prep_data.py


import unittest
from unittest.mock import patch, mock_open
import tempfile
import os

import pandas as pd
import numpy as np

from training.components.prep_data import PrepData


class TestPrepData(unittest.TestCase):

    @patch("training.components.prep_data.EnvLoader")
    def setUp(self, MockEnvLoader):
        self.prep_data = PrepData()

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
            self.prep_data.module_path = module_path  

            # Call `_import_function` and check if it correctly loads `test_function`
            imported_function = self.prep_data._import_function("test_function")
            self.assertEqual(imported_function(), "Hello, World!")

    def test_convert_sqlite_text_to_types(self):
        """Test that columns are converted from text to the correct types."""
        df = pd.DataFrame({
            "col1": ["1", "2", "3"],
            "col2": ["a", "b", "c"],
            "col3": ["1.1", "None", "3.3"],
            "col4": ["1.1", "2.2", "3.3"],
        })
        expected_df = pd.DataFrame(df.copy())
        for col in expected_df.columns:
            expected_df[col] = expected_df[col].replace(["None", "NULL"], np.nan)   
        expected_df["col1"] = expected_df["col1"].astype(np.int64)
        expected_df["col2"] = expected_df["col2"].astype(str)
        expected_df["col3"] = expected_df["col3"].astype(str)
        expected_df["col4"] = expected_df["col4"].astype(np.float32)

        result_df = self.prep_data._convert_sqlite_text_to_types(df)
        pd.testing.assert_frame_equal(result_df, expected_df)

    @patch("builtins.open", new_callable=mock_open, read_data='{"col1": "int64", "col2": "str", "col3": "object", "col4": "float32"}')
    @patch("os.path.exists", return_value=True)
    def test_apply_feature_description(self, mock_exists, mock_file):
        """Test that feature descriptions are applied to the dataframe."""
        
        df = pd.DataFrame({
            "col1": ["1", "2", "3"],
            "col2": ["a", "b", "c"],
            "col3": ["1.1", "None", "3.3"],
            "col4": ["1.1", "2.2", "3.3"],
        })

        expected_df = df.copy()
        expected_df["col1"] = expected_df["col1"].astype(np.int64)
        expected_df["col2"] = expected_df["col2"].astype(str)
        expected_df["col3"] = expected_df["col3"].astype(str)
        expected_df["col4"] = expected_df["col4"].astype(np.float32)

        self.prep_data.prepped_data_dir = "/mock"

        table = "config"
        result_df = self.prep_data._apply_feature_description(table, df)

        pd.testing.assert_frame_equal(result_df, expected_df)


if __name__ == '__main__':
    unittest.main()