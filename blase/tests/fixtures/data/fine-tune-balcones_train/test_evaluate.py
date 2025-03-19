# test_evaluate.py


import unittest
from unittest.mock import patch, mock_open, MagicMock
import tempfile
import os

import numpy as np
import tensorflow as tf
import pandas as pd

from evaluation.components.evaluate import Eval


class TestEval(unittest.TestCase):

    def setUp(self):
        self.eval = Eval()

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
            self.eval.module_path = module_path  

            # Call `_import_function` and check if it correctly loads `test_function`
            imported_function = self.eval._import_function(module_path, "test_function")
            self.assertEqual(imported_function(), "Hello, World!")

    @patch("builtins.open", new_callable=mock_open, read_data='{"0": 0.5, "2": 3.5, "1": 3.5}')
    @patch("os.path.exists", return_value=True)
    def test_initial_bias(self, mock_exists, mock_file):
        """Test initial bias is correctly calculated."""
        model_args = {
            "initial_bias": True 
        }
        expected_result = np.array([-2.639057, -0.133531, -0.133531])
        result = self.eval._initial_bias(model_args, weight_dict_path="test_path")
        np.testing.assert_array_almost_equal(result, expected_result)

    @patch.object(Eval, "_import_function")
    def test_model_config(self, mock_import_function):
        """Test model_config is returned."""
        config = {
            "dir": "/mock/dir",
            "model_config_src": "config.py",
            "config_dict_name": "config"
        }
        mock_fn = MagicMock()
        mock_import_function.return_value = mock_fn
        expected_result = mock_fn
        result = self.eval._model_config(config)
        self.assertEqual(result, expected_result)
        mock_import_function.assert_called_once_with("/mock/dir/config.py", "config") 

    def test_group_pred_files(self):
        """Test that files are grouped correctly."""
        with tempfile.TemporaryDirectory() as test_dir:
            self.eval.data_dir = test_dir

            mode = "train"
            for i in range(3):
                open(os.path.join(self.eval.data_dir, f"file_{i}_train.tfrecord"), "w").close()
                open(os.path.join(self.eval.data_dir, f"targets_{i}_train.tfrecord"), "w").close()
            
            feature_category = "file"
            result = self.eval._group_pred_files(self.eval.data_dir, feature_category, mode)

            expected_result = {
                0: {'features': [f'{self.eval.data_dir}/file_0_train.tfrecord'], 
                    'targets': [f'{self.eval.data_dir}/targets_0_train.tfrecord']}, 
                1: {'features': [f'{self.eval.data_dir}/file_1_train.tfrecord'], 
                    'targets': [f'{self.eval.data_dir}/targets_1_train.tfrecord']}, 
                2: {'features': [f'{self.eval.data_dir}/file_2_train.tfrecord'], 
                    'targets': [f'{self.eval.data_dir}/targets_2_train.tfrecord']}}
            self.assertEqual(result, expected_result)

    @patch("builtins.open", new_callable=mock_open, read_data='{"feature1": "metadata1", "feature2": "metadata2"}')
    @patch("glob.glob")
    @patch("os.path.join", side_effect=lambda *args: "/".join(args))
    def test_load_feature_metadata_exact_match(self, mock_path_join, mock_glob, mock_file):
        """Test that feature metadata is loaded correctly."""
        data_dir = "/mock/data"
        mock_glob.return_value = ["/mock/data/category_feature_description.json"]

        result = self.eval._load_feature_metadata("category", data_dir)

        expected_result = {"feature1": "metadata1", "feature2": "metadata2"}
        self.assertEqual(result, expected_result)

        mock_file.assert_called_once_with("/mock/data/category_feature_description.json", "r")

    @patch("builtins.open", new_callable=mock_open, read_data='{"col1": "int64", "col2": "string", "col3": "object", "col4": "float32"}')
    @patch("os.path.exists", return_value=True)
    @patch("os.path.join", return_value="/mock/data/targets_feature_description.json")
    def test_load_target_metadata(self, mock_join, mock_exists, mock_file):
        """Test metadata is loaded correcrtly."""
        expected_result = {
            "col1": tf.io.FixedLenFeature([], tf.int64),
            "col2": tf.io.VarLenFeature(tf.string),
            "col3": tf.io.VarLenFeature(tf.string),
            "col4": tf.io.FixedLenFeature([], tf.float32)
        }
        result = self.eval._load_target_metadata(data_dir="/mock/data")
        for key in expected_result:
            self.assertEqual(type(result[key]), type(expected_result[key]))

    def test_parse_tfrecord_fn(self):
        """Test that the correct parsing function is returned."""
        feature_dict = {
            "feature1": tf.train.Feature(float_list=tf.train.FloatList(value=[0.5])),
            "feature2": tf.train.Feature(int64_list=tf.train.Int64List(value=[42])),
            "feature3": tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"100"]))
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        serialized_example = example_proto.SerializeToString() 
        
        feature_metadata = {
            "feature1": "float32",
            "feature2": "int64",
            "feature3": "object"
        }

        category = "test"
        feature_categories = {
            "test": {
                "reshape": False,
                "shape": None
            }
        }
        result = self.eval._parse_tfrecord_fn(serialized_example, feature_metadata, category, feature_categories)
        expected_result = tf.convert_to_tensor([0.5, 42.0, 100.0], dtype=tf.float32)
        
        # Convert tensors to np.arrays for testing
        result = result.numpy()
        expected_result = expected_result.numpy()
        np.testing.assert_array_equal(result, expected_result)

    @patch.object(Eval, "_import_function")
    @patch.object(Eval, "_model_config")
    @patch.object(Eval, "_initial_bias")
    def test_load_model(self, mock_initial_bias, mock_model_config, mock_import_function):
        """Test model is loaded correctly."""
        config = {
            "dir": "/mock/dir",
            "create_model_module_src": "test_custom_train_funcs.py",
            "model_module": "create_model",
            "model_args": "model_args",
        }

        mock_model_args = {"arg1": 10, "arg2": "relu"}
        mock_weight_dict_path = "mock/weight_dict_path"

        mock_create_model = MagicMock()
        mock_import_function.return_value = mock_create_model
        mock_model_config.return_value = {
            "model_args": mock_model_args,
            "weight_dict_path": mock_weight_dict_path
        }
        mock_initial_bias.return_value = None

        result = self.eval._load_model(config)

        mock_import_function.assert_called_once_with("/mock/dir/test_custom_train_funcs.py", "create_model")
        mock_model_config.assert_called_once_with(config)
        mock_initial_bias.assert_called_once_with(mock_model_args, mock_weight_dict_path)
        mock_create_model.assert_called_once_with(arg1=10, arg2="relu")
        self.assertEqual(result, mock_create_model())

    def test_load_weights(self):
        """Test weights are loaded."""
        config = {
            "dir": "/mock",
            "model_weights": "weights.h5"
        }
        mock_model = MagicMock()
        result = self.eval._load_weights(mock_model, config)
        expected_result = mock_model
        mock_model.load_weights.assert_called_once_with("/mock/weights.h5")
        self.assertEqual(result, expected_result)

    def test_group_lime_files(self):
        """Test files are grouped correctly."""
        with tempfile.TemporaryDirectory() as test_dir:

            for i in range(3):
                open(os.path.join(test_dir, f"feature_{i}_train.tfrecord"), "w").close()
                open(os.path.join(test_dir, f"targets_{i}_train.tfrecord"), "w").close()

            self.eval.explain_config["file_num"] = "0"
            
            result = self.eval._group_lime_files(test_dir)
            expected_result = {
                0: {'features': [f'{test_dir}/feature_0_train.tfrecord'], 
                    'targets': [f'{test_dir}/targets_0_train.tfrecord']}}
            self.assertEqual(result, expected_result)

    def test_extract_single_feature(self):
        """Test extracting features from a simple TF dataset with one feature."""
        dataset = tf.data.Dataset.from_tensor_slices((tf.constant([[1.0, 2.0], [3.0, 4.0]]), tf.constant([0, 1])))
        result = self.eval._extract_numpy_features(dataset, batch_size=2)
        expected = np.array([[1.], [2.], [3.], [4.]]) 
        np.testing.assert_array_almost_equal(result, expected)

    def test_extract_multi_input(self):
        """Test extracting features from a multi-input TF dataset."""
        dataset = tf.data.Dataset.from_tensor_slices((
            (tf.constant([[1.0], [2.0]]), tf.constant([[3.0], [4.0]])),
            tf.constant([0, 1])
        ))
        result = self.eval._extract_numpy_features(dataset, batch_size=2)
        expected = np.array([[1.0, 3.0], [2.0, 4.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_extract_empty_dataset(self):
        """Test extracting features from an empty dataset."""
        dataset = tf.data.Dataset.from_tensor_slices(([], []))
        result = self.eval._extract_numpy_features(dataset, batch_size=2)
        expected = np.empty((0, 1))
        np.testing.assert_array_equal(result, expected)

    def test_save_feature_shapes_single_input(self):
        """Test extracting shape from a dataset with a single input tensor."""
        dataset = tf.data.Dataset.from_tensor_slices((tf.random.uniform((10, 5)),))
        dataset = dataset.batch(1)

        lime_input_shapes, lime_cum_indices = self.eval._save_feature_shapes(dataset)

        expected_shapes = [(5,)]
        expected_cum_indices = np.array([0, 5])

        self.assertEqual(lime_input_shapes, expected_shapes)
        np.testing.assert_array_equal(lime_cum_indices, expected_cum_indices)

    def test_save_feature_shapes_multi_input(self):
        """Test extracting shapes from a dataset with multiple input tensors."""
        dataset = tf.data.Dataset.from_tensor_slices((
            tf.random.uniform((10, 4)),  
            tf.random.uniform((10, 6))
        ))
        dataset = dataset.batch(1)

        lime_input_shapes, lime_cum_indices = self.eval._save_feature_shapes(dataset)

        expected_shapes = [(4,), (6,)]
        expected_cum_indices = np.array([0, 4, 10])

        self.assertEqual(lime_input_shapes, expected_shapes)
        np.testing.assert_array_equal(lime_cum_indices, expected_cum_indices)

    def test_save_feature_shapes_empty_dataset(self):
        """Test handling an empty dataset."""
        dataset = tf.data.Dataset.from_tensor_slices((tf.zeros((0, 5)),))
        dataset = dataset.batch(1)

        lime_input_shapes, lime_cum_indices = self.eval._save_feature_shapes(dataset)

        expected_shapes = [(5,)]
        expected_cum_indices = np.array([0, 5])

        self.assertEqual(lime_input_shapes, expected_shapes)
        np.testing.assert_array_equal(lime_cum_indices, expected_cum_indices)

    def test_lime_predict(self):
        """Test that _lime_predict correctly slices, reshapes, and calls model.predict"""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[0.8, 0.2], [0.6, 0.4]])
        
        X = np.array([
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12]
        ])
        
        lime_input_shapes = [(2,), (4,)]
        lime_cum_indices = np.array([0, 2, 6])
        
        result = self.eval._lime_predict(X, mock_model, lime_input_shapes, lime_cum_indices)

        expected_input_1 = np.array([[1, 2], [7, 8]])
        expected_input_2 = np.array([[3, 4, 5, 6], [9, 10, 11, 12]])

        args, _ = mock_model.predict.call_args
        np.testing.assert_array_equal(args[0][0], expected_input_1)
        np.testing.assert_array_equal(args[0][1], expected_input_2)
        np.testing.assert_array_equal(result, np.array([[0.8, 0.2], [0.6, 0.4]]))

    def test_list_cat_cols(self):
        """Test categorical features are identified."""
        self.eval.explain_config["categorical_feature_id"]: "pair_"
        feature_names = ["pair_AUDUSD", "sma20"]
        result_cat, result_cont = self.eval._list_cat_cols(feature_names)
        expected_cat = ["pair_AUDUSD"]
        expected_cont = ["sma20"]
        self.assertEqual(result_cat, expected_cat)
        self.assertEqual(result_cont, expected_cont)

    def test_clean_bytes(self):
        """Test that bytes are cleaned correctly."""
        test_cases = [
            # (input_value, expected_output)
            ([b"hello"], "hello"),            # List containing bytes
            ([b"hello", b"world"], "hello"),  # List containing multiple bytes (only first should be taken)
            (["hello"], "hello"),             # List containing string
            ([b""], ""),                      # List containing empty bytes
            ("hello", "hello"),               # Normal string (no change expected)
            ([], ""),                         # Empty list
            (None, None),                     # None input (should return None)
            (b"", ""),                        # Empty bytes input
        ]

        for input_value, expected_output in test_cases:
            with self.subTest(input_value=input_value):
                result = self.eval._clean_bytes(input_value)
                self.assertEqual(result, expected_output)

    @patch.object(Eval, "_clean_bytes")
    def test_extract_target_info_sparse(self, mock_clean_bytes):
        """Test _extract_target_info handles SparseTensors correctly."""
        target_column_names = ["target", "extra1"]
        sample_num = 0
        mock_clean_bytes.side_effect = lambda x: x.decode('utf-8') if isinstance(x, bytes) else x

        data_dict = {
            "target": tf.sparse.from_dense(tf.constant([[0], [1]], dtype=tf.int64)),
            "extra1": tf.sparse.from_dense(tf.constant([[b"val1"], [b"val2"]], dtype=tf.string)),
        }

        dataset = tf.data.Dataset.from_tensor_slices(data_dict).batch(1)
        expected_df = pd.DataFrame({
            "target": [0, 1],
            "extra1": ["val1", "val2"]
        })

        result = self.eval._extract_target_info(dataset, target_column_names, sample_num)
        pd.testing.assert_series_equal(result, expected_df.iloc[sample_num], check_dtype=False)

    @patch.object(Eval, "_clean_bytes")
    def test_extract_target_info_dense(self, mock_clean_bytes):
        """Test _extract_target_info handles SparseTensors correctly."""
        target_column_names = ["target", "extra1", "extra2"]
        sample_num = 1
        mock_clean_bytes.side_effect = lambda x: x.decode('utf-8') if isinstance(x, bytes) else x

        # Create a simple TF dataset with 3 rows
        data_dict = {
            "target": tf.constant([0, 1, 2], dtype=tf.int64),
            "extra1": tf.constant([b"data1", b"data2", b"data3"], dtype=tf.string),
            "extra2": tf.constant([3.1, 4.2, 5.3], dtype=tf.float32),
        }

        dataset = tf.data.Dataset.from_tensor_slices(data_dict).batch(2)  # Batch of 2

        # Expected DataFrame after processing
        expected_df = pd.DataFrame({
            "target": [0, 1, 2],
            "extra1": ["data1", "data2", "data3"],  # Cleaned bytes
            "extra2": [3.1, 4.2, 5.3]
        })

        # Run the function
        result = self.eval._extract_target_info(dataset, target_column_names, sample_num)

        # Validate the extracted row
        pd.testing.assert_series_equal(result, expected_df.iloc[sample_num], check_dtype=False)

    @patch("sqlite3.connect")
    def test_complete_lime_obs_info(self, mock_connect):
        """Test specified data is pulled from db."""
        self.eval.explain_config = {
            "prediction_dir": "/mock/path/to/db.sqlite",
            "id_cols": ["date", "pair"]
        }

        target_info = pd.Series({
            "date": "2025-03-14",
            "pair": "EURUSD"
        })

        expected_df = pd.DataFrame({
            "date": ["2025-03-14"],
            "pair": ["EURUSD"],
            "target": [1],
            "prediction": [0.8]
        })

        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        with patch("pandas.read_sql_query", return_value=expected_df) as mock_read_sql:
            result = self.eval._complete_lime_obs_info(target_info)

            expected_query = """
            SELECT * FROM predictions
            WHERE date = ? AND pair = ?
            """
            expected_params = ("2025-03-14", "EURUSD")

            mock_read_sql.assert_called_once_with(expected_query, mock_conn, params=expected_params)

            pd.testing.assert_frame_equal(result, expected_df)
            mock_conn.close.assert_called_once()

    def test_metric_cats(self):
        """Test that categories are grouped."""
        df = pd.DataFrame({
            "split": ["train", "train", "val", "test"],
            "pair": ["AUDUSD", "EURUSD", "USDJPY", "GBPUSD"],
            "predicted_category": [0, 1, 2, 0],
        })
        config_dict = {
            "metric_categories": {
                "split_col": "split",
                "asset_col": 'pair',
                "target_col": 'predicted_category',
                }
        }
        sub_dict = "metric_categories"
        expected_result = {
            "split_col": np.array(["train", "val", "test"]),
            "asset_col": np.array(["AUDUSD", "EURUSD", "USDJPY", "GBPUSD"]),
            "target_col": np.array([0, 1, 2])
        }
        result = self.eval._metric_cats(df, config_dict, sub_dict)
        for key in expected_result:
            np.testing.assert_array_equal(result[key], expected_result[key])

    def test_adjust_actual_target(self):
        """Test that targets are adjusted."""
        df = pd.DataFrame({
            "target": [0, 0, 1, 2, 0],
            "buy_sl_time": ["None", 6, "None", 3, 12],
            "sell_sl_time": [10, 2, 3, "None", "None"]
        })
        expected_result = pd.DataFrame({
            "target": [1, 0, 1, 2, 2],
            "buy_sl_time": ["None", 6, "None", 3, 12],
            "sell_sl_time": [10, 2, 3, "None", "None"]       
        })
        result = self.eval._adjust_actual_target(df)
        pd.testing.assert_frame_equal(result, expected_result)

    def test_fit_cal_fn(self):
        """Test calibration binning."""
        df = pd.DataFrame({
            "wl": [0, 1] * 500,
            "confidence": np.linspace(0.5, 1.0, num=1000)
        })
        self.eval.cal_config["y_conf"] = "confidence"
        expected_x_list = [round(df.iloc[split]["confidence"].mean(), 4) for split in np.array_split(df.index, 100)]
        expected_y_list = [round(df.iloc[split]["wl"].mean(), 4) for split in np.array_split(df.index, 100)]
        result_x_list, result_y_list = self.eval._fit_cal_fn(df)
        self.assertEqual(expected_x_list, result_x_list)
        self.assertEqual(expected_y_list, result_y_list)

    def test_candidate_screen(self):
        """Test candidate screening logic based on cumulative accuracy threshold."""
        df = pd.DataFrame({
            "wl": [0, 1, 1, 0, 1, 1, 0, 1, 1, 1],
            "confidence": [0.55, 0.60, 0.65, 0.50, 0.70, 0.75, 0.55, 0.80, 0.85, 0.90]
        })

        accuracy_threshold = 0.6
        volume = 3

        df["cumulative_avg_wl"] = df["wl"].expanding().mean()

        expected_result = {
            "cumulative_avg_wl": df.loc[9, "cumulative_avg_wl"],
            "confidence": df.loc[9, "confidence"],
            "rows_from_top": 9
        }
        result = self.eval._candidate_screen(df, accuracy_threshold, volume)
        self.assertEqual(result, expected_result)

    def test_candidate_screen_no_match(self):
        """Test when no row meets the accuracy threshold."""
        df = pd.DataFrame({
            "wl": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "confidence": [0.55, 0.60, 0.65, 0.50, 0.70, 0.75, 0.55, 0.80, 0.85, 0.90]
        })

        accuracy_threshold = 0.6
        volume = 3

        expected_result = {}
        result = self.eval._candidate_screen(df, accuracy_threshold, volume)
        self.assertEqual(result, expected_result)


if __name__ == "__main__":
    unittest.main()
