# test_train.py


import unittest
from unittest.mock import patch, mock_open, MagicMock
import tempfile
import os

import numpy as np
import tensorflow as tf

from training.components.train import Train


class TestTrain(unittest.TestCase):

    def setUp(self):
        self.train = Train()
        self.train.weight_dict_path = "/mock/path/to/weights.json"
        self.train.model_modules_path = "/mock/path/to/module"
        self.train.model_function = "mock_create_model"
        self.train.model_args = {"initial_bias": True, "param1": 10, "param2": "relu"}        
        self.train.optimizer = {"type": "adam"}
        self.train.metrics = ["accuracy"]
        self.train.custom_loss = {}
        self.train.data_dir = "/mock/data"
        self.train.feature_categories = {
            "test": {
                "reshape": False,
                "shape": None
            }
        }
        self.train.iteration_dir = "/mock/iteration_dir"
        self.train.requirements_paths = {
            "config": "/mock/path/to/config.py",
            "model": "/mock/path/to/model.py"
        }

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
            self.train.module_path = module_path  

            # Call `_import_function` and check if it correctly loads `test_function`
            imported_function = self.train._import_function(module_path, "test_function")
            self.assertEqual(imported_function(), "Hello, World!")

    @patch("builtins.open", new_callable=mock_open, read_data='{"0": 0.5, "2": 3.5, "1": 3.5}')
    @patch("os.path.exists", return_value=True)
    def test_initial_bias(self, mock_exists, mock_file):
        """Test initial bias is correctly calculated."""
        expected_result = np.array([-2.639057, -0.133531, -0.133531])
        result = self.train._initial_bias()
        np.testing.assert_array_almost_equal(result, expected_result)

    @patch.object(Train, "_import_function")
    @patch.object(Train, "_initial_bias")
    def test_load_model_with_initial_bias(self, mock_initial_bias, mock_import_function):
        """Test loading a model when initial_bias is used."""
        
        mock_create_model = MagicMock()
        mock_import_function.return_value = mock_create_model

        mock_initial_bias.return_value = np.array([0.5, -0.5])

        result = self.train.load_model()
        self.assertEqual(result, mock_create_model())

    @patch.object(Train, "_import_function")
    @patch.object(Train, "_initial_bias", return_value=False)
    def test_load_model_withouy_initial_bias(self, mock_initial_bias, mock_import_function):
        """Test loading a model when initial_bias is used."""
        mock_create_model = MagicMock()
        mock_import_function.return_value = mock_create_model
        
        self.train.model_args = {**self.train.model_args, "initial_bias": False}
        result = self.train.load_model()
        self.assertEqual(result, mock_create_model())

    @patch.object(Train, "_import_function", return_value="Not Callable")
    @patch.object(Train, "_initial_bias", return_value=False)
    def test_load_model_non_callable(self, mock_initial_bias, mock_import_function):
        """Test that TypeError is raised when create_model is not callable."""
        with self.assertRaises(TypeError):
            self.train.load_model()

    def test_compile_model_with_dict_optimizer(self):
        """Test compiling the model with a dictionary optimizer config."""
        model = MagicMock()
        result = self.train.compile_model(model)
        model.compile.assert_called_once()  # Ensure compile() was called
        self.assertEqual(result, model)

    def test_compile_model_with_string_optimizer(self):
        """Test compiling the model with a string-based optimizer."""
        self.train.optimizer = "sgd"  # Use optimizer string instead of a dictionary
        model = MagicMock()
        result = self.train.compile_model(model)
        model.compile.assert_called_once()
        self.assertEqual(result, model)

    def test_compile_model_with_invalid_optimizer(self):
        """Test that ValueError is raised for an invalid optimizer type."""
        self.train.optimizer = {"type": "invalid_optimizer"}
        model = MagicMock()

        with self.assertRaises(ValueError) as context:
            self.train.compile_model(model)

        self.assertIn("Optimizer type 'invalid_optimizer' is not recognized.", str(context.exception))

    @patch.object(Train, "_import_function")
    @patch("tensorflow.keras.optimizers.Adam")
    def test_compile_model_with_custom_loss(self, mock_adam, mock_import_function):
        """Test compiling the model with a valid custom loss function."""
        mock_loss_fn = MagicMock()
        mock_import_function.return_value = mock_loss_fn

        mock_optimizer = MagicMock()
        mock_adam.return_value = mock_optimizer

        self.train.custom_loss = {
            "custom_loss_path": "/mock/path/to/custom_loss",
            "module_name": "custom_loss_fn"
        }
        model = MagicMock()
        result = self.train.compile_model(model)

        mock_import_function.assert_called_once_with("/mock/path/to/custom_loss", "custom_loss_fn")
        mock_adam.assert_called_once()
        model.compile.assert_called_once_with(
            optimizer=mock_optimizer,
            loss=mock_loss_fn,
            metrics=["accuracy"]
        )
        self.assertEqual(result, model)

    @patch.object(Train, "_import_function")
    def test_compile_model_with_invalid_custom_loss(self, mock_import_function):
        """Test that an error is raised when the custom loss function cannot be imported."""
        mock_import_function.side_effect = ImportError("Could not import loss function.")

        self.train.custom_loss = {
            "custom_loss_path": "/mock/path/to/custom_loss",
            "module_name": "custom_loss_fn"
        }
        model = MagicMock()

        with self.assertRaises(ImportError) as context:
            self.train.compile_model(model)

        self.assertIn("Could not import loss function.", str(context.exception))

    def test_compile_model_with_no_optimizer(self):
        """Test that a model compiles correctly when no optimizer is provided."""
        self.train.optimizer = {}
        model = MagicMock()
        result = self.train.compile_model(model)

        model.compile.assert_called_once()
        self.assertEqual(result, model)

    def test_compile_model_with_no_loss(self):
        """Test that the model compiles with no loss when loss is None."""
        self.train.loss = None
        model = MagicMock()
        result = self.train.compile_model(model)

        model.compile.assert_called_once()
        self.assertEqual(result, model)

    @patch("builtins.open", new_callable=mock_open, read_data='{"feature1": "metadata1", "feature2": "metadata2"}')
    @patch("glob.glob")
    @patch("os.path.join", side_effect=lambda *args: "/".join(args))
    def test_load_feature_metadata_exact_match(self, mock_path_join, mock_glob, mock_file):
        """Test that feature metadata is loaded correctly."""
        mock_glob.return_value = ["/mock/data/category_feature_description.json"]

        result = self.train._load_feature_metadata("category")

        expected_result = {"feature1": "metadata1", "feature2": "metadata2"}
        self.assertEqual(result, expected_result)

        mock_file.assert_called_once_with("/mock/data/category_feature_description.json", "r")

    @patch("builtins.open", new_callable=mock_open, read_data='{"feature1": "metadata1"}')
    @patch("glob.glob")
    @patch("os.path.join", side_effect=lambda *args: "/".join(args))
    def test_load_feature_metadata_closest_match(self, mock_path_join, mock_glob, mock_file):
        """Test loading feature metadata when exact match is missing but one close match exists."""
        mock_glob.return_value = ["/mock/data/category_v2_feature_description.json"]

        result = self.train._load_feature_metadata("category")

        expected_result = {"feature1": "metadata1"}
        self.assertEqual(result, expected_result)

        mock_file.assert_called_once_with("/mock/data/category_v2_feature_description.json", "r")

    @patch("glob.glob")
    @patch("os.path.join", side_effect=lambda *args: "/".join(args))
    def test_load_feature_metadata_multiple_matches(self, mock_path_join, mock_glob):
        """Test error when multiple matching files are found."""
        mock_glob.return_value = [
            "/mock/data/category_v1_feature_description.json",
            "/mock/data/category_v2_feature_description.json"
        ]

        with self.assertRaises(FileNotFoundError) as context:
            self.train._load_feature_metadata("category")
        
        self.assertIn("Multiple feature description JSON files found", str(context.exception))

    @patch("glob.glob")
    @patch("os.path.join", side_effect=lambda *args: "/".join(args))
    def test_load_feature_metadata_no_match(self, mock_path_join, mock_glob):
        """Test error when no matching files are found."""
        mock_glob.return_value = []

        with self.assertRaises(FileNotFoundError) as context:
            self.train._load_feature_metadata("category")
        
        self.assertIn("Feature description JSON not found", str(context.exception))

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
        result = self.train._load_target_metadata()
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
        result = self.train._parse_tfrecord_fn(serialized_example, feature_metadata, category)
        expected_result = tf.convert_to_tensor([0.5, 42.0, 100.0], dtype=tf.float32)
        
        # Convert tensors to np.arrays for testing
        result = result.numpy()
        expected_result = expected_result.numpy()
        np.testing.assert_array_equal(result, expected_result)

    def test_group_files_by_number(self):
        """Test files are grouped correctly."""
        with tempfile.TemporaryDirectory() as test_dir:
            self.train.data_dir = test_dir

            dataset_type = "train"

            for i in range(3):
                open(os.path.join(self.train.data_dir, f"file_{i}_train.tfrecord"), "w").close()
                open(os.path.join(self.train.data_dir, f"targets_{i}_train.tfrecord"), "w").close()
            
            feature_category = "file"
            result = self.train._group_files_by_number(feature_category, dataset_type)

            expected_result = {
                0: {'features': [f'{self.train.data_dir}/file_0_train.tfrecord'], 
                    'targets': [f'{self.train.data_dir}/targets_0_train.tfrecord']}, 
                1: {'features': [f'{self.train.data_dir}/file_1_train.tfrecord'], 
                    'targets': [f'{self.train.data_dir}/targets_1_train.tfrecord']}, 
                2: {'features': [f'{self.train.data_dir}/file_2_train.tfrecord'], 
                    'targets': [f'{self.train.data_dir}/targets_2_train.tfrecord']}}
            self.assertEqual(result, expected_result)

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
                result = self.train._clean_bytes(input_value)
                self.assertEqual(result, expected_output)

    @patch("training.components.train.datetime")  # Patch datetime to return a fixed timestamp
    @patch("training.components.train.os.makedirs")  # Patch os.makedirs to prevent actual directory creation
    @patch("training.components.train.os.path.exists", return_value=False)  # Pretend save_dir doesn't exist
    @patch.object(Train, "_copy_file")  # Mock _copy_file to track calls
    def test_create_save_dir_and_store_success(self, mock_copy_file, mock_exists, mock_makedirs, mock_datetime):
        """Test that the function successfully creates a directory and copies files."""

        # Mock datetime to return a fixed value
        mock_datetime.now.return_value.strftime.return_value = "10_03_2025_12_30_00"

        expected_save_dir = "/mock/iteration_dir/10_03_2025_12_30_00"

        # Run function
        result = self.train._create_save_dir_and_store()

        # Assertions
        mock_exists.assert_called_once_with(expected_save_dir)
        mock_makedirs.assert_called_once_with(expected_save_dir)
        mock_copy_file.assert_any_call("/mock/path/to/config.py", expected_save_dir)
        mock_copy_file.assert_any_call("/mock/path/to/model.py", expected_save_dir)
        self.assertEqual(result, expected_save_dir)

    @patch("training.components.train.datetime")
    @patch("training.components.train.os.makedirs")
    @patch("training.components.train.os.path.exists", return_value=True)  # Simulate directory already existing
    @patch.object(Train, "_copy_file")
    def test_create_save_dir_and_store_already_exists(self, mock_copy_file, mock_exists, mock_makedirs, mock_datetime):
        """Test behavior when save directory already exists (should not call makedirs)."""

        # Mock datetime
        mock_datetime.now.return_value.strftime.return_value = "10_03_2025_12_30_00"
        expected_save_dir = "/mock/iteration_dir/10_03_2025_12_30_00"

        # Run function
        result = self.train._create_save_dir_and_store()

        # Assertions
        mock_exists.assert_called_once_with(expected_save_dir)
        mock_makedirs.assert_not_called()  # Since directory exists, it shouldn't create one
        mock_copy_file.assert_any_call("/mock/path/to/config.py", expected_save_dir)
        mock_copy_file.assert_any_call("/mock/path/to/model.py", expected_save_dir)
        self.assertEqual(result, expected_save_dir)

    @patch("training.components.train.datetime")
    @patch("training.components.train.os.makedirs")
    @patch("training.components.train.os.path.exists", return_value=False)
    @patch.object(Train, "_copy_file")
    def test_create_save_dir_and_store_missing_config(self, mock_copy_file, mock_exists, mock_makedirs, mock_datetime):
        """Test behavior when `config` key is missing from requirements_paths."""

        # Mock datetime
        mock_datetime.now.return_value.strftime.return_value = "10_03_2025_12_30_00"
        self.train.requirements_paths.pop("config")  # Remove config key

        with self.assertRaises(KeyError):
            self.train._create_save_dir_and_store()

        mock_exists.assert_not_called()  # Function should exit early
        mock_makedirs.assert_not_called()
        mock_copy_file.assert_not_called()

    @patch("training.components.train.datetime")
    @patch("training.components.train.os.makedirs")
    @patch("training.components.train.os.path.exists", return_value=False)
    @patch.object(Train, "_copy_file", side_effect=FileNotFoundError("Mocked missing file"))
    def test_create_save_dir_and_store_copy_failure(self, mock_copy_file, mock_exists, mock_makedirs, mock_datetime):
        """Test behavior when `_copy_file` raises an error (e.g., missing file)."""

        # Mock datetime
        mock_datetime.now.return_value.strftime.return_value = "10_03_2025_12_30_00"

        with self.assertRaises(FileNotFoundError) as context:
            self.train._create_save_dir_and_store()

        self.assertIn("Mocked missing file", str(context.exception))
        mock_exists.assert_called_once()
        mock_makedirs.assert_called_once()
        mock_copy_file.assert_called()  # Should still attempt copying

    @patch("os.path.exists", return_value=True)  # Mock file existence check
    @patch("builtins.open", new_callable=mock_open, read_data='{"0": 1.0, "1": 2.5, "2": 0.8}')  # Mock file content
    def test_load_class_weights_success(self, mock_file, mock_exists):
        """Test successfully loading class weights."""
        self.train.use_weight_dict = True  # Enable class weights
        expected_tensor = tf.constant([1.0, 2.5, 0.8], dtype=tf.float32)

        result = self.train._load_class_weights()
        
        # Ensure the tensor is correctly returned
        tf.debugging.assert_near(result, expected_tensor)
        mock_exists.assert_called_once_with(self.train.weight_dict_path)
        mock_file.assert_called_once_with(self.train.weight_dict_path, 'r')

    @patch("os.path.exists", return_value=False)  # File does not exist
    def test_load_class_weights_no_file(self, mock_exists):
        """Test when the weights file is missing."""
        self.train.use_weight_dict = True  # Enable class weights
        result = self.train._load_class_weights()
        self.assertIsNone(result)
        mock_exists.assert_called_once_with(self.train.weight_dict_path)

    @patch("os.path.exists", return_value=True)  # File exists but use_weight_dict is False
    @patch("builtins.open", new_callable=mock_open, read_data='{"0": 1.0, "1": 2.5, "2": 0.8}') 
    def test_load_class_weights_disabled(self, mock_file, mock_exists):
        """Test when `use_weight_dict` is False."""
        self.train.use_weight_dict = False  # Disable class weights
        result = self.train._load_class_weights()
        self.assertIsNone(result)
        
        # Ensure the function exits early and does not open the file
        mock_exists.assert_not_called()
        mock_file.assert_not_called()


if __name__ == "__main__":
    unittest.main()
