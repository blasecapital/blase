# eval_inference.py


import importlib.util
import json
import re
import glob
import gc
import sqlite3
import functools

import os
# Suppress TensorFlow INFO logs and disable oneDNN custom operations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, log_loss, roc_auc_score
import matplotlib.pyplot as plt
import lime.lime_tabular

from utils import EnvLoader, public_method


class Eval:
    def __init__(self):        
        # Initialize the EnvLoader
        self.env_loader = EnvLoader()
        
        # Retrieve the path to the config module using the env_loader
        config_path = self.env_loader.get("EVALUATE_CONFIG")
        
        # Dynamically import the config module
        self.config = self.env_loader.load_config_module(config_path)
        
        self.backtest_config = self.config.get("backtest")
        self.explain_config = self.config.get("explain")
        self.metrics_config = self.config.get("metrics")
        self.cal_config = self.config.get("calibration")
        self.candidate_config = self.config.get("candidates")
        
    #-----------------------------------------------------#
    # General Utils
    #-----------------------------------------------------#
        
    def _import_function(self, module_path, function_name):
        """
        Dynamically import a module specified in `self.module_path` and 
        return the function from the arg.
        """
        spec = importlib.util.spec_from_file_location(
            "module", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    
        # Access the specified function dynamically
        function = getattr(module, function_name)
        
        return function
    
    def _initial_bias(self, model_args, weight_dict_path):
        """
        Compute initial bias for model output layer based on class distribution.
        
        Args:
            mode: self.backtest_config or self.forwardtest_config
        
        Returns:
            np.array: Initial bias values for each target class.
        """
        # Check if initial_bias is enabled in model_args
        if not model_args.get("initial_bias", False):
            return None  # No initial bias required
    
        # Load class weights from the JSON file
        if not os.path.exists(weight_dict_path):
            raise FileNotFoundError(f"Weight dictionary not found at: {weight_dict_path}")
    
        with open(weight_dict_path, "r") as f:
            class_weights = json.load(f)
    
        # Convert keys to integers (as JSON keys are stored as strings)
        class_weights = {int(k): v for k, v in class_weights.items()}
    
        # Compute class probabilities
        total_weight = sum(class_weights.values())
        class_probs = {k: v / total_weight for k, v in class_weights.items()}
    
        # Compute log-odds for initial bias
        epsilon = 1e-8  # Small value to prevent log(0)
        initial_bias = np.log([class_probs[i] / (1 - class_probs[i] + epsilon) for i in sorted(class_probs)])
    
        print("Computed Initial Bias.")
        return initial_bias   
    
    #-----------------------------------------------------#
    # Evaluate on preprocessed backtest data
    #-----------------------------------------------------#   
    
    # backtest predict utils --------------------------------#
    
    def _model_config(self, config):
        """
        Return the original iteration's config.py dict.
        """
        config_path = os.path.join(
            config["dir"], config["model_config_src"])
        model_config = self._import_function(
            config_path, config["config_dict_name"])
        return model_config
    
    def _group_pred_files(self, data_dir, feature_category, mode):
        """
        Go through the prepped data directory and group feature and target
        sets by their number.
        """
        tfrecord_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.tfrecord')])
        
        if mode in ['train', 'val', 'test']:
            feature_files = [f for f in tfrecord_files if f.startswith(feature_category) and mode in f]
            target_files = [f for f in tfrecord_files if "targets" in f and mode in f]
        elif mode == 'full':
            feature_files = [f for f in tfrecord_files if "targets" not in f]
            target_files = [f for f in tfrecord_files if "targets" in f]
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'full', 'train', 'val', or 'test'.")

        file_dict = {}

        for f in feature_files:
            match = re.search(r'_(\d+)_', f)
            if match:
                num = int(match.group(1))
                if num not in file_dict:
                    file_dict[num] = {'features': [], 'targets': []}
                file_dict[num]['features'].append(os.path.join(data_dir, f))

        for f in target_files:
            match = re.search(r'_(\d+)_', f)
            if match:
                num = int(match.group(1))
                if num in file_dict:  # Ensuring targets align with features
                    file_dict[num]['targets'].append(os.path.join(data_dir, f))

        return file_dict 
    
    def _load_feature_metadata(self, feature_category, data_dir):
        """
        Load feature metadata from the corresponding JSON file dynamically.
        - Searches for files matching the expected pattern.
        - If exact match is missing, searches for similar files.
        """
        # Define expected JSON filename pattern
        expected_filename = f"{feature_category}_feature_description.json"
        
        # Get all JSON files in the directory
        json_files = glob.glob(os.path.join(data_dir, "*.json"))
    
        # Check for exact match
        json_file_path = os.path.join(data_dir, expected_filename)
        if json_file_path in json_files:
            pass  # File exists, proceed with loading
    
        # If exact match is missing, try to find a closely matching file
        else:
            matching_files = [f for f in json_files if feature_category in os.path.basename(f)]
            
            if len(matching_files) == 1:
                json_file_path = matching_files[0]  # Use closest match
            elif len(matching_files) > 1:
                raise FileNotFoundError(f"Multiple feature description JSON files found for {feature_category}: {matching_files}. "
                                        f"Please specify the correct one.")
            else:
                raise FileNotFoundError(f"Feature description JSON not found for {feature_category}. "
                                        f"Expected: {expected_filename}, but found: {json_files}")
    
        # Load JSON metadata
        with open(json_file_path, "r") as f:
            metadata = json.load(f)
    
        return metadata
    
    def _load_target_metadata(self, data_dir):
        """
        Load target metadata from the corresponding JSON file.
        """
        json_file = os.path.join(data_dir, "targets_feature_description.json")
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"Target description JSON not found: {json_file}")

        with open(json_file, "r") as f:
            metadata = json.load(f)
            
            # Auto-fix incorrect data types
        for key, value in metadata.items():
            if value == "int64":
                metadata[key] = tf.io.FixedLenFeature([], tf.int64)  # Ensure correct int64 format
            elif value == "float32":
                metadata[key] = tf.io.FixedLenFeature([], tf.float32)  # Ensure correct float32 format
            elif value == "string":
                metadata[key] = tf.io.VarLenFeature(tf.string)  # Use VarLenFeature for string values
            elif value == "object":
                metadata[key] = tf.io.VarLenFeature(tf.string) # Comvert objects to strings
            else:
                raise ValueError(f"Unsupported data type '{value}' for key '{key}' in target metadata.")

        return metadata
    
    def _parse_tfrecord_fn(self, example_proto, feature_metadata, category,
                           feature_categories):
        """
        Parses a TFRecord example into features while ensuring proper reshaping.
        """
        # Dynamically assign TFRecord feature types based on metadata
        feature_description = {}
        for key, dtype in feature_metadata.items():
            if dtype in ["int64", "int"]:
                feature_description[key] = tf.io.FixedLenFeature([], tf.int64)
            elif dtype in ["float32", "float", "float64"]:
                feature_description[key] = tf.io.FixedLenFeature([], tf.float32)
            elif dtype in ["string", "object"]:
                feature_description[key] = tf.io.FixedLenFeature([], tf.string)
            else:
                raise ValueError(f"Unsupported dtype {dtype} for feature {key}")
                
        # Parse only the feature part (exclude the target)
        parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    
        # Group features by datatype.
        converted_features = {}
        for key, dtype in feature_metadata.items():
            value = parsed_example[key]
            if dtype in ["int64", "int"]:
                # If desired, you can choose to keep ints as ints; however, for stacking
                # you need a uniform type. So we cast them to float32.
                converted_features[key] = tf.cast(value, tf.float32)
            elif dtype in ["float32", "float", "float64"]:
                converted_features[key] = value  # Already float32.
            elif dtype in ["string", "object"]:
                # If the string represents a number, convert it; otherwise, handle separately.
                # Here we attempt to convert to float32.
                converted_features[key] = tf.strings.to_number(value, out_type=tf.float32)
        
        # Reassemble features in the original order.
        ordered_features = [converted_features[key] for key in feature_metadata.keys()]
        feature_tensor = tf.stack(ordered_features, axis=0)
    
        # Apply reshaping if required
        category_config = feature_categories[category]
        if category_config["reshape"]:
            feature_tensor = tf.reshape(feature_tensor, category_config["shape"])
    
        return feature_tensor
    
    @tf.autograph.experimental.do_not_convert
    def _load_tfrecord_dataset(self, feature_files, target_files, feature_categories,
                               data_dir, batch_size):
        """
        Load and process datasets from multiple aligned feature and target TFRecord files.
        Ensures correct structure for model input.
        """
        feature_metadata = {category: self._load_feature_metadata(category, data_dir) for category in feature_categories}
        target_metadata = self._load_target_metadata(data_dir)
        
        feature_datasets = {
            category: tf.data.TFRecordDataset(files).map(
                lambda x: self._parse_tfrecord_fn(x, feature_metadata[category], 
                                                  category, feature_categories),
                num_parallel_calls=tf.data.AUTOTUNE
            ) for category, files in feature_files.items()
        }
    
        # Load and parse target dataset separately (return full data + `target`)
        def parse_target_fn(x):
            parsed = tf.io.parse_single_example(x, target_metadata)
            return parsed, parsed['target']  # Return full parsed targets & just `target` for training
    
        target_dataset = tf.data.TFRecordDataset(target_files).map(
            parse_target_fn, num_parallel_calls=tf.data.AUTOTUNE
        )
    
        # Extract only the target column for training
        target_labels_dataset = target_dataset.map(lambda x, y: y, num_parallel_calls=tf.data.AUTOTUNE)
    
        # Zip full target data separately for saving CSV
        full_target_dataset = target_dataset.map(lambda x, y: x, num_parallel_calls=tf.data.AUTOTUNE)
    
        # Convert feature dictionary to tuple (preserves ordering)
        feature_tuple_dataset = tf.data.Dataset.zip((
            tuple(feature_datasets.values()),  # Tuple of feature inputs
            target_labels_dataset  # Only target label for training
        ))
    
        # Batch and prefetch for efficiency
        feature_tuple_dataset = feature_tuple_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        full_target_dataset = full_target_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
        return feature_tuple_dataset, full_target_dataset, list(target_metadata.keys())
    
    def _store_predictions(self, predictions, full_target_dataset, 
                           target_column_names, dataset_type):
        """
        Store predictions in the location specified in config.
        """
        # Extract iteration folder name for database name
        iteration_folder = os.path.basename(self.backtest_config["dir"])
        
        # Extract epoch number from model weights
        epoch_number = self.backtest_config["model_weights"].split("_")[0].replace("epoch", "")
        
        # Define Database Path
        db_name = f"{iteration_folder}_epoch{epoch_number}.db"
        db_path = os.path.join(self.backtest_config["save_pred_dir"], db_name)
        
        # Connect to Database (Create if not exists)
        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.cursor()
            
            # Extract primary key columns from config (ensure they exist in target_column_names)
            primary_keys = self.backtest_config["primary_keys"] 
            
            if not set(primary_keys).issubset(set(target_column_names)):
                raise ValueError(f"Primary keys {primary_keys} are missing from target_column_names.")

            # Extract target values
            actual_targets = []
            extra_target_values = []  # Stores extra columns from target dataset
            
            for full_targets in full_target_dataset:  # Looping through target values
                target_dict = {
                    col: (
                        tf.sparse.to_dense(full_targets[col]).numpy().tolist()  # Convert SparseTensor to dense
                        if isinstance(full_targets[col], tf.sparse.SparseTensor)  # Check if SparseTensor
                        else full_targets[col].numpy().tolist()
                    ) 
                    for col in target_column_names
                }
                
                # Extract main target
                actual_targets.extend(target_dict['target'])
                
                # Store all extra target values
                extra_target_values.extend([
                    [target_dict[col][i] for col in target_column_names if col != 'target']
                    for i in range(len(target_dict['target']))
                ])
        
            # Convert Predictions to NumPy
            predictions = np.array(predictions)
            
            # Extract Predicted Categories & Confidence
            predicted_categories = np.argmax(predictions, axis=1)
            confidences = np.max(predictions, axis=1)
            
            # Prepare columns for insertion
            extra_columns = [col for col in target_column_names if col != "target"]
            extra_column_definitions = ", ".join([f"{col} REAL" for col in extra_columns])
        
            # Construct Primary Key Clause
            primary_key_clause = ", ".join(primary_keys)
            
            # Create Table if it Doesn't Exist
            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS predictions (
                target INTEGER,
                predicted_category INTEGER,
                confidence REAL,
                split TEXT
                {', ' + extra_column_definitions if extra_columns else ''},
                PRIMARY KEY ({primary_key_clause})
            );
            """
            cursor.execute(create_table_query)
            
            # Prepare Data for Insertion
            data_to_insert = [
                (
                    int(actual_targets[i]), 
                    int(predicted_categories[i]), 
                    float(confidences[i]), 
                    str(dataset_type)  # Ensure dataset_type is a string
                ) + tuple(
                    str(x[0].decode('utf-8')) if isinstance(x, list) and x and isinstance(x[0], bytes) else  # Convert [b'value'] → 'value'
                    str(x) if isinstance(x, list) and x else  # Convert ['value'] → 'value'
                    "NULL" if x is None else  # Replace None with "NULL"
                    str(x)  # Convert all remaining values to strings
                    for x in extra_target_values[i]
                )
                for i in range(len(actual_targets))
            ]

            # Insert Data (Using ON CONFLICT IGNORE to prevent duplicate keys)
            insert_query = f"""
            INSERT INTO predictions (target, predicted_category, confidence, split
                {', ' + ', '.join(extra_columns) if extra_columns else ''})
            VALUES ({', '.join(['?'] * (4 + len(extra_columns)))})
            ON CONFLICT ({primary_key_clause}) DO UPDATE SET
                target = excluded.target,
                predicted_category = excluded.predicted_category,
                confidence = excluded.confidence,
                split = excluded.split
                {''.join([f", {col} = excluded.{col}" for col in extra_columns]) if extra_columns else ''};
            """
            cursor.executemany(insert_query, data_to_insert)
            
            # Commit and Close Connection
            conn.commit()
        finally:
            conn.close()
        
        print(f"Predictions successfully stored in {db_name} ({dataset_type} set)")
        
    # backtest predict main functions -----------------------#
    
    def _load_model(self, config):
        """
        Use the backtest config to source the model weights, architecture,
        and arguements.
        """
        # Create the args to import model architecture
        model_modules_path = os.path.join(
            config["dir"], config["create_model_module_src"])
        model_function = config["model_module"]
        
        # Load the create_model function for the iteration
        create_model = self._import_function(
            model_modules_path, model_function)
        
        # Load the model args
        model_config = self._model_config(config)
        model_args = model_config[config["model_args"]]
        weight_dict_path = model_config["weight_dict_path"]
        
        # Compute initial bias if needed
        initial_bias = self._initial_bias(model_args, weight_dict_path)
        
        # Pass model arguments, updating the initial_bias field if applicable
        if initial_bias is not None:
            model_args["initial_bias"] = initial_bias
        
        # Pass the model's parameters from the config dict
        if callable(create_model):
            model = create_model(**model_args)
        else:
            raise TypeError("create_model is not callable")
            
        return model
    
    def _load_weights(self, model, config):
        """
        Use the backtest config's specified weights file to load into model
        architecture.
        """
        weights_path = os.path.join(
            config["dir"], config["model_weights"])
        
        model.load_weights(weights_path)
        print("Weights loaded successfully.")
        return model
    
    def _backtest_predict(self, model, mode, config):
        """
        Group the prepped data, load them as datasets, predict, and save 
        outcomes. 
        
        Args:
            model (obj): tf model with loaded weights
            mode (str): 'full', 'train', 'val', or 'test' specify which 
                        datasets to make predictions with
        """
        # Loop through all numbered datasets (0, 1, 2, etc.)
        model_config = self._model_config(config)
        data_dir = model_config["data_dir"]
        batch_size = model_config["batch_size"]
        feature_categories = model_config["feature_categories"]
        
        # Group the prepped data so feature sets are bundled with corresponding
        # target sets
        predict_groups = {category: self._group_pred_files(data_dir, category, mode)
                          for category in feature_categories}
        
        for num in sorted(predict_groups[next(iter(feature_categories))].keys()):
            feature_files = {
                category: [
                    f for f in predict_groups[category].get(num, {}).get('features', []) 
                    if category in os.path.basename(f)  # Ensures files are correctly matched
                ]
                for category in feature_categories
            }
            target_files = predict_groups[
                next(iter(feature_categories))].get(num, {}).get('targets', [])
            
            # Extract dataset type (train, val, or test) from any feature file name
            dataset_type = None
            if feature_files:
                sample_file = next(iter(feature_files[next(iter(feature_files))]), None)
                if sample_file:
                    dataset_type = os.path.basename(sample_file).split("_")[-1].split(".")[0]  # Extract last part before extension
                    
            # Ensure all feature categories are present
            if any(not files for files in feature_files.values()) or not target_files:
                continue  # Skip if any feature category is missing

            print(f"Predicting set {num}")
            
            # Load dataset with all feature sets
            dataset, full_target_dataset, target_column_names = (
                self._load_tfrecord_dataset(feature_files, target_files, feature_categories,
                                            data_dir, batch_size))
            
            predictions = model.predict(dataset)
            self._store_predictions(
                predictions, full_target_dataset, target_column_names, dataset_type)
            
            del dataset, full_target_dataset, target_column_names
            gc.collect()
        
    @public_method
    def predict_and_store(self, mode='full'):
        """
        Load data, model, and respective specs from the iteration specified in
        config. Perform model.predict and save results.
        
        Args:
            mode (str): 'full', 'train', 'val', or 'test' specify which 
                        datasets to make predictions with
        """
        config = self.backtest_config
        model = self._load_model(config)
        model = self._load_weights(model, config)
        self._backtest_predict(model, mode, config)
        
    #-----------------------------------------------------#
    # Backtest explain
    #-----------------------------------------------------#
    
    # backtest explain utils -----------------------#
    
    def _group_lime_files(self, data_dir):
        tfrecord_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.tfrecord')])
        
        file_num = str(self.explain_config["file_num"])
        
        feature_files = [f for f in tfrecord_files if ("features" in f or "feature" in f) and file_num in f]
        target_files = [f for f in tfrecord_files if "targets" in f and file_num in f]

        file_dict = {}

        for f in feature_files:
            match = re.search(r'_(\d+)_', f)
            if match:
                num = int(match.group(1))
                if num not in file_dict:
                    file_dict[num] = {'features': [], 'targets': []}
                file_dict[num]['features'].append(os.path.join(data_dir, f))

        for f in target_files:
            match = re.search(r'_(\d+)_', f)
            if match:
                num = int(match.group(1))
                if num in file_dict:  # Ensuring targets align with features
                    file_dict[num]['targets'].append(os.path.join(data_dir, f))
        
        return file_dict 
    
    def _extract_numpy_features(self, dataset, batch_size=1):
        """
        Extract a NumPy array of feature samples from the dataset.
        """
        feature_list = []
    
        for batch in dataset.take(batch_size):  # Get only a small batch for LIME
            features, _ = batch  # Unpack dataset tuple
            if isinstance(features, tuple):
                # Flatten the tuple elements and concatenate them if multi-input
                features = np.hstack([f.numpy().reshape(len(f), -1) for f in features])
            else:
                features = features.numpy().reshape(len(features), -1)
    
            feature_list.append(features)
    
        return np.vstack(feature_list) if feature_list else np.empty((0, 1))
    
    def _save_feature_shapes(self, dataset):
        """
        Dynamically extract and store input shapes from the dataset's element_spec.
        """        
        input_specs = dataset.element_spec
        
        if not isinstance(input_specs, (list, tuple)):
            input_specs = [input_specs]
        
        # Extract the true input shapes, handling multiple input tensors
        lime_input_shapes = []
        for spec in input_specs:
            if isinstance(spec, tuple):  # If the spec itself is a tuple (multi-input case)
                lime_input_shapes.extend([tuple(s.shape[1:]) for s in spec])  
            else:
                lime_input_shapes.append(tuple(spec.shape[1:]))
        
        # Compute the flattened dimensions for each input.
        lime_flat_dims = [np.prod(shape) for shape in lime_input_shapes]
        
        # Compute cumulative indices to know where each input's flattened features begin and end.
        lime_cum_indices = np.cumsum([0] + lime_flat_dims)
        
        return lime_input_shapes, lime_cum_indices
    
    def _lime_predict(self, X, model, lime_input_shapes, lime_cum_indices):
        """
        Wrapper function for model.predict to ensure LIME receives 
        probabilities as a NumPy array.
        """
        # Prepare the list of inputs for the model.
        inputs = []
        for i in range(len(lime_input_shapes)):
            start = int(lime_cum_indices[i])
            end = int(lime_cum_indices[i + 1])

            # Slice the corresponding columns from X.
            input_i_flat = X[:, start:end]

            # Check if the input is empty
            if input_i_flat.size == 0:
                print(f"Warning: Empty input detected for index {i} (start={start}, end={end}) - Skipping")
                continue  # Skip this input to avoid errors
            
            # Reshape the flat array back to its original shape: (n_samples, *input_shape).
            input_i = input_i_flat.reshape(-1, *lime_input_shapes[i])
            inputs.append(input_i)

        if not inputs:
            raise ValueError("No valid inputs found for model prediction!")

        predictions = model.predict(inputs)
        
        return predictions
            
    def _list_cat_cols(self, feature_names):
        """
        Identify categorical feature names based on regex pattern from explain_config.
        
        Args:
            feature_names (list): List of all feature names extracted from the dataset.
        
        Returns:
            tuple: (categorical_feature_names, continuous_feature_names)
        """
        # Retrieve regex pattern for categorical features
        cat_feature_pattern = self.explain_config["categorical_feature_id"]
    
        # Separate categorical and continuous features
        categorical_features = [col for col in feature_names if re.search(cat_feature_pattern, col)]
        continuous_features = [col for col in feature_names if col not in categorical_features]
    
        return categorical_features, continuous_features
    
    def _clean_bytes(self, value):
        if isinstance(value, list):  # Extract from list
            value = value[0] if len(value) > 0 else ''
        
        if isinstance(value, bytes):  # Decode bytes
            value = value.decode('utf-8')

        if isinstance(value, str):  # Remove leading "b'" and trailing "'"
            value = value.replace("[b'", "").replace("']", "")

        return value
    
    def _extract_target_info(self, full_target_dataset, target_column_names, sample_num):
        """
        Extract target information from the full_target_dataset and combine all batches
        into one big target_dict.
        """
        # Initialize a dictionary with empty lists for each target column
        big_target_dict = {col: [] for col in target_column_names}
        
        actual_targets = []
        extra_target_values = []  # Stores extra columns from target dataset
    
        # Loop through each batch in the dataset
        for full_targets in full_target_dataset:
            # Convert each column from the batch to a list, handling SparseTensors if needed
            current_target_dict = {
                col: (
                    [val[0] if isinstance(val, list) and len(val) == 1 else val for val in 
                        tf.sparse.to_dense(full_targets[col]).numpy().tolist()]
                    if isinstance(full_targets[col], tf.sparse.SparseTensor)
                    else [val[0] if isinstance(val, list) and len(val) == 1 else val for val in 
                        full_targets[col].numpy().tolist()]
                )
                for col in target_column_names
            }
            
            # Extend the big_target_dict with the values from the current batch
            for col in target_column_names:
                big_target_dict[col].extend(current_target_dict[col])
            
            # Extract main target values (for additional processing)
            actual_targets.extend(current_target_dict['target'])
            
            # Store extra target values (all columns except 'target')
            extra_target_values.extend([
                [current_target_dict[col][i] for col in target_column_names if col != 'target']
                for i in range(len(current_target_dict['target']))
            ])
        
        # Now, big_target_dict contains all rows from all batches.
        df = pd.DataFrame(big_target_dict)
        df = df.applymap(self._clean_bytes)
        return df.iloc[sample_num]
    
    def _complete_lime_obs_info(self, target_info):
        """
        Connect to the prediction database and return the observation data
        related to the LIME analysis.
        """
        conn = sqlite3.connect(self.explain_config["prediction_dir"])
        
        try:
            # Extract column names for filtering
            id_cols = self.explain_config["id_cols"]  # List of column names like ["date", "pair", "target"]
    
            # Ensure all id_cols exist in target_info
            if not all(col in target_info.index for col in id_cols):
                raise ValueError(f"Missing required ID columns in target_info: {id_cols}")
    
            # Construct the WHERE clause dynamically
            where_conditions = " AND ".join([f"{col} = ?" for col in id_cols])
            query = f"""
            SELECT * FROM predictions
            WHERE {where_conditions}
            """
    
            # Extract filter values from target_info
            filter_values = tuple(target_info[col] for col in id_cols)
            
            # Execute the query
            df = pd.read_sql_query(query, conn, params=filter_values)
    
        finally:
            conn.close()
        
        return df
        
    # backtest explain main functions -----------------------#
    
    def _load_lime_data(self, config):
        model_config = self._model_config(config)
        data_dir = model_config["data_dir"]
        batch_size = model_config["batch_size"]
        feature_categories = model_config["feature_categories"]
        
        # Group the prepped data so feature sets are bundled with corresponding
        # target sets
        predict_groups = {category: self._group_lime_files(data_dir)
                          for category in feature_categories}
        
        for num in sorted(predict_groups[next(iter(feature_categories))].keys()):
            feature_files = {
                category: [
                    f for f in predict_groups[category].get(num, {}).get('features', []) 
                    if category in os.path.basename(f)  # Ensures files are correctly matched
                ]
                for category in feature_categories
            }
            target_files = predict_groups[
                next(iter(feature_categories))].get(num, {}).get('targets', [])
            
            # Ensure all feature categories are present
            if any(not files for files in feature_files.values()) or not target_files:
                print("Skipping files...")
                continue  # Skip if any feature category is missing

            print(f"Loading set {num}...")
            
            # Load dataset with all feature sets
            dataset, full_target_dataset, target_column_names = (
                self._load_tfrecord_dataset(feature_files, target_files, feature_categories,
                                            data_dir, batch_size))
            
            feature_metadata = {category: self._load_feature_metadata(category, data_dir) for category in feature_categories}
                        
            feature_names = []
            for category, metadata in feature_metadata.items():
                feature_names.extend(metadata.keys())
            
            print("LIME data successfully loaded.")
            
        return dataset, feature_names, full_target_dataset, target_column_names
    
    def _run_lime(self, dataset, feature_names, full_target_dataset, target_column_names, model):
        """
        Run LIME on a given dataset with proper feature name separation.
        """
        sample_num = self.explain_config["sample_num"]
        
        # Extract categorical and continuous feature names
        categorical_features, continuous_features = self._list_cat_cols(feature_names)
        lime_input_shapes, lime_cum_indices = self._save_feature_shapes(dataset)
    
        # Extract NumPy feature samples
        X_sample = self._extract_numpy_features(dataset, batch_size=1)
        print("Features extracted...")
        
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_sample,
            feature_names=feature_names,  # Full feature names
            categorical_features=[feature_names.index(f) for f in categorical_features],  # Pass indices
            class_names=self.explain_config["class_names"],
            mode="classification"
        )
        print("explainer initialized...")
        
        # Store display settings
        prev_display_width = pd.get_option("display.width")
        prev_max_columns = pd.get_option("display.max_columns")
    
        # Temporarily expand console output for better display
        pd.set_option("display.width", 150)
        pd.set_option("display.max_columns", None)
        
        target_info = self._extract_target_info(full_target_dataset, target_column_names, sample_num)
        complete_prediction_info = self._complete_lime_obs_info(target_info)
        print("\n ***Explaining the following:***")
        print(complete_prediction_info, "\n")
        
        # Use `functools.partial` to pass the model implicitly
        predict_fn = functools.partial(
            self._lime_predict, model=model, lime_input_shapes=lime_input_shapes, 
            lime_cum_indices=lime_cum_indices)
    
        explanation = explainer.explain_instance(
            X_sample[sample_num],  
            predict_fn  
        )
        
        lime_df = pd.DataFrame(explanation.as_list(), columns=["Feature", "Importance"])
        print(lime_df)
        explanation.as_pyplot_figure()
        plt.show()
       
        # Reset display settings
        pd.set_option("display.width", prev_display_width)
        pd.set_option("display.max_columns", prev_max_columns)
        
    @public_method
    def explain(self):
        """
        Predict with model and generate LIME explanations.
        
        Steps:
            1) Load model
            2) Load specific dataset
            3) Specify features (categorical and continuous)
        """
        # Load model
        config = self.explain_config
        model = self._load_model(config)
        model = self._load_weights(model, config)
        
        dataset, feature_names, full_target_dataset, target_column_names = self._load_lime_data(config)
        self._run_lime(dataset, feature_names, full_target_dataset, target_column_names, model)
        
    #-----------------------------------------------------#
    # Backtest metrics
    #-----------------------------------------------------#
    
    # backtest metrics utils -----------------------#
    
    def _load_data(self, db_path, query):
        """
        Load predictions data from the database using the query in config.
        """
        conn = sqlite3.connect(db_path)
        try:
            df = pd.read_sql(query, conn)
        finally:
            conn.close()
        
        return df
    
    def _metric_cats(self, df, config_dict, sub_dict):
        """
        Return a dictionary of categories to group metrics by. Each category
        refers to a list of its unique values.
        """
        metric_cats = {}
        for key in config_dict[sub_dict].keys():
            col_name = config_dict[sub_dict][key]
            metric_cats[key] = df[f'{col_name}'].unique()
        return metric_cats        
    
    def _accuracy_score(self, y_true, y_pred):
        acc = np.where(y_true == y_pred, 1, 0)
        return round(acc.mean(), 4)
    
    def _precision_score(self, y_true, y_pred, average="weighted", zero_division=0):
        """
        Compute precision using sklearn, handling division by zero.
        """
        return round(precision_score(y_true, y_pred, average=average, zero_division=zero_division), 4)
    
    def _recall_score(self, y_true, y_pred, average="weighted", zero_division=0):
        """
        Compute recall using sklearn, handling division by zero.
        """
        return round(recall_score(y_true, y_pred, average=average, zero_division=zero_division), 4)
    
    def _f1_score(self, y_true, y_pred, average="weighted", zero_division=0):
        """
        Compute F1-score using sklearn, handling division by zero.
        """
        return round(f1_score(y_true, y_pred, average=average, zero_division=zero_division), 4)
    
    def _log_loss(self, y_true, y_pred, y_conf):
        """
        Compute log loss, converting y_conf (confidence for predicted class) into 
        a full probability distribution.
        """
        # Ensure consistent class labels across y_true and y_pred
        unique_classes = np.unique(y_true)  # Get all possible classes in y_true
        n_classes = len(unique_classes)  # Total expected classes
        
        # If only one class exists, return None (ROC AUC requires multiple classes)
        if len(np.unique(y_true)) == 1 or len(np.unique(y_pred)) == 1:
            return None
    
        # Convert inputs to NumPy arrays
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        y_conf = np.array(y_conf).flatten()
    
        # Ensure y_pred contains only valid class indices
        y_pred = np.searchsorted(unique_classes, y_pred)  # Map y_pred to valid indices
    
        # Create a zero-matrix for probabilities
        y_prob = np.zeros((len(y_pred), n_classes))
    
        # Assign confidence to the predicted class column
        y_prob[np.arange(len(y_pred)), y_pred] = y_conf
    
        # Ensure sum of probabilities is 1 (assign remaining probability mass)
        remaining_prob = (1 - y_conf) / (n_classes - 1)
        y_prob[y_prob == 0] = np.repeat(remaining_prob, y_prob.shape[1] - 1)
    
        # Normalize probability distribution
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    
        # Compute log loss
        return round(log_loss(y_true, y_prob, labels=unique_classes), 4)
    
    def _roc_auc_score(self, y_true, y_pred, y_conf, multi_class="ovo"):
        """
        Compute ROC AUC score for multi-class classification.
        If only one class exists in y_true, return None.
        """
        unique_classes = np.unique(y_true)  # Get all unique classes from y_true
        n_classes = len(unique_classes)  # Number of expected classes
    
        # If only one class exists, return None (ROC AUC requires multiple classes)
        if len(np.unique(y_true)) == 1 or len(np.unique(y_pred)) == 1:
            return None
    
        # Convert arrays to NumPy format and ensure 1D shape
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        y_conf = np.array(y_conf).flatten()
    
        # Ensure y_pred only contains valid indices based on y_true classes
        y_pred = np.searchsorted(unique_classes, y_pred)  # Map y_pred to valid indices
    
        # Initialize zero-matrix for probability distribution
        y_prob = np.zeros((len(y_pred), n_classes))
    
        # Assign confidence scores to predicted classes
        y_prob[np.arange(len(y_pred)), y_pred] = y_conf
    
        # Ensure sum of probabilities = 1 (spread remaining probability mass)
        remaining_prob = (1 - y_conf) / (n_classes - 1)
        y_prob[y_prob == 0] = np.repeat(remaining_prob, y_prob.shape[1] - 1)
    
        # Normalize probability matrix to sum to 1 per row
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    
        # Compute ROC AUC
        return round(roc_auc_score(y_true, y_prob, multi_class=multi_class), 4)
    
    def _adjust_actual_target(self, df):
        """
        Modify the 'target' column:
        - If 'target' is 0 and 'buy_sl_time' is 'None', set it to 1.
        - If 'target' is 0 and 'sell_sl_time' is 'None', set it to 2.
        
        Args:
            df (pd.DataFrame): DataFrame containing 'target', 'buy_sl_time', and 'sell_sl_time'.
        
        Returns:
            pd.DataFrame: Updated DataFrame with modified 'target' values.
        """
        # Ensure 'target' is modified only where it's 0
        mask_buy = df["buy_sl_time"] == "None"
        mask_sell = df["sell_sl_time"] == "None"
    
        df.loc[mask_buy, "target"] = 1
        df.loc[mask_sell, "target"] = 2
    
        return df
    
    def _compute_metrics(self, y_true, y_pred, y_conf):
        """
        Compute classification metrics dynamically.
        """
        metrics = self.metrics_config.get("metrics", [])
        results = {}
        if "accuracy" in metrics:
            results["accuracy"] = self._accuracy_score(y_true, y_pred)
        if "precision" in metrics:
            results["precision"] = self._precision_score(y_true, y_pred, average="weighted", zero_division=0)
        if "recall" in metrics:
            results["recall"] = self._recall_score(y_true, y_pred, average="weighted", zero_division=0)
        if "f1_score" in metrics:
            results["f1_score"] = self._f1_score(y_true, y_pred, average="weighted", zero_division=0)
        if "log_loss" in metrics:
            results["log_loss"] = self._log_loss(y_true, y_pred, y_conf) if len(set(y_true)) > 1 else None
        if "roc_auc" in metrics:
            results["roc_auc"] = self._roc_auc_score(y_true, y_pred, y_conf, multi_class="ovo") if len(set(y_true)) > 1 else None
        
        return results        
    
    def _add_wl_col(self, df, config):
        pd.options.mode.chained_assignment = None
        y_true_col = config["y_true"]
        y_pred_col = config["y_pred"]
        df["wl"] = np.where(df[y_true_col] == df[y_pred_col], 1, 0)
        return df
    
    def _sort_by_conf(self, df, config):
        y_conf_col = config["y_conf"]
        df = df.sort_values(by=y_conf_col, ascending=False)
        return df
    
    def _fit_cal_fn(self, df):
        if df.empty:
            return [], []
        
        n = 100
        df = df.reset_index(drop=True)
        
        splits = np.array_split(df.index, n)
        if not splits:
            return [], []
        
        y_conf_col = self.cal_config["y_conf"]
        
        x_list = []
        y_list = []
        for i, split in enumerate(splits):
            if split.empty:  # Ensure the split is not empty
                continue
            beg, end = split[0], split[-1] + 1
            df_split = df.iloc[beg:end]
            if df_split.empty:
                continue
            
            x_list.append(round(df_split[y_conf_col].mean(), 4))
            y_list.append(round(df_split["wl"].mean(), 4))
        return x_list, y_list
    
    def _mse(self, x_list, y_list):
        if len(x_list) != len(y_list):
            raise ValueError("List lengths do not match")
    
        mse = np.mean((np.array(y_list) - np.array(x_list)) ** 2)
        results = {"mse": round(mse, 6)}
        return results
        
    def _plot_cal(self, x_list, y_list, title):
        comp = np.linspace(np.min(x_list), np.max(x_list))
        plt.plot(x_list, y_list)
        plt.plot(comp, comp)
        plt.xlabel("Pred Conf")
        plt.ylabel("Accuracy")
        plt.title(title)
        plt.legend(labels=["model", "benchmark"])
        plt.show()
        
    def _candidate_screen(self, df, accuracy_threshold, volume):
        df = df.reset_index(drop=True)
        df['cumulative_avg_wl'] = df['wl'].expanding().mean()
    
        last_valid_row = None  # Store the last row that meets the condition
    
        # Iterate through the DataFrame starting from the specified volume row
        for i in range(volume, len(df)):
            if df.loc[i, "cumulative_avg_wl"] > accuracy_threshold:
                last_valid_row = {
                    "cumulative_avg_wl": df.loc[i, "cumulative_avg_wl"],
                    "confidence": df.loc[i, "confidence"],
                    "rows_from_top": i
                }
    
        # Return the last valid row found, or an empty dict if none met the condition
        return last_valid_row if last_valid_row else {}
        
    # backtest metrics main functions -----------------------#
        
    def _display_results(self, db_path, query, mode='convert'):
        """
        Print metrics by split, category, and pair. Includes functions to convert
        target to its original category if it was engineered during
        data preparation (for example, if 'buy' or 'sell' signals were limited 
        if the trade time is less than x hours). 
        
        Args:
            mode (str): must be either 'convert' (default) or 'original', convert
                        will undo any target engineering
        
        Metrics:
            Classification accuracy, log loss, AUC, precision, recall, F1 score,
            and confusion matrix.
            
        Categories:
            Split, pair, and category.
        """
        df = self._load_data(db_path, query)
        if mode == 'convert':
            df = self._adjust_actual_target(df)
        metric_cats = self._metric_cats(df, self.metrics_config, "metric_categories")
        
        y_true_col = self.metrics_config["y_true"]
        y_pred_col = self.metrics_config["y_pred"]
        y_conf_col = self.metrics_config["y_conf"]
        split_col = self.metrics_config["metric_categories"]["split_col"]
        asset_col = self.metrics_config["metric_categories"]["asset_col"]
        target_col = self.metrics_config["metric_categories"]["target_col"]
    
        # Store display settings
        prev_display_width = pd.get_option("display.width")
        prev_max_columns = pd.get_option("display.max_columns")
    
        # Temporarily expand console output for better display
        pd.set_option("display.width", 150)
        pd.set_option("display.max_columns", None)
    
        #  Overall Metrics Table
        print("\n **Overall Metrics:**")
        overall_metrics = self._compute_metrics(df[y_true_col], df[y_pred_col], df[y_conf_col])
        overall_df = pd.DataFrame([overall_metrics])
        print(overall_df.to_string(index=False))
        
        #  Split-Level Breakdown
        for split in metric_cats["split_col"]:
            df_split = df[df[split_col] == split].copy()
            split_metrics = self._compute_metrics(df_split[y_true_col], df_split[y_pred_col], df_split[y_conf_col])
            split_df = pd.DataFrame([split_metrics])
    
            print(f"\n **Metrics for Split: {split}**")
            print(split_df.to_string(index=False))
    
            #  Asset (Pair) Breakdown
            pair_data = []
            for pair in metric_cats["asset_col"]:
                df_pair = df_split[df_split[asset_col] == pair].copy()
                if df_pair.empty:
                    continue
    
                pair_metrics = self._compute_metrics(df_pair[y_true_col], df_pair[y_pred_col], df_pair[y_conf_col])
                pair_metrics["pair"] = pair
                pair_data.append(pair_metrics)
    
            if pair_data:
                pair_df = pd.DataFrame(pair_data).set_index("pair")
                print("\n **Metrics by Pair:**")
                print(pair_df.to_string())
    
            #  Target Category Breakdown within Each Pair
            target_data = []
            for pair in metric_cats["asset_col"]:
                df_pair = df_split[df_split[asset_col] == pair].copy()
                for target in metric_cats["target_col"]:
                    df_target = df_pair[df_pair[target_col] == target].copy()
                    if df_target.empty:
                        continue
    
                    target_metrics = self._compute_metrics(df_target[y_true_col], df_target[y_pred_col], df_target[y_conf_col])
                    target_metrics["pair"] = pair
                    target_metrics["target_category"] = target
                    target_data.append(target_metrics)
    
            if target_data:
                target_df = pd.DataFrame(target_data).set_index(["pair", "target_category"])
                print("\n **Metrics by Pair & Target Category:**")
                print(target_df.to_string())
                
        # Reset display settings
        pd.set_option("display.width", prev_display_width)
        pd.set_option("display.max_columns", prev_max_columns)
        
    @public_method
    def report_metrics(self,
                       display_mode='original'):
        """
        Load the predictions from the database specified in config and display 
        specified metrics.
            
        display_results:
            Print metrics by split, category, and pair. Includes functions to convert
            target to its original category if it was engineered during
            data preparation.
            
        display_mode: arg for display_results, must be 'convert' which undos 
                      target engineering, or 'original'
        """
        db_path = self.metrics_config["db"]
        query = self.metrics_config["query"]
        self._display_results(db_path, query, display_mode)
        
    def _calculate_calibration(self, df, mode):
        """
        Optionally modify the y_true column according to mode, create the 
        win/loss column, sort df by prediction confidence, fit calibration function
        (x=pred_conf, y=interval win/loss rate), compare calibration function
        to ideal calibration (y=x) and report metric, and plot the comparison.  
        Perform these operations holistically and for any grouping specification
        in config.
        """
        if mode == 'convert':
            df = self._adjust_actual_target(df)
        cal_cats = self._metric_cats(df, self.cal_config, "cal_categories")
            
        df = self._add_wl_col(df, self.cal_config)
        df = self._sort_by_conf(df, self.cal_config)
        
        def calibration(df, title):
            x_list, y_list = self._fit_cal_fn(df)
            mse = self._mse(x_list, y_list)
            self._plot_cal(x_list, y_list, title)
            return mse
        
        split_col = self.metrics_config["metric_categories"]["split_col"]
        asset_col = self.metrics_config["metric_categories"]["asset_col"]
        target_col = self.metrics_config["metric_categories"]["target_col"]
    
        #  Overall Metrics Table
        print("\n **Overall Metrics:**")
        title = "Overall"
        mse = calibration(df, title)
        overall_df = pd.DataFrame([mse])
        print(overall_df.to_string(index=False))
        
        #  Split-Level Breakdown
        for split in cal_cats["split_col"]:
            df_split = df[df[split_col] == split].copy()
            split_metrics = calibration(df_split, split)
            split_df = pd.DataFrame([split_metrics])
    
            print(f"\n **Metrics for Split: {split}**")
            print(split_df.to_string(index=False))
            
            #  Asset (Pair) Breakdown
            pair_data = []
            for pair in cal_cats["asset_col"]:
                df_pair = df_split[df_split[asset_col] == pair].copy()
                if df_pair.empty:
                    continue
    
                pair_metrics = calibration(df_pair, split + " " + pair)
                pair_metrics["pair"] = pair
                pair_data.append(pair_metrics)
    
            if pair_data:
                pair_df = pd.DataFrame(pair_data).set_index("pair")
                print("\n **Metrics by Pair:**")
                print(pair_df.to_string())
            
            #  Target Category Breakdown within Each Pair
            target_data = []
            for pair in cal_cats["asset_col"]:
                df_pair = df_split[df_split[asset_col] == pair].copy()
                for target in cal_cats["target_col"]:
                    df_target = df_pair[df_pair[target_col] == target].copy()
                    if df_target.empty:
                        continue
    
                    target_metrics = calibration(
                        df_target, split + " " + pair + " " + str(target))
                    target_metrics["pair"] = pair
                    target_metrics["target_category"] = target
                    target_data.append(target_metrics)
    
            if target_data:
                target_df = pd.DataFrame(target_data).set_index(["pair", "target_category"])
                print("\n **Metrics by Pair & Target Category:**")
                print(target_df.to_string())
               
    @public_method
    def report_calibration(self,
                           mode='original'):
        """
        Load the predictions from the database specified in the config, calculate
        calibration using prediction confidence, plot calibration chart(s),
        and report calibration metrics.
        
        Args:
            mode (str): 'original' to keep targets as they are or 'convert' to
                          undo any target engineering (revert to real outcomes)
        """
        db_path = self.cal_config["db"]
        query = self.cal_config["query"]
        df = self._load_data(db_path, query)
        self._calculate_calibration(df, mode)
        
    def _find_candidates(self, df, accuracy_threshold, volume):
        """
        Filter the df by the specified classes in config, create the win/loss 
        column, sort df by prediction confidence, calculate cumulative accuracy, 
        check if the highest confidence observations meet threshold criteria,
        and run custom reporting module if specified in the config.
        """
        filter_col = self.candidate_config["class_filter"]["column_name"]
        class_filter = self.candidate_config["class_filter"]["classes"]
        df = df[df[filter_col].isin(class_filter)]
        df = self._add_wl_col(df, self.candidate_config)
        df = self._sort_by_conf(df, self.candidate_config)
        asset_col = self.candidate_config["asset_col"]
        asset_list = df[asset_col].unique()
        
        candidate_dict = {}
        for asset in asset_list:
            asset_df = df[df[asset_col].isin([asset])].copy()
            findings = self._candidate_screen(asset_df, accuracy_threshold, volume)
            candidate_dict[asset] = findings
        
        filtered_candidates = [
            {"asset": asset, 
             "win_rate": data["cumulative_avg_wl"], 
             "confidence_threshold": data["confidence"], 
             "volume": data["rows_from_top"]}
            for asset, data in candidate_dict.items() if data]  # Filter out empty dicts
            
        candidate_df = pd.DataFrame(filtered_candidates)
        print("\n", candidate_df, "\n")
        
        if self.candidate_config["custom_func"]:
            module_path = self.candidate_config["custom_func_path"]
            function_name = self.candidate_config["custom_func_name"]
            custom_eval_func = self._import_function(module_path, function_name)
            if callable(custom_eval_func):
                custom_eval_func(df)
            else:
                raise TypeError("custom_eval_func is not callable")
        
    @public_method
    def report_candidates(self, 
                          mode='convert',
                          accuracy_threshold=.35,
                          volume=300
                          ):
        """
        Load the predictions from the database specified in the config, calculate
        performance metrics based on arg specifications, report which assets
        meet specifications, and optionally run a custom evaluation function,
        such as running profit.
        
        Args:
            mode (str): 'original' to keep targets as they are or 'convert' to
                          undo any target engineering (revert to real outcomes)
            accuracy_threshold (float): specify the minimum accuracy
            volume (int): specify minimum sample size (i.e. number of trades
                          above desired accuracy)
        """        
        db_path = self.candidate_config["db"]
        query = self.candidate_config["query"]
        df = self._load_data(db_path, query)
        if mode == 'convert':
            df = self._adjust_actual_target(df)
        self._find_candidates(df, accuracy_threshold, volume)
        