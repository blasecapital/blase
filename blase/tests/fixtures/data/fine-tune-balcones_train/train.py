# train.py


import importlib.util
import gc
import json
import re
import glob
import os
from datetime import datetime
import shutil

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

# Suppress TensorFlow INFO logs and disable oneDNN custom operations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
keras.mixed_precision.set_global_policy("mixed_float16")
import numpy as np
import pandas as pd

from utils import EnvLoader, public_method


class Train:
    def __init__(self):
        # Initialize the EnvLoader
        self.env_loader = EnvLoader()
        
        # Retrieve the path to the config module using the env_loader
        config_path = self.env_loader.get("DATA_TRAIN_CONFIG")
        
        # Dynamically import the config module
        self.config = self.env_loader.load_config_module(config_path)
        
        # Initialize config attributes
        self.model_modules_path = self.config.get("model_modules_path")
        self.model_args = self.config.get("model_args")
        self.model_function = self.config.get("model_function")
        self.callback_function = self.config.get("callback_function")
        self.custom_loss = self.config.get("custom_loss")
        self.loss = self.config.get("loss")
        self.metrics = self.config.get("metrics")
        self.optimizer = self.config.get("optimizer")
        self.data_dir = self.config.get("data_dir")
        self.epochs = self.config.get("epochs")
        self.batch_size = self.config.get("batch_size")
        self.feature_categories = self.config.get("feature_categories")
        self.use_weight_dict = self.config.get("use_weight_dict")
        self.weight_dict_path = self.config.get("weight_dict_path")
        self.iteration_dir = self.config.get("iteration_dir")
        self.requirements_paths = self.config.get("requirements_paths")
        
    def _import_function(self, module_path, function_name):
        """
        Dynamically import a module specified in `self.module_path` and 
        return the function from the arg.
        """
        spec = importlib.util.spec_from_file_location(
            "module", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    
        # Access the specified filter function dynamically
        function = getattr(module, function_name)
        
        return function
    
    def _initial_bias(self):
        """
        Compute initial bias for model output layer based on class distribution.
        
        Returns:
            np.array: Initial bias values for each target class.
        """
        # Check if initial_bias is enabled in model_args
        if not self.model_args.get("initial_bias", False):
            return None  # No initial bias required
    
        # Load class weights from the JSON file
        if not os.path.exists(self.weight_dict_path):
            raise FileNotFoundError(f"Weight dictionary not found at: {self.weight_dict_path}")
    
        with open(self.weight_dict_path, "r") as f:
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
        
    @public_method
    def load_model(self):
        """
        Load a model defined in the model_specs.
        """
        # Load the create_model function for the iteration
        create_model = self._import_function(self.model_modules_path, self.model_function)
        
        # Compute initial bias if needed
        initial_bias = self._initial_bias()
        
        # Pass model arguments, updating the initial_bias field if applicable
        model_args = self.model_args.copy()  # Avoid modifying the original config
        if initial_bias is not None:
            model_args["initial_bias"] = initial_bias
        
        # Pass the model's parameters from the config dict
        if callable(create_model):
            model = create_model(**self.model_args)
        else:
            raise TypeError("create_model is not callable")
            
        return model
    
    @public_method
    def compile_model(self, model):
        """
        Compile the model based on its specifications.
        
        Args:
            model (tf.keras.Model): The TensorFlow model to compile.
            model_name (str): Name of the model for retrieving its specifications.
            
        Returns:
            tf.keras.Model: The compiled TensorFlow model.
        """
        # Retrieve model specifications
        optimizer_config = self.optimizer
        
        # Dynamically create the optimizer
        if isinstance(optimizer_config, dict):
            optimizer_type = optimizer_config.pop("type", "adam")  # Default to Adam if type is not specified
            optimizer_class = getattr(tf.keras.optimizers, optimizer_type.capitalize(), None)
            
            if optimizer_class is None:
                raise ValueError(f"Optimizer type '{optimizer_type}' is not recognized.")
            
            # Instantiate the optimizer with remaining parameters in optimizer_config
            optimizer = optimizer_class(**optimizer_config)
        else:
            # If `optimizer_config` is not a dictionary, assume it's a valid TensorFlow optimizer
            optimizer = tf.keras.optimizers.get(optimizer_config)
            
        # Determine the loss function:
        # If custom_loss is provided (non-empty dict), import and use that custom loss.
        if self.custom_loss and isinstance(self.custom_loss, dict) and len(self.custom_loss) > 0:
            # Use self.model_modules_path for the module path and "custom_loss" as the function name.
            path = self.custom_loss["custom_loss_path"]
            module_name = self.custom_loss["module_name"]
            loss_fn = self._import_function(path, module_name)
            print("Using custom loss function.")
        else:
            loss_fn = self.loss
            print("Using standard loss function:", self.loss)
        
        # Compile the model
        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=self.metrics
        )
        
        print("Successfully compiled model...")
        
        return model
        
    def _load_feature_metadata(self, feature_category):
        """
        Load feature metadata from the corresponding JSON file dynamically.
        - Searches for files matching the expected pattern.
        - If exact match is missing, searches for similar files.
        """
        # Define expected JSON filename pattern
        expected_filename = f"{feature_category}_feature_description.json"
        
        # Get all JSON files in the directory
        json_files = glob.glob(os.path.join(self.data_dir, "*.json"))
    
        # Check for exact match
        json_file_path = os.path.join(self.data_dir, expected_filename)
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
        
    def _load_target_metadata(self):
        """
        Load target metadata from the corresponding JSON file.
        """
        json_file = os.path.join(self.data_dir, "targets_feature_description.json")
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

    def _parse_tfrecord_fn(self, example_proto, feature_metadata, category):
        """
        Parses a TFRecord example into features while ensuring proper reshaping.
        """
        # Build the feature description based on metadata.
        feature_description = {}
        for key, dtype in feature_metadata.items():
            if dtype in ["int64", "int"]:
                # Keep original type during parsing.
                feature_description[key] = tf.io.FixedLenFeature([], tf.int64)
            elif dtype in ["float32", "float", "float64"]:
                feature_description[key] = tf.io.FixedLenFeature([], tf.float32)
            elif dtype in ["string", "object"]:
                feature_description[key] = tf.io.FixedLenFeature([], tf.string)
            else:
                raise ValueError(f"Unsupported dtype {dtype} for feature {key}")
    
        # Parse the TFRecord.
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
    
        # Now, all tensors in ordered_features are of type float32, so we can stack them.
        feature_tensor = tf.stack(ordered_features, axis=0)
    
        # Apply reshaping if the category configuration requires it.
        category_config = self.feature_categories[category]
        if category_config["reshape"]:
            feature_tensor = tf.reshape(feature_tensor, category_config["shape"])
    
        return feature_tensor
    
    @tf.autograph.experimental.do_not_convert
    def _load_tfrecord_dataset(self, feature_files, target_files):
        """
        Load and process datasets from multiple aligned feature and target TFRecord files.
        Ensures correct structure for model input.
        """
        feature_metadata = {category: self._load_feature_metadata(category) for category in self.feature_categories}
        target_metadata = self._load_target_metadata()
    
        feature_datasets = {
            category: tf.data.TFRecordDataset(files).map(
                lambda x: self._parse_tfrecord_fn(x, feature_metadata[category], category),
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
        feature_tuple_dataset = feature_tuple_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        full_target_dataset = full_target_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
    
        return feature_tuple_dataset, full_target_dataset, list(target_metadata.keys())

    def _group_files_by_number(self, feature_category, dataset_type):
        """
        Groups feature and target files by their number (0,1,2,...) for training order consistency.
        """
        tfrecord_files = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.tfrecord')])

        feature_files = [f for f in tfrecord_files if f.startswith(feature_category) and dataset_type in f]
        target_files = [f for f in tfrecord_files if "targets" in f and dataset_type in f]

        file_dict = {}

        for f in feature_files:
            match = re.search(r'_(\d+)_', f)
            if match:
                num = int(match.group(1))
                if num not in file_dict:
                    file_dict[num] = {'features': [], 'targets': []}
                file_dict[num]['features'].append(os.path.join(self.data_dir, f))

        for f in target_files:
            match = re.search(r'_(\d+)_', f)
            if match:
                num = int(match.group(1))
                if num in file_dict:  # Ensuring targets align with features
                    file_dict[num]['targets'].append(os.path.join(self.data_dir, f))

        return file_dict  # Returns {number: {features: [...], targets: [...]}}
        
    def _copy_file(self, source_path, export_path):
        """
        Copies a file from source_path to export_path.
        
        If export_path is a directory, the file is copied into that directory with its original filename.
        If export_path is a full file path, the file is copied to that location.
        
        Args:
            source_path (str): The path of the file to copy.
            export_path (str): The destination directory or file path where the file should be saved.
        
        Returns:
            None
        """
        # Check if the source file exists.
        if not os.path.isfile(source_path):
            print(f"Source file does not exist: {source_path}")
            return
    
        # Determine the final destination path.
        if os.path.isdir(export_path):
            # If export_path is a directory, append the source filename.
            destination_path = os.path.join(export_path, os.path.basename(source_path))
        else:
            # If export_path is a full file path, use it directly.
            destination_path = export_path
            # Ensure the destination directory exists.
            dest_dir = os.path.dirname(destination_path)
            if dest_dir and not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
        
        try:
            shutil.copy2(source_path, destination_path)
            file_name = os.path.basename(source_path)
            print(f"Successfully copied {file_name}.")
        except Exception as e:
            print(f"Error copying file: {e}")
            
    def _clean_bytes(self, value):
        if isinstance(value, list):  # Extract from list
            value = value[0] if len(value) > 0 else ''
        
        if isinstance(value, bytes):  # Decode bytes
            value = value.decode('utf-8')

        if isinstance(value, str):  # Remove leading "b'" and trailing "'"
            value = value.replace("[b'", "").replace("']", "")

        return value
    
    def _create_save_dir_and_store(self):
        """
        Create the directory to store the iteration metadata using the current
        time as the unqiue identifier. Store the iterations' config.py and
        model creation .py file.
        """
        # Ensure required files exist in `requirements_paths`
        required_keys = ["config", "model"]
        for key in required_keys:
            if key not in self.requirements_paths:
                raise KeyError(f"Missing required key '{key}' in `requirements_paths`.")

        # Create the unique save directory name
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        save_dir = os.path.join(self.iteration_dir, dt_string)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"Successfully created the save directory with {dt_string} suffix.")
            
        # Copy the config.py and file with model architecture information
        self._copy_file(self.requirements_paths["config"], save_dir)
        self._copy_file(self.requirements_paths["model"], save_dir)
        
        return save_dir
    
    def _load_class_weights(self):
        """
        Load class_weights if they exist, return None otherwise.
        Converts to TensorFlow tensor for efficient lookup.
        """
        if self.use_weight_dict and os.path.exists(self.weight_dict_path):
            with open(self.weight_dict_path, 'r') as f:
                class_weights = json.load(f)
            class_weights = {int(k): v for k, v in class_weights.items()}  # Convert keys to ints
            print("Successfully loaded class weights.")
    
            # Convert to TensorFlow tensor for faster access
            class_weights_tensor = tf.constant([class_weights[i] for i in sorted(class_weights)], dtype=tf.float32)
            return class_weights_tensor
    
        return None  # No class weights available
    
    def _add_sample_weights(self, features, target, class_weights_tensor):
        """
        Assigns a sample weight based on class_weights_tensor lookup.
        """
        sample_weight = tf.gather(class_weights_tensor, target)  # Lookup weight using target class
        return features, target, sample_weight
    
    def _val_prediction_data(self, model, val_dataset, full_target_dataset, target_column_names):
        """
        Run model.predict on the validation set and return predictions and
        confidences for export.
        """
        # Now, run predictions on this validation chunk.
        # Since our dataset is a tf.data.Dataset of (features, target),
        # we use model.predict to get the output probabilities.
        predictions = model.predict(val_dataset)
        # Collect actual targets from the dataset.
        # (We assume that the dataset yields (features, target) tuples.)
        actual_targets = []
        extra_target_values = []  # Store extra target column values
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
        # Ensure predictions is a NumPy array.
        predictions = np.array(predictions)
        # Extract Predicted Categories & Confidence
        predicted_categories = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1)
        
        return predicted_categories, actual_targets, confidences, extra_target_values
        
    @public_method
    @tf.autograph.experimental.do_not_convert
    def fit_model(self, model):
        """
        Iteratively loads aligned feature and target TFRecord files in batches,
        ensuring each epoch processes all aligned sets (0,1,2,...) before validation.
        Supports dynamically specified feature categories.
        """
        save_dir = self._create_save_dir_and_store()
        class_weights = self._load_class_weights()
        if self.callback_function:
            aggregated_callbacks = self._import_function(self.model_modules_path, self.callback_function)
            if callable(aggregated_callbacks):
                # Create an instance of AggregatedCallbacks with any parameters you want.
                log_dir = os.path.join(save_dir, "logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
                aggregated_callbacks = aggregated_callbacks(save_dir=save_dir, log_dir=log_dir)
                if aggregated_callbacks.use_tensorboard:
                    train_log_dir = os.path.join(log_dir, "train")
                    callbacks=[tf.keras.callbacks.TensorBoard(
                                log_dir=train_log_dir,
                                histogram_freq=1,
                                write_graph=False,
                                write_images=True,
                                update_freq='epoch',
                                profile_batch=0)]
                    
                    val_log_dir = os.path.join(log_dir, "val")  # Separate validation logs
                    val_callbacks = [tf.keras.callbacks.TensorBoard(
                        log_dir=val_log_dir,
                        histogram_freq=1,
                        write_graph=False,  # Avoid duplicate graph logs
                        write_images=False,  # No need to save images again
                        update_freq='epoch',
                        profile_batch=0
                    )]
                else:
                    callbacks=[]
                    val_callbacks = []
            else:
                raise TypeError("callback function is not callable")
        else:
            aggregated_callbacks = None
            callbacks = []
            val_callbacks = []
    
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            
            # Initialize accumulators for training metrics.
            train_loss_sum = 0.0
            train_acc_sum = 0.0
            num_train_batches = 0
            
            # -------------------------------
            # Training Phase: Process each training chunk.
            # -------------------------------
    
            # Training Phase (ensure all feature sets are loaded together)
            train_groups = {category: self._group_files_by_number(category, 'train') 
                            for category in self.feature_categories}
    
            # Loop through all numbered datasets (0, 1, 2, etc.)
            for num in sorted(train_groups[next(iter(self.feature_categories))].keys()):
                feature_files = {
                    category: train_groups[category].get(num, {}).get('features', []) 
                    for category in self.feature_categories}
                target_files = train_groups[
                    next(iter(self.feature_categories))].get(num, {}).get('targets', [])
    
                # Ensure all feature categories are present
                if any(not files for files in feature_files.values()) or not target_files:
                    continue  # Skip if any feature category is missing
    
                print(f"Training on set {num}")
    
                # Load dataset with all feature sets
                train_dataset, full_target_dataset, target_column_names = (
                    self._load_tfrecord_dataset(feature_files, target_files))
                
                # Add class_weights as a tensor
                if class_weights is not None:
                    train_dataset = train_dataset.map(
                        lambda x, y: self._add_sample_weights(x, y, class_weights))
    
                # Train model with all feature inputs; include class_weight if available
                history = model.fit(train_dataset, epochs=1, verbose=1, 
                                        callbacks=callbacks)
                    
                # Assuming 'loss' and 'accuracy' are tracked metrics
                # (Update the keys if your metric names differ.)
                batch_loss = history.history.get('loss', [0])[-1]
                batch_acc = history.history.get('accuracy', [0])[-1]
                train_loss_sum += batch_loss
                train_acc_sum += batch_acc
                num_train_batches += 1
    
                del train_dataset
                gc.collect()
                
            # Compute average training metrics for this epoch
            avg_train_loss = train_loss_sum / num_train_batches if num_train_batches else 0
            avg_train_acc = train_acc_sum / num_train_batches if num_train_batches else 0
    
            # -------------------------------
            # Validation Phase: Process each validation chunk.
            # -------------------------------
            # Initialize accumulators for validation metrics.
            val_loss_sum = 0.0
            val_acc_sum = 0.0
            num_val_batches = 0
            pred_rows = []  # Each element: [epoch, predicted_category, actual_target, prediction_confidence]
                
            val_groups = {category: self._group_files_by_number(category, 'val') 
                          for category in self.feature_categories}
    
            for num in sorted(val_groups[next(iter(self.feature_categories))].keys()):
                feature_files = {category: val_groups[category].get(num, {}).get('features', []) 
                                 for category in self.feature_categories}
                target_files = val_groups[next(iter(self.feature_categories))].get(num, {}).get('targets', [])
    
                if any(not files for files in feature_files.values()) or not target_files:
                    continue  
    
                print(f"Validating on set {num}")
    
                val_dataset, full_target_dataset, target_column_names = self._load_tfrecord_dataset(
                    feature_files, target_files)
                
                results = model.evaluate(val_dataset, verbose=1, callbacks=val_callbacks)
                
                # Typically, results[0] is loss and results[1] is accuracy.
                val_loss_sum += results[0]
                if len(results) > 1:
                    val_acc_sum += results[1]
                num_val_batches += 1
            
                # Run model.predict on validation set
                predicted_categories, actual_targets, confidences, extra_target_values = (
                    self._val_prediction_data(
                    model, val_dataset, full_target_dataset, target_column_names))
            
                # Prepare CSV Rows
                for i, (pred, actual, conf) in enumerate(zip(predicted_categories, actual_targets, confidences)):
                    pred_rows.append([epoch + 1, pred, actual, conf] + extra_target_values[i])
            
                del val_dataset
                gc.collect()
                
            # Log Confusion Matrix (Validation Only)
            if aggregated_callbacks and aggregated_callbacks.use_conf_matrix:
                if not isinstance(predicted_categories, np.ndarray):
                    predicted_categories = np.array(predicted_categories).reshape(-1)
                if not isinstance(actual_targets, np.ndarray):
                    actual_targets = np.array(actual_targets).reshape(-1)
                aggregated_callbacks.conf_matrix_callback.log_conf_matrix(epoch, predicted_categories, actual_targets)
            
            # Compute Average Validation Metrics
            avg_val_loss = val_loss_sum / num_val_batches if num_val_batches else 0
            avg_val_acc = val_acc_sum / num_val_batches if num_val_batches else 0
            
            # Save Model
            model_filename = (
                f"epoch{epoch}_"
                f"trainLoss_{avg_train_loss:.4f}_"
                f"trainAcc_{avg_train_acc:.4f}_"
                f"valLoss_{avg_val_loss:.4f}_"
                f"valAcc_{avg_val_acc:.4f}.h5"
            )
            model_path = os.path.join(save_dir, model_filename)
            model.save_weights(model_path)
            print(f"Saved model for epoch {epoch+1} at: {model_path}")
            
            # Append Predictions & Target Columns to CSV
            predictions_csv_path = os.path.join(save_dir, f"{epoch}_predictions.csv")
            
            # CSV Header
            csv_headers = ["epoch", "predicted_category", "actual_target", "prediction_confidence"] + [
                col for col in target_column_names if col != 'target'
            ]
            
            df_predictions = pd.DataFrame(pred_rows, columns=csv_headers)

            # Apply cleaning to all DataFrame cells
            df_predictions = df_predictions.applymap(self._clean_bytes)
            
            # Save DataFrame to CSV
            predictions_csv_path = os.path.join(save_dir, f"{epoch}_predictions.csv")
            df_predictions.to_csv(predictions_csv_path, mode="a", index=False, 
                                  header=not os.path.exists(predictions_csv_path))
            
            print(f"Appended predictions for epoch {epoch+1} to CSV file: {predictions_csv_path}")
            
            # -------------------------------
            # Aggregate metrics and update aggregated callbacks.
            # -------------------------------
            if aggregated_callbacks:
                aggregated_logs = {
                'train_loss': avg_train_loss,
                'train_accuracy': avg_train_acc,
                'val_loss': avg_val_loss,
                'val_accuracy': avg_val_acc
                }
                # Call the custom aggregated callbacks with overall epoch metrics.
                aggregated_callbacks.on_aggregated_epoch_end(epoch, aggregated_logs, model)
                
                # Optionally, check if early stopping was triggered.
                if model.stop_training:
                    print("Stopping further training due to early stopping.")
                    break

    @public_method
    def train_models(self):
        """
        Train all models defined in model_specs.

        Returns:
            dict: Training histories for each model.
        """
        model = self.load_model()
        model = self.compile_model(model)
        self.fit_model(model)
    