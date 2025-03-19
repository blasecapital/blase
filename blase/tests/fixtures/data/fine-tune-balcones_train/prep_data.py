# prep_data.py


import os
import sqlite3
import warnings
import importlib.util
import re
import json

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib
import numpy as np
import tensorflow as tf
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

from .load_training_data import LoadTrainingData
from utils import EnvLoader, public_method


class PrepData:
    def __init__(self):
        # Initialize the EnvLoader
        self.env_loader = EnvLoader()
        
        # Retrieve the path to the config module using the env_loader
        config_path = self.env_loader.get("DATA_TRAIN_CONFIG")
        
        # Dynamically import the config module
        self.config = self.env_loader.load_config_module(config_path)
        
        # Config specs
        self.module_path = self.config.get("data_processing_modules_path")
        self.primary_key = self.config.get("primary_key")
        self.feature_eng = self.config.get("feature_engineering")
        self.target_eng = self.config.get("target_engineering")
        self.feature_scaling = self.config.get("feature_scaling")
        self.scaler_save_dir = self.config.get("scaler_save_dir")
        self.prep_and_save_dict = self.config.get("prep_and_save")
        self.prepped_data_dir = self.config.get("prepped_data_dir")
        self.weight_dict_save_path = self.config.get("weight_dict_save_path")
        
        # Initialize weights_dict
        self.weights_dict = {}
        
        # Initialize data loader
        self.ltd = LoadTrainingData()
        
    def _import_function(self, function_name):
        """
        Dynamically import a module specified in `self.module_path` and 
        return the function from the arg.
        """
        spec = importlib.util.spec_from_file_location(
            "module", self.module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    
        # Access the specified filter function dynamically
        function = getattr(module, function_name)
        
        return function
    
    def _save_engineered_data(self, data, db_path, save_table):
        """
        Save the engineered data to the database using INSERT OR REPLACE.
    
        - Ensures the database file exists.
        - Dynamically creates the table if it does not exist.
        - Stores all columns as TEXT to ensure compatibility.
        - Uses `INSERT OR REPLACE` to avoid duplicate primary key conflicts.
    
        Args:
            data (pd.DataFrame): The engineered data.
            db_path (str): Path to the SQLite database.
            save_table (str): Table name where the data should be stored.
        """
        if data.empty:
            print(f"No data to save for table: {save_table}")
            return
    
        # Ensure the database path is valid
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
        # Convert all columns to string format for storage
        data = data.astype(str)
    
        # Define columns and enforce primary key
        column_definitions = ", ".join(f"{col} TEXT" for col in data.columns)
        primary_key_str = ", ".join(self.primary_key)
    
        create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {save_table} (
                {column_definitions},
                PRIMARY KEY ({primary_key_str})
            )
        """
        columns = ", ".join(data.columns)
        placeholders = ", ".join(["?"] * len(data.columns))
    
        insert_query = f"""
            INSERT OR REPLACE INTO {save_table} ({columns})
            VALUES ({placeholders})
        """
    
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
    
            # Create the table if it does not exist
            cursor.execute(create_table_query)
    
            # Execute batch insert
            cursor.executemany(insert_query, data.to_records(index=False))
    
            conn.commit()
    
        print(f"Engineered data successfully saved to {save_table} in {db_path}")
        
    @public_method
    def engineer(self, mode="all"):
        """
        Create new features or targets and add them to new database tables.
        
        Args:
            mode = Options: all, feature, or target.
        """
        # Determine which dictionaries to process
        if mode == 'feature':
            dict_list = [self.feature_eng]
        elif mode == 'target':
            dict_list = [self.target_eng]
        elif mode == 'all':
            dict_list = [self.feature_eng, self.target_eng]
        else:
            raise ValueError(f"Invalid mode: {mode}. Choose from 'all', 'feature', or 'target'.")

        for item in dict_list:
            for key in item.keys():
                if item is self.feature_eng:
                    source = 'feature_engineering'
                elif item is self.target_eng:
                    source = 'target_engineering'
                else:
                    raise ValueError(f"Unknown source for key: {key}")
                chunk_keys = self.ltd.chunk_keys(
                    mode='config',
                    source=source, 
                    key=key)  
                function_name = item[key][2]
                filter_function = self._import_function(function_name)
                for idx, chunk_key in enumerate(chunk_keys):
                    print(f"Processing chunk {chunk_key[0][0]} - {chunk_key[1][0]}...")
                    data = self.ltd.load_chunk(
                        mode='config',
                        source=source, 
                        key=key, 
                        chunk_key=chunk_key)
                    data = filter_function(data)
                    db_path = self.env_loader.get(item[key][0])
                    save_table = item[key][3]
                    self._save_engineered_data(data, db_path, save_table)
                    
    @public_method
    def scale(self, scaler_type='minmax'):
        """
        Load each features set, fit scaler on the training batches, and
        save the scale file.
        
        Args:
            scaler_type (str): 'minmax' or 'standard'
        """
        for key in self.feature_scaling.keys():
            db_path = self.env_loader.get(self.feature_scaling[key][0])
            table_list = self.feature_scaling[key][1]
            
            # Choose scaler
            scaler_class = MinMaxScaler if scaler_type == 'minmax' else StandardScaler
            scaler = scaler_class()
            
            for table in table_list:
                print(f"Beginning to process {table}...")
                query = f"""SELECT * FROM {table}"""
                chunk_keys = self.ltd.chunk_keys(
                    mode='manual',
                    db_path=db_path,
                    query=query
                    )
                
                for idx, chunk_key in enumerate(chunk_keys):
                    # Only load training data (60% of total dataset)
                    if idx < int(len(chunk_keys) * 0.6):
                        print(f"Processing chunk {chunk_key[0][0]} - {chunk_key[1][0]}...")
                        data = self.ltd.load_chunk(
                            mode='manual',
                            db_path=db_path,
                            query=query,
                            chunk_key=chunk_key
                            )
                        
                        # Remove primary keys before scaling
                        data = data.drop(columns=self.primary_key, 
                                                 errors='ignore')
                        
                        # Ensure there are numeric features to scale
                        if not data.empty:
                            scaler.partial_fit(data)
                    
                    # Stop loading chunks after training portion
                    else:
                        break
                
                # Save the fitted scaler
                scaler_save_path = os.path.join(self.scaler_save_dir, 
                                                f'{table}_scaler' + '.pkl')
                joblib.dump(scaler, scaler_save_path)
                print(f"Scaler saved at: {scaler_save_path}")
                    
    @public_method
    def encode_targets(self):
        """
        Use encoding logic to transform targets into numerical categories
        or appropiately format regression-based targets.
        """
        print("This would be a great feature!")
        
    def _save_metadata(self):
        """
        Keep track of primary-key based splits, completed processes, etc.
        """
        print("This would be a great feature!")
    
    def _convert_sqlite_text_to_types(self, df):
        """
        Convert all columns from SQLite text format to their appropriate data types.
        
        - If any value is a float, cast the entire column as float.
        - If all values are numeric and integers, cast as int.
        - Otherwise, cast as string.
        
        Special Handling:
        - Ensures 0.0, 1.0, and -1.0 are **not** converted to strings.
        
        Args:
            df (pd.DataFrame): DataFrame where all values are initially strings.
            
        Returns:
            pd.DataFrame: Converted DataFrame with numeric and categorical data.
        """
        for col in df.columns:
            sample_values = df[col].dropna().unique()[:min(100, len(df[col]))]  # Sample up to 100 values
    
            # Convert "None" or "NULL" to NaN
            df[col] = df[col].replace(["None", "NULL"], np.nan)
            
            # If any NaN values exist in the column, force it to be a string column
            if df[col].isna().any():
                df[col] = df[col].astype(str)
                continue
    
            # Step 1: Check if column contains only numeric values (integers or floats)
            is_numeric = any(re.match(r"^-?\d+(\.\d+)?$", str(val)) for val in sample_values)
    
            # Check if any value contains a decimal point
            contains_decimal = any("." in str(val) for val in sample_values)
    
            if is_numeric:
                df[col] = pd.to_numeric(df[col], errors="coerce")  # Convert to numeric
    
                # Step 2: If any value has a decimal, treat the whole column as float
                if contains_decimal:
                    df[col] = df[col].astype(np.float32)
                else:
                    # Ensure whole number floats like `1.0` → `1`
                    if all(df[col].dropna() % 1 == 0):  
                        df[col] = df[col].astype(np.int64)
                    else:
                        df[col] = df[col].astype(np.float32)
    
            else:
                df[col] = df[col].astype(str)  # Convert to string if not fully numeric
    
        return df
    
    def _save_feature_description(self, table, data):
        """
        Store each column's datatype in a .json file for TFRecord loading downstream.
        """
        json_path  = os.path.join(self.prepped_data_dir, f"{table}_feature_description.json")
        feature_description = {col: str(data[col].dtype) for col in data.columns}
        with open(json_path , "w") as f:
            json.dump(feature_description , f, indent=4)
        print(f"Feature description saved to {json_path }")
        
    def _apply_feature_description(self, table, df):
        """
        Loads the feature description JSON and applies the correct dtypes 
        to the incoming DataFrame.
        
        Args:
            table (str): Table name (to find the correct schema file).
            df (pd.DataFrame): The batch of data to be retyped.
        
        Returns:
            pd.DataFrame: DataFrame with consistent column types.
        """
        json_path = os.path.join(self.prepped_data_dir, f"{table}_feature_description.json")
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Feature description JSON not found: {json_path}")
    
        # Load the stored schema
        with open(json_path, "r") as json_file:
            feature_description = json.load(json_file)
    
        # Apply stored dtypes
        for col, dtype in feature_description.items():
            if dtype.startswith("float"):
                df[col] = df[col].astype(np.float32)
            elif dtype.startswith("int"):
                df[col] = df[col].astype(np.int64)
            else:
                df[col] = df[col].astype(str)
    
        return df
        
    def _bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):  # Handles EagerTensor
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))
    
    def _float_feature(self, value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[float(value)]))
    
    def _int64_feature(self, value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))
    
    def _save_tfrecord_chunk(self, data, filename):
        """
        Saves a Pandas DataFrame to a TFRecord file.
        
        Args:
            data (pd.DataFrame): The chunked DataFrame.
            filename (str): Path to the TFRecord file.
        """
        def serialize_example(row):
            feature_dict = {
                col: self._int64_feature(getattr(row, col)) if isinstance(getattr(row, col), (np.integer, int)) else
                      self._float_feature(getattr(row, col)) if isinstance(getattr(row, col), (np.floating, float)) else
                      self._bytes_feature(str(getattr(row, col))) 
                for col in data.columns
            }
            return tf.train.Example(features=tf.train.Features(feature=feature_dict)).SerializeToString()
    
        with tf.io.TFRecordWriter(filename) as writer:
            for row in data.itertuples(index=False):
                example = serialize_example(row)
                writer.write(example)

        print(f"Saved TFRecord: {filename}")
        
    def _create_and_save_weights_dict(self, data, target_col, fin):
        """
        Update and optionally finalize the weights dictionary with counts from the provided data.
        
        Args:
            data (pd.DataFrame): DataFrame containing the target column.
            target_col (str): Name of the column to use for computing class weights.
            fin (bool): True if this is the final training chunk, triggering 
            calculation of final weights and saving them.
            
        The function updates self.weights_dict with the count of each class from the current chunk.
        If fin is True, it computes the final class weights using an inverse frequency method, 
        then saves the resulting dictionary to self.weight_dict_save_path.
        """
        # Ensure weights_dict is initialized (should already be set in __init__)
        if not hasattr(self, 'weights_dict'):
            self.weights_dict = {}
        
        # Get a dictionary of counts for each unique value in the target column
        chunk_counts = data[target_col].value_counts().to_dict()
        
        # Update the weights_dict with these counts
        for cls, count in chunk_counts.items():
            self.weights_dict[cls] = self.weights_dict.get(cls, 0) + count
    
        # If this is the final chunk, compute the final class weights and save the dictionary.
        if fin:
            total_samples = sum(self.weights_dict.values())
            num_classes = len(self.weights_dict)
            # Compute weights: weight for class i = total_samples / (num_classes * count_i)
            final_weights = {cls: total_samples / (num_classes * count) for cls, 
                             count in self.weights_dict.items()}
            
            # Save the final weights to the specified JSON file.
            with open(self.weight_dict_save_path, 'w') as f:
                json.dump(final_weights, f)
            
            print("Final class weights saved to {}: {}".format(
                self.weight_dict_save_path, final_weights))
        
    @public_method
    def save_batches(self):
        """
        Split, type features and targets, and scale features, and
        save completely prepped data in batches to TFRecord files for efficient,
        iterative loading in model training.
    
        This function ensures that each table uses the same set of chunk_keys.
        The steps are:
          1. For each key in self.prep_and_save_dict, determine the list of tables.
          2. For each table, compute the chunk_keys.
          3. Find the set of chunk_keys from the table that has the maximum length.
          4. Process each table using that global set of chunk_keys.
        """
        # Step 1: Collect chunk keys from all tables across all keys
        chunk_keys_by_table = {}
    
        for key in self.prep_and_save_dict.keys():
            db_path = self.env_loader.get(self.prep_and_save_dict[key][0])
            table_list = list(self.prep_and_save_dict[key][1].keys())
    
            for table in table_list:
                if table not in chunk_keys_by_table:
                    print(f"Determining chunk keys for table: {table}...")
                    query = f"SELECT * FROM {table}"
                    chunk_keys = self.ltd.chunk_keys(
                        mode='manual',
                        db_path=db_path,
                        query=query
                    )
                    chunk_keys_by_table[table] = chunk_keys
    
        # Step 2: Find the single max_table across all keys
        max_table = max(chunk_keys_by_table, key=lambda t: len(chunk_keys_by_table[t]))
        global_chunk_keys = chunk_keys_by_table[max_table]
    
        print(f"Using global chunk keys from table '{max_table}' with {len(global_chunk_keys)} chunks for alignment.")

        # Step 3: Process all tables using the global chunk keys
        for key in self.prep_and_save_dict.keys():
            db_path = self.env_loader.get(self.prep_and_save_dict[key][0])
            table_list = list(self.prep_and_save_dict[key][1].keys())
    
            for table in table_list:
                print(f"Beginning to process {table}...")
                query = f"""SELECT * FROM {table}"""
                
                # Load scaler if needed
                scale = False
                if self.prep_and_save_dict[key][1][table]['scaler']:
                    scaler_path = os.path.join(self.scaler_save_dir, 
                                          f'{table}_scaler' + '.pkl')
                    scaler = joblib.load(scaler_path)
                    scale = True
                
                # Process each chunk using the global chunk keys.
                n_chunks = len(global_chunk_keys)
                n_train = int(n_chunks * 0.6)
                for idx, chunk_key in enumerate(global_chunk_keys):
                    print(f"Processing chunk {chunk_key[0][0]} - {chunk_key[1][0]}...")
                    data = self.ltd.load_chunk(
                        mode='manual',
                        db_path=db_path,
                        query=query,
                        chunk_key=chunk_key
                        )
                    
                    # Remove primary keys before scaling
                    if not self.prep_and_save_dict[key][1][table].get('keep_primary_key'):
                        data = data.drop(columns=self.primary_key, 
                                                 errors='ignore')
                    
                    # Convert text to appropriate types'
                    if idx == 0:
                        data = self._convert_sqlite_text_to_types(data)
                        self._save_feature_description(table, data)
                    else:
                        data = self._apply_feature_description(table, data)
                    
                    # Scale data if needed
                    if scale:
                        data_columns = [col for col in data.columns if col not in self.primary_key]  # Features only
                        data_index = data.index
                        original_dtypes = data.dtypes  # Save original dtypes
                    
                        # Apply scaling only to feature columns (excluding `date, pair`)
                        scaled_data = scaler.transform(data[data_columns])
                        scaled_df = pd.DataFrame(scaled_data, columns=data_columns, index=data_index)
                    
                        # Restore original dtypes to prevent unintended changes
                        for col in scaled_df.columns:
                            scaled_df[col] = scaled_df[col].astype(original_dtypes[col])
                        
                    # Determine dataset split: 60% train, 20% val, 20% test
                    if idx < n_train:
                        split_type = "train"
                        if self.prep_and_save_dict[key][1][table].get('weights_dict'):
                            target_col = self.prep_and_save_dict[key][1][table]['weights_dict']
                            fin = idx == n_train - 1
                            self._create_and_save_weights_dict(data, target_col, fin)
                    elif idx < int(n_chunks * 0.8):
                        split_type = "val"
                    else:
                        split_type = "test"
                        
                    # Define TFRecord filename
                    filename = os.path.join(
                        self.prepped_data_dir,
                        f"{table}_{idx}_{split_type}.tfrecord"
                    )
                        
                    # Save chunk to TFRecord
                    self._save_tfrecord_chunk(data, filename)
                    
    @public_method
    def validate_record(self, max_features=10):
        """
        Read the saved TFRecords and print the first entry for each unique
        table-file, limiting the number of features displayed.
        
        Args:
            max_features (int): Maximum number of features to display per record.
        """
        prepped_data_dir = self.prepped_data_dir
        file_list = os.listdir(prepped_data_dir)
        file_list = [file for file in file_list if '0' in file]
        
        for file in file_list:
            file_path = os.path.join(prepped_data_dir, file)
            raw_dataset = tf.data.TFRecordDataset(file_path)
            
            for raw_record in raw_dataset.take(1):
                example = tf.train.Example()
                example.ParseFromString(raw_record.numpy())
    
                # Extract feature dictionary
                feature_dict = example.features.feature
                
                # Select only the first `max_features` keys
                selected_features = list(feature_dict.keys())[:max_features]
                
                # Print condensed feature set
                print(f"{file} - Displaying {len(selected_features)} / {len(feature_dict)} features:")
                for feature in selected_features:
                    print(f"  {feature}: {feature_dict[feature]}")
    
                print("..." if len(feature_dict) > max_features else "")
                