# create_targets.py


import importlib.util
import sqlite3
import pandas as pd
import gc

from .load_data import LoadData
from utils import EnvLoader, public_method


class CreateTargets():
    def __init__(self):        
        # Instance of the EnvLoader
        self.env_loader = EnvLoader()
        
        # Retrieve the path to the config module using the env_loader
        config_path = self.env_loader.get("DATA_PREP_CONFIG")
        
        # Dynamically import the config module
        self.config = self.env_loader.load_config_module(config_path)
        
        # Load specifications on the ETL process 
        # Options include loading, processing, and storing all data at once
        # or breaking the pipeline down by pair or further by target
        self.load_mode = self.config.get("load_mode")
        self.collect_pairs = self.config.get("collect_pairs")
        self.pair_query = self.config.get("pair_query")
        self.pair_list = self.config.get("pair_list")
        self.df_batch = self.config.get("df_batch")
        self.group_by = self.config.get("group_by")
        self.source_mode = self.config.get("source_mode")
        # May include the following for future functionality for granular control
        # over saving
        self.target_save_mode = self.config.get("target_save_mode")
        self.target_batch = self.config.get("target_batch")
        
        # Primary key columns
        self.primary_key = self.config.get("primary_key", [])
    
        # Path of this iteration's target module file
        self.module_path = self.config.get("target_modules_path")
        self.main_target_module = self.config.get("main_target_module")
        self.storage_map = self.config.get("target_storage_map")
        
        
    def _import_target_module(self):
        """
        Dynamically import the target module specified in `self.module_path` and 
        return the `targets` function.
        """
        spec = importlib.util.spec_from_file_location("target_module", self.module_path)
        target_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(target_module)
    
        # Retrieve the name of the main target function from config
        function_name = self.main_target_module
    
        # Access the specified target function dynamically
        targets_function = getattr(target_module, function_name)
        
        return targets_function
    
    def _drop_base_data_columns(self, df, cols_to_drop):
        """
        Remove the original columns from the DataFrame, leaving only targets.
        Args:
            df (pd.DataFrame): The DataFrame to process.
        Returns:
            pd.DataFrame: The DataFrame with only the target columns.
        """
        return df.drop(columns=cols_to_drop, errors="ignore")
    
    def _import_storage_map(self):
        """
        Dynamically import the module specified in `self.module_path` and return the 
        storage map defined in the module.
        """
        # Dynamically load the module
        spec = importlib.util.spec_from_file_location("storage_map_module", self.module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Retrieve the storage map name from config
        map_name = self.storage_map  # The name of the storage map object (e.g., "storage_map")
    
        # Access the storage map dynamically
        storage_map = getattr(module, map_name)
        
        return storage_map
    
    @public_method
    def store_targets(self, df, db_source):
        """
        Store the target DataFrame into its respective database tables based on 
        the storage map.
        
        Args:
            df (pd.DataFrame): The DataFrame containing all target columns.
        """
        # Retrieve the storage map
        storage_map = self._import_storage_map()
    
        # Get database path
        db_path = self.env_loader.get(db_source)
    
        # Connect to the database
        with sqlite3.connect(db_path) as conn:
            for table, columns in storage_map.items():
                primary_key = self.primary_key
                table_df = df[primary_key + columns]
    
                # Check if table exists
                cursor = conn.cursor()
                cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
                table_exists = cursor.fetchone()
    
                if not table_exists:
                    # Create table with primary key if it doesn't exist
                    column_defs = ", ".join([f"{col} TEXT" for col in primary_key + columns])
                    pk_defs = ", ".join(primary_key)
                    create_table_query = f"""
                    CREATE TABLE {table} (
                        {column_defs},
                        PRIMARY KEY ({pk_defs})
                    );
                    """
                    cursor.execute(create_table_query)
                    print(f"Table '{table}' created successfully.")
    
                # Prepare the `INSERT OR REPLACE` query
                columns_list = ", ".join(table_df.columns)
                placeholders = ", ".join(["?"] * len(table_df.columns))
                insert_query = f"""
                INSERT OR REPLACE INTO {table} ({columns_list})
                VALUES ({placeholders});
                """
    
                # Execute the batch insertion using executemany
                data_to_insert = table_df.to_records(index=False).tolist()
                cursor.executemany(insert_query, data_to_insert)
    
                print(f"Successfully stored targets in table: {table}")
                
    def _store_original_columns(self, df):
        """
        Create a list of original columns. Helps trim excess data when saving
        targets.
        """
        original_columns = [
            col for col in df.columns if col not in self.primary_key
        ]
        return original_columns
    
    def _group_dataframe(self, df, group_by_col):
        """
        Group the DataFrame by the specified column.
    
        Args:
            df (pd.DataFrame): The DataFrame to group.
            group_by_col (str): Column name to group by.
    
        Returns:
            pd.core.groupby.DataFrameGroupBy: Grouped DataFrame.
        """
        return df.groupby(group_by_col)
        
    def _create_pairs_list(self, pair_query):
        """
        Fetch a list of pairs using the pair query.
    
        Args:
            pair_query (dict): A dictionary with database path and query.
    
        Returns:
            list: A list of unique pairs.
        """
        source_type = list(pair_query.keys())[0]
        db_path = self.env_loader.get(source_type)
        query = list(pair_query.values())[0]
        with sqlite3.connect(db_path) as conn:
            return pd.read_sql_query(query, conn)["pair"].tolist()
        
    def _create_batches(self, pairs_list, batch_size):
        """
        Split pairs list into batches.
    
        Args:
            pairs_list (list): List of pairs to batch.
            batch_size (int): Size of each batch.
    
        Yields:
            list: A batch of pairs.
        """
        for i in range(0, len(pairs_list), batch_size):
            yield pairs_list[i:i + batch_size]
        
    def _process_and_save_dataframe(self, df, original_columns):
        """
        Process a DataFrame and save the calculated targets.
    
        Args:
            df (pd.DataFrame): The DataFrame to process.
        """
        target_function = self._import_target_module()
        processed_df = target_function(df)  # Apply target function
        target_df = self._drop_base_data_columns(processed_df, original_columns)  # Remove base columns
        self.store_targets(target_df, self.config["target_source"])  # Save targets
        del processed_df, target_df  # Free memory
        gc.collect()
        
    def _process_and_save_grouped(self, grouped_df, original_columns):
        """
        Process grouped DataFrames and save targets incrementally.
    
        Args:
            grouped_df (pd.DataFrameGroupBy): Grouped DataFrame.
        """
        target_function = self._import_target_module()
        for _, group in grouped_df:
            processed_df = target_function(group)  # Apply target function to group
            target_df = self._drop_base_data_columns(processed_df, original_columns)  # Remove base columns
            self.store_targets(target_df, self.config["target_source"])  # Save targets
            del processed_df, target_df  # Free memory
            gc.collect()
                    
    @public_method
    def calculate_and_store_targets(self):
        """
        Load data, calculate targets, and store them incrementally.
        Handles pair-by-pair, batch-by-batch, or full dataset based on config.
        """
        loader = LoadData()
    
        if self.load_mode == "full":
            # Load full dataset
            df = loader.load_data(self.source_mode)
            original_columns = self._store_original_columns(df)
            if self.group_by:
                grouped_df = self._group_dataframe(df, self.group_by)
                del df
                gc.collect()
                self._process_and_save_grouped(grouped_df, original_columns)
            else:
                self._process_and_save_dataframe(df, original_columns)
    
        elif self.load_mode == "pair":
            # Process data pair by pair
            pairs_list = (
                self._create_pairs_list(self.pair_query)
                if self.collect_pairs == "query"
                else self.pair_list
            )
            for pair in pairs_list:
                pair_df = loader.load_data(self.source_mode, pair=pair)
                original_columns = self._store_original_columns(pair_df)
                self._process_and_save_dataframe(pair_df, original_columns)
    
        elif self.load_mode == "batch":
            # Process data batch by batch
            pairs_list = (
                self._create_pairs_list(self.pair_query)
                if self.collect_pairs == "query"
                else self.pair_list
            )
            batched_pairs = self._create_batches(pairs_list, self.df_batch)
            for batch in batched_pairs:
                batch_df = loader.load_data(self.source_mode, batch=batch)
                original_columns = self._store_original_columns(batch_df)
                if self.group_by:
                    grouped_df = self._group_dataframe(batch_df, self.group_by)
                    del batch_df
                    gc.collect()
                    self._process_and_save_grouped(grouped_df, original_columns)
                else:
                    self._process_and_save_dataframe(batch_df, original_columns)
    
        else:
            raise ValueError(f"Invalid load_mode: {self.load_mode}")
