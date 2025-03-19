# load_data.py


import sqlite3
import pandas as pd

from utils import EnvLoader, public_method


class LoadData:
    def __init__(self):
        """
        Initialize the LoadData class with configuration and utility modules.
        """
        # Instance of the EnvLoader
        self.env_loader = EnvLoader()
        
        # Retrieve the path to the config module using the env_loader
        config_path = self.env_loader.get("DATA_PREP_CONFIG")
        
        # Dynamically import the config module
        self.config = self.env_loader.load_config_module(config_path)
        
        # Configuration for features and targets
        self.sources = {
            "base": {
                "source": self.config.get("base_source"),
                "query": self.config.get("base_query")
                },
            "features": {
                "source": self.config.get("feature_source"),
                "query": self.config.get("feature_query"),
            },
            "targets": {
                "source": self.config.get("target_source"),
                "query": self.config.get("target_query"),
            },
        }

    def _load_from_database(self, db_path, query, params=None):
        """
        Internal helper function to load data from a database.
        
        Args:
            db_path (str): Path to the database.
            query (str): SQL query to execute.
            params (list): Parameters to substitute in the query.
        
        Returns:
            pd.DataFrame: The loaded data.
        """
        if not query:
            raise ValueError("Query is empty or not provided.")
        
        try:
            with sqlite3.connect(db_path) as conn:
                df = pd.read_sql_query(query, conn, params=params)
                print("Data loaded successfully...", df)
                return df
        except Exception as e:
            raise RuntimeError(f"Error loading data from the database: {e}")

    @public_method
    def load_data(self, mode, pair=None, batch=None):
        """
        Load data based on the specified mode (base, feature, or target).
        
        Args:
            mode (str): The mode of data to load ('base', 'features', or 'targets').
            pair (str): Optional; specific pair to load data for.
            batch (list): Optional; batch of pairs to load data for.
        
        Returns:
            pd.DataFrame: The loaded data.
        """
        if mode not in self.sources:
            raise ValueError(f"Invalid mode: {mode}. Must be 'base', 'features', or 'targets'.")

        source_config = self.sources[mode]
        source_type = source_config["source"]
        query_template = source_config["query"]

        print(f"Beginning to load {mode} data...")

        db_path = self.env_loader.get(source_type)
        if pair:
            # Single pair case
            query = query_template.format(placeholders="?")
            return self._load_from_database(db_path, query, params=[pair])
        elif batch:
            # Batch case
            placeholders = ",".join("?" for _ in batch)
            query = query_template.format(placeholders=placeholders)
            return self._load_from_database(db_path, query, params=batch)
        else:
            # Full dataset
            query = query_template.replace("WHERE pair IN ({placeholders})", "")
            return self._load_from_database(db_path, query)
            