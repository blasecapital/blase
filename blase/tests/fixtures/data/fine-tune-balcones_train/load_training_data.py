# load_training_data.py


import sqlite3
import math
import re

import pandas as pd

from utils import EnvLoader, public_method


class LoadTrainingData:
    def __init__(self):
        """
        Initialize the LoadData class with configuration and utility modules.
        """
        # Initialize the EnvLoader
        self.env_loader = EnvLoader()
        
        # Retrieve the path to the config module using the env_loader
        config_path = self.env_loader.get("DATA_TRAIN_CONFIG")
        
        # Dynamically import the config module
        self.config = self.env_loader.load_config_module(config_path)
        
        self.source_query = self.config.get('source_query')
        self.primary_key = self.config.get('primary_key')
        self.max_chunk_size = 20_000_000  # Limit on rows * columns per chunk
        
    def _get_row_count(self, db_path, query):
        """
        Get the number of rows the query will return.

        Args:
            db_path (str): Path to the SQLite database.
            query (str): SQL query.

        Returns:
            int: Number of rows the query will return.
        """
        cleaned_query = query.strip().rstrip(";")
        count_query = f"SELECT COUNT(*) FROM ({cleaned_query}) as temp_table"
        with sqlite3.connect(db_path) as conn:
            row_count = conn.execute(count_query).fetchone()[0]
        return row_count

    def _get_column_count(self, db_path, query):
        """
        Get the number of columns in the query result.

        Args:
            db_path (str): Path to the SQLite database.
            query (str): SQL query.

        Returns:
            int: Number of columns in the query result.
        """
        cleaned_query = query.strip().rstrip(";")
        limit_query = f"SELECT * FROM ({cleaned_query}) as temp_table LIMIT 1"
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute(limit_query)
            column_count = len(cursor.description)  # Number of columns
        return column_count

    @public_method
    def chunk_keys(self, mode, source=None, key=None, db_path=None, query=None):
        """
        Calculate primary key ranges for chunks based on dataset size
        and the maximum allowable chunk size. Supports composite primary keys.
    
        Args:
            mode (str): "config" to use source/key from config, "manual" to provide db_path and query.
            source (str, optional): Config key to retrieve data from (used in "config" mode).
            key (str, optional): Key from the source_query config (used in "config" mode).
            db_path (str, optional): Database path (used in "manual" mode).
            query (str, optional): SQL query to fetch data (used in "manual" mode).
    
        Returns:
            list of tuple: List of ((start_date, start_pair), (end_date, end_pair)) tuples for chunking.
        """
        if mode == "config":
            if not source or not key:
                raise ValueError("When using 'config' mode, 'source' and 'key' must be provided.")
            source = self.config.get(source)
            if key not in source:
                raise ValueError(f"Key '{key}' not found in configuration.")
            db_ref = source[key][0]
            query = source[key][1]
            db_path = self.env_loader.get(db_ref)
        elif mode == "manual":
            if not db_path or not query:
                raise ValueError("When using 'manual' mode, 'db_path' and 'query' must be provided.")
        else:
            raise ValueError("Invalid mode. Choose either 'config' or 'manual'.")
        
        # Get row and column counts
        row_count = self._get_row_count(db_path, query)
        column_count = self._get_column_count(db_path, query)
    
        # Calculate the initial chunk size
        max_rows_per_chunk = self.max_chunk_size // column_count
        split_factor = 5
    
        while row_count // split_factor > max_rows_per_chunk:
            split_factor += 5
    
        # Calculate the number of rows per chunk
        rows_per_chunk = math.ceil(row_count / split_factor)
    
        # Prepare the primary key and ordering logic
        primary_key_columns = ", ".join(self.primary_key)
        order_by_clause = ", ".join(self.primary_key)
    
        # Extract filtering logic from the source query
        base_query = query.strip().rstrip(";")
        select_primary_keys = f"""
            SELECT DISTINCT {primary_key_columns}
            FROM ({base_query})
            ORDER BY {order_by_clause}
        """
    
        # Retrieve all distinct primary key tuples for splitting
        with sqlite3.connect(db_path) as conn:
            primary_key_values = [tuple(row) for row in conn.execute(select_primary_keys)]
    
        # Split the primary key values into chunks
        chunk_keys = []
        for i in range(0, len(primary_key_values), rows_per_chunk):
            end_index = min(i + rows_per_chunk, len(primary_key_values))
            chunk_keys.append((primary_key_values[i], primary_key_values[end_index - 1]))
    
        return chunk_keys

    @public_method
    def load_chunk(self, mode, source=None, key=None, db_path=None, query=None, chunk_key=None): 
        """
        Load a chunk of data based on the key and chunk key.
    
        Args:
            mode (str): "config" to use source/key from config, "manual" to provide db_path and query.
            source (str, optional): Config key to retrieve data from (used in "config" mode).
            key (str, optional): Key from the source_query config (used in "config" mode).
            db_path (str, optional): Database path (used in "manual" mode).
            query (str, optional): SQL query to fetch data (used in "manual" mode).
            chunk_key (tuple): Tuple of (start_date, end_date) for chunk boundaries.
    
        Returns:
            pandas.DataFrame: DataFrame containing the data for the chunk.
        """
        if mode == "config":
            if not source or not key:
                raise ValueError("When using 'config' mode, 'source' and 'key' must be provided.")
            source = self.config.get(source)
            if key not in source:
                raise ValueError(f"Key '{key}' not found in configuration.")
            db_ref = source[key][0]
            query = source[key][1]
            db_path = self.env_loader.get(db_ref)
        elif mode == "manual":
            if not db_path or not query:
                raise ValueError("When using 'manual' mode, 'db_path' and 'query' must be provided.")
        else:
            raise ValueError("Invalid mode. Choose either 'config' or 'manual'.")
    
        if not chunk_key:
            raise ValueError("chunk_key must be provided to specify the date range.")
    
        # Strip and clean query
        base_query = query.strip().rstrip(";")
    
        # Ensure correct query formatting
        query_upper = base_query.upper()
    
        # Check if `WHERE` already exists in base_query
        where_exists = "WHERE" in query_upper
        order_exists = "ORDER BY" in query_upper
    
        # Extract `ORDER BY` if present
        if order_exists:
            order_index = query_upper.index("ORDER BY")
            base_query, order_part = base_query[:order_index].strip(), base_query[order_index:].strip()
        else:
            order_part = "ORDER BY date, pair"
    
        # Remove any existing date filtering** to prevent conflicts
        base_query = re.sub(r"AND\s+date\s*[<>]=?\s*'\d{4}-\d{2}-\d{2}'", "", base_query, flags=re.IGNORECASE)
    
        # Construct new WHERE condition correctly
        new_where_condition = "date >= ? AND date <= ?"
        if where_exists:
            modified_query = f"{base_query} AND {new_where_condition}"
        else:
            modified_query = f"{base_query} WHERE {new_where_condition}"
    
        # Reconstruct final query
        final_query = f"{modified_query} {order_part}"
        
        # Extract start_date and end_date as strings
        start_date = str(chunk_key[0][0])
        end_date = str(chunk_key[1][0])
    
        with sqlite3.connect(db_path) as conn:
            data = pd.read_sql_query(final_query, conn, params=(start_date, end_date))
        return data
               