# clean_raw_data.py


import os
import re
import sqlite3
import warnings
import importlib.util
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

from .load_training_data import LoadTrainingData
from utils import EnvLoader, public_method


class CleanRawData:
    def __init__(self):
        
        # Initialize the EnvLoader
        self.env_loader = EnvLoader()
        
        # Retrieve the path to the config module using the env_loader
        config_path = self.env_loader.get("DATA_TRAIN_CONFIG")
        
        # Dynamically import the config module
        self.config = self.env_loader.load_config_module(config_path)
        
        # Config specs
        self.source_query = self.config.get('source_query')
        self.module_path = self.config.get("data_processing_modules_path")
        self.clean_functions = self.config.get("clean_functions")
        self.bad_keys_path = self.config.get("bad_keys_path")
        self.primary_key = self.config.get("primary_key")
        
        # Initialize data loader
        self.ltd = LoadTrainingData()
    
    def _get_progress_log(self, key):
        """
        Retrieve the progress log for the specified key. If the log does not exist, initialize it.
    
        Args:
            key (str): The key identifying the dataset being processed.
    
        Returns:
            dict: The progress log for the specified key.
        """
        if not hasattr(self, "_progress_logs"):
            self._progress_logs = {}  # Initialize a dictionary to store progress logs
    
        if key not in self._progress_logs:
            # Initialize the progress log for the given key
            self._progress_logs[key] = {
                "running_sum": {},
                "running_square_sum": {},
                "running_count": {},
                "running_min": {},
                "running_max": {},
                "bin_edges": {},
                "bin_frequencies": {},
                "chunk_stats": {},
                "bad_keys": [],
                "final_stats": None,
            }
        
        return self._progress_logs[key]
    
    def _update_bin_frequencies(self, feature, data, bin_edges, bin_frequencies):
        """
        Update bin frequencies for a feature based on the current chunk of data.
    
        Args:
            feature (str): The feature/column name.
            data (DataFrame): The current chunk of data.
            bin_edges (array): The edges of the bins.
            bin_frequencies (dict): A dictionary storing the frequencies of values in each bin,
                                    including underflow and overflow bins.
    
        Example:
            bin_edges = [0, 10, 20, 30]
            bin_frequencies = {"underflow": 0, "overflow": 0, 0: 0, 1: 0, 2: 0}
        """
        if feature not in data.columns:
            print(f"Feature '{feature}' not found in data columns.")
            return
    
        # Extract values for the feature, dropping NaNs
        values = data[feature].dropna().values
    
        # Digitize the values into bins
        bin_indices = np.digitize(values, bin_edges, right=False)
    
        # Ensure bins are initialized
        bin_frequencies.setdefault(feature, {"underflow": 0, "overflow": 0})
        for i in range(len(bin_edges) - 1):
            bin_frequencies[feature].setdefault(i, 0)
    
        # Update bin frequencies
        for value, idx in zip(values, bin_indices):
            if idx == 0:
                # Value is below the smallest bin
                bin_frequencies[feature]["underflow"] += 1
            elif idx > len(bin_edges) - 1:
                # Value is above the largest bin
                bin_frequencies[feature]["overflow"] += 1
            else:
                # Value falls within a valid bin
                bin_frequencies[feature][idx - 1] += 1
    
    def _calculate_percentiles(self, bin_edges, bin_frequencies, total_count):
        """
        Calculate percentiles (e.g., 25%, 50%, 75%) using binned data.
    
        Args:
            bin_edges (array): The edges of the bins.
            bin_frequencies (dict): A dictionary storing bin frequencies, including underflow and overflow bins.
            total_count (int): The total number of valid values for the feature.
    
        Returns:
            dict: A dictionary containing percentiles (25%, 50%, 75%).
        """
        if total_count == 0:
            return {"25%": None, "50%": None, "75%": None}
    
        # Extract bin frequencies as a list in bin order
        frequencies = [bin_frequencies.get(i, 0) for i in range(len(bin_edges) - 1)]
        cumulative_frequency = np.cumsum(frequencies)  # Cumulative frequency for interpolation
        
        percentiles = {}
        for percentile, label in [(0.25, "25%"), (0.5, "50%"), (0.75, "75%")]:
            target_count = total_count * percentile
    
            # Find the bin where the target count falls
            bin_idx = np.searchsorted(cumulative_frequency, target_count)
    
            if bin_idx == 0:
                # If the target count is in the first bin
                percentiles[label] = bin_edges[0]
            elif bin_idx >= len(bin_edges) - 1:
                # If the target count is in the last bin
                percentiles[label] = bin_edges[-1]
            else:
                # Interpolate within the bin
                bin_start = bin_edges[bin_idx - 1]
                bin_end = bin_edges[bin_idx]
                bin_frequency = frequencies[bin_idx - 1]
                prev_cumulative = cumulative_frequency[bin_idx - 1]
                curr_cumulative = cumulative_frequency[bin_idx]
    
                if bin_frequency > 0:
                    # Linear interpolation within the bin
                    interpolated_value = bin_start + (
                        (target_count - prev_cumulative) / (curr_cumulative - prev_cumulative)
                    ) * (bin_end - bin_start)
                    percentiles[label] = interpolated_value
                else:
                    # If the bin has zero frequency, fallback to the bin start
                    percentiles[label] = bin_start
    
        return percentiles
    
    def _finalize_describe_report(self, key, progress_log):
        """
        Finalize and generate aggregated statistics after processing all chunks.
        """
        final_stats = {}
    
        for feature in progress_log["running_sum"]:
            count = progress_log["running_count"][feature]
            if count == 0:
                continue
    
            # Calculate mean, std, min, and max
            mean = progress_log["running_sum"][feature] / count
            variance = (
                progress_log["running_square_sum"][feature] / count - mean ** 2
            )
            std_dev = np.sqrt(variance)
            min_value = progress_log["running_min"][feature]
            max_value = progress_log["running_max"][feature]
    
            # Calculate percentiles using bin data
            percentiles = self._calculate_percentiles(
                progress_log["bin_edges"][feature],
                progress_log["bin_frequencies"][feature],
                count,
            )
    
            # Store aggregated statistics
            final_stats[feature] = {
                "count": count,
                "mean": mean,
                "std": std_dev,
                "min": min_value,
                "25%": percentiles.get("25%"),
                "50%": percentiles.get("50%"),
                "75%": percentiles.get("75%"),
                "max": max_value,
            }
    
        # Save the final stats to the progress log
        progress_log["final_stats"] = final_stats
        
    def _initialize_bins_from_sql(self, col_list, key):
        """
        Initialize bin edges for each feature by querying the global MIN and MAX using SQL, 
        incorporating any filtering conditions from the source query.
        
        Args:
            col_list (list): List of feature/column names.
            key (str): The key identifying the dataset, used to derive the table name and database path.
        
        Returns:
            dict: A dictionary containing bin edges for each feature.
        """
        # Get the database path from the environment loader
        db_path = self.env_loader.get(self.source_query[key][0])
        query = self.source_query[key][1]
        
        # Extract table name and WHERE clause
        table_name = query.split("FROM")[1].split("WHERE")[0].strip()
        where_clause = query.split("WHERE")[1].strip() if "WHERE" in query else ""
    
        def process_feature(feature):
            """
            Helper function to query MIN and MAX for a single feature.
            """
            try:
                # Connect to the database
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
    
                # Build the query for MIN and MAX, including the WHERE clause if it exists
                min_max_query = f"SELECT MIN({feature}) AS min_val, MAX({feature}) AS max_val FROM {table_name}"
                if where_clause:
                    min_max_query += f" WHERE {where_clause}"
                
                # Execute the query and fetch results
                cursor.execute(min_max_query)
                result = cursor.fetchone()
                conn.close()
    
                if result and result[0] is not None and result[1] is not None:
                    min_val = float(result[0])
                    max_val = float(result[1])
                    return feature, np.linspace(min_val, max_val, num=101)
                else:
                    raise ValueError(f"No valid data found for feature '{feature}' in table '{table_name}' with filter '{where_clause}'.")
            except Exception as e:
                print(f"Error initializing bins for feature '{feature}': {e}")
                # Default fallback if MIN and MAX cannot be computed
                return feature, np.linspace(0, 1, num=101)
    
        # Use ThreadPoolExecutor to parallelize the feature processing with progress bar
        bin_edges = {}
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(process_feature, feature): feature for feature in col_list}
            with tqdm(total=len(futures), desc="Processing Features", unit="feature") as pbar:
                for future in as_completed(futures):
                    feature, edges = future.result()
                    bin_edges[feature] = edges
                    pbar.update(1)
    
        return bin_edges
    
    def _display_descriptive_stats(self, progress_log):
        """
        Display final stats for each feature as a well-structured DataFrame.
        """
        final_stats = progress_log['final_stats']
        
        # Create a DataFrame where rows are features and columns are statistics
        stats_df = pd.DataFrame(final_stats).T  # Transpose so features are rows
        
        # Temporarily adjust Pandas display options
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
            print("Descriptive Statistics for All Features:")
            print(stats_df)
        
    def _plot_descriptive_stats(self, progress_log):
        """
        Generate plots for features using the progress log.
        
        Args:
            progress_log (dict): The progress log containing stats and bin frequencies.
        """
        for feature, stats in progress_log["final_stats"].items():
            bin_edges = progress_log["bin_edges"][feature]
            bin_frequencies = progress_log["bin_frequencies"][feature]
    
            # Extract mean and standard deviation from final stats
            mean = stats["mean"]
            std_dev = stats["std"]
    
            # Define 3-sigma range
            lower_bound = mean - 4 * std_dev
            upper_bound = mean + 4 * std_dev
    
            # Identify bins inside the 3σ range
            valid_bins = [i for i, edge in enumerate(bin_edges[:-1]) if lower_bound <= edge <= upper_bound]
    
            # Extract valid bin frequencies
            frequencies = [bin_frequencies.get(i, 0) for i in valid_bins]
    
            # Add underflow and overflow bins
            underflow_count = sum(
                bin_frequencies[i] for i, edge in enumerate(bin_edges[:-1]) if edge < lower_bound
            )
            overflow_count = sum(
                bin_frequencies[i] for i, edge in enumerate(bin_edges[:-1]) if edge > upper_bound
            )
    
            # Add special bins for underflow and overflow
            frequencies.insert(0, underflow_count)
            frequencies.append(overflow_count)
            labels = ["< 4σ"] + [f"{bin_edges[i]:.2f}" for i in valid_bins] + ["> 4σ"]
    
            # Plot
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(frequencies)), frequencies, width=0.9, align="center", alpha=0.7)
            plt.title(f"Feature: {feature} - Bin Distribution with Overflow & Underflow")
            plt.xlabel("Bins")
            plt.ylabel("Frequency")
            plt.xticks(range(len(frequencies)), labels, rotation=45, ha="right")
            plt.tight_layout()
            plt.show() 
            
    def _plot_rates(self, data, plot_skip):
        """
        Plot candlestick charts from OHLC standardized data using mplfinance.
    
        Steps:
        1. Identify OHLC-standardized features and ensure they form complete sets.
        2. Verify that all unique numbers (time steps) are sequential.
        3. Reconstruct each row into a time series DataFrame for plotting.
        4. Use a dummy date range to avoid NaT errors.
        5. Use `pair` and `date` for the plot title.
        6. Plot the full candlestick sequence from highest to lowest standard.
    
        Args:
            data (DataFrame): Data containing `pair`, `date`, and OHLC standardized columns.
        """
        # Ensure required columns exist
        if "date" not in data.columns or "pair" not in data.columns:
            print("Missing 'date' or 'pair' column in dataset.")
            return
    
        # Identify OHLC feature sets based on standardized naming pattern
        ohlc_pattern = re.compile(r"^(open|high|low|close)_standard_(\d+)$")
        feature_groups = {}
    
        for col in data.columns:
            match = ohlc_pattern.match(col)
            if match:
                ohlc_type, num = match.groups()
                num = int(num)
                if num not in feature_groups:
                    feature_groups[num] = {"open": None, "high": None, "low": None, "close": None}
                feature_groups[num][ohlc_type] = col  # Store column names under OHLC category
    
        # Ensure each group contains all four OHLC components
        valid_groups = {num: cols for num, cols in feature_groups.items() if all(cols.values())}
        
        if not valid_groups:
            print("Missing OHLC components in the dataset.")
            return
        
        # Ensure numbers are sequential
        sorted_numbers = sorted(valid_groups.keys(), reverse=True)
        if sorted_numbers != list(range(sorted_numbers[0], sorted_numbers[-1] - 1, -1)):
            print("Numbers are not sequential.")
            return
    
        # Process each row as a full candlestick sequence
        for index in range(0, len(data), plot_skip):
            row = data.iloc[index]
            pair = row["pair"]  # Extract currency pair name
            date_str = row["date"]  # Extract date as string
    
            # Construct DataFrame for mplfinance
            ohlc_data = pd.DataFrame({
                "Open": [row[valid_groups[num]["open"]] for num in sorted_numbers],
                "High": [row[valid_groups[num]["high"]] for num in sorted_numbers],
                "Low": [row[valid_groups[num]["low"]] for num in sorted_numbers],
                "Close": [row[valid_groups[num]["close"]] for num in sorted_numbers],
            })
            
            # Ensure data is numeric
            ohlc_data = ohlc_data.apply(pd.to_numeric, errors="coerce")
            
            # Drop rows with all NaN values in OHLC columns
            if ohlc_data.isnull().all(axis=None):
                continue
    
            # Generate a dummy datetime index (starting at 2024-01-01)
            dummy_start_date = pd.Timestamp("2024-01-01")
            date_range = pd.date_range(start=dummy_start_date, periods=len(sorted_numbers), freq='H')  # Hourly intervals
            ohlc_data.index = date_range  
    
            # Plot using mplfinance
            fig, ax = plt.subplots(figsize=(10, 6))
            mpf.plot(ohlc_data, type="candle", style="charles", ax=ax)
            ax.set_title(f"Candlestick Chart for {pair} - {date_str}")
            plt.show()
        
    def _describe_features(self, key, chunk_key, data, progress_log, bin_edges, col_list, finish):
        """
        Calculate and store statistics for features in the current data chunk.
        Aggregate statistics for accurate reporting across chunks.
    
        Args:
            key (str): Dataset key.
            chunk_key (str): Current chunk identifier.
            data (DataFrame): Data chunk.
            progress_log (dict): Progress log for the dataset.
            bin_edges (dict): Pre-initialized bin edges for all features.
            col_list (list): List of features to process.
            finish (bool): Whether this is the last chunk.
        """
        chunk_stats = data.describe().T
    
        for feature in col_list:
            if feature not in progress_log["running_sum"]:
                # Initialize progress log for this feature
                progress_log["running_sum"][feature] = 0
                progress_log["running_square_sum"][feature] = 0
                progress_log["running_count"][feature] = 0
                progress_log["running_min"][feature] = float("inf")
                progress_log["running_max"][feature] = float("-inf")
                progress_log["bin_frequencies"][feature] = {i: 0 for i in range(len(bin_edges[feature]) - 1)}
                progress_log["bin_frequencies"][feature].update({"underflow": 0, "overflow": 0})
                progress_log["bin_edges"][feature] = bin_edges[feature]
    
            if feature in chunk_stats.index:
                # Update stats
                progress_log["running_sum"][feature] += chunk_stats.loc[feature, "mean"] * chunk_stats.loc[feature, "count"]
                progress_log["running_square_sum"][feature] += (
                    chunk_stats.loc[feature, "std"] ** 2 + chunk_stats.loc[feature, "mean"] ** 2
                ) * chunk_stats.loc[feature, "count"]
                progress_log["running_count"][feature] += chunk_stats.loc[feature, "count"]
                progress_log["running_min"][feature] = min(progress_log["running_min"][feature], chunk_stats.loc[feature, "min"])
                progress_log["running_max"][feature] = max(progress_log["running_max"][feature], chunk_stats.loc[feature, "max"])
    
                # Update bin frequencies
                self._update_bin_frequencies(
                    feature, data, bin_edges[feature], progress_log["bin_frequencies"]
                )
    
        if finish:
            self._finalize_describe_report(key, progress_log)
            self._display_descriptive_stats(progress_log)
            
    def _initialize_target_progress(self, key):
        """
        Create and retrieve target metric dictionary for persistent storage and updates.
        """
        if not hasattr(self, "_target_progress"):
            self._target_progress = {}  # Initialize a dictionary to store progress logs
    
        if key not in self._target_progress:
            # Initialize the progress log for the given key
            self._target_progress[key] = {
                "cat_counts": {},  # Stores categorical target counts
                "time_bins": {  # Stores time-based categorical bins
                    "hours_passed": {label: 0 for label in ["==1", ">1 & <=5", ">5 & <=24", ">24 & <=96", ">96"]},
                    "buy_sl_time": {label: 0 for label in ["==1", ">1 & <=5", ">5 & <=24", ">24 & <=96", ">96"]},
                    "sell_sl_time": {label: 0 for label in ["==1", ">1 & <=5", ">5 & <=24", ">24 & <=96", ">96"]},
                },
                "quant_stats": {  # Stores cumulative statistics for numerical targets
                    "sum": {},
                    "sum_sq": {},
                    "count": {},
                    "min": {},
                    "max": {}
                }
            }
    
        return self._target_progress[key]
    
    def _display_cat_target_stats(self, progress):
        """
        Display final stats for categorical targets as structured DataFrames.
    
        - Summarizes categorical target counts.
        - Summarizes time-based categorical target bins.
        - Prints them separately for clarity.
        """
        if "cat_counts" not in progress or "time_bins" not in progress:
            print("No categorical target data available.")
            return
    
        # Convert categorical target counts into a DataFrame
        target_counts_df = pd.DataFrame.from_dict(progress["cat_counts"]["target"], orient="index").T
        target_counts_df.index = ["count"]  # Rename row index
    
        # Convert counts to integers
        target_counts_df = target_counts_df.astype(int)
    
        # Compute percentage representation of each category
        total_count = target_counts_df.sum(axis=1).values[0]
        percent_row = ((target_counts_df / total_count) * 100).round(2)
        percent_row.index = ["percent (%)"]
    
        # Convert time-based categorical bins into a DataFrame
        time_bin_dfs = []
        for time_col, bins in progress["time_bins"].items():
            bin_df = pd.DataFrame.from_dict(bins, orient="index", columns=[time_col])
            time_bin_dfs.append(bin_df)
    
        # Concatenate all time bin DataFrames horizontally
        time_bin_df = pd.concat(time_bin_dfs, axis=1) if time_bin_dfs else None
    
        # Temporarily adjust Pandas display options for better readability
        with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
            print("\n **Categorical Target Counts**:")
            print(pd.concat([target_counts_df, percent_row]))  # Show counts and percentages
    
            if time_bin_df is not None:
                print("\n **Time-Based Target Bins**:")
                print(time_bin_df)  # Show time bin stats separately
                
    def _display_quant_target_stats(self, progress):
        """
        Display final stats for quantitative targets as a structured DataFrame.
    
        - Summarizes sum, sum of squares, count, min, and max for each quantitative target.
        """
        if "quant_stats" not in progress:
            print("No quantitative target data available.")
            return
    
        # Extract all statistical metrics from progress["quant_stats"]
        stats_dict = {
            "sum": progress["quant_stats"]["sum"],
            "sum_sq": progress["quant_stats"]["sum_sq"],
            "count": progress["quant_stats"]["count"],
            "min": progress["quant_stats"]["min"],
            "max": progress["quant_stats"]["max"]
        }
    
        # Convert dictionary to DataFrame (rows: features, columns: statistics)
        stats_df = pd.DataFrame(stats_dict)
    
        # Ensure count column is integer
        stats_df["count"] = stats_df["count"].astype(int)
    
        # Temporarily adjust Pandas display options for better readability
        with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
            print("\n** Quantitative Target Statistics **")
            print(stats_df)   
        
    def _describe_targets(self, key, chunk_key, data, col_list,
                           finish, target_type):
        """
        Describe target distributions based on type across multiple chunks.
        
        - For categorical targets, maintain cumulative category counts.
        - For time-based categorical targets (`hours_passed`, `buy_sl_time`, `sell_sl_time`),
          group them into bins and accumulate counts across chunks.
        - For quantitative targets, maintain running statistics for final aggregation.
    
        Args:
            key (str): Dataset key.
            chunk_key (str): Current chunk identifier.
            data (DataFrame): Data chunk.
            col_list (list): List of target columns.
            finish (bool): Whether this is the last chunk.
            target_type (str): Type of target ('cat' for categorical, 'quant' for quantitative).
        """
        progress = self._initialize_target_progress(key)  
    
        if target_type == 'cat':
            # Update category counts
            for col in col_list:
                if col not in progress["cat_counts"]:
                    progress["cat_counts"][col] = {}
    
                counts = data[col].value_counts().to_dict()
                for category, count in counts.items():
                    progress["cat_counts"][col][category] = progress["cat_counts"][col].get(category, 0) + count
    
            # Define bins for time-based categorical targets
            time_bins = [1, 5, 24, 96]
            time_labels = ["==1", ">1 & <=5", ">5 & <=24", ">24 & <=96", ">96"]
    
            for time_col in ["hours_passed", "buy_sl_time", "sell_sl_time"]:
                if time_col in data.columns:
                    if time_col not in progress["time_bins"]:
                        progress["time_bins"][time_col] = {label: 0 for label in time_labels}
    
                    # Bin the data and update counts
                    time_counts = pd.cut(
                        data[time_col], 
                        bins=[-float("inf")] + time_bins + [float("inf")], 
                        labels=time_labels
                    ).value_counts().to_dict()
    
                    for label, count in time_counts.items():
                        progress["time_bins"][time_col][label] += count  # No more KeyError
    
            if finish:
                self._display_cat_target_stats(progress)
            
        elif target_type == 'quant':
            for col in col_list:
                # Skip columns that are not numeric (int or float)
                if data[col].dtype.kind not in {'i', 'f'}:  # 'i' -> integer, 'f' -> float
                    print(f"Skipping non-numeric column: {col}")
                    continue  # Skip non-numeric columns
                    
                if col not in progress["quant_stats"]["sum"]:
                    progress["quant_stats"]["sum"][col] = 0
                    progress["quant_stats"]["sum_sq"][col] = 0
                    progress["quant_stats"]["count"][col] = 0
                    progress["quant_stats"]["min"][col] = float("inf")
                    progress["quant_stats"]["max"][col] = float("-inf")
    
                chunk = data[col].dropna()
                progress["quant_stats"]["sum"][col] += chunk.sum()
                progress["quant_stats"]["sum_sq"][col] += (chunk ** 2).sum()
                progress["quant_stats"]["count"][col] += chunk.count()
                progress["quant_stats"]["min"][col] = min(progress["quant_stats"]["min"][col], chunk.min())
                progress["quant_stats"]["max"][col] = max(progress["quant_stats"]["max"][col], chunk.max())
                
            if finish:
                self._display_quant_target_stats(progress)
    
        if target_type not in {'cat', 'quant'}:
            raise ValueError(f"Invalid target_type: {target_type}. Use 'cat' or 'quant'")
        
    def _plot_features(self, key, chunk_key, data, progress_log, bin_edges, 
                       col_list, finish, describe_features):
        """
        Calculate and store statistics for features in the current data chunk.
        Aggregate statistics for accurate plotting across chunks.
    
        Args:
            key (str): Dataset key.
            chunk_key (str): Current chunk identifier.
            data (DataFrame): Data chunk.
            progress_log (dict): Progress log for the dataset.
            bin_edges (dict): Pre-initialized bin edges for all features.
            col_list (list): List of features to process.
            finish (bool): Whether this is the last chunk.
            describe_features (bool): Is progress_log being created elsewhere.
        """
        # If describe_features is True, it will handle progress_log creation
        if describe_features:
            if finish:
                self._plot_descriptive_stats(progress_log)
        # If describe_features is False, create progress_log
        else:
            chunk_stats = data.describe().T
        
            for feature in col_list:
                if feature not in progress_log["running_sum"]:
                    # Initialize progress log for this feature
                    progress_log["running_sum"][feature] = 0
                    progress_log["running_square_sum"][feature] = 0
                    progress_log["running_count"][feature] = 0
                    progress_log["running_min"][feature] = float("inf")
                    progress_log["running_max"][feature] = float("-inf")
                    progress_log["bin_frequencies"][feature] = {i: 0 for i in range(len(bin_edges[feature]) - 1)}
                    progress_log["bin_frequencies"][feature].update({"underflow": 0, "overflow": 0})
                    progress_log["bin_edges"][feature] = bin_edges[feature]
        
                if feature in chunk_stats.index:
                    # Update stats
                    progress_log["running_sum"][feature] += chunk_stats.loc[feature, "mean"] * chunk_stats.loc[feature, "count"]
                    progress_log["running_square_sum"][feature] += (
                        chunk_stats.loc[feature, "std"] ** 2 + chunk_stats.loc[feature, "mean"] ** 2
                    ) * chunk_stats.loc[feature, "count"]
                    progress_log["running_count"][feature] += chunk_stats.loc[feature, "count"]
                    progress_log["running_min"][feature] = min(progress_log["running_min"][feature], chunk_stats.loc[feature, "min"])
                    progress_log["running_max"][feature] = max(progress_log["running_max"][feature], chunk_stats.loc[feature, "max"])
        
                    # Update bin frequencies
                    self._update_bin_frequencies(
                        feature, data, bin_edges[feature], progress_log["bin_frequencies"]
                    )
        
            if finish:
                self._finalize_describe_report(key, progress_log)
                self._plot_descriptive_stats(progress_log)
 
    def _collect_cols(self, key):
        """
        Collect the feature (column) names from the source query for a given key.
    
        Args:
            key (str): The key identifying the dataset.
    
        Returns:
            list: A list of column names to process, excluding the primary key.
        """
        # Extract the query string for the given key
        query = self.source_query[key][1]
    
        # Parse the query to collect column headers
        # Extract column names between `SELECT` and `FROM`
        select_section = query.split("FROM")[0].split("SELECT")[-1].strip()
        columns = [col.strip() for col in select_section.split(",")]
    
        # Check for wildcard (*), fetch all columns if present
        if "*" in columns:
            # Establish a connection to the database
            database_name = self.source_query[key][0]
            db_path = self.env_loader.get(database_name)
            table_name = query.split("FROM")[1].split("WHERE")[0].strip()
    
            try:
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.cursor()
                    query_for_columns = f"PRAGMA table_info({table_name})"
                    column_info = cursor.execute(query_for_columns).fetchall()
                    columns = [col[1] for col in column_info]  # Assuming column names are in the second position
            except Exception as e:
                raise RuntimeError(f"Failed to fetch column information: {e}")
    
        # Remove primary key columns
        dummy_df = pd.DataFrame(columns=columns)  # Create a dummy DataFrame with the column names
        filtered_df = self._remove_primary_key(dummy_df)
        col_list = filtered_df.columns.tolist()
        return col_list
        
    def _remove_primary_key(self, data):
        """
        Remove the primary key from the dataframe.
        """
        if not hasattr(self, "primary_key") or not self.primary_key:
            raise AttributeError("Primary key is not defined or empty.")
    
        # Ensure primary_key columns exist in the dataframe before dropping
        missing_keys = [key for key in self.primary_key if key not in data.columns]
        if missing_keys:
            raise ValueError(f"Primary key columns not found in dataframe: {missing_keys}")
    
        # Drop the primary key columns
        return data.drop(columns=self.primary_key)
    
    def _convert_to_numeric(self, data):
        """
        Convert column data from text to numeric.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
            
        # Columns to exclude from conversion
        exclude_cols = {'target', 'date', 'pair'}
    
        # Attempt to convert all columns to numeric
        for col in data.columns:
            try:
                if col in exclude_cols:
                    continue
                data[col] = pd.to_numeric(data[col], errors="coerce")
            except Exception as e:
                print(f"Could not convert column '{col}' to numeric: {e}")
    
        return data
        
    @public_method
    def inspect_data(self,
                   data_keys,
                   describe_features=False,
                   describe_targets=False,
                   target_type='cat',
                   plot_features=False,
                   plot_mode='rate',
                   plot_skip=48):
        """
        Loop through the data keys and apply the selected functions to each
        data set in chunks. Supports iterative evaluation and cleaning.
        
        ONLY SET ONE OPTION TO TRUE FOR EVERY RUN. VERIFY THE DATA KEY MATCHES
        THE DESIRED FUNCTION.
        
        target_type options:
            - 'cat' for categorical
            - 'quant' for numerical/quantitative
        
        plot_mode options:
            - 'rate' will plot standardized OHLC data
            - 'stat' will plot histogram of feature distributions
        """
        # Separate describe/plot phase from cleaning/aligning
        if any([describe_features, describe_targets, plot_features]):
            source = 'source_query'
            for key in data_keys:
                print(f"Beginning descriptive functions for {key}...")
                chunk_keys = self.ltd.chunk_keys(
                    mode='config', 
                    source=source, 
                    key=key)  # Determine chunk splits
    
                for idx, chunk_key in enumerate(chunk_keys):
                    print(f"Processing chunk {chunk_key[0][0]} - {chunk_key[1][0]}...")
                    # Set finish to True for the last chunk
                    finish = idx == len(chunk_keys) - 1
                    # Initialize bins only if needed
                    if idx == 0:
                        if not (plot_features and plot_mode == 'rate' and not any(
                                [describe_features, describe_targets])):
                            col_list = self._collect_cols(key)
                            if not describe_targets:
                                bin_edges = self._initialize_bins_from_sql(col_list, key)
                    
                    data = self.ltd.load_chunk(
                        mode='config',
                        source=source,
                        key=key, 
                        chunk_key=chunk_key)
                    
                    if not (plot_features and plot_mode == 'rate' and not any(
                            [describe_features, describe_targets])):
                        data = self._remove_primary_key(data)
                        data = self._convert_to_numeric(data)
                        progress_log = self._get_progress_log(key)
                    # Perform operations on the current chunk
                    if describe_features:
                        self._describe_features(
                            key, chunk_key, data, progress_log, bin_edges, 
                            col_list, finish)
                    if describe_targets:
                        self._describe_targets(key, chunk_key, data, col_list,
                                               finish, target_type)
                    if plot_features:
                        if plot_mode == 'rate':
                            self._plot_rates(data, plot_skip)
                        elif plot_mode == 'stat':
                            self._plot_features(key, chunk_key, data, progress_log, 
                                                bin_edges, col_list, finish, 
                                                describe_features)
                    
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
        clean_functions = getattr(module, function_name)
        
        return clean_functions
    
    def _save_bad_keys(self, key, bad_keys):
        """
        Save the bad_keys list to the self.bad_keys_path.
        
        Args:
            key (str): Name of clean_functions metadata.
            bad_keys (list): List of bad primary keys.
        """
        save_dir = self.bad_keys_path
        os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists
    
        save_path = os.path.join(save_dir, f"{key}_bad_keys.txt")
        
        if bad_keys:  # Only save if there are bad keys
            with open(save_path, "w") as output:
                output.write("\n".join(map(str, bad_keys)))  # Write keys line by line
        else:
            print(f"No bad keys found for {key}. Skipping save.")
        
    @public_method
    def clean(self, data_keys=[]):
        """
        Apply the iteration's filter functions on their corresponding database
        tables. Store the bad data's primary keys in a file for downstream
        processing.
        
        Args:
            data_keys (list): Default is empty and indicates all keys in the 
                config['clean_functions'] dict should be processed. Specify
                the keys in a list to only process select tables/data.
        """
        def clean_steps(data_keys):
            for key in data_keys:
                print(f"Processing clean functions for: {key}")
                # Determine chunk splits
                bad_keys_list = []
                source = 'clean_functions'
                chunk_keys = self.ltd.chunk_keys(
                    mode='config',
                    source=source, 
                    key=key)  
                # Import the filter function
                function_name = self.clean_functions[key][2]
                filter_function = self._import_function(function_name)
                for idx, chunk_key in enumerate(chunk_keys):
                    print(f"Processing chunk {chunk_key[0][0]} - {chunk_key[1][0]}...")
                    data = self.ltd.load_chunk(
                        mode='config',
                        source=source, 
                        key=key, 
                        chunk_key=chunk_key)
                    data = self._convert_to_numeric(data)
                    bad_keys = filter_function(data)
                    bad_keys_list.extend(bad_keys)
                self._save_bad_keys(key, bad_keys_list)
        
        # Process all keys in config's clean_functions
        if not data_keys:
            data_keys = list(self.clean_functions.keys())
            
        clean_steps(data_keys)
        
    def _create_bad_key_set(self, bad_keys_path):
        """
        Loop through all bad_keys.txt files in the given directory, extract bad keys, 
        and store them in a set.
    
        Args:
            bad_keys_path (str): Path to the directory containing bad_keys.txt files.
    
        Returns:
            set: A set of tuples containing bad primary keys (pair, date).
        """
        full_set = set()
    
        # Iterate through all text files in the directory
        for file in os.listdir(bad_keys_path):
            if file.endswith("_bad_keys.txt"):  # Ensure only relevant files are processed
                full_path = os.path.join(bad_keys_path, file)
                
                # Read and extract keys
                with open(full_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:  # Skip empty lines
                            try:
                                # Parse the line as a tuple
                                pair, date = eval(line)  # Convert string tuple to actual tuple
                                full_set.add((pair, date))  # Add to the set
                            except Exception as e:
                                print(f"Error parsing line in {file}: {line} | {e}")
    
        return full_set
    
    def _remove_bad_keys(self, data, bad_key_set):
        """
        Filter out data with matching primary keys and return the cleaned 
        data.
        """
        # Ensure the required primary key columns exist in the DataFrame
        primary_keys = self.primary_key
        for key in primary_keys:
            if key not in data.columns:
                raise KeyError(f"Primary key '{key}' not found in data columns: {data.columns}")
    
        # Filter out rows that have a primary key match in bad_key_set
        mask = data.apply(lambda row: (row["pair"], row["date"]) not in bad_key_set, axis=1)
        cleaned_data = data[mask].reset_index(drop=True)
        
        return cleaned_data
    
    def _save_clean_data(self, clean_data, db_path, table):
        """
        Save the cleaned data to the clean database and its corresponding table.
    
        - If `reset_db` is True, it **deletes the old clean database** before inserting new data.
        - Uses `if_exists="replace"` for first write and `if_exists="append"` for additional tables.
        
        Args:
            clean_data (pd.DataFrame): The cleaned DataFrame to be saved.
            db_path (str): Path to the clean database.
            table (str): Name of the table in which data should be stored.
            reset_db (bool): If True, removes the old clean database before writing fresh data.
        """
        if clean_data.empty:
            print(f"No clean data to save for table: {table}")
            return
    
        # Ensure the database filename includes "_clean"
        db_dir, db_filename = os.path.split(db_path)
        if "_clean" not in db_filename:
            db_filename = db_filename.replace(".db", "_clean.db")
            db_path = os.path.join(db_dir, db_filename)
    
        # Convert all columns to string format for storage
        clean_data = clean_data.astype(str)
    
        # Define columns and enforce primary key
        column_definitions = ", ".join(f"{col} TEXT" for col in clean_data.columns)
        primary_key_str = ", ".join(self.primary_key)
    
        create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {table} (
                {column_definitions},
                PRIMARY KEY ({primary_key_str})
            )
        """
        columns = ", ".join(clean_data.columns)
        placeholders = ", ".join(["?"] * len(clean_data.columns))
    
        insert_query = f"""
            INSERT OR REPLACE INTO {table} ({columns})
            VALUES ({placeholders})
        """
    
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
    
            # Create the table if it does not exist
            cursor.execute(create_table_query)
    
            # Execute batch insert
            cursor.executemany(insert_query, clean_data.to_records(index=False))
    
            conn.commit()
    
        print(f"Data successfully saved to {table} in {db_path}")
        
    def _clean_and_save(self, feature_path, target_path, bad_key_set):
        """
        Load data from raw or clean database, remove data with bad keys, and
        save the clean data to the clean database.
        """
        def clean_and_save_process(db_paths, bad_key_set):
            for db_path in db_paths:
                
                # Ensure the database filename includes "_clean"
                db_dir, db_filename = os.path.split(db_path)
                if "_clean" not in db_filename:
                    db_filename = db_filename.replace(".db", "_clean.db")
                
                delete_path = os.path.join(db_dir, db_filename)
                
                # Reset the clean database only once before first insert
                if os.path.exists(delete_path):
                    os.remove(delete_path)
                    print(f"Deleted old clean database: {delete_path}")
                
                conn = sqlite3.connect(db_path)
                try:
                    with sqlite3.connect(db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                        tables = [row[0] for row in cursor.fetchall()] 
                    
                    for table in tables:
                        query = f"""
                        SELECT * FROM {table}
                        """
                        chunk_keys = self.ltd.chunk_keys(
                            mode='manual',
                            db_path=db_path,
                            query=query
                            )
                        for idx, chunk_key in enumerate(chunk_keys):
                            print(f"Processing chunk {chunk_key[0][0]} - {chunk_key[1][0]}...")
                            data = self.ltd.load_chunk(
                                mode='manual',
                                db_path=db_path,
                                query=query,
                                chunk_key=chunk_key
                                )
                            clean_data = self._remove_bad_keys(data, bad_key_set)
                            self._save_clean_data(clean_data, db_path, table)
                finally:
                    conn.close()
                    
        # Check if feature and target paths are the same
        if feature_path == target_path:
            db_paths = [feature_path]
        else:
            db_paths = [feature_path, target_path]
            
        clean_and_save_process(db_paths, bad_key_set)
        
    @public_method
    def align(self):
        """
        Loop through the iteration's database tables, remove observations based 
        on the stored bad primary key file, and save or update the clean data 
        tables.
        """        
        # Get paths for clean and raw databases
        raw_feature_path = self.env_loader.get("FEATURE_DATABASE")
        raw_target_path = self.env_loader.get("TARGET_DATABASE")
        
        # Create a set of all bad primary keys
        bad_key_set = self._create_bad_key_set(self.bad_keys_path)
        
        # Loop through the raw or clean database tables, remove bad keys,
        # and save or overwrite clean database
        self._clean_and_save(raw_feature_path, raw_target_path, bad_key_set)
        