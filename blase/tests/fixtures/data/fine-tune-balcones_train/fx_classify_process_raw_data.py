# fx_classify_process_raw_data.py


import pandas as pd
import numpy as np


def filter_hourly(df):
    """
    Filters rows that contain any NaN values (in any column) are also flagged.

    The function returns a list of tuples containing the ('pair', 'date') values
    for the rows that meet filter conditions.
    """
    if 'pair' not in df.columns or 'date' not in df.columns:
        raise ValueError("DataFrame must contain 'pair' and 'date' columns.")
    
    # Find rows with NaN values in any column.
    nan_rows = df[df.isna().any(axis=1)]

    # Extract (pair, date) tuples.
    return list(zip(nan_rows['pair'], nan_rows['date']))


def filter_targets(df):
    """
    Find rows in df['target'] with values == "wait".
    
    Returns:
        List of tuples containing ('pair', 'date') for matching rows.
    """
    if 'pair' not in df.columns or 'date' not in df.columns or 'target' not in df.columns:
        raise ValueError("DataFrame must contain 'pair', 'date', and 'target' columns.")

    # Apply filtering condition
    filtered_rows = df[df['target'] == "wait"]
    # Return as a list of tuples (pair, date)
    return list(zip(filtered_rows['pair'], filtered_rows['date']))


def feature_engineering(df):
    """
    Perform feature engineering by:
        - Adding sine and cosine transformations for the hour of the day.
        - One-hot encoding the 'pair' column while retaining it.

    Args:
        df (pd.DataFrame): DataFrame with 'date' and 'pair' columns.

    Returns:
        pd.DataFrame: DataFrame with new engineered features.
    """
    
    # Ensure 'date' is in datetime format
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Handle invalid date conversions
    if df['date'].isna().any():
        raise ValueError("Some dates could not be converted to datetime. Check 'date' column.")

    # Hour-of-day cyclic encoding
    df['hour_sin'] = np.sin(2 * np.pi * (df['date'].dt.hour / 24))
    df['hour_cos'] = np.cos(2 * np.pi * (df['date'].dt.hour / 24))

    # One-hot encode 'pair' column while keeping the original
    df_encoded = pd.get_dummies(df, columns=['pair'], prefix='pair', dtype=int)
    
    # Re-insert the original 'pair' column at its original position
    df_encoded.insert(df.columns.get_loc('pair'), 'pair', df['pair'])

    return df_encoded


def target_engineering(df):
    """
    Encodes target column into string categories.
    
    Args:
        df (pd.DataFrame): DataFrame with a 'date' column.

    Returns:
        pd.DataFrame: DataFrame with new targets.
    """
    category_to_index = {'loss': 0, 'buy': 1, 'sell': 2, 'wait': 0}    
    df['target'] = df['target'].map(category_to_index)
    
    df['date'] = pd.to_datetime(df['date'])
    df['hour'] = df['date'].dt.hour
    eod_condition = df['hour'].isin([23, 0, 1])
    df['hours_passed'] = df['hours_passed'].fillna(0).astype(float)
    condition_hours_passed = df['hours_passed'] > 12
    
    df['target'] = np.where(
        eod_condition | condition_hours_passed,
        0,
        df['target']
        )
    
    df.drop(['hour'], axis=1, inplace=True)
    
    df['target'] = df['target'].fillna(0).astype(int)
    df['target'] = df['target'].astype(int)
    
    return df


def df_features(df):
    """
    Example function that returns the df's list of training features.
    
    Args:
        df (pd.DataFrame): DataFrame with a 'date' column.
        
    Returns:
        list: List of feature columns.
    """
    return [col for col in df.columns if col not in [
        'date', 'pair', 'target', 'hours_passed', 'buy_sl_time', 'sell_sl_time']]
