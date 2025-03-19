# fx_classify_feature_engineering.py

'''
Create features test/template for development:
    
    Accept the pd.DataFrame from the iteration's base_data table for 
    feature engineering.
    
    Map the features to their storage location.
    
    Calculate each feature and add each to a features function to
    package for easy import into CreateFeatures.
'''

import numpy as np
import pandas as pd


def standardize_open_hourly_rates(df, shift=14):
    """
    Standardizes the Open values over the past 'shift' periods.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the OHLC data.
        shift (int): Defines the shift from the current hour.
    
    Returns:
        pd.DataFrame: Updated DataFrame with standardized OHLC columns.
    """
    # Ensure that the dependent columns are numeric and contain no NaNs
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    
    # Create all shifted columns at once
    shifted_data = {
        f'open_standard_{hour + 1}': df['open'].shift(hour)
        for hour in range(shift)
    }
    shifted_data['baseline_close'] = df['close'].shift(shift)
    
    # Add shifted columns to DataFrame in one operation to avoid fragmentation
    shifted_df = pd.DataFrame(shifted_data)
    
    # Calculate standardized values in one step
    for hour in range(shift):
        column_name = f'open_standard_{hour + 1}'
        shifted_df[column_name] = (
            ((shifted_df[column_name] - shifted_df['baseline_close']) /
             shifted_df['baseline_close']).round(5) + 1
        )
    
    # Merge back into the original DataFrame
    return pd.concat([df, shifted_df.drop(columns=['baseline_close'])], axis=1)


def standardize_high_hourly_rates(df, shift=14):
    """
    Standardizes the High values over the past 14 periods.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the OHLC data.
        start_index (int): Index from where to start the calculation.
        shift (int): Defines the shift from the current hour.
    
    Returns:
        pd.DataFrame: Updated DataFrame with standardized OHLC columns.
    """
    # Ensure that the dependent columns are numeric and contain no NaNs
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    
    # Create all shifted columns at once
    shifted_data = {
        f'high_standard_{hour + 1}': df['high'].shift(hour)
        for hour in range(shift)
    }
    shifted_data['baseline_close'] = df['close'].shift(shift)
    
    # Add shifted columns to DataFrame in one operation to avoid fragmentation
    shifted_df = pd.DataFrame(shifted_data)
    
    # Calculate standardized values in one step
    for hour in range(shift):
        column_name = f'high_standard_{hour + 1}'
        shifted_df[column_name] = (
            ((shifted_df[column_name] - shifted_df['baseline_close']) /
             shifted_df['baseline_close']).round(5) + 1
        )
    
    # Merge back into the original DataFrame
    return pd.concat([df, shifted_df.drop(columns=['baseline_close'])], axis=1)


def standardize_low_hourly_rates(df, shift=14):
    """
    Standardizes the Low values over the past 14 periods.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the OHLC data.
        start_index (int): Index from where to start the calculation.
        shift (int): Defines the shift from the current hour.
    
    Returns:
        pd.DataFrame: Updated DataFrame with standardized OHLC columns.
    """
    # Ensure that the dependent columns are numeric and contain no NaNs
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    
    # Create all shifted columns at once
    shifted_data = {
        f'low_standard_{hour + 1}': df['low'].shift(hour)
        for hour in range(shift)
    }
    shifted_data['baseline_close'] = df['close'].shift(shift)
    
    # Add shifted columns to DataFrame in one operation to avoid fragmentation
    shifted_df = pd.DataFrame(shifted_data)
    
    # Calculate standardized values in one step
    for hour in range(shift):
        column_name = f'low_standard_{hour + 1}'
        shifted_df[column_name] = (
            ((shifted_df[column_name] - shifted_df['baseline_close']) /
             shifted_df['baseline_close']).round(5) + 1
        )
    
    # Merge back into the original DataFrame
    return pd.concat([df, shifted_df.drop(columns=['baseline_close'])], axis=1)


def standardize_close_hourly_rates(df, shift=14):
    """
    Standardizes the Close values over the past 14 periods.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the OHLC data.
        start_index (int): Index from where to start the calculation.
        shift (int): Defines the shift from the current hour.
    
    Returns:
        pd.DataFrame: Updated DataFrame with standardized OHLC columns.
    """    
    # Ensure that the dependent columns are numeric and contain no NaNs
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    
    # Create all shifted columns at once
    shifted_data = {
        f'close_standard_{hour + 1}': df['close'].shift(hour)
        for hour in range(shift)
    }
    shifted_data['baseline_close'] = df['close'].shift(shift)
    
    # Add shifted columns to DataFrame in one operation to avoid fragmentation
    shifted_df = pd.DataFrame(shifted_data)
    
    # Calculate standardized values in one step
    for hour in range(shift):
        column_name = f'close_standard_{hour + 1}'
        shifted_df[column_name] = (
            ((shifted_df[column_name] - shifted_df['baseline_close']) /
             shifted_df['baseline_close']).round(5) + 1
        )
    
    # Merge back into the original DataFrame
    return pd.concat([df, shifted_df.drop(columns=['baseline_close'])], axis=1)


def high_vol(df):
    df['hourly_range'] = (df['high'] - df['low']) / df['open']
    df['avg_range'] = df['hourly_range'].rolling(window=4).mean()
    
    df['high_vol'] = np.where(
        df['avg_range'] > 0.002, 1, 0)
    return df['high_vol']


def sma_roc(df):
    df['sma'] = df['close'].rolling(window=4).mean()
    df['sma_roc'] = (df['sma'] - df['sma'].shift(1)) / df['close'].shift(1)
    return df['sma_roc']


def close_to_std_dev(df):
    df['sma'] = df['close'].rolling(window=4).mean()
    df['std_dev'] = df['close'].rolling(window=4).std()
    df['close_to_std_dev'] = (df['sma'] - df['close']) / df['std_dev']
    return df['close_to_std_dev']


#########################################################
# Required functions for the data_preparation.py module
#########################################################


def features(df):
    shift = 12
    processed_dfs = []
    for pair in ["AUDUSD", "CHFJPY", "EURUSD", "USDJPY", "USDMXN"]:
        df_copy = df[df["pair"] == pair].copy()  # Isolate pair subset
        
        df_copy = standardize_open_hourly_rates(df_copy, shift=shift)
        df_copy = standardize_high_hourly_rates(df_copy, shift=shift)
        df_copy = standardize_low_hourly_rates(df_copy, shift=shift)
        df_copy = standardize_close_hourly_rates(df_copy, shift=shift)
        
        df_copy['high_vol'] = high_vol(df_copy)
        df_copy['sma_roc'] = sma_roc(df_copy)
        df_copy['close_to_std_dev'] = close_to_std_dev(df_copy)
        
        processed_dfs.append(df_copy) 
        
    df_final = pd.concat(processed_dfs, ignore_index=False, sort=False)
    return df_final


storage_map = {
    'test_hourly_feature_data': [                        
        'open_standard_12', 'high_standard_12', 
        'low_standard_12', 'close_standard_12', 
        'open_standard_11', 'high_standard_11', 
        'low_standard_11', 'close_standard_11', 
        'open_standard_10', 'high_standard_10', 
        'low_standard_10', 'close_standard_10', 
        'open_standard_9', 'high_standard_9', 
        'low_standard_9', 'close_standard_9', 
        'open_standard_8', 'high_standard_8', 
        'low_standard_8', 'close_standard_8', 
        'open_standard_7', 'high_standard_7', 
        'low_standard_7', 'close_standard_7', 
        'open_standard_6', 'high_standard_6', 
        'low_standard_6', 'close_standard_6', 
        'open_standard_5', 'high_standard_5', 
        'low_standard_5', 'close_standard_5', 
        'open_standard_4', 'high_standard_4', 
        'low_standard_4', 'close_standard_4', 
        'open_standard_3', 'high_standard_3', 
        'low_standard_3', 'close_standard_3', 
        'open_standard_2', 'high_standard_2', 
        'low_standard_2', 'close_standard_2', 
        'open_standard_1', 'high_standard_1', 
        'low_standard_1', 'close_standard_1'],
    
    'test_technical_feature_data': [
        'high_vol', 'sma_roc', 'close_to_std_dev']
    }