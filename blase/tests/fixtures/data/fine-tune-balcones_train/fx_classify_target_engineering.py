# fx_classify_target_engineering.py


'''
Create targets test/template for development:
    
    Accept the pd.DataFrame from the iteration's base_data table for 
    target engineering.
    
    Map the targets to their storage location.
    
    Calculate each target and add each to a targets function to
    package for easy import into CreateTargets.
'''


import re
from datetime import time
from tqdm import tqdm

import numpy as np
import pandas as pd


multipliers = {
    "AUD/USD": 0.000280151,
    "CHF/JPY": 0.000205996,
    "EUR/USD": 0.000110819,
    "USD/JPY": 9.60859E-05,
    "USD/MXN": 0.000283136,
    }


widemultipliers = {
    "AUD/USD": 0.000650051,
    "CHF/JPY": 0.001063064,
    "EUR/USD": 0.000368475,
    "USD/JPY": 0.000455197,
    "USD/MXN": 0.002011915, 
    }


def normalize_pair_name(pair_name):
    """
    Extracts and normalizes the currency pair name from a pair name.
    Normalizes formats like 'AUDUSD', 'aud/usd', 'AUD/usd' to 'AUD/USD'.
    """
    # Extract the base name (without path and extension)
    base_name = re.sub(r'[^a-zA-Z]', '', pair_name).upper()  # Remove non-alphabetic chars and convert to upper case
    
    # Assuming currency pairs are 6 characters long (e.g., 'AUDUSD')
    if len(base_name) == 6:
        return base_name[:3] + '/' + base_name[3:]
    else:
        raise ValueError("Currency pair name format not recognized.")


def get_multiplier(date, normalized_pair_name, multipliers, widemultipliers):
    """
    Determines the appropriate multiplier to use based on the timestamp.
    """
    date = pd.to_datetime(date)
    is_wide_multiplier = False
    if date.time() == time(0, 0):
        factor = widemultipliers.get(normalized_pair_name, 0)
        is_wide_multiplier = True
    else:
        factor = multipliers.get(normalized_pair_name, 0)
    
    return factor, is_wide_multiplier


def add_open_prices(df, pair_name, multipliers, widemultipliers):
    """
    Calculate trade open price using the adjusted hour's open rate.
    """
    
    # Normalize the pair name based on the file name
    normalized_pair_name = normalize_pair_name(pair_name)
        
    def calculate_open_price_bid(row):
        open_price_bid = row['open']
        return open_price_bid

    def calculate_open_price_ask(row):
        date = pd.to_datetime(row['date']).tz_localize(None)
        factor, _ = get_multiplier(date, normalized_pair_name, multipliers, widemultipliers)
        adjustment_factor = 1 + factor
        open_price_bid = row['open_price_bid']
        open_price_ask = open_price_bid * adjustment_factor
        return open_price_ask

    df['open_price_bid'] = df.apply(calculate_open_price_bid, axis=1)
    df['open_price_ask'] = df.apply(calculate_open_price_ask, axis=1)

    return df


def calc_sl_tp(df, pair):
    """
    Calculate the stop loss and take profit based on recent volatility.
    
    Parameters:
    df (DataFrame): A pandas DataFrame with at least 'high', 'low', 
                    'open_price_ask', and 'open_price_bid' columns.
    pair (str): The currency pair to normalize and retrieve spread multipliers for.
    
    Returns:
    DataFrame: The DataFrame with new columns 'true_range', 'avg_true_range',
               'buy_sl', 'buy_tp', 'sell_sl', and 'sell_tp'.
    """
    # Calculate true range
    df['true_range'] = df['high'] - df['low']
    
    # Find the normal spread amount
    spread = multipliers[pair]
    
    # Calculate the rolling 14-period average of the true range
    df['avg_true_range'] = df['true_range'].shift(1).rolling(window=4).mean() * (1 + spread)
    df['avg_true_range'] = df['avg_true_range'].fillna(0)
     
    # Calculate buy sl and tp
    df['buy_sl'] = df['open_price_ask'] - np.maximum(
        df['avg_true_range'], 
        df['open_price_ask'] * 0.00125)
    df['buy_tp'] = df['open_price_ask'] + 2 * np.maximum(
        df['avg_true_range'], 
        df['open_price_ask'] * 0.00125)
    
    # Calculate sell sl and tp
    df['sell_sl'] = df['open_price_bid'] + np.maximum(
        df['avg_true_range'], 
        df['open_price_ask'] * 0.00125)
    df['sell_tp'] = df['open_price_bid'] - 2 * np.maximum(
        df['avg_true_range'], 
        df['open_price_ask'] * 0.00125)
    
    return df


def true_outcome(index, df, multipliers, widemultipliers, n):
    outcome = 'wait'  # Default target
    hours_passed = 0  # Initialize hours passed
    
    # Check if out of bounds
    if index + 1 >= len(df):
        return outcome, hours_passed, None, None
    
    row = df.iloc[index]
    start_row = df.iloc[index+1]
    pair_name = normalize_pair_name(row['pair'])
    date = pd.to_datetime(row['date'])
    
    # Get the multiplier for the pair
    multiplier, is_wide_multiplier = get_multiplier(
        date, pair_name, multipliers, widemultipliers)
    
    # Calculate the take profit and stop loss for 'Buy'
    buy_take_profit = start_row['buy_tp']
    buy_stop_loss = start_row['buy_sl']
    
    # Calculate the take profit and stop loss for 'Sell'
    sell_take_profit = start_row['sell_tp']
    sell_stop_loss = start_row['sell_sl']
    
    # Flags to track if stop loss for buy and sell are hit
    buy_stop_loss_hit = False
    sell_stop_loss_hit = False
    
    buy_sl_time = None
    sell_sl_time = None
    
    # Loop through the next set of hours
    for i in range(index, min(index + n, len(df) - 1)):
        next_row = df.iloc[i+1]
        current_timestamp = pd.to_datetime(next_row['date'])
        
        # Determine the appropriate multiplier based on the is_wide_multiplier flag
        effective_multiplier = 1 + multiplier if not is_wide_multiplier else 1 + is_wide_multiplier
        
        # Check for 'Buy' and 'Sell' stop loss conditions
        if next_row['low'] < buy_stop_loss and not buy_sl_time:
            buy_stop_loss_hit = True
            buy_sl_time = (current_timestamp - date).total_seconds() / 3600
        if next_row['high'] * effective_multiplier > sell_stop_loss and not sell_sl_time:
            sell_stop_loss_hit = True
            sell_sl_time = (current_timestamp - date).total_seconds() / 3600

        # If both stop losses are hit, assign 'Loss' and exit loop
        if buy_stop_loss_hit and sell_stop_loss_hit:
            outcome = 'loss'
            hours_passed = (current_timestamp - date).total_seconds() / 3600
            break

        # Check for 'Buy' condition
        if not buy_stop_loss_hit and next_row['high'] >= buy_take_profit:
            outcome = 'buy'
            hours_passed = (current_timestamp - date).total_seconds() / 3600
            break  # Take profit was hit, stop checking further
        
        # Check for 'Sell' condition
        if not sell_stop_loss_hit and next_row['low'] * effective_multiplier <= sell_take_profit:
            outcome = 'sell'
            hours_passed = (current_timestamp - date).total_seconds() / 3600
            break  # Take profit was hit, stop checking further
    
    return outcome, hours_passed, buy_sl_time, sell_sl_time


def update_targets(df, multipliers, widemultipliers, n=336):
    pair_name = normalize_pair_name(df['pair'].iloc[0])
    df = add_open_prices(df, pair_name, multipliers, widemultipliers)
    df = calc_sl_tp(df, pair_name)
    
    # Initialize an empty list to hold the targets
    outcomes = []
    hours_passed = []
    buy_sl = []
    sell_sl = []
    
    # Iterate over each row index in the DataFrame
    for index in tqdm(range(len(df)), desc="Determining targets"):
        # Call true_outcome for each row and append the result to the targets list
        outcome, hours, buy_sl_time, sell_sl_time = true_outcome(
            index, df, multipliers, widemultipliers, n)
        outcomes.append(outcome)
        hours_passed.append(hours)
        buy_sl.append(buy_sl_time)
        sell_sl.append(sell_sl_time)
    
    # Assign the list of targets back to the DataFrame's 'Target' column
    df['target'] = outcomes
    df['hours_passed'] = hours_passed
    df['buy_sl_time'] = buy_sl
    df['sell_sl_time'] = sell_sl
    
    output_df = df[['date', 'pair', 'target', 'hours_passed', 'buy_sl_time',
                    'sell_sl_time']]
    return output_df


#########################################################
# Required functions for the data_preparation.py module
#########################################################


def targets(df):
    """
    Call all feature functions and add them to the df.

    Args:
        df (pd.DataFrame): DataFrame containing exchange rates and other features.

    Returns:
        pd.DataFrame: Updated DataFrame with new features added for each pair.
    """
    output_df = update_targets(df, multipliers, widemultipliers, n=336)
    return output_df


storage_map = {'targets': ['target', 'hours_passed', 'buy_sl_time',
                           'sell_sl_time']}