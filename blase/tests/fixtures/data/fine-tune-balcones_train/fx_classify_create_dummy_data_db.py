# fx_classify_create_dummy_data_db.py 


import sqlite3
from datetime import datetime, timedelta
import os

import pandas as pd
import numpy as np


"""
Create the raw data database to initiate the test workflow. This step is a 
prerequisite to feature and target engineering. This iteration simply creates
random OHLC data for five pairs which will be used for creating technical
indicator and timeseries features and calculating targets.
"""


def create_dummy_table(db_path):
    query = """
    CREATE TABLE IF NOT EXISTS test_hourly_exchange_rate (
        date    TEXT,
        pair    TEXT,
        open    REAL,
        high    REAL,
        low    REAL,
        close    REAL,
        PRIMARY KEY (date, pair)
        );
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute(query)
        conn.commit()
        print("Table created successfully or already exists.")
        
    except sqlite3.Error as e:
        print(f"An error occurred: {e}.")
    
    finally:
        conn.close()
    

def populate_dummy_table(db_path, n_samples):
    # Initialize data dict to pass to pd.DataFrame(data)
    data = {
        "date": [],
        "pair": [],
        "open": [],
        "high": [],
        "low": [],
        "close": []
        }
    
    # Specify pairs and initial price
    pair_dict = {
        "AUDUSD": 0.63,
        "CHFJPY": 166,
        "EURUSD": 1.03,
        "USDJPY": 150,
        "USDMXN": 18
        }
    
    start_date = datetime(2025, 1, 1)
    
    for i, pair in enumerate(pair_dict.keys()):
        rand_multi = abs(np.random.uniform(0.001,0.003) ) # Volatility scaler
        trend_scaler = np.random.uniform(0.2,0.5) # Trendiness scaler
        up_down_bias = np.random.randint(2)
        up_down_freq = 0.04
        
        for j in range(n_samples):
            date = start_date + timedelta(hours=1*j)
            if j == 0:
                open_price = pair_dict[pair]
            else:
                open_price = round(data['close'][(i*n_samples)+j-1] * (1 + np.random.uniform(low=-1.0) * 0.000001), 5)
                
            rand_nums = np.sort(abs(np.random.randn(3)))
            
            # Randomly scale rand_nums to prevent large high/low gap
            if rand_nums[1] > 2 * rand_nums[0]:
                if np.random.rand() > 0.2:
                    rand_nums[2] *= np.random.uniform()
                    rand_nums[1] *= np.random.uniform()
                    rand_nums = np.sort(rand_nums)
            
            # Randomly make the high or low the bigger multiplier
            high_low_flag = np.random.randint(2)
            if high_low_flag:
                max_rand = rand_nums[2]
                min_rand = rand_nums[1]
            else:
                max_rand = rand_nums[1]
                min_rand = rand_nums[2]
            
            # Randomly make the close positive or negative relative to open
            close_flag = np.random.randint(2)
            if np.random.uniform() < up_down_freq:
                close_flag = up_down_bias
            mid_rand = rand_nums[0] if close_flag else -rand_nums[0]
            
            high_price = round(open_price * (max_rand * rand_multi + 1), 5)
            low_price = round(open_price * (-min_rand * rand_multi + 1), 5)
            close_price = round(open_price * (mid_rand * rand_multi + 1), 5)
            
            if np.random.rand() < trend_scaler:
                if mid_rand > 0:
                    close_price = high_price - ((high_price - close_price) * np.random.rand())
                else:
                    close_price = ((close_price - low_price) * np.random.rand()) + low_price
            close_price = round(close_price, 5)
                    
            data['date'].append(date.strftime("%Y-%m-%d %H:%M:%S"))
            data['pair'].append(pair)
            data['open'].append(open_price)
            data['high'].append(high_price)
            data['low'].append(low_price)
            data['close'].append(close_price)
    
    conn = sqlite3.connect(db_path)
    try:
        df = pd.DataFrame(data)
        df.to_sql('test_hourly_exchange_rate', conn, if_exists='append', index=False)
        print("Hourly rates inserted successfully.")
    except Exception as e:
        print(e)
    finally:
        conn.close()
    

def main(fresh=False):
    db_path = (r"/workspace/tests/data/test_raw_data.db")
    
    # Optionally create a new db and data
    if fresh:
        if os.path.isfile(db_path):
            os.remove(db_path)
    
    n_samples = 1000
    create_dummy_table(db_path)
    populate_dummy_table(db_path, n_samples)
    
    
if __name__ == "__main__":
    main(fresh=True)
    