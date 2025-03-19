# fx_classify_custom_eval_funcs.py


import pandas as pd


def calculate_running_profit(df, initial_value=100000):
    
    df = df[df['confidence'] >= .45]
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by="date", inplace=True)
    
    running_profit = [initial_value]

    open_trades = {}  # Dictionary to track open trades for each pair

    # Initialize lists to store column values
    pair_columns = df['pair'].unique()
    open_trade_counts = {pair: [0] * len(df) for pair in pair_columns}
    total_open_trades = [0] * len(df)
    trade_bool = [0] * len(df)

    MAX_TRADES = 10 # Maximum number of trades allowed per pair
    MAX_TOTAL_TRADES = 55  # Maximum number of total trades allowed

    for i in range(len(df)):
        previous_value = running_profit[-1]
        row = df.iloc[i]
        current_time = row['date']
        pair = row['pair']
        predicted_target = row['predicted_category']
        hours_passed = row['hours_passed']
        buy_sl_time = row['buy_sl_time']
        sell_sl_time = row['sell_sl_time']
        wl = row['wl']
        
        profit = .0025
        risk = profit * .5
        
        # Check and close any open trades if their hours have passed
        for trade_pair in list(open_trades.keys()):
            new_open_trades = []
            for trade_time, trade_signal, trade_hours_passed, trade_wl in open_trades[trade_pair]:
                time_difference = (current_time - trade_time).total_seconds() / 3600  # Convert to hours

                if time_difference >= trade_hours_passed:
                    # Close the trade
                    if trade_wl == 1:
                        previous_value += previous_value * (profit)
                    else:
                        previous_value -= previous_value * (risk)
                else:
                    new_open_trades.append((trade_time, trade_signal, trade_hours_passed, trade_wl))

            if new_open_trades:
                open_trades[trade_pair] = new_open_trades
            else:
                del open_trades[trade_pair]

        # Initialize the pair's open trades list if not present
        if pair not in open_trades:
            open_trades[pair] = []

        # Calculate the total open trades for this index
        for p in pair_columns:
            if p in open_trades:
                open_trade_counts[p][i] = len(open_trades[p])
            else:
                open_trade_counts[p][i] = 0

        total_open_trades[i] = sum(open_trade_counts[p][i] for p in pair_columns)
        if total_open_trades[i] > MAX_TOTAL_TRADES:
            total_open_trades[i] = MAX_TOTAL_TRADES

        # Determine trade duration based on `w_l` and `predicted_category`
        if wl == 0:
            if predicted_target == 1:
                trade_hours_passed = buy_sl_time
            elif predicted_target == 2:
                trade_hours_passed = sell_sl_time
        elif wl == 1:
            trade_hours_passed = hours_passed
        else:
            print("trade hours passed issue")
            break

        # Check if we can open a new trade
        trade_opened = 0
        if total_open_trades[i] < MAX_TOTAL_TRADES:
            if len(open_trades[pair]) < MAX_TRADES:
                if not open_trades[pair] or open_trades[pair][0][1] == predicted_target:
                    if trade_hours_passed is not None:
                        open_trades[pair].append((current_time, predicted_target, trade_hours_passed, wl))
                        trade_opened = 1

        trade_bool[i] = trade_opened

        # Update open trade counts for all pairs
        for p in pair_columns:
            if p in open_trades:
                open_trade_counts[p][i] = len(open_trades[p])
            else:
                open_trade_counts[p][i] = 0

        # Recalculate the total open trades value after potentially opening new trades
        total_open_trades[i] = sum(open_trade_counts[p][i] for p in pair_columns)
        if total_open_trades[i] > MAX_TOTAL_TRADES:
            total_open_trades[i] = MAX_TOTAL_TRADES

        # Append the current running profit value
        running_profit.append(previous_value)

    # Ensure the length of running_profit matches the length of the DataFrame
    while len(running_profit) < len(df):
        running_profit.append(running_profit[-1])
    while len(running_profit) > len(df):
        running_profit.pop()
        
    df["Running Profit"] = running_profit

    # Convert lists to columns in the DataFrame
    for pair in pair_columns:
        df[f'{pair} open trade count'] = open_trade_counts[pair]
    df['Total Open Trades'] = total_open_trades
    df['trade_bool'] = trade_bool

    # Calculate the annual rate of return
    initial_profit = df['Running Profit'].iloc[0]
    final_profit = df['Running Profit'].iloc[-1]
    time_elapsed_years = ((df['date'].iloc[-1] - df['date'].iloc[0]).total_seconds() / 31536000) - 1 # Time in years adjusted for missing data
    annual_rate_of_return = ((final_profit / initial_profit) ** (1 / time_elapsed_years) - 1) * 100

    # Calculate the maximum drawdown as a percentage of the peak
    df['Cumulative Max'] = df['Running Profit'].cummax()
    df['Drawdown'] = df['Cumulative Max'] - df['Running Profit']
    df['Drawdown Percent'] = df['Drawdown'] / df['Cumulative Max'] * 100
    max_drawdown_percent = df['Drawdown Percent'].max()

    # Print results
    print(f"Initial Value: {initial_value}")
    print(f"Final Value: {final_profit:.2f}")
    print(f"Time Elapsed: {time_elapsed_years:.2f} years")
    print(f"Annual Rate of Return: {annual_rate_of_return:.2f}%")
    print(f"Maximum Drawdown: {max_drawdown_percent:.2f}%")

    # Print the annual returns
    df['Year'] = df['date'].dt.year
    def calculate_annual_return(group):
        start_value = group['Running Profit'].iloc[0]
        end_value = group['Running Profit'].iloc[-1]
        return (end_value - start_value) / start_value * 100

    annual_returns = df.groupby('Year').apply(calculate_annual_return)
    for year, return_value in annual_returns.items():
        print(f'Return for {year}: {return_value:.2f}%')
        
    # Add year and month columns
    df['Month'] = df['date'].dt.month

    def calculate_monthly_return(group):
        start_value = group['Running Profit'].iloc[0]
        end_value = group['Running Profit'].iloc[-1]
        return (end_value - start_value) / start_value * 100

    # Group by year and month, and apply the calculation
    monthly_returns = df.groupby(['Year', 'Month']).apply(calculate_monthly_return)

    # Print the monthly returns
    for (year, month), return_value in monthly_returns.items():
        print(f'Return for {year}-{month:02d}: {return_value:.2f}%')

    #print(list(monthly_returns))
    #print(sum(wins) / sum(losses))
    #print(len(wins) + len(losses))

    # Plot the running profit
    import matplotlib.pyplot as plt
    plt.plot(df['date'], df['Running Profit'], label='Running Profit')
    plt.xlabel('date')
    plt.ylabel('Running Profit')
    plt.title('Running Profit Over Time')
    plt.legend()
    plt.show()