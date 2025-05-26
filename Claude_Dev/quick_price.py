import os
import pandas as pd
import mplfinance as mpf
from datetime import datetime, time
import matplotlib.pyplot as plt
from tabulate import tabulate

def analyze_stock_data(base_dir, subdir, ticker, buy_date, sell_date):
    """
    Analyze stock data for a given ticker and date range.

    Args:
        base_dir (str): Base directory path containing Ticker_data
        subdir (str): Name of subdirectory in Ticker_data
        ticker (str): Stock ticker symbol
        buy_date (str): Buy date in YYYY-MM-DD format
        sell_date (str): Sell date in YYYY-MM-DD format
    """
    # Construct directory path
    dir_path = os.path.join(base_dir, "Ticker_data", subdir)

    # Find the relevant file
    stock_files = [f for f in os.listdir(dir_path) if f.endswith('.csv') and ticker in f.upper()]

    if not stock_files:
        print(f"No data found for ticker {ticker}")
        return

    # Load the data
    file_path = os.path.join(dir_path, stock_files[0])
    df = pd.read_csv(file_path)

    # Convert datetime column
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)

    # Calculate Typical Price
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3

    # 1. Create candlestick plot
    plt.close()
    mpf.plot(df, type='candle', style='yahoo', volume=False,
             title=f'{ticker} Price Chart\n{buy_date} to {sell_date}')
    plt.show()

    # 2. Buy date morning price information (6:30 AM to 6:40 AM)
    buy_date = datetime.strptime(buy_date, '%Y-%m-%d').date()
    buy_mask = (
        (df.index.date == buy_date) &
        (df.index.time >= time(6, 30)) &
        (df.index.time <= time(6, 40))
    )
    buy_data = df[buy_mask]

    print("\nBuy Date Morning Prices:")
    print(tabulate(
        buy_data[['open', 'high', 'low', 'close', 'typical_price']],
        headers=['Time', 'Open', 'High', 'Low', 'Close', 'Typical Price'],
        tablefmt='grid',
        showindex=True,
        floatfmt='.2f'
    ))

    buy_mask = (
        (df.index.date == buy_date) &
        (df.index.time >= time(12, 55)) &
        (df.index.time <= time(13, 00))
    )
    buy_data = df[buy_mask]

    print("\nBuy Date Afternoon Prices:")
    print(tabulate(
        buy_data[['open', 'high', 'low', 'close', 'typical_price']],
        headers=['Time', 'Open', 'High', 'Low', 'Close', 'Typical Price'],
        tablefmt='grid',
        showindex=True,
        floatfmt='.2f'
    ))


    # 3. Sell date afternoon price information (12:55 PM to 1:00 PM)
    sell_date = datetime.strptime(sell_date, '%Y-%m-%d').date()
    sell_mask = (
        (df.index.date == sell_date) &
        (df.index.time >= time(12, 55)) &
        (df.index.time <= time(13, 0))
    )
    sell_data = df[sell_mask]

    print("\nSell Date Afternoon Prices (12:55 PM - 1:00 PM):")
    print(tabulate(
        sell_data[['open', 'high', 'low', 'close', 'typical_price']],
        headers=['Time', 'Open', 'High', 'Low', 'Close', 'Typical Price'],
        tablefmt='grid',
        showindex=True,
        floatfmt='.2f'
    ))

if __name__ == "__main__":
    # Base directory from the provided code
    BASE_DIR = "C:/Users/ryant/Documents/Stock_Market/Python/universe_data/VM Weekly Breakout"

    # Get user inputs
    subdir = input("Enter subdirectory name from Ticker_data: ")
    ticker = input("Enter ticker symbol: ").upper()
    buy_date = input("Enter buy date (YYYY-MM-DD): ")
    sell_date = input("Enter sell date (YYYY-MM-DD): ")

    # Run analysis
    analyze_stock_data(BASE_DIR, subdir, ticker, buy_date, sell_date)