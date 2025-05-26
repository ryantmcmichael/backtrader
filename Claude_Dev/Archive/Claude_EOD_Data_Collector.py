# -*- coding: utf-8 -*-
"""
EOD Historical Data Collection Script
Created using Spyder 3.9
"""

from eodhd import APIClient
import os
import pandas as pd
import datetime
import time
import yfinance as yf
import multiprocessing as mp
from datetime import timedelta
import pytz
import logging
import sys
import math

class DataCollector:
    def __init__(self, api_key, base_dir):
        """Initialize the DataCollector with API key and base directory."""
        self.api_key = api_key
        self.base_dir = base_dir

        # Create necessary directories
        self.data_dir = os.path.join(base_dir, "Claude_data")
        self.log_dir = os.path.join(base_dir, "Claude_log")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # Initialize logging
        logging.basicConfig(
            filename=os.path.join(self.log_dir, 'data_collection.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        # Initialize error tracking
        self.eod_failures = set()
        self.sector_failures = set()
        self.ticker_sectors = {}

        print(f"Initialized DataCollector. Data directory: {self.data_dir}")
        print(f"Log directory: {self.log_dir}")

def get_unix_timestamp(date_str, time_str="00:00:00"):
    """Convert date string to Unix timestamp."""
    dt = datetime.datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
    return int(dt.timestamp())

def get_sector_info(ticker, base_dir, sector_failures, ticker_sectors):
    """Get sector information from Yahoo Finance or backup CSV."""
    try:
        stock = yf.Ticker(ticker)
        sector = stock.info.get('sector', None)

        if sector == "Financial Services":
            sector = "Financial"

        if sector is None:
            # Check backup CSV
            sector_lookup_path = os.path.join(base_dir, "sector_lookup.csv")
            if os.path.exists(sector_lookup_path):
                sector_df = pd.read_csv(sector_lookup_path)
                sector_info = sector_df[sector_df['Ticker'] == ticker]

                if not sector_info.empty:
                    sector = sector_info.iloc[0]['Sector']
                else:
                    sector_failures.add(ticker)
                    return None
            else:
                print(f"Warning: sector_lookup.csv not found at {sector_lookup_path}")
                sector_failures.add(ticker)
                return None

        ticker_sectors[ticker] = sector
        return sector

    except Exception as e:
        logging.error(f"Error getting sector for {ticker}: {str(e)}")
        sector_failures.add(ticker)
        return None

def process_minute_data(df):
    """Process and clean minute-level data."""
    if df.empty:
        return df

    # Convert timestamp to datetime and localize to UTC
    df['Datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.set_index('Datetime')

    # Localize to UTC and convert to Pacific time
    df.index = df.index.tz_localize('UTC').tz_convert('US/Pacific')

    # Filter for market hours (6:30 AM to 1:00 PM PT)
    market_mask = (df.index.time >= datetime.time(6, 30)) & \
                 (df.index.time <= datetime.time(13, 0))
    df = df[market_mask].copy()

    # Select and rename columns
    df = df[['open', 'high', 'low', 'close', 'volume']].copy()

    # Reset index to get Datetime as a column
    df.reset_index(inplace=True)

    # Format Datetime column
    df['Datetime'] = df['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

    return df

def download_ticker_data(api_key, ticker, start_date, end_date, output_dir, eod_failures):
    """Download and process minute data for a single ticker."""
    try:
        print(f"Downloading data for {ticker} from {start_date} to {end_date}")

        # Create API client for this process
        api = APIClient(api_key)

        # Convert end_date to include end of day
        end_date_adj = f"{end_date} 23:59:59"

        # Get Unix timestamps
        start_unix = get_unix_timestamp(start_date)
        end_unix = get_unix_timestamp(end_date, "23:59:59")

        # Download data from EOD
        resp = api.get_intraday_historical_data(
            symbol=ticker,
            from_unix_time=start_unix,
            to_unix_time=end_unix,
            interval='1m'
        )

        # Convert response to DataFrame
        df = pd.DataFrame(resp)

        if df.empty:
            print(f"No data received for {ticker}")
            eod_failures.add(f"{ticker}_{start_date}_{end_date}")
            return False

        # Process the data
        df = process_minute_data(df)

        if df.empty:
            print(f"No market hours data for {ticker}")
            eod_failures.add(f"{ticker}_{start_date}_{end_date}")
            return False

        # Save to CSV
        output_file = f"0_0_1_0_{ticker}_{start_date}_{end_date}.csv"
        output_path = os.path.join(output_dir, output_file)
        df.to_csv(output_path, index=False)
        print(f"Saved data for {ticker} to {output_path}")

        # Add delay to respect rate limits
        time.sleep(0.5)

        return True

    except Exception as e:
        print(f"Error processing {ticker}: {str(e)}")
        logging.error(f"Error processing {ticker}: {str(e)}")
        eod_failures.add(f"{ticker}_{start_date}_{end_date}")
        return False

def process_date_group(args):
    """Process a group of tickers for a specific date."""
    api_key, base_dir, date_info = args
    date = date_info['date']
    start_date = date_info['start_date']
    end_date = date_info['end_date']
    tickers = date_info['tickers']

    # Create shared data structures for this process
    eod_failures = set()
    sector_failures = set()
    ticker_sectors = {}

    print(f"\nProcessing date group: {date.strftime('%Y-%m-%d')}")
    print(f"Start date: {start_date.strftime('%Y-%m-%d')}")
    print(f"End date: {end_date.strftime('%Y-%m-%d')}")
    print(f"Number of tickers: {len(tickers)}")

    # Create date-specific directory
    data_dir = os.path.join(base_dir, "Claude_data")
    date_dir = os.path.join(data_dir, date.strftime('%Y-%m-%d'))
    os.makedirs(date_dir, exist_ok=True)

    results = []
    for ticker in tickers:
        success = download_ticker_data(
            api_key,
            ticker,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d'),
            date_dir,
            eod_failures
        )
        results.append(success)

        # Get sector info if not already collected
        if ticker not in ticker_sectors and ticker not in sector_failures:
            get_sector_info(ticker, base_dir, sector_failures, ticker_sectors)

    return {
        'results': results,
        'eod_failures': eod_failures,
        'sector_failures': sector_failures,
        'ticker_sectors': ticker_sectors,
        'num_tickers': len(tickers)
    }

def print_progress(start_time, current, total):
    """Print progress bar and estimated time remaining."""
    elapsed_time = time.time() - start_time
    progress = current / total

    if progress > 0:
        estimated_total_time = elapsed_time / progress
        remaining_time = estimated_total_time - elapsed_time

        # Convert to hours, minutes, seconds
        hours = int(remaining_time // 3600)
        minutes = int((remaining_time % 3600) // 60)
        seconds = int(remaining_time % 60)

        # Clear previous line and print progress
        sys.stdout.write('\r' + ' ' * 80 + '\r')
        sys.stdout.write(f'Progress: {current}/{total} ({progress:.1%}) - Est. time remaining: {hours:02d}:{minutes:02d}:{seconds:02d}')
        sys.stdout.flush()

def main():
    # User-provided configuration
    api_key = '5e98828c864a88.41999009'
    base_dir = "C:/Users/ryant/Documents/Stock_Market/Python/universe_data/VM Weekly Breakout"
    input_csv = "JD_RAW_MDA_Weekly_Picks.csv"

    print("Starting data collection process...")
    print(f"Base directory: {base_dir}")
    print(f"Input CSV: {input_csv}")

    # Initialize collector for main process
    collector = DataCollector(api_key, base_dir)

    # Read input CSV
    input_path = os.path.join(base_dir, input_csv)
    print(f"Reading input CSV from: {input_path}")

    try:
        df = pd.read_csv(input_path)
        df['End Date'] = pd.to_datetime(df['End Date'], format='%m/%d/%Y')
        print(f"Successfully read input CSV. Found {len(df)} rows.")
    except Exception as e:
        print(f"Error reading input CSV: {str(e)}")
        return

    # Get unique dates and tickers
    unique_dates = df['End Date'].unique()
    print(f"Found {len(unique_dates)} unique dates to process.")

    # Prepare date windows
    date_groups = []
    for date in unique_dates:
        start_date = date - timedelta(days=7)
        end_date = date + timedelta(days=21)
        date_tickers = df[df['End Date'] == date]['Ticker'].unique().tolist()

        date_groups.append({
            'date': date,
            'start_date': start_date,
            'end_date': end_date,
            'tickers': date_tickers
        })

    # Calculate total operations for progress bar
    total_operations = sum(len(group['tickers']) for group in date_groups)
    current_operation = 0
    start_time = time.time()

    # Process groups in parallel
    print(f"\nStarting parallel processing with {min(mp.cpu_count(), len(date_groups))} processes...")

    # Create pool and process groups
    with mp.Pool(processes=min(mp.cpu_count(), len(date_groups))) as pool:
        # Prepare arguments for each process
        args = [(api_key, base_dir, group) for group in date_groups]

        # Process groups and collect results
        all_results = []
        for result in pool.imap_unordered(process_date_group, args):
            all_results.append(result)
            current_operation += result['num_tickers']
            print_progress(start_time, current_operation, total_operations)

    print("\nCombining results...")

    # Combine results from all processes
    combined_eod_failures = set()
    combined_sector_failures = set()
    combined_ticker_sectors = {}

    for result in all_results:
        combined_eod_failures.update(result['eod_failures'])
        combined_sector_failures.update(result['sector_failures'])
        combined_ticker_sectors.update(result['ticker_sectors'])

    print("\nSaving results...")

    # Save combined results
    pd.DataFrame(list(combined_eod_failures), columns=['Failure']).to_csv(
        os.path.join(collector.log_dir, 'EOD_ticker_failures.csv'), index=False
    )

    pd.DataFrame(list(combined_sector_failures), columns=['Ticker']).to_csv(
        os.path.join(collector.log_dir, 'sector_failures.csv'), index=False
    )

    pd.DataFrame(combined_ticker_sectors.items(), columns=['Ticker', 'Sector']).to_csv(
        os.path.join(collector.log_dir, 'ticker_sectors.csv'), index=False
    )

    print("\nData collection process completed!")

if __name__ == "__main__":
    main()

'''
    API_KEY = '5e98828c864a88.41999009'
    BASE_DIR = "C:/Users/ryant/Documents/Stock_Market/Python/universe_data/VM Weekly Breakout"
    INPUT_CSV = BASE_DIR + "/JD_RAW_MDA_Weekly_Picks.csv"

    api_key = '5e98828c864a88.41999009'
    base_dir = "C:/Users/ryant/Documents/Stock_Market/Python/universe_data/VM Weekly Breakout"
    input_csv = "JD_RAW_MDA_Weekly_Picks.csv"
'''