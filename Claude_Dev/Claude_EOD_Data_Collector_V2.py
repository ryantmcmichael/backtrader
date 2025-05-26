import os
import sys
import pandas as pd
import time
import yfinance as yf
from eodhd import APIClient
import multiprocessing as mp
from datetime import datetime, timedelta
import logging
import warnings
import io
from contextlib import redirect_stdout, redirect_stderr

# Suppress pandas warnings
warnings.filterwarnings('ignore')

def LOG_insert(text, level):
    fpath = ("C:/Users/ryant/Documents/Stock_Market/Python/universe_data/" +
             "VM Weekly Breakout/Ticker_Download_Log/data_collection.log")
    infoLog = logging.FileHandler(fpath)
    infoLog.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    logger = logging.getLogger(fpath)
    logger.setLevel(level)

    if not logger.handlers:
        logger.addHandler(infoLog)
        if (level == logging.INFO):
            logger.info(text)
        if (level == logging.ERROR):
            logger.error(text)
        if (level == logging.WARNING):
            logger.warning(text)

    handlers = logger.handlers[:]
    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()

    return

def print_progress(start_time, current, total):
    """Print progress bar to console."""
    elapsed_time = time.time() - start_time
    progress = current / total

    if progress > 0:
        estimated_total_time = elapsed_time / progress
        remaining_time = estimated_total_time - elapsed_time

        sys.stdout.write(f'\rProgress: {current}/{total} tickers processed '
                        f'(ETA: {remaining_time:.1f}s remaining)')
        sys.stdout.flush()

def get_sector_from_file(ticker, base_dir):
    """Get sector information from backup CSV file."""
    try:
        sector_df = pd.read_csv(os.path.join(base_dir, 'sector_lookup.csv'))
        sector = sector_df.loc[sector_df['Ticker'] == ticker, 'Sector'].iloc[0]
        source = 'file'
        return sector, source
    except:
        return None, None

def get_sector_info(ticker, base_dir):
    """Get sector information from yfinance or backup file."""
    # Redirect yfinance output
    stdout = io.StringIO()
    stderr = io.StringIO()

    # First try backup file. Then try yfinance.
    sector, source = get_sector_from_file(ticker, base_dir)

    if sector is None:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                stock = yf.Ticker(ticker)
                sector = stock.info['sector']

                if sector == "Financial Services":
                    sector = "Financial"

                if sector:
                    source = 'yfinance'
                    return sector, source

            except Exception as e:
                LOG_insert(f"YFinance failed for {ticker}: {str(e)}", logging.WARNING)
    return sector, source

def process_ticker_data(ticker, start_date, end_date, api_client):
    """Process and validate ticker data from EOD API."""
    try:
        # Convert dates to Unix timestamps
        start_unix = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
        end_unix = int(datetime.strptime(f"{end_date} 23:59:59", "%Y-%m-%d %H:%M:%S").timestamp())

        # Get data from API
        resp = api_client.get_intraday_historical_data(
            symbol=ticker,
            from_unix_time=start_unix,
            to_unix_time=end_unix,
            interval='1m'
        )

        if not resp:
            LOG_insert(f"EOD API ERROR: {ticker} on {end_date}", logging.ERROR)
            return None

        # Convert to DataFrame
        df = pd.DataFrame(resp)

        # Handle timestamp conversion
        try:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        except:
            df['datetime'] = pd.to_datetime(df['timestamp'])

        # Localize to UTC and convert to Pacific time
        df['datetime'] = df['datetime'].dt.tz_localize('UTC').dt.tz_convert('US/Pacific')

        # Filter for market hours (6:30 AM to 1:00 PM PT)
        df['time'] = df['datetime'].dt.time
        market_mask = ((df['time'] >= datetime.strptime('06:30:00', '%H:%M:%S').time()) &
                      (df['time'] <= datetime.strptime('13:00:00', '%H:%M:%S').time()))
        df = df.loc[market_mask]

        # Select and rename columns
        df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]

        return df

    except Exception as e:
        LOG_insert(f"Error processing {ticker}: {str(e)}", logging.ERROR)
        return None

def process_date_group(args):
    """Process a group of tickers for a specific date range."""
    api_key, base_dir, group = args
    date, tickers = group

    # Initialize tracking sets and dictionaries
    eod_failures = set()
    sector_failures = set()
    ticker_sectors = {}

    # Create API client for this process
    api_client = APIClient(api_key)

    # Calculate date window
    date_obj = datetime.strptime(date, "%m/%d/%Y")
    start_date = (date_obj - timedelta(days=10)).strftime("%Y-%m-%d")
    end_date = (date_obj + timedelta(days=21)).strftime("%Y-%m-%d")

    # Create output directory
    output_dir = os.path.join(base_dir, 'Ticker_data', date_obj.strftime("%Y-%m-%d"))
    os.makedirs(output_dir, exist_ok=True)

    for ticker in tickers:
        # Get sector information (respecting rate limit)
        if ticker not in ticker_sectors:
            sector, source = get_sector_info(ticker, base_dir)
            if sector:
                ticker_sectors[ticker] = sector
            else:
                sector_failures.add(ticker)
            if source == 'yfinance':
                time.sleep(2.4)  # Rate limit for yfinance (25 requests per minute)

        # Check if file already exists
        output_file = f"0_0_1_0_{ticker}_{start_date}_{end_date}.csv"
        if os.path.exists(os.path.join(output_dir, output_file)):
            continue

        # Get ticker data
        df = process_ticker_data(ticker, start_date, end_date, api_client)

        if df is None:
            eod_failures.add(ticker)
            continue

        # Save data
        df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df.to_csv(os.path.join(output_dir, output_file), index=False)

        # Rate limit for EOD API (100 requests per minute)
        time.sleep(0.6)

    return {
        'eod_failures': eod_failures,
        'sector_failures': sector_failures,
        'ticker_sectors': ticker_sectors,
        'num_tickers': len(tickers)
    }

def save_results(base_dir, all_results):
    """Save accumulated results to output files."""
    log_dir = os.path.join(base_dir, 'Ticker_Download_Log')

    # Combine results from all processes
    all_eod_failures = set()
    all_sector_failures = set()
    all_ticker_sectors = {}

    for result in all_results:
        all_eod_failures.update(result['eod_failures'])
        all_sector_failures.update(result['sector_failures'])
        all_ticker_sectors.update(result['ticker_sectors'])

    # Save EOD failures
    pd.DataFrame(list(all_eod_failures), columns=['Ticker']).to_csv(
        os.path.join(log_dir, 'EOD_ticker_failures.csv'), index=False)

    # Save sector information
    pd.DataFrame(list(all_ticker_sectors.items()),
                columns=['Ticker', 'Sector']).to_csv(
        os.path.join(log_dir, 'ticker_sectors.csv'), index=False)

    # Save sector failures
    pd.DataFrame(list(all_sector_failures), columns=['Ticker']).to_csv(
        os.path.join(log_dir, 'sector_failures.csv'), index=False)

def main():
    # User-provided configuration
    api_key = '5e98828c864a88.41999009'
    base_dir = "C:/Users/ryant/Documents/Stock_Market/Python/universe_data/VM Weekly Breakout"
    input_csv = "JD_RAW_MDA_Weekly_Picks.csv"

    os.makedirs(os.path.join(base_dir,'Ticker_Download_Log'), exist_ok=True)
    os.makedirs(os.path.join(base_dir,'Ticker_data'), exist_ok=True)

    # Setup logging
    try:
        os.remove(os.path.join(base_dir,'Ticker_Download_Log/data_collection.log'))
    except FileNotFoundError:
        None

    LOG_insert('Script Starting!', logging.INFO)

    # Redirect stderr to log file
    sys.stderr = open(os.path.join(base_dir, 'Ticker_Download_Log', 'data_collection.log'), 'a')

    try:
        # Read input CSV
        df = pd.read_csv(os.path.join(base_dir, input_csv))

        # Group tickers by date
        date_groups = [(date, df[df['End Date'] == date]['Ticker'].unique().tolist())
                      for date in df['End Date'].unique()]

        # Calculate total operations (sum of tickers across all date groups)
        total_operations = sum(len(tickers) for _, tickers in date_groups)
        current_operation = 0
        start_time = time.time()

        # Process in parallel
        with mp.Pool(processes=min(mp.cpu_count(), len(date_groups))) as pool:
            args = [(api_key, base_dir, group) for group in date_groups]

            all_results = []
            for result in pool.imap_unordered(process_date_group, args):
                all_results.append(result)
                current_operation += result['num_tickers']
                print_progress(start_time, current_operation, total_operations)

        # Save results
        save_results(base_dir, all_results)

        print("\nProcessing complete!")

    except Exception as e:
        LOG_insert(f"Main process error: {str(e)}", logging.ERROR)
        raise

if __name__ == "__main__":
    main()

'''

    api_key = '5e98828c864a88.41999009'
    base_dir = "C:/Users/ryant/Documents/Stock_Market/Python/universe_data/VM Weekly Breakout"
    input_csv = "JD_RAW_MDA_Weekly_Picks - Truncated.csv"
'''