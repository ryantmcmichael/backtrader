import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, time, timedelta
import random
import mplfinance as mpf
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configure logging
def setup_logging():
    # Individual trades logger
    trade_logger = logging.getLogger('trade_logger')
    trade_logger.handlers = []  # Clear existing handlers
    trade_logger.setLevel(logging.INFO)
    trade_handler = logging.FileHandler('Individual Trades.log', mode='w')
    trade_handler.setFormatter(logging.Formatter('%(message)s'))
    trade_logger.addHandler(trade_handler)

    # Summary logger
    summary_logger = logging.getLogger('summary_logger')
    summary_logger.handlers = []  # Clear existing handlers
    summary_logger.setLevel(logging.INFO)
    summary_handler = logging.FileHandler('Backtest summary.log', mode='w')
    summary_handler.setFormatter(logging.Formatter('%(message)s'))
    summary_logger.addHandler(summary_handler)

    return trade_logger, summary_logger

# Load and process market signals
def load_market_signals(base_dir):
    gauges_dir = os.path.join(base_dir, 'Gauges')

    # Load MG Gauge
    mg_df = pd.read_csv(os.path.join(gauges_dir, 'daily-market-momentum-ga.csv'))
    mg_df['DateTime'] = pd.to_datetime(mg_df['DateTime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    mg_df = mg_df.dropna(subset=['DateTime'])
    mg_df = mg_df.rename(columns={
        'Positive Momentum Gauge®': 'Pos MG',
        'Negative Momentum Gauge®': 'Neg MG'
    })

    # Load SP500 Gauge
    sp500_df = pd.read_csv(os.path.join(gauges_dir, 'daily-sp-500-momentum-ga.csv'))
    sp500_df['DateTime'] = pd.to_datetime(sp500_df['DateTime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    sp500_df = sp500_df.dropna(subset=['DateTime'])
    sp500_df = sp500_df.rename(columns={
        'Positive Momentum Gauge®': 'Pos SP500',
        'Negative Momentum Gauge®': 'Neg SP500'
    })

    # Load Sector Gauges
    sector_gauges = {}
    for filename in os.listdir(gauges_dir):
        if filename not in ['daily-market-momentum-ga.csv', 'daily-sp-500-momentum-ga.csv']:
            sector_name = os.path.splitext(filename)[0]
            sector_df = pd.read_csv(os.path.join(gauges_dir, filename))
            sector_df['DateTime'] = pd.to_datetime(sector_df['DateTime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
            sector_df = sector_df.dropna(subset=['DateTime'])
            sector_df = sector_df.rename(columns={
                'Positive Momentum Gauge®': f'Pos {sector_name}',
                'Negative Momentum Gauge®': f'Neg {sector_name}'
            })
            sector_gauges[sector_name] = sector_df

    return mg_df, sp500_df, sector_gauges

# Load sector information
def load_sector_info(base_dir):
    sector_file = os.path.join(base_dir, 'Ticker_Download_Log', 'ticker_sectors.csv')
    sector_df = pd.read_csv(sector_file)
    return dict(zip(sector_df['Ticker'], sector_df['Sector']))

# Check market signals
def check_mg_signal(date, mg_df):
    if isinstance(date, datetime):
        date = date.date()
    signal_data = mg_df[mg_df['DateTime'].dt.date == date]
    if signal_data.empty:
        return False
    return (signal_data['Pos MG'].iloc[0] > signal_data['Neg MG'].iloc[0]) and (signal_data['Neg MG'].iloc[0] < 40)

def check_sp500_signal(date, sp500_df):
    if isinstance(date, datetime):
        date = date.date()
    signal_data = sp500_df[sp500_df['DateTime'].dt.date == date]
    if signal_data.empty:
        return False
    return signal_data['Pos SP500'].iloc[0] > signal_data['Neg SP500'].iloc[0]

def check_sector_signal(date, sector, sector_gauges):
    if isinstance(date, datetime):
        date = date.date()
    sector_df = sector_gauges.get(sector)
    if sector_df is None:
        return False
    signal_data = sector_df[sector_df['DateTime'].dt.date == date]
    if signal_data.empty:
        return False
    pos_col = f'Pos {sector}'
    neg_col = f'Neg {sector}'
    return signal_data[pos_col].iloc[0] > signal_data[neg_col].iloc[0]

# Generate random entry time
def get_random_entry_time(date):
    base_time = datetime.combine(date, time(6, 32))
    max_minutes = 3
    random_minutes = random.randint(0, max_minutes)
    return base_time + timedelta(minutes=random_minutes)

# Process stock data
def load_stock_data(file_path):
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    return df

    ############################### PART 2
    ###############################

# Trading logic
class Trade:
    def __init__(self, ticker, entry_datetime, entry_price, investment):
        self.ticker = ticker
        self.entry_datetime = entry_datetime
        self.entry_price = entry_price
        self.investment = investment
        self.shares = investment / entry_price
        self.exit_datetime = None
        self.exit_price = None
        self.exit_reason = None

    def close_trade(self, exit_datetime, exit_price, reason):
        self.exit_datetime = exit_datetime
        self.exit_price = exit_price
        self.exit_reason = reason

    def get_pnl(self):
        if self.exit_price is None:
            return 0, 0
        dollar_pnl = (self.exit_price - self.entry_price) * self.shares
        percent_pnl = (self.exit_price / self.entry_price - 1) * 100
        return dollar_pnl, percent_pnl

def process_trade(stock_data, trade, mg_df, sp500_df, sector_gauges, sector, second_friday):
    profit_target = 1.07  # 7% profit
    loss_target = 0.95   # 5% loss

    # Process each day's data
    current_date = trade.entry_datetime.date()
    while current_date <= second_friday:
        # Get data for current day
        day_data = stock_data[stock_data.index.date == current_date]

        if not day_data.empty:
            # Check profit target
            profit_hits = day_data[day_data['high'].ge(trade.entry_price * profit_target)]
            if not profit_hits.empty:
                exit_datetime = profit_hits.index[0]
                trade.close_trade(exit_datetime, trade.entry_price * profit_target, "Profit Target")
                return trade

            # Check loss target
            loss_hits = day_data[day_data['low'].le(trade.entry_price * loss_target)]
            if not loss_hits.empty:
                exit_datetime = loss_hits.index[0]
                trade.close_trade(exit_datetime, trade.entry_price * loss_target, "Loss Target")
                return trade

            # Check market signals for next day
            next_date = current_date + timedelta(days=1)

            # MG Signal check
            if not check_mg_signal(next_date, mg_df):
                next_day_data = stock_data[stock_data.index.date == next_date]
                if not next_day_data.empty:
                    exit_datetime = next_day_data.index[0]
                    exit_price = next_day_data['open'].iloc[0]
                    trade.close_trade(exit_datetime, exit_price, "MG Signal")
                    return trade

            # SP500 Signal check
            if not check_sp500_signal(next_date, sp500_df):
                next_day_data = stock_data[stock_data.index.date == next_date]
                if not next_day_data.empty:
                    exit_datetime = next_day_data.index[0]
                    exit_price = next_day_data['open'].iloc[0]
                    trade.close_trade(exit_datetime, exit_price, "SP500 Signal")
                    return trade

            # Sector Signal check
            if not check_sector_signal(next_date, sector, sector_gauges):
                next_day_data = stock_data[stock_data.index.date == next_date]
                if not next_day_data.empty:
                    exit_datetime = next_day_data.index[0]
                    exit_price = next_day_data['open'].iloc[0]
                    trade.close_trade(exit_datetime, exit_price, "Sector Signal")
                    return trade

        # Check if we've reached second Friday
        if current_date == second_friday:
            day_data = stock_data[stock_data.index.date == current_date]
            if not day_data.empty:
                # Find the timestamp 1 hour before market close
                close_time = datetime.combine(current_date, time(15, 0))  # 3:00 PM
                exit_data = day_data[day_data.index <= close_time]
                if not exit_data.empty:
                    exit_datetime = exit_data.index[-1]
                    exit_price = exit_data['close'].iloc[-1]
                    trade.close_trade(exit_datetime, exit_price, "Second Friday")
                    return trade

        current_date += timedelta(days=1)

    return trade

def find_first_friday(start_date, end_date):
    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() == 4:  # Friday
            return current_date
        current_date += timedelta(days=1)
    return None

def find_second_friday(start_date, end_date):
    first_friday = find_first_friday(start_date, end_date)
    if first_friday:
        current_date = first_friday + timedelta(days=1)
        while current_date <= end_date:
            if current_date.weekday() == 4:  # Friday
                return current_date
            current_date += timedelta(days=1)
    return None


    ############################### PART 3
    ###############################

# Visualization functions
def plot_trade_candlestick(stock_data, trade, date_id):
    # Extend the plot range
    start_date = trade.entry_datetime.date()
    end_date = (trade.exit_datetime + timedelta(days=3)).date()
    plot_data = stock_data[start_date:end_date].copy()

    if plot_data.empty:
        raise ValueError("No data available for the specified date range")

    # Create entry/exit data points that align with the plot data index
    entry_data = pd.Series(np.nan, index=plot_data.index, name='Entry')
    exit_data = pd.Series(np.nan, index=plot_data.index, name='Exit')

    # Find nearest timestamps for entry and exit
    entry_idx = plot_data.index[plot_data.index.get_indexer([trade.entry_datetime], method='nearest')[0]]
    exit_idx = plot_data.index[plot_data.index.get_indexer([trade.exit_datetime], method='nearest')[0]]

    # Set the values at the correct indices
    entry_data[entry_idx] = trade.entry_price
    exit_data[exit_idx] = trade.exit_price

    # Create the plot
    fig, axes = mpf.plot(
        plot_data,
        type='hollow_and_filled',
        style='yahoo',
        figsize=(12, 6),
        warn_too_much_data=1000000000,
        title=f'{trade.ticker} - {date_id}',
        addplot=[
            mpf.make_addplot(entry_data, type='scatter', marker='^', markersize=200, color='g'),
            mpf.make_addplot(exit_data, type='scatter', marker='v', markersize=200, color='r')
        ],
        returnfig=True
    )

    # Add annotations
    entry_text = f'Entry: ${trade.entry_price:.2f}'
    exit_text = f'Exit: ${trade.exit_price:.2f}\n{trade.exit_reason}'

    # Convert timestamps to plot indices
    entry_plot_idx = list(plot_data.index).index(entry_idx)
    exit_plot_idx = list(plot_data.index).index(exit_idx)

    # Add text annotations
    axes[0].annotate(entry_text,
                    xy=(entry_plot_idx, trade.entry_price),
                    xytext=(10, 10), textcoords='offset points')
    axes[0].annotate(exit_text,
                    xy=(exit_plot_idx, trade.exit_price),
                    xytext=(10, -20), textcoords='offset points')

    # Adjust layout
    fig.set_tight_layout(True)
    fig.align_labels()

    return fig

def plot_portfolio_returns(portfolio_returns):
    dates = sorted(portfolio_returns.keys())
    cumulative_returns = []
    instantaneous_returns = []
    running_total = 0

    for date in dates:
        running_total += portfolio_returns[date]
        cumulative_returns.append(running_total)
        instantaneous_returns.append(portfolio_returns[date])

    x = np.arange(len(dates))
    width = 0.35

    fig, ax = plt.subplots(figsize=(15, 7))

    # Plot cumulative returns
    cum_bars = ax.bar(x - width/2, cumulative_returns, width, label='Cumulative Returns',
                     color=['g' if x >= 0 else 'r' for x in cumulative_returns])

    # Plot instantaneous returns
    inst_bars = ax.bar(x + width/2, instantaneous_returns, width, label='Period Returns',
                      color=['cyan' if x >= 0 else 'magenta' for x in instantaneous_returns])

    ax.set_xlabel('Date ID')
    ax.set_ylabel('Returns (%)')
    ax.set_title('Portfolio Returns by Period')
    ax.set_xticks(x)
    ax.set_xticklabels(dates, rotation=45)
    ax.legend()

    fig.set_tight_layout(True)
    return fig

def plot_annual_returns(portfolio_returns):
    # Convert dates to years and aggregate returns
    annual_returns = {}
    for date_id, returns in portfolio_returns.items():
        year = datetime.strptime(date_id, '%Y-%m-%d').year
        annual_returns[year] = annual_returns.get(year, 0) + returns

    years = sorted(annual_returns.keys())
    cumulative_returns = []
    instantaneous_returns = []
    running_total = 0

    for year in years:
        running_total += annual_returns[year]
        cumulative_returns.append(running_total)
        instantaneous_returns.append(annual_returns[year])

    x = np.arange(len(years))
    width = 0.35

    fig, ax = plt.subplots(figsize=(15, 7))

    # Plot cumulative returns
    cum_bars = ax.bar(x - width/2, cumulative_returns, width, label='Cumulative Returns',
                     color=['g' if x >= 0 else 'r' for x in cumulative_returns])

    # Plot instantaneous returns
    inst_bars = ax.bar(x + width/2, instantaneous_returns, width, label='Annual Returns',
                      color=['cyan' if x >= 0 else 'magenta' for x in instantaneous_returns])

    ax.set_xlabel('Year')
    ax.set_ylabel('Returns (%)')
    ax.set_title('Portfolio Returns by Year')
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.legend()

    fig.set_tight_layout(True)
    return fig

def main():
    # Set base directory
    base_dir = "C:/Users/ryant/Documents/Stock_Market/Python/universe_data/VM Weekly Breakout"

    # Setup logging
    trade_logger, summary_logger = setup_logging()

    # Load market signals and sector information
    mg_df, sp500_df, sector_gauges = load_market_signals(base_dir)
    sector_info = load_sector_info(base_dir)

    # Initialize portfolio tracking
    initial_portfolio = 100000
    portfolio_value = initial_portfolio
    portfolio_returns = {}
    all_trades = []
    max_drawdown = 0

    # Get all date directories
    ticker_data_dir = os.path.join(base_dir, 'Ticker_data')
    date_dirs = sorted([d for d in os.listdir(ticker_data_dir)
                       if os.path.isdir(os.path.join(ticker_data_dir, d))])

    # Process each date directory
    for date_id in tqdm(date_dirs, desc="Processing date ranges"):
        dir_path = os.path.join(ticker_data_dir, date_id)
        stock_files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]

        if not stock_files:
            continue

        # Parse date range from first file
        sample_file = stock_files[0]
        start_date_str = sample_file.split('_')[5]
        end_date_str = sample_file.split('_')[6].replace('.csv', '')
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()

        # Find first and second Fridays
        first_friday = find_first_friday(start_date, end_date)
        second_friday = find_second_friday(start_date, end_date)

        if not first_friday or not second_friday:
            continue

        # Check market signals for day before first Friday
        signal_date = first_friday - timedelta(days=1)
        if not (check_mg_signal(signal_date, mg_df) and
                check_sp500_signal(signal_date, sp500_df)):
            continue

        # Calculate per-stock investment
        num_stocks = len(stock_files)
        per_stock_investment = initial_portfolio / num_stocks

        # Process each stock
        period_trades = []
        for stock_file in stock_files:
            ticker = stock_file.split('_')[4]
            sector = sector_info.get(ticker)
            if not sector:
                continue

            stock_data = load_stock_data(os.path.join(dir_path, stock_file))

            # Get entry price and time
            entry_datetime = get_random_entry_time(first_friday)
            entry_data = stock_data[stock_data.index >= entry_datetime].iloc[0]
            entry_price = entry_data['open']

            # Create and process trade
            trade = Trade(ticker, entry_datetime, entry_price, per_stock_investment)
            trade = process_trade(stock_data, trade, mg_df, sp500_df,
                                sector_gauges, sector, second_friday)

            if trade.exit_price is not None:
                period_trades.append(trade)
                all_trades.append(trade)

                # Log trade details
                dollar_pnl, percent_pnl = trade.get_pnl()
                trade_logger.info(
                    f"Ticker: {trade.ticker}, "
                    f"Entry Datetime: {trade.entry_datetime}, "
                    f"Entry Price: ${trade.entry_price:.2f}, "
                    f"Exit Datetime: {trade.exit_datetime}, "
                    f"Exit Price: ${trade.exit_price:.2f}, "
                    f"Exit Reason: {trade.exit_reason}, "
                    f"P/L ($): ${dollar_pnl:.2f}, "
                    f"P/L (%): {percent_pnl:.2f}%"
                )

        # Calculate period returns
        period_pnl = sum(trade.get_pnl()[0] for trade in period_trades)
        period_return_pct = (period_pnl / initial_portfolio) * 100
        portfolio_returns[date_id] = period_return_pct

        # Update portfolio value and track max drawdown
        portfolio_value += period_pnl
        drawdown = initial_portfolio - portfolio_value
        max_drawdown = max(max_drawdown, drawdown)

    # Log and print final results
    total_gain = portfolio_value - initial_portfolio
    total_gain_pct = (total_gain / initial_portfolio) * 100
    summary = (
        f"Starting Portfolio Value: ${initial_portfolio:,.2f}\n"
        f"Final Portfolio Value: ${portfolio_value:,.2f}\n"
        f"Total Gain: ${total_gain:,.2f}\n"
        f"Total Gain Percentage: {total_gain_pct:.2f}%\n"
        f"Max Drawdown: ${max_drawdown:,.2f}"
    )
    print(summary)
    summary_logger.info(summary)

    # Plot best and worst performers
    print("\nGenerating candlestick plots for best and worst performers...")
    sorted_trades = sorted(all_trades, key=lambda x: x.get_pnl()[1], reverse=True)
    best_trades = sorted_trades[:3]
    worst_trades = sorted_trades[-3:]

    for trade in best_trades + worst_trades:
        try:
            # Find the correct date directory
            date_dirs = [d for d in os.listdir(ticker_data_dir)
                        if os.path.isdir(os.path.join(ticker_data_dir, d))]

            # Look through date directories to find the stock file
            stock_file = None
            stock_path = None

            for date_dir in date_dirs:
                dir_path = os.path.join(ticker_data_dir, date_dir)
                files = [f for f in os.listdir(dir_path) if f.endswith('.csv') and trade.ticker in f]

                for file in files:
                    file_path = os.path.join(dir_path, file)
                    # Load a small sample to check date range
                    sample_data = pd.read_csv(file_path, nrows=1)
                    file_start_date = pd.to_datetime(sample_data['datetime'].iloc[0]).date()

                    if file_start_date <= trade.entry_datetime.date() <= file_start_date + timedelta(days=14):
                        stock_file = file
                        stock_path = file_path
                        break

                if stock_path:
                    break

            if stock_path and os.path.exists(stock_path):
                print(f"Plotting {trade.ticker} from {trade.entry_datetime.strftime('%Y-%m-%d')}")
                stock_data = load_stock_data(stock_path)
                fig = plot_trade_candlestick(stock_data, trade,
                                           trade.entry_datetime.strftime('%Y-%m-%d'))
                plt.show()
            else:
                print(f"Could not find data file for {trade.ticker} traded on {trade.entry_datetime.strftime('%Y-%m-%d')}")

        except Exception as e:
            print(f"Error plotting {trade.ticker}: {str(e)}")

    # Plot portfolio returns
    fig = plot_portfolio_returns(portfolio_returns)
    plt.show()

    # Plot annual returns
    fig = plot_annual_returns(portfolio_returns)
    plt.show()

if __name__ == "__main__":
    main()