import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, time, timedelta
import mplfinance as mpf
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas_market_calendars as mcal
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import yfinance as yf
import matplotlib.dates as mdates

# Close all existing plots
plt.close('all')

@dataclass
class StrategyParams:
    # Entry signals
    entry_use_mg_pos_signal: bool = False
    entry_use_mg_40_signal: bool = False
    entry_use_sp500_signal: bool = True

    # Exit signals
    exit_use_mg_pos_signal: bool = False
    exit_use_mg_40_signal: bool = True
    exit_use_sp500_signal: bool = False
    exit_use_sector_signals: bool = False

    # Other parameters
    profit_target: float = 0.05 # Percentage gain (0.1 means price * 1.1)
    loss_target: float = 0.2 # Percentage loss (0.1 means price * 0.9)
    entry_time_offset: int = 1  # Offset in minutes from baseline entry time
    start_date: Optional[str] = '2024-01-01'  # Format: 'YYYY-MM-DD'
    end_date: Optional[str] = '2025-01-01'    # Format: 'YYYY-MM-DD'

# Configuration constants
BASE_DIR = "C:/Users/ryant/Documents/Stock_Market/Python/universe_data/VM Weekly Breakout"
INITIAL_PORTFOLIO = 100000

BENCHMARK_TICKER = "SPY"  # Standard benchmark using S&P 500 ETF

# Time constants
EXIT_TIME = time(12, 59)  # 3:00 PM

def filter_date_dirs(date_dirs: List[str], start_date: Optional[str], end_date: Optional[str]) -> List[str]:
    """
    Filter directory list based on date range.

    Args:
        date_dirs: List of directory names (dates in YYYY-MM-DD format)
        start_date: Optional start date string in YYYY-MM-DD format
        end_date: Optional end date string in YYYY-MM-DD format

    Returns:
        List of filtered directory names
    """
    if not (start_date or end_date):
        return date_dirs

    filtered_dirs = []
    for dir_name in date_dirs:
        dir_date = datetime.strptime(dir_name, '%Y-%m-%d').date()

        if start_date:
            start = datetime.strptime(start_date, '%Y-%m-%d').date()
            if dir_date < start:
                continue

        if end_date:
            end = datetime.strptime(end_date, '%Y-%m-%d').date()
            if dir_date > end:
                continue

        filtered_dirs.append(dir_name)

    return filtered_dirs

class BenchmarkManager:
    """Class to manage benchmark data and calculations"""

    def __init__(self, benchmark_ticker: str, base_dir: str):
        self.benchmark_ticker = benchmark_ticker
        self.benchmark_dir = os.path.join(base_dir, 'Benchmarks')
        self.benchmark_file = os.path.join(self.benchmark_dir, f'{benchmark_ticker}_Benchmark.csv')
        self.benchmark_data = None
        self.setup_benchmark_directory()

    def setup_benchmark_directory(self):
        """Create Benchmarks directory if it doesn't exist"""
        os.makedirs(self.benchmark_dir, exist_ok=True)

    def load_or_download_benchmark(self):
        """Load existing benchmark data or download if not available"""
        if os.path.exists(self.benchmark_file):
            self.benchmark_data = pd.read_csv(self.benchmark_file)
            self.benchmark_data['Date'] = pd.to_datetime(self.benchmark_data['Date']).dt.date
        else:
            self._download_benchmark_data()

    def _download_benchmark_data(self):
        """Download benchmark data using yfinance"""
        print(f"Downloading benchmark data for {self.benchmark_ticker}...")
        benchmark = yf.Ticker(self.benchmark_ticker)
        data = benchmark.history(period='10y', interval='1d')

        # Reset index to make Date a column and strip time component
        data.reset_index(inplace=True)
        data['Date'] = pd.to_datetime(data['Date']).dt.date

        # Save to CSV
        data.to_csv(self.benchmark_file, index=False)
        self.benchmark_data = data
        print(f"Benchmark data saved to {self.benchmark_file}")

    def calculate_benchmark_returns(self, start_date, end_date):
        """Calculate benchmark returns for the given period"""
        if self.benchmark_data is None:
            self.load_or_download_benchmark()

        mask = (self.benchmark_data['Date'] >= start_date) & \
               (self.benchmark_data['Date'] <= end_date)
        period_data = self.benchmark_data[mask].copy()

        if len(period_data) < 2:
            return 0.0

        initial_price = period_data.iloc[0]['Open']
        final_price = period_data.iloc[-1]['Close']
        return ((final_price - initial_price) / initial_price) * 100

    def calculate_benchmark_metrics(self, start_date, end_date):
        """Calculate comprehensive benchmark metrics"""
        if self.benchmark_data is None:
            self.load_or_download_benchmark()

        mask = (self.benchmark_data['Date'] >= start_date) & \
               (self.benchmark_data['Date'] <= end_date)
        period_data = self.benchmark_data[mask].copy()

        if len(period_data) < 2:
            return None

        # Calculate daily returns
        period_data['Daily_Return'] = period_data['Close'].pct_change()

        # Basic metrics
        total_return = (period_data.iloc[-1]['Close'] / period_data.iloc[0]['Open'] - 1) * 100
        trading_days = len(period_data)
        years = trading_days / 252  # Approximate number of trading days in a year

        # Annualized return
        annualized_return = ((1 + total_return/100) ** (1/years) - 1) * 100 if years > 0 else 0

        # Risk metrics
        daily_volatility = period_data['Daily_Return'].std()
        annualized_volatility = daily_volatility * (252 ** 0.5) * 100  # Convert to percentage

        # Calculate drawdown
        rolling_max = period_data['Close'].expanding().max()
        drawdowns = (period_data['Close'] - rolling_max) / rolling_max * 100
        max_drawdown = abs(drawdowns.min())

        # Sharpe ratio (assuming risk-free rate of 2%)
        risk_free_rate = 0.02
        excess_return = annualized_return/100 - risk_free_rate
        sharpe_ratio = excess_return / (annualized_volatility/100) if annualized_volatility > 0 else 0

        # Sortino ratio
        negative_returns = period_data['Daily_Return'][period_data['Daily_Return'] < 0]
        downside_std = negative_returns.std() * (252 ** 0.5) if len(negative_returns) > 0 else 0
        sortino_ratio = excess_return / downside_std if downside_std > 0 else float('inf')

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': annualized_volatility,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'years': years
        }

    def get_benchmark_prices(self, start_date, end_date):
        """Get benchmark price data for the given period"""
        if self.benchmark_data is None:
            self.load_or_download_benchmark()

        mask = (self.benchmark_data['Date'] >= start_date) & \
               (self.benchmark_data['Date'] <= end_date)
        return self.benchmark_data[mask]

def calculate_strategy_benchmark_returns(trades, benchmark_manager):
    """
    Calculate daily benchmark returns for the entire strategy period.

    Args:
        trades: List of Trade objects from the strategy
        benchmark_manager: BenchmarkManager instance

    Returns:
        Dict mapping dates to cumulative returns
    """
    # Find strategy start and end dates from trades
    start_date = min(trade.entry_datetime.date() for trade in trades)
    end_date = max(trade.exit_datetime.date() for trade in trades)

    # Get benchmark data for the full period
    benchmark_data = benchmark_manager.get_benchmark_prices(start_date, end_date)

    if benchmark_data.empty:
        return {}

    # Calculate daily returns - using proper pandas methods to avoid warning
    initial_price = benchmark_data['Open'].iloc[0]
    benchmark_data = benchmark_data.copy()  # Create a copy to avoid warning
    benchmark_data.loc[:, 'Return'] = ((benchmark_data['Close'] - initial_price) / initial_price * 100)

    # Create dictionary mapping dates to cumulative returns
    return {row['Date']: row['Return'] for _, row in benchmark_data.iterrows()}


def calculate_performance_metrics(portfolio_returns, trades):
    """
    Calculate comprehensive performance metrics for the trading strategy.

    Args:
        portfolio_returns (dict): Dictionary of date to return percentage
        trades (list): List of Trade objects

    Returns:
        dict: Dictionary of calculated metrics
    """
    returns_series = pd.Series(list(portfolio_returns.values())) / 100

    # Time period calculations
    dates = list(portfolio_returns.keys())
    start_date = datetime.strptime(min(dates), '%Y-%m-%d')
    end_date = datetime.strptime(max(dates), '%Y-%m-%d')
    years = (end_date - start_date).days / 365.25
    trading_periods = len(returns_series)

    # Basic return metrics
    cumulative_return = (1 + returns_series).prod() - 1
    annualized_return = (1 + cumulative_return) ** (1/years) - 1 if years > 0 else 0

    # Risk metrics
    trading_periods_per_year = 26  # Approximate number of bi-weekly periods
    volatility = returns_series.std() * np.sqrt(trading_periods_per_year)
    risk_free_rate = 0.02  # 2% annual risk-free rate
    excess_return = annualized_return - risk_free_rate
    sharpe_ratio = excess_return / volatility if volatility > 0 else 0

    # Drawdown analysis
    cumulative_returns = (1 + returns_series).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdowns = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = abs(drawdowns.min())
    calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0

    # Trade statistics
    winning_trades = [t for t in trades if t.get_pnl()[0] > 0]
    losing_trades = [t for t in trades if t.get_pnl()[0] <= 0]

    win_rate = len(winning_trades) / len(trades) if trades else 0

    if winning_trades:
        avg_win = np.mean([t.get_pnl()[1] for t in winning_trades])
        max_win = max([t.get_pnl()[1] for t in winning_trades])
    else:
        avg_win = 0
        max_win = 0

    if losing_trades:
        avg_loss = np.mean([t.get_pnl()[1] for t in losing_trades])
        max_loss = min([t.get_pnl()[1] for t in losing_trades])
    else:
        avg_loss = 0
        max_loss = 0

    profit_factor = abs(sum(t.get_pnl()[0] for t in winning_trades) /
                       sum(t.get_pnl()[0] for t in losing_trades)) if losing_trades else float('inf')

    # Recovery factor and MAR ratio
    recovery_factor = cumulative_return / max_drawdown if max_drawdown > 0 else float('inf')
    mar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else float('inf')

    # Sortino ratio (using negative returns only for denominator)
    negative_returns = returns_series[returns_series < 0]
    downside_std = np.sqrt(trading_periods_per_year) * np.std(negative_returns) if len(negative_returns) > 0 else 0
    sortino_ratio = (annualized_return - risk_free_rate) / downside_std if downside_std > 0 else float('inf')

    return {
        # Return metrics
        'cumulative_return': cumulative_return * 100,  # Convert to percentage
        'annualized_return': annualized_return * 100,  # Convert to percentage
        'trading_periods': trading_periods,
        'years': years,

        # Risk metrics
        'volatility': volatility * 100,  # Convert to percentage
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'max_drawdown': max_drawdown * 100,  # Convert to percentage

        # Trade metrics
        'total_trades': len(trades),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'win_rate': win_rate * 100,  # Convert to percentage
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'max_win': max_win,
        'max_loss': max_loss,
        'profit_factor': profit_factor,

        # Additional ratios
        'recovery_factor': recovery_factor,
        'mar_ratio': mar_ratio
    }


def setup_logging():
    """Configure logging for both trade details and summary."""
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

    trade_logger.propagate = False
    summary_logger.propagate = False

    return trade_logger, summary_logger

def load_purchase_dates(base_dir: str) -> pd.DataFrame:
    """Load purchase dates CSV and convert dates/times to appropriate formats."""
    purchase_dates_file = os.path.join(base_dir, 'JD Trading Record.csv')
    df = pd.read_csv(purchase_dates_file)

    # Convert dates to datetime.date objects
    df['Purchase Date'] = pd.to_datetime(df['Purchase Date'], format='%Y-%m-%d').dt.date

    # Convert times to datetime.time objects - assuming format HH:MM or H:MM
    # First ensure consistent format
    df['Purchase Time'] = df['Purchase Time'].str.zfill(5)  # Pad single-digit hours
    df['Purchase Time'] = pd.to_datetime(df['Purchase Time'], format='%H:%M').dt.time

    # Set Trading Period as index for faster lookups
    df.set_index('Trading Period', inplace=True)

    return df

def load_market_signals(base_dir):
    """Load and process all market signals from gauge files with date indexing."""
    gauges_dir = os.path.join(base_dir, 'Gauges')

    def process_gauge_file(file_path, columns_map=None):
        df = pd.read_csv(file_path)

        df['DateTime'] = pd.to_datetime(df['DateTime'], errors = 'coerce')
        df = df.dropna(subset=['DateTime'])

        df['DateTime'] = pd.to_datetime(df['DateTime'])

        if columns_map:
            df = df.rename(columns=columns_map)
        # Create date index for faster lookups
        df['date'] = df['DateTime'].dt.date
        df.set_index('date', inplace=True)
        return df

    # Load MG Gauge with optimized structure
    mg_df = process_gauge_file(
        os.path.join(gauges_dir, 'daily-market-momentum-ga.csv'),
        {'Positive Momentum Gauge®': 'Pos MG', 'Negative Momentum Gauge®': 'Neg MG'}
    )

    # Load SP500 Gauge with optimized structure
    sp500_df = process_gauge_file(
        os.path.join(gauges_dir, 'daily-sp-500-momentum-ga.csv'),
        {'Positive Momentum Gauge®': 'Pos SP500', 'Negative Momentum Gauge®': 'Neg SP500'}
    )

    # Load Sector Gauges with optimized structure
    sector_gauges = {}
    for filename in os.listdir(gauges_dir):
        if filename not in ['daily-market-momentum-ga.csv', 'daily-sp-500-momentum-ga.csv',
                            'weekly-market-momentum-g.csv', 'weekly-sp-500-momentum-g.csv']:
            sector_name = os.path.splitext(filename)[0]
            sector_gauges[sector_name] = process_gauge_file(
                os.path.join(gauges_dir, filename),
                {
                    'Positive Momentum Gauge®': f'Pos {sector_name}',
                    'Negative Momentum Gauge®': f'Neg {sector_name}'
                }
            )

    return mg_df, sp500_df, sector_gauges

def load_sector_info(base_dir):
    """Load sector information for tickers with caching."""
    if not hasattr(load_sector_info, 'cache'):
        load_sector_info.cache = None

    if load_sector_info.cache is None:
        sector_file = os.path.join(base_dir, 'Ticker_Download_Log', 'ticker_sectors.csv')
        sector_df = pd.read_csv(sector_file)
        load_sector_info.cache = dict(zip(sector_df['Ticker'], sector_df['Sector']))

    return load_sector_info.cache

def load_stock_data(file_path):
    """Load and process stock price data with caching."""
    if not hasattr(load_stock_data, 'cache'):
        load_stock_data.cache = {}

    if file_path not in load_stock_data.cache:
        df = pd.read_csv(file_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        load_stock_data.cache[file_path] = df

    return load_stock_data.cache[file_path]

def get_entry_time(baseline_time: datetime.time, offset_minutes: int) -> datetime.time:
    """Calculate entry time by adding offset to baseline time."""
    base_dt = datetime.combine(datetime.today(), baseline_time)
    offset_dt = base_dt + timedelta(minutes=offset_minutes)
    return offset_dt.time()

def get_trading_days(start_date, end_date):
    """Get all trading days for the given date range using caching."""
    cache_key = f"{start_date}_{end_date}"
    if not hasattr(get_trading_days, 'cache'):
        get_trading_days.cache = {}

    if cache_key not in get_trading_days.cache:
        nyse = mcal.get_calendar('NYSE')
        trading_days = nyse.valid_days(start_date=start_date, end_date=end_date)
        get_trading_days.cache[cache_key] = set(d.date() for d in trading_days)

    return get_trading_days.cache[cache_key]

def is_trading_day(date, trading_days):
    """Check if given date is a trading day."""
    if isinstance(date, datetime):
        date = date.date()
    return date in trading_days

def find_prior_trading_day(date: datetime.date, trading_days: set) -> datetime.date:
    """Find the most recent trading day before the given date."""
    current = date - timedelta(days=1)
    while current not in trading_days:
        current -= timedelta(days=1)
    return current

def find_entry_day(start_date, end_date, trading_days):
    """Find first trading day in the date range."""
    current = start_date
    while current <= end_date:
        if current in trading_days:
            return current
        current += timedelta(days=1)
    return None

def get_next_trading_day(date, trading_days):
    """Get next trading day after given date."""
    if isinstance(date, datetime):
        date = date.date()
    next_date = date + timedelta(days=1)
    while next_date not in trading_days:
        next_date += timedelta(days=1)
    return next_date

def find_second_friday_exit(start_date: datetime.date, end_date: datetime.date, trading_days: set) -> datetime.date:
    """
    Find the last valid trading day before or on the second calendar Friday.

    Args:
        start_date: Start date of the trading period
        end_date: End date of the trading period
        trading_days: Set of valid trading days

    Returns:
        datetime.date: Last valid trading day before or on the second calendar Friday
    """
    # Initialize counter for Fridays
    friday_count = 0
    current = start_date

    # Find the second calendar Friday
    while current <= end_date:
        if current.weekday() == 4:  # Friday (0=Monday, 4=Friday)
            friday_count += 1
            if friday_count == 2:
                # Found second Friday, now find the last trading day before or on this date
                check_date = current
                while check_date not in trading_days and check_date >= start_date:
                    check_date -= timedelta(days=1)

                if check_date >= start_date:
                    return check_date
                else:
                    # If we couldn't find a valid trading day before the second Friday,
                    # return None to indicate no valid exit day
                    return None
        current += timedelta(days=1)

    # If we couldn't find a second Friday in the date range
    return None

def check_mg_signal_entry(date, mg_df, params=None):
    """Check if MG signals are True for entry on the given date."""
    if params is None:
        return True

    if isinstance(date, datetime):
        date = date.date()

    try:
        signal_data = mg_df.loc[date]

        # Split conditions based on entry parameters
        pos_condition = True
        neg_40_condition = True

        if params.entry_use_mg_pos_signal:
            pos_condition = signal_data['Pos MG'] > signal_data['Neg MG']

        if params.entry_use_mg_40_signal:
            neg_40_condition = signal_data['Neg MG'] < 40

        return pos_condition and neg_40_condition
    except KeyError:
        return False

def check_mg_signal_exit(date, mg_df, params=None):
    """Check if MG signals are True for exit on the given date."""
    if params is None:
        return True

    if isinstance(date, datetime):
        date = date.date()

    try:
        signal_data = mg_df.loc[date]

        # Split conditions based on exit parameters
        pos_condition = True
        neg_40_condition = True

        if params.exit_use_mg_pos_signal:
            pos_condition = signal_data['Pos MG'] > signal_data['Neg MG']

        if params.exit_use_mg_40_signal:
            neg_40_condition = signal_data['Neg MG'] < 40

        return pos_condition and neg_40_condition
    except KeyError:
        return False

def check_sp500_signal_entry(date, sp500_df, params=None):
    """Check if SP500 signal is True for entry on the given date."""
    if params is None or not params.entry_use_sp500_signal:
        return True

    if isinstance(date, datetime):
        date = date.date()

    try:
        signal_data = sp500_df.loc[date]
        return signal_data['Pos SP500'] > signal_data['Neg SP500']
    except KeyError:
        return False

def check_sp500_signal_exit(date, sp500_df, params=None):
    """Check if SP500 signal is True for exit on the given date."""
    if params is None or not params.exit_use_sp500_signal:
        return True

    if isinstance(date, datetime):
        date = date.date()

    try:
        signal_data = sp500_df.loc[date]
        return signal_data['Pos SP500'] > signal_data['Neg SP500']
    except KeyError:
        return False

def check_sector_signal_exit(date, sector, sector_gauges, params=None):
    """Check if sector signal is True for exit on the given date."""
    if params is None or not params.exit_use_sector_signals:
        return True

    if isinstance(date, datetime):
        date = date.date()

    sector_df = sector_gauges.get(sector)
    if sector_df is None:
        return False

    try:
        signal_data = sector_df.loc[date]
        pos_col = f'Pos {sector}'
        neg_col = f'Neg {sector}'
        return signal_data[pos_col] > signal_data[neg_col]
    except KeyError:
        return False

def plot_benchmark_period(benchmark_manager, portfolio_start_date, portfolio_end_date):
    """Create a daily candlestick plot for the benchmark over the full portfolio period."""
    # Get benchmark data for the full period
    benchmark_data = benchmark_manager.get_benchmark_prices(portfolio_start_date, portfolio_end_date)

    if benchmark_data.empty:
        raise ValueError("No benchmark data available for the specified date range")

    # Convert benchmark data to format required for plotting
    plot_data = pd.DataFrame({
        'Open': benchmark_data['Open'].astype(float),
        'High': benchmark_data['High'].astype(float),
        'Low': benchmark_data['Low'].astype(float),
        'Close': benchmark_data['Close'].astype(float),
        'Volume': benchmark_data['Volume'].astype(float)
    })

    # Set datetime index
    plot_data.index = pd.to_datetime(benchmark_data['Date'])

    # Remove any rows with NaN values
    plot_data = plot_data.dropna()

    if plot_data.empty:
        raise ValueError("No valid benchmark data after cleaning")

    # Calculate cumulative return
    total_return = (
        (plot_data['Close'].iloc[-1] - plot_data['Open'].iloc[0]) /
        plot_data['Open'].iloc[0] * 100
    )

    # Create figure and axes
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.1)
    ax1 = fig.add_subplot(gs[0])
    ax_volume = fig.add_subplot(gs[1], sharex=ax1)

    # Plot candlesticks
    width = 0.6
    width2 = 0.05

    up = plot_data[plot_data.Close >= plot_data.Open]
    down = plot_data[plot_data.Close < plot_data.Open]

    # Plot up candles
    ax1.bar(up.index, up.Close-up.Open, width, bottom=up.Open, color='g', alpha=0.5)
    ax1.bar(up.index, up.High-up.Close, width2, bottom=up.Close, color='g')
    ax1.bar(up.index, up.Low-up.Open, width2, bottom=up.Open, color='g')

    # Plot down candles
    ax1.bar(down.index, down.Close-down.Open, width, bottom=down.Open, color='r', alpha=0.5)
    ax1.bar(down.index, down.High-down.Open, width2, bottom=down.Open, color='r')
    ax1.bar(down.index, down.Low-down.Close, width2, bottom=down.Close, color='r')

    # Plot volume
    ax_volume.bar(up.index, up.Volume, width, color='g', alpha=0.5)
    ax_volume.bar(down.index, down.Volume, width, color='r', alpha=0.5)

    # Create secondary y-axis for percentage changes
    ax2 = ax1.twinx()

    # Calculate percentage changes for secondary axis using evenly spaced points
    y1_min, y1_max = ax1.get_ylim()
    initial_price = plot_data['Open'].iloc[0]

    # Create evenly spaced price points
    y_axis_values = np.linspace(y1_min, y1_max, num=10)

    # Calculate percentage changes relative to initial price
    pct_changes = ((y_axis_values - initial_price) / initial_price) * 100

    # Set the secondary y-axis limits and ticks
    ax2.set_ylim(y1_min, y1_max)
    ax2.set_yticks(y_axis_values)
    ax2.set_yticklabels([f'{pct:.1f}%' for pct in pct_changes])

    # Add zero line (initial price level) if it falls within the axis range
    if y1_min < initial_price < y1_max:
        ax1.axhline(y=initial_price, color='gray', linestyle='--', alpha=0.5)

    # Add annotation for total return
    ax1.annotate(
        f'Total Return: {total_return:.1f}%',
        xy=(0.02, 0.95),
        xycoords='axes fraction',
        bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8),
        fontsize=10
    )

    # Configure grid and labels
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.set_ylabel(f'{BENCHMARK_TICKER} Price ($)')
    ax2.set_ylabel('Price Change (%)')
    ax_volume.set_ylabel('Volume')

    # Format x-axis
    ax_volume.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_volume.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax_volume.get_xticklabels(), rotation=45, ha='right')

    # Set title
    ax1.set_title(f'{BENCHMARK_TICKER} Performance: {portfolio_start_date} to {portfolio_end_date}')

    # Adjust layout
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.15, top=0.9)

    return fig


def plot_trade_candlestick(stock_data, trade, date_id):
    """Create candlestick plot for a single trade."""
    # Extend the plot range
    start_date = (trade.entry_datetime - timedelta(days=1)).date()
    end_date = min(trade.exit_datetime.date() + timedelta(days=3), stock_data.index[-1].date())
    plot_data = stock_data[start_date:end_date].copy()

    if plot_data.empty:
        raise ValueError("No data available for the specified date range")

    # Create entry/exit markers
    entry_data = pd.Series(np.nan, index=plot_data.index)
    exit_data = pd.Series(np.nan, index=plot_data.index)

    # Set values for entry and exit points
    entry_idx = plot_data.index[plot_data.index.get_indexer([trade.entry_datetime], method='nearest')[0]]
    exit_idx = plot_data.index[plot_data.index.get_indexer([trade.exit_datetime], method='nearest')[0]]

    entry_data[entry_idx] = trade.entry_price
    exit_data[exit_idx] = trade.exit_price

    # Create the addplots
    additional_plots = [
        mpf.make_addplot(entry_data, type='scatter', marker='^', markersize=200, color='g'),
        mpf.make_addplot(exit_data, type='scatter', marker='v', markersize=200, color='r'),
    ]

    # Create the plot with mplfinance
    fig, axlist = mpf.plot(
        plot_data,
        type='hollow_and_filled',
        style='yahoo',
        figsize=(12, 6),
        volume=False,
        addplot=additional_plots,
        warn_too_much_data=1000000000,
        title=f'{trade.ticker} - {date_id}',
        returnfig=True
    )

    # Get the main price axis
    ax1 = axlist[0]

    # Add horizontal lines at entry and exit prices
    ax1.axhline(y=trade.entry_price, color='g', linestyle='-', alpha=0.3)
    ax1.axhline(y=trade.exit_price, color='r', linestyle='-', alpha=0.3)

    # Create secondary y-axis
    ax2 = ax1.twinx()

    # Remove primary y-axis label and set secondary y-axis label
    ax1.set_ylabel('')
    ax2.set_ylabel('Price Change (%)')

    # Calculate percentage changes for secondary axis
    y1_min, y1_max = ax1.get_ylim()

    # Create evenly spaced price points
    y_axis_values = np.linspace(y1_min, y1_max, num=15)

    # Calculate percentage changes relative to entry price
    pct_changes = ((y_axis_values - trade.entry_price) / trade.entry_price) * 100

    # Set the secondary y-axis limits and ticks
    ax2.set_ylim(y1_min, y1_max)
    ax2.set_yticks(y_axis_values)
    ax2.set_yticklabels([f'{pct:.1f}%' for pct in pct_changes])

    # Add zero line if it falls within the axis range
    if y1_min < trade.entry_price < y1_max:
        ax1.axhline(y=trade.entry_price, color='gray', linestyle='--', alpha=0.5)

    # Add annotations
    entry_x = len(plot_data.loc[:entry_idx]) - 1
    exit_x = len(plot_data.loc[:exit_idx]) - 1

    ax1.annotate(f'Entry: ${trade.entry_price:.2f}',
                xy=(entry_x, trade.entry_price),
                xytext=(10, 10),
                textcoords='offset points',
                zorder=10)

    ax1.annotate(f'Exit: ${trade.exit_price:.2f}\n{trade.exit_reason}',
                xy=(exit_x, trade.exit_price),
                xytext=(10, -20),
                textcoords='offset points',
                zorder=10)

    # Configure grid
    ax1.grid(False)
    ax2.grid(True)

    # Adjust subplot parameters manually with more space on the right for cursor coordinates
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.15, top=0.9)

    return fig


def plot_portfolio_returns(portfolio_returns, benchmark_returns=None):
    """
    Create bar plot of portfolio returns by date with benchmark comparison.

    Args:
        portfolio_returns: Dict mapping date IDs to portfolio returns
        benchmark_returns: Dict mapping dates to benchmark cumulative returns
    """
    dates = sorted(portfolio_returns.keys())
    instantaneous_returns = [portfolio_returns[date] for date in dates]

    # Calculate cumulative returns for portfolio
    returns_series = pd.Series(instantaneous_returns) / 100
    cumulative_returns = (1 + returns_series).cumprod()
    running_totals = cumulative_returns * 100 - 100

    # Calculate drawdowns
    rolling_max = cumulative_returns.expanding().max()
    drawdowns = (cumulative_returns - rolling_max) / rolling_max * 100

    # Find the point of maximum drawdown
    max_drawdown_idx = drawdowns.idxmin()
    max_drawdown = drawdowns.min()

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(15, 7))
    x = np.arange(len(dates))
    width = 0.35

    # Plot instantaneous returns
    inst_colors = ['gray' if val >= 0 else 'gray' for val in instantaneous_returns]
    ax.bar(x + width/2, instantaneous_returns, width, label='Period Returns', color=inst_colors)

    # Plot cumulative returns
    cum_colors = ['g' if val >= 0 else 'r' for val in running_totals]
    ax.bar(x - width/2, running_totals, width, label='Cumulative Returns', color=cum_colors)

    # Add benchmark returns if provided
    if benchmark_returns is not None:
        benchmark_cumulative = []
        for date_id in dates:
            # Convert date_id to datetime.date
            strategy_date = datetime.strptime(date_id, '%Y-%m-%d').date()

            # Find the closest benchmark date
            if strategy_date in benchmark_returns:
                benchmark_return = benchmark_returns[strategy_date]
            else:
                closest_date = min(benchmark_returns.keys(),
                                 key=lambda x: abs((x - strategy_date).days))
                benchmark_return = benchmark_returns[closest_date]

            benchmark_cumulative.append(benchmark_return)

        ax.plot(x, benchmark_cumulative, 'b--', label=f'{BENCHMARK_TICKER} Returns', linewidth=2)

    # Mark maximum drawdown point
    if not np.isnan(max_drawdown):
        ax.annotate(
            f'Max Drawdown: {abs(max_drawdown):.1f}%\n'
            f'Date: {dates[max_drawdown_idx]}',
            xy=(max_drawdown_idx - width/2, running_totals[max_drawdown_idx]),
            xytext=(30, -30),
            textcoords='offset points',
            ha='left',
            va='top',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            arrowprops=dict(
                arrowstyle='->',
                connectionstyle='arc3,rad=0.2',
                color='red'
            )
        )

    # Configure axis
    ax.set_xlabel('Date ID')
    ax.set_ylabel('Returns (%)')
    ax.set_title('Portfolio Returns by Period')
    ax.set_xticks(x)
    ax.set_xticklabels(dates, rotation=45)
    ax.legend()

    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    return fig

def plot_annual_returns(portfolio_returns, benchmark_returns=None):
    """Create bar plot of portfolio returns by year with benchmark comparison."""
    # Convert dates to years and aggregate returns
    annual_returns = {}
    for date_id, returns in portfolio_returns.items():
        # Handle both string dates and datetime.date objects
        if isinstance(date_id, str):
            year = datetime.strptime(date_id, '%Y-%m-%d').year
        else:
            year = date_id.year
        annual_returns[year] = annual_returns.get(year, 0) + returns

    # Calculate benchmark annual returns if provided
    if benchmark_returns is not None:
        benchmark_annual = {}
        for date_id, returns in benchmark_returns.items():
            # Benchmark dates should already be datetime.date objects
            year = date_id.year
            benchmark_annual[year] = benchmark_annual.get(year, 0) + returns

    years = sorted(annual_returns.keys())
    cumulative_returns = []
    instantaneous_returns = []
    running_total = 0

    for year in years:
        running_total += annual_returns[year]
        cumulative_returns.append(running_total)
        instantaneous_returns.append(annual_returns[year])

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(15, 7))
    x = np.arange(len(years))
    width = 0.35

    # Plot instantaneous returns
    inst_colors = ['gray' if val >= 0 else 'gray' for val in instantaneous_returns]
    ax.bar(x + width/2, instantaneous_returns, width, label='Annual Returns', color=inst_colors)

    # Plot cumulative returns
    cum_colors = ['g' if val >= 0 else 'r' for val in cumulative_returns]
    ax.bar(x - width/2, cumulative_returns, width, label='Cumulative Returns', color=cum_colors)

    # Add benchmark returns if provided
    if benchmark_returns is not None:
        benchmark_cumulative = []
        running_total = 0
        for year in years:
            # Find the last day of each year in the benchmark data
            year_dates = [date for date in benchmark_returns.keys() if date.year == year]
            if year_dates:
                last_date = max(year_dates)
                running_total = benchmark_returns[last_date]  # Use the cumulative return directly
            benchmark_cumulative.append(running_total)
        ax.plot(x, benchmark_cumulative, 'b--', label=f'{BENCHMARK_TICKER} Returns', linewidth=2)

    # Configure axis
    ax.set_xlabel('Year')
    ax.set_ylabel('Returns (%)')
    ax.set_title('Portfolio Returns by Year')
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.legend()

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)

    # Adjust layout
    plt.tight_layout()

    return fig


class Trade:
    """Class to represent and manage individual trades."""

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
        """Close the trade with exit information."""
        self.exit_datetime = exit_datetime
        self.exit_price = exit_price
        self.exit_reason = reason

    def get_pnl(self):
        """Calculate profit/loss in dollars and percentage."""
        if self.exit_price is None:
            return 0, 0
        dollar_pnl = (self.exit_price - self.entry_price) * self.shares
        percent_pnl = (self.exit_price / self.entry_price - 1) * 100
        return dollar_pnl, percent_pnl

@dataclass
class BacktestResults:
    total_gain: float
    max_drawdown: float
    win_rate: float
    metrics: Dict
    portfolio_returns: Dict
    trades: List[Trade]
    benchmark_metrics: Dict  # Added for benchmark comparison
    benchmark_returns: Dict  # Added for benchmark comparison

    def to_dict(self) -> Dict:
        """Convert BacktestResults to dictionary for storage"""
        return {
            'total_gain': self.total_gain,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'metrics': self.metrics,
            'portfolio_returns': self.portfolio_returns,
            'benchmark_metrics': self.benchmark_metrics,
            'benchmark_returns': self.benchmark_returns,
            'trades': [
                {
                    'ticker': t.ticker,
                    'entry_datetime': t.entry_datetime.isoformat(),
                    'entry_price': t.entry_price,
                    'investment': t.investment,
                    'shares': t.shares,
                    'exit_datetime': t.exit_datetime.isoformat() if t.exit_datetime else None,
                    'exit_price': t.exit_price,
                    'exit_reason': t.exit_reason
                }
                for t in self.trades
            ]
        }


def process_trade(stock_data, trade, mg_df, sp500_df, sector_gauges, sector,
                 second_friday, trading_days, params=None):
    """Process a trade with updated exit logic using separate exit signals."""
    if params is None:
        params = StrategyParams()

    # Set up logging
    logger = logging.getLogger('trade_logger')

    # Pre-calculate target prices
    profit_price = trade.entry_price * (1 + params.profit_target)
    loss_price = trade.entry_price * (1 - params.loss_target)

    # Create a mask for the date range we're interested in
    date_mask = (stock_data.index >= trade.entry_datetime) & (stock_data.index.date <= second_friday)
    period_data = stock_data[date_mask]

    # Group by date for faster processing
    daily_data = period_data.groupby(period_data.index.date)

    for current_date, day_data in daily_data:
        if not is_trading_day(current_date, trading_days):
            continue

        # Process normal trading days
        day_data_to_check = day_data
        is_second_friday = current_date == second_friday

        # If it's second Friday, only look at data up to EXIT_TIME
        if is_second_friday:
            day_data_to_check = day_data[day_data.index <= datetime.combine(current_date, EXIT_TIME)]
            if day_data_to_check.empty:
                continue

        # Find all potential profit and loss triggers in this day's data
        profit_triggers = day_data_to_check[day_data_to_check['high'] >= profit_price].index
        loss_triggers = day_data_to_check[day_data_to_check['low'] <= loss_price].index

        all_triggers = []

        # Process loss triggers
        if not loss_triggers.empty:
            trigger_time = loss_triggers[0]
            # Get the OHLC bar at trigger time
            trigger_bar = day_data_to_check.loc[trigger_time]

            # If the bar opens below loss target, use opening price
            if trigger_bar['open'] <= loss_price:
                logger.info(
                    f"Loss trigger for {trade.ticker} at {trigger_time}. "
                    f"Bar opens (${trigger_bar['open']:.2f}) below loss target (${loss_price:.2f}). "
                    f"Using bar's opening price."
                )
                exit_price = trigger_bar['open']
            else:
                exit_price = loss_price

            all_triggers.append((trigger_time, exit_price, "Loss Target"))

        # Process profit triggers
        if not profit_triggers.empty:
            trigger_time = profit_triggers[0]
            # Get the OHLC bar at trigger time
            trigger_bar = day_data_to_check.loc[trigger_time]

            # If the bar opens above profit target, use opening price
            if trigger_bar['open'] >= profit_price:
                logger.info(
                    f"Profit trigger for {trade.ticker} at {trigger_time}. "
                    f"Bar opens (${trigger_bar['open']:.2f}) above profit target (${profit_price:.2f}). "
                    f"Using bar's opening price."
                )
                exit_price = trigger_bar['open']
            else:
                exit_price = profit_price

            all_triggers.append((trigger_time, exit_price, "Profit Target"))

        # Check for simultaneous triggers
        if len(all_triggers) >= 2 and all_triggers[0][0] == all_triggers[1][0]:
            logger.warning(
                f"Simultaneous profit/loss triggers for {trade.ticker} at {all_triggers[0][0]}. "
                f"Entry: ${trade.entry_price:.2f}. Taking loss exit for safety."
            )
            # Filter to keep only the loss trigger
            all_triggers = [t for t in all_triggers if "Loss" in t[2]]

        # If we have any triggers, take the earliest one
        if all_triggers:
            # Sort by timestamp
            all_triggers.sort(key=lambda x: x[0])
            exit_datetime, exit_price, exit_reason = all_triggers[0]
            trade.close_trade(exit_datetime, exit_price, exit_reason)
            return trade

        # Check exit signals (at end of day)
        if (not check_mg_signal_exit(current_date, mg_df, params) or
            not check_sp500_signal_exit(current_date, sp500_df, params) or
            (params.exit_use_sector_signals and not check_sector_signal_exit(current_date, sector, sector_gauges, params))):

            next_date = get_next_trading_day(current_date, trading_days)
            next_day_data = stock_data[stock_data.index.date == next_date]

            if not next_day_data.empty:
                exit_datetime = next_day_data.index[0]
                exit_price = next_day_data['open'].iloc[0]

                reason = ("MG Signal" if not check_mg_signal_exit(current_date, mg_df, params)
                         else "SP500 Signal" if not check_sp500_signal_exit(current_date, sp500_df, params)
                         else f"{sector} Signal")

                trade.close_trade(exit_datetime, exit_price, reason)
                return trade

        # Handle second Friday exit if we reach this point
        if is_second_friday:
            exit_datetime = day_data_to_check.index[-1]
            exit_price = day_data_to_check['close'].iloc[-1]
            trade.close_trade(exit_datetime, exit_price, "Second Friday")
            return trade

    return trade

def main(params: Optional[StrategyParams] = None, base_dir: str = BASE_DIR) -> BacktestResults:
    """
    Main execution function for the backtest with modified entry logic using purchase dates CSV.

    Args:
        params: StrategyParams object containing strategy parameters
        base_dir: Base directory for data files

    Returns:
        BacktestResults object containing backtest results and metrics
    """
    if params is None:
        params = StrategyParams()  # Use defaults if no params provided

    # Initialize benchmark manager
    benchmark_manager = BenchmarkManager(BENCHMARK_TICKER, base_dir)
    benchmark_manager.load_or_download_benchmark()

    # Load purchase dates
    purchase_dates_df = load_purchase_dates(base_dir)

    # Determine if this is an optimization run
    is_optimization = params != StrategyParams()

    # Setup logging based on run type
    if is_optimization:
        # Minimal logging for optimization runs
        trade_logger = logging.getLogger('trade_logger')
        trade_logger.handlers = []
        summary_logger = logging.getLogger('summary_logger')
        summary_logger.handlers = []
    else:
        # Full logging for standalone runs
        trade_logger, summary_logger = setup_logging()

    # Load market signals and sector information
    mg_df, sp500_df, sector_gauges = load_market_signals(base_dir)
    sector_info = load_sector_info(base_dir)

    # Initialize portfolio tracking
    portfolio_value = INITIAL_PORTFOLIO
    portfolio_returns = {}
    all_trades = []

    # Get all date directories
    ticker_data_dir = os.path.join(base_dir, 'Ticker_data')
    date_dirs = sorted([d for d in os.listdir(ticker_data_dir)
                       if os.path.isdir(os.path.join(ticker_data_dir, d))])

    # Filter date directories based on date range parameters
    date_dirs = filter_date_dirs(date_dirs, params.start_date, params.end_date)

    if not date_dirs:
        raise ValueError("No trading periods found in the specified date range")

    # Find overall date range for trading calendar
    all_dates = []
    for date_id in date_dirs:
        dir_path = os.path.join(ticker_data_dir, date_id)
        stock_files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
        if stock_files:
            start_date = datetime.strptime(stock_files[0].split('_')[5], '%Y-%m-%d').date()
            end_date = datetime.strptime(stock_files[0].split('_')[6].replace('.csv', ''), '%Y-%m-%d').date()
            all_dates.extend([start_date, end_date])

    # Get trading days once for the entire date range
    trading_days = get_trading_days(min(all_dates), max(all_dates))

    # Process each date directory
    for date_id in tqdm(date_dirs, desc="Processing date ranges", disable=is_optimization):
        dir_path = os.path.join(ticker_data_dir, date_id)
        stock_files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]

        if not stock_files:
            continue

        # Look up purchase date and time from CSV
        try:
            purchase_info = purchase_dates_df.loc[date_id]
            entry_day = purchase_info['Purchase Date']
            baseline_time = purchase_info['Purchase Time']
        except KeyError:
            # Skip if no purchase date info found for this trading period
            if not is_optimization:
                print(f"Warning: No purchase date information found for trading period {date_id}")
            continue

        # Find prior trading day for signal check
        signal_date = find_prior_trading_day(entry_day, trading_days)

        # Check market signals for prior trading day
        if not (check_mg_signal_entry(signal_date, mg_df, params) and
                check_sp500_signal_entry(signal_date, sp500_df, params)):
            continue

        # Calculate actual entry time with offset
        entry_time = get_entry_time(baseline_time, params.entry_time_offset)
        entry_datetime = datetime.combine(entry_day, entry_time)

        # Parse date range from first file (for second Friday calculation)
        sample_file = stock_files[0]
        period_start_date = datetime.strptime(sample_file.split('_')[5], '%Y-%m-%d').date()
        period_end_date = datetime.strptime(sample_file.split('_')[6].replace('.csv', ''), '%Y-%m-%d').date()

        # Find second Friday exit day
        second_friday_exit = find_second_friday_exit(period_start_date, period_end_date, trading_days)

        if not second_friday_exit:
            if not is_optimization:
                print(f"Warning: Could not find second Friday exit for period {date_id}")
            continue

        # Calculate per-stock investment
        num_stocks = len(stock_files)
        per_stock_investment = INITIAL_PORTFOLIO / num_stocks

        # Process each stock
        period_trades = []
        for stock_file in stock_files:
            ticker = stock_file.split('_')[4]
            sector = sector_info.get(ticker)
            if not sector:
                continue

            # Load stock data
            stock_data = load_stock_data(os.path.join(dir_path, stock_file))

            # Get entry price at specified time
            entry_data = stock_data[stock_data.index >= entry_datetime]
            if entry_data.empty:
                if not is_optimization:
                    print(f"Warning: No data found for {ticker} at entry time {entry_datetime}")
                continue

            entry_price = (entry_data.iloc[0]['high'] +
                        entry_data.iloc[0]['low'] +
                        entry_data.iloc[0]['close']) / 3

            # Create and process trade
            trade = Trade(ticker, entry_datetime, entry_price, per_stock_investment)
            trade = process_trade(stock_data, trade, mg_df, sp500_df,
                              sector_gauges, sector, second_friday_exit,
                              trading_days, params)

            if trade.exit_price is not None:
                period_trades.append(trade)
                all_trades.append(trade)

                # Log trade details only for non-optimization runs
                if not is_optimization:
                    dollar_pnl, percent_pnl = trade.get_pnl()
                    trade_logger.info(
                        f"Ticker: {trade.ticker}, "
                        f"Investment: ${trade.investment:.2f}, "
                        f"Entry: {trade.entry_datetime}, ${trade.entry_price:.5f}, "
                        f"Exit: {trade.exit_datetime}, ${trade.exit_price:.5f}, "
                        f"Reason: {trade.exit_reason}, "
                        f"P/L: ${dollar_pnl:.2f} ({percent_pnl:.2f}%)"
                    )


        # Calculate period returns and update portfolio
        period_pnl = sum(trade.get_pnl()[0] for trade in period_trades)
        period_return_pct = (period_pnl / INITIAL_PORTFOLIO) * 100
        portfolio_returns[date_id] = period_return_pct

        portfolio_value += period_pnl

    # Calculate performance metrics
    metrics = calculate_performance_metrics(portfolio_returns, all_trades)

    # Calculate final results
    total_gain = portfolio_value - INITIAL_PORTFOLIO
    total_gain_pct = (total_gain / INITIAL_PORTFOLIO) * 100

    returns_series = pd.Series(list(portfolio_returns.values())) / 100
    cumulative_returns = (1 + returns_series).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdowns = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = abs(drawdowns.min()) * 100  # Convert to percentage

    # Calculate benchmark metrics and returns
    benchmark_metrics = None
    benchmark_returns = None

    if all_trades:  # Only calculate if we have trades
        # Calculate continuous benchmark returns for the entire strategy period
        benchmark_returns = calculate_strategy_benchmark_returns(all_trades, benchmark_manager)

        # Calculate benchmark metrics using the full period
        start_date = min(trade.entry_datetime.date() for trade in all_trades)
        end_date = max(trade.exit_datetime.date() for trade in all_trades)
        benchmark_metrics = benchmark_manager.calculate_benchmark_metrics(start_date, end_date)

    # Generate summary text for non-optimization runs
        if not is_optimization:
            summary = f"""

Strategy Parameters:
------------------
Entry Conditions:
    MG Positive Signal: {params.entry_use_mg_pos_signal}
    MG 40 Signal: {params.entry_use_mg_40_signal}
    SP500 Signal: {params.entry_use_sp500_signal}

Exit Conditions:
    MG Positive Signal: {params.exit_use_mg_pos_signal}
    MG 40 Signal: {params.exit_use_mg_40_signal}
    SP500 Signal: {params.exit_use_sp500_signal}
    Sector Signals: {params.exit_use_sector_signals}

Trade Parameters:
    Profit Target: {(params.profit_target) * 100:.1f}%
    Loss Target: {(params.loss_target) * 100:.1f}%
    Entry Time Offset: {params.entry_time_offset} minutes

Performance Metrics:
------------------
Cumulative Return: {metrics['cumulative_return']:.2f}%
Annualized Return: {metrics['annualized_return']:.2f}%
Trading Periods: {metrics['trading_periods']}
Years: {metrics['years']:.2f}

Risk Metrics:
------------------
Annual Volatility: {metrics['volatility']:.2f}%
Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
Sortino Ratio: {metrics['sortino_ratio']:.2f}
Calmar Ratio: {metrics['calmar_ratio']:.2f}
Maximum Drawdown: {metrics['max_drawdown']:.2f}%

Trade Statistics:
------------------
Total Trades: {metrics['total_trades']}
Win Rate: {metrics['win_rate']:.2f}%
Average Win: {metrics['avg_win']:.2f}%
Average Loss: {metrics['avg_loss']:.2f}%
Maximum Win: {metrics['max_win']:.2f}%
Maximum Loss: {metrics['max_loss']:.2f}%
Profit Factor: {metrics['profit_factor']:.2f}

Additional Ratios:
------------------
Recovery Factor: {metrics['recovery_factor']:.2f}
MAR Ratio: {metrics['mar_ratio']:.2f}

Performance Summary:
------------------
Starting Portfolio Value: ${INITIAL_PORTFOLIO:,.2f}
Final Portfolio Value: ${portfolio_value:,.2f}
Total Gain: ${total_gain:,.2f} ({total_gain_pct:.2f}%)

Benchmark Comparison:
------------------
Benchmark: {BENCHMARK_TICKER}
Total Return: {benchmark_metrics['total_return']:.2f}%
Annualized Return: {benchmark_metrics['annualized_return']:.2f}%
Volatility: {benchmark_metrics['volatility']:.2f}%
Max Drawdown: {benchmark_metrics['max_drawdown']:.2f}%
Sharpe Ratio: {benchmark_metrics['sharpe_ratio']:.2f}
Sortino Ratio: {benchmark_metrics['sortino_ratio']:.2f}"""

            print(summary)
            summary_logger.info(summary)

            # Generate all plots
            print("\nGenerating performance plots...")
            plot_portfolio_returns(portfolio_returns, benchmark_returns)
            plt.show()
            plot_annual_returns(portfolio_returns, benchmark_returns)
            plt.show()

            # Plot benchmark performance
            if date_dirs:
                first_date = datetime.strptime(min(date_dirs), '%Y-%m-%d').date()
                last_date = datetime.strptime(max(date_dirs), '%Y-%m-%d').date()
                try:
                    print("\nGenerating benchmark performance plot...")
                    plot_benchmark_period(benchmark_manager, first_date, last_date)
                    plt.show()
                except Exception as e:
                    print(f"Error plotting benchmark performance: {str(e)}")

            # Plot trade details
            print("\nGenerating trade detail plots...")
            sorted_trades = sorted(all_trades, key=lambda x: x.get_pnl()[1], reverse=True)
            best_trades = sorted_trades[:4]
            worst_trades = sorted_trades[-4:]

            for trade in best_trades + worst_trades:
                try:
                    for date_id in date_dirs:
                        dir_path = os.path.join(ticker_data_dir, date_id)
                        stock_files = [f for f in os.listdir(dir_path)
                                     if f.endswith('.csv') and trade.ticker in f]

                        if stock_files:
                            stock_path = os.path.join(dir_path, stock_files[0])
                            start_date_str = stock_files[0].split('_')[5]
                            file_start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()

                            if file_start_date <= trade.entry_datetime.date() <= file_start_date + timedelta(days=14):
                                stock_data = load_stock_data(stock_path)
                                plot_trade_candlestick(stock_data, trade, date_id)
                                plt.show()
                                break

                except Exception as e:
                    print(f"Error plotting {trade.ticker}: {str(e)}")

    # Return structured results
    return BacktestResults(
        total_gain=total_gain,
        max_drawdown=max_drawdown,
        win_rate=metrics['win_rate'],
        metrics=metrics,
        portfolio_returns=portfolio_returns,
        trades=all_trades,
        benchmark_metrics=benchmark_metrics,
        benchmark_returns=benchmark_returns
    )

if __name__ == "__main__":
    # When run directly, use default parameters
    results = main()