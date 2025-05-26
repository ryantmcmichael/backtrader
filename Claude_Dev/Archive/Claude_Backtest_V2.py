import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, time, timedelta
import mplfinance as mpf
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas_market_calendars as mcal

# Close all existing plots
plt.close('all')

# Configuration constants
BASE_DIR = "C:/Users/ryant/Documents/Stock_Market/Python/universe_data/VM Weekly Breakout Validation"
INITIAL_PORTFOLIO = 100000
PROFIT_TARGET = 1.20  # 20% profit
LOSS_TARGET = 0.85   # 15% loss

# Time constants
ENTRY_TIME = time(6, 35)
EXIT_TIME = time(13, 0)  # 3:00 PM

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

    return trade_logger, summary_logger

def load_market_signals(base_dir):
    """Load and process all market signals from gauge files."""
    gauges_dir = os.path.join(base_dir, 'Gauges')

    # Load MG Gauge
    mg_df = pd.read_csv(os.path.join(gauges_dir, 'daily-market-momentum-ga.csv'))
    mg_df['DateTime'] = pd.to_datetime(mg_df['DateTime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    mg_df = mg_df.dropna(subset=['DateTime'])
    mg_df = mg_df.rename(columns={
        'Positive Momentum Gauge®': 'Pos MG',
        'Negative Momentum Gauge®': 'Neg MG'
    })
    mg_df = mg_df[['DateTime', 'Pos MG', 'Neg MG']]

    # Load SP500 Gauge
    sp500_df = pd.read_csv(os.path.join(gauges_dir, 'daily-sp-500-momentum-ga.csv'))
    sp500_df['DateTime'] = pd.to_datetime(sp500_df['DateTime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    sp500_df = sp500_df.dropna(subset=['DateTime'])
    sp500_df = sp500_df.rename(columns={
        'Positive Momentum Gauge®': 'Pos SP500',
        'Negative Momentum Gauge®': 'Neg SP500'
    })
    sp500_df = sp500_df[['DateTime', 'Pos SP500', 'Neg SP500']]

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
            sector_df = sector_df[['DateTime', f'Pos {sector_name}', f'Neg {sector_name}']]
            sector_gauges[sector_name] = sector_df

    return mg_df, sp500_df, sector_gauges

def load_sector_info(base_dir):
    """Load sector information for tickers."""
    sector_file = os.path.join(base_dir, 'Ticker_Download_Log', 'ticker_sectors.csv')
    sector_df = pd.read_csv(sector_file)
    return dict(zip(sector_df['Ticker'], sector_df['Sector']))

def load_stock_data(file_path):
    """Load and process stock price data."""
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    return df

def get_entry_time(date):
    """Generate entry time at 6:35 AM"""
    base_time = datetime.combine(date, ENTRY_TIME)
    return base_time


def get_trading_days(start_date, end_date):
    """Get all trading days for the given date range."""
    nyse = mcal.get_calendar('NYSE')
    trading_days = nyse.valid_days(start_date=start_date, end_date=end_date)
    return set(d.date() for d in trading_days)

def is_trading_day(date, trading_days):
    """Check if given date is a trading day."""
    if isinstance(date, datetime):
        date = date.date()
    return date in trading_days

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

def find_second_friday_exit(start_date, end_date, trading_days):
    """Find second Friday that's a trading day."""
    current = start_date
    friday_count = 0
    while current <= end_date:
        if current.weekday() == 4 and current in trading_days:  # Friday
            friday_count += 1
            if friday_count == 2:
                return current
        current += timedelta(days=1)
    return None

def check_mg_signal(date, mg_df):
    """
    Check if MG signal is True for the given date.
    Uses signal published previous evening.
    """
    if isinstance(date, datetime):
        date = date.date()

    signal_data = mg_df[mg_df['DateTime'].dt.date == date]
    if signal_data.empty:
        return False

    return (signal_data['Pos MG'].iloc[0] > signal_data['Neg MG'].iloc[0]) and (signal_data['Neg MG'].iloc[0] < 40)

def check_sp500_signal(date, sp500_df):
    """
    Check if SP500 signal is True for the given date.
    Uses signal published previous evening.
    """
    if isinstance(date, datetime):
        date = date.date()

    signal_data = sp500_df[sp500_df['DateTime'].dt.date == date]
    if signal_data.empty:
        return False

    return signal_data['Pos SP500'].iloc[0] > signal_data['Neg SP500'].iloc[0]

def check_sector_signal(date, sector, sector_gauges):
    """
    Check if sector signal is True for the given date.
    Uses signal published previous evening.
    """
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


def process_trade(stock_data, trade, mg_df, sp500_df, sector_gauges, sector,
                 second_friday, trading_days):
    """Process a trade according to strategy rules."""
    current_date = trade.entry_datetime.date()

    while current_date <= second_friday:
        # Check if trading day
        if not is_trading_day(current_date, trading_days):
            current_date += timedelta(days=1)
            continue

        # Get data for current day
        day_data = stock_data[stock_data.index.date == current_date].copy()

        if not day_data.empty:
            # Check profit target
            profit_hits = day_data[day_data['high'].ge(trade.entry_price * PROFIT_TARGET)]
            if not profit_hits.empty:
                exit_datetime = profit_hits.index[0]
                trade.close_trade(exit_datetime, trade.entry_price * PROFIT_TARGET, "Profit Target")
                return trade

            # Check loss target
            loss_hits = day_data[day_data['low'].le(trade.entry_price * LOSS_TARGET)]
            if not loss_hits.empty:
                exit_datetime = loss_hits.index[0]
                trade.close_trade(exit_datetime, trade.entry_price * LOSS_TARGET, "Loss Target")
                return trade

            # Check today's signals
            if (not check_mg_signal(current_date, mg_df) or
                not check_sp500_signal(current_date, sp500_df) or
                not check_sector_signal(current_date, sector, sector_gauges)):

                next_date = get_next_trading_day(current_date, trading_days)
                next_day_data = stock_data[stock_data.index.date == next_date]

                if not next_day_data.empty:
                    exit_datetime = next_day_data.index[0]
                    exit_price = next_day_data['open'].iloc[0]

                    # Determine exit reason
                    if not check_mg_signal(current_date, mg_df):
                        reason = "MG Signal"
                    elif not check_sp500_signal(current_date, sp500_df):
                        reason = "SP500 Signal"
                    else:
                        reason = f"{sector} Signal"

                    trade.close_trade(exit_datetime, exit_price, reason)
                    return trade

            # Check if we've reached second Friday
            if current_date == second_friday:
                exit_time = datetime.combine(current_date, EXIT_TIME)
                exit_data = day_data[day_data.index <= exit_time]
                if not exit_data.empty:
                    exit_datetime = exit_data.index[-1]
                    exit_price = exit_data['close'].iloc[-1]
                    trade.close_trade(exit_datetime, exit_price, "Second Friday")
                    return trade

        current_date += timedelta(days=1)

    return trade

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
    y_axis_values = ax1.get_yticks()
    pct_changes = ((y_axis_values - trade.entry_price) / trade.entry_price) * 100
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_yticks(y_axis_values)
    ax2.set_yticklabels([f'{pct:.1f}%' for pct in pct_changes])

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

def plot_portfolio_returns(portfolio_returns):
    """Create bar plot of portfolio returns by date."""
    dates = sorted(portfolio_returns.keys())
    cumulative_returns = []
    instantaneous_returns = []
    running_total = 0

    for date in dates:
        running_total += portfolio_returns[date]
        cumulative_returns.append(running_total)
        instantaneous_returns.append(portfolio_returns[date])

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(15, 7))
    x = np.arange(len(dates))
    width = 0.35

    # Plot cumulative returns
    cum_colors = ['g' if val >= 0 else 'r' for val in cumulative_returns]
    ax.bar(x - width/2, cumulative_returns, width, label='Cumulative Returns', color=cum_colors)

    # Plot instantaneous returns
    inst_colors = ['cyan' if val >= 0 else 'magenta' for val in instantaneous_returns]
    ax.bar(x + width/2, instantaneous_returns, width, label='Period Returns', color=inst_colors)

    # Configure axis
    ax.set_xlabel('Date ID')
    ax.set_ylabel('Returns (%)')
    ax.set_title('Portfolio Returns by Period')
    ax.set_xticks(x)
    ax.set_xticklabels(dates, rotation=45)
    ax.legend()

    return fig

def plot_annual_returns(portfolio_returns):
    """Create bar plot of portfolio returns by year."""
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

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(15, 7))
    x = np.arange(len(years))
    width = 0.35

    # Plot cumulative returns
    cum_colors = ['g' if val >= 0 else 'r' for val in cumulative_returns]
    ax.bar(x - width/2, cumulative_returns, width, label='Cumulative Returns', color=cum_colors)

    # Plot instantaneous returns
    inst_colors = ['cyan' if val >= 0 else 'magenta' for val in instantaneous_returns]
    ax.bar(x + width/2, instantaneous_returns, width, label='Annual Returns', color=inst_colors)

    # Configure axis
    ax.set_xlabel('Year')
    ax.set_ylabel('Returns (%)')
    ax.set_title('Portfolio Returns by Year')
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.legend()

    return fig

def main():
    """Main execution function for the backtest."""
    # Setup logging
    trade_logger, summary_logger = setup_logging()

    # Load market signals and sector information
    mg_df, sp500_df, sector_gauges = load_market_signals(BASE_DIR)
    sector_info = load_sector_info(BASE_DIR)

    # Initialize portfolio tracking
    portfolio_value = INITIAL_PORTFOLIO
    portfolio_returns = {}
    all_trades = []
    max_drawdown = 0

    # Get all date directories
    ticker_data_dir = os.path.join(BASE_DIR, 'Ticker_data')
    date_dirs = sorted([d for d in os.listdir(ticker_data_dir)
                       if os.path.isdir(os.path.join(ticker_data_dir, d))])

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
    for date_id in tqdm(date_dirs, desc="Processing date ranges"):
        dir_path = os.path.join(ticker_data_dir, date_id)
        stock_files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]

        if not stock_files:
            continue

        # Parse date range from first file
        sample_file = stock_files[0]
        start_date = datetime.strptime(sample_file.split('_')[5], '%Y-%m-%d').date()
        end_date = datetime.strptime(sample_file.split('_')[6].replace('.csv', ''), '%Y-%m-%d').date()

        # Find entry day and second Friday exit day
        entry_day = find_entry_day(start_date, end_date, trading_days)
        second_friday_exit = find_second_friday_exit(start_date, end_date, trading_days)

        if not entry_day or not second_friday_exit:
            continue

        # Check market signals for day before entry
        signal_date = entry_day - timedelta(days=1)
        if not (check_mg_signal(signal_date, mg_df) and
                check_sp500_signal(signal_date, sp500_df)):
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

            # Get entry price and time
            entry_datetime = get_entry_time(entry_day)
            entry_data = stock_data[stock_data.index >= entry_datetime]
            if entry_data.empty:
                continue

            entry_price = entry_data.iloc[0]['open']

            # Create and process trade
            trade = Trade(ticker, entry_datetime, entry_price, per_stock_investment)
            trade = process_trade(stock_data, trade, mg_df, sp500_df,
                                sector_gauges, sector, second_friday_exit, trading_days)

            if trade.exit_price is not None:
                period_trades.append(trade)
                all_trades.append(trade)

                # Log trade details
                dollar_pnl, percent_pnl = trade.get_pnl()
                trade_logger.info(
                    f"Ticker: {trade.ticker}, "
                    f"Investment: ${trade.investment:.2f}, "
                    f"Entry: {trade.entry_datetime}, ${trade.entry_price:.2f}, "
                    f"Exit: {trade.exit_datetime}, ${trade.exit_price:.2f}, "
                    f"Reason: {trade.exit_reason}, "
                    f"P/L: ${dollar_pnl:.2f} ({percent_pnl:.2f}%)"
                )

        # Calculate period returns and update portfolio
        period_pnl = sum(trade.get_pnl()[0] for trade in period_trades)
        period_return_pct = (period_pnl / INITIAL_PORTFOLIO) * 100
        portfolio_returns[date_id] = period_return_pct

        portfolio_value += period_pnl
        drawdown = max(0, INITIAL_PORTFOLIO - portfolio_value)
        max_drawdown = max(max_drawdown, drawdown)

    # Calculate performance metrics
    metrics = calculate_performance_metrics(portfolio_returns, all_trades)

    # Log and print final results
    total_gain = portfolio_value - INITIAL_PORTFOLIO
    total_gain_pct = (total_gain / INITIAL_PORTFOLIO) * 100

    summary = f"""
Performance Summary:
------------------
Starting Portfolio Value: ${INITIAL_PORTFOLIO:,.2f}
Final Portfolio Value: ${portfolio_value:,.2f}
Total Gain: ${total_gain:,.2f} ({total_gain_pct:.2f}%)

Cumulative Return: {metrics['cumulative_return']:.2f}%
Annualized Return: {metrics['annualized_return']:.2f}%
Trading Periods: {metrics['trading_periods']}
Years: {metrics['years']:.2f}

Annual Volatility: {metrics['volatility']:.2f}%
Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
Sortino Ratio: {metrics['sortino_ratio']:.2f}
Calmar Ratio: {metrics['calmar_ratio']:.2f}
Maximum Drawdown: {metrics['max_drawdown']:.2f}%

Total Trades: {metrics['total_trades']}
Win Rate: {metrics['win_rate']:.2f}%
Average Win: {metrics['avg_win']:.2f}%
Average Loss: {metrics['avg_loss']:.2f}%
Maximum Win: {metrics['max_win']:.2f}%
Maximum Loss: {metrics['max_loss']:.2f}%
Profit Factor: {metrics['profit_factor']:.2f}

Recovery Factor: {metrics['recovery_factor']:.2f}
MAR Ratio: {metrics['mar_ratio']:.2f}"""

    print(summary)
    summary_logger.info(summary)

    # Plot best and worst performers
    print("\nGenerating candlestick plots for best and worst performers...")
    sorted_trades = sorted(all_trades, key=lambda x: x.get_pnl()[1], reverse=True)
    best_trades = sorted_trades[:4]
    worst_trades = sorted_trades[-4:]

    ticker_data_dir = os.path.join(BASE_DIR, 'Ticker_data')

    for trade in best_trades + worst_trades:
        try:
            # Find the correct date directory for this trade
            for date_id in date_dirs:
                dir_path = os.path.join(ticker_data_dir, date_id)
                stock_files = [f for f in os.listdir(dir_path)
                             if f.endswith('.csv') and trade.ticker in f]

                if stock_files:
                    # Check if this file contains our trade's date range
                    stock_path = os.path.join(dir_path, stock_files[0])
                    # Parse date from filename instead of reading file
                    start_date_str = stock_files[0].split('_')[5]
                    file_start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()

                    if file_start_date <= trade.entry_datetime.date() <= file_start_date + timedelta(days=14):
                        stock_data = load_stock_data(stock_path)
                        plot_trade_candlestick(stock_data, trade, date_id)
                        plt.show()
                        break

        except Exception as e:
            print(f"Error plotting {trade.ticker}: {str(e)}")

    # Plot portfolio returns
    plot_portfolio_returns(portfolio_returns)
    plt.show()

    # Plot annual returns
    plot_annual_returns(portfolio_returns)
    plt.show()

if __name__ == "__main__":
    main()