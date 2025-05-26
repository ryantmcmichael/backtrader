import numpy as np
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import multiprocessing as mp
from itertools import product
from tqdm import tqdm
import warnings

# Suppress pandas warnings
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
pd.options.mode.chained_assignment = None

class StockSimulator:
    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)

    def generate_minute_prices(self, start_date, end_date):
        """Generate synthetic minute-level price data for one stock"""
        business_days = pd.date_range(start=start_date, end=end_date, freq='B')
        minutes_per_day = []

        for day in business_days:
            minutes = pd.date_range(
                start=day.replace(hour=6, minute=30),
                end=day.replace(hour=13, minute=0),
                freq='1min'
            )
            minutes_per_day.extend(minutes)

        index = pd.DatetimeIndex(minutes_per_day)
        n_minutes = len(index)

        # Initialize price series with random walk
        price = 100
        prices = np.zeros(n_minutes)
        prices[0] = price

        # Parameters for price evolution
        volatility = 0.001
        target_reached = False

        for i in range(1, n_minutes):
            if prices[i-1] < 70:
                drift = 0.0002  # Upward drift when price is low
            elif prices[i-1] > 130:
                drift = -0.0002  # Downward drift when price is high
            else:
                drift = 0

            # Add random component
            change = np.random.normal(drift, volatility)

            # Ensure 60% chance of hitting target before last day
            if not target_reached and i < (n_minutes - 390):  # 390 minutes in last day
                if np.random.random() < 0.001:  # Small chance each minute
                    if np.random.random() < 0.6:  # 60% chance
                        change = 0.02 if np.random.random() < 0.5 else -0.02
                        target_reached = True

            new_price = prices[i-1] * (1 + change)
            prices[i] = np.clip(new_price, 70, 130)

        # Generate OHLC data
        df = pd.DataFrame(index=index)
        df['Open'] = prices
        df['Close'] = prices

        # Add small random variations for High and Low
        df['High'] = df['Open'] * (1 + np.random.uniform(0, 0.002, len(df)))
        df['Low'] = df['Open'] * (1 - np.random.uniform(0, 0.002, len(df)))

        return df

    def generate_market_signals(self, dates):
        """Generate synthetic market signals for given dates"""
        np.random.seed(self.seed)
        n_days = len(dates)
        n_signals = 6

        # 5% chance of False for each signal
        signals = np.random.random((n_days, n_signals)) > 0.05

        signal_df = pd.DataFrame(
            signals,
            index=dates,
            columns=[f'Signal_{i+1}' for i in range(n_signals)]
        )

        return signal_df

class Backtester:
    def __init__(self, profit_target=0.07, loss_target=0.05, required_signals=None):
        self.profit_target = profit_target
        self.loss_target = loss_target
        self.required_signals = required_signals if required_signals else list(range(6))

    def run_backtest(self, price_data, signal_data, investment_amount=250):
        """Run backtest for one stock"""
        first_thursday = pd.Timestamp(price_data.index[0].date())  # Convert to Timestamp
        first_friday = first_thursday + pd.Timedelta(days=1)

        # Check if signals are True on Thursday
        thursday_signals = signal_data.loc[first_thursday.strftime('%Y-%m-%d')]  # Use string format
        required_signals_met = all(thursday_signals[f'Signal_{i+1}'] for i in self.required_signals)

        if not required_signals_met:
            return None

        # Generate random buy time between 6:32 and 6:35 AM on Friday
        buy_time = pd.Timestamp.combine(
            first_friday.date(),
            pd.Timestamp('06:32:00').time()
        ) + pd.Timedelta(minutes=np.random.randint(0, 4))

        # Find the actual price data point closest to our desired buy time
        buy_time = price_data.index[price_data.index.searchsorted(buy_time)]
        buy_price = price_data.loc[buy_time, 'Open']
        shares = investment_amount / buy_price

        # Initialize tracking variables
        current_position = True
        current_value = investment_amount
        max_value = current_value
        min_value = current_value
        sell_time = None
        sell_price = None
        sell_reason = None

        # Track position through time
        for current_time in price_data.index[price_data.index > buy_time]:
            if not current_position:
                break

            current_price = price_data.loc[current_time, 'Open']
            current_value = shares * current_price
            max_value = max(max_value, current_value)
            min_value = min(min_value, current_value)

            # Check profit target
            if current_price >= buy_price * (1 + self.profit_target):
                sell_time = current_time
                sell_price = current_price
                sell_reason = 'Profit Target'
                current_position = False
                break

            # Check loss target
            if current_price <= buy_price * (1 - self.loss_target):
                sell_time = current_time
                sell_price = current_price
                sell_reason = 'Loss Target'
                current_position = False
                break

            # Check daily signals
            current_date = pd.Timestamp(current_time.date())
            if current_date.strftime('%Y-%m-%d') in signal_data.index:
                day_signals = signal_data.loc[current_date.strftime('%Y-%m-%d')]
                if not all(day_signals[f'Signal_{i+1}'] for i in self.required_signals):
                    next_day = current_date + pd.Timedelta(days=1)
                    while next_day.strftime('%Y-%m-%d') not in price_data.index:
                        next_day += pd.Timedelta(days=1)
                    sell_time = price_data.index[price_data.index.searchsorted(next_day)]
                    sell_price = price_data.loc[sell_time, 'Open']
                    sell_reason = 'Signal Exit'
                    current_position = False
                    break

        # If still holding at end of period, sell 1 hour before close
        if current_position:
            last_day = price_data.index[-1].date()
            sell_time = pd.Timestamp.combine(last_day, pd.Timestamp('12:00:00').time())
            sell_time = price_data.index[price_data.index.searchsorted(sell_time)]
            sell_price = price_data.loc[sell_time, 'Open']
            sell_reason = 'End of Period'

        final_value = shares * sell_price
        max_drawdown = (max_value - min_value) / max_value

        return {
            'buy_time': buy_time,
            'buy_price': buy_price,
            'sell_time': sell_time,
            'sell_price': sell_price,
            'sell_reason': sell_reason,
            'final_value': final_value,
            'max_drawdown': max_drawdown,
            'return_pct': (final_value - investment_amount) / investment_amount * 100
        }

def plot_stock(price_data, trade_info, title):
    """Plot candlestick chart with trade annotations"""
    if trade_info is None:
        return

    # Convert to mpf format
    df = price_data.copy()
    df.index.name = 'Date'

    # Create empty series spanning the full date range
    annotations = pd.Series(np.nan, index=df.index)

    # Add buy and sell points at specific times
    annotations[trade_info['buy_time']] = trade_info['buy_price']
    annotations[trade_info['sell_time']] = trade_info['sell_price']

    # Create buy annotation
    buy_data = pd.Series(np.nan, index=df.index)
    buy_data[trade_info['buy_time']] = trade_info['buy_price']

    # Create sell annotation
    sell_data = pd.Series(np.nan, index=df.index)
    sell_data[trade_info['sell_time']] = trade_info['sell_price']

    # Create addplot objects
    addplots = [
        mpf.make_addplot(buy_data, type='scatter', marker='^', markersize=100, color='g'),
        mpf.make_addplot(sell_data, type='scatter', marker='v', markersize=100, color='r')
    ]

    # Plot
    fig, axes = mpf.plot(
        df,
        type='hollow_and_filled',
        style='yahoo',
        title=title,
        addplot=addplots,
        warn_too_much_data=1000000000,
        returnfig=True
    )

    # Add text annotation for sell reason
    ax = axes[0]
    ax.annotate(
        trade_info['sell_reason'],
        xy=(df.index.get_loc(trade_info['sell_time']), trade_info['sell_price']),
        xytext=(10, 10),
        textcoords='offset points'
    )

    plt.show()

def objective_function(params):
    """Objective function for optimization"""
    required_signals, profit_target, loss_target = params

    # Create simulator and backtester instances
    simulator = StockSimulator()
    backtester = Backtester(
        profit_target=profit_target,
        loss_target=loss_target,
        required_signals=required_signals
    )

    # Run backtest
    total_value = 0
    max_drawdown = 0

    # Generate data and run backtest for multiple periods
    start_date = pd.Timestamp('2024-03-07')
    for _ in range(4):  # 4 periods
        end_date = start_date + pd.Timedelta(days=8)

        price_data = simulator.generate_minute_prices(start_date, end_date)
        signal_data = simulator.generate_market_signals(
            pd.date_range(start=start_date, end=end_date, freq='B')
        )

        result = backtester.run_backtest(price_data, signal_data)
        if result:
            total_value += result['final_value']
            max_drawdown = max(max_drawdown, result['max_drawdown'])

        start_date += pd.Timedelta(days=7)

    # Combine objectives (maximize value, minimize drawdown)
    score = total_value * (1 - max_drawdown)
    return -score  # Negative because we want to maximize

def optimize_strategy():
    """Optimize strategy parameters using parallel processing"""
    # Define parameter space
    signal_combinations = list(product([0, 1], repeat=6))  # Binary choices for signals
    profit_targets = np.linspace(0.01, 0.25, 3)
    loss_targets = np.linspace(0.01, 0.20, 3)

    # Create parameter combinations
    params = []
    for signals in signal_combinations:
        required_signals = [i for i, use in enumerate(signals) if use]
        for pt in profit_targets:
            for lt in loss_targets:
                params.append((required_signals, pt, lt))

    # Run optimization in parallel
    with mp.Pool() as pool:
        results = list(tqdm(
            pool.imap(objective_function, params),
            total=len(params),
            desc="Optimizing"
        ))

    # Find best parameters
    best_idx = np.argmin(results)
    best_params = params[best_idx]

    return best_params

# Main execution
if __name__ == "__main__":
    # Initialize simulation
    simulator = StockSimulator()
    backtester = Backtester()

    # Track portfolio performance
    portfolio_values = []
    all_trades = []

    # Run simulation for multiple periods
    start_date = pd.Timestamp('2024-03-07')
    period_end_dates = []

    for period in range(4):  # 4 periods
        end_date = start_date + pd.Timedelta(days=8)
        period_end_dates.append(end_date)

        # Generate data for 4 stocks
        period_trades = []
        period_value = 0

        for stock in range(4):
            price_data = simulator.generate_minute_prices(start_date, end_date)
            signal_data = simulator.generate_market_signals(
                pd.date_range(start=start_date, end=end_date, freq='B')
            )

            result = backtester.run_backtest(price_data, signal_data)
            if result:
                period_trades.append((price_data, result))
                period_value += result['final_value']
                all_trades.append({
                    'Period': period + 1,
                    'Stock': stock + 1,
                    'Buy Date': result['buy_time'].date(),
                    'Sell Date': result['sell_time'].date(),
                    'Return': result['return_pct'],
                    'Reason': result['sell_reason']
                })

        portfolio_values.append(period_value)

        # Plot best and worst performing stocks
        if period_trades:
            sorted_trades = sorted(period_trades, key=lambda x: x[1]['return_pct'])

            # Plot worst performing stock
            plot_stock(
                sorted_trades[0][0],
                sorted_trades[0][1],
                f'Period {period+1} - Worst Performing Stock'
            )

            # Plot best performing stock
            plot_stock(
                sorted_trades[-1][0],
                sorted_trades[-1][1],
                f'Period {period+1} - Best Performing Stock'
            )

        start_date += pd.Timedelta(days=7)

    # Print portfolio statistics
    initial_value = len(all_trades) * 250
    final_value = sum(portfolio_values)
    total_gain_dollars = final_value - initial_value
    total_gain_percent = (final_value - initial_value) / initial_value * 100
    max_drawdown = min(portfolio_values) - initial_value

    print("\nPortfolio Statistics:")
    print(f"Starting Value: ${initial_value:,.2f}")
    print(f"Final Value: ${final_value:,.2f}")
    print(f"Total Gain: ${total_gain_dollars:,.2f}")
    print(f"Total Gain %: {total_gain_percent:.2f}%")
    print(f"Max Drawdown: ${max_drawdown:,.2f}")

    # Print individual trade statistics
    print("\nIndividual Trade Statistics:")
    for trade in all_trades:
        print(f"\nPeriod {trade['Period']}, Stock {trade['Stock']}")
        print(f"Buy Date: {trade['Buy Date']}")
        print(f"Sell Date: {trade['Sell Date']}")
        print(f"Return: {trade['Return']:.2f}%")
        print(f"Exit Reason: {trade['Reason']}")

    # Create bar plot of portfolio returns
    plt.figure(figsize=(12, 6))
    returns = [(v - len(all_trades)*250)/(len(all_trades)*250) * 100 for v in portfolio_values]
    colors = ['g' if r >= 0 else 'r' for r in returns]

    plt.bar(range(len(returns)), returns, color=colors)
    plt.title('Portfolio Returns by Period')
    plt.xlabel('Period End Date')
    plt.ylabel('Return (%)')
    plt.xticks(range(len(returns)), [d.strftime('%Y-%m-%d') for d in period_end_dates], rotation=45)
    plt.tight_layout()
    plt.show()

    # Run optimization
    print("\nStarting strategy optimization...")
    best_signals, best_profit_target, best_loss_target = optimize_strategy()

    print("\nOptimal Parameters:")
    print("Required Signals:", [i+1 for i in best_signals])
    print(f"Profit Target: {best_profit_target:.2f}%")
    print(f"Loss Target: {best_loss_target:.2f}%")

    # Run backtest with optimal parameters
    print("\nRunning backtest with optimal parameters...")
    optimized_backtester = Backtester(
        profit_target=best_profit_target,
        loss_target=best_loss_target,
        required_signals=best_signals
    )

    # Track optimized portfolio performance
    optimized_portfolio_values = []
    optimized_trades = []

    # Run simulation with optimal parameters
    start_date = pd.Timestamp('2024-03-07')

    for period in range(4):
        end_date = start_date + pd.Timedelta(days=8)
        period_value = 0

        for stock in range(4):
            price_data = simulator.generate_minute_prices(start_date, end_date)
            signal_data = simulator.generate_market_signals(
                pd.date_range(start=start_date, end=end_date, freq='B')
            )

            result = optimized_backtester.run_backtest(price_data, signal_data)
            if result:
                period_value += result['final_value']
                optimized_trades.append({
                    'Period': period + 1,
                    'Stock': stock + 1,
                    'Return': result['return_pct']
                })

        optimized_portfolio_values.append(period_value)
        start_date += pd.Timedelta(days=7)

    # Print optimized results
    opt_initial_value = len(optimized_trades) * 250
    opt_final_value = sum(optimized_portfolio_values)
    opt_total_gain_dollars = opt_final_value - opt_initial_value
    opt_total_gain_percent = (opt_final_value - opt_initial_value) / opt_initial_value * 100
    opt_max_drawdown = min(optimized_portfolio_values) - opt_initial_value

    print("\nOptimized Portfolio Statistics:")
    print(f"Starting Value: ${opt_initial_value:,.2f}")
    print(f"Final Value: ${opt_final_value:,.2f}")
    print(f"Total Gain: ${opt_total_gain_dollars:,.2f}")
    print(f"Total Gain %: {opt_total_gain_percent:.2f}%")
    print(f"Max Drawdown: ${opt_max_drawdown:,.2f}")