# visualize_result.py
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, time, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional

# Import from backtest_strategy
from backtest_strategy import (
    plot_portfolio_returns,
    plot_annual_returns,
    plot_trade_candlestick,
    plot_benchmark_period,
    load_stock_data,
    Trade,
    StrategyParams,
    BenchmarkManager,
    BENCHMARK_TICKER,
    BASE_DIR
)

# ============== USER CONFIGURATION ==============
# Set these parameters to match the run you want to visualize

STRATEGY_PARAMS = StrategyParams(
    use_mg_pos_signal=False,
    use_mg_40_signal=False,
    use_sp500_signal=True,
    use_sector_signals=True,
    profit_target=1.05,
    loss_target=0.9,
    entry_time_offset=1
)

# Directory settings
RESULTS_DIR = "Optimization Results"

@dataclass
class StoredTrade:
    ticker: str
    entry_datetime: datetime
    entry_price: float
    investment: float
    shares: float
    exit_datetime: datetime
    exit_price: float
    exit_reason: str

    @classmethod
    def from_dict(cls, data: Dict):
        """Create StoredTrade from dictionary"""
        return cls(
            ticker=data['ticker'],
            entry_datetime=datetime.fromisoformat(data['entry_datetime']),
            entry_price=data['entry_price'],
            investment=data['investment'],
            shares=data['shares'],
            exit_datetime=datetime.fromisoformat(data['exit_datetime']) if data['exit_datetime'] else None,
            exit_price=data['exit_price'],
            exit_reason=data['exit_reason']
        )

    def to_trade(self) -> Trade:
        """Convert StoredTrade to Trade object"""
        trade = Trade(self.ticker, self.entry_datetime, self.entry_price, self.investment)
        trade.shares = self.shares
        trade.exit_datetime = self.exit_datetime
        trade.exit_price = self.exit_price
        trade.exit_reason = self.exit_reason
        return trade

def find_matching_run(results_df: pd.DataFrame, params: StrategyParams) -> pd.Series:
    """Find the run that matches the specified parameters"""
    mask = (
        (results_df['use_mg_pos_signal'] == params.use_mg_pos_signal) &
        (results_df['use_mg_40_signal'] == params.use_mg_40_signal) &
        (results_df['use_sp500_signal'] == params.use_sp500_signal) &
        (results_df['use_sector_signals'] == params.use_sector_signals) &
        (results_df['profit_target'] == params.profit_target) &
        (results_df['loss_target'] == params.loss_target) &
        (results_df['entry_time_offset'] == params.entry_time_offset)
    )

    matching_runs = results_df[mask]

    if len(matching_runs) == 0:
        raise ValueError("No matching run found with the specified parameters")
    elif len(matching_runs) > 1:
        print("Warning: Multiple matching runs found. Using the first one.")

    return matching_runs.iloc[0]

def load_run_data(run_data: pd.Series) -> Dict:
    """Load data for a specific optimization run."""
    # Get the run ID (1-based index in the CSV)
    results_df = pd.read_csv(os.path.join(RESULTS_DIR, 'detailed_results.csv'))
    run_id = results_df[
        (results_df[['use_mg_pos_signal', 'use_mg_40_signal', 'use_sp500_signal',
                    'use_sector_signals', 'profit_target', 'loss_target',
                    'entry_time_offset']] ==
         run_data[['use_mg_pos_signal', 'use_mg_40_signal', 'use_sp500_signal',
                  'use_sector_signals', 'profit_target', 'loss_target',
                  'entry_time_offset']]).all(axis=1)
    ].index[0] + 1

    # Load the stored trade data
    trades_file = os.path.join(RESULTS_DIR, 'trades', f'trades_{run_id}.json')
    with open(trades_file, 'r') as f:
        trades_data = json.load(f)

    # Convert trade dictionaries to StoredTrade objects
    trades = [StoredTrade.from_dict(t) for t in trades_data['trades']]

    # Load portfolio returns
    returns_file = os.path.join(RESULTS_DIR, 'returns', f'returns_{run_id}.json')
    with open(returns_file, 'r') as f:
        portfolio_returns = json.load(f)

    # Initialize benchmark manager and get benchmark returns
    benchmark_manager = BenchmarkManager(BENCHMARK_TICKER, BASE_DIR)
    benchmark_manager.load_or_download_benchmark()

    # Calculate benchmark returns for visualization
    trade_objects = [t.to_trade() for t in trades]
    start_date = min(t.entry_datetime.date() for t in trade_objects)
    end_date = max(t.exit_datetime.date() for t in trade_objects)

    # Get benchmark data for the period
    benchmark_data = benchmark_manager.get_benchmark_prices(start_date, end_date).copy()

    # Calculate daily benchmark returns
    initial_price = benchmark_data.iloc[0]['Open']
    benchmark_data.loc[:, 'Return'] = ((benchmark_data['Close'] - initial_price) / initial_price * 100)

    # Create dictionary mapping dates to returns
    benchmark_returns = {row['Date']: row['Return'] for _, row in benchmark_data.iterrows()}

    return {
        'run_data': run_data,
        'trades': trades,
        'portfolio_returns': portfolio_returns,
        'benchmark_returns': benchmark_returns,
        'benchmark_manager': benchmark_manager
    }

def print_run_summary(run_data: pd.Series):
    """Print summary statistics for the run"""
    print("\nRun Summary:")
    print("-" * 50)
    print("\nParameters:")
    print(f"  Use MG Pos Signal: {run_data['use_mg_pos_signal']}")
    print(f"  Use MG 40 Signal: {run_data['use_mg_40_signal']}")
    print(f"  Use SP500 Signal: {run_data['use_sp500_signal']}")
    print(f"  Use Sector Signals: {run_data['use_sector_signals']}")
    print(f"  Profit Target: {(run_data['profit_target'] - 1) * 100:.1f}%")
    print(f"  Loss Target: {(1 - run_data['loss_target']) * 100:.1f}%")
    print(f"  Entry Time Offset: {run_data['entry_time_offset']} minutes")

    print("\nPerformance Metrics:")
    print(f"  Total Gain: ${run_data['total_gain']:,.2f}")
    print(f"  Cumulative Return: {run_data['cumulative_return']:.2f}%")
    print(f"  Annualized Return: {run_data['annualized_return']:.2f}%")
    print(f"  Max Drawdown: {run_data['max_drawdown']:,.2f}%")
    print(f"  Win Rate: {run_data['win_rate']:.1f}%")
    print(f"  Sharpe Ratio: {run_data['sharpe_ratio']:.2f}")
    print(f"  Sortino Ratio: {run_data['sortino_ratio']:.2f}")
    print(f"  Calmar Ratio: {run_data['calmar_ratio']:.2f}")
    print(f"  Total Trades: {run_data['total_trades']}")

def create_visualizations(run_data: Dict):
    """Create all visualizations for the run"""
    print("\nCreating visualizations...")

    # Plot portfolio returns with benchmark comparison
    print("Plotting portfolio returns...")
    plot_portfolio_returns(run_data['portfolio_returns'], run_data['benchmark_returns'])
    plt.show()

    # Plot annual returns with benchmark comparison
    print("Plotting annual returns...")
    plot_annual_returns(run_data['portfolio_returns'], run_data['benchmark_returns'])
    plt.show()

    # Plot benchmark performance
    print("Plotting benchmark performance...")
    trades = [t.to_trade() for t in run_data['trades']]
    start_date = min(t.entry_datetime.date() for t in trades)
    end_date = max(t.exit_datetime.date() for t in trades)
    plot_benchmark_period(run_data['benchmark_manager'], start_date, end_date)
    plt.show()

    # Create candlestick plots for best and worst trades
    print("Creating candlestick plots for best and worst performers...")
    sorted_trades = sorted(trades, key=lambda x: x.get_pnl()[1], reverse=True)
    best_trades = sorted_trades[:4]
    worst_trades = sorted_trades[-4:]

    for trade in best_trades + worst_trades:
        # Find the stock data file
        ticker_data_dir = os.path.join(BASE_DIR, 'Ticker_data')
        date_dirs = sorted([d for d in os.listdir(ticker_data_dir)
                          if os.path.isdir(os.path.join(ticker_data_dir, d))])

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

def main():
    """Main execution function"""
    try:
        # Load results
        results_df = pd.read_csv(os.path.join(RESULTS_DIR, 'detailed_results.csv'))

        # Find the matching run
        run_data = find_matching_run(results_df, STRATEGY_PARAMS)

        # Print summary
        print_run_summary(run_data)

        # Load full run data
        full_run_data = load_run_data(run_data)

        # Create visualizations
        create_visualizations(full_run_data)

    except ValueError as e:
        print(f"Error: {str(e)}")
        return
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()