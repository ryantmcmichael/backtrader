# optimization.py
import multiprocessing as mp
from multiprocessing import Pool
import itertools
from datetime import time
from datetime import timedelta
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from dataclasses import asdict
from typing import List, Dict, Tuple
import os
import time as time_module
from backtest_strategy import StrategyParams, BacktestResults, main as run_backtest
import json

# Define directories
BASE_DIR = "C:/Users/ryant/Documents/Stock_Market/Python/universe_data/VM Weekly Breakout"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "Optimization Results")

def create_parameter_grid():
    """Create a grid of all parameter combinations to test"""
    # Entry signal parameters (3 binary parameters)
    entry_params = list(itertools.product([True, False], repeat=3))

    # Exit signal parameters (4 binary parameters)
    exit_params = [
     (True, True, True, False),
     (True, True, False, False),
     (True, False, True, False),
     (True, False, False, False),
     (False, True, True, False),
     (False, True, False, False),
     (False, False, True, False),
     (False, False, False, False)]

    # Profit targets
    profit_targets = [0.05, 0.1, 0.15, 0.2]

    # Loss targets
    loss_targets = [0.05, 0.1, 0.15, 0.2]

    # Entry time offsets (in minutes)
    entry_time_offsets = [0]  # Test delays

    # Create all combinations
    param_grid = []
    for entry in entry_params:
        for exit in exit_params:
            for profit in profit_targets:
                for loss in loss_targets:
                    for offset in entry_time_offsets:
                        param_grid.append({
                            # Entry parameters
                            'entry_use_mg_pos_signal': entry[0],
                            'entry_use_mg_40_signal': entry[1],
                            'entry_use_sp500_signal': entry[2],

                            # Exit parameters
                            'exit_use_mg_pos_signal': exit[0],
                            'exit_use_mg_40_signal': exit[1],
                            'exit_use_sp500_signal': exit[2],
                            'exit_use_sector_signals': exit[3],

                            # Other parameters
                            'profit_target': profit,
                            'loss_target': loss,
                            'entry_time_offset': offset
                        })
    return param_grid

def run_single_backtest(params_dict):
    """Wrapper function to run a single backtest with given parameters"""
    try:
        # Convert dictionary back to StrategyParams
        params = StrategyParams(
            # Entry parameters
            entry_use_mg_pos_signal=params_dict['entry_use_mg_pos_signal'],
            entry_use_mg_40_signal=params_dict['entry_use_mg_40_signal'],
            entry_use_sp500_signal=params_dict['entry_use_sp500_signal'],

            # Exit parameters
            exit_use_mg_pos_signal=params_dict['exit_use_mg_pos_signal'],
            exit_use_mg_40_signal=params_dict['exit_use_mg_40_signal'],
            exit_use_sp500_signal=params_dict['exit_use_sp500_signal'],
            exit_use_sector_signals=params_dict['exit_use_sector_signals'],

            # Other parameters
            profit_target = params_dict['profit_target'],  # Convert to multiplier
            loss_target = params_dict['loss_target'],      # Convert to multiplier
            entry_time_offset=params_dict['entry_time_offset']
        )

        # Run backtest and return full results object
        results = run_backtest(params, BASE_DIR)

        return {
            'params': params_dict,
            'results': results
        }
    except Exception as e:
        print(f"Error in backtest: {str(e)}")
        return None

def calculate_scores(results_list):
    """Calculate normalized scores for all results"""
    if not results_list:
        return []

    # Find maximum values using BacktestResults object
    max_gain = max(r['results'].total_gain for r in results_list)
    max_drawdown = max(r['results'].max_drawdown for r in results_list)

    # Weights
    gain_weight = 0.8
    drawdown_weight = 0.1
    winrate_weight = 0.1

    scored_results = []
    for result in results_list:
        results_obj = result['results']

        # Normalize gain
        max_gain_diff = max(max_gain - r['results'].total_gain for r in results_list)
        if max_gain_diff == 0:
            norm_gain = 1.0
        else:
            gain_diff = max_gain - results_obj.total_gain
            norm_gain = 1 - (gain_diff / max_gain_diff)

        # Normalize drawdown
        norm_drawdown = 1 - (results_obj.max_drawdown / max_drawdown) if max_drawdown > 0 else 1.0

        # Win rate normalization
        norm_winrate = results_obj.win_rate / 100

        # Calculate score
        score = (gain_weight * norm_gain +
                drawdown_weight * norm_drawdown +
                winrate_weight * norm_winrate)

        scored_results.append({
            'params': result['params'],
            'results': results_obj,
            'score': score
        })

    return scored_results

def save_optimization_results(results: List[Dict], output_dir: str):
    """Save complete optimization results to files for future analysis"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for trades and returns
    trades_dir = output_path / 'trades'
    returns_dir = output_path / 'returns'
    trades_dir.mkdir(exist_ok=True)
    returns_dir.mkdir(exist_ok=True)

    # Prepare detailed results data
    detailed_results = []
    for i, result in enumerate(results, 1):
        params = result['params']
        results_data = result['results']

        # Save trades data
        trades_data = {
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
                for t in results_data.trades
            ]
        }
        with open(trades_dir / f'trades_{i}.json', 'w') as f:
            json.dump(trades_data, f, indent=2)

        # Save portfolio returns
        with open(returns_dir / f'returns_{i}.json', 'w') as f:
            json.dump(results_data.portfolio_returns, f, indent=2)

        row = {
            # Entry Parameters
            'entry_use_mg_pos_signal': params['entry_use_mg_pos_signal'],
            'entry_use_mg_40_signal': params['entry_use_mg_40_signal'],
            'entry_use_sp500_signal': params['entry_use_sp500_signal'],

            # Exit Parameters
            'exit_use_mg_pos_signal': params['exit_use_mg_pos_signal'],
            'exit_use_mg_40_signal': params['exit_use_mg_40_signal'],
            'exit_use_sp500_signal': params['exit_use_sp500_signal'],
            'exit_use_sector_signals': params['exit_use_sector_signals'],

            # Trade Parameters
            'profit_target': params['profit_target'],
            'loss_target': params['loss_target'],
            'entry_time_offset': params['entry_time_offset'],

            # Core metrics
            'total_gain': results_data.total_gain,
            'max_drawdown': results_data.max_drawdown,
            'win_rate': results_data.win_rate,

            # Additional metrics
            'cumulative_return': results_data.metrics['cumulative_return'],
            'annualized_return': results_data.metrics['annualized_return'],
            'volatility': results_data.metrics['volatility'],
            'sharpe_ratio': results_data.metrics['sharpe_ratio'],
            'sortino_ratio': results_data.metrics['sortino_ratio'],
            'calmar_ratio': results_data.metrics['calmar_ratio'],
            'total_trades': results_data.metrics['total_trades'],
            'avg_win': results_data.metrics['avg_win'],
            'avg_loss': results_data.metrics['avg_loss'],
            'max_win': results_data.metrics['max_win'],
            'max_loss': results_data.metrics['max_loss'],
            'profit_factor': results_data.metrics['profit_factor'],
            'recovery_factor': results_data.metrics['recovery_factor'],
            'mar_ratio': results_data.metrics['mar_ratio'],

            # Score
            'score': result['score']
        }
        detailed_results.append(row)

    # Save full results to CSV
    df_results = pd.DataFrame(detailed_results)
    df_results.to_csv(output_path / 'detailed_results.csv', index=False)

    # Save scoring information
    scoring_info = {
        'weights': {
            'gain_weight': 0.4,
            'drawdown_weight': 0.3,
            'winrate_weight': 0.3
        },
        'normalization': {
            'gain_normalization': '1 - ((max_gain - gain) / max(max_gain - gains))',
            'drawdown_normalization': '1 - (drawdown / max_drawdown)',
            'winrate_normalization': 'winrate / 100'
        }
    }

    with open(output_path / 'scoring_info.txt', 'w') as f:
        f.write("Original Scoring Configuration:\n")
        f.write("-----------------------------\n\n")
        f.write("Weights:\n")
        for metric, weight in scoring_info['weights'].items():
            f.write(f"{metric}: {weight}\n")
        f.write("\nNormalization Methods:\n")
        for metric, method in scoring_info['normalization'].items():
            f.write(f"{metric}: {method}\n")

# Save rescore helper function
    rescore_code = """
import pandas as pd
import numpy as np

def rescore_results(csv_path, gain_weight=0.4, drawdown_weight=0.3, winrate_weight=0.3):
    '''
    Rescore optimization results with new weights

    Args:
        csv_path: Path to detailed_results.csv
        gain_weight: Weight for total gain (default: 0.4)
        drawdown_weight: Weight for max drawdown (default: 0.3)
        winrate_weight: Weight for win rate (default: 0.3)

    Returns:
        DataFrame with new scores and rankings
    '''
    # Read results
    df = pd.read_csv(csv_path)

    # Calculate maximums
    max_gain = df['total_gain'].max()
    max_drawdown = df['max_drawdown'].max()

    # Normalize gain
    max_gain_diff = max(max_gain - df['total_gain'])
    if max_gain_diff == 0:
        df['norm_gain'] = 1.0
    else:
        df['norm_gain'] = 1 - ((max_gain - df['total_gain']) / max_gain_diff)

    # Normalize drawdown
    df['norm_drawdown'] = 1 - (df['max_drawdown'] / max_drawdown) if max_drawdown > 0 else 1.0

    # Normalize win rate
    df['norm_winrate'] = df['win_rate'] / 100

    # Calculate new scores
    df['new_score'] = (
        gain_weight * df['norm_gain'] +
        drawdown_weight * df['norm_drawdown'] +
        winrate_weight * df['norm_winrate']
    )

    # Sort by new scores
    df_sorted = df.sort_values('new_score', ascending=False).reset_index(drop=True)

    # Add rank column
    df_sorted['rank'] = df_sorted.index + 1

    return df_sorted

def analyze_results(df_sorted):
    '''
    Analyze and print detailed results including parameter frequency analysis

    Args:
        df_sorted: Sorted DataFrame from rescore_results
    '''
    # Print top 10 results with new weights
    print("\nTop 10 Results with New Weights:")
    for _, row in df_sorted.head(10).iterrows():
        print(f"\nRank {row['rank']} (Score: {row['new_score']:.4f})")
        print(f"Parameters:")
        print("  Entry Conditions:")
        print(f"    MG Pos Signal: {row['entry_use_mg_pos_signal']}")
        print(f"    MG 40 Signal: {row['entry_use_mg_40_signal']}")
        print(f"    SP500 Signal: {row['entry_use_sp500_signal']}")
        print("  Exit Conditions:")
        print(f"    MG Pos Signal: {row['exit_use_mg_pos_signal']}")
        print(f"    MG 40 Signal: {row['exit_use_mg_40_signal']}")
        print(f"    SP500 Signal: {row['exit_use_sp500_signal']}")
        print(f"    Sector Signals: {row['exit_use_sector_signals']}")
        print("  Trade Parameters:")
        print(f"    Profit Target: {row['profit_target'] * 100:.1f}%")
        print(f"    Loss Target: {row['loss_target'] * 100:.1f}%")
        print(f"    Entry Time Offset: {row['entry_time_offset']} minutes")
        print(f"Results:")
        print(f"    Total Gain: ${row['total_gain']:,.2f}")
        print(f"    Max Drawdown: {row['max_drawdown']:,.2f}%")
        print(f"    Win Rate: {row['win_rate']:.1f}%")
        print(f"    Sharpe Ratio: {row['sharpe_ratio']:.2f}")
        print(f"    Sortino Ratio: {row['sortino_ratio']:.2f}")
        print(f"    Calmar Ratio: {row['calmar_ratio']:.2f}")
        print(f"    Total Trades: {row['total_trades']}")
        print(f"    Profit Factor: {row['profit_factor']:.2f}\\n")

    # Parameter Frequency Analysis for Top 10%
    print("\nParameter Frequency Analysis for Top 10%:")
    num_top = len(df_sorted) // 10  # Top 10%
    top_results = df_sorted.head(num_top)

    # Entry parameters
    print("\nEntry Parameters:")
    print(f"MG Pos Signal True: {(top_results['entry_use_mg_pos_signal'] == True).mean():.1%}")
    print(f"MG 40 Signal True: {(top_results['entry_use_mg_40_signal'] == True).mean():.1%}")
    print(f"SP500 Signal True: {(top_results['entry_use_sp500_signal'] == True).mean():.1%}")

    # Exit parameters
    print("\nExit Parameters:")
    print(f"MG Pos Signal True: {(top_results['exit_use_mg_pos_signal'] == True).mean():.1%}")
    print(f"MG 40 Signal True: {(top_results['exit_use_mg_40_signal'] == True).mean():.1%}")
    print(f"SP500 Signal True: {(top_results['exit_use_sp500_signal'] == True).mean():.1%}")
    print(f"Sector Signals True: {(top_results['exit_use_sector_signals'] == True).mean():.1%}")

    # Trade parameters
    print("\nTrade Parameters (Average):")
    print(f"Profit Target: {top_results['profit_target'].mean() * 100:.1f}%")
    print(f"Loss Target: {top_results['loss_target'].mean() * 100:.1f}%")
    print(f"Entry Time Offset: {top_results['entry_time_offset'].mean():.1f} minutes")

if __name__ == '__main__':
    # Example usage
    results_df = rescore_results('detailed_results.csv',
                               gain_weight=0.5,
                               drawdown_weight=0.3,
                               winrate_weight=0.2)

    # Analyze and print results
    analyze_results(results_df)
"""

    with open(output_path / 'rescore_helper.py', 'w') as f:
        f.write(rescore_code.strip())

    print(f"\nResults saved to {output_path}")
    print("Files created:")
    print("  - detailed_results.csv: Complete results for all parameter combinations")
    print("  - scoring_info.txt: Original scoring configuration")
    print("  - rescore_helper.py: Helper script for rescoring with new weights")

def optimize_strategy():
    """Run parallel optimization to find best parameter combination"""
    print("Starting optimization...")
    start_time = time_module.time()

    # Create parameter grid
    param_grid = create_parameter_grid()
    total_combinations = len(param_grid)
    print(f"Testing {total_combinations} parameter combinations")
    print(f"Using {max(1, mp.cpu_count() - 1)} processes")

    # Run parallel backtests
    results_list = []
    with Pool(max(1, mp.cpu_count() - 1)) as pool:
        for i, result in enumerate(pool.imap_unordered(run_single_backtest, param_grid), 1):
            if result is not None:
                results_list.append(result)

            # Calculate time statistics
            elapsed_time = time_module.time() - start_time
            progress = i / total_combinations
            if progress > 0:
                estimated_total_time = elapsed_time / progress
                remaining_time = estimated_total_time - elapsed_time

                # Format times
                elapsed_str = str(timedelta(seconds=int(elapsed_time)))
                remaining_str = str(timedelta(seconds=int(remaining_time)))

                print(f"\rProgress: {i:3d}/{total_combinations} combinations tested "
                      f"({progress*100:6.1f}%) | "
                      f"Elapsed: {elapsed_str} | "
                      f"Remaining: {remaining_str}", end='', flush=True)

    print("\n\nCalculating scores...")
    scored_results = calculate_scores(results_list)

    # Sort by score
    sorted_results = sorted(scored_results, key=lambda x: x['score'], reverse=True)

    # Save results
    save_optimization_results(sorted_results, OUTPUT_DIR)

    # Print timing information
    end_time = time_module.time()
    duration = end_time - start_time
    print(f"\nOptimization completed in {duration:.1f} seconds "
          f"({duration/60:.1f} minutes)")

    def print_result_summary(result, rank, total_results):
        """Helper function to print a single result summary"""
        params = result['params']
        results_obj = result['results']
        rank_str = f"{rank}" if rank <= total_results/2 else f"{total_results-rank+1} (from bottom)"

        print(f"\n{rank_str}. Score: {result['score']:.4f}")
        print(f"Parameters:")
        print("  Entry Conditions:")
        print(f"    MG Pos Signal: {params['entry_use_mg_pos_signal']}")
        print(f"    MG 40 Signal: {params['entry_use_mg_40_signal']}")
        print(f"    SP500 Signal: {params['entry_use_sp500_signal']}")
        print("  Exit Conditions:")
        print(f"    MG Pos Signal: {params['exit_use_mg_pos_signal']}")
        print(f"    MG 40 Signal: {params['exit_use_mg_40_signal']}")
        print(f"    SP500 Signal: {params['exit_use_sp500_signal']}")
        print(f"    Sector Signals: {params['exit_use_sector_signals']}")
        print("  Trade Parameters:")
        print(f"    Profit Target: {params['profit_target'] * 100:.1f}%")
        print(f"    Loss Target: {params['loss_target'] * 100:.1f}%")
        print(f"    Entry Time Offset: {params['entry_time_offset']} minutes")
        print(f"Results:")
        print(f"  Total Gain: ${results_obj.total_gain:,.2f}")
        print(f"  Max Drawdown: {results_obj.max_drawdown:,.2f}%")
        print(f"  Win Rate: {results_obj.win_rate:.1f}%")

    # Print top 5 combinations
    print("\n=== Top 5 Parameter Combinations ===")
    for i, result in enumerate(sorted_results[:5], 1):
        print_result_summary(result, i, len(sorted_results))

    # Print bottom 5 combinations
    print("\n=== Bottom 5 Parameter Combinations ===")
    for i, result in enumerate(sorted_results[-5:], len(sorted_results)-4):
        print_result_summary(result, i, len(sorted_results))

    return sorted_results

if __name__ == "__main__":
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Run optimization
    results = optimize_strategy()