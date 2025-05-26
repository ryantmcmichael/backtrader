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
    print("Top 10 Results with New Weights:")
    for _, row in df_sorted.head(10).iterrows():
        print(f"Rank {row['rank']} (Score: {row['new_score']:.4f})")
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
        print(f"    Profit Factor: {row['profit_factor']:.2f}\n")

    # Parameter Frequency Analysis for Top 10%
    print("Parameter Frequency Analysis for Top 10%:")
    num_top = len(df_sorted) // 10  # Top 10%
    top_results = df_sorted.head(num_top)

    # Entry parameters
    print("Entry Parameters:")
    print(f"MG Pos Signal True: {(top_results['entry_use_mg_pos_signal'] == True).mean():.1%}")
    print(f"MG 40 Signal True: {(top_results['entry_use_mg_40_signal'] == True).mean():.1%}")
    print(f"SP500 Signal True: {(top_results['entry_use_sp500_signal'] == True).mean():.1%}")

    # Exit parameters
    print("Exit Parameters:")
    print(f"MG Pos Signal True: {(top_results['exit_use_mg_pos_signal'] == True).mean():.1%}")
    print(f"MG 40 Signal True: {(top_results['exit_use_mg_40_signal'] == True).mean():.1%}")
    print(f"SP500 Signal True: {(top_results['exit_use_sp500_signal'] == True).mean():.1%}")
    print(f"Sector Signals True: {(top_results['exit_use_sector_signals'] == True).mean():.1%}")

    # Trade parameters
    print("Trade Parameters (Average):")
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