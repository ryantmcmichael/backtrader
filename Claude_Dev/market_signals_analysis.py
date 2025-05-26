import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from typing import List, Dict, Callable

class SignalAnalyzer:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.gauges_dir = os.path.join(base_dir, 'Gauges')

        # Load and process all data files
        self.ground_truth = self._load_ground_truth()
        self.daily_mg = self._load_gauge_data('daily-market-momentum-ga.csv')
        self.daily_sp500 = self._load_gauge_data('daily-sp-500-momentum-ga.csv')
        self.weekly_mg = self._load_gauge_data('weekly-market-momentum-g.csv')
        self.weekly_sp500 = self._load_gauge_data('weekly-sp-500-momentum-g.csv')

    def _load_ground_truth(self) -> pd.DataFrame:
        """Load and process ground truth data."""
        df = pd.read_csv(os.path.join(self.base_dir, 'JD Purchase Dates.csv'))
        df['Purchase Date'] = pd.to_datetime(df['Purchase Date'])
        return df.drop(columns=['Purchase Time'])

    def _load_gauge_data(self, filename: str) -> pd.DataFrame:
        """Load and process gauge data files."""
        df = pd.read_csv(os.path.join(self.gauges_dir, filename))

        # Convert datetime and filter valid rows
        df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        df = df.dropna(subset=['DateTime'])

        # Keep only first 3 columns and rename for consistency
        df = df.iloc[:, :3]
        df.columns = ['DateTime', 'Positive', 'Negative']

        return df.sort_values('DateTime')

    def _get_prior_valid_day(self, date: datetime, df: pd.DataFrame) -> pd.Series:
        """Get the first valid day's data prior to given date."""
        mask = df['DateTime'] < date
        if mask.any():
            return df[mask].iloc[-1]
        return pd.Series()

    def _get_prior_valid_days(self, date: datetime, df: pd.DataFrame, n_days: int) -> pd.DataFrame:
        """Get the n valid days' data prior to given date."""
        mask = df['DateTime'] < date
        if mask.any():
            return df[mask].tail(n_days)
        return pd.DataFrame()

    def _get_prior_valid_week(self, date: datetime, df: pd.DataFrame) -> pd.Series:
        """Get the first valid week's data prior to given date."""
        mask = df['DateTime'] < date
        if mask.any():
            return df[mask].iloc[-1]
        return pd.Series()

    def check_daily_mg_positive_vs_negative(self, date: datetime) -> bool:
        """Check if Daily MG Positive >= Daily MG Negative on prior day."""
        prior_day = self._get_prior_valid_day(date, self.daily_mg)
        if not prior_day.empty:
            return prior_day['Positive']+2 >= prior_day['Negative']
        return False

    def check_daily_mg_negative_below_40(self, date: datetime) -> bool:
        """Check if Daily MG Negative < 40 on prior day."""
        prior_day = self._get_prior_valid_day(date, self.daily_mg)
        if not prior_day.empty:
            return prior_day['Negative'] < 40
        return False

    def check_daily_sp500_positive_vs_negative(self, date: datetime) -> bool:
        """Check if Daily SP500 Positive >= Daily SP500 Negative on prior day."""
        prior_day = self._get_prior_valid_day(date, self.daily_sp500)
        if not prior_day.empty:
            return prior_day['Positive'] >= prior_day['Negative']
        return False

    def check_4_day_mg_positive_vs_negative(self, date: datetime) -> bool:
        """Check if all 4 prior days have Daily MG Positive >= Daily MG Negative."""
        prior_days = self._get_prior_valid_days(date, self.daily_mg, 4)
        if len(prior_days) == 4:
            return all(prior_days['Positive'] >= prior_days['Negative'])
        return False

    def check_increasing_mg_positive(self, date: datetime) -> bool:
        """Check if there are 3 consecutive days of increasing Daily MG Positive values."""
        prior_days = self._get_prior_valid_days(date, self.daily_mg, 3)
        if len(prior_days) == 3:
            values = prior_days['Positive'].values
            return all(values[i] < values[i+1] for i in range(len(values)-1))
        return False

    def check_weekly_mg_positive_vs_negative(self, date: datetime) -> bool:
        """Check if Weekly MG Positive >= Weekly MG Negative on prior week."""
        prior_week = self._get_prior_valid_week(date, self.weekly_mg)
        if not prior_week.empty:
            return prior_week['Positive'] >= prior_week['Negative']
        return False

    def check_weekly_sp500_positive_vs_negative(self, date: datetime) -> bool:
        """Check if Weekly SP500 Positive >= Weekly SP500 Negative on prior week."""
        prior_week = self._get_prior_valid_week(date, self.weekly_sp500)
        if not prior_week.empty:
            return prior_week['Positive'] >= prior_week['Negative']
        return False

    def generate_prediction(self, boolean_combination: Callable[[Dict[str, bool]], bool]) -> pd.DataFrame:
        """Generate predictions using the specified boolean combination function."""
        results = []

        for _, row in self.ground_truth.iterrows():
            date = row['Purchase Date']

            # Calculate all boolean conditions
            conditions = {
                'Daily_MG_Pos_vs_Neg': self.check_daily_mg_positive_vs_negative(date),
                'Daily_MG_Neg_Below_40': self.check_daily_mg_negative_below_40(date),
                'Daily_SP500_Pos_vs_Neg': self.check_daily_sp500_positive_vs_negative(date),
                '4_Day_MG_Pos_vs_Neg': self.check_4_day_mg_positive_vs_negative(date),
                'Increasing_MG_Pos': self.check_increasing_mg_positive(date),
                'Weekly_MG_Pos_vs_Neg': self.check_weekly_mg_positive_vs_negative(date),
                'Weekly_SP500_Pos_vs_Neg': self.check_weekly_sp500_positive_vs_negative(date)
            }

            # Calculate predicted decision using the provided boolean combination
            predicted_decision = boolean_combination(conditions)

            # Create result row
            result = {
                'Trading_Period': row['Trading Period'],
                'Purchase_Date': date,
                'Gauge_Decision': row['Gauge Decision'],
                'Predicted_Decision': predicted_decision,
                **conditions
            }

            results.append(result)

        # Create DataFrame
        results_df = pd.DataFrame(results)

        # Add Success column after Predicted_Decision
        results_df.insert(
            results_df.columns.get_loc('Predicted_Decision') + 1,
            'Success',
            results_df['Gauge_Decision'] == results_df['Predicted_Decision']
        )

        # Calculate accuracy
        accuracy = results_df['Success'].mean()
        print(f"\nPrediction Accuracy: {accuracy:.2%}")

        return results_df

def example_boolean_combination(conditions: Dict[str, bool]) -> bool:
    """Example boolean combination of conditions."""
    return (
        (conditions['Daily_MG_Pos_vs_Neg'] or conditions['Increasing_MG_Pos'])
        and conditions['4_Day_MG_Pos_vs_Neg']
    )

# Example usage
if __name__ == "__main__":
    base_dir = r"C:\Users\ryant\Documents\Stock_Market\Python\universe_data\VM Weekly Breakout"

    # Initialize analyzer
    analyzer = SignalAnalyzer(base_dir)

    # Generate predictions using example combination
    results = analyzer.generate_prediction(example_boolean_combination)

    # Display results
    pd.set_option('display.max_columns', None)
    print("\nFirst few rows of results:")
    print(results.head())

    # Save results to CSV
    output_file = 'signal_analysis_results.csv'
    results.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")