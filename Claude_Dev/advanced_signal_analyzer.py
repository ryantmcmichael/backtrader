import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from typing import Dict, List, Tuple
import logging
import warnings
import os
from datetime import datetime, timedelta
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import make_scorer, f1_score
import xgboost as xgb

class GaugePatternAnalyzer:
    """
    Analyzes trading signal patterns from market gauge data to predict trading decisions.
    Implements strict temporal constraints to avoid look-ahead bias.
    """

    def __init__(self, base_path: str, random_state: int = np.random.randint(0,100)):
        """
        Initialize the analyzer with paths and parameters.

        Args:
            base_path: Base directory containing data files
            random_state: Random seed for reproducibility
        """
        self.base_path = base_path
        self.random_state = random_state
        self.logger = self._setup_logging()

        # Data containers
        self.ground_truth = None
        self.daily_mg = None
        self.daily_sp500 = None
        self.weekly_mg = None
        self.weekly_sp500 = None
        self.features_df = None
        self.model = None
        self.feature_importance = None

        # Constants
        self.DAILY_LOOKBACK = 15
        self.WEEKLY_LOOKBACK = 5
        self.TRAIN_SPLIT = 0.5

    def _setup_logging(self) -> logging.Logger:
        """Configure logging for the analyzer."""
        logger = logging.getLogger('GaugePatternAnalyzer')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.FileHandler('gauge_analysis.log', mode='w')
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def load_data(self) -> None:
        """Load and preprocess all required data files."""
        self.logger.info("Loading data files...")

        try:
            # Load ground truth data
            gt_path = os.path.join(self.base_path, 'JD Purchase Dates.csv')
            self.ground_truth = pd.read_csv(gt_path)
            self.ground_truth['Purchase_Date'] = pd.to_datetime(self.ground_truth['Purchase Date']).dt.date

            # Load gauge data
            gauges_dir = os.path.join(self.base_path, 'Gauges')

            self.daily_mg = self._load_gauge_file(os.path.join(gauges_dir, 'daily-market-momentum-ga.csv'))
            self.daily_sp500 = self._load_gauge_file(os.path.join(gauges_dir, 'daily-sp-500-momentum-ga.csv'))
            self.weekly_mg = self._load_gauge_file(os.path.join(gauges_dir, 'weekly-market-momentum-g.csv'))
            self.weekly_sp500 = self._load_gauge_file(os.path.join(gauges_dir, 'weekly-sp-500-momentum-g.csv'))

            self.logger.info("Data loading completed successfully")

        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def _load_gauge_file(self, file_path: str) -> pd.DataFrame:
        """
        Load and preprocess a gauge file with validation.

        Args:
            file_path: Path to the gauge CSV file

        Returns:
            Preprocessed DataFrame with validated gauge data
        """
        # Read file
        df = pd.read_csv(file_path)

        # Convert DateTime and validate format
        df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        df = df.dropna(subset=['DateTime'])

        # Keep only first 3 columns and rename
        df = df.iloc[:, :3]
        df.columns = ['DateTime', 'Positive', 'Negative']

        return df

    def _calculate_features(self, date: datetime.date, gauge_df: pd.DataFrame,
                              lookback_days: int) -> Dict[str, float]:
        """
        Calculate features with safe handling of edge cases and numerical stability.
        """
        end_date = pd.Timestamp(date)
        start_date = end_date - pd.Timedelta(days=lookback_days)

        # Get historical data
        mask = (gauge_df['DateTime'] >= start_date) & (gauge_df['DateTime'] < end_date)
        hist_data = gauge_df[mask].copy()

        if hist_data.empty:
            return self._get_default_features()

        features = {}

        try:
            # Latest values
            latest = hist_data.iloc[-1]
            features['latest_positive'] = latest['Positive']
            features['latest_negative'] = latest['Negative']
            features['latest_diff'] = latest['Positive'] - latest['Negative']

            # Safe ratio calculation
            if latest['Negative'] > 0:
                features['latest_ratio'] = min(latest['Positive'] / latest['Negative'], 100)
            else:
                features['latest_ratio'] = 100 if latest['Positive'] > 0 else 1

            # Non-linear transformations with bounds
            features['pos_squared'] = min(latest['Positive'] ** 2, 10000)  # Cap at 100^2
            features['neg_squared'] = min(latest['Negative'] ** 2, 10000)
            features['pos_root'] = np.sqrt(max(latest['Positive'], 0))
            features['neg_root'] = np.sqrt(max(latest['Negative'], 0))

            # Moving averages
            features['ma_positive'] = hist_data['Positive'].mean()
            features['ma_negative'] = hist_data['Negative'].mean()

            # Standard deviations with minimum value
            features['std_positive'] = max(hist_data['Positive'].std(), 0.0001)
            features['std_negative'] = max(hist_data['Negative'].std(), 0.0001)

            # Rate of change with safety checks
            if len(hist_data) > 1:
                initial_pos = hist_data['Positive'].iloc[0]
                initial_neg = hist_data['Negative'].iloc[0]

                if initial_pos > 0:
                    features['roc_pos'] = min(((latest['Positive'] - initial_pos) / initial_pos * 100), 1000)
                else:
                    features['roc_pos'] = 0

                if initial_neg > 0:
                    features['roc_neg'] = min(((latest['Negative'] - initial_neg) / initial_neg * 100), 1000)
                else:
                    features['roc_neg'] = 0
            else:
                features['roc_pos'] = 0
                features['roc_neg'] = 0

            # Trend calculations with safety checks
            if len(hist_data) > 1:
                # Normalize time index to [0, 1]
                time_index = np.linspace(0, 1, len(hist_data))

                # Calculate trends
                pos_trend = np.polyfit(time_index, hist_data['Positive'], 1)
                neg_trend = np.polyfit(time_index, hist_data['Negative'], 1)

                features['trend_positive'] = pos_trend[0]
                features['trend_negative'] = neg_trend[0]
                features['trend_diff'] = pos_trend[0] - neg_trend[0]

                # Safe R-squared calculation
                p_fit = np.poly1d(pos_trend)
                n_fit = np.poly1d(neg_trend)

                p_mean = hist_data['Positive'].mean()
                n_mean = hist_data['Negative'].mean()

                p_ss_tot = max(((hist_data['Positive'] - p_mean) ** 2).sum(), 0.0001)
                n_ss_tot = max(((hist_data['Negative'] - n_mean) ** 2).sum(), 0.0001)

                p_ss_res = ((hist_data['Positive'] - p_fit(time_index)) ** 2).sum()
                n_ss_res = ((hist_data['Negative'] - n_fit(time_index)) ** 2).sum()

                features['trend_pos_r2'] = max(min(1 - (p_ss_res / p_ss_tot), 1), 0)
                features['trend_neg_r2'] = max(min(1 - (n_ss_res / n_ss_tot), 1), 0)
            else:
                features['trend_positive'] = 0
                features['trend_negative'] = 0
                features['trend_diff'] = 0
                features['trend_pos_r2'] = 0
                features['trend_neg_r2'] = 0

            # Momentum indicators with bounds
            if len(hist_data) >= 2:
                features['momentum_pos'] = min(latest['Positive'] - hist_data['Positive'].iloc[0], 100)
                features['momentum_neg'] = min(latest['Negative'] - hist_data['Negative'].iloc[0], 100)
            else:
                features['momentum_pos'] = 0
                features['momentum_neg'] = 0

            # Crossover analysis
            hist_data['crossover'] = (hist_data['Positive'] > hist_data['Negative']).astype(int).diff()
            features['crossover_count'] = abs(hist_data['crossover'].fillna(0)).sum()

            # Relative position metrics
            features['time_pos_above'] = (hist_data['Positive'] > hist_data['Negative']).mean()
            features['time_pos_above_ma'] = (hist_data['Positive'] > hist_data['Positive'].mean()).mean()

            # Acceleration with bounds
            if len(hist_data) > 2:
                pos_vel = hist_data['Positive'].diff()
                neg_vel = hist_data['Negative'].diff()

                # Bound velocities
                pos_vel = pos_vel.clip(-100, 100)
                neg_vel = neg_vel.clip(-100, 100)

                features['pos_acceleration'] = pos_vel.diff().mean()
                features['neg_acceleration'] = neg_vel.diff().mean()

                # Clip acceleration values
                features['pos_acceleration'] = max(min(features['pos_acceleration'], 100), -100)
                features['neg_acceleration'] = max(min(features['neg_acceleration'], 100), -100)
            else:
                features['pos_acceleration'] = 0
                features['neg_acceleration'] = 0

        except Exception as e:
            self.logger.warning(f"Error calculating features for date {date}: {str(e)}")
            return self._get_default_features()

        # Final safety check to catch any remaining infinities
        for key, value in features.items():
            if not np.isfinite(value):
                features[key] = 0
                self.logger.warning(f"Infinite value caught in feature {key}, setting to 0")

        return features

    def _get_default_features(self) -> Dict[str, float]:
        """Return default feature values for missing data."""
        return {
            'latest_positive': np.nan,
            'latest_negative': np.nan,
            'latest_diff': np.nan,
            'latest_ratio': np.nan,
            'ma_positive': np.nan,
            'ma_negative': np.nan,
            'std_positive': np.nan,
            'std_negative': np.nan,
            'ma_diff': np.nan,
            'std_diff': np.nan,
            'trend_positive': np.nan,
            'trend_negative': np.nan,
            'trend_diff': np.nan,
            'crossover_count': np.nan,
            'pos_acceleration': np.nan,
            'neg_acceleration': np.nan
        }

    def _generate_cross_gauge_features(self, daily_features: Dict[str, float],
                                     weekly_features: Dict[str, float]) -> Dict[str, float]:
        """Generate features comparing daily and weekly gauge data."""
        cross_features = {}

        try:
            # Ratios between daily and weekly values
            cross_features['daily_weekly_pos_ratio'] = (
                daily_features['latest_positive'] / weekly_features['latest_positive']
                if weekly_features['latest_positive'] != 0 else np.nan
            )

            cross_features['daily_weekly_neg_ratio'] = (
                daily_features['latest_negative'] / weekly_features['latest_negative']
                if weekly_features['latest_negative'] != 0 else np.nan
            )

            # Trend alignment
            cross_features['trend_alignment_positive'] = (
                1 if np.sign(daily_features['trend_positive']) == np.sign(weekly_features['trend_positive'])
                else 0 if not np.isnan(daily_features['trend_positive']) and not np.isnan(weekly_features['trend_positive'])
                else np.nan
            )

            cross_features['trend_alignment_negative'] = (
                1 if np.sign(daily_features['trend_negative']) == np.sign(weekly_features['trend_negative'])
                else 0 if not np.isnan(daily_features['trend_negative']) and not np.isnan(weekly_features['trend_negative'])
                else np.nan
            )

        except Exception as e:
            self.logger.warning(f"Error generating cross gauge features: {str(e)}")
            cross_features = {
                'daily_weekly_pos_ratio': np.nan,
                'daily_weekly_neg_ratio': np.nan,
                'trend_alignment_positive': np.nan,
                'trend_alignment_negative': np.nan
            }

        return cross_features

    def generate_features(self) -> None:
        """Generate features for all purchase dates in ground truth data."""
        self.logger.info("Generating features...")

        features_list = []
        for _, row in self.ground_truth.iterrows():
            date = row['Purchase_Date']

            # Calculate features for each gauge type
            daily_mg_features = self._calculate_features(date, self.daily_mg, self.DAILY_LOOKBACK)
            daily_sp500_features = self._calculate_features(date, self.daily_sp500, self.DAILY_LOOKBACK)
            weekly_mg_features = self._calculate_features(date, self.weekly_mg, self.WEEKLY_LOOKBACK * 7)
            weekly_sp500_features = self._calculate_features(date, self.weekly_sp500, self.WEEKLY_LOOKBACK * 7)

            # Generate cross-gauge features
            mg_cross_features = self._generate_cross_gauge_features(daily_mg_features, weekly_mg_features)
            sp500_cross_features = self._generate_cross_gauge_features(daily_sp500_features, weekly_sp500_features)

            # Combine all features
            features = {
                'Trading_Period': row['Trading Period'],
                'Purchase_Date': date,
                'Gauge_Decision': row['Gauge Decision']
            }

            # Add prefixes to distinguish feature sources
            features.update({f'daily_mg_{k}': v for k, v in daily_mg_features.items()})
            features.update({f'daily_sp500_{k}': v for k, v in daily_sp500_features.items()})
            features.update({f'weekly_mg_{k}': v for k, v in weekly_mg_features.items()})
            features.update({f'weekly_sp500_{k}': v for k, v in weekly_sp500_features.items()})
            features.update({f'mg_cross_{k}': v for k, v in mg_cross_features.items()})
            features.update({f'sp500_cross_{k}': v for k, v in sp500_cross_features.items()})

            features_list.append(features)

        self.features_df = pd.DataFrame(features_list)

        # Log feature generation statistics
        total_features = len(self.features_df.columns) - 3
        missing_rates = self.features_df.iloc[:, 3:].isna().mean()
        features_with_missing = missing_rates[missing_rates > 0]

        self.logger.info(f"Generated {total_features} features for {len(self.features_df)} periods")
        if not features_with_missing.empty:
            self.logger.info("\nFeatures with missing values:")
            for feature, rate in features_with_missing.items():
                self.logger.info(f"{feature}: {rate*100:.1f}% missing")

    def train_model(self) -> None:
        """Train an ensemble model with proper train/test split."""
        self.logger.info("Training model...")

        try:
            # Prepare features and target
            feature_cols = [col for col in self.features_df.columns
                          if col not in ['Trading_Period', 'Purchase_Date', 'Gauge_Decision']]
            X = self.features_df[feature_cols]
            y = self.features_df['Gauge_Decision']

            # Split into train/test sets
            split_idx = int(len(X) * self.TRAIN_SPLIT)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            self.logger.info(f"\nDataset Split:")
            self.logger.info(f"Training samples: {len(X_train)}")
            self.logger.info(f"Testing samples: {len(X_test)}")

            # Initialize preprocessing
            imputer = SimpleImputer(strategy='mean')
            scaler = StandardScaler()

            # Preprocess training data
            X_train_imputed = imputer.fit_transform(X_train)
            X_train_scaled = scaler.fit_transform(X_train_imputed)

            # Preprocess test data
            X_test_imputed = imputer.transform(X_test)
            X_test_scaled = scaler.transform(X_test_imputed)

            # Initialize models
            rf = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state
            )

            # Train model
            rf.fit(X_train_scaled, y_train)
            self.model = rf

            # Make predictions
            train_pred = rf.predict(X_train_scaled)
            test_pred = rf.predict(X_test_scaled)

            # Store results
            self.features_df['Dataset'] = ['Train'] * len(X_train) + ['Test'] * len(X_test)
            self.features_df['Predicted_Decision'] = np.concatenate([train_pred, test_pred])
            self.features_df['Success'] = self.features_df['Predicted_Decision'] == self.features_df['Gauge_Decision']

            # Calculate feature importance
            self.feature_importance = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': rf.feature_importances_
            }).sort_values('Importance', ascending=False)

            # Log separate performance metrics for train and test sets
            self._log_model_performance(X_train_scaled, y_train, train_pred,
                                      X_test_scaled, y_test, test_pred)

        except Exception as e:
            self.logger.error(f"Error during model training: {str(e)}")
            raise

    def _get_feature_description(self, feature_name: str) -> str:
        """
        Get description for a given feature.

        Args:
            feature_name: Base name of the feature without prefix

        Returns:
            Description of the feature's calculation and meaning
        """
        descriptions = {
            'latest_positive': 'Most recent positive gauge value before the purchase date',
            'latest_negative': 'Most recent negative gauge value before the purchase date',
            'latest_diff': 'Difference between latest positive and negative gauge values',
            'latest_ratio': 'Ratio of latest positive to negative gauge values',
            'ma_positive': 'Moving average of positive gauge values over the lookback period',
            'ma_negative': 'Moving average of negative gauge values over the lookback period',
            'std_positive': 'Standard deviation of positive gauge values over the lookback period',
            'std_negative': 'Standard deviation of negative gauge values over the lookback period',
            'ma_diff': 'Moving average of differences between positive and negative values',
            'std_diff': 'Standard deviation of differences between positive and negative values',
            'trend_positive': 'Linear trend (slope) of positive gauge values over the lookback period',
            'trend_negative': 'Linear trend (slope) of negative gauge values over the lookback period',
            'trend_diff': 'Linear trend (slope) of differences between positive and negative values',
            'crossover_count': 'Number of times positive and negative values crossed each other',
            'pos_acceleration': 'Rate of change in positive gauge values (second derivative)',
            'neg_acceleration': 'Rate of change in negative gauge values (second derivative)',
            'daily_weekly_pos_ratio': 'Ratio between daily and weekly positive gauge values',
            'daily_weekly_neg_ratio': 'Ratio between daily and weekly negative gauge values',
            'trend_alignment_positive': 'Boolean indicating if daily and weekly positive trends align (1 if same direction)',
            'trend_alignment_negative': 'Boolean indicating if daily and weekly negative trends align (1 if same direction)'
        }

        return descriptions.get(feature_name, 'Custom calculated feature')

    def _log_data_quality_metrics(self, X_train, X_test):
        """Log metrics about data quality and missing values."""
        self.logger.info("\nData Quality Metrics:")

        # Training set metrics
        train_missing = X_train.isna().sum()
        train_missing_pct = (train_missing / len(X_train) * 100)

        self.logger.info("\nTraining Set:")
        self.logger.info(f"Total samples: {len(X_train)}")
        self.logger.info(f"Features with missing values: {(train_missing > 0).sum()}")

        if (train_missing > 0).any():
            self.logger.info("\nFeatures with highest missing rates (Training):")
            for feat, pct in train_missing_pct.nlargest(5).items():
                self.logger.info(f"{feat}: {pct:.1f}%")

        # Test set metrics
        test_missing = X_test.isna().sum()
        test_missing_pct = (test_missing / len(X_test) * 100)

        self.logger.info("\nTest Set:")
        self.logger.info(f"Total samples: {len(X_test)}")
        self.logger.info(f"Features with missing values: {(test_missing > 0).sum()}")

        if (test_missing > 0).any():
            self.logger.info("\nFeatures with highest missing rates (Test):")
            for feat, pct in test_missing_pct.nlargest(5).items():
                self.logger.info(f"{feat}: {pct:.1f}%")

    def _log_model_performance(self, X_train, y_train, train_pred,
                             X_test, y_test, test_pred) -> None:
        """Log comprehensive model performance metrics."""
        self.logger.info("\nModel Performance Metrics:")

        # Training metrics
        self.logger.info("\nTraining Set Metrics:")
        self.logger.info(classification_report(y_train, train_pred))

        train_cm = confusion_matrix(y_train, train_pred)
        self.logger.info("\nTraining Confusion Matrix:")
        self.logger.info(f"Accuracy: {(train_cm[0,0]+train_cm[1,1])/(train_cm[0,0]+train_cm[1,1]+train_cm[0,1]+train_cm[1,0]):.4f}")
        self.logger.info(f"True Negative: {train_cm[0,0]}")
        self.logger.info(f"False Positive: {train_cm[0,1]}")
        self.logger.info(f"False Negative: {train_cm[1,0]}")
        self.logger.info(f"True Positive: {train_cm[1,1]}")

        # Testing metrics
        self.logger.info("\nTest Set Metrics:")
        self.logger.info(classification_report(y_test, test_pred))

        test_cm = confusion_matrix(y_test, test_pred)
        self.logger.info("\nTest Confusion Matrix:")
        self.logger.info(f"Accuracy: {(test_cm[0,0]+test_cm[1,1])/(test_cm[0,0]+test_cm[1,1]+test_cm[0,1]+test_cm[1,0]):.4f}")
        self.logger.info(f"True Negative: {test_cm[0,0]}")
        self.logger.info(f"False Positive: {test_cm[0,1]}")
        self.logger.info(f"False Negative: {test_cm[1,0]}")
        self.logger.info(f"True Positive: {test_cm[1,1]}")

        # Feature importance analysis
        self.logger.info("\nTop 10 Most Important Features:")
        for _, row in self.feature_importance.head(10).iterrows():
            feature = row['Feature']
            importance = row['Importance']

            self.logger.info(f"\n{feature}:")
            self.logger.info(f"  Importance: {importance:.4f}")

    def save_results(self) -> None:
        """Save analysis results to files."""
        self.logger.info("Saving results to files...")

        try:
            # Save results DataFrame
            results_path = os.path.join(self.base_path, 'pattern_analysis_results.csv')
            self.features_df.to_csv(results_path, index=False)

            # Save feature importance
            importance_path = os.path.join(self.base_path, 'feature_importance.csv')
            self.feature_importance.to_csv(importance_path, index=False)

            # Create feature description file
            description_path = os.path.join(self.base_path, 'feature_description.txt')
            with open(description_path, 'w') as f:
                f.write("Feature Descriptions:\n\n")

                # Group features by type
                feature_groups = {
                    'Daily Market Gauge': 'daily_mg_',
                    'Daily SP500 Gauge': 'daily_sp500_',
                    'Weekly Market Gauge': 'weekly_mg_',
                    'Weekly SP500 Gauge': 'weekly_sp500_',
                    'Cross-Gauge Market': 'mg_cross_',
                    'Cross-Gauge SP500': 'sp500_cross_'
                }

                for group_name, prefix in feature_groups.items():
                    f.write(f"\n{group_name} Features:\n")
                    f.write("-" * (len(group_name) + 10) + "\n")

                    group_features = [col for col in self.features_df.columns if col.startswith(prefix)]
                    for feature in group_features:
                        base_name = feature.replace(prefix, '')
                        description = self._get_feature_description(base_name)
                        f.write(f"{feature}:\n  {description}\n\n")

            self.logger.info("Results saved successfully")

        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            raise

    def analyze(self) -> None:
        """Run the complete analysis pipeline."""
        self.load_data()
        self.generate_features()
        self.train_model()
        self.save_results()
        self.logger.info("Analysis completed successfully")


def main():
    """Main execution function."""
    # Set base path
    base_path = r'C:\Users\ryant\Documents\Stock_Market\Python\universe_data\VM Weekly Breakout'

    # Create and run analyzer
    analyzer = GaugePatternAnalyzer(base_path)

    try:
        analyzer.analyze()
        print("Analysis completed successfully. Check gauge_analysis.log for detailed results.")
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        logging.error(f"Analysis failed: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()