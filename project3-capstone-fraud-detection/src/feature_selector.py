import pandas as pd
import numpy as np
import lightgbm as lgb
import logging
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from itertools import chain
from typing import Dict, List, Any, Optional, Union, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureSelector:
    """
    Class for performing feature selection for machine learning or data preprocessing.
    
    Implements five different methods to identify features for removal:
        1. High percentage of missing values.
        2. Single unique value.
        3. High collinearity.
        4. Zero importance from a gradient boosting machine.
        5. Low cumulative importance from a gradient boosting machine.
    """
    
    def __init__(self, data: pd.DataFrame, labels: Optional[Union[pd.Series, np.ndarray]] = None):
        """
        Initialize the FeatureSelector with data and optional labels.
        
        Args:
            data: Dataframe with features.
            labels: Optional labels for importance-based methods.
        """
        self.data = data
        self.labels = labels

        if labels is None:
            logger.warning('No labels provided. Feature importance based methods are not available.')
        
        self.base_features = list(data.columns)
        self.one_hot_features = None
        
        # Records for features to remove
        self.record_missing = None
        self.record_single_unique = None
        self.record_collinear = None
        self.record_zero_importance = None
        self.record_low_importance = None
        
        self.missing_stats = None
        self.unique_stats = None
        self.corr_matrix = None
        self.feature_importances = None
        
        # Dictionary to hold removal operations
        self.ops = {}
        self.one_hot_correlated = False

    def identify_missing(self, missing_threshold: float) -> None:
        """
        Find features with missing fraction above threshold.
        
        Args:
            missing_threshold: Fraction of missing values above which to drop.
        """
        self.missing_threshold = missing_threshold

        missing_series = self.data.isnull().sum() / self.data.shape[0]
        self.missing_stats = pd.DataFrame(missing_series).rename(columns={0: 'missing_fraction'})
        self.missing_stats = self.missing_stats.sort_values('missing_fraction', ascending=False)

        record_missing = pd.DataFrame(missing_series[missing_series > missing_threshold]).reset_index().rename(
            columns={'index': 'feature', 0: 'missing_fraction'}
        )

        self.record_missing = record_missing
        self.ops['missing'] = list(record_missing['feature'])
        
        logger.info(f"{len(self.ops['missing'])} features identified with > {missing_threshold:.2f} missing values.")

    def identify_single_unique(self) -> None:
        """Find features with only a single unique value (ignoring NaNs)."""
        unique_counts = self.data.nunique()
        self.unique_stats = pd.DataFrame(unique_counts).rename(columns={0: 'nunique'})
        self.unique_stats = self.unique_stats.sort_values('nunique', ascending=True)
        
        record_single_unique = pd.DataFrame(unique_counts[unique_counts == 1]).reset_index().rename(
            columns={'index': 'feature', 0: 'nunique'}
        )

        self.record_single_unique = record_single_unique
        self.ops['single_unique'] = list(record_single_unique['feature'])
        
        logger.info(f"{len(self.ops['single_unique'])} features identified with a single unique value.")

    def identify_collinear(self, correlation_threshold: float, one_hot: bool = False) -> None:
        """
        Find collinear features based on correlation coefficient.
        
        Args:
            correlation_threshold: Pearson correlation coefficient threshold.
            one_hot: Whether to one-hot encode categorical features before calculation.
        """
        self.correlation_threshold = correlation_threshold
        self.one_hot_correlated = one_hot
        
        if one_hot:
            features = pd.get_dummies(self.data)
            self.one_hot_features = [col for col in features.columns if col not in self.base_features]
            self.data_all = pd.concat([features[self.one_hot_features], self.data], axis=1)
            corr_matrix = features.corr()
        else:
            corr_matrix = self.data.corr()
        
        self.corr_matrix = corr_matrix
    
        # Extract upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Select features with correlations above threshold
        to_drop = [column for column in upper.columns if any(upper[column].abs() > correlation_threshold)]

        record_collinear_list = []
        for column in to_drop:
            corr_features = list(upper.index[upper[column].abs() > correlation_threshold])
            corr_values = list(upper[column][upper[column].abs() > correlation_threshold])
            
            for f, v in zip(corr_features, corr_values):
                record_collinear_list.append({'drop_feature': column, 'corr_feature': f, 'corr_value': v})

        self.record_collinear = pd.DataFrame(record_collinear_list)
        self.ops['collinear'] = to_drop
        
        logger.info(f"{len(self.ops['collinear'])} features identified with correlation magnitude > {correlation_threshold:.2f}.")

    def identify_zero_importance(self, 
                                 task: str, 
                                 eval_metric: Optional[str] = None, 
                                 n_iterations: int = 10, 
                                 early_stopping: bool = True) -> None:
        """
        Identify features with zero importance using a Gradient Boosting Machine.
        
        Args:
            task: 'classification' or 'regression'.
            eval_metric: Metric for early stopping (e.g., 'auc', 'l2').
            n_iterations: Number of training iterations to average importances.
            early_stopping: Whether to use a validation set for early stopping.
        """
        if early_stopping and eval_metric is None:
            raise ValueError('eval_metric must be provided if early_stopping is True.')
            
        if self.labels is None:
            raise ValueError("Labels must be provided for importance-based identification.")
        
        features = pd.get_dummies(self.data)
        self.one_hot_features = [col for col in features.columns if col not in self.base_features]
        self.data_all = pd.concat([features[self.one_hot_features], self.data], axis=1)

        feature_names = list(features.columns)
        X = np.array(features)
        y = np.array(self.labels).reshape((-1, ))

        feature_importance_values = np.zeros(len(feature_names))
        
        logger.info('Training Gradient Boosting Model...')
        
        for i in range(n_iterations):
            if task == 'classification':
                model = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, verbose=-1)
            elif task == 'regression':
                model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, verbose=-1)
            else:
                raise ValueError('Task must be "classification" or "regression".')
                
            if early_stopping:
                X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15)
                model.fit(X_train, y_train, eval_metric=eval_metric,
                          eval_set=[(X_valid, y_valid)],
                          callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=0)])
                
                gc.collect()
            else:
                model.fit(X, y)

            feature_importance_values += model.feature_importances_ / n_iterations

        self.feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})
        self.feature_importances = self.feature_importances.sort_values('importance', ascending=False).reset_index(drop=True)

        # Normalize and calculate cumulative importance
        self.feature_importances['normalized_importance'] = self.feature_importances['importance'] / self.feature_importances['importance'].sum()
        self.feature_importances['cumulative_importance'] = np.cumsum(self.feature_importances['normalized_importance'])

        record_zero_importance = self.feature_importances[self.feature_importances['importance'] == 0.0]
        self.record_zero_importance = record_zero_importance
        self.ops['zero_importance'] = list(record_zero_importance['feature'])
        
        logger.info(f"{len(self.ops['zero_importance'])} features identified with zero importance.")

    def identify_low_importance(self, cumulative_importance: float) -> None:
        """Identify features that do not contribute to the specified cumulative importance threshold."""
        if self.feature_importances is None:
            raise RuntimeError("Must call `identify_zero_importance` before `identify_low_importance`.")
            
        self.cumulative_importance = cumulative_importance
        record_low_importance = self.feature_importances[self.feature_importances['cumulative_importance'] > cumulative_importance]

        self.record_low_importance = record_low_importance
        self.ops['low_importance'] = list(record_low_importance['feature'])
        
        logger.info(f"{len(self.ops['low_importance'])} features identified as contributing to the remaining "
                    f"{1-cumulative_importance:.2f} of cumulative importance.")

    def identify_all(self, selection_params: Dict[str, Any]) -> None:
        """Run all five identification methods."""
        required_params = ['missing_threshold', 'correlation_threshold', 'eval_metric', 'task', 'cumulative_importance']
        for p in required_params:
            if p not in selection_params:
                raise ValueError(f"Missing required parameter: {p}")
        
        self.identify_missing(selection_params['missing_threshold'])
        self.identify_single_unique()
        self.identify_collinear(selection_params['correlation_threshold'])
        self.identify_zero_importance(task=selection_params['task'], eval_metric=selection_params['eval_metric'])
        self.identify_low_importance(selection_params['cumulative_importance'])
        
        self.all_identified = set(chain(*self.ops.values()))
        logger.info(f"{len(self.all_identified)} total features identified for removal.")

    def remove(self, methods: Union[str, List[str]], keep_one_hot: bool = True) -> pd.DataFrame:
        """
        Remove identified features from the data.
        
        Args:
            methods: 'all' or list of specific methods to use.
            keep_one_hot: Whether to keep one-hot encoded columns if they were created.
            
        Returns:
            pd.DataFrame: Data with identified features removed.
        """
        features_to_drop = []
      
        if methods == 'all':
            data = self.data_all if hasattr(self, 'data_all') else self.data
            features_to_drop = set(chain(*self.ops.values()))
        else:
            if any(m in ['zero_importance', 'low_importance'] for m in methods) or self.one_hot_correlated:
                data = self.data_all
            else:
                data = self.data
                
            for m in methods:
                if m not in self.ops:
                    raise ValueError(f"Method '{m}' has not been run or identified no features.")
                features_to_drop.extend(self.ops[m])
        
            features_to_drop = set(features_to_drop)
            
        if not keep_one_hot and self.one_hot_features:
            features_to_drop = features_to_drop.union(set(self.one_hot_features))
       
        data = data.drop(columns=list(features_to_drop))
        logger.info(f"Removed {len(features_to_drop)} features.")
        return data

    def plot_missing(self) -> None:
        """Plot histogram of missing values fraction."""
        if self.missing_stats is None:
            raise RuntimeError("Run `identify_missing` first.")
        
        plt.figure(figsize=(8, 6))
        plt.hist(self.missing_stats['missing_fraction'], bins=10, edgecolor='k', color='coral')
        plt.xlabel('Missing Fraction')
        plt.ylabel('Count of Features')
        plt.title('Distribution of Missing Values')
        plt.show()

    def plot_feature_importances(self, plot_n: int = 15) -> None:
        """Plot the most important features."""
        if self.feature_importances is None:
            raise RuntimeError("Run `identify_zero_importance` first.")
            
        plot_df = self.feature_importances.head(plot_n).copy()
        plot_df = plot_df.sort_values('importance', ascending=True)
        
        plt.figure(figsize=(10, 8))
        plt.barh(plot_df['feature'], plot_df['normalized_importance'], color='teal', edgecolor='k')
        plt.xlabel('Normalized Importance')
        plt.title(f'Top {plot_n} Feature Importances')
        plt.show()
