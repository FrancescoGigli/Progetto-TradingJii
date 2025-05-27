#!/usr/bin/env python3
"""
Advanced Preprocessing Module for TradingJii ML System

Contains sophisticated preprocessing components including scaling, outlier handling,
missing value imputation, and data validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer
)
# Enable experimental IterativeImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer, SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, mutual_info_classif, chi2, f_classif
)
from sklearn.pipeline import Pipeline
import warnings

from .base import (
    BasePreprocessor, BaseValidator, PreprocessingError, ValidationError,
    validate_data_integrity, detect_outliers, compute_feature_importance_scores
)
from .config import get_preprocessing_config

warnings.filterwarnings('ignore')

# ====================================================================
# SCALING COMPONENTS
# ====================================================================

class MultiScaler(BasePreprocessor):
    """
    Advanced scaling component with multiple scaling methods.
    Supports standard, minmax, robust, and quantile scaling.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.method = self.config.get('method', 'robust')
        self.feature_range = self.config.get('feature_range', (0, 1))
        self.quantile_range = self.config.get('quantile_range', (25.0, 75.0))
        self.clip = self.config.get('clip', True)
        self.scaler_ = None
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'MultiScaler':
        """Fit the scaler to the data."""
        self._validate_input(X)
        
        # Select scaling method
        if self.method == 'standard':
            self.scaler_ = StandardScaler()
        elif self.method == 'minmax':
            self.scaler_ = MinMaxScaler(feature_range=self.feature_range, clip=self.clip)
        elif self.method == 'robust':
            self.scaler_ = RobustScaler(quantile_range=self.quantile_range)
        elif self.method == 'quantile':
            self.scaler_ = QuantileTransformer(
                output_distribution='uniform',
                n_quantiles=min(1000, len(X))
            )
        else:
            raise PreprocessingError(f"Unknown scaling method: {self.method}")
        
        # Fit the scaler
        self.scaler_.fit(X)
        self.is_fitted = True
        
        self.logger.info(f"Fitted {self.method} scaler on {X.shape[1]} features")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data using the fitted scaler."""
        if not self.is_fitted:
            raise PreprocessingError("Scaler must be fitted before transform")
        
        self._validate_input(X)
        
        # Transform the data
        X_scaled = self.scaler_.transform(X)
        
        # Create DataFrame with original column names
        X_transformed = pd.DataFrame(
            X_scaled, 
            columns=X.columns, 
            index=X.index
        )
        
        self._update_feature_info(X, X_transformed)
        return X_transformed
    
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform the scaled data."""
        if not self.is_fitted:
            raise PreprocessingError("Scaler must be fitted before inverse_transform")
        
        X_original = self.scaler_.inverse_transform(X)
        return pd.DataFrame(X_original, columns=X.columns, index=X.index)

# ====================================================================
# OUTLIER HANDLING
# ====================================================================

class OutlierHandler(BasePreprocessor):
    """
    Advanced outlier detection and handling component.
    Supports IQR, Z-score, Isolation Forest, and Local Outlier Factor methods.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.method = self.config.get('method', 'iqr')
        self.threshold = self.config.get('threshold', 1.5)
        self.zscore_threshold = self.config.get('zscore_threshold', 3.0)
        self.contamination = self.config.get('contamination', 0.1)
        self.n_neighbors = self.config.get('n_neighbors', 20)
        self.action = self.config.get('action', 'clip')  # 'remove', 'clip', 'transform'
        
        self.outlier_detector_ = None
        self.outlier_mask_ = None
        self.bounds_ = {}
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'OutlierHandler':
        """Fit the outlier detector to the data."""
        self._validate_input(X)
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        if self.method in ['iqr', 'zscore']:
            # Statistical methods - compute bounds
            for col in numeric_cols:
                if self.method == 'iqr':
                    Q1 = X[col].quantile(0.25)
                    Q3 = X[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - self.threshold * IQR
                    upper_bound = Q3 + self.threshold * IQR
                elif self.method == 'zscore':
                    mean = X[col].mean()
                    std = X[col].std()
                    lower_bound = mean - self.zscore_threshold * std
                    upper_bound = mean + self.zscore_threshold * std
                
                self.bounds_[col] = (lower_bound, upper_bound)
                
        elif self.method == 'isolation_forest':
            self.outlier_detector_ = IsolationForest(
                contamination=self.contamination,
                random_state=42
            )
            self.outlier_detector_.fit(X[numeric_cols])
            
        elif self.method == 'lof':
            self.outlier_detector_ = LocalOutlierFactor(
                n_neighbors=self.n_neighbors,
                contamination=self.contamination
            )
            # Note: LOF doesn't have separate fit/predict, so we store the data
            self.outlier_detector_.fit_predict(X[numeric_cols])
            
        self.is_fitted = True
        self.logger.info(f"Fitted {self.method} outlier detector")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in the data."""
        if not self.is_fitted:
            raise PreprocessingError("OutlierHandler must be fitted before transform")
        
        self._validate_input(X)
        X_transformed = X.copy()
        
        if self.action == 'remove':
            # Identify outliers and remove rows
            outlier_mask = self._detect_outliers(X_transformed)
            X_transformed = X_transformed[~outlier_mask.any(axis=1)]
            self.logger.info(f"Removed {outlier_mask.any(axis=1).sum()} outlier rows")
            
        elif self.action == 'clip':
            # Clip outliers to bounds
            for col, (lower_bound, upper_bound) in self.bounds_.items():
                if col in X_transformed.columns:
                    X_transformed[col] = X_transformed[col].clip(lower_bound, upper_bound)
            self.logger.info("Clipped outliers to computed bounds")
            
        elif self.action == 'transform':
            # Transform outliers (e.g., log transformation)
            outlier_mask = self._detect_outliers(X_transformed)
            for col in X_transformed.select_dtypes(include=[np.number]).columns:
                if col in outlier_mask.columns:
                    # Apply log1p transformation to outliers
                    mask = outlier_mask[col]
                    if mask.any():
                        X_transformed.loc[mask, col] = np.log1p(
                            np.abs(X_transformed.loc[mask, col])
                        )
        
        self._update_feature_info(X, X_transformed)
        return X_transformed
    
    def _detect_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        """Detect outliers in the data."""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        outlier_mask = pd.DataFrame(False, index=X.index, columns=X.columns)
        
        if self.method in ['iqr', 'zscore']:
            for col, (lower_bound, upper_bound) in self.bounds_.items():
                if col in X.columns:
                    outlier_mask[col] = (X[col] < lower_bound) | (X[col] > upper_bound)
                    
        elif self.method == 'isolation_forest':
            outlier_pred = self.outlier_detector_.predict(X[numeric_cols])
            for i, col in enumerate(numeric_cols):
                outlier_mask[col] = outlier_pred == -1
                
        elif self.method == 'lof':
            outlier_pred = self.outlier_detector_.fit_predict(X[numeric_cols])
            for i, col in enumerate(numeric_cols):
                outlier_mask[col] = outlier_pred == -1
        
        return outlier_mask

# ====================================================================
# MISSING VALUE HANDLING
# ====================================================================

class MissingValueHandler(BasePreprocessor):
    """
    Advanced missing value imputation component.
    Supports mean, median, mode, KNN, and iterative imputation.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.method = self.config.get('method', 'knn')
        self.n_neighbors = self.config.get('n_neighbors', 5)
        self.max_iter = self.config.get('max_iter', 10)
        self.random_state = self.config.get('random_state', 42)
        self.initial_strategy = self.config.get('initial_strategy', 'mean')
        
        self.imputer_ = None
        self.missing_columns_ = None
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'MissingValueHandler':
        """Fit the imputer to the data."""
        self._validate_input(X)
        
        # Identify columns with missing values
        self.missing_columns_ = X.columns[X.isnull().any()].tolist()
        
        if not self.missing_columns_:
            self.logger.info("No missing values found in data")
            self.is_fitted = True
            return self
        
        # Select imputation method
        if self.method in ['mean', 'median', 'most_frequent']:
            self.imputer_ = SimpleImputer(strategy=self.method)
        elif self.method == 'knn':
            self.imputer_ = KNNImputer(n_neighbors=self.n_neighbors)
        elif self.method == 'iterative':
            self.imputer_ = IterativeImputer(
                max_iter=self.max_iter,
                random_state=self.random_state,
                initial_strategy=self.initial_strategy
            )
        elif self.method == 'forward_fill':
            # Forward fill doesn't need fitting
            pass
        else:
            raise PreprocessingError(f"Unknown imputation method: {self.method}")
        
        # Fit the imputer
        if self.imputer_ is not None:
            self.imputer_.fit(X)
        
        self.is_fitted = True
        self.logger.info(f"Fitted {self.method} imputer for {len(self.missing_columns_)} columns")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values in the data."""
        if not self.is_fitted:
            raise PreprocessingError("MissingValueHandler must be fitted before transform")
        
        self._validate_input(X)
        
        if not self.missing_columns_:
            return X.copy()
        
        X_transformed = X.copy()
        
        if self.method == 'forward_fill':
            X_transformed = X_transformed.fillna(method='ffill')
        else:
            # Use fitted imputer
            X_imputed = self.imputer_.transform(X)
            X_transformed = pd.DataFrame(
                X_imputed,
                columns=X.columns,
                index=X.index
            )
        
        # Log imputation statistics
        imputed_count = X.isnull().sum().sum() - X_transformed.isnull().sum().sum()
        self.logger.info(f"Imputed {imputed_count} missing values")
        
        self._update_feature_info(X, X_transformed)
        return X_transformed

# ====================================================================
# DATA VALIDATION
# ====================================================================

class DataValidator(BaseValidator):
    """
    Comprehensive data validation component.
    Performs various data quality checks and validations.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.check_infinite = self.config.get('check_infinite', True)
        self.check_missing = self.config.get('check_missing', True)
        self.check_duplicates = self.config.get('check_duplicates', True)
        self.check_target_balance = self.config.get('check_target_balance', True)
        self.min_samples_per_class = self.config.get('min_samples_per_class', 10)
        self.max_correlation = self.config.get('max_correlation', 0.95)
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DataValidator':
        """Fit the validator (no-op for validator)."""
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform is pass-through for validator."""
        return X.copy()
    
    def validate(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Perform comprehensive validation of the data.
        
        Args:
            X: Feature matrix
            y: Target variable (optional)
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'passed': True,
            'warnings': [],
            'errors': [],
            'checks': {}
        }
        
        # Basic data integrity check
        integrity_results = validate_data_integrity(X, 'label' if y is not None else None)
        results['checks']['data_integrity'] = integrity_results
        
        # Check for infinite values
        if self.check_infinite:
            inf_check = self._check_infinite_values(X)
            results['checks']['infinite_values'] = inf_check
            if inf_check['has_infinite']:
                results['warnings'].append("Infinite values detected")
        
        # Check missing values
        if self.check_missing:
            missing_check = self._check_missing_values(X)
            results['checks']['missing_values'] = missing_check
            if missing_check['has_missing']:
                results['warnings'].append("Missing values detected")
        
        # Check duplicates
        if self.check_duplicates:
            duplicate_check = self._check_duplicate_rows(X)
            results['checks']['duplicates'] = duplicate_check
            if duplicate_check['has_duplicates']:
                results['warnings'].append("Duplicate rows detected")
        
        # Check feature correlations
        correlation_check = self._check_high_correlations(X)
        results['checks']['correlations'] = correlation_check
        if correlation_check['has_high_correlations']:
            results['warnings'].append("High feature correlations detected")
        
        # Target validation
        if y is not None and self.check_target_balance:
            target_check = self._check_target_distribution(y)
            results['checks']['target_distribution'] = target_check
            if target_check['is_imbalanced']:
                results['warnings'].append("Imbalanced target distribution")
            if target_check['insufficient_samples']:
                results['errors'].append("Insufficient samples per class")
                results['passed'] = False
        
        # Overall validation status
        if results['errors']:
            results['passed'] = False
        
        self.validation_results_ = results
        return results
    
    def _check_infinite_values(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Check for infinite values in the data."""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        infinite_counts = {}
        
        for col in numeric_cols:
            inf_count = np.isinf(X[col]).sum()
            if inf_count > 0:
                infinite_counts[col] = int(inf_count)
        
        return {
            'has_infinite': len(infinite_counts) > 0,
            'infinite_counts': infinite_counts,
            'total_infinite': sum(infinite_counts.values())
        }
    
    def _check_missing_values(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Check for missing values in the data."""
        missing_counts = X.isnull().sum()
        missing_percentages = (missing_counts / len(X) * 100)
        
        return {
            'has_missing': missing_counts.sum() > 0,
            'missing_counts': missing_counts[missing_counts > 0].to_dict(),
            'missing_percentages': missing_percentages[missing_percentages > 0].to_dict(),
            'total_missing': int(missing_counts.sum())
        }
    
    def _check_duplicate_rows(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Check for duplicate rows in the data."""
        duplicate_count = X.duplicated().sum()
        
        return {
            'has_duplicates': duplicate_count > 0,
            'duplicate_count': int(duplicate_count),
            'duplicate_percentage': duplicate_count / len(X) * 100
        }
    
    def _check_high_correlations(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Check for highly correlated features."""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {
                'has_high_correlations': False,
                'high_correlations': {},
                'correlation_pairs': []
            }
        
        correlation_matrix = X[numeric_cols].corr().abs()
        
        # Find high correlations (excluding diagonal)
        high_correlations = {}
        correlation_pairs = []
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                col1 = correlation_matrix.columns[i]
                col2 = correlation_matrix.columns[j]
                corr_value = correlation_matrix.iloc[i, j]
                
                if corr_value > self.max_correlation:
                    pair = (col1, col2)
                    high_correlations[f"{col1}__{col2}"] = corr_value
                    correlation_pairs.append({
                        'feature1': col1,
                        'feature2': col2,
                        'correlation': corr_value
                    })
        
        return {
            'has_high_correlations': len(high_correlations) > 0,
            'high_correlations': high_correlations,
            'correlation_pairs': correlation_pairs
        }
    
    def _check_target_distribution(self, y: pd.Series) -> Dict[str, Any]:
        """Check target variable distribution."""
        value_counts = y.value_counts()
        min_samples = value_counts.min()
        max_samples = value_counts.max()
        imbalance_ratio = max_samples / min_samples if min_samples > 0 else float('inf')
        
        return {
            'class_distribution': value_counts.to_dict(),
            'min_samples_per_class': int(min_samples),
            'max_samples_per_class': int(max_samples),
            'imbalance_ratio': imbalance_ratio,
            'is_imbalanced': imbalance_ratio > 5,
            'insufficient_samples': min_samples < self.min_samples_per_class
        }

# ====================================================================
# MAIN PREPROCESSING PIPELINE
# ====================================================================

class AdvancedPreprocessor(BasePreprocessor):
    """
    Main preprocessing pipeline that orchestrates all preprocessing steps.
    Handles scaling, outliers, missing values, and validation.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        if config is None:
            config = get_preprocessing_config()
        super().__init__(config)
        
        # Initialize components
        self.validator = DataValidator(self.config.get('validation', {}))
        self.missing_handler = MissingValueHandler(self.config.get('missing_values', {}))
        self.outlier_handler = OutlierHandler(self.config.get('outliers', {}))
        self.scaler = MultiScaler(self.config.get('scaling', {}))
        
        self.validation_results_ = {}
        self.preprocessing_steps_ = []
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'AdvancedPreprocessor':
        """Fit all preprocessing components."""
        self._validate_input(X)
        
        self.logger.info("Starting preprocessing pipeline fitting...")
        
        # Step 1: Validate data
        self.logger.info("Step 1: Validating data...")
        self.validator.fit(X, y)
        validation_results = self.validator.validate(X, y)
        self.validation_results_ = validation_results
        
        if not validation_results['passed']:
            raise ValidationError(f"Data validation failed: {validation_results['errors']}")
        
        # Step 2: Handle missing values
        if validation_results['checks']['missing_values']['has_missing']:
            self.logger.info("Step 2: Fitting missing value handler...")
            self.missing_handler.fit(X, y)
            X = self.missing_handler.transform(X)
            self.preprocessing_steps_.append('missing_values')
        
        # Step 3: Handle outliers
        self.logger.info("Step 3: Fitting outlier handler...")
        self.outlier_handler.fit(X, y)
        if self.outlier_handler.action != 'remove':  # Don't transform if removing
            X = self.outlier_handler.transform(X)
        self.preprocessing_steps_.append('outliers')
        
        # Step 4: Scale features
        self.logger.info("Step 4: Fitting scaler...")
        self.scaler.fit(X, y)
        self.preprocessing_steps_.append('scaling')
        
        self.is_fitted = True
        self.logger.info("Preprocessing pipeline fitted successfully!")
        
        self._update_feature_info(X, X)  # Update with final X
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all preprocessing transformations."""
        if not self.is_fitted:
            raise PreprocessingError("AdvancedPreprocessor must be fitted before transform")
        
        self._validate_input(X)
        X_transformed = X.copy()
        
        self.logger.info("Applying preprocessing pipeline...")
        
        # Apply transformations in order
        if 'missing_values' in self.preprocessing_steps_:
            X_transformed = self.missing_handler.transform(X_transformed)
            self.logger.info("Applied missing value imputation")
        
        if 'outliers' in self.preprocessing_steps_:
            X_transformed = self.outlier_handler.transform(X_transformed)
            self.logger.info("Applied outlier handling")
        
        if 'scaling' in self.preprocessing_steps_:
            X_transformed = self.scaler.transform(X_transformed)
            self.logger.info("Applied feature scaling")
        
        self.logger.info(f"Preprocessing complete: {X.shape} -> {X_transformed.shape}")
        return X_transformed
    
    def get_preprocessing_report(self) -> Dict[str, Any]:
        """Get comprehensive preprocessing report."""
        report = {
            'validation_results': self.validation_results_,
            'preprocessing_steps': self.preprocessing_steps_,
            'components': {
                'validator': self.validator.get_params(),
                'missing_handler': self.missing_handler.get_params(),
                'outlier_handler': self.outlier_handler.get_params(),
                'scaler': self.scaler.get_params()
            },
            'feature_info': {
                'input_features': self.feature_names_in_,
                'output_features': self.feature_names_out_,
                'n_features_in': self.n_features_in_,
                'n_features_out': self.n_features_out_
            }
        }
        return report

if __name__ == "__main__":
    # Test the preprocessing components
    print("✅ Advanced preprocessing components loaded successfully!")
    print(f"🔧 Available components: MultiScaler, OutlierHandler, MissingValueHandler, DataValidator, AdvancedPreprocessor")
    
    # Create sample data for testing
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.exponential(2, 1000),
        'feature3': np.random.uniform(-5, 5, 1000)
    })
    
    # Add some missing values and outliers
    sample_data.loc[sample_data.sample(50).index, 'feature1'] = np.nan
    sample_data.loc[sample_data.sample(20).index, 'feature2'] = 1000  # Outliers
    
    print(f"📊 Sample data shape: {sample_data.shape}")
    print(f"🔍 Missing values: {sample_data.isnull().sum().sum()}")
