#!/usr/bin/env python3
"""
Base Classes and Utilities for TradingJii ML System

Contains abstract base classes, custom exceptions, and common utilities
for the machine learning pipeline.
"""

import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import joblib
import json
from datetime import datetime
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ====================================================================
# CUSTOM EXCEPTIONS
# ====================================================================

class MLError(Exception):
    """Base exception class for ML operations."""
    def __init__(self, message: str, details: Optional[Dict] = None):
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.now()
        super().__init__(self.message)
    
    def __str__(self):
        return f"{self.__class__.__name__}: {self.message}"

class PreprocessingError(MLError):
    """Exception raised during preprocessing operations."""
    pass

class FeatureEngineeringError(MLError):
    """Exception raised during feature engineering operations."""
    pass

class ValidationError(MLError):
    """Exception raised during validation operations."""
    pass

class ModelTrainingError(MLError):
    """Exception raised during model training operations."""
    pass

class PredictionError(MLError):
    """Exception raised during prediction operations."""
    pass

# ====================================================================
# BASE CLASSES
# ====================================================================

class BaseMLComponent(ABC):
    """Abstract base class for all ML components."""
    
    def __init__(self, config: Optional[Dict] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize the base ML component.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config or {}
        self.logger = logger or self._setup_logger()
        self.is_fitted = False
        self.metadata = {
            'created_at': datetime.now(),
            'version': '1.0.0',
            'component_type': self.__class__.__name__
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for the component."""
        logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'BaseMLComponent':
        """Fit the component to the data."""
        pass
    
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data."""
        pass
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit the component and transform the data."""
        return self.fit(X, y).transform(X)
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save the component to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save the main component
            joblib.dump(self, filepath)
            
            # Save metadata separately
            metadata_path = filepath.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                # Convert datetime to string for JSON serialization
                metadata = self.metadata.copy()
                metadata['created_at'] = metadata['created_at'].isoformat()
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Component saved to {filepath}")
            
        except Exception as e:
            raise MLError(f"Failed to save component: {str(e)}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'BaseMLComponent':
        """Load the component from disk."""
        filepath = Path(filepath)
        
        try:
            component = joblib.load(filepath)
            
            # Load metadata if available
            metadata_path = filepath.with_suffix('.json')
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    metadata['created_at'] = datetime.fromisoformat(metadata['created_at'])
                    component.metadata.update(metadata)
            
            component.logger.info(f"Component loaded from {filepath}")
            return component
            
        except Exception as e:
            raise MLError(f"Failed to load component: {str(e)}")
    
    def get_params(self) -> Dict[str, Any]:
        """Get component parameters."""
        return self.config.copy()
    
    def set_params(self, **params) -> 'BaseMLComponent':
        """Set component parameters."""
        self.config.update(params)
        return self
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get component metadata."""
        return self.metadata.copy()

class BasePreprocessor(BaseMLComponent):
    """Abstract base class for preprocessing components."""
    
    def __init__(self, config: Optional[Dict] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config, logger)
        self.feature_names_in_ = None
        self.feature_names_out_ = None
        self.n_features_in_ = None
        self.n_features_out_ = None
    
    def _validate_input(self, X: pd.DataFrame) -> None:
        """Validate input data."""
        if not isinstance(X, pd.DataFrame):
            raise PreprocessingError("Input must be a pandas DataFrame")
        
        if X.empty:
            raise PreprocessingError("Input DataFrame is empty")
        
        if self.is_fitted and self.feature_names_in_ is not None:
            missing_features = set(self.feature_names_in_) - set(X.columns)
            if missing_features:
                raise PreprocessingError(
                    f"Missing features in input: {missing_features}"
                )
    
    def _update_feature_info(self, X_in: pd.DataFrame, X_out: pd.DataFrame) -> None:
        """Update feature information after fitting."""
        self.feature_names_in_ = list(X_in.columns)
        self.feature_names_out_ = list(X_out.columns)
        self.n_features_in_ = len(X_in.columns)
        self.n_features_out_ = len(X_out.columns)

class BaseFeatureEngineer(BaseMLComponent):
    """Abstract base class for feature engineering components."""
    
    def __init__(self, config: Optional[Dict] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config, logger)
        self.feature_names_in_ = None
        self.feature_names_out_ = None
        self.new_features_ = None
    
    def _validate_input(self, X: pd.DataFrame) -> None:
        """Validate input data."""
        if not isinstance(X, pd.DataFrame):
            raise FeatureEngineeringError("Input must be a pandas DataFrame")
        
        if X.empty:
            raise FeatureEngineeringError("Input DataFrame is empty")
    
    def get_new_features(self) -> List[str]:
        """Get list of newly created features."""
        return self.new_features_ or []

class BaseValidator(BaseMLComponent):
    """Abstract base class for validation components."""
    
    def __init__(self, config: Optional[Dict] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config, logger)
        self.validation_results_ = {}
    
    @abstractmethod
    def validate(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Validate the data and return validation results."""
        pass
    
    def get_validation_results(self) -> Dict[str, Any]:
        """Get validation results."""
        return self.validation_results_.copy()

# ====================================================================
# UTILITY FUNCTIONS
# ====================================================================

def validate_data_integrity(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
    """
    Validate data integrity and return comprehensive report.
    
    Args:
        df: DataFrame to validate
        target_col: Name of target column
        
    Returns:
        Dictionary with validation results
    """
    results = {
        'shape': df.shape,
        'memory_usage': df.memory_usage(deep=True).sum(),
        'missing_values': {},
        'infinite_values': {},
        'duplicate_rows': 0,
        'data_types': {},
        'warnings': [],
        'errors': []
    }
    
    # Check missing values
    missing = df.isnull().sum()
    results['missing_values'] = {
        col: {'count': int(count), 'percentage': count / len(df) * 100}
        for col, count in missing.items() if count > 0
    }
    
    # Check infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            results['infinite_values'][col] = {
                'count': int(inf_count),
                'percentage': inf_count / len(df) * 100
            }
    
    # Check duplicates
    results['duplicate_rows'] = int(df.duplicated().sum())
    
    # Data types
    results['data_types'] = {col: str(dtype) for col, dtype in df.dtypes.items()}
    
    # Generate warnings
    if results['missing_values']:
        results['warnings'].append("Missing values detected")
    
    if results['infinite_values']:
        results['warnings'].append("Infinite values detected")
    
    if results['duplicate_rows'] > 0:
        results['warnings'].append(f"{results['duplicate_rows']} duplicate rows found")
    
    # Target validation
    if target_col and target_col in df.columns:
        target_info = analyze_target_distribution(df[target_col])
        results['target_analysis'] = target_info
        
        if target_info.get('imbalance_ratio', 0) > 10:
            results['warnings'].append("Highly imbalanced target variable")
    
    return results

def analyze_target_distribution(y: pd.Series) -> Dict[str, Any]:
    """
    Analyze target variable distribution.
    
    Args:
        y: Target variable series
        
    Returns:
        Dictionary with distribution analysis
    """
    value_counts = y.value_counts()
    total_samples = len(y)
    
    analysis = {
        'unique_values': len(value_counts),
        'class_distribution': value_counts.to_dict(),
        'class_percentages': (value_counts / total_samples * 100).to_dict(),
        'most_common_class': value_counts.index[0],
        'least_common_class': value_counts.index[-1],
        'imbalance_ratio': value_counts.iloc[0] / value_counts.iloc[-1] if len(value_counts) > 1 else 1.0
    }
    
    return analysis

def compute_feature_importance_scores(X: pd.DataFrame, y: pd.Series, method: str = 'mutual_info') -> pd.Series:
    """
    Compute feature importance scores using various methods.
    
    Args:
        X: Feature matrix
        y: Target variable
        method: Method to use ('mutual_info', 'chi2', 'f_score')
        
    Returns:
        Series with feature importance scores
    """
    from sklearn.feature_selection import mutual_info_classif, chi2, f_classif
    from sklearn.preprocessing import MinMaxScaler
    
    # Ensure all values are non-negative for chi2
    if method == 'chi2':
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    else:
        X_scaled = X
    
    if method == 'mutual_info':
        scores = mutual_info_classif(X_scaled, y, random_state=42)
    elif method == 'chi2':
        scores, _ = chi2(X_scaled, y)
    elif method == 'f_score':
        scores, _ = f_classif(X_scaled, y)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return pd.Series(scores, index=X.columns).sort_values(ascending=False)

def detect_outliers(df: pd.DataFrame, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
    """
    Detect outliers in numerical columns.
    
    Args:
        df: DataFrame to analyze
        method: Detection method ('iqr', 'zscore', 'isolation_forest')
        threshold: Threshold for outlier detection
        
    Returns:
        Boolean DataFrame indicating outliers
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outliers = pd.DataFrame(False, index=df.index, columns=df.columns)
    
    for col in numeric_cols:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers[col] = z_scores > threshold
            
        elif method == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_pred = iso_forest.fit_predict(df[[col]])
            outliers[col] = outlier_pred == -1
    
    return outliers

def create_time_based_splits(df: pd.DataFrame, n_splits: int = 5, 
                           gap: int = 0) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create time-based train/test splits for time series data.
    
    Args:
        df: DataFrame with time-ordered data
        n_splits: Number of splits
        gap: Gap between train and test sets
        
    Returns:
        List of (train_indices, test_indices) tuples
    """
    n_samples = len(df)
    test_size = n_samples // (n_splits + 1)
    
    splits = []
    for i in range(n_splits):
        test_start = (i + 1) * test_size
        test_end = test_start + test_size
        train_end = test_start - gap
        
        if train_end <= 0 or test_end > n_samples:
            break
        
        train_indices = np.arange(0, train_end)
        test_indices = np.arange(test_start, min(test_end, n_samples))
        
        splits.append((train_indices, test_indices))
    
    return splits

def safe_divide(numerator: Union[float, np.ndarray], 
                denominator: Union[float, np.ndarray], 
                default: float = 0.0) -> Union[float, np.ndarray]:
    """
    Safely divide two numbers/arrays, handling division by zero.
    
    Args:
        numerator: Numerator value(s)
        denominator: Denominator value(s)
        default: Default value for division by zero
        
    Returns:
        Result of division with safe handling of division by zero
    """
    if isinstance(denominator, (int, float)):
        return numerator / denominator if denominator != 0 else default
    else:
        return np.where(denominator != 0, numerator / denominator, default)

def memory_usage_mb(df: pd.DataFrame) -> float:
    """Get memory usage of DataFrame in MB."""
    return df.memory_usage(deep=True).sum() / 1024 / 1024

def reduce_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce memory usage of DataFrame by optimizing data types.
    
    Args:
        df: DataFrame to optimize
        
    Returns:
        Optimized DataFrame
    """
    df_optimized = df.copy()
    
    for col in df_optimized.columns:
        col_type = df_optimized[col].dtype
        
        if col_type != 'object':
            c_min = df_optimized[col].min()
            c_max = df_optimized[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df_optimized[col] = df_optimized[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df_optimized[col] = df_optimized[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df_optimized[col] = df_optimized[col].astype(np.int32)
                    
            elif str(col_type)[:5] == 'float':
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df_optimized[col] = df_optimized[col].astype(np.float32)
    
    return df_optimized

if __name__ == "__main__":
    # Test the base classes and utilities
    print("✅ Base classes and utilities loaded successfully!")
    print(f"📝 Available exceptions: {[cls.__name__ for cls in [MLError, PreprocessingError, FeatureEngineeringError, ValidationError]]}")
    print(f"🔧 Available base classes: {[cls.__name__ for cls in [BaseMLComponent, BasePreprocessor, BaseFeatureEngineer, BaseValidator]]}")
    print(f"🛠️ Available utilities: validate_data_integrity, analyze_target_distribution, compute_feature_importance_scores, detect_outliers")
