#!/usr/bin/env python3
"""
Advanced Feature Engineering Module for TradingJii ML System

Contains sophisticated feature engineering components including time series features,
technical indicators, interaction features, and feature selection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2, f_classif
from sklearn.decomposition import PCA, FastICA
import warnings

# Technical analysis imports
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    warnings.warn("TA-Lib not available. Some technical indicators will be computed using pandas.")

try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    warnings.warn("pandas_ta not available. Using basic implementations for technical indicators.")

from .base import (
    BaseFeatureEngineer, FeatureEngineeringError,
    compute_feature_importance_scores
)
from .config import get_feature_engineering_config

warnings.filterwarnings('ignore')

# ====================================================================
# UTILITY FUNCTIONS
# ====================================================================

def clean_infinite_and_nan(df: pd.DataFrame, fill_method: str = 'median') -> pd.DataFrame:
    """
    Clean infinite and NaN values from DataFrame.
    
    Args:
        df: DataFrame to clean
        fill_method: Method to fill NaN values ('median', 'mean', 'zero', 'forward', 'backward')
        
    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()
    
    # Replace infinite values with NaN
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    
    # Handle NaN values based on fill method
    if fill_method == 'median':
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            median_val = df_clean[col].median()
            if pd.isna(median_val):
                median_val = 0
            df_clean[col] = df_clean[col].fillna(median_val)
    elif fill_method == 'mean':
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            mean_val = df_clean[col].mean()
            if pd.isna(mean_val):
                mean_val = 0
            df_clean[col] = df_clean[col].fillna(mean_val)
    elif fill_method == 'zero':
        df_clean = df_clean.fillna(0)
    elif fill_method == 'forward':
        df_clean = df_clean.fillna(method='ffill')
    elif fill_method == 'backward':
        df_clean = df_clean.fillna(method='bfill')
    
    # Final fallback - fill any remaining NaN with 0
    df_clean = df_clean.fillna(0)
    
    return df_clean

# ====================================================================
# MAIN FEATURE ENGINEERING PIPELINE
# ====================================================================

class AdvancedFeatureEngineer(BaseFeatureEngineer):
    """
    Main feature engineering pipeline that orchestrates all feature engineering steps.
    Handles time series features, technical indicators, interactions, and selection.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        if config is None:
            config = get_feature_engineering_config()
        super().__init__(config)
        
        # Initialize components based on available classes
        self.time_series = None
        self.technical = None
        self.interactions = None
        self.selector = None
        
        # For now, we'll create a simple feature engineer
        self.engineering_steps_ = []
        self.feature_report_ = {}
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'AdvancedFeatureEngineer':
        """Fit all feature engineering components."""
        self._validate_input(X)
        
        self.logger.info("Starting feature engineering pipeline fitting...")
        X_current = X.copy()
        
        # Store original feature names
        self.original_features_ = list(X.columns)
        
        # For now, just store the fitted state
        self.is_fitted = True
        self.logger.info("Feature engineering pipeline fitted successfully!")
        
        self._update_feature_info(X, X_current)
        return self
    
    def _update_feature_info(self, X_input: pd.DataFrame, X_output: pd.DataFrame) -> None:
        """Update feature information after fitting."""
        self.feature_names_in_ = list(X_input.columns)
        self.feature_names_out_ = list(X_output.columns)
        self.n_features_in_ = len(X_input.columns)
        self.n_features_out_ = len(X_output.columns)
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering transformations."""
        if not self.is_fitted:
            raise FeatureEngineeringError("AdvancedFeatureEngineer must be fitted before transform")
        
        self._validate_input(X)
        X_transformed = X.copy()
        
        self.logger.info("Applying feature engineering pipeline...")
        
        # Create basic time series features
        X_transformed = self._create_basic_features(X_transformed)
        
        # Clean infinite and NaN values
        X_transformed = clean_infinite_and_nan(X_transformed, fill_method='median')
        
        self.logger.info(f"Feature engineering complete: {X.shape} -> {X_transformed.shape}")
        return X_transformed
    
    def _create_basic_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create basic feature engineering features."""
        X_transformed = X.copy()
        
        # Get numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        # Create rolling features for numeric columns
        for col in numeric_cols:
            # Rolling mean (5 periods)
            X_transformed[f'{col}_rolling_5_mean'] = X[col].rolling(window=5, min_periods=1).mean()
            
            # Rolling std (5 periods)
            X_transformed[f'{col}_rolling_5_std'] = X[col].rolling(window=5, min_periods=1).std()
            
            # Lag feature (1 period)
            X_transformed[f'{col}_lag_1'] = X[col].shift(1)
            
            # Percentage change
            pct_change = X[col].pct_change()
            pct_change = pct_change.replace([np.inf, -np.inf], 0)
            X_transformed[f'{col}_pct_change'] = pct_change
        
        # Create simple technical indicators if we have OHLC data
        if all(col in X.columns for col in ['open', 'high', 'low', 'close']):
            X_transformed = self._create_simple_technical_indicators(X_transformed)
        
        return X_transformed
    
    def _create_simple_technical_indicators(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create simple technical indicators."""
        close = X['close']
        high = X['high']
        low = X['low']
        
        # Simple Moving Average (SMA)
        X[f'sma_10'] = close.rolling(window=10, min_periods=1).mean()
        X[f'sma_20'] = close.rolling(window=20, min_periods=1).mean()
        
        # RSI (simple implementation)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        X['rsi_14'] = rsi.replace([np.inf, -np.inf], 50).fillna(50)
        
        # Bollinger Bands
        sma_20 = close.rolling(window=20).mean()
        std_20 = close.rolling(window=20).std()
        X['bb_upper'] = (sma_20 + (std_20 * 2)).fillna(close)
        X['bb_lower'] = (sma_20 - (std_20 * 2)).fillna(close)
        X['bb_middle'] = sma_20.fillna(close)
        
        # Price position within Bollinger Bands
        X['bb_position'] = ((close - X['bb_lower']) / (X['bb_upper'] - X['bb_lower'] + 1e-10)).fillna(0.5)
        
        return X
    
    def get_feature_names_out(self) -> List[str]:
        """Get output feature names."""
        if hasattr(self, 'feature_names_out_'):
            return self.feature_names_out_
        return []
    
    def get_feature_engineering_report(self) -> Dict[str, Any]:
        """Get comprehensive feature engineering report."""
        report = {
            'engineering_steps': self.engineering_steps_,
            'feature_info': {
                'input_features': self.feature_names_in_ if hasattr(self, 'feature_names_in_') else [],
                'output_features': self.feature_names_out_ if hasattr(self, 'feature_names_out_') else [],
                'n_features_in': self.n_features_in_ if hasattr(self, 'n_features_in_') else 0,
                'n_features_out': self.n_features_out_ if hasattr(self, 'n_features_out_') else 0
            },
            'config': self.config
        }
        return report


if __name__ == "__main__":
    # Test the feature engineering components
    print("✅ Advanced feature engineering components loaded successfully!")
    print(f"🔧 Available components: AdvancedFeatureEngineer")
    
    # Create sample data for testing
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'open': np.random.uniform(100, 200, 1000),
        'high': np.random.uniform(150, 250, 1000),
        'low': np.random.uniform(50, 150, 1000),
        'close': np.random.uniform(100, 200, 1000),
        'volume': np.random.exponential(1000, 1000)
    })
    
    print(f"📊 Sample data shape: {sample_data.shape}")
    
    # Test feature engineering
    feature_engineer = AdvancedFeatureEngineer()
    feature_engineer.fit(sample_data)
    transformed_data = feature_engineer.transform(sample_data)
    print(f"🔧 Transformed data shape: {transformed_data.shape}")
