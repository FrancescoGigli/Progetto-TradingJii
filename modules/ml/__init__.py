#!/usr/bin/env python3
"""
TradingJii ML Package

Advanced machine learning components for cryptocurrency trading.
Includes preprocessing, feature engineering, model training, and monitoring.
"""

from .config import (
    PREPROCESSING_CONFIG,
    FEATURE_ENGINEERING_CONFIG,
    MODEL_CONFIG,
    TRAINING_CONFIG
)

from .base import (
    BasePreprocessor,
    BaseFeatureEngineer,
    BaseValidator,
    MLError,
    PreprocessingError,
    FeatureEngineeringError,
    ValidationError
)

from .preprocessor import (
    AdvancedPreprocessor,
    MultiScaler,
    OutlierHandler,
    MissingValueHandler,
    DataValidator
)

from .feature_engineer import (
    AdvancedFeatureEngineer
)

__version__ = "1.0.0"
__author__ = "TradingJii ML Team"

__all__ = [
    # Config
    'PREPROCESSING_CONFIG',
    'FEATURE_ENGINEERING_CONFIG', 
    'MODEL_CONFIG',
    'TRAINING_CONFIG',
    
    # Base classes
    'BasePreprocessor',
    'BaseFeatureEngineer',
    'BaseValidator',
    'MLError',
    'PreprocessingError',
    'FeatureEngineeringError',
    'ValidationError',
    
    # Preprocessor
    'AdvancedPreprocessor',
    'MultiScaler',
    'OutlierHandler',
    'MissingValueHandler',
    'DataValidator',
    
    # Feature Engineering
    'AdvancedFeatureEngineer'
]
