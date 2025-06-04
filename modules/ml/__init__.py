#!/usr/bin/env python3
"""
ML Module for TradingJii

This module provides machine learning capabilities for cryptocurrency trading signal generation:
- Model training and management
- Real-time prediction
- Feature extraction from market data
- Signal generation with confidence scoring
- Robust error handling and fallback mechanisms
"""

from .config import (
    MODEL_CONFIG,
    SIGNAL_THRESHOLDS, 
    FALLBACK_CONFIG,
    LOGGING_CONFIG,
    VALIDATION_CONFIG,
    FEATURE_CONFIG,
    MLConfigError,
    ModelNotFoundError,
    FeatureValidationError,
    SchemaValidationError,
    InsufficientDataError,
    get_model_config,
    get_threshold_config,
    validate_config,
    get_available_models,
    get_ensemble_models
)

__version__ = "1.0.0"
__author__ = "TradingJii ML Team"

# Module exports
__all__ = [
    # Configuration
    "MODEL_CONFIG",
    "SIGNAL_THRESHOLDS", 
    "FALLBACK_CONFIG",
    "LOGGING_CONFIG",
    "VALIDATION_CONFIG",
    "FEATURE_CONFIG",
    
    # Exceptions
    "MLConfigError",
    "ModelNotFoundError",
    "FeatureValidationError", 
    "SchemaValidationError",
    "InsufficientDataError",
    
    # Utility functions
    "get_model_config",
    "get_threshold_config",
    "validate_config",
    "get_available_models",
    "get_ensemble_models"
]
