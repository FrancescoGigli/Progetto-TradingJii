#!/usr/bin/env python3
"""
ML Configuration Module for TradingJii

Contains all ML-specific configurations including:
- Model configurations with versioning
- Signal thresholds 
- Fallback configurations
- Logging settings
"""

import os
import hashlib
from datetime import datetime
from typing import Dict, List, Optional

# Model Configuration with versioning and validation
MODEL_CONFIG = {
    "volatility_classifier": {
        "path": "models/volatility_classifier_*.pkl",  # Pattern-based loading
        "version": "v1.2.0",
        "type": "classification",
        "features": ["vol_window_7", "vol_trend", "vol_std"],
        "feature_columns": ["x_1", "x_2", "x_3", "x_4", "x_5", "x_6", "x_7"],  # Exact order from training
        "accuracy": 0.78,
        "trained_on": "2025-01-01",
        "min_samples": 32,
        "description": "Volatility pattern classifier based on 7-day windows",
        "target_classes": ["HOLD", "BUY", "SELL"],
        "feature_schema_hash": None,  # Will be calculated dynamically
        "symbol_specific": True,  # This model has symbol-specific variants
        "pattern_match": "volatility_classifier_{symbol}_{timestamp}.pkl"
    },
    "trend_predictor": {
        "path": "models/trend_model_v1.0.1.pkl", 
        "version": "v1.0.1",
        "type": "classification",
        "features": ["rsi14", "macd", "ema20_slope", "macd_signal"],
        "feature_columns": ["rsi14", "macd", "macd_signal", "ema20"],
        "accuracy": 0.73,
        "trained_on": "2024-12-15",
        "min_samples": 24,
        "description": "Technical analysis trend predictor",
        "target_classes": ["HOLD", "BUY", "SELL"],
        "feature_schema_hash": None,
        "enabled": False  # Disabled because no actual model file exists
    },
    "ensemble_v1": {
        "type": "ensemble",
        "version": "v1.0.0",
        "models": ["volatility_classifier"],  # Only volatility classifier available
        "weights": [1.0],  # Single model ensemble
        "description": "Volatility-based ensemble (single model due to missing trend predictor)",
        "min_models_required": 1,  # Minimum models needed for ensemble to work
        "target_classes": ["HOLD", "BUY", "SELL"],
        "enabled": True
    }
}

# Signal threshold configurations
SIGNAL_THRESHOLDS = {
    "conservative": {
        "buy": 0.7,
        "sell": 0.7, 
        "hold_range": 0.4,
        "description": "High confidence required for signals"
    },
    "moderate": {
        "buy": 0.6,
        "sell": 0.6,
        "hold_range": 0.3,
        "description": "Balanced risk/reward approach"
    },
    "aggressive": {
        "buy": 0.55,
        "sell": 0.55,
        "hold_range": 0.2,
        "description": "Lower threshold for more frequent signals"
    },
    "default": "moderate"
}

# Fallback and error handling configuration
FALLBACK_CONFIG = {
    "default_model": "volatility_classifier",
    "fallback_model": None,  # No fallback model available - prevents recursion
    "default_threshold": "moderate",
    "min_confidence": 0.5,
    "max_retries": 2,  # Reduced to prevent excessive retries
    "enable_graceful_degradation": True,
    "fallback_signal": "HOLD",  # Safe default when all else fails
    "max_fallback_rate": 0.2,  # Alert if fallback rate > 20%
    "enable_health_monitoring": True,
    "recursion_depth_limit": 3,  # Prevent infinite recursion
    "safe_mode_enabled": True  # Always return HOLD if all models fail
}

# Logging configuration
LOGGING_CONFIG = {
    "log_dir": "logs",
    "predictor_log": "logs/predictor_run.log",
    "fallback_log": "logs/fallback_stats.log", 
    "system_health_log": "logs/system_health.log",
    "audit_log": "logs/audit_trail.log",
    "log_level": "INFO",
    "max_log_size_mb": 50,
    "log_retention_days": 30,
    "enable_console_logging": True,
    "enable_file_logging": True
}

# System validation settings
VALIDATION_CONFIG = {
    "enable_pre_checks": True,
    "validate_models_on_startup": True,
    "validate_features_on_predict": True,
    "enable_schema_validation": True,
    "enable_data_quality_checks": True,
    "min_data_quality_score": 0.8,
    "enable_performance_monitoring": True
}

# Feature extraction settings
FEATURE_CONFIG = {
    "volatility_window_size": 7,
    "ta_lookback_periods": 50,  # For technical analysis indicators
    "min_data_points": 100,     # Minimum data points needed for reliable features
    "enable_feature_scaling": True,
    "feature_cache_enabled": False,  # Set to True for production performance
    "cache_ttl_minutes": 30
}

# Custom exception classes
class MLConfigError(Exception):
    """Base exception for ML configuration errors"""
    pass

class ModelNotFoundError(MLConfigError):
    """Raised when a model file is not found"""
    pass

class FeatureValidationError(MLConfigError):
    """Raised when feature validation fails"""
    pass

class SchemaValidationError(MLConfigError):
    """Raised when schema validation fails"""
    pass

class InsufficientDataError(MLConfigError):
    """Raised when insufficient data is available"""
    pass

# Utility functions
def generate_schema_hash(dtypes, columns) -> str:
    """
    Generate a hash for feature schema validation
    
    Args:
        dtypes: pandas dtype series
        columns: list of column names
        
    Returns:
        Hash string for validation
    """
    schema_str = f"{list(columns)}_{list(dtypes.astype(str))}"
    return hashlib.md5(schema_str.encode()).hexdigest()[:12]

def get_model_config(model_name: str) -> Dict:
    """
    Get configuration for a specific model
    
    Args:
        model_name: Name of the model
        
    Returns:
        Model configuration dictionary
        
    Raises:
        ModelNotFoundError: If model is not configured
    """
    if model_name not in MODEL_CONFIG:
        raise ModelNotFoundError(f"Model '{model_name}' not found in configuration")
    
    return MODEL_CONFIG[model_name].copy()

def get_threshold_config(threshold_name: str = None) -> Dict:
    """
    Get threshold configuration
    
    Args:
        threshold_name: Name of threshold profile or None for default
        
    Returns:
        Threshold configuration dictionary
    """
    if threshold_name is None:
        threshold_name = SIGNAL_THRESHOLDS["default"]
    
    if threshold_name not in SIGNAL_THRESHOLDS:
        threshold_name = SIGNAL_THRESHOLDS["default"]
    
    return SIGNAL_THRESHOLDS[threshold_name].copy()

def validate_config() -> bool:
    """
    Validate the entire ML configuration
    
    Returns:
        True if configuration is valid
        
    Raises:
        MLConfigError: If configuration is invalid
    """
    # Check that default model exists
    default_model = FALLBACK_CONFIG["default_model"]
    if default_model not in MODEL_CONFIG:
        raise MLConfigError(f"Default model '{default_model}' not configured")
    
    # Check that fallback model exists
    fallback_model = FALLBACK_CONFIG.get("fallback_model")
    if fallback_model and fallback_model not in MODEL_CONFIG:
        raise MLConfigError(f"Fallback model '{fallback_model}' not configured")
    
    # Check that default threshold exists
    default_threshold = SIGNAL_THRESHOLDS["default"]
    if default_threshold not in SIGNAL_THRESHOLDS:
        raise MLConfigError(f"Default threshold '{default_threshold}' not configured")
    
    # Validate ensemble configurations
    for model_name, config in MODEL_CONFIG.items():
        if config["type"] == "ensemble":
            for sub_model in config["models"]:
                if sub_model not in MODEL_CONFIG:
                    raise MLConfigError(f"Ensemble model '{model_name}' references unknown model '{sub_model}'")
    
    # Check log directory exists
    log_dir = LOGGING_CONFIG["log_dir"]
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    return True

def get_available_models() -> List[str]:
    """
    Get list of available model names
    
    Returns:
        List of model names
    """
    return [name for name, config in MODEL_CONFIG.items() if config["type"] != "ensemble"]

def get_ensemble_models() -> List[str]:
    """
    Get list of available ensemble model names
    
    Returns:
        List of ensemble model names
    """
    return [name for name, config in MODEL_CONFIG.items() if config["type"] == "ensemble"]

# Initialize and validate configuration on import
try:
    validate_config()
except Exception as e:
    print(f"Warning: ML configuration validation failed: {e}")
