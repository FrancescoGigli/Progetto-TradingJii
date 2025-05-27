#!/usr/bin/env python3
"""
ML Configuration Module for TradingJii

Contains all configuration settings for preprocessing, feature engineering,
model training, and validation.
"""

import numpy as np
from typing import Dict, List, Any, Optional

# ====================================================================
# PREPROCESSING CONFIGURATION
# ====================================================================

PREPROCESSING_CONFIG = {
    'scaling': {
        'method': 'robust',  # 'standard', 'minmax', 'robust', 'quantile'
        'feature_range': (0, 1),
        'quantile_range': (25.0, 75.0),
        'clip': True
    },
    
    'outliers': {
        'method': 'iqr',  # 'iqr', 'zscore', 'isolation_forest', 'lof'
        'threshold': 1.5,
        'zscore_threshold': 3.0,
        'contamination': 0.1,
        'n_neighbors': 20,
        'action': 'clip'  # 'remove', 'clip', 'transform'
    },
    
    'missing_values': {
        'method': 'knn',  # 'mean', 'median', 'mode', 'knn', 'iterative', 'forward_fill'
        'n_neighbors': 5,
        'max_iter': 10,
        'random_state': 42,
        'initial_strategy': 'mean'
    },
    
    'feature_selection': {
        'method': 'mutual_info',  # 'variance', 'correlation', 'mutual_info', 'rfe', 'lasso'
        'variance_threshold': 0.01,
        'correlation_threshold': 0.95,
        'k_best': 50,
        'alpha': 0.01,
        'random_state': 42
    },
    
    'validation': {
        'check_infinite': True,
        'check_missing': True,
        'check_duplicates': True,
        'check_target_balance': True,
        'min_samples_per_class': 10,
        'max_correlation': 0.95
    }
}

# ====================================================================
# FEATURE ENGINEERING CONFIGURATION
# ====================================================================

FEATURE_ENGINEERING_CONFIG = {
    'time_features': {
        'rolling_windows': [3, 5, 10, 20, 50],
        'rolling_functions': ['mean', 'std', 'min', 'max', 'median'],
        'lag_features': [1, 2, 3, 5, 10],
        'diff_features': [1, 2],
        'seasonal_features': True,
        'trend_features': True,
        'cyclic_features': True
    },
    
    'technical_indicators': {
        'momentum': {
            'rsi_periods': [14, 21],
            'stoch_periods': [14],
            'williams_r_periods': [14],
            'roc_periods': [12, 25],
            'momentum_periods': [10]
        },
        'trend': {
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'adx_periods': [14],
            'aroon_periods': [14, 25],
            'cci_periods': [20]
        },
        'volatility': {
            'bbands_periods': [20],
            'atr_periods': [14, 21],
            'kc_periods': [20],
            'donchian_periods': [20]
        },
        'volume': {
            'obv': True,
            'ad': True,
            'cmf_periods': [20],
            'mfi_periods': [14]
        }
    },
    
    'interactions': {
        'max_degree': 2,
        'selection_method': 'mutual_info',  # 'correlation', 'mutual_info', 'chi2'
        'max_features': 100,
        'include_bias': False
    },
    
    'polynomial': {
        'degree': 2,
        'interaction_only': True,
        'include_bias': False,
        'selection_threshold': 0.01
    },
    
    'decomposition': {
        'use_pca': False,
        'n_components': 0.95,  # Variance to retain
        'use_ica': False,
        'ica_components': 10
    }
}

# ====================================================================
# MODEL CONFIGURATION
# ====================================================================

MODEL_CONFIG = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'sqrt',
        'bootstrap': True,
        'random_state': 42,
        'n_jobs': -1,
        'class_weight': 'balanced'
    },
    
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'eval_metric': 'mlogloss',
        'verbosity': 0,
        'use_label_encoder': False
    },
    
    'lightgbm': {
        'n_estimators': 100,
        'max_depth': -1,
        'learning_rate': 0.1,
        'num_leaves': 31,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'verbose': -1,
        'force_row_wise': True,
        'class_weight': 'balanced'
    },
    
    'mlp': {
        'hidden_layer_sizes': (100, 50),
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 0.0001,
        'learning_rate': 'constant',
        'learning_rate_init': 0.001,
        'max_iter': 500,
        'shuffle': True,
        'random_state': 42,
        'early_stopping': True,
        'validation_fraction': 0.1,
        'n_iter_no_change': 10
    },
    
    'svm': {
        'C': 1.0,
        'kernel': 'rbf',
        'gamma': 'scale',
        'probability': True,
        'random_state': 42,
        'class_weight': 'balanced'
    },
    
    'logistic_regression': {
        'C': 1.0,
        'penalty': 'l2',
        'solver': 'liblinear',
        'max_iter': 1000,
        'random_state': 42,
        'class_weight': 'balanced'
    }
}

# ====================================================================
# TRAINING CONFIGURATION
# ====================================================================

TRAINING_CONFIG = {
    'data_split': {
        'test_size': 0.2,
        'validation_size': 0.1,
        'shuffle': False,  # Temporal split for time series
        'stratify': True,
        'random_state': 42
    },
    
    'cross_validation': {
        'method': 'time_series_split',  # 'kfold', 'stratified_kfold', 'time_series_split'
        'n_splits': 5,
        'shuffle': False,
        'random_state': 42,
        'gap': 0,  # Gap between train and test for time series
        'max_train_size': None
    },
    
    'hyperparameter_tuning': {
        'method': 'optuna',  # 'grid_search', 'random_search', 'optuna'
        'n_trials': 100,
        'timeout': 3600,  # seconds
        'n_jobs': -1,
        'random_state': 42,
        'pruning': True,
        'cv_folds': 3
    },
    
    'imbalanced_learning': {
        'use_sampling': True,
        'method': 'smote',  # 'smote', 'adasyn', 'random_over', 'random_under'
        'sampling_strategy': 'auto',
        'random_state': 42,
        'k_neighbors': 5
    },
    
    'ensemble': {
        'use_voting': True,
        'voting_type': 'soft',  # 'hard', 'soft'
        'use_stacking': True,
        'meta_learner': 'logistic_regression',
        'use_blending': False,
        'blend_ratio': 0.5
    },
    
    'early_stopping': {
        'monitor': 'val_loss',
        'patience': 10,
        'min_delta': 0.001,
        'restore_best_weights': True
    },
    
    'logging': {
        'log_level': 'INFO',
        'log_to_file': True,
        'log_dir': 'ml_system/reports/training',
        'experiment_tracking': True,
        'mlflow_uri': 'ml_system/models/experiments'
    }
}

# ====================================================================
# EVALUATION CONFIGURATION
# ====================================================================

EVALUATION_CONFIG = {
    'metrics': {
        'classification': [
            'accuracy', 'precision', 'recall', 'f1',
            'roc_auc', 'log_loss', 'matthews_corrcoef'
        ],
        'per_class': True,
        'confusion_matrix': True,
        'classification_report': True
    },
    
    'plots': {
        'confusion_matrix': True,
        'roc_curve': True,
        'precision_recall_curve': True,
        'feature_importance': True,
        'learning_curves': True,
        'validation_curves': True
    },
    
    'interpretability': {
        'use_shap': True,
        'shap_sample_size': 1000,
        'feature_importance': True,
        'partial_dependence': True,
        'lime_explanations': False
    }
}

# ====================================================================
# MONITORING CONFIGURATION
# ====================================================================

MONITORING_CONFIG = {
    'data_drift': {
        'method': 'evidently',
        'reference_period': '30d',
        'drift_threshold': 0.1,
        'features_to_monitor': 'all'
    },
    
    'model_performance': {
        'performance_threshold': 0.05,  # Degradation threshold
        'monitoring_window': '7d',
        'alert_channels': ['log', 'email'],
        'retrain_threshold': 0.1
    },
    
    'prediction_monitoring': {
        'confidence_threshold': 0.7,
        'uncertainty_threshold': 0.3,
        'outlier_threshold': 0.05
    }
}

# ====================================================================
# CLASS MAPPINGS
# ====================================================================

CLASS_MAPPINGS = {
    'labels': {
        0: 'HOLD',
        1: 'BUY', 
        2: 'SELL'
    },
    'colors': {
        0: '#FFA500',  # Orange for HOLD
        1: '#00FF00',  # Green for BUY
        2: '#FF0000'   # Red for SELL
    }
}

# ====================================================================
# HELPER FUNCTIONS
# ====================================================================

def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get configuration for a specific model."""
    return MODEL_CONFIG.get(model_name, {})

def get_preprocessing_config() -> Dict[str, Any]:
    """Get preprocessing configuration."""
    return PREPROCESSING_CONFIG.copy()

def get_feature_engineering_config() -> Dict[str, Any]:
    """Get feature engineering configuration."""
    return FEATURE_ENGINEERING_CONFIG.copy()

def get_training_config() -> Dict[str, Any]:
    """Get training configuration."""
    return TRAINING_CONFIG.copy()

def update_config(config_dict: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Update configuration with new values."""
    config_copy = config_dict.copy()
    config_copy.update(updates)
    return config_copy

def get_class_name(label: int) -> str:
    """Get class name from label."""
    return CLASS_MAPPINGS['labels'].get(label, f'Unknown_{label}')

def get_class_color(label: int) -> str:
    """Get color for class label."""
    return CLASS_MAPPINGS['colors'].get(label, '#CCCCCC')

# ====================================================================
# VALIDATION
# ====================================================================

def validate_config() -> bool:
    """Validate all configurations."""
    try:
        # Basic validation
        assert isinstance(PREPROCESSING_CONFIG, dict)
        assert isinstance(FEATURE_ENGINEERING_CONFIG, dict)
        assert isinstance(MODEL_CONFIG, dict)
        assert isinstance(TRAINING_CONFIG, dict)
        
        # Check required keys
        required_preprocessing = ['scaling', 'outliers', 'missing_values']
        for key in required_preprocessing:
            assert key in PREPROCESSING_CONFIG
        
        required_training = ['data_split', 'cross_validation']
        for key in required_training:
            assert key in TRAINING_CONFIG
            
        return True
    except AssertionError:
        return False

if __name__ == "__main__":
    # Test configuration validation
    if validate_config():
        print("✅ All configurations are valid!")
        print(f"📊 Preprocessing methods: {list(PREPROCESSING_CONFIG.keys())}")
        print(f"🔧 Feature engineering features: {list(FEATURE_ENGINEERING_CONFIG.keys())}")
        print(f"🤖 Available models: {list(MODEL_CONFIG.keys())}")
        print(f"🚀 Training configurations: {list(TRAINING_CONFIG.keys())}")
    else:
        print("❌ Configuration validation failed!")
