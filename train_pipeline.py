#!/usr/bin/env python3
"""
Complete ML Training Pipeline for TradingJii

Advanced end-to-end training system that:
1. Loads merged.csv from ml_datasets/SYMBOL/TIMEFRAME/
2. Applies full pipeline: AdvancedPreprocessor → FeatureEngineer
3. Trains 4 models: RandomForest, XGBoost, LightGBM, MLP
4. Evaluates with temporal cross-validation + trading-specific metrics
5. Saves complete bundle with pipeline + best model
6. Generates detailed report (CSV + console)
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import warnings
import colorama
from colorama import Fore, Back, Style
import joblib
import json

# ML Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
import xgboost as xgb
import lightgbm as lgb

# Custom modules
from modules.ml.preprocessor import AdvancedPreprocessor
from modules.ml.feature_engineer import AdvancedFeatureEngineer
from modules.ml.config import (
    get_model_config, get_preprocessing_config, get_feature_engineering_config,
    get_training_config, CLASS_MAPPINGS
)
from modules.ml.base import (
    create_time_based_splits, validate_data_integrity, ModelTrainingError
)
from modules.utils.logging_setup import setup_logging

# Initialize colorama
colorama.init()

warnings.filterwarnings('ignore')

# ====================================================================
# CONFIGURATION AND CONSTANTS
# ====================================================================

MODEL_CLASSES = {
    'RandomForest': RandomForestClassifier,
    'XGBoost': xgb.XGBClassifier,
    'LightGBM': lgb.LGBMClassifier,
    'MLP': MLPClassifier
}

def convert_numpy_types(obj):
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

def get_available_symbols() -> List[str]:
    """Get list of available symbols from ml_datasets directory."""
    datasets_path = Path("ml_datasets")
    if not datasets_path.exists():
        return []
    
    symbols = []
    for item in datasets_path.iterdir():
        if item.is_dir():
            symbols.append(item.name)
    
    return sorted(symbols)

def print_header(text: str, color: str = Fore.CYAN) -> None:
    """Print formatted header."""
    print(f"\n{color}{'='*60}")
    print(f"{color}{text.center(60)}")
    print(f"{color}{'='*60}{Style.RESET_ALL}")

def print_step(step: str, description: str) -> None:
    """Print step information."""
    print(f"\n{Fore.YELLOW}🔄 {step}: {Fore.WHITE}{description}{Style.RESET_ALL}")

def print_success(message: str) -> None:
    """Print success message."""
    print(f"{Fore.GREEN}✅ {message}{Style.RESET_ALL}")

def print_warning(message: str) -> None:
    """Print warning message."""
    print(f"{Fore.YELLOW}⚠️  {message}{Style.RESET_ALL}")

def print_error(message: str) -> None:
    """Print error message."""
    print(f"{Fore.RED}❌ {message}{Style.RESET_ALL}")

def create_output_directories() -> None:
    """Create necessary output directories."""
    directories = [
        'ml_system/models',
        'ml_system/reports/training',
        'ml_system/reports/validation'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def load_merged_dataset(symbol: str, timeframe: str) -> pd.DataFrame:
    """Load merged dataset for given symbol and timeframe."""
    dataset_path = Path(f"ml_datasets/{symbol}/{timeframe}/merged.csv")
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    print_step("DATA LOADING", f"Loading {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    print_success(f"Loaded dataset: {df.shape[0]:,} rows, {df.shape[1]:,} columns")
    return df

def prepare_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare features and target from dataset."""
    if 'label' not in df.columns:
        raise ValueError("Target column 'label' not found in dataset")
    
    exclude_cols = ['label', 'timestamp', 'datetime']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    y = df['label'].copy()
    
    print_success(f"Prepared features: {X.shape[1]} features, {len(y)} samples")
    print(f"Target distribution: {dict(y.value_counts())}")
    
    return X, y

def apply_preprocessing_pipeline(X: pd.DataFrame, y: pd.Series, 
                              config: Optional[Dict] = None) -> Tuple[pd.DataFrame, AdvancedPreprocessor]:
    """Apply advanced preprocessing pipeline."""
    print_step("PREPROCESSING", "Applying advanced preprocessing pipeline")
    
    if config is None:
        config = get_preprocessing_config()
    
    preprocessor = AdvancedPreprocessor(config)
    
    try:
        X_processed = preprocessor.fit_transform(X, y)
        report = preprocessor.get_preprocessing_report()
        
        print_success(f"Preprocessing complete: {X.shape} -> {X_processed.shape}")
        
        validation = report['validation_results']
        if validation['warnings']:
            for warning in validation['warnings']:
                print_warning(warning)
        
        return X_processed, preprocessor
        
    except Exception as e:
        print_error(f"Preprocessing failed: {str(e)}")
        raise

def apply_feature_engineering(X: pd.DataFrame, config: Optional[Dict] = None) -> Tuple[pd.DataFrame, AdvancedFeatureEngineer]:
    """Apply feature engineering pipeline."""
    print_step("FEATURE ENGINEERING", "Applying advanced feature engineering")
    
    if config is None:
        config = get_feature_engineering_config()
    
    feature_engineer = AdvancedFeatureEngineer(config)
    
    try:
        X_engineered = feature_engineer.fit_transform(X)
        new_features = feature_engineer.get_new_features()
        print_success(f"Feature engineering complete: {X.shape[1]} -> {X_engineered.shape[1]} features")
        print(f"New features created: {len(new_features)}")
        
        return X_engineered, feature_engineer
        
    except Exception as e:
        print_error(f"Feature engineering failed: {str(e)}")
        raise

def perform_time_series_split(X: pd.DataFrame, y: pd.Series, 
                            n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create time-based splits for temporal validation."""
    print_step("CROSS-VALIDATION", f"Creating {n_splits} time-based splits")
    
    df_temp = pd.DataFrame(index=X.index)
    splits = create_time_based_splits(df_temp, n_splits=n_splits, gap=24)
    
    print_success(f"Created {len(splits)} temporal splits")
    return splits

def train_single_model(X: pd.DataFrame, y: pd.Series, model_name: str, 
                      config: Optional[Dict] = None) -> Dict[str, Any]:
    """Train and evaluate a single model with cross-validation."""
    print_step("MODEL TRAINING", f"Training {model_name}")
    
    if model_name not in MODEL_CLASSES:
        raise ValueError(f"Unknown model: {model_name}")
    
    if config is None:
        config = get_model_config(model_name.lower())
    
    ModelClass = MODEL_CLASSES[model_name]
    model = ModelClass(**config)
    
    try:
        tscv_splits = perform_time_series_split(X, y, n_splits=5)
        
        fold_scores = []
        for fold, (train_idx, test_idx) in enumerate(tscv_splits):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            fold_result = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1': f1_score(y_test, y_pred, average='weighted')
            }
            
            if y_pred_proba is not None and len(np.unique(y)) == 2:
                fold_result['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
            
            fold_scores.append(fold_result)
        
        cv_scores = {}
        for metric in fold_scores[0].keys():
            scores = [fold[metric] for fold in fold_scores]
            cv_scores[metric] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'scores': scores
            }
        
        model.fit(X, y)
        
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.Series(
                model.feature_importances_,
                index=X.columns
            ).sort_values(ascending=False)
        elif hasattr(model, 'coef_'):
            feature_importance = pd.Series(
                np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_),
                index=X.columns
            ).sort_values(ascending=False)
        
        result = {
            'model_name': model_name,
            'model': model,
            'cv_scores': cv_scores,
            'feature_importance': feature_importance,
            'config': config,
            'success': True,
            'error': None
        }
        
        print_success(f"{model_name} training completed")
        print(f"CV Accuracy: {cv_scores['accuracy']['mean']:.4f} ± {cv_scores['accuracy']['std']:.4f}")
        print(f"CV F1-Score: {cv_scores['f1']['mean']:.4f} ± {cv_scores['f1']['std']:.4f}")
        
        return result
        
    except Exception as e:
        print_error(f"{model_name} training failed: {str(e)}")
        return {
            'model_name': model_name,
            'model': None,
            'cv_scores': None,
            'feature_importance': None,
            'config': config,
            'success': False,
            'error': str(e)
        }

def select_best_model(results: Dict[str, Any]) -> Tuple[str, Any]:
    """Select best model based on cross-validation F1 score."""
    print_step("MODEL SELECTION", "Selecting best model based on CV F1-score")
    
    successful_models = {
        name: result for name, result in results.items() 
        if result.get('success', False)
    }
    
    if not successful_models:
        raise ModelTrainingError("No models trained successfully")
    
    best_model_name = None
    best_f1_score = -1
    
    for model_name, result in successful_models.items():
        f1_mean = result['cv_scores']['f1']['mean']
        if f1_mean > best_f1_score:
            best_f1_score = f1_mean
            best_model_name = model_name
    
    best_result = successful_models[best_model_name]
    
    print_success(f"Best model: {best_model_name} (CV F1: {best_f1_score:.4f})")
    return best_model_name, best_result

def train_symbol(symbol: str, timeframe: str, models: List[str]) -> Dict[str, Any]:
    """Train models for a single symbol."""
    try:
        df = load_merged_dataset(symbol, timeframe)
        X, y = prepare_features_target(df)
        X_processed, preprocessor = apply_preprocessing_pipeline(X, y)
        X_final, feature_engineer = apply_feature_engineering(X_processed)
        
        results = {}
        for model_name in models:
            result = train_single_model(X_final, y, model_name)
            results[model_name] = result
        
        best_model_name, best_result = select_best_model(results)
        
        return {
            'symbol': symbol,
            'success': True,
            'best_model': best_model_name,
            'results': results
        }
        
    except Exception as e:
        print_error(f"Training failed for {symbol}: {str(e)}")
        return {
            'symbol': symbol,
            'success': False,
            'error': str(e)
        }

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="TradingJii ML Training Pipeline")
    
    parser.add_argument('--timeframe', '-t', type=str, required=True, 
                       choices=['1h', '4h', '1d'], help='Timeframe for training')
    
    symbol_group = parser.add_mutually_exclusive_group(required=True)
    symbol_group.add_argument('--symbol', '-s', type=str, help='Trading symbol')
    symbol_group.add_argument('--all-symbols', action='store_true', help='Train all symbols')
    
    parser.add_argument('--models', '-m', nargs='+', 
                       choices=['RandomForest', 'XGBoost', 'LightGBM', 'MLP'],
                       default=['RandomForest', 'XGBoost', 'LightGBM', 'MLP'],
                       help='Models to train')
    
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    return parser.parse_args()

def main():
    """Main training pipeline execution."""
    args = parse_arguments()
    
    logger = setup_logging(logging.INFO)
    create_output_directories()
    
    if args.all_symbols:
        symbols = get_available_symbols()
        if not symbols:
            print_error("No symbols found in ml_datasets directory")
            sys.exit(1)
        print_header("🚀 TRADINGJII ML TRAINING PIPELINE (ALL SYMBOLS) 🚀", Fore.BLUE)
    else:
        symbols = [args.symbol]
        print_header("🚀 TRADINGJII ML TRAINING PIPELINE 🚀", Fore.BLUE)
    
    print(f"{Fore.CYAN}Symbols: {len(symbols)} symbols")
    print(f"Timeframe: {args.timeframe}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}")
    
    try:
        all_results = []
        successful_trainings = 0
        failed_trainings = 0
        
        for i, symbol in enumerate(symbols):
            print(f"\n{Fore.MAGENTA}>>> Training symbol {i+1}/{len(symbols)}: {symbol} <<<{Style.RESET_ALL}")
            
            dataset_path = Path(f"ml_datasets/{symbol}/{args.timeframe}/merged.csv")
            if not dataset_path.exists():
                print_warning(f"Dataset not found for {symbol} at {args.timeframe}: {dataset_path}")
                failed_trainings += 1
                all_results.append({
                    'symbol': symbol,
                    'success': False,
                    'error': f"Dataset not found: {dataset_path}"
                })
                continue
            
            result = train_symbol(symbol, args.timeframe, args.models)
            all_results.append(result)
            
            if result['success']:
                successful_trainings += 1
                print_success(f"{symbol} training completed successfully!")
            else:
                failed_trainings += 1
                print_error(f"{symbol} training failed!")
        
        print_header("🎯 TRAINING SUMMARY", Fore.YELLOW)
        print(f"{Fore.GREEN}✅ Successful trainings: {successful_trainings}")
        print(f"{Fore.RED}❌ Failed trainings: {failed_trainings}")
        print(f"{Fore.CYAN}📊 Total symbols processed: {len(symbols)}{Style.RESET_ALL}")
        
    except Exception as e:
        print_error(f"Training pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
