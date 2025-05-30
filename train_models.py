#!/usr/bin/env python3
"""
Binary Classification Training Pipeline for TradingJii
====================================================

Converts 3-class system (BUY, SELL, HOLD) to binary (BUY/SELL) with runtime confidence management.

Features:
1. Loads merged.csv and excludes HOLD labels (0)
2. Remaps BUY(1)→1, SELL(2)→0 (or vice versa)
3. Applies SMOTE balancing (optional)
4. Trains binary models with class_weight='balanced'
5. Saves best model as best_binary_model.pkl
6. Maintains existing preprocessing & feature engineering
7. Auto-training on all available datasets when no arguments provided
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
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb

# Custom modules
from modules.ml.preprocessor import AdvancedPreprocessor
from modules.ml.feature_engineer import AdvancedFeatureEngineer
from modules.ml.config import (
    get_model_config, get_preprocessing_config, get_feature_engineering_config,
    get_training_config
)
from modules.ml.base import (
    create_time_based_splits, validate_data_integrity, ModelTrainingError
)
from modules.utils.logging_setup import setup_logging

# Initialize colorama
colorama.init()
warnings.filterwarnings('ignore')

# ====================================================================
# BINARY CLASSIFICATION CONFIGURATION
# ====================================================================

BINARY_MODEL_CLASSES = {
    'RandomForest': RandomForestClassifier,
    'XGBoost': xgb.XGBClassifier, 
    'LightGBM': lgb.LGBMClassifier,
    'MLP': MLPClassifier
}

BINARY_CLASS_MAPPING = {
    'BUY_AS_1': {1: 1, 2: 0},   # BUY=1, SELL=0
    'SELL_AS_1': {1: 0, 2: 1}   # BUY=0, SELL=1
}

def print_header(text: str, color: str = Fore.CYAN) -> None:
    """Print formatted header."""
    print(f"\n{color}{'='*70}")
    print(f"{color}{text.center(70)}")
    print(f"{color}{'='*70}{Style.RESET_ALL}")

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

def discover_available_datasets() -> List[Tuple[str, str]]:
    """Discover all available symbol/timeframe combinations."""
    ml_datasets_path = Path("ml_datasets")
    combinations = []
    
    if not ml_datasets_path.exists():
        print_warning("ml_datasets directory not found")
        return combinations
    
    for symbol_dir in ml_datasets_path.iterdir():
        if symbol_dir.is_dir():
            symbol = symbol_dir.name
            for timeframe_dir in symbol_dir.iterdir():
                if timeframe_dir.is_dir():
                    timeframe = timeframe_dir.name
                    merged_file = timeframe_dir / "merged.csv"
                    if merged_file.exists():
                        combinations.append((symbol, timeframe))
    
    return combinations

def create_output_directories() -> None:
    """Create necessary output directories."""
    directories = [
        'ml_system/models/binary_models',
        'ml_system/reports/binary_training',
        'ml_system/logs/predictions'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def load_and_prepare_binary_dataset(symbol: str, timeframe: str, 
                                  mapping_strategy: str = 'BUY_AS_1',
                                  use_smote: bool = True) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """
    Load merged dataset and convert to binary classification.
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe 
        mapping_strategy: 'BUY_AS_1' or 'SELL_AS_1'
        use_smote: Whether to apply SMOTE balancing
        
    Returns:
        Tuple of (X, y, conversion_info)
    """
    dataset_path = Path(f"ml_datasets/{symbol}/{timeframe}/merged.csv")
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    print_step("BINARY DATA LOADING", f"Loading {dataset_path}")
    df = pd.read_csv(dataset_path)
    print_success(f"Loaded dataset: {df.shape[0]:,} rows, {df.shape[1]:,} columns")
    
    # Original distribution
    original_dist = dict(df['label'].value_counts().sort_index())
    print(f"Original distribution: {original_dist}")
    
    # Filter out HOLD labels (0)
    print_step("FILTERING", "Removing HOLD labels (0)")
    df_filtered = df[df['label'] != 0].copy()
    after_filter_dist = dict(df_filtered['label'].value_counts().sort_index())
    print_success(f"After filtering: {df_filtered.shape[0]:,} rows")
    print(f"Filtered distribution: {after_filter_dist}")
    
    # Remap labels to binary
    print_step("REMAPPING", f"Converting to binary using {mapping_strategy}")
    label_mapping = BINARY_CLASS_MAPPING[mapping_strategy]
    df_filtered['label'] = df_filtered['label'].map(label_mapping)
    
    binary_dist = dict(df_filtered['label'].value_counts().sort_index())
    print_success(f"Binary distribution: {binary_dist}")
    
    # Prepare features and target
    exclude_cols = ['label', 'timestamp', 'datetime']
    feature_cols = [col for col in df_filtered.columns if col not in exclude_cols]
    
    X = df_filtered[feature_cols].copy()
    y = df_filtered['label'].copy()
    
    # Apply SMOTE if requested
    conversion_info = {
        'original_samples': len(df),
        'filtered_samples': len(df_filtered),
        'original_distribution': original_dist,
        'filtered_distribution': after_filter_dist,
        'binary_distribution': binary_dist,
        'mapping_strategy': mapping_strategy,
        'label_mapping': label_mapping,
        'smote_applied': False
    }
    
    if use_smote and len(binary_dist) == 2:
        print_step("SMOTE BALANCING", "Applying SMOTE for class balancing")
        try:
            smote = SMOTE(random_state=42, k_neighbors=min(5, min(binary_dist.values())-1))
            X_balanced, y_balanced = smote.fit_resample(X, y)
            
            balanced_dist = dict(pd.Series(y_balanced).value_counts().sort_index())
            print_success(f"SMOTE applied: {X.shape[0]} → {X_balanced.shape[0]} samples")
            print(f"Balanced distribution: {balanced_dist}")
            
            conversion_info['smote_applied'] = True
            conversion_info['balanced_distribution'] = balanced_dist
            
            X, y = X_balanced, y_balanced
            
        except Exception as e:
            print_warning(f"SMOTE failed: {e}, continuing without balancing")
    
    print_success(f"Final binary dataset: {X.shape[1]} features, {len(y)} samples")
    return X, y, conversion_info

def get_binary_model_config(model_name: str) -> Dict[str, Any]:
    """Get model configuration adapted for binary classification."""
    config = get_model_config(model_name.lower()).copy()
    
    # Ensure class_weight is set for models that support it
    if model_name in ['RandomForest', 'LightGBM']:
        config['class_weight'] = 'balanced'
    
    # Adjust XGBoost for binary classification
    if model_name == 'XGBoost':
        config['objective'] = 'binary:logistic'
        config['eval_metric'] = 'logloss'
    
    # Adjust LightGBM for binary classification
    if model_name == 'LightGBM':
        config['objective'] = 'binary'
        config['metric'] = 'binary_logloss'
    
    return config

def train_binary_model(X: pd.DataFrame, y: pd.Series, model_name: str,
                      conversion_info: Dict) -> Dict[str, Any]:
    """Train and evaluate a binary classification model."""
    print_step("BINARY MODEL TRAINING", f"Training {model_name}")
    
    if model_name not in BINARY_MODEL_CLASSES:
        raise ValueError(f"Unknown model: {model_name}")
    
    config = get_binary_model_config(model_name)
    ModelClass = BINARY_MODEL_CLASSES[model_name]
    model = ModelClass(**config)
    
    try:
        # Time-based cross-validation
        print_step("CROSS-VALIDATION", "Performing temporal cross-validation")
        tscv_splits = create_time_based_splits(pd.DataFrame(index=X.index), n_splits=5, gap=24)
        
        fold_scores = []
        fold_roc_curves = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv_splits):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            fold_result = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='binary'),
                'recall': recall_score(y_test, y_pred, average='binary'),
                'f1': f1_score(y_test, y_pred, average='binary'),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            # Store ROC curve data
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            fold_roc_curves.append({'fpr': fpr, 'tpr': tpr, 'auc': fold_result['roc_auc']})
            
            fold_scores.append(fold_result)
            print(f"  Fold {fold+1}: Accuracy={fold_result['accuracy']:.4f}, AUC={fold_result['roc_auc']:.4f}")
        
        # Aggregate CV scores
        cv_scores = {}
        for metric in fold_scores[0].keys():
            scores = [fold[metric] for fold in fold_scores]
            cv_scores[metric] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'scores': scores
            }
        
        # Train final model on full dataset
        print_step("FINAL TRAINING", "Training on full dataset")
        model.fit(X, y)
        
        # Feature importance
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
            'roc_curves': fold_roc_curves,
            'feature_importance': feature_importance,
            'config': config,
            'conversion_info': conversion_info,
            'success': True,
            'error': None
        }
        
        print_success(f"{model_name} training completed")
        print(f"CV Accuracy: {cv_scores['accuracy']['mean']:.4f} ± {cv_scores['accuracy']['std']:.4f}")
        print(f"CV ROC-AUC: {cv_scores['roc_auc']['mean']:.4f} ± {cv_scores['roc_auc']['std']:.4f}")
        print(f"CV F1-Score: {cv_scores['f1']['mean']:.4f} ± {cv_scores['f1']['std']:.4f}")
        
        return result
        
    except Exception as e:
        print_error(f"{model_name} training failed: {str(e)}")
        return {
            'model_name': model_name,
            'model': None,
            'cv_scores': None,
            'roc_curves': None,
            'feature_importance': None,
            'config': config,
            'conversion_info': conversion_info,
            'success': False,
            'error': str(e)
        }

def select_best_binary_model(results: Dict[str, Any]) -> Tuple[str, Any]:
    """Select best model based on ROC-AUC score."""
    print_step("MODEL SELECTION", "Selecting best model based on CV ROC-AUC")
    
    successful_models = {
        name: result for name, result in results.items() 
        if result.get('success', False)
    }
    
    if not successful_models:
        raise ModelTrainingError("No models trained successfully")
    
    best_model_name = None
    best_auc_score = -1
    
    for model_name, result in successful_models.items():
        auc_mean = result['cv_scores']['roc_auc']['mean']
        if auc_mean > best_auc_score:
            best_auc_score = auc_mean
            best_model_name = model_name
    
    best_result = successful_models[best_model_name]
    
    print_success(f"Best model: {best_model_name} (CV ROC-AUC: {best_auc_score:.4f})")
    return best_model_name, best_result

def save_binary_model_bundle(symbol: str, timeframe: str, best_result: Dict[str, Any],
                           preprocessor: AdvancedPreprocessor, 
                           feature_engineer: AdvancedFeatureEngineer) -> str:
    """Save complete model bundle with preprocessing pipeline."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create model bundle
    model_bundle = {
        'model': best_result['model'],
        'preprocessor': preprocessor,
        'feature_engineer': feature_engineer,
        'model_name': best_result['model_name'],
        'cv_scores': best_result['cv_scores'],
        'conversion_info': best_result['conversion_info'],
        'config': best_result['config'],
        'symbol': symbol,
        'timeframe': timeframe,
        'training_timestamp': timestamp,
        'version': '1.0.0_binary'
    }
    
    # Save paths
    model_dir = Path(f"ml_system/models/binary_models")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Main model file
    model_file = model_dir / f"best_binary_model_{symbol}_{timeframe}.pkl"
    joblib.dump(model_bundle, model_file)
    
    # Generic "best" model (latest trained)
    best_model_file = model_dir / "best_binary_model.pkl"
    joblib.dump(model_bundle, best_model_file)
    
    print_success(f"Model bundle saved: {model_file}")
    print_success(f"Best model updated: {best_model_file}")
    
    return str(model_file)

def generate_training_report(symbol: str, timeframe: str, results: Dict[str, Any],
                           best_result: Dict[str, Any]) -> None:
    """Generate detailed training report."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_dir = Path("ml_system/reports/binary_training")
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Training report
    report_file = report_dir / f"binary_training_report_{symbol}_{timeframe}_{timestamp}.csv"
    
    report_data = []
    for model_name, result in results.items():
        if result.get('success', False):
            cv_scores = result['cv_scores']
            row = {
                'symbol': symbol,
                'timeframe': timeframe,
                'model_name': model_name,
                'cv_accuracy_mean': cv_scores['accuracy']['mean'],
                'cv_accuracy_std': cv_scores['accuracy']['std'],
                'cv_precision_mean': cv_scores['precision']['mean'],
                'cv_precision_std': cv_scores['precision']['std'],
                'cv_recall_mean': cv_scores['recall']['mean'],
                'cv_recall_std': cv_scores['recall']['std'],
                'cv_f1_mean': cv_scores['f1']['mean'],
                'cv_f1_std': cv_scores['f1']['std'],
                'cv_roc_auc_mean': cv_scores['roc_auc']['mean'],
                'cv_roc_auc_std': cv_scores['roc_auc']['std'],
                'is_best_model': model_name == best_result['model_name'],
                'training_timestamp': timestamp
            }
            
            # Add conversion info
            conv_info = result['conversion_info']
            row.update({
                'original_samples': conv_info['original_samples'],
                'filtered_samples': conv_info['filtered_samples'],
                'mapping_strategy': conv_info['mapping_strategy'],
                'smote_applied': conv_info['smote_applied']
            })
            
            report_data.append(row)
    
    report_df = pd.DataFrame(report_data)
    report_df.to_csv(report_file, index=False)
    print_success(f"Training report saved: {report_file}")
    
    # Feature importance report
    if best_result.get('feature_importance') is not None:
        fi_file = report_dir / f"binary_feature_importance_{symbol}_{timeframe}_{timestamp}.csv"
        fi_df = pd.DataFrame({
            'feature': best_result['feature_importance'].index,
            'importance': best_result['feature_importance'].values,
            'rank': range(1, len(best_result['feature_importance']) + 1)
        })
        fi_df.to_csv(fi_file, index=False)
        print_success(f"Feature importance saved: {fi_file}")

def train_single_combination(symbol: str, timeframe: str, models: List[str],
                           mapping: str, use_smote: bool) -> bool:
    """Train models for a single symbol/timeframe combination."""
    try:
        print_header(f"🎯 TRAINING: {symbol} ({timeframe})", Fore.BLUE)
        
        # Load and prepare binary dataset
        X, y, conversion_info = load_and_prepare_binary_dataset(
            symbol, timeframe, mapping, use_smote
        )
        
        # Apply preprocessing pipeline
        print_step("PREPROCESSING", "Applying advanced preprocessing pipeline")
        preprocessor = AdvancedPreprocessor(get_preprocessing_config())
        X_processed = preprocessor.fit_transform(X, y)
        print_success(f"Preprocessing: {X.shape} -> {X_processed.shape}")
        
        # Apply feature engineering
        print_step("FEATURE ENGINEERING", "Applying feature engineering")
        feature_engineer = AdvancedFeatureEngineer(get_feature_engineering_config())
        X_final = feature_engineer.fit_transform(X_processed)
        print_success(f"Feature engineering: {X_processed.shape} -> {X_final.shape}")
        
        # Train models
        results = {}
        for model_name in models:
            result = train_binary_model(X_final, y, model_name, conversion_info)
            results[model_name] = result
        
        # Select best model
        best_model_name, best_result = select_best_binary_model(results)
        
        # Save model bundle
        model_file = save_binary_model_bundle(
            symbol, timeframe, best_result, 
            preprocessor, feature_engineer
        )
        
        # Generate reports
        generate_training_report(symbol, timeframe, results, best_result)
        
        print_header(f"✅ {symbol} ({timeframe}) COMPLETED", Fore.GREEN)
        print(f"Best Model: {Fore.YELLOW}{best_model_name}{Style.RESET_ALL}")
        print(f"CV ROC-AUC: {Fore.GREEN}{best_result['cv_scores']['roc_auc']['mean']:.4f} ± {best_result['cv_scores']['roc_auc']['std']:.4f}{Style.RESET_ALL}")
        
        return True
        
    except Exception as e:
        print_error(f"Training failed for {symbol} ({timeframe}): {str(e)}")
        return False

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="TradingJii Binary Classification Training")
    
    parser.add_argument('--symbol', '-s', type=str, 
                       help='Trading symbol (if not specified, trains on all available)')
    parser.add_argument('--timeframe', '-t', type=str,
                       choices=['1h', '4h', '1d'], 
                       help='Timeframe (if not specified, trains on all available)')
    parser.add_argument('--models', '-m', nargs='+',
                       choices=['RandomForest', 'XGBoost', 'LightGBM', 'MLP'],
                       default=['RandomForest', 'XGBoost', 'LightGBM'],
                       help='Models to train')
    parser.add_argument('--mapping', type=str, choices=['BUY_AS_1', 'SELL_AS_1'],
                       default='BUY_AS_1', help='Binary mapping strategy')
    parser.add_argument('--no-smote', action='store_true',
                       help='Disable SMOTE balancing')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    return parser.parse_args()

def main():
    """Main training pipeline execution."""
    args = parse_arguments()
    
    logger = setup_logging(logging.INFO)
    create_output_directories()
    
    # Determine combinations to train
    if args.symbol and args.timeframe:
        # Train specific combination
        combinations = [(args.symbol, args.timeframe)]
        mode = "SPECIFIC"
    elif args.symbol:
        # Train specific symbol, all timeframes
        available = discover_available_datasets()
        combinations = [(s, t) for s, t in available if s == args.symbol]
        mode = f"SYMBOL: {args.symbol}"
    elif args.timeframe:
        # Train specific timeframe, all symbols
        available = discover_available_datasets()
        combinations = [(s, t) for s, t in available if t == args.timeframe]
        mode = f"TIMEFRAME: {args.timeframe}"
    else:
        # Train all available combinations
        combinations = discover_available_datasets()
        mode = "ALL AVAILABLE"
    
    if not combinations:
        print_error("No datasets found to train on!")
        print_warning("Make sure to generate ML datasets first:")
        print_warning("python real_time.py --timeframes 1h --generate-ml-datasets")
        sys.exit(1)
    
    print_header("🚀 TRADINGJII BINARY CLASSIFICATION TRAINING 🚀", Fore.BLUE)
    print(f"{Fore.CYAN}Training Mode: {mode}")
    print(f"Combinations to train: {len(combinations)}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Mapping Strategy: {args.mapping}")
    print(f"SMOTE: {'Disabled' if args.no_smote else 'Enabled'}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}")
    
    # Show combinations
    print(f"\n{Fore.YELLOW}📋 Training Schedule:{Style.RESET_ALL}")
    for i, (symbol, timeframe) in enumerate(combinations, 1):
        print(f"  {i:2d}. {symbol} ({timeframe})")
    
    successful_trains = 0
    failed_trains = 0
    
    # Train each combination
    for i, (symbol, timeframe) in enumerate(combinations, 1):
        print(f"\n{Back.MAGENTA}{Fore.WHITE} PROGRESS: {i}/{len(combinations)} {Style.RESET_ALL}")
        
        success = train_single_combination(
            symbol, timeframe, args.models, 
            args.mapping, not args.no_smote
        )
        
        if success:
            successful_trains += 1
        else:
            failed_trains += 1
    
    # Final summary
    print_header("🎯 TRAINING SUMMARY", Fore.GREEN if failed_trains == 0 else Fore.YELLOW)
    print(f"Total combinations: {len(combinations)}")
    print(f"{Fore.GREEN}Successful: {successful_trains}{Style.RESET_ALL}")
    if failed_trains > 0:
        print(f"{Fore.RED}Failed: {failed_trains}{Style.RESET_ALL}")
    
    if successful_trains > 0:
        print(f"\n{Fore.GREEN}✅ Training completed! Models saved to ml_system/models/binary_models/{Style.RESET_ALL}")
        print(f"{Fore.BLUE}💡 Run prediction test: python predict.py --test{Style.RESET_ALL}")
        print(f"{Fore.BLUE}🚀 Start real-time system: python real_time.py --timeframes 1h{Style.RESET_ALL}")
    
    if failed_trains > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
