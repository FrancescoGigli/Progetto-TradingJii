#!/usr/bin/env python3
"""
Tabular ML Models Training Script for TradingJii

This script trains multiple machine learning models for cryptocurrency trading signal classification.
It loads merged datasets, preprocesses features, trains models, and generates comprehensive reports.

Models trained:
- RandomForestClassifier
- XGBoostClassifier  
- LightGBMClassifier
- MLPClassifier

Usage:
    python train_tabular_models.py --timeframe 1h
    python train_tabular_models.py --timeframe 4h --symbol BTC_USDTUSDT
    python train_tabular_models.py --timeframe 1h --verbose
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from datetime import datetime

# ML imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)
import xgboost as xgb
import lightgbm as lgb

# Local imports
from colorama import Fore, Style, init
from modules.utils.logging_setup import setup_logging

# Initialize colorama for Windows compatibility
init()

def setup_directories():
    """Create necessary directories for models and reports."""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    return models_dir

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train tabular ML models for trading signal classification"
    )
    
    parser.add_argument(
        "--timeframe", 
        type=str, 
        required=True,
        choices=["1h", "4h", "1d"],
        help="Timeframe for training (e.g., 1h, 4h)"
    )
    
    parser.add_argument(
        "--symbol", 
        type=str, 
        default=None,
        help="Specific symbol to train on (e.g., BTC_USDTUSDT). If not specified, trains on all available symbols."
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save trained models (default: models)"
    )
    
    return parser.parse_args()

def load_and_combine_datasets(timeframe: str, symbol: Optional[str] = None) -> pd.DataFrame:
    """
    Load and combine all merged datasets for the specified timeframe.
    
    Args:
        timeframe: The timeframe (e.g., "1h", "4h")
        symbol: Optional specific symbol to load. If None, loads all symbols.
        
    Returns:
        Combined DataFrame with all datasets
    """
    ml_datasets_dir = Path("ml_datasets")
    
    if not ml_datasets_dir.exists():
        logging.error(f"ML datasets directory not found: {ml_datasets_dir}")
        return pd.DataFrame()
    
    datasets = []
    symbols_loaded = []
    
    # Get list of symbol directories
    if symbol:
        symbol_dirs = [ml_datasets_dir / symbol] if (ml_datasets_dir / symbol).exists() else []
    else:
        symbol_dirs = [d for d in ml_datasets_dir.iterdir() if d.is_dir()]
    
    if not symbol_dirs:
        logging.error(f"No symbol directories found in {ml_datasets_dir}")
        return pd.DataFrame()
    
    # Load datasets from each symbol
    for symbol_dir in symbol_dirs:
        dataset_file = symbol_dir / timeframe / "merged.csv"
        
        if dataset_file.exists():
            try:
                df = pd.read_csv(dataset_file)
                if not df.empty:
                    # Add symbol column for tracking
                    df['symbol'] = symbol_dir.name
                    datasets.append(df)
                    symbols_loaded.append(symbol_dir.name)
                    logging.info(f"Loaded {len(df)} records from {symbol_dir.name} ({timeframe})")
                else:
                    logging.warning(f"Empty dataset found: {dataset_file}")
            except Exception as e:
                logging.error(f"Error loading {dataset_file}: {e}")
        else:
            logging.warning(f"Dataset not found: {dataset_file}")
    
    if not datasets:
        logging.error(f"No datasets found for timeframe {timeframe}")
        return pd.DataFrame()
    
    # Combine all datasets
    combined_df = pd.concat(datasets, ignore_index=True)
    
    # Remove rows with NaN or missing labels
    initial_rows = len(combined_df)
    combined_df = combined_df.dropna()
    
    # Ensure label column exists and is valid
    if 'label' not in combined_df.columns:
        logging.error("Label column not found in datasets")
        return pd.DataFrame()
    
    # Remove rows with invalid labels (should be 0, 1, or 2)
    valid_labels = combined_df['label'].isin([0, 1, 2])
    combined_df = combined_df[valid_labels]
    
    final_rows = len(combined_df)
    dropped_rows = initial_rows - final_rows
    
    if dropped_rows > 0:
        logging.warning(f"Dropped {dropped_rows} rows ({dropped_rows/initial_rows:.1%}) due to NaN or invalid labels")
    
    logging.info(f"Combined dataset: {final_rows} records from {len(symbols_loaded)} symbols")
    logging.info(f"Symbols loaded: {', '.join(symbols_loaded)}")
    
    return combined_df

def prepare_features_and_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features (X) and labels (y) from the combined dataset.
    
    Args:
        df: Combined DataFrame
        
    Returns:
        Tuple of (X, y) where X contains features and y contains labels
    """
    # Exclude non-feature columns
    exclude_columns = ['label', 'timestamp', 'pattern', 'symbol']
    feature_columns = [col for col in df.columns if col not in exclude_columns]
    
    X = df[feature_columns].copy()
    y = df['label'].copy()
    
    # Log feature information
    logging.info(f"Features prepared: {len(feature_columns)} columns")
    logging.debug(f"Feature columns: {feature_columns}")
    
    # Log class distribution
    class_counts = y.value_counts().sort_index()
    logging.info("Class distribution:")
    for label, count in class_counts.items():
        class_name = {0: "HOLD", 1: "BUY", 2: "SELL"}.get(label, f"Unknown({label})")
        percentage = count / len(y) * 100
        logging.info(f"  {class_name} ({label}): {count} ({percentage:.1f}%)")
    
    return X, y

def evaluate_model(name: str, model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
    """
    Evaluate a trained model and return comprehensive metrics.
    
    Args:
        name: Model name
        model: Trained model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary containing all evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Per-class metrics
    precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
    
    # Weighted averages
    precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Print detailed results
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"Model: {Fore.YELLOW}{name}{Style.RESET_ALL}")
    print(f"{'='*60}{Style.RESET_ALL}")
    
    print(f"\n{Fore.GREEN}Overall Metrics:{Style.RESET_ALL}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted Precision: {precision_weighted:.4f}")
    print(f"Weighted Recall: {recall_weighted:.4f}")
    print(f"Weighted F1-Score: {f1_weighted:.4f}")
    
    print(f"\n{Fore.GREEN}Per-Class Metrics:{Style.RESET_ALL}")
    class_names = ["HOLD", "BUY", "SELL"]
    for i, class_name in enumerate(class_names):
        if i < len(precision_per_class):
            print(f"{class_name} (Class {i}):")
            print(f"  Precision: {precision_per_class[i]:.4f}")
            print(f"  Recall: {recall_per_class[i]:.4f}")
    
    print(f"\n{Fore.GREEN}Classification Report:{Style.RESET_ALL}")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    print(f"\n{Fore.GREEN}Confusion Matrix:{Style.RESET_ALL}")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Show probability predictions for a sample
    if y_proba is not None:
        print(f"\n{Fore.GREEN}Sample Probability Predictions:{Style.RESET_ALL}")
        sample_indices = np.random.choice(len(X_test), min(10, len(X_test)), replace=False)
        for idx in sample_indices:
            actual = y_test.iloc[idx]
            proba = y_proba[idx]
            predicted = y_pred[idx]
            print(f"Sample {idx}: Actual={actual}, Predicted={predicted}, "
                  f"Proba=[HOLD: {proba[0]:.3f}, BUY: {proba[1]:.3f}, SELL: {proba[2]:.3f}]")
    
    # Prepare metrics dictionary for saving
    metrics = {
        'model_name': name,
        'accuracy': accuracy,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'precision_0': precision_per_class[0] if len(precision_per_class) > 0 else 0,
        'recall_0': recall_per_class[0] if len(recall_per_class) > 0 else 0,
        'precision_1': precision_per_class[1] if len(precision_per_class) > 1 else 0,
        'recall_1': recall_per_class[1] if len(recall_per_class) > 1 else 0,
        'precision_2': precision_per_class[2] if len(precision_per_class) > 2 else 0,
        'recall_2': recall_per_class[2] if len(recall_per_class) > 2 else 0,
    }
    
    return metrics

def save_training_report(metrics_list: List[Dict], timeframe: str, output_dir: Path):
    """
    Save training report to CSV file.
    
    Args:
        metrics_list: List of metrics dictionaries from all models
        timeframe: Timeframe used for training
        output_dir: Output directory for saving reports
    """
    if not metrics_list:
        logging.warning("No metrics to save in training report")
        return
    
    # Create DataFrame from metrics
    report_df = pd.DataFrame(metrics_list)
    
    # Add metadata
    report_df['timeframe'] = timeframe
    report_df['training_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Reorder columns for better readability
    column_order = [
        'model_name', 'timeframe', 'training_date', 'accuracy',
        'precision_weighted', 'recall_weighted', 'f1_weighted',
        'precision_0', 'recall_0', 'precision_1', 'recall_1', 
        'precision_2', 'recall_2'
    ]
    
    report_df = report_df[column_order]
    
    # Save report
    report_file = output_dir / f"training_report_{timeframe}.csv"
    
    # Check if file exists to determine if we should append or create new
    if report_file.exists():
        # Append to existing file
        existing_df = pd.read_csv(report_file)
        combined_df = pd.concat([existing_df, report_df], ignore_index=True)
        combined_df.to_csv(report_file, index=False)
        logging.info(f"Appended training report to: {report_file}")
    else:
        # Create new file
        report_df.to_csv(report_file, index=False)
        logging.info(f"Created training report: {report_file}")

def main():
    """Main training function."""
    args = parse_arguments()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    
    # Setup directories
    output_dir = setup_directories()
    
    logging.info(f"{Fore.GREEN}Starting ML model training...{Style.RESET_ALL}")
    logging.info(f"Timeframe: {args.timeframe}")
    logging.info(f"Symbol filter: {args.symbol or 'All symbols'}")
    logging.info(f"Output directory: {output_dir}")
    
    # Load and combine datasets
    logging.info(f"\n{Fore.CYAN}Loading datasets...{Style.RESET_ALL}")
    df = load_and_combine_datasets(args.timeframe, args.symbol)
    
    if df.empty:
        logging.error("No data loaded. Exiting.")
        return
    
    # Prepare features and labels
    logging.info(f"\n{Fore.CYAN}Preparing features and labels...{Style.RESET_ALL}")
    X, y = prepare_features_and_labels(df)
    
    if X.empty or y.empty:
        logging.error("Failed to prepare features and labels. Exiting.")
        return
    
    # Split data temporally (no shuffle to preserve time order)
    logging.info(f"\n{Fore.CYAN}Splitting data (80/20, temporal split)...{Style.RESET_ALL}")
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    logging.info(f"Training set: {len(X_train)} samples")
    logging.info(f"Test set: {len(X_test)} samples")
    
    # Define models to train
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': xgb.XGBClassifier(
            random_state=42,
            eval_metric='mlogloss',
            verbosity=0
        ),
        'LightGBM': lgb.LGBMClassifier(
            random_state=42,
            verbose=-1,
            force_row_wise=True
        ),
        'MLP': MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
    }
    
    # Train models and collect metrics
    metrics_list = []
    trained_models = {}
    
    logging.info(f"\n{Fore.CYAN}Training models...{Style.RESET_ALL}")
    
    for model_name, model in models.items():
        try:
            logging.info(f"\n{Fore.YELLOW}Training {model_name}...{Style.RESET_ALL}")
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Evaluate the model
            metrics = evaluate_model(model_name, model, X_test, y_test)
            metrics_list.append(metrics)
            trained_models[model_name] = model
            
            # Save the model
            model_file = output_dir / f"{model_name}_{args.timeframe}.pkl"
            joblib.dump(model, model_file)
            logging.info(f"Saved model: {model_file}")
            
        except Exception as e:
            logging.error(f"Error training {model_name}: {e}")
            import traceback
            logging.error(traceback.format_exc())
    
    # Save training report
    if metrics_list:
        logging.info(f"\n{Fore.CYAN}Saving training report...{Style.RESET_ALL}")
        save_training_report(metrics_list, args.timeframe, output_dir)
        
        # Print summary comparison
        print(f"\n{Fore.GREEN}{'='*60}")
        print("TRAINING SUMMARY COMPARISON")
        print(f"{'='*60}{Style.RESET_ALL}")
        
        summary_df = pd.DataFrame(metrics_list)
        summary_df = summary_df.sort_values('accuracy', ascending=False)
        
        print(f"\n{Fore.YELLOW}Model Rankings by Accuracy:{Style.RESET_ALL}")
        for idx, row in summary_df.iterrows():
            print(f"{row['model_name']}: {row['accuracy']:.4f}")
        
        print(f"\n{Fore.YELLOW}Best Model: {summary_df.iloc[0]['model_name']} "
              f"(Accuracy: {summary_df.iloc[0]['accuracy']:.4f}){Style.RESET_ALL}")
    else:
        logging.warning("No models were successfully trained")
    
    logging.info(f"\n{Fore.GREEN}Training completed!{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
