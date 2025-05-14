#!/usr/bin/env python3
"""
TFT Trainer Module for TradingJii

This module trains Temporal Fusion Transformer (TFT) models on volatility datasets:
- Scans dataset directories for each symbol and timeframe
- Loads datasets with sufficient samples
- Trains TFT models on the data
- Saves trained models to disk
- Generates model registry and evaluation metrics
"""

import os
import logging
import pandas as pd
import argparse
import json
from typing import Dict, List, Tuple, Optional, Set, Union
import pickle
from colorama import Fore, Style
from datetime import datetime
import numpy as np
import time

# Import necessary darts modules
from darts import TimeSeries
from darts.models import TFTModel
from darts.dataprocessing.transformers import Scaler
from darts.utils.likelihood_models import QuantileRegression
from darts.metrics import mape, mae, rmse

# Set PyTorch settings at the beginning
import torch
import pytorch_lightning as pl

# Configure PyTorch Lightning to be less verbose
import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*TensorBoard support.*")
warnings.filterwarnings("ignore", ".*you defined a `validation_step` but have no `val_dataloader`*")

# Disable PyTorch Lightning's excessive logging
pl._logger.setLevel(logging.WARNING)

# Print GPU information once
if torch.cuda.is_available():
    logging.info(f"{Fore.GREEN}=== HARDWARE INFORMATION ==={Style.RESET_ALL}")
    logging.info(f"{Fore.GREEN}GPU available:{Style.RESET_ALL} {torch.cuda.get_device_name(0)}")
    logging.info(f"{Fore.GREEN}GPU memory:{Style.RESET_ALL} {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    logging.info(f"{Fore.GREEN}Using PyTorch version:{Style.RESET_ALL} {torch.__version__}")
    logging.info(f"{Fore.GREEN}Setting precision to 'high' for tensor cores{Style.RESET_ALL}")
    # Set float32 matmul precision to improve performance with Tensor Cores
    torch.set_float32_matmul_precision('high')
    logging.info(f"{Fore.GREEN}=============================={Style.RESET_ALL}")
else:
    logging.info(f"{Fore.YELLOW}Warning: GPU not available. Using CPU for training (slower){Style.RESET_ALL}")

# Configure PyTorch Lightning Trainer to be less verbose
pl.trainer.trainer.Trainer.progress_bar_callback = None

# Import project modules
from modules.utils.logging_setup import setup_logging


def train_all_tft_models(
    dataset_base_dir: str = "datasets",
    model_base_dir: str = "models",
    min_samples: int = 15,
    input_size: int = 7,
    num_epochs: int = 100,
    hidden_size: int = 64,
    n_heads: int = 4,
    dropout: float = 0.1
) -> Dict[str, Dict[str, str]]:
    """
    Train TFT models for all pattern categories under each symbol and timeframe.

    Args:
        dataset_base_dir: Root path of datasets (e.g., 'datasets')
        model_base_dir: Output path to store trained models (e.g., 'models')
        min_samples: Minimum number of rows in a dataset file to allow training
        input_size: Number of input features per sample (should match x_1 to x_7)
        num_epochs: Number of training epochs
        hidden_size: Hidden layer dimension in the TFT
        n_heads: Number of attention heads
        dropout: Dropout rate

    Returns:
        A nested dictionary mapping symbols and patterns to model paths:
        {
            "BTC_USDT": {
                "1010110": "models/BTC_USDT/5m/tft_cat_1010110.pkl",
                ...
            },
            ...
        }
    """
    # Results dictionary to track all trained models
    results_dict = {}
    
    # List to collect all model info for the registry
    all_models_info = []

    # Scan dataset directory for symbols
    for symbol_dir in os.listdir(dataset_base_dir):
        symbol_path = os.path.join(dataset_base_dir, symbol_dir)
        
        # Skip if not a directory
        if not os.path.isdir(symbol_path):
            continue
        
        # Initialize inner dictionary for this symbol
        symbol_dict = {}
        results_dict[symbol_dir] = symbol_dict
        
        # Scan for timeframes
        for timeframe_dir in os.listdir(symbol_path):
            timeframe_path = os.path.join(symbol_path, timeframe_dir)
            
            # Skip if not a directory
            if not os.path.isdir(timeframe_path):
                continue
            
            # Scan for category files (cat_*.csv)
            for filename in os.listdir(timeframe_path):
                if not filename.startswith("cat_") or not filename.endswith(".csv"):
                    continue
                
                # Extract pattern from filename
                pattern = filename.replace("cat_", "").replace(".csv", "")
                dataset_path = os.path.join(timeframe_path, filename)
                
                try:
                    # Train model for this dataset
                    model_info = _train_and_save_model(
                        symbol_dir, 
                        timeframe_dir, 
                        pattern,
                        dataset_path,
                        model_base_dir,
                        min_samples,
                        input_size,
                        num_epochs,
                        hidden_size,
                        n_heads,
                        dropout
                    )
                    
                    # Add to results if training was successful
                    if model_info:
                        model_path = model_info["path"]
                        if symbol_dir not in results_dict:
                            results_dict[symbol_dir] = {}
                        results_dict[symbol_dir][pattern] = model_path
                        
                        # Add to models info for registry
                        all_models_info.append(model_info)
                        
                except Exception as e:
                    logging.error(f"Error training model for {symbol_dir}/{timeframe_dir}/{pattern}: {e}")
                    import traceback
                    logging.error(traceback.format_exc())
    
    # Create model registry
    if all_models_info:
        # Create registry directory
        registry_dir = os.path.join(model_base_dir, "registry")
        os.makedirs(registry_dir, exist_ok=True)
        
        # Save as CSV
        registry_df = pd.DataFrame(all_models_info)
        registry_csv_path = os.path.join(registry_dir, "model_registry.csv")
        registry_df.to_csv(registry_csv_path, index=False)
        
        # Save as JSON
        registry_json_path = os.path.join(registry_dir, "model_registry.json")
        with open(registry_json_path, 'w') as f:
            json.dump(all_models_info, f, indent=2)
        
        # Save training summary
        summary = {
            "trained_at": datetime.now().isoformat(),
            "total_models": len(all_models_info),
            "symbols": len(set(m["symbol"] for m in all_models_info)),
            "timeframes": set(m["timeframe"] for m in all_models_info),
            "avg_metrics": {
                "mape": sum(m["mape"] for m in all_models_info) / len(all_models_info),
                "mae": sum(m["mae"] for m in all_models_info) / len(all_models_info),
                "rmse": sum(m["rmse"] for m in all_models_info) / len(all_models_info)
            },
            "training_params": {
                "min_samples": min_samples,
                "input_size": input_size,
                "num_epochs": num_epochs,
                "hidden_size": hidden_size,
                "n_heads": n_heads,
                "dropout": dropout
            }
        }
        
        summary_path = os.path.join(registry_dir, "training_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        logging.info(f"Model registry created at: {Fore.MAGENTA}{registry_csv_path}{Style.RESET_ALL}")
    
    # Log summary
    total_models = sum(len(patterns) for patterns in results_dict.values())
    logging.info(f"\n{Fore.GREEN}=== MODEL TRAINING SUMMARY ==={Style.RESET_ALL}")
    logging.info(f"Trained {Fore.GREEN}{total_models}{Style.RESET_ALL} models "
               f"across {Fore.YELLOW}{len(results_dict)}{Style.RESET_ALL} symbols")
    logging.info(f"Models saved to: {Fore.MAGENTA}{model_base_dir}{Style.RESET_ALL}")
    
    return results_dict


def _train_and_save_model(
    symbol: str,
    timeframe: str,
    pattern: str,
    dataset_path: str,
    model_base_dir: str,
    min_samples: int,
    input_size: int,
    num_epochs: int,
    hidden_size: int,
    n_heads: int,
    dropout: float
) -> Optional[Dict]:
    """
    Train and save a TFT model for a specific dataset.
    
    Args:
        symbol: Cryptocurrency symbol
        timeframe: Time interval
        pattern: Binary pattern category
        dataset_path: Path to the dataset CSV file
        model_base_dir: Base directory to save the model
        min_samples: Minimum sample size required
        input_size: Number of input features
        num_epochs: Training epochs
        hidden_size: Hidden layer dimension
        n_heads: Number of attention heads
        dropout: Dropout rate
        
    Returns:
        Dictionary with model path and metrics, or None if training was skipped
    """
    # Load dataset
    df = pd.read_csv(dataset_path)
    
    # Check if enough samples
    if len(df) < min_samples:
        logging.warning(f"[SKIP] {symbol} ({timeframe}), pattern {pattern} — "
                      f"Not enough samples ({len(df)} < {min_samples})")
        return None
    
    logging.info(f"[INFO] Training TFT for {Fore.YELLOW}{symbol}{Style.RESET_ALL} "
               f"({timeframe}), pattern {Fore.CYAN}{pattern}{Style.RESET_ALL} — "
               f"{Fore.GREEN}{len(df)}{Style.RESET_ALL} samples")
    
    # Create model output directory
    model_dir = os.path.join(model_base_dir, symbol, timeframe)
    os.makedirs(model_dir, exist_ok=True)
    
    # Prepare the data
    # Create a time index (we don't have actual timestamps, so we'll create a synthetic index)
    df['time'] = pd.date_range(start='2023-01-01', periods=len(df), freq='h')
    df.set_index('time', inplace=True)
    
    # Extract feature columns (x_1 to x_7) and target column (y)
    feature_cols = [f'x_{i+1}' for i in range(input_size)]
    target_col = 'y'
    
    # Check if all required columns exist
    if not all(col in df.columns for col in feature_cols):
        logging.error(f"Missing expected feature columns in {dataset_path}")
        return None
    if target_col not in df.columns:
        logging.error(f"Missing target column 'y' in {dataset_path}")
        return None
    
    # Convert to TimeSeries
    features_df = df[feature_cols]
    target_df = df[[target_col]]
    
    # Create TimeSeries objects
    features_series = TimeSeries.from_dataframe(features_df)
    target_series = TimeSeries.from_dataframe(target_df)
    
    # Scale the data
    scaler_features = Scaler()
    scaler_target = Scaler()
    
    features_scaled = scaler_features.fit_transform(features_series)
    target_scaled = scaler_target.fit_transform(target_series)
    
    # Split into training and validation sets (80/20 split)
    train_size = int(0.8 * len(features_scaled))
    
    train_features = features_scaled[:train_size]
    train_target = target_scaled[:train_size]
    
    val_features = features_scaled[train_size:]
    val_target = target_scaled[train_size:]
    
    # Create and train the model
    model = TFTModel(
        input_chunk_length=input_size,
        output_chunk_length=1,
        hidden_size=hidden_size,
        lstm_layers=1,
        num_attention_heads=n_heads,
        dropout=dropout,
        batch_size=16,
        n_epochs=num_epochs,
        likelihood=QuantileRegression(
            quantiles=[0.1, 0.5, 0.9]
        ),
        random_state=42,
        force_reset=True,
        add_relative_index=True  # Add relative index as future covariate
    )
    
    # Train the model with validation data - using verbose=False to reduce output
    model.fit(
        series=train_target,
        past_covariates=train_features,
        val_series=val_target,
        val_past_covariates=val_features,
        verbose=False
    )
    
    # Print only a simple training completion message
    logging.info(f"Model training completed with {num_epochs} epochs")
    
    # Calculate validation metrics - predict one step at a time to avoid the future covariates issue
    # We'll predict one step ahead for each validation point
    prediction_scaled = model.predict(
        n=1,  # Predict only one step ahead to avoid needing future covariates
        series=train_target,
        past_covariates=train_features
    )
    prediction = scaler_target.inverse_transform(prediction_scaled)
    actual = scaler_target.inverse_transform(val_target)
    
    # Calculate error metrics on validation data
    mape_score = mape(actual, prediction)
    mae_score = mae(actual, prediction)
    rmse_score = rmse(actual, prediction)
    
    logging.info(f"[INFO] Validation metrics: MAPE={mape_score:.4f}, MAE={mae_score:.4f}, RMSE={rmse_score:.4f}")
    
    # Save the model, scaler, and metadata
    model_path = os.path.join(model_dir, f"tft_cat_{pattern}.pkl")
    
    # Create a dictionary containing everything needed for inference
    model_package = {
        "model": model,
        "feature_scaler": scaler_features,
        "target_scaler": scaler_target,
        "metadata": {
            "symbol": symbol,
            "timeframe": timeframe,
            "pattern": pattern,
            "input_size": input_size,
            "trained_at": datetime.now().isoformat(),
            "num_samples": len(df),
            "train_samples": len(train_target),
            "val_samples": len(val_target),
            "n_epochs": num_epochs,
            "hidden_size": hidden_size,
            "n_heads": n_heads,
            "dropout": dropout,
            "metrics": {
                "mape": float(mape_score),
                "mae": float(mae_score),
                "rmse": float(rmse_score)
            }
        }
    }
    
    # Save the package to disk
    with open(model_path, 'wb') as f:
        pickle.dump(model_package, f)
    
    logging.info(f"[INFO] Model saved to {Fore.MAGENTA}{model_path}{Style.RESET_ALL}")
    
    # Return information about this model
    return {
        "path": model_path,
        "symbol": symbol,
        "timeframe": timeframe,
        "pattern": pattern,
        "samples": len(df),
        "mape": float(mape_score),
        "mae": float(mae_score),
        "rmse": float(rmse_score)
    }


def process_symbols(symbols: List[str], args: argparse.Namespace) -> List[Dict]:
    """
    Process multiple symbols with given arguments.
    
    Args:
        symbols: List of symbol names to process
        args: Command line arguments
        
    Returns:
        List of model information dictionaries
    """
    # Print progress header
    total_symbols = len(symbols)
    logging.info(f"{Fore.GREEN}========================================{Style.RESET_ALL}")
    logging.info(f"{Fore.GREEN}Starting training for {total_symbols} symbols{Style.RESET_ALL}")
    logging.info(f"{Fore.GREEN}========================================{Style.RESET_ALL}")
    
    start_time = time.time()
    all_model_info = []
    
    # Track progress
    completed_symbols = 0
    
    for symbol in symbols:
        symbol_start_time = time.time()
        symbol_path = os.path.join(args.dataset_dir, symbol)
        
        if not os.path.isdir(symbol_path):
            logging.warning(f"Directory not found: {symbol_path}, skipping")
            continue
            
        logging.info(f"{Fore.CYAN}[{completed_symbols+1}/{total_symbols}] Processing symbol: {symbol}{Style.RESET_ALL}")
            
        # If timeframe is specified, only process that timeframe
        if args.timeframe:
            timeframe_paths = [os.path.join(symbol_path, args.timeframe)]
        else:
            # Otherwise process all timeframes
            timeframe_paths = [os.path.join(symbol_path, d) for d in os.listdir(symbol_path) 
                              if os.path.isdir(os.path.join(symbol_path, d))]
        
        # Process each timeframe
        for timeframe_path in timeframe_paths:
            if not os.path.isdir(timeframe_path):
                continue
                
            timeframe = os.path.basename(timeframe_path)
            
            # Process each category file
            for filename in os.listdir(timeframe_path):
                if not filename.startswith("cat_") or not filename.endswith(".csv"):
                    continue
                
                # Extract pattern from filename
                pattern = filename.replace("cat_", "").replace(".csv", "")
                dataset_path = os.path.join(timeframe_path, filename)
                
                try:
                    # Train model for this dataset
                    model_info = _train_and_save_model(
                        symbol,
                        timeframe,
                        pattern,
                        dataset_path,
                        args.model_dir,
                        args.min_samples,
                        args.input_size,
                        args.epochs,
                        args.hidden_size,
                        args.n_heads,
                        args.dropout
                    )
                    
                    # Add to results if training was successful
                    if model_info:
                        all_model_info.append(model_info)
                        
                except Exception as e:
                    logging.error(f"Error training model for {symbol}/{timeframe}/{pattern}: {e}")
                    import traceback
                    logging.error(traceback.format_exc())
        # Update completed counter and show progress
        completed_symbols += 1
        symbol_time = time.time() - symbol_start_time
        logging.info(f"{Fore.GREEN}Completed symbol {symbol} in {symbol_time:.1f} seconds{Style.RESET_ALL}")
        
        # Show overall progress
        if completed_symbols < total_symbols:
            elapsed_time = time.time() - start_time
            avg_time_per_symbol = elapsed_time / completed_symbols
            estimated_remaining = avg_time_per_symbol * (total_symbols - completed_symbols)
            
            logging.info(f"{Fore.YELLOW}Progress: {completed_symbols}/{total_symbols} symbols " 
                       f"({100*completed_symbols/total_symbols:.1f}%){Style.RESET_ALL}")
            logging.info(f"{Fore.YELLOW}Estimated time remaining: {estimated_remaining/60:.1f} minutes{Style.RESET_ALL}")
            logging.info(f"{Fore.GREEN}----------------------------------------{Style.RESET_ALL}")
    
    # Show final timing info
    total_time = time.time() - start_time
    logging.info(f"{Fore.GREEN}========================================{Style.RESET_ALL}")
    logging.info(f"{Fore.GREEN}Completed all {total_symbols} symbols in {total_time/60:.1f} minutes{Style.RESET_ALL}")
    logging.info(f"{Fore.GREEN}Average time per symbol: {total_time/total_symbols:.1f} seconds{Style.RESET_ALL}")
    logging.info(f"{Fore.GREEN}========================================{Style.RESET_ALL}")
    
    return all_model_info


def create_model_registry(all_model_info: List[Dict], model_base_dir: str, args: argparse.Namespace) -> None:
    """
    Create a model registry from trained model information.
    
    Args:
        all_model_info: List of dictionaries with model information
        model_base_dir: Base directory for models
        args: Command line arguments with training parameters
    """
    if not all_model_info:
        logging.warning("No models were successfully trained. Registry not created.")
        return
        
    # Create registry directory
    registry_dir = os.path.join(model_base_dir, "registry")
    os.makedirs(registry_dir, exist_ok=True)
    
    # Save as CSV
    registry_df = pd.DataFrame(all_model_info)
    registry_csv_path = os.path.join(registry_dir, "model_registry.csv")
    registry_df.to_csv(registry_csv_path, index=False)
    
    # Save as JSON
    registry_json_path = os.path.join(registry_dir, "model_registry.json")
    with open(registry_json_path, 'w') as f:
        json.dump(all_model_info, f, indent=2)
    
    # Save training summary
    summary = {
        "trained_at": datetime.now().isoformat(),
        "total_models": len(all_model_info),
        "symbols": len(set(m["symbol"] for m in all_model_info)),
        "timeframes": list(set(m["timeframe"] for m in all_model_info)),
        "avg_metrics": {
            "mape": sum(m["mape"] for m in all_model_info) / len(all_model_info),
            "mae": sum(m["mae"] for m in all_model_info) / len(all_model_info),
            "rmse": sum(m["rmse"] for m in all_model_info) / len(all_model_info)
        },
        "training_params": {
            "min_samples": args.min_samples,
            "input_size": args.input_size,
            "num_epochs": args.epochs,
            "hidden_size": args.hidden_size,
            "n_heads": args.n_heads,
            "dropout": args.dropout
        }
    }
    
    summary_path = os.path.join(registry_dir, "training_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
        
    # Generate a log filename with timestamp
    log_filename = f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(registry_dir, log_filename)
    
    # Write detailed log
    with open(log_path, 'w') as f:
        f.write(f"=== TFT MODEL TRAINING LOG ===\n")
        f.write(f"Time: {datetime.now().isoformat()}\n\n")
        
        # Group by symbol and timeframe
        by_symbol = {}
        for info in all_model_info:
            symbol = info["symbol"]
            if symbol not in by_symbol:
                by_symbol[symbol] = {}
                
            timeframe = info["timeframe"]
            if timeframe not in by_symbol[symbol]:
                by_symbol[symbol][timeframe] = []
                
            by_symbol[symbol][timeframe].append(info)
        
        # Log by symbol and timeframe
        for symbol, timeframes in by_symbol.items():
            f.write(f"\nSymbol: {symbol}\n")
            
            for timeframe, models in timeframes.items():
                f.write(f"  Timeframe: {timeframe}\n")
                f.write(f"  Models trained: {len(models)}\n")
                
                # Calculate average metrics
                avg_mape = sum(m["mape"] for m in models) / len(models)
                avg_mae = sum(m["mae"] for m in models) / len(models)
                avg_rmse = sum(m["rmse"] for m in models) / len(models)
                
                f.write(f"  Avg metrics: MAPE={avg_mape:.4f}, MAE={avg_mae:.4f}, RMSE={avg_rmse:.4f}\n")
                
                # Log details for each model
                for model in models:
                    f.write(f"    Pattern: {model['pattern']}, "
                           f"Samples: {model['samples']}, "
                           f"MAPE: {model['mape']:.4f}, "
                           f"MAE: {model['mae']:.4f}, "
                           f"RMSE: {model['rmse']:.4f}\n")
    
    logging.info(f"Model registry created at: {Fore.MAGENTA}{registry_csv_path}{Style.RESET_ALL}")
    logging.info(f"Training log saved to: {Fore.MAGENTA}{log_path}{Style.RESET_ALL}")


def main():
    """Entry point for command-line execution."""
    parser = argparse.ArgumentParser(description='Train TFT models on volatility datasets')
    parser.add_argument('--symbol', type=str, help='Filter by symbol (e.g., BTC_USDT or ALL for all symbols)')
    parser.add_argument('--symbols', type=str, help='Comma-separated list of symbols to process')
    parser.add_argument('--timeframe', type=str, help='Filter by timeframe (e.g., 5m)')
    parser.add_argument('--dataset-dir', type=str, default='datasets', 
                        help='Root directory containing datasets')
    parser.add_argument('--model-dir', type=str, default='models', 
                        help='Output directory for trained models')
    parser.add_argument('--min-samples', type=int, default=30, 
                        help='Minimum number of samples required for training')
    parser.add_argument('--input-size', type=int, default=7, 
                        help='Number of input features (x_1 to x_7)')
    parser.add_argument('--epochs', type=int, default=10, 
                        help='Number of training epochs')
    parser.add_argument('--hidden-size', type=int, default=64, 
                        help='Hidden dimension in TFT model')
    parser.add_argument('--n-heads', type=int, default=4, 
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, 
                        help='Dropout rate')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(level=logging.INFO)
    
    # Get base dirs
    dataset_dir = args.dataset_dir
    model_dir = args.model_dir
    
    # Determine which symbols to process
    if args.symbols:
        # Process comma-separated list of symbols
        symbols_to_process = [s.strip() for s in args.symbols.split(',')]
        logging.info(f"Processing multiple symbols: {', '.join(symbols_to_process)}")
        all_model_info = process_symbols(symbols_to_process, args)
        
    elif args.symbol == "ALL" or args.symbol == "all":
        # Process all symbols
        symbols_to_process = [d for d in os.listdir(dataset_dir) 
                              if os.path.isdir(os.path.join(dataset_dir, d))]
        logging.info(f"Processing ALL symbols: {len(symbols_to_process)} found")
        all_model_info = process_symbols(symbols_to_process, args)
        
    elif args.symbol:
        # Process a single symbol
        all_model_info = process_symbols([args.symbol], args)
        
    else:
        # Use the main function
        logging.info("No symbol filter specified. Processing all symbols and timeframes.")
        results_dict = train_all_tft_models(
            dataset_base_dir=dataset_dir,
            model_base_dir=model_dir,
            min_samples=args.min_samples,
            input_size=args.input_size,
            num_epochs=args.epochs,
            hidden_size=args.hidden_size,
            n_heads=args.n_heads,
            dropout=args.dropout
        )
        # Results are already processed in train_all_tft_models
        return
    
    # Create registry from collected model info
    create_model_registry(all_model_info, model_dir, args)
    
    # Log summary
    total_models = len(all_model_info)
    total_symbols = len(set(info["symbol"] for info in all_model_info))
    total_timeframes = len(set(info["timeframe"] for info in all_model_info))
    
    logging.info(f"\n{Fore.GREEN}=== MODEL TRAINING SUMMARY ==={Style.RESET_ALL}")
    logging.info(f"Trained {Fore.GREEN}{total_models}{Style.RESET_ALL} models "
               f"across {Fore.YELLOW}{total_symbols}{Style.RESET_ALL} symbols "
               f"and {Fore.YELLOW}{total_timeframes}{Style.RESET_ALL} timeframes")
    logging.info(f"Models saved to: {Fore.MAGENTA}{model_dir}{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
