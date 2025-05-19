#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train and Backtest ML Model Script

This script implements an end-to-end workflow for training a trading model:
1. Data loading from merged.csv files
2. Feature and target preparation
3. Time-series train/test split
4. Model training with XGBoost
5. Model evaluation
6. Backtest simulation
"""

import glob
import os
import logging
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data():
    """
    Recursively search for all merged.csv files under ml_datasets folder
    and concatenate them into a single DataFrame.
    
    Returns:
        pd.DataFrame: Concatenated DataFrame sorted by timestamp
    """
    logger.info("Searching for merged.csv files...")
    
    # Search for all merged.csv files
    csv_files = glob.glob('ml_datasets/**/**/merged.csv', recursive=True)
    
    if not csv_files:
        logger.warning("No merged.csv files found!")
        return None
    
    logger.info(f"Found {len(csv_files)} merged.csv files")
    
    # Load and concatenate all CSV files
    dfs = []
    for file in csv_files:
        try:
            logger.info(f"Loading {file}...")
            df = pd.read_csv(file, parse_dates=['timestamp'])
            
            # Extract symbol and timeframe from the path
            parts = file.split(os.sep)
            if len(parts) >= 3:
                symbol = parts[-3]  # e.g., BTC_USDT
                timeframe = parts[-2]  # e.g., 15m
                
                # Add symbol and timeframe columns if not already present
                if 'symbol' not in df.columns:
                    df['symbol'] = symbol
                if 'timeframe' not in df.columns:
                    df['timeframe'] = timeframe
            
            dfs.append(df)
        except Exception as e:
            logger.error(f"Error loading {file}: {e}")
    
    if not dfs:
        logger.warning("No valid DataFrames to concatenate!")
        return None
    
    # Concatenate all DataFrames
    all_data = pd.concat(dfs, ignore_index=True)
    
    # Sort by timestamp
    all_data = all_data.sort_values(by='timestamp')
    
    logger.info(f"Loaded dataset with {all_data.shape[0]} rows and {all_data.shape[1]} columns")
    
    return all_data

def prepare_features_target(data):
    """
    Prepare features (X) and target (y) from the data.
    
    Args:
        data (pd.DataFrame): Input DataFrame
        
    Returns:
        tuple: (X, y) where X is features DataFrame and y is target Series
    """
    if data is None or data.empty:
        logger.warning("Empty or None DataFrame provided to prepare_features_target!")
        return None, None
    
    # Columns to exclude from features
    exclude_cols = ['y', 'y_class', 'timestamp', 'pattern', 'symbol', 'timeframe']
    
    # Select features (all columns except excluded ones)
    feature_cols = [col for col in data.columns if col not in exclude_cols]
    X = data[feature_cols]
    
    # Fill missing values with zero
    X = X.fillna(0)
    
    # Use y_class as target
    y = data['y_class']
    
    logger.info(f"Prepared {X.shape[1]} features and {y.shape[0]} target values")
    
    return X, y

def train_test_split_timeseries(X, y, train_size=0.8):
    """
    Split data into train and test sets respecting chronological order.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        train_size (float): Proportion of data to use for training
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    if X is None or y is None:
        logger.warning("None X or y provided to train_test_split_timeseries!")
        return None, None, None, None
    
    n = len(X)
    train_idx = int(n * train_size)
    
    X_train = X.iloc[:train_idx]
    X_test = X.iloc[train_idx:]
    y_train = y.iloc[:train_idx]
    y_test = y.iloc[train_idx:]
    
    logger.info(f"Split data into {X_train.shape[0]} training samples and {X_test.shape[0]} testing samples")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """
    Train an XGBClassifier model on the training data.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        
    Returns:
        XGBClassifier: Trained model
    """
    if X_train is None or y_train is None:
        logger.warning("None X_train or y_train provided to train_model!")
        return None
    
    logger.info("Training XGBClassifier model...")
    
    # Initialize model
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    
    # Train model
    model.fit(X_train, y_train)
    
    logger.info("Model training completed")
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on test data and print metrics.
    
    Args:
        model: Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
        
    Returns:
        tuple: (y_pred, confusion_mat, class_report)
    """
    if model is None or X_test is None or y_test is None:
        logger.warning("None model or test data provided to evaluate_model!")
        return None, None, None
    
    logger.info("Evaluating model on test data...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    # Print metrics
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)
    
    return y_pred, conf_matrix, class_report

def run_backtest(X_test, y_test, y_pred, y_values):
    """
    Run a backtest simulation using the model's predictions.
    
    Args:
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target classes
        y_pred (np.array): Predicted classes
        y_values (pd.Series): Actual percentage changes
        
    Returns:
        tuple: (final_capital, account_balance_history, win_rate, trade_count)
    """
    if X_test is None or y_test is None or y_pred is None or y_values is None:
        logger.warning("None data provided to run_backtest!")
        return None, None, None, None
    
    logger.info("Running backtest simulation...")
    
    # Initialize account
    starting_capital = 100.0  # €100
    capital = starting_capital
    leverage = 10
    account_balance_history = [capital]
    
    # Counters for win rate calculation
    correct_predictions = 0
    trade_count = 0
    
    # Loop through test data in chronological order
    for i in range(len(y_test)):
        pred_class = y_pred[i]
        actual_class = y_test.iloc[i]
        percentage_change = y_values.iloc[i]  # Real percentage change
        
        # Track trades
        if pred_class != 0:  # If model predicted BUY or SELL
            trade_count += 1
            
            # BUY signal
            if pred_class == 1:
                new_capital = capital * (1 + (percentage_change/100) * leverage)
                # Check if prediction direction was correct
                if percentage_change > 0:
                    correct_predictions += 1
                    
            # SELL signal
            elif pred_class == 2:
                new_capital = capital * (1 - (percentage_change/100) * leverage)
                # Check if prediction direction was correct
                if percentage_change < 0:
                    correct_predictions += 1
            
            # Update capital
            capital = new_capital
            
        # Record balance
        account_balance_history.append(capital)
    
    # Calculate win rate
    win_rate = (correct_predictions / trade_count * 100) if trade_count > 0 else 0
    
    # Print backtest results
    print("\nBacktest Results:")
    print(f"Starting capital: {starting_capital:.2f} €")
    print(f"Final capital: {capital:.2f} €")
    print(f"Return: {((capital - starting_capital) / starting_capital * 100):.2f}%")
    print(f"Win rate: {win_rate:.2f}%")
    print(f"Total trades executed: {trade_count}")
    
    return capital, account_balance_history, win_rate, trade_count

def save_model(model, filepath='models/trading_model.pkl'):
    """
    Save the trained model to a file.
    
    Args:
        model: Trained model to save
        filepath (str): Path to save the model
        
    Returns:
        bool: True if successful, False otherwise
    """
    if model is None:
        logger.warning("None model provided to save_model!")
        return False
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    try:
        logger.info(f"Saving model to {filepath}...")
        joblib.dump(model, filepath)
        logger.info("Model saved successfully")
        return True
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return False

def main():
    """
    Main function to orchestrate the entire workflow.
    """
    logger.info("Starting train_and_backtest.py")
    
    # Step 1: Load data
    data = load_data()
    if data is None or data.empty:
        logger.error("Failed to load data. Exiting.")
        return
    
    # Step 2: Prepare features and target
    X, y = prepare_features_target(data)
    if X is None or y is None:
        logger.error("Failed to prepare features and target. Exiting.")
        return
    
    # Get the y values (percentage change) for backtest
    y_values = data['y']
    
    # Step 3: Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split_timeseries(X, y)
    if X_train is None or X_test is None or y_train is None or y_test is None:
        logger.error("Failed to split data. Exiting.")
        return
    
    # Step 4: Train model
    model = train_model(X_train, y_train)
    if model is None:
        logger.error("Failed to train model. Exiting.")
        return
    
    # Step 5: Save model
    save_model(model)
    
    # Step 6: Evaluate model
    y_pred, conf_matrix, class_report = evaluate_model(model, X_test, y_test)
    if y_pred is None:
        logger.error("Failed to evaluate model. Exiting.")
        return
    
    # Step 7: Run backtest simulation
    y_values_test = y_values.iloc[len(X_train):]
    run_backtest(X_test, y_test, y_pred, y_values_test)
    
    logger.info("train_and_backtest.py completed successfully")

if __name__ == '__main__':
    main()
