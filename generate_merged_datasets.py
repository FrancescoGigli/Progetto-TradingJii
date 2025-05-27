#!/usr/bin/env python3
"""
Generate Merged ML Datasets from Database

This script generates merged CSV datasets for ML training by:
1. Loading price data from data_<timeframe> tables
2. Creating volatility features (x_1...x_7) as sliding windows
3. Calculating labels based on price changes
4. Joining with technical indicators from ta_<timeframe> tables
5. Saving as merged.csv files in ml_datasets/<symbol>/<timeframe>/

Usage:
    python generate_merged_datasets.py --timeframe 1h
    python generate_merged_datasets.py --timeframe 4h --symbol BTC_USDTUSDT
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# Local imports
from colorama import Fore, Style, init
from modules.utils.config import DB_FILE
from modules.utils.logging_setup import setup_logging

# Initialize colorama for Windows compatibility
init()

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate merged ML datasets from database"
    )
    
    parser.add_argument(
        "--timeframe", 
        type=str, 
        required=True,
        choices=["1h", "4h", "1d"],
        help="Timeframe for dataset generation (e.g., 1h, 4h)"
    )
    
    parser.add_argument(
        "--symbol", 
        type=str, 
        default=None,
        help="Specific symbol to process (e.g., BTC_USDTUSDT). If not specified, processes all symbols."
    )
    
    parser.add_argument(
        "--window-size",
        type=int,
        default=7,
        help="Size of volatility sliding window (default: 7)"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Overwrite existing merged.csv files"
    )
    
    return parser.parse_args()

def get_available_symbols(timeframe: str) -> List[str]:
    """
    Get list of available symbols for the given timeframe.
    
    Args:
        timeframe: The timeframe (e.g., "1h", "4h")
        
    Returns:
        List of symbol names
    """
    table_name = f"data_{timeframe}"
    
    try:
        with sqlite3.connect(DB_FILE) as conn:
            query = f"SELECT DISTINCT symbol FROM {table_name} ORDER BY symbol"
            df = pd.read_sql_query(query, conn)
            return df['symbol'].tolist()
    except Exception as e:
        logging.error(f"Error getting symbols from {table_name}: {e}")
        return []

def load_price_data(symbol: str, timeframe: str) -> pd.DataFrame:
    """
    Load OHLCV price data for a symbol and timeframe.
    
    Args:
        symbol: The cryptocurrency symbol
        timeframe: The timeframe
        
    Returns:
        DataFrame with OHLCV data sorted by timestamp
    """
    table_name = f"data_{timeframe}"
    
    try:
        with sqlite3.connect(DB_FILE) as conn:
            query = f"""
                SELECT timestamp, open, high, low, close, volume
                FROM {table_name}
                WHERE symbol = ?
                ORDER BY timestamp ASC
            """
            
            df = pd.read_sql_query(query, conn, params=(symbol,))
            
            if df.empty:
                logging.warning(f"No price data found for {symbol} in {table_name}")
                return pd.DataFrame()
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
            
    except Exception as e:
        logging.error(f"Error loading price data for {symbol} ({timeframe}): {e}")
        return pd.DataFrame()

def create_volatility_features(df: pd.DataFrame, window_size: int = 7) -> pd.DataFrame:
    """
    Create volatility features using sliding windows.
    
    Args:
        df: DataFrame with OHLCV data
        window_size: Size of the sliding window
        
    Returns:
        DataFrame with volatility features x_1...x_n
    """
    if len(df) < window_size + 1:
        logging.warning(f"Insufficient data for window size {window_size}")
        return pd.DataFrame()
    
    # Calculate price changes (volatility proxy)
    df = df.copy()
    df['price_change'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    df['volatility'] = df['price_change'].fillna(0)
    
    # Create sliding windows
    rows = []
    
    for i in range(window_size, len(df)):
        # Get volatility window (x_1 to x_n)
        window = df['volatility'].iloc[i-window_size:i].tolist()
        
        # Get current row data
        current_row = df.iloc[i]
        timestamp = current_row['timestamp']
        
        # Calculate future price change for label
        if i < len(df) - 1:
            next_close = df['close'].iloc[i + 1]
            current_close = current_row['close']
            future_price_change = (next_close - current_close) / current_close
        else:
            # Skip last row as we can't calculate future price change
            continue
        
        # Create row data
        row = {f'x_{j+1}': val for j, val in enumerate(window)}
        row['y'] = future_price_change
        row['timestamp'] = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        
        # Create binary pattern
        pattern = ''.join('1' if v > 0 else '0' for v in window)
        row['pattern'] = pattern
        
        rows.append(row)
    
    result_df = pd.DataFrame(rows)
    
    if result_df.empty:
        return result_df
    
    # Create labels based on price change thresholds
    result_df['label'] = 0  # Default to HOLD
    result_df.loc[result_df['y'] >= 0.01, 'label'] = 1  # BUY
    result_df.loc[result_df['y'] <= -0.01, 'label'] = 2  # SELL
    
    return result_df

def load_technical_indicators(symbol: str, timeframe: str, timestamps: List[str]) -> pd.DataFrame:
    """
    Load technical indicators for the given timestamps.
    
    Args:
        symbol: The cryptocurrency symbol
        timeframe: The timeframe
        timestamps: List of timestamps to match
        
    Returns:
        DataFrame with technical indicators
    """
    if not timestamps:
        return pd.DataFrame()
    
    table_name = f"ta_{timeframe}"
    
    try:
        # Process in batches to avoid SQLite parameter limit
        batch_size = 500
        all_results = []
        
        for i in range(0, len(timestamps), batch_size):
            batch_timestamps = timestamps[i:i+batch_size]
            
            with sqlite3.connect(DB_FILE) as conn:
                # Create placeholders for the timestamps
                placeholders = ','.join(['?' for _ in batch_timestamps])
                
                query = f"""
                    SELECT *
                    FROM {table_name}
                    WHERE symbol = ? AND timestamp IN ({placeholders})
                """
                
                params = [symbol] + batch_timestamps
                batch_df = pd.read_sql_query(query, conn, params=params)
                
                if not batch_df.empty:
                    all_results.append(batch_df)
        
        # Combine all batches
        if all_results:
            df = pd.concat(all_results, ignore_index=True)
            
            # Remove non-feature columns
            columns_to_drop = ['id', 'symbol'] if 'id' in df.columns else ['symbol']
            for col in columns_to_drop:
                if col in df.columns:
                    df = df.drop(col, axis=1)
            
            return df
        else:
            logging.warning(f"No technical indicators found for {symbol} ({timeframe})")
            return pd.DataFrame()
            
    except Exception as e:
        logging.error(f"Error loading technical indicators for {symbol} ({timeframe}): {e}")
        return pd.DataFrame()

def generate_merged_dataset(symbol: str, timeframe: str, window_size: int = 7, force: bool = False) -> bool:
    """
    Generate a merged dataset for a single symbol and timeframe.
    
    Args:
        symbol: The cryptocurrency symbol
        timeframe: The timeframe
        window_size: Size of the volatility sliding window
        force: Whether to overwrite existing files
        
    Returns:
        Boolean indicating success
    """
    # Setup output directory
    symbol_safe = symbol.replace('/', '_').replace(':', '').replace('\\', '').replace('*', '')
    symbol_safe = symbol_safe.replace('?', '').replace('"', '').replace('<', '').replace('>', '')
    symbol_safe = symbol_safe.replace('|', '')
    
    output_dir = Path("ml_datasets") / symbol_safe / timeframe
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "merged.csv"
    
    if output_file.exists() and not force:
        logging.info(f"Dataset already exists: {output_file} (use --force to overwrite)")
        return True
    
    logging.info(f"Generating dataset for {symbol} ({timeframe})...")
    
    # Load price data
    price_df = load_price_data(symbol, timeframe)
    
    if price_df.empty:
        logging.error(f"No price data available for {symbol} ({timeframe})")
        return False
    
    logging.info(f"Loaded {len(price_df)} price records")
    
    # Create volatility features and labels
    features_df = create_volatility_features(price_df, window_size)
    
    if features_df.empty:
        logging.error(f"Failed to create features for {symbol} ({timeframe})")
        return False
    
    logging.info(f"Created {len(features_df)} feature records")
    
    # Load technical indicators
    timestamps = features_df['timestamp'].tolist()
    indicators_df = load_technical_indicators(symbol, timeframe, timestamps)
    
    if indicators_df.empty:
        logging.warning(f"No technical indicators found for {symbol} ({timeframe})")
        logging.warning("Proceeding with volatility-only dataset")
        merged_df = features_df.copy()
    else:
        logging.info(f"Loaded {len(indicators_df)} indicator records")
        
        # Merge features with indicators
        merged_df = pd.merge(features_df, indicators_df, on='timestamp', how='inner')
        
        if merged_df.empty:
            logging.error(f"Failed to merge data for {symbol} ({timeframe})")
            return False
    
    # Remove any remaining NaN values
    initial_rows = len(merged_df)
    merged_df = merged_df.dropna()
    final_rows = len(merged_df)
    
    if initial_rows > final_rows:
        logging.warning(f"Dropped {initial_rows - final_rows} rows due to NaN values")
    
    if final_rows == 0:
        logging.error(f"No valid data remaining for {symbol} ({timeframe})")
        return False
    
    # Log class distribution
    class_counts = merged_df['label'].value_counts().sort_index()
    logging.info("Label distribution:")
    for label, count in class_counts.items():
        class_name = {0: "HOLD", 1: "BUY", 2: "SELL"}.get(label, f"Unknown({label})")
        percentage = count / len(merged_df) * 100
        logging.info(f"  {class_name} ({label}): {count} ({percentage:.1f}%)")
    
    # Save the dataset
    merged_df.to_csv(output_file, index=False)
    logging.info(f"Saved dataset: {output_file} ({final_rows} records)")
    
    return True

def generate_dataset_for_symbol(symbol: str, timeframe: str, force: bool = False) -> bool:
    """
    Wrapper function for integration with real_time.py
    
    Args:
        symbol: The cryptocurrency symbol
        timeframe: The timeframe (e.g., "1h", "4h") 
        force: Whether to overwrite existing files
        
    Returns:
        Boolean indicating success
    """
    return generate_merged_dataset(symbol, timeframe, window_size=7, force=force)

def main():
    """Main function."""
    args = parse_arguments()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    
    logging.info(f"{Fore.GREEN}Starting dataset generation...{Style.RESET_ALL}")
    logging.info(f"Timeframe: {args.timeframe}")
    logging.info(f"Window size: {args.window_size}")
    logging.info(f"Force overwrite: {args.force}")
    
    # Get symbols to process
    if args.symbol:
        symbols = [args.symbol]
        logging.info(f"Processing single symbol: {args.symbol}")
    else:
        symbols = get_available_symbols(args.timeframe)
        logging.info(f"Processing {len(symbols)} symbols: {', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''}")
    
    if not symbols:
        logging.error("No symbols found to process")
        return
    
    # Process each symbol
    successful = 0
    failed = 0
    
    for symbol in symbols:
        try:
            success = generate_merged_dataset(symbol, args.timeframe, args.window_size, args.force)
            if success:
                successful += 1
            else:
                failed += 1
        except Exception as e:
            logging.error(f"Error processing {symbol}: {e}")
            failed += 1
    
    # Summary
    total = successful + failed
    logging.info(f"\n{Fore.GREEN}Dataset generation completed!{Style.RESET_ALL}")
    logging.info(f"Successful: {successful}/{total}")
    logging.info(f"Failed: {failed}/{total}")
    
    if failed > 0:
        logging.warning(f"{failed} symbols failed to process")

if __name__ == "__main__":
    main()
