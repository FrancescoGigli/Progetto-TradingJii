#!/usr/bin/env python3
"""
Full Dataset Generator Module for TradingJii - FIXED VERSION

Fixed timestamp alignment issue between volatility patterns and technical indicators.
The main fix is in the merge_patterns_with_indicators function that normalizes 
timestamp formats before merging.

This module generates merged ML datasets directly from volatility and technical indicator data:
- Extracts volatility sliding windows from the SQLite database
- Creates labeled datasets with X (features) and y (target)
- Joins with technical indicators for the corresponding timestamps
- Exports to a single merged CSV file for training ML models

Eliminates the need for intermediate pattern files, providing a streamlined pipeline.
"""

import os
import logging
import sqlite3
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from colorama import Fore, Style
from modules.utils.config import DB_FILE, DEFAULT_WINDOW_SIZE, BUY_THRESHOLD, SELL_THRESHOLD
from modules.utils.logging_setup import setup_logging

async def generate_full_ml_dataset(
    symbol: str, 
    timeframe: str, 
    window_size: int = 7, 
    output_dir: Optional[str] = None,
    force: bool = True,
    filter_flat_patterns: bool = False
) -> bool:
    """
    Generate a full ML dataset by combining volatility patterns with technical indicators.
    
    Args:
        symbol: The cryptocurrency symbol (e.g., "BTC_USDT")
        timeframe: The timeframe (e.g., "5m")
        window_size: Size of the sliding window (default: 7)
        output_dir: Directory where the dataset will be saved (default: ml_datasets/<symbol>/<timeframe>)
        force: Whether to overwrite existing file (default: True)
        filter_flat_patterns: Whether to filter out flat patterns (all 0s or all 1s) (default: False)
        
    Returns:
        Boolean indicating success
    """
    # Initialize variables for tracking statistics
    total_records = 0
    records_retained = 0
    records_dropped = 0
    
    # Sanitize symbol for safe directory name
    symbol_safe = symbol.replace('/', '_').replace(':', '').replace('\\', '').replace('*', '')
    symbol_safe = symbol_safe.replace('?', '').replace('"', '').replace('<', '').replace('>', '')
    symbol_safe = symbol_safe.replace('|', '')
    
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = os.path.join("ml_datasets", symbol_safe, timeframe)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output file path
    output_file = os.path.join(output_dir, "merged.csv")
    
    # Check if file exists and if we should skip
    if not force and os.path.exists(output_file):
        logging.info(f"{Fore.YELLOW}Dataset already exists at {output_file}. Use force=True to regenerate.{Style.RESET_ALL}")
        return True
    
    logging.info(f"{Fore.GREEN}Generating dataset for {Fore.YELLOW}{symbol}{Style.RESET_ALL} ({timeframe}){Style.RESET_ALL}")
    logging.debug(f"Window size: {window_size}, Output: {output_file}")
    
    # Step 1: Load volatility series
    volatility_data = load_volatility_series(symbol, timeframe)
    
    if volatility_data.empty or len(volatility_data) < window_size + 1:
        logging.warning(f"Insufficient volatility data for {symbol} ({timeframe}) - need at least {window_size + 1} data points")
        return False
    
    total_volatility_records = len(volatility_data)
    logging.info(f"Loaded {total_volatility_records} price records")
    
    # Step 2: Generate sliding windows for patterns and targets
    patterns_df = generate_pattern_windows(volatility_data, window_size, filter_flat_patterns)
    
    if patterns_df.empty:
        logging.warning(f"Failed to generate pattern windows for {symbol} ({timeframe})")
        return False
    
    total_records = len(patterns_df)
    logging.info(f"Created {total_records} feature records")
    
    # Step 3: Load technical indicators for this symbol/timeframe
    indicators_df = load_technical_indicators(symbol, timeframe, patterns_df['timestamp'].tolist())
    
    # If no indicators are available, warn but continue with just volatility data
    if indicators_df.empty:
        logging.warning(f"No technical indicators found for {symbol} ({timeframe})")
        logging.warning(f"Proceeding with volatility-only dataset")
        merged_df = patterns_df.copy()  # Use only the pattern data
    else:
        logging.debug(f"Loaded {len(indicators_df)} technical indicator records")
        
        # Step 4: Merge patterns with indicators (FIXED VERSION)
        merged_df = merge_patterns_with_indicators(patterns_df, indicators_df)
        
    if merged_df.empty:
        logging.warning(f"Failed to create dataset for {symbol} ({timeframe})")
        return False
    
    records_retained = len(merged_df)
    records_dropped = total_records - records_retained
    retention_rate = (records_retained / total_records) * 100 if total_records > 0 else 0
    
    # Step 5: Save the merged dataset
    if records_retained < 5000 and total_volatility_records >= 5000:
        logging.warning(f"{Fore.YELLOW}Warning: Final dataset contains fewer than 5000 records ({records_retained}), "
                       f"while there are {total_volatility_records} volatility records available. "
                       f"This might indicate issues with technical indicators.{Style.RESET_ALL}")
    
    # Add labels
    merged_df = add_labels(merged_df)
    
    # Save the dataset
    merged_df.to_csv(output_file, index=False)
    
    # Count labels
    if 'y_class' in merged_df.columns:
        label_counts = merged_df['y_class'].value_counts().sort_index()
        logging.info("Label distribution:")
        for label, count in label_counts.items():
            label_name = "SELL" if label == 0 else "BUY"
            percentage = (count / len(merged_df)) * 100
            logging.info(f"  {label_name} ({label}): {count} ({percentage:.1f}%)")
    
    logging.info(f"Saved dataset: {output_file} ({len(merged_df)} records)")
    
    return True

def load_volatility_series(symbol: str, timeframe: str) -> pd.DataFrame:
    """
    Load timestamp, volatility for a given symbol and timeframe from the volatility table.
    
    Args:
        symbol: The cryptocurrency symbol (e.g., "BTC/USDT")
        timeframe: The timeframe (e.g., "5m")
        
    Returns:
        A pandas DataFrame with timestamp and volatility columns, sorted chronologically
    """
    table_name = f"volatility_{timeframe}"
    
    try:
        with sqlite3.connect(DB_FILE) as conn:
            query = f"""
                SELECT timestamp, volatility
                FROM {table_name}
                WHERE symbol = ?
                ORDER BY timestamp ASC
            """
            
            df = pd.read_sql_query(query, conn, params=(symbol,))
            
            if df.empty:
                logging.warning(f"No volatility data found for {symbol} in timeframe {timeframe}")
                return pd.DataFrame(columns=['timestamp', 'volatility'])
            
            # Convert timestamp to datetime for easier manipulation
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
            
    except Exception as e:
        logging.error(f"Error loading volatility series for {symbol} ({timeframe}): {e}")
        return pd.DataFrame(columns=['timestamp', 'volatility'])

def generate_pattern_windows(
    df: pd.DataFrame, 
    window_size: int = 7,
    filter_flat_patterns: bool = False
) -> pd.DataFrame:
    """
    Generate sliding windows of volatility values with targets and pattern strings.
    
    Args:
        df: DataFrame with timestamp and volatility columns
        window_size: Size of the sliding window (default: 7)
        filter_flat_patterns: Whether to filter out flat patterns (all 0s or all 1s)
        
    Returns:
        DataFrame with x_1 to x_n columns, y, y_class, pattern, and timestamp
    """
    if df.empty or len(df) < window_size + 1:
        logging.warning(f"Insufficient data to generate pattern windows (need at least {window_size + 1} data points)")
        return pd.DataFrame()
    
    # Initialize lists to store row data
    rows = []
    
    # Generate sliding windows
    for i in range(len(df) - window_size):
        window = df['volatility'].iloc[i:i+window_size].tolist()
        target = df['volatility'].iloc[i+window_size]  # Next value is the target
        timestamp = df['timestamp'].iloc[i+window_size]  # Timestamp corresponding to the target
        
        # Create binary pattern string (1 if volatility > 0, else 0)
        pattern = ''.join('1' if v > 0 else '0' for v in window)
        
        # Skip flat patterns if requested
        if filter_flat_patterns and (pattern == '0' * window_size or pattern == '1' * window_size):
            continue
        
        # Create row with x_1, x_2, ..., x_n columns, y, pattern, and timestamp
        row = {f'x_{i+1}': val for i, val in enumerate(window)}
        row['y'] = target  # Keep original target value for regression or comparison
        row['pattern'] = pattern
        
        # Safely convert timestamp to string format before adding to row
        try:
            # Check if timestamp is a pandas Timestamp or datetime object
            if hasattr(timestamp, 'strftime'):
                row['timestamp'] = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            elif isinstance(timestamp, str):
                # If it's already a string, keep it as is
                row['timestamp'] = timestamp
            elif isinstance(timestamp, (int, float)):
                # Convert numeric timestamp to datetime
                row['timestamp'] = pd.to_datetime(timestamp, unit='s').strftime('%Y-%m-%d %H:%M:%S')
            else:
                # For any other type, convert to string and log a warning
                logging.warning(f"Unexpected timestamp type: {type(timestamp)} for value: {timestamp}")
                row['timestamp'] = str(timestamp)
        except Exception as e:
            logging.warning(f"Error converting timestamp {timestamp} (type: {type(timestamp)}) to string: {e}")
            # Use a default timestamp format as fallback
            row['timestamp'] = str(timestamp)
        
        rows.append(row)
    
    # Convert to DataFrame
    result_df = pd.DataFrame(rows)
    
    return result_df

def load_technical_indicators(symbol: str, timeframe: str, timestamps: List[str]) -> pd.DataFrame:
    """
    Load technical indicators for a given symbol, timeframe, and list of timestamps.
    
    Args:
        symbol: The cryptocurrency symbol
        timeframe: The timeframe
        timestamps: List of timestamps to match
        
    Returns:
        DataFrame with technical indicators for each timestamp
    """
    table_name = f"ta_{timeframe}"
    
    try:
        with sqlite3.connect(DB_FILE) as conn:
            # First, check if there are any indicators for this symbol
            check_query = f"SELECT COUNT(*) FROM {table_name} WHERE symbol = ?"
            cursor = conn.cursor()
            cursor.execute(check_query, (symbol,))
            total_indicators = cursor.fetchone()[0]
            
            if total_indicators == 0:
                logging.warning(f"No technical indicators found in database for {symbol} ({timeframe})")
                return pd.DataFrame()
            
            logging.debug(f"Found {total_indicators} total indicators for {symbol} ({timeframe})")
            
            # Load ALL indicators for this symbol and timeframe (simpler and more reliable)
            query = f"""
                SELECT *
                FROM {table_name}
                WHERE symbol = ?
                ORDER BY timestamp ASC
            """
            
            df = pd.read_sql_query(query, conn, params=(symbol,))
            
            if df.empty:
                logging.warning(f"No technical indicators loaded for {symbol} ({timeframe})")
                return pd.DataFrame()
            
            # Remove id and symbol columns as they're not needed for ML
            if 'id' in df.columns:
                df = df.drop('id', axis=1)
            if 'symbol' in df.columns:
                df = df.drop('symbol', axis=1)
            
            logging.debug(f"Successfully loaded {len(df)} technical indicators for {symbol} ({timeframe})")
            return df
            
    except Exception as e:
        logging.error(f"Error loading technical indicators for {symbol} ({timeframe}): {e}")
        import traceback
        logging.error(traceback.format_exc())
        return pd.DataFrame()

def normalize_timestamp_format(timestamp_str: str) -> str:
    """
    Normalize timestamp format to ensure consistent merging.
    Converts both 'YYYY-MM-DD HH:MM:SS' and 'YYYY-MM-DDTHH:MM:SS' to 'YYYY-MM-DD HH:MM:SS'.
    
    Args:
        timestamp_str: Timestamp string in any supported format
        
    Returns:
        Normalized timestamp string in 'YYYY-MM-DD HH:MM:SS' format
    """
    if isinstance(timestamp_str, str):
        # Replace 'T' with space if present (ISO format)
        return timestamp_str.replace('T', ' ')
    return str(timestamp_str)

def merge_patterns_with_indicators(patterns_df: pd.DataFrame, indicators_df: pd.DataFrame) -> pd.DataFrame:
    """
    FIXED VERSION: Merge pattern data with technical indicators based on timestamp.
    
    The main fix is normalizing timestamp formats before merging:
    - Pattern timestamps: 'YYYY-MM-DD HH:MM:SS' 
    - Indicator timestamps: 'YYYY-MM-DDTHH:MM:SS' (ISO format)
    
    Args:
        patterns_df: DataFrame with pattern features, target, and timestamp
        indicators_df: DataFrame with technical indicators indexed by timestamp
        
    Returns:
        Merged DataFrame with all features
    """
    if patterns_df.empty or indicators_df.empty:
        return pd.DataFrame()
    
    # Make copies to avoid modifying the original dataframes
    patterns_df = patterns_df.copy()
    indicators_df = indicators_df.copy()
    
    # Handle potential timestamp format issues
    logging.debug(f"Converting timestamps to uniform format for merging")
    logging.debug(f"Pattern timestamps sample: {patterns_df['timestamp'].iloc[0]} (type: {type(patterns_df['timestamp'].iloc[0])})")
    logging.debug(f"Indicator timestamps sample: {indicators_df['timestamp'].iloc[0]} (type: {type(indicators_df['timestamp'].iloc[0])})")
    
    # Convert timestamps to strings and normalize format (FIX APPLIED HERE)
    try:
        patterns_df['timestamp'] = patterns_df['timestamp'].astype(str).apply(normalize_timestamp_format)
        indicators_df['timestamp'] = indicators_df['timestamp'].astype(str).apply(normalize_timestamp_format)
    except Exception as e:
        logging.error(f"Error converting timestamps to string: {e}")
        # Try a more robust approach if simple casting fails
        patterns_df['timestamp'] = patterns_df['timestamp'].apply(lambda x: normalize_timestamp_format(str(x)))
        indicators_df['timestamp'] = indicators_df['timestamp'].apply(lambda x: normalize_timestamp_format(str(x)))
    
    # Log sample values for debugging
    logging.debug(f"After normalization - Pattern timestamps: {patterns_df['timestamp'].iloc[0]}")
    logging.debug(f"After normalization - Indicator timestamps: {indicators_df['timestamp'].iloc[0]}")
    
    # Check for timestamp overlap before merge (ENHANCED DEBUGGING)
    pattern_timestamps = set(patterns_df['timestamp'].tolist())
    indicator_timestamps = set(indicators_df['timestamp'].tolist())
    overlap = pattern_timestamps.intersection(indicator_timestamps)
    
    logging.debug(f"Timestamp overlap: {len(overlap)} out of {len(pattern_timestamps)} patterns")
    
    if len(overlap) == 0:
        logging.warning(f"{Fore.YELLOW}No timestamp overlap found between patterns and indicators!{Style.RESET_ALL}")
        logging.warning(f"Sample pattern timestamps: {list(pattern_timestamps)[:3]}")
        logging.warning(f"Sample indicator timestamps: {list(indicator_timestamps)[:3]}")
        return pd.DataFrame()
    
    # Merge the dataframes on timestamp
    merged_df = pd.merge(patterns_df, indicators_df, on='timestamp', how='inner')
    
    logging.debug(f"Merge resulted in {len(merged_df)} records")
    
    # Check for any NaN values and drop rows that contain them
    rows_before = len(merged_df)
    merged_df = merged_df.dropna()
    rows_after = len(merged_df)
    
    if rows_before > rows_after:
        dropped_rows = rows_before - rows_after
        logging.warning(f"{Fore.YELLOW}Dropped {dropped_rows} rows due to NaN values ({dropped_rows/rows_before:.2%} of total){Style.RESET_ALL}")
    
    return merged_df

def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add refined binary classification labels excluding neutral zone.
    
    Uses both BUY_THRESHOLD and SELL_THRESHOLD to create cleaner classes:
    - Class 1 (BUY): y > BUY_THRESHOLD (strong positive volatility)
    - Class 0 (SELL): y < SELL_THRESHOLD (strong negative volatility) 
    - Excluded: SELL_THRESHOLD <= y <= BUY_THRESHOLD (neutral zone)
    
    This approach creates more distinct classes by excluding ambiguous signals.
    """
    if 'y' not in df.columns:
        return df
    
    df = df.copy()
    initial_count = len(df)
    
    # Create masks for clear signals
    buy_mask = df['y'] > BUY_THRESHOLD
    sell_mask = df['y'] < SELL_THRESHOLD
    neutral_mask = (df['y'] >= SELL_THRESHOLD) & (df['y'] <= BUY_THRESHOLD)
    
    # Only keep samples with clear signals (exclude neutral zone)
    clear_signal_mask = buy_mask | sell_mask
    df_filtered = df[clear_signal_mask].copy()
    
    # Assign refined binary labels: 1 for BUY, 0 for SELL
    df_filtered['y_class'] = buy_mask[clear_signal_mask].astype(int)
    
    # Log filtering statistics
    final_count = len(df_filtered)
    excluded_count = initial_count - final_count
    neutral_count = neutral_mask.sum()
    
    logging.info(f"Refined binary labeling statistics:")
    logging.info(f"  Initial samples: {initial_count}")
    logging.info(f"  BUY samples (y > {BUY_THRESHOLD}): {buy_mask.sum()}")
    logging.info(f"  SELL samples (y < {SELL_THRESHOLD}): {sell_mask.sum()}")
    logging.info(f"  Neutral zone excluded ({SELL_THRESHOLD} <= y <= {BUY_THRESHOLD}): {neutral_count}")
    logging.info(f"  Final samples retained: {final_count} ({(final_count/initial_count)*100:.1f}%)")
    logging.info(f"  Samples excluded: {excluded_count} ({(excluded_count/initial_count)*100:.1f}%)")
    
    if final_count < initial_count * 0.5:
        logging.warning(f"{Fore.YELLOW}Warning: More than 50% of samples excluded due to neutral zone filtering. "
                       f"Consider adjusting BUY_THRESHOLD ({BUY_THRESHOLD}) or SELL_THRESHOLD ({SELL_THRESHOLD}){Style.RESET_ALL}")
    
    return df_filtered

if __name__ == "__main__":
    # Set up logging
    setup_logging(level=logging.INFO)
    
    # Example usage
    import asyncio
    
    async def test():
        success = await generate_full_ml_dataset(
            symbol="BTC/USDT",
            timeframe="5m",
            window_size=7
        )
        print(f"Generation {'successful' if success else 'failed'}")
    
    asyncio.run(test())
