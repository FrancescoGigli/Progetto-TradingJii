#!/usr/bin/env python3
"""
Volatility Processor Module for TradingJii

This module computes and manages cryptocurrency price volatility data:
- Extracts close price series from the SQLite database
- Computes percentage volatility (Pt / Pt-1 - 1) * 100
- Cleans the data to remove extreme values
- Stores the result in a dedicated volatility table

Integrates with the real_time.py update flow.
"""

import sqlite3
import logging
import pandas as pd
from typing import Tuple, Optional
from colorama import Fore, Style
from modules.utils.config import DB_FILE

def load_close_series(symbol: str, timeframe: str) -> pd.DataFrame:
    """
    Extract the close price time series for a given symbol and timeframe from the SQLite database.
    
    Args:
        symbol: The cryptocurrency symbol (e.g., "BTC/USDT")
        timeframe: The timeframe (e.g., "5m")
        
    Returns:
        A pandas DataFrame with timestamp and close columns, sorted chronologically
    """
    table_name = f"market_data_{timeframe}"
    
    try:
        with sqlite3.connect(DB_FILE) as conn:
            query = f"""
                SELECT timestamp, close
                FROM {table_name}
                WHERE symbol = ?
                ORDER BY timestamp ASC
            """
            
            df = pd.read_sql_query(query, conn, params=(symbol,))
            
            if df.empty:
                logging.warning(f"No data found for {symbol} in timeframe {timeframe}")
                return pd.DataFrame(columns=['timestamp', 'close'])
                
            # Ensure timestamp is string in the format '%Y-%m-%dT%H:%M:%S'
            # (already in this format based on db_manager.py save_ohlcv_data)
            return df
            
    except Exception as e:
        logging.error(f"Error loading close series for {symbol} ({timeframe}): {e}")
        return pd.DataFrame(columns=['timestamp', 'close'])

def compute_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute percentage volatility from the close prices.
    
    Args:
        df: A DataFrame with columns timestamp, close
        
    Returns:
        A DataFrame with timestamp and volatility columns
    """
    if df.empty or len(df) < 2:
        logging.warning("Insufficient data to compute volatility (need at least 2 data points)")
        return pd.DataFrame(columns=['timestamp', 'volatility'])
        
    # Create a copy to avoid modifying the original dataframe
    result_df = df.copy()
    
    # Compute percentage change and multiply by 100
    result_df['volatility'] = result_df['close'].pct_change() * 100
    
    # Drop the first row which contains NaN for volatility
    result_df = result_df.dropna()
    
    # Select only the columns we need
    return result_df[['timestamp', 'volatility']]

def clean_volatility(df: pd.DataFrame, clip_range: Tuple[int, int] = (-100, 100)) -> pd.DataFrame:
    """
    Clean the volatility series by removing invalid or extreme values.
    
    Args:
        df: A DataFrame with timestamp, volatility
        clip_range: Default (-100, 100) - values outside this range will be clipped
        
    Returns:
        Cleaned DataFrame with no NaN or inf values and clipped volatility
    """
    if df.empty:
        return df
        
    # Create a copy to avoid modifying the original dataframe
    cleaned_df = df.copy()
    
    # Remove any NaN or inf values
    cleaned_df = cleaned_df.replace([float('inf'), float('-inf')], pd.NA)
    cleaned_df = cleaned_df.dropna()
    
    # Clip values to the specified range
    cleaned_df['volatility'] = cleaned_df['volatility'].clip(clip_range[0], clip_range[1])
    
    return cleaned_df

def save_volatility(symbol: str, timeframe: str, df: pd.DataFrame) -> bool:
    """
    Update the volatility column in the unified market data table.
    
    Args:
        symbol: The cryptocurrency symbol
        timeframe: The timeframe (e.g., "5m")
        df: A DataFrame with columns timestamp, volatility
        
    Returns:
        Boolean indicating success
    """
    if df.empty:
        logging.warning(f"No volatility data to save for {symbol} ({timeframe})")
        return False
        
    table_name = f"market_data_{timeframe}"
    
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            
            # Verify the table exists
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
            if not cursor.fetchone():
                logging.warning(f"{Fore.YELLOW}Table {table_name} does not exist. Cannot save volatility data.{Style.RESET_ALL}")
                return False
            
            # Prepare data for UPDATE
            records_updated = 0
            for _, row in df.iterrows():
                try:
                    # Update only the volatility column in the unified table
                    cursor.execute(f"""
                        UPDATE {table_name}
                        SET volatility = ?
                        WHERE symbol = ? AND timestamp = ?
                    """, (row['volatility'], symbol, row['timestamp']))
                    
                    # Check if the update affected any rows
                    if cursor.rowcount > 0:
                        records_updated += 1
                    else:
                        logging.debug(f"No matching record found for {symbol} at {row['timestamp']} - volatility update skipped")
                except Exception as e:
                    logging.error(f"Error updating volatility for {symbol} at {row['timestamp']}: {e}")
                    continue
            
            conn.commit()
            
            logging.info(f"Updated {Fore.GREEN}{records_updated}{Style.RESET_ALL} volatility values for {Fore.YELLOW}{symbol}{Style.RESET_ALL} ({timeframe})")
            return records_updated > 0
            
    except Exception as e:
        logging.error(f"Error saving volatility data for {symbol} ({timeframe}): {e}")
        return False

def process_and_save_volatility(symbol: str, timeframe: str) -> None:
    """
    Run the full pipeline for a given symbol and timeframe:
    - Load close series
    - Compute percentage volatility
    - Clean the volatility data
    - Save the result to database
    
    Args:
        symbol: The cryptocurrency symbol
        timeframe: The timeframe
    """
    try:
        logging.debug(f"Processing volatility for {symbol} ({timeframe})")
        
        # Step 1: Load close price series
        close_df = load_close_series(symbol, timeframe)
        if close_df.empty:
            logging.warning(f"No close price data available for {symbol} ({timeframe})")
            return
            
        # Step 2: Compute volatility
        volatility_df = compute_volatility(close_df)
        if volatility_df.empty:
            logging.warning(f"Could not compute volatility for {symbol} ({timeframe})")
            return
            
        # Step 3: Clean volatility data
        cleaned_df = clean_volatility(volatility_df)
        if cleaned_df.empty:
            logging.warning(f"No valid volatility data after cleaning for {symbol} ({timeframe})")
            return
            
        # Step 4: Save to database
        success = save_volatility(symbol, timeframe, cleaned_df)
        
        if success:
            logging.info(f"Successfully processed and saved volatility for {Fore.YELLOW}{symbol}{Style.RESET_ALL} ({timeframe})")
        else:
            logging.warning(f"Failed to save volatility for {symbol} ({timeframe})")
            
    except Exception as e:
        logging.error(f"Error in volatility processing pipeline for {symbol} ({timeframe}): {e}")
        import traceback
        logging.error(traceback.format_exc())
