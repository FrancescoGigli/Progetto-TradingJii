# add_volatility_columns.py

import logging
import pandas as pd
import numpy as np
import sys
import os
import sqlite3
import argparse
from datetime import datetime, timedelta
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.volatility_utils import calculate_volatility_rate
from src.data.subseries_utils import get_all_subseries_with_categories, binary_to_pattern
from src.utils.config import DB_FILE, SUBSERIES_LENGTH, ENABLED_TIMEFRAMES
from src.data.db_manager import (
    init_data_tables, save_subseries_data, get_top_categories, get_category_transitions
)

def add_volatility_columns(timeframe):
    """
    Add volatility columns to the data table for a specific timeframe.
    
    Args:
        timeframe: The timeframe to modify (e.g., '5m', '15m')
        
    Returns:
        True if successful, False otherwise
    """
    logging.info(f"Adding volatility columns to data_{timeframe} table...")
    
    # Connect to database
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    try:
        # Check if the columns already exist
        cursor.execute(f"PRAGMA table_info(data_{timeframe})")
        columns = [col[1] for col in cursor.fetchall()]
        
        # Columns to add if they don't exist
        volatility_columns = [
            "close_volatility REAL", 
            "open_volatility REAL", 
            "high_volatility REAL", 
            "low_volatility REAL",
            "volume_change REAL",
            "historical_volatility REAL"
        ]
        
        # Only add columns that don't exist
        for col in volatility_columns:
            col_name = col.split()[0]
            if col_name not in columns:
                cursor.execute(f"ALTER TABLE data_{timeframe} ADD COLUMN {col}")
                logging.info(f"Added column {col_name} to data_{timeframe} table")
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logging.error(f"Error adding volatility columns to data_{timeframe} table: {e}")
        conn.rollback()
        conn.close()
        return False

def process_symbol_volatility(symbol, timeframe, lookback_days=30):
    """
    Process volatility data for a specific symbol and timeframe.
    
    Args:
        symbol: The symbol to process
        timeframe: The timeframe to process
        lookback_days: Number of days to look back
        
    Returns:
        Tuple of (DataFrame with volatility, Dictionary of categorized subseries)
    """
    logging.info(f"Processing volatility for {symbol} ({timeframe})...")
    
    # Connect to database
    conn = sqlite3.connect(DB_FILE)
    
    try:
        # Calculate cutoff date
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%dT%H:%M:%S')
        
        # Query data for this symbol
        query = f"""
            SELECT timestamp, open, high, low, close, volume
            FROM data_{timeframe}
            WHERE symbol = ? AND timestamp > ?
            ORDER BY timestamp
        """
        
        df = pd.read_sql_query(query, conn, params=(symbol, cutoff_date))
        
        if df.empty:
            logging.warning(f"No data found for {symbol} ({timeframe}) after {cutoff_date}")
            conn.close()
            return None, None
        
        # Process timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Calculate volatility
        df_vol = calculate_volatility_rate(df)
        
        # Update volatility in database
        cursor = conn.cursor()
        
        # Update each row with volatility values
        for idx, row in df_vol.iterrows():
            # Only update if volatility columns have values
            if pd.notna(row.get('close_volatility')):
                ts = idx.strftime('%Y-%m-%dT%H:%M:%S')
                
                update_cols = []
                update_vals = []
                
                # Add all volatility columns that exist
                for col in ['close_volatility', 'open_volatility', 'high_volatility', 
                           'low_volatility', 'volume_change', 'historical_volatility']:
                    if col in df_vol.columns and pd.notna(row.get(col)):
                        update_cols.append(f"{col} = ?")
                        update_vals.append(row.get(col))
                
                if update_cols:
                    query = f"""
                        UPDATE data_{timeframe}
                        SET {", ".join(update_cols)}
                        WHERE symbol = ? AND timestamp = ?
                    """
                    cursor.execute(query, update_vals + [symbol, ts])
        
        conn.commit()
        
        # Get all subseries and categorize them
        categorized = None
        if len(df_vol) >= SUBSERIES_LENGTH:
            categorized = get_all_subseries_with_categories(df_vol)
            
            # Save subseries data
            if categorized:
                saved = save_subseries_data(symbol, timeframe, categorized)
                logging.info(f"Saved {saved} subseries for {symbol} ({timeframe})")
        
        conn.close()
        return df_vol, categorized
        
    except Exception as e:
        logging.error(f"Error processing volatility for {symbol} ({timeframe}): {e}")
        conn.close()
        return None, None

def process_timeframe(timeframe, symbol_limit=None, lookback_days=30):
    """
    Process all symbols for a specific timeframe.
    
    Args:
        timeframe: The timeframe to process
        symbol_limit: Maximum number of symbols to process
        lookback_days: Number of days to look back
        
    Returns:
        Dictionary with statistics
    """
    logging.info(f"Processing {timeframe} data...")
    
    # First add volatility columns if needed
    add_volatility_columns(timeframe)
    
    # Connect to database
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    try:
        # Get available symbols
        cursor.execute(f"SELECT DISTINCT symbol FROM data_{timeframe} LIMIT {symbol_limit if symbol_limit else 1000000}")
        symbols = [row[0] for row in cursor.fetchall()]
        
        if not symbols:
            logging.warning(f"No symbols found for timeframe {timeframe}")
            conn.close()
            return {'timeframe': timeframe, 'symbols_processed': 0, 'categories': 0, 'subseries': 0}
        
        if symbol_limit:
            symbols = symbols[:symbol_limit]
        
        logging.info(f"Found {len(symbols)} symbols for timeframe {timeframe}")
        
        # Process each symbol
        categories_count = 0
        subseries_count = 0
        symbols_processed = 0
        
        for symbol in tqdm(symbols, desc=f"Processing {timeframe} symbols"):
            df_vol, categorized = process_symbol_volatility(symbol, timeframe, lookback_days)
            
            if df_vol is not None:
                symbols_processed += 1
                
                if categorized:
                    categories_count += len(categorized)
                    
                    # Count subseries
                    for category, subseries_list in categorized.items():
                        subseries_count += len(subseries_list)
        
        conn.close()
        
        return {
            'timeframe': timeframe,
            'symbols_processed': symbols_processed,
            'categories': categories_count,
            'subseries': subseries_count
        }
        
    except Exception as e:
        logging.error(f"Error processing {timeframe}: {e}")
        conn.close()
        return {'timeframe': timeframe, 'error': str(e)}

def verify_results():
    """Verify the processing results."""
    logging.info("Verifying results...")
    
    # Get top categories
    top_categories = get_top_categories(limit=10)
    print("\n==== Top Categories ====")
    for category_id, pattern, count in top_categories:
        print(f"Category: {category_id}, Pattern: {pattern}, Count: {count}")
    
    # Get top transitions
    transitions = get_category_transitions()
    print("\n==== Sample Category Transitions ====")
    count = 0
    for from_cat, to_cats in transitions.items():
        if count >= 5:
            break
        
        print(f"From {from_cat} ({binary_to_pattern(from_cat)}):")
        top_transitions = sorted(to_cats.items(), key=lambda x: x[1], reverse=True)[:3]
        for to_cat, prob in top_transitions:
            print(f"  -> {to_cat} ({binary_to_pattern(to_cat)}): {prob:.4f}")
        count += 1

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Add volatility columns and process real data")
    parser.add_argument('--timeframes', type=str, default="5m,15m", help="Comma-separated list of timeframes to process")
    parser.add_argument('--symbols', type=int, default=3, help="Maximum number of symbols to process per timeframe")
    parser.add_argument('--days', type=int, default=14, help="Number of days to look back for data")
    parser.add_argument('--log-level', type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("\n===== ADDING VOLATILITY COLUMNS AND PROCESSING DATA =====")
    
    # Initialize database tables if they don't exist
    init_data_tables()
    
    # Process each timeframe
    timeframes = args.timeframes.split(',')
    results = []
    
    for timeframe in timeframes:
        result = process_timeframe(
            timeframe.strip(),
            symbol_limit=args.symbols,
            lookback_days=args.days
        )
        results.append(result)
    
    # Print summary
    print("\n==== Processing Summary ====")
    for result in results:
        if 'error' in result:
            print(f"Error processing {result['timeframe']}: {result['error']}")
        else:
            print(f"Timeframe: {result['timeframe']}")
            print(f"  Symbols processed: {result['symbols_processed']}")
            print(f"  Categories: {result['categories']}")
            print(f"  Subseries: {result['subseries']}")
    
    # Verify results
    verify_results()
    
    print("\n===== DONE =====")
