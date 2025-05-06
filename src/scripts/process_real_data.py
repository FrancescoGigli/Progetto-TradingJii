# process_real_data.py

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
from src.utils.config import DB_FILE, SUBSERIES_LENGTH
from src.data.db_manager import (
    get_symbol_data, save_data, init_data_tables,
    save_subseries_data, get_top_categories, get_category_transitions
)

def process_timeframe_data(timeframe, symbol_limit=None, lookback_days=30):
    """
    Process data for a specific timeframe, calculate volatility,
    and save subseries categorization.
    
    Args:
        timeframe: The timeframe to process (e.g., '5m', '15m')
        symbol_limit: Maximum number of symbols to process (None for all)
        lookback_days: Number of days to look back for data
        
    Returns:
        Dictionary with statistics
    """
    logging.info(f"Processing {timeframe} data...")
    
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
            return {'timeframe': timeframe, 'symbols_processed': 0, 'rows_processed': 0, 'categories': 0, 'subseries': 0}
        
        if symbol_limit:
            symbols = symbols[:symbol_limit]
        
        logging.info(f"Found {len(symbols)} symbols for timeframe {timeframe}")
        
        # Calculate cutoff date
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%dT%H:%M:%S')
        
        # Process each symbol
        categories_count = 0
        subseries_count = 0
        rows_processed = 0
        
        for symbol in tqdm(symbols, desc=f"Processing {timeframe} symbols"):
            # Get data for this symbol
            cursor.execute(f"""
                SELECT * FROM data_{timeframe}
                WHERE symbol = ? AND timestamp > ?
                ORDER BY timestamp
            """, (symbol, cutoff_date))
            
            columns = [col[0] for col in cursor.description]
            rows = cursor.fetchall()
            
            if not rows:
                logging.warning(f"No data found for {symbol} ({timeframe}) after {cutoff_date}")
                continue
            
            # Convert to DataFrame
            df = pd.DataFrame(rows, columns=columns)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Keep only OHLCV columns for volatility calculation
            ohlcv_df = df[['open', 'high', 'low', 'close', 'volume']].copy()
            
            # Calculate volatility
            volatility_df = calculate_volatility_rate(ohlcv_df)
            
            # Update the original dataframe with volatility columns
            for col in volatility_df.columns:
                if col not in ohlcv_df.columns:
                    df[col] = volatility_df[col]
            
            # Save updated data back to database
            save_data(symbol, df, timeframe)
            rows_processed += len(df)
            
            # Get subseries categorization
            if len(df) >= SUBSERIES_LENGTH:
                categorized = get_all_subseries_with_categories(df)
                if categorized:
                    # Save subseries data
                    saved = save_subseries_data(symbol, timeframe, categorized)
                    subseries_count += saved
                    categories_count += len(categorized)
                    logging.debug(f"Saved {saved} subseries for {symbol} ({timeframe})")
        
        conn.close()
        
        return {
            'timeframe': timeframe,
            'symbols_processed': len(symbols),
            'rows_processed': rows_processed,
            'categories': categories_count,
            'subseries': subseries_count
        }
        
    except Exception as e:
        logging.error(f"Error processing {timeframe} data: {e}")
        import traceback
        traceback.print_exc()
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
    parser = argparse.ArgumentParser(description="Process real data for volatility analysis")
    parser.add_argument('--timeframes', type=str, default="5m,15m", help="Comma-separated list of timeframes to process")
    parser.add_argument('--symbols', type=int, default=None, help="Maximum number of symbols to process per timeframe")
    parser.add_argument('--days', type=int, default=30, help="Number of days to look back for data")
    parser.add_argument('--log-level', type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("\n===== PROCESSING REAL DATA =====")
    
    # Initialize database tables if they don't exist
    init_data_tables()
    
    # Process each timeframe
    timeframes = args.timeframes.split(',')
    results = []
    
    for timeframe in timeframes:
        result = process_timeframe_data(
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
            print(f"  Rows processed: {result['rows_processed']}")
            print(f"  Categories: {result['categories']}")
            print(f"  Subseries: {result['subseries']}")
    
    # Verify results
    verify_results()
    
    print("\n===== DONE =====")
