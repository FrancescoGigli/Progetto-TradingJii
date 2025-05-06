# volatility_test.py

import logging
import pandas as pd
import numpy as np
import sys
import os
import sqlite3
import argparse
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.volatility_utils import calculate_volatility_rate, convert_to_volatility_series, calculate_historical_volatility
from src.data.subseries_utils import create_subseries, categorize_subseries, get_all_subseries_with_categories, binary_to_pattern
from src.utils.config import DB_FILE, SUBSERIES_LENGTH
from src.data.db_manager import (
    get_symbol_data, init_data_tables, save_subseries_data,
    save_category, save_subseries_occurrence, update_category_transition,
    get_category_transitions, get_top_categories
)

def test_volatility_calculation():
    """Test volatility rate calculation on sample data."""
    logging.info("Testing volatility rate calculation...")
    
    # Create sample OHLCV data
    dates = pd.date_range(start='2023-01-01', periods=10)
    data = {
        'open': [100, 102, 101, 103, 104, 105, 103, 102, 101, 104],
        'high': [105, 106, 105, 107, 108, 107, 106, 105, 103, 106],
        'low': [98, 100, 99, 101, 102, 102, 100, 99, 98, 102],
        'close': [102, 101, 103, 104, 105, 103, 102, 101, 104, 105],
        'volume': [1000, 1200, 900, 1100, 1300, 1250, 1150, 1050, 1200, 1400]
    }
    df = pd.DataFrame(data, index=dates)
    
    # Calculate volatility rates
    df_vol = calculate_volatility_rate(df)
    print("\n==== Sample data with volatility rates ====")
    print(df_vol[['close', 'close_volatility']].tail())
    
    # Convert to volatility series only
    vol_series = convert_to_volatility_series(df)
    print("\n==== Volatility series only ====")
    print(vol_series.tail())
    
    # Calculate historical volatility
    df_hist_vol = calculate_historical_volatility(df)
    print("\n==== Historical volatility ====")
    print(df_hist_vol[['close', 'historical_volatility']].tail())
    
    return df_vol

def test_subseries_creation(df):
    """Test subseries creation and categorization."""
    logging.info("Testing subseries creation and categorization...")
    
    # Create subseries
    subseries_list = create_subseries(df, length=SUBSERIES_LENGTH)
    print(f"\n==== Created {len(subseries_list)} subseries of length {SUBSERIES_LENGTH} ====")
    
    if subseries_list:
        # Show first subseries
        print("\n==== First subseries ====")
        print(subseries_list[0][['close', 'close_volatility']].head())
        
        # Categorize first subseries
        try:
            category = categorize_subseries(subseries_list[0])
            pattern = binary_to_pattern(category)
            print(f"\n==== Category: {category} ====")
            print(f"Pattern: {pattern}")
            
            # Show direction of price movement
            directions = (subseries_list[0]['close_volatility'] >= 0).astype(int)
            print("\n==== Price movement directions (1=up, 0=down) ====")
            print(directions)
            
        except Exception as e:
            print(f"Error categorizing subseries: {e}")
    
    return subseries_list

def test_with_real_data(symbol="BTC/USDT", timeframe="1h", limit=100):
    """Test with real data from the database."""
    logging.info(f"Testing with real data: {symbol} ({timeframe})...")
    
    try:
        # Get data from database
        df = get_symbol_data(symbol, timeframe, limit)
        
        if df.empty:
            print(f"No data found for {symbol} ({timeframe})")
            return None
        
        print(f"\n==== Retrieved {len(df)} rows for {symbol} ({timeframe}) ====")
        print(df.head())
        
        # Calculate volatility rates
        df_vol = calculate_volatility_rate(df)
        print("\n==== Real data with volatility rates ====")
        print(df_vol[['close', 'close_volatility']].head())
        
        # Get all subseries and categorize them
        categorized = get_all_subseries_with_categories(df_vol)
        
        print(f"\n==== Found {len(categorized)} categories ====")
        for category, subseries in list(categorized.items())[:5]:  # Show top 5 categories
            pattern = binary_to_pattern(category)
            print(f"Category: {category}, Pattern: {pattern}, Count: {len(subseries)}")
        
        return categorized
    
    except Exception as e:
        print(f"Error testing with real data: {e}")
        import traceback
        print(traceback.format_exc())
        return None

def test_database_operations(symbol="Sample", timeframe="test", save_to_db=False):
    """Test the database operations for subseries categorization."""
    logging.info("Testing database operations for subseries categorization...")
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=30)
    data = {
        'open': np.random.normal(100, 2, 30),
        'high': np.random.normal(102, 2, 30),
        'low': np.random.normal(98, 2, 30),
        'close': np.random.normal(101, 2, 30),
        'volume': np.random.normal(1000, 200, 30)
    }
    df = pd.DataFrame(data, index=dates)
    
    # Calculate volatility
    df_vol = calculate_volatility_rate(df)
    
    # Get categorized subseries
    categorized = get_all_subseries_with_categories(df_vol)
    
    print(f"\n==== Found {len(categorized)} categories in sample data ====")
    
    if not save_to_db:
        print("Skipping database save (dry run)")
        return
    
    # Initialize database tables if they don't exist
    init_data_tables()
    
    # Save subseries data to database
    saved_count = save_subseries_data(symbol, timeframe, categorized)
    print(f"\n==== Saved {saved_count} subseries to database ====")
    
    # Get top categories
    top_categories = get_top_categories(limit=5)
    print("\n==== Top Categories ====")
    for category_id, pattern, count in top_categories:
        print(f"Category: {category_id}, Pattern: {pattern}, Count: {count}")
    
    # Get category transitions
    transitions = get_category_transitions()
    print("\n==== Sample Category Transitions ====")
    for from_cat, to_cats in list(transitions.items())[:3]:
        print(f"From {from_cat}:")
        for to_cat, prob in list(to_cats.items())[:2]:
            print(f"  -> {to_cat}: {prob:.4f}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Test and manage volatility analysis and subseries categorization")
    parser.add_argument('--init-db', action='store_true', help="Initialize database tables")
    parser.add_argument('--save', action='store_true', help="Save data to database")
    parser.add_argument('--sample', action='store_true', help="Test with sample data only")
    parser.add_argument('--symbol', type=str, default="BTC/USDT", help="Symbol to use for testing")
    parser.add_argument('--timeframe', type=str, default="1h", help="Timeframe to use for testing")
    parser.add_argument('--limit', type=int, default=200, help="Limit for querying data")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize database if requested
    if args.init_db:
        print("Initializing database tables...")
        init_data_tables()
        print("Database initialization complete.")
    
    # Test with sample data
    print("\n===== TESTING WITH SAMPLE DATA =====")
    df_vol = test_volatility_calculation()
    subseries = test_subseries_creation(df_vol)
    
    # Test database operations with sample data
    print("\n===== TESTING DATABASE OPERATIONS =====")
    test_database_operations(symbol="Sample", timeframe="test", save_to_db=args.save)
    
    # Skip real data test if sample only
    if args.sample:
        print("\nSkipping real data test as requested with --sample")
        sys.exit(0)
    
    # Test with real data if available
    print("\n===== TESTING WITH REAL DATA =====")
    # Check if database exists
    if os.path.exists(DB_FILE):
        # Check available symbols
        try:
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            
            # Try to get the requested symbol
            table_name = f"data_{args.timeframe}"
            try:
                cursor.execute(f"SELECT DISTINCT symbol FROM {table_name} WHERE symbol = ? LIMIT 1", (args.symbol,))
                symbol_result = cursor.fetchone()
                
                if symbol_result:
                    symbol = symbol_result[0]
                else:
                    # If requested symbol not found, get any symbol
                    cursor.execute(f"SELECT DISTINCT symbol FROM {table_name} LIMIT 1")
                    symbol_result = cursor.fetchone()
                    
                    if symbol_result:
                        symbol = symbol_result[0]
                        print(f"Requested symbol {args.symbol} not found, using {symbol} instead.")
                    else:
                        print(f"No symbols found in {table_name} table.")
                        symbol = None
            except Exception as e:
                print(f"Error checking for symbol: {e}")
                symbol = None
                
            conn.close()
            
            if symbol:
                categorized = test_with_real_data(symbol=symbol, timeframe=args.timeframe, limit=args.limit)
                
                # If save is enabled and we have categories, save them to database
                if args.save and categorized:
                    print("\n==== Saving real data categories to database ====")
                    saved_count = save_subseries_data(symbol, args.timeframe, categorized)
                    print(f"Saved {saved_count} subseries to database.")
            else:
                print("No suitable symbols found in database")
        except Exception as e:
            print(f"Error checking database: {e}")
    else:
        print(f"Database file {DB_FILE} not found. Skipping real data test.")
        
    print("\n===== DONE =====")
