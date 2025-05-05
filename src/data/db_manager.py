# db_manager.py

import sqlite3
import pandas as pd
import logging
import os
from datetime import datetime, timedelta
from src.utils.config import DB_FILE, ENABLED_TIMEFRAMES

def init_data_tables():
    """Initialize database tables for all enabled timeframes."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    for timeframe in ENABLED_TIMEFRAMES:
        # Create table for each timeframe
        table_name = f"data_{timeframe}"
        
        try:
            # Use VARCHAR for timestamp to maintain ISO format for easier readability
            # This could be converted to Unix timestamp if needed for better performance
            cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {table_name} (
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                ema5 REAL,
                ema10 REAL,
                ema20 REAL,
                macd REAL,
                macd_signal REAL,
                macd_histogram REAL,
                rsi_fast REAL,
                stoch_rsi REAL,
                atr REAL,
                bollinger_hband REAL,
                bollinger_lband REAL,
                bollinger_pband REAL,
                vwap REAL,
                adx REAL,
                roc REAL,
                log_return REAL,
                tenkan_sen REAL,
                kijun_sen REAL,
                senkou_span_a REAL,
                senkou_span_b REAL,
                chikou_span REAL,
                williams_r REAL,
                obv REAL,
                sma_fast REAL,
                sma_slow REAL,
                sma_fast_trend REAL,
                sma_slow_trend REAL,
                sma_cross REAL,
                close_lag_1 REAL,
                volume_lag_1 REAL,
                weekday_sin REAL,
                weekday_cos REAL,
                hour_sin REAL,
                hour_cos REAL,
                mfi REAL,
                cci REAL,
                PRIMARY KEY (timestamp, symbol)
            )
            ''')
            
            # Create index on symbol for faster lookups by symbol
            cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_symbol ON {table_name} (symbol)")
            
            # Create index on timestamp for faster lookups by time
            cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_timestamp ON {table_name} (timestamp)")
            
        except Exception as e:
            logging.error(f"Error creating table {table_name}: {e}")
    
    conn.commit()
    conn.close()

def save_data(symbol, df, timeframe):
    """
    Save pandas DataFrame to SQLite database for a specific symbol and timeframe.
    """
    if df.empty:
        logging.warning(f"No data to save for {symbol} ({timeframe})")
        return False
        
    # Reset index to make timestamp a regular column
    df_copy = df.reset_index()
    
    # Convert timestamp to ISO format string for better human readability
    df_copy['timestamp'] = df_copy['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S')
    
    # Add symbol column
    df_copy['symbol'] = symbol
    
    # Convert DataFrame to list of tuples for efficient insertion
    records = df_copy.to_records(index=False)
    
    # Create string of column names dynamically based on DataFrame columns
    columns = ', '.join(df_copy.columns)
    placeholders = ', '.join(['?'] * len(df_copy.columns))
    
    table_name = f"data_{timeframe}"
    
    # Connect to database
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    try:
        # Use INSERT OR REPLACE to handle existing records
        insert_query = f'''
        INSERT OR REPLACE INTO {table_name} ({columns})
        VALUES ({placeholders})
        '''
        
        # Execute insert for all records
        cursor.executemany(insert_query, list(records))
        
        conn.commit()
        logging.debug(f"Saved {len(df_copy)} rows for {symbol} ({timeframe})")
        return True
    except Exception as e:
        conn.rollback()
        logging.error(f"Error saving data for {symbol} ({timeframe}): {e}")
        return False
    finally:
        conn.close()

def get_symbol_data(symbol, timeframe, limit=None):
    """
    Retrieve data for a specific symbol and timeframe from the database.
    Returns a pandas DataFrame.
    """
    table_name = f"data_{timeframe}"
    
    # Connect to database
    conn = sqlite3.connect(DB_FILE)
    
    # Build query
    query = f"SELECT * FROM {table_name} WHERE symbol = ? ORDER BY timestamp DESC"
    
    if limit:
        query += f" LIMIT {limit}"
    
    # Execute query and load into DataFrame
    try:
        df = pd.read_sql_query(query, conn, params=(symbol,))
        
        if not df.empty:
            # Convert timestamp string back to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)  # Sort by timestamp in ascending order
            
        conn.close()
        return df
    except Exception as e:
        logging.error(f"Error retrieving data for {symbol} ({timeframe}): {e}")
        conn.close()
        return pd.DataFrame()

def check_data_freshness(symbol, timeframe, max_age_days=1):
    """
    Check if we already have fresh data for a symbol and timeframe.
    Returns a tuple of (is_fresh, last_timestamp).
    """
    table_name = f"data_{timeframe}"
    
    # Connect to database
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    try:
        # Get the most recent timestamp for this symbol
        cursor.execute(f"SELECT MAX(timestamp) FROM {table_name} WHERE symbol = ?", (symbol,))
        result = cursor.fetchone()
        
        if not result or not result[0]:
            # No data exists for this symbol
            conn.close()
            return False, None
            
        # Parse the timestamp string to a datetime object
        last_timestamp = datetime.strptime(result[0], '%Y-%m-%dT%H:%M:%S')
        
        # Check if the last timestamp is within max_age_days
        now = datetime.utcnow()
        age_days = (now - last_timestamp).total_seconds() / (24 * 3600)
        
        is_fresh = age_days <= max_age_days
        
        conn.close()
        return is_fresh, last_timestamp
        
    except Exception as e:
        logging.error(f"Error checking data freshness for {symbol} ({timeframe}): {e}")
        conn.close()
        return False, None

def get_data_last_timestamp(symbol, timeframe):
    """
    Get the last timestamp for a symbol and timeframe.
    Returns a datetime object or None if no data exists.
    """
    table_name = f"data_{timeframe}"
    
    # Connect to database
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    try:
        # Get the most recent timestamp for this symbol
        cursor.execute(f"SELECT MAX(timestamp) FROM {table_name} WHERE symbol = ?", (symbol,))
        result = cursor.fetchone()
        
        if not result or not result[0]:
            # No data exists for this symbol
            conn.close()
            return None
            
        # Parse the timestamp string to a datetime object
        last_timestamp = datetime.strptime(result[0], '%Y-%m-%dT%H:%M:%S')
        
        conn.close()
        return last_timestamp
        
    except Exception as e:
        logging.error(f"Error getting last timestamp for {symbol} ({timeframe}): {e}")
        conn.close()
        return None

def get_symbol_data_info(symbol, timeframe):
    """
    Get information about the data for a symbol and timeframe.
    Returns a dictionary with count, first_date, and last_date.
    """
    table_name = f"data_{timeframe}"
    
    # Connect to database
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    try:
        # Get the count, min and max timestamps
        cursor.execute(f"""
            SELECT COUNT(*), MIN(timestamp), MAX(timestamp)
            FROM {table_name} 
            WHERE symbol = ?
        """, (symbol,))
        
        count, min_timestamp, max_timestamp = cursor.fetchone()
        
        # Parse timestamps if they exist
        first_date = datetime.strptime(min_timestamp, '%Y-%m-%dT%H:%M:%S') if min_timestamp else None
        last_date = datetime.strptime(max_timestamp, '%Y-%m-%dT%H:%M:%S') if max_timestamp else None
        
        conn.close()
        
        return {
            'count': count,
            'first_date': first_date,
            'last_date': last_date
        }
        
    except Exception as e:
        logging.error(f"Error getting data info for {symbol} ({timeframe}): {e}")
        conn.close()
        return {'count': 0, 'first_date': None, 'last_date': None}
