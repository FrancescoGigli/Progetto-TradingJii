# db_manager.py

import sqlite3
import pandas as pd
import logging
import os
from datetime import datetime, timedelta
from src.utils.config import DB_FILE, ENABLED_TIMEFRAMES

def init_data_tables():
    """Initialize database tables for all enabled timeframes and subseries data."""
    conn = sqlite3.connect(DB_FILE)
    
    # Abilita WAL mode per prestazioni superiori
    conn.execute('PRAGMA journal_mode = WAL')
    
    cursor = conn.cursor()
    
    # Create subseries categories table
    try:
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS subseries_categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_id TEXT NOT NULL,
            pattern TEXT NOT NULL,
            description TEXT,
            count INTEGER DEFAULT 0,
            UNIQUE(category_id)
        )
        ''')
        
        # Create index on category_id for faster lookups
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_subseries_categories_id ON subseries_categories (category_id)")
        
    except Exception as e:
        logging.error(f"Error creating subseries_categories table: {e}")

    # Create subseries occurrences table
    try:
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS subseries_occurrences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            start_timestamp TEXT NOT NULL,
            end_timestamp TEXT NOT NULL,
            FOREIGN KEY (category_id) REFERENCES subseries_categories(category_id),
            UNIQUE(symbol, timeframe, start_timestamp)
        )
        ''')
        
        # Create indexes for faster lookups
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_occurrences_category_id ON subseries_occurrences (category_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_occurrences_symbol ON subseries_occurrences (symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_occurrences_timeframe ON subseries_occurrences (timeframe)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_occurrences_timestamps ON subseries_occurrences (start_timestamp, end_timestamp)")
        
    except Exception as e:
        logging.error(f"Error creating subseries_occurrences table: {e}")

    # Create category transitions table
    try:
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS category_transitions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            from_category TEXT NOT NULL,
            to_category TEXT NOT NULL,
            count INTEGER DEFAULT 0,
            probability REAL DEFAULT 0,
            UNIQUE(from_category, to_category),
            FOREIGN KEY (from_category) REFERENCES subseries_categories(category_id),
            FOREIGN KEY (to_category) REFERENCES subseries_categories(category_id)
        )
        ''')
        
        # Create indexes for faster lookups
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_transitions_from ON category_transitions (from_category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_transitions_to ON category_transitions (to_category)")
        
    except Exception as e:
        logging.error(f"Error creating category_transitions table: {e}")
    
    # Create OHLCV data tables for each timeframe
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
                close_volatility REAL,
                open_volatility REAL,
                high_volatility REAL,
                low_volatility REAL,
                volume_change REAL,
                historical_volatility REAL,
                PRIMARY KEY (timestamp, symbol)
            )
            ''')
            
            # Create index on symbol for faster lookups by symbol
            cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_symbol ON {table_name} (symbol)")
            
            # Create index on timestamp for faster lookups by time
            cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_timestamp ON {table_name} (timestamp)")
            
            # Create a composite index on symbol and timestamp for frequent joined lookups
            cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_symbol_timestamp ON {table_name} (symbol, timestamp)")
            
        except Exception as e:
            logging.error(f"Error creating table {table_name}: {e}")
    
    conn.commit()
    conn.close()

def save_data(symbol, df, timeframe):
    """
    Save pandas DataFrame to SQLite database for a specific symbol and timeframe.
    Only saves columns that exist in the database table.
    """
    if df.empty:
        logging.warning(f"No data to save for {symbol} ({timeframe})")
        return False
    
    # Abilita WAL mode per prestazioni superiori
    conn = sqlite3.connect(DB_FILE)
    conn.execute('PRAGMA journal_mode = WAL')
    
    # Get the list of columns that exist in the table
    table_name = f"data_{timeframe}"
    cursor = conn.cursor()
    
    try:
        cursor.execute(f"PRAGMA table_info({table_name})")
        existing_columns = [row[1] for row in cursor.fetchall()]
        
        # Reset index to make timestamp a regular column
        df_copy = df.reset_index()
        
        # Convert timestamp to ISO format string for better human readability
        df_copy['timestamp'] = df_copy['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S')
        
        # Add symbol column
        df_copy['symbol'] = symbol
        
        # Filter DataFrame to include only columns that exist in the table
        df_columns = set(df_copy.columns)
        valid_columns = list(df_columns.intersection(existing_columns))
        
        if len(valid_columns) < 2:
            logging.error(f"Not enough valid columns to save for {symbol} ({timeframe})")
            conn.close()
            return False
        
        # Select only the existing columns
        df_filtered = df_copy[valid_columns]
        
        # Convert filtered DataFrame to list of tuples for efficient insertion
        records = df_filtered.to_records(index=False)
        
        # Create string of column names dynamically based on filtered DataFrame columns
        columns = ', '.join(valid_columns)
        placeholders = ', '.join(['?'] * len(valid_columns))
        
        # Use INSERT OR REPLACE to handle existing records
        insert_query = f'''
        INSERT OR REPLACE INTO {table_name} ({columns})
        VALUES ({placeholders})
        '''
        
        # Execute insert for all records
        cursor.executemany(insert_query, list(records))
        
        conn.commit()
        logging.debug(f"Saved {len(df_filtered)} rows for {symbol} ({timeframe}) with {len(valid_columns)} columns")
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
    
    The freshness threshold is based on the timeframe:
    - For timeframes <= 1h: data should be updated every 15 minutes
    - For timeframes > 1h: data can be updated less frequently (few hours)
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
        
        # Determine freshness threshold based on timeframe
        # For small timeframes (5m, 15m, 30m, 1h), use minutes
        # For larger timeframes (4h, 1d), use hours or days
        now = datetime.utcnow()
        age_hours = (now - last_timestamp).total_seconds() / 3600  # Age in hours
        
        # Adjust threshold based on timeframe
        if timeframe in ['5m', '15m', '30m']:
            # For small timeframes, require data to be less than 15 minutes old
            is_fresh = age_hours <= 0.25  # 15 minutes = 0.25 hours
        elif timeframe in ['1h']:
            # For hourly data, require data to be less than 1 hour old
            is_fresh = age_hours <= 1.0
        elif timeframe in ['4h']:
            # For 4h data, require data to be less than 4 hours old
            is_fresh = age_hours <= 4.0
        else:
            # For any other timeframe, use the default max_age_days parameter
            is_fresh = age_hours <= (max_age_days * 24)
        
        # Log the freshness check result
        if not is_fresh:
            logging.info(f"Data for {symbol} ({timeframe}) is stale: last update {age_hours:.1f} hours ago")
        
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


# ===== Subseries Category Management Functions =====

def save_category(category_id, pattern, description=None):
    """
    Save a subseries category to the database.
    
    Args:
        category_id: Binary string representing the category (e.g., "01101")
        pattern: Human-readable pattern (e.g., "↓↑↑↓↑")
        description: Optional description of the pattern
        
    Returns:
        True if successful, False otherwise
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    try:
        # Check if category already exists
        cursor.execute("SELECT count FROM subseries_categories WHERE category_id = ?", (category_id,))
        result = cursor.fetchone()
        
        if result:
            # Category exists, update count
            current_count = result[0]
            cursor.execute("""
                UPDATE subseries_categories 
                SET count = ?, description = COALESCE(?, description)
                WHERE category_id = ?
            """, (current_count + 1, description, category_id))
        else:
            # New category, insert
            cursor.execute("""
                INSERT INTO subseries_categories (category_id, pattern, description, count)
                VALUES (?, ?, ?, 1)
            """, (category_id, pattern, description))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        conn.rollback()
        logging.error(f"Error saving category {category_id}: {e}")
        conn.close()
        return False

def save_subseries_occurrence(category_id, symbol, timeframe, start_timestamp, end_timestamp):
    """
    Save a subseries occurrence to the database.
    
    Args:
        category_id: The category ID of the subseries
        symbol: The cryptocurrency symbol
        timeframe: The timeframe of the data
        start_timestamp: Start timestamp of the subseries
        end_timestamp: End timestamp of the subseries
        
    Returns:
        True if successful, False otherwise
    """
    # Make sure we have timestamps as strings
    if isinstance(start_timestamp, datetime):
        start_timestamp = start_timestamp.strftime('%Y-%m-%dT%H:%M:%S')
    if isinstance(end_timestamp, datetime):
        end_timestamp = end_timestamp.strftime('%Y-%m-%dT%H:%M:%S')
    
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT OR REPLACE INTO subseries_occurrences 
            (category_id, symbol, timeframe, start_timestamp, end_timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (category_id, symbol, timeframe, start_timestamp, end_timestamp))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        conn.rollback()
        logging.error(f"Error saving subseries occurrence: {e}")
        conn.close()
        return False

def update_category_transition(from_category, to_category):
    """
    Update the transition count between two categories and recalculate probability.
    
    Args:
        from_category: The starting category ID
        to_category: The ending category ID
        
    Returns:
        True if successful, False otherwise
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    try:
        # Check if transition already exists
        cursor.execute("""
            SELECT count FROM category_transitions 
            WHERE from_category = ? AND to_category = ?
        """, (from_category, to_category))
        
        result = cursor.fetchone()
        
        if result:
            # Transition exists, update count
            current_count = result[0]
            cursor.execute("""
                UPDATE category_transitions 
                SET count = ?
                WHERE from_category = ? AND to_category = ?
            """, (current_count + 1, from_category, to_category))
        else:
            # New transition, insert
            cursor.execute("""
                INSERT INTO category_transitions (from_category, to_category, count)
                VALUES (?, ?, 1)
            """, (from_category, to_category))
        
        # Recalculate probabilities for all transitions from this category
        cursor.execute("""
            SELECT SUM(count) FROM category_transitions WHERE from_category = ?
        """, (from_category,))
        
        total_count = cursor.fetchone()[0]
        
        if total_count:
            cursor.execute("""
                UPDATE category_transitions
                SET probability = CAST(count AS REAL) / ?
                WHERE from_category = ?
            """, (total_count, from_category))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        conn.rollback()
        logging.error(f"Error updating category transition: {e}")
        conn.close()
        return False

def get_category_transitions(from_category=None):
    """
    Get transition probabilities for categories.
    
    Args:
        from_category: Optional category ID to get transitions from.
                      If None, get all transitions.
        
    Returns:
        Dictionary mapping from_category to {to_category: probability}
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    try:
        if from_category:
            cursor.execute("""
                SELECT to_category, probability
                FROM category_transitions
                WHERE from_category = ?
                ORDER BY probability DESC
            """, (from_category,))
            
            results = cursor.fetchall()
            conn.close()
            
            # Return as dictionary {to_category: probability}
            return {to_category: prob for to_category, prob in results}
        else:
            cursor.execute("""
                SELECT from_category, to_category, probability
                FROM category_transitions
                ORDER BY from_category, probability DESC
            """)
            
            results = cursor.fetchall()
            conn.close()
            
            # Return as nested dictionary {from_category: {to_category: probability}}
            transitions = {}
            for from_cat, to_cat, prob in results:
                if from_cat not in transitions:
                    transitions[from_cat] = {}
                transitions[from_cat][to_cat] = prob
            
            return transitions
    except Exception as e:
        logging.error(f"Error getting category transitions: {e}")
        conn.close()
        return {} if from_category else {}

def get_top_categories(limit=10):
    """
    Get the most common subseries categories.
    
    Args:
        limit: Maximum number of categories to return
        
    Returns:
        List of tuples (category_id, pattern, count)
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT category_id, pattern, count
            FROM subseries_categories
            ORDER BY count DESC
            LIMIT ?
        """, (limit,))
        
        results = cursor.fetchall()
        conn.close()
        return results
    except Exception as e:
        logging.error(f"Error getting top categories: {e}")
        conn.close()
        return []

def save_subseries_data(symbol, timeframe, categorized_subseries):
    """
    Save subseries data to the database, including categories, occurrences, and transitions.
    
    Args:
        symbol: The cryptocurrency symbol
        timeframe: The timeframe of the data
        categorized_subseries: Dictionary mapping category IDs to lists of subseries DataFrames
        
    Returns:
        Number of subseries saved
    """
    from src.data.subseries_utils import binary_to_pattern
    
    count = 0
    prev_category = None
    
    for category_id, subseries_list in categorized_subseries.items():
        # Save category
        pattern = binary_to_pattern(category_id)
        save_category(category_id, pattern)
        
        # Save occurrences and transitions
        for subseries in subseries_list:
            start_time = subseries.index.min()
            end_time = subseries.index.max()
            
            save_subseries_occurrence(category_id, symbol, timeframe, start_time, end_time)
            count += 1
            
            # If we have a previous category, update transition
            if prev_category:
                update_category_transition(prev_category, category_id)
            
            prev_category = category_id
    
    return count
