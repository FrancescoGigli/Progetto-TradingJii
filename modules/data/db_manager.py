#!/usr/bin/env python3
"""
Database management module for TradingJii

Handles database initialization, queries, and data management.
"""

import sqlite3
from datetime import datetime
import logging
from colorama import Fore, Style
from modules.utils.config import DB_FILE, TIMEFRAME_CONFIG

def init_data_tables(timeframes):
    """
    Initialize database tables for each timeframe.
    
    Args:
        timeframes: List of timeframes to create tables for
    """
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        for timeframe in timeframes:
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS data_{timeframe} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    UNIQUE(symbol, timestamp)
                )
            """)
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{timeframe}_symbol_timestamp
                ON data_{timeframe} (symbol, timestamp)
            """)

def get_timestamp_range(symbol, timeframe):
    """
    Get the first and last timestamp available for a symbol.
    
    Args:
        symbol: The cryptocurrency symbol
        timeframe: The timeframe to check
        
    Returns:
        Tuple of (first_date, last_date) as datetime objects or (None, None) if no data
    """
    table_name = f"data_{timeframe}"
    
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT MIN(timestamp), MAX(timestamp) 
            FROM {table_name}
            WHERE symbol = ?
        """, (symbol,))
        
        result = cursor.fetchone()
        if result and result[0] and result[1]:
            return (
                datetime.strptime(result[0], '%Y-%m-%dT%H:%M:%S'),
                datetime.strptime(result[1], '%Y-%m-%dT%H:%M:%S')
            )
    return None, None

def check_data_freshness(symbol, timeframe):
    """
    Check if we already have fresh data for a symbol and timeframe.
    
    Args:
        symbol: The cryptocurrency symbol
        timeframe: The timeframe to check
        
    Returns:
        Tuple of (is_fresh, last_timestamp)
    """
    now = datetime.now()
    first_date, last_date = get_timestamp_range(symbol, timeframe)

    if last_date:
        time_diff = now - last_date
        if time_diff < TIMEFRAME_CONFIG[timeframe]['max_age']:
            logging.info(f"Saltato {Fore.YELLOW}{symbol}{Style.RESET_ALL} ({timeframe}): Dati recenti già esistenti")
            logging.info(f"  • Prima data: {Fore.CYAN}{first_date.strftime('%Y-%m-%d %H:%M')}{Style.RESET_ALL}")
            logging.info(f"  • Ultima data: {Fore.CYAN}{last_date.strftime('%Y-%m-%d %H:%M')}{Style.RESET_ALL}")
            return True, last_date
            
    return False, last_date

def save_ohlcv_data(symbol, timeframe, ohlcv_data):
    """
    Save OHLCV data to the database.
    
    Args:
        symbol: The cryptocurrency symbol
        timeframe: The timeframe of the data
        ohlcv_data: List of OHLCV data from the exchange
        
    Returns:
        Tuple of (success, num_records_saved)
    """
    if not ohlcv_data:
        return False, 0
        
    table_name = f"data_{timeframe}"
    
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            
            # Convert OHLCV data to records format
            records = [
                (symbol, datetime.fromtimestamp(r[0]/1000).strftime('%Y-%m-%dT%H:%M:%S'), *r[1:])
                for r in ohlcv_data
            ]
            
            # Insert data using INSERT OR REPLACE
            cursor.executemany(f"""
                INSERT OR REPLACE INTO {table_name}
                (symbol, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, records)
            
            conn.commit()
            
            # Log data range after saving
            first_date, last_date = get_timestamp_range(symbol, timeframe)
            if first_date and last_date:
                logging.info(f"Completato {Fore.YELLOW}{symbol}{Style.RESET_ALL} ({timeframe}) - salvati {Fore.GREEN}{len(records)}{Style.RESET_ALL} record")
                logging.info(f"  • Prima data: {Fore.CYAN}{first_date.strftime('%Y-%m-%d %H:%M')}{Style.RESET_ALL}")
                logging.info(f"  • Ultima data: {Fore.CYAN}{last_date.strftime('%Y-%m-%d %H:%M')}{Style.RESET_ALL}")
            
            return True, len(records)
    except Exception as e:
        logging.error(f"Errore nel salvataggio dei dati per {symbol} ({timeframe}): {e}")
        return False, 0
