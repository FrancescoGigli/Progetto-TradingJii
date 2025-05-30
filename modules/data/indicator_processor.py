#!/usr/bin/env python3
"""
Technical Indicator Processor Module for TradingJii

This module computes and manages technical indicators for cryptocurrency price data:
- Extracts OHLCV data from the SQLite database
- Computes various technical indicators (SMA, EMA, RSI, etc.)
- Stores the results in dedicated indicator tables

Integrates with the real_time.py update flow.
"""

import sqlite3
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from colorama import Fore, Style
from datetime import datetime
import traceback
import sys
import subprocess
from modules.utils.config import DB_FILE, TA_PARAMS

# Global variables to track if we've checked for TA-Lib
talib_checked = False
pandas_ta_checked = False
has_talib = False
has_pandas_ta = False
talib = None  # Global reference to talib module

def _check_talib():
    """
    Check if TA-Lib is available, attempt installation if not.
    
    Returns:
        Boolean indicating if TA-Lib is available
    """
    global talib_checked, has_talib, pandas_ta_checked, has_pandas_ta, talib
    
    if talib_checked:
        return has_talib
    
    talib_checked = True
    
    # Try importing TA-Lib
    try:
        import talib as talib_module
        talib = talib_module
        has_talib = True
        logging.info(f"{Fore.GREEN}TA-Lib is available. Using TA-Lib for technical indicators.{Style.RESET_ALL}")
        return True
    except ImportError:
        logging.warning(f"{Fore.YELLOW}TA-Lib not found. Attempting to install...{Style.RESET_ALL}")
        
        try:
            # Attempt to install TA-Lib
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ta-lib"])
            logging.info(f"{Fore.GREEN}TA-Lib installed successfully.{Style.RESET_ALL}")
            
            # Try importing again
            import talib as talib_module
            talib = talib_module
            has_talib = True
            logging.info(f"{Fore.GREEN}TA-Lib is now available.{Style.RESET_ALL}")
            return True
        except (ImportError, subprocess.CalledProcessError) as e:
            logging.warning(f"{Fore.YELLOW}Failed to install TA-Lib: {e}.{Style.RESET_ALL}")
            logging.warning(f"{Fore.YELLOW}For Windows users: Please install from a pre-built wheel:{Style.RESET_ALL}")
            logging.warning(f"{Fore.CYAN}pip install --no-cache-dir https://download.lfd.uci.edu/pythonlibs/archived/ta_lib-0.4.24-cp310-cp310-win_amd64.whl{Style.RESET_ALL}")
            logging.warning(f"{Fore.YELLOW}or follow instructions at: https://github.com/mrjbq7/ta-lib/issues{Style.RESET_ALL}")
            logging.warning(f"{Fore.YELLOW}Will fall back to pandas-ta.{Style.RESET_ALL}")
            
            # Check for pandas-ta as fallback
            return _check_pandas_ta()

def _check_pandas_ta():
    """
    Check if pandas-ta is available, attempt installation if not.
    
    Returns:
        Boolean indicating if pandas-ta is available
    """
    global pandas_ta_checked, has_pandas_ta, pandas_ta
    
    if pandas_ta_checked:
        return has_pandas_ta
    
    pandas_ta_checked = True
    
    # Try importing pandas-ta
    try:
        import pandas_ta as pandas_ta_module
        pandas_ta = pandas_ta_module
        has_pandas_ta = True
        logging.info(f"{Fore.GREEN}Using pandas-ta for technical indicators.{Style.RESET_ALL}")
        return True
    except ImportError:
        logging.warning(f"{Fore.YELLOW}pandas-ta not found. Attempting to install...{Style.RESET_ALL}")
        
        try:
            # Attempt to install pandas-ta
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas-ta"])
            logging.info(f"{Fore.GREEN}pandas-ta installed successfully.{Style.RESET_ALL}")
            
            # Try importing again
            import pandas_ta as pandas_ta_module
            pandas_ta = pandas_ta_module
            has_pandas_ta = True
            logging.info(f"{Fore.GREEN}pandas-ta is now available.{Style.RESET_ALL}")
            return True
        except (ImportError, subprocess.CalledProcessError) as e:
            logging.error(f"{Fore.RED}Failed to install pandas-ta: {e}. Technical indicators will not be available.{Style.RESET_ALL}")
            return False

def init_indicator_tables(timeframes):
    """
    Initialize database tables for technical indicators for each timeframe.
    
    Args:
        timeframes: List of timeframes to create tables for
    """
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            tables_created = []
            
            for timeframe in timeframes:
                table_name = f"ta_{timeframe}"
                # Ensure the table name is valid - remove any problematic characters
                table_name = table_name.replace('-', '_')
                
                try:
                    logging.info(f"Creating table {table_name} if it doesn't exist...")
                    cursor.execute(f"""
                        CREATE TABLE IF NOT EXISTS {table_name} (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            symbol TEXT NOT NULL,
                            timestamp TEXT NOT NULL,
                            
                            /* Simple Moving Averages */
                            sma9 REAL,
                            sma20 REAL,
                            sma50 REAL,
                            
                            /* Exponential Moving Averages */
                            ema20 REAL,
                            ema50 REAL,
                            ema200 REAL,
                            
                            /* Momentum Indicators */
                            rsi14 REAL,
                            stoch_k REAL,
                            stoch_d REAL,
                            macd REAL,
                            macd_signal REAL,
                            macd_hist REAL,
                            
                            /* Volatility Indicators */
                            atr14 REAL,
                            bbands_upper REAL,
                            bbands_middle REAL,
                            bbands_lower REAL,
                            
                            /* Volume-based Indicators */
                            obv REAL,
                            vwap REAL,
                            volume_sma20 REAL,
                            
                            /* Trend Strength */
                            adx14 REAL,
                            
                            /* Ensure no duplicate entries */
                            UNIQUE(symbol, timestamp)
                        )
                    """)
                    cursor.execute(f"""
                        CREATE INDEX IF NOT EXISTS idx_{timeframe}_ta_symbol_timestamp
                        ON {table_name} (symbol, timestamp)
                    """)
                    tables_created.append(table_name)
                    
                    # Verify table was created successfully
                    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
                    if cursor.fetchone():
                        logging.info(f"{Fore.GREEN}Table {table_name} created or already exists{Style.RESET_ALL}")
                    else:
                        logging.error(f"{Fore.RED}Failed to verify table {table_name} after creation{Style.RESET_ALL}")
                except Exception as e:
                    logging.error(f"{Fore.RED}Error creating table {table_name}: {e}{Style.RESET_ALL}")
                    raise
            
            conn.commit()
            
    except Exception as e:
        logging.error(f"{Fore.RED}Database initialization error for indicator tables: {e}{Style.RESET_ALL}")
        logging.error(traceback.format_exc())
        raise

def load_ohlcv_data(symbol: str, timeframe: str) -> pd.DataFrame:
    """
    Load historical OHLCV data for a specific symbol and timeframe from the database.
    Automatically checks coverage and triggers full recalculation if needed.
    
    Args:
        symbol: Cryptocurrency symbol
        timeframe: Timeframe (e.g., '5m')
        
    Returns:
        DataFrame with OHLCV data
    """
    table_name = f"data_{timeframe}"
    ta_table_name = f"ta_{timeframe}"
    
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            
            # Check indicator coverage first
            cursor.execute(f"SELECT COUNT(*) FROM {ta_table_name} WHERE symbol = ?", (symbol,))
            existing_indicators = cursor.fetchone()[0]
            
            cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE symbol = ?", (symbol,))
            total_data = cursor.fetchone()[0]
            
            if total_data == 0:
                logging.info(f"No data available for {Fore.YELLOW}{symbol}{Style.RESET_ALL} ({timeframe})")
                return pd.DataFrame()
            
            coverage_ratio = existing_indicators / total_data
            
            # If coverage is very low, trigger full recalculation
            if coverage_ratio < 0.1:
                logging.info(f"🔧 Low indicator coverage ({coverage_ratio:.1%}) detected for {symbol} ({timeframe}) - loading ALL data for recalculation")
                
                # Clear existing indicators
                cursor.execute(f"DELETE FROM {ta_table_name} WHERE symbol = ?", (symbol,))
                conn.commit()
                
                # Load ALL data for complete recalculation
                query = f"""
                    SELECT timestamp, open, high, low, close, volume
                    FROM {table_name}
                    WHERE symbol = ?
                    ORDER BY timestamp ASC
                """
                df = pd.read_sql_query(query, conn, params=(symbol,))
                
                if not df.empty:
                    logging.info(f"📊 Loaded {len(df)} records for complete indicator recalculation")
                
            else:
                # Normal incremental update - only missing data with lookback
                logging.debug(f"✅ Good coverage ({coverage_ratio:.1%}) - performing incremental update")
                
                # Get the latest timestamp with indicators
                cursor.execute(f"SELECT MAX(timestamp) FROM {ta_table_name} WHERE symbol = ?", (symbol,))
                latest_indicator_ts = cursor.fetchone()[0]
                
                lookback = _get_longest_lookback_period()
                
                if latest_indicator_ts and lookback > 0:
                    # Include sufficient lookback data for proper calculation
                    if timeframe == '1h':
                        lookback_hours = lookback
                    elif timeframe == '4h':
                        lookback_hours = lookback * 4
                    elif timeframe == '1d':
                        lookback_hours = lookback * 24
                    else:
                        lookback_hours = lookback
                    
                    query = f"""
                        SELECT timestamp, open, high, low, close, volume
                        FROM {table_name}
                        WHERE symbol = ? AND timestamp >= datetime(?, '-{lookback_hours} hours')
                        ORDER BY timestamp ASC
                    """
                    df = pd.read_sql_query(query, conn, params=(symbol, latest_indicator_ts))
                else:
                    # Get all missing data
                    query = f"""
                        SELECT d.timestamp, d.open, d.high, d.low, d.close, d.volume
                        FROM {table_name} d
                        LEFT JOIN {ta_table_name} t ON d.symbol = t.symbol AND d.timestamp = t.timestamp
                        WHERE d.symbol = ? AND t.timestamp IS NULL
                        ORDER BY d.timestamp ASC
                    """
                    df = pd.read_sql_query(query, conn, params=(symbol,))
            
            if df.empty:
                logging.info(f"No new data to process for {Fore.YELLOW}{symbol}{Style.RESET_ALL} ({timeframe})")
                return pd.DataFrame()
            
            # Ensure the timestamp is properly formatted
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            logging.debug(f"Loaded {len(df)} OHLCV records for {symbol} ({timeframe})")
            return df
            
    except Exception as e:
        logging.error(f"{Fore.RED}Error loading OHLCV data for {symbol} ({timeframe}): {e}{Style.RESET_ALL}")
        logging.error(traceback.format_exc())
        return pd.DataFrame()

def _get_longest_lookback_period() -> int:
    """
    Get the longest lookback period required for any indicator.
    
    Returns:
        The maximum lookback period
    """
    # Get the custom parameters from config, or use defaults
    ta_params = getattr(sys.modules['modules.utils.config'], 'TA_PARAMS', {})
    
    # Determine the longest lookback needed based on longest period indicators
    longest_lookback = max([
        ta_params.get('sma50', {}).get('timeperiod', 50),
        ta_params.get('ema200', {}).get('timeperiod', 200),
        ta_params.get('bbands', {}).get('timeperiod', 20) + 10,  # Add buffer for convergence
        200  # Default safe value to capture most indicators
    ])
    
    return longest_lookback

def _calculate_indicators_talib(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Calculate technical indicators using TA-Lib.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        Dictionary of indicator name to Series
    """
    # Using the global talib module
    global talib
    
    # Get parameters from config or use defaults
    ta_params = getattr(sys.modules['modules.utils.config'], 'TA_PARAMS', {})
    
    # Extract data series
    open_prices = df['open'].values
    high_prices = df['high'].values
    low_prices = df['low'].values
    close_prices = df['close'].values
    volume = df['volume'].values
    
    indicators = {}
    
    try:
        # Simple Moving Averages
        sma9_params = ta_params.get('sma9', {'timeperiod': 9})
        indicators['sma9'] = talib.SMA(close_prices, **sma9_params)
        
        sma20_params = ta_params.get('sma20', {'timeperiod': 20})
        indicators['sma20'] = talib.SMA(close_prices, **sma20_params)
        
        sma50_params = ta_params.get('sma50', {'timeperiod': 50})
        indicators['sma50'] = talib.SMA(close_prices, **sma50_params)
        
        # Exponential Moving Averages
        ema20_params = ta_params.get('ema20', {'timeperiod': 20})
        indicators['ema20'] = talib.EMA(close_prices, **ema20_params)
        
        ema50_params = ta_params.get('ema50', {'timeperiod': 50})
        indicators['ema50'] = talib.EMA(close_prices, **ema50_params)
        
        ema200_params = ta_params.get('ema200', {'timeperiod': 200})
        indicators['ema200'] = talib.EMA(close_prices, **ema200_params)
        
        # Momentum Indicators
        rsi14_params = ta_params.get('rsi14', {'timeperiod': 14})
        indicators['rsi14'] = talib.RSI(close_prices, **rsi14_params)
        
        stoch_params = ta_params.get('stoch', {'fastk_period': 14, 'slowk_period': 3, 'slowd_period': 3})
        indicators['stoch_k'], indicators['stoch_d'] = talib.STOCH(
            high_prices, low_prices, close_prices,
            fastk_period=stoch_params.get('fastk_period', 14),
            slowk_period=stoch_params.get('slowk_period', 3),
            slowd_period=stoch_params.get('slowd_period', 3)
        )
        
        macd_params = ta_params.get('macd', {'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9})
        indicators['macd'], indicators['macd_signal'], indicators['macd_hist'] = talib.MACD(
            close_prices,
            fastperiod=macd_params.get('fastperiod', 12),
            slowperiod=macd_params.get('slowperiod', 26),
            signalperiod=macd_params.get('signalperiod', 9)
        )
        
        # Volatility Indicators
        atr14_params = ta_params.get('atr14', {'timeperiod': 14})
        indicators['atr14'] = talib.ATR(high_prices, low_prices, close_prices, **atr14_params)
        
        bbands_params = ta_params.get('bbands', {'timeperiod': 20, 'nbdevup': 2, 'nbdevdn': 2})
        indicators['bbands_upper'], indicators['bbands_middle'], indicators['bbands_lower'] = talib.BBANDS(
            close_prices,
            timeperiod=bbands_params.get('timeperiod', 20),
            nbdevup=bbands_params.get('nbdevup', 2),
            nbdevdn=bbands_params.get('nbdevdn', 2)
        )
        
        # Volume-based Indicators
        indicators['obv'] = talib.OBV(close_prices, volume)
        
        # Calculate VWAP (this is not in TA-Lib, so we'll calculate manually)
        typical_price = (high_prices + low_prices + close_prices) / 3
        vwap_values = np.cumsum(typical_price * volume) / np.cumsum(volume)
        indicators['vwap'] = vwap_values
        
        # Volume SMA
        vol_sma20_params = ta_params.get('volume_sma20', {'timeperiod': 20})
        indicators['volume_sma20'] = talib.SMA(volume, **vol_sma20_params)
        
        # Trend Strength
        adx14_params = ta_params.get('adx14', {'timeperiod': 14})
        indicators['adx14'] = talib.ADX(high_prices, low_prices, close_prices, **adx14_params)
        
    except Exception as e:
        logging.error(f"{Fore.RED}Error calculating indicators with TA-Lib: {e}{Style.RESET_ALL}")
        logging.error(traceback.format_exc())
    
    # Convert all numpy arrays to pandas Series
    return {k: pd.Series(v) for k, v in indicators.items()}

def _calculate_indicators_pandas_ta(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Calculate technical indicators using pandas_ta.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        Dictionary of indicator name to Series
    """
    # Using the global pandas_ta module
    global pandas_ta
    ta = pandas_ta
    
    # Create a copy of the DataFrame to ensure we don't modify the original
    data = df.copy()
    
    # Get parameters from config or use defaults
    ta_params = getattr(sys.modules['modules.utils.config'], 'TA_PARAMS', {})
    
    indicators = {}
    
    try:
        # Simple Moving Averages
        sma9_params = ta_params.get('sma9', {'length': 9})
        indicators['sma9'] = ta.sma(data['close'], **sma9_params)
        
        sma20_params = ta_params.get('sma20', {'length': 20})
        indicators['sma20'] = ta.sma(data['close'], **sma20_params)
        
        sma50_params = ta_params.get('sma50', {'length': 50})
        indicators['sma50'] = ta.sma(data['close'], **sma50_params)
        
        # Exponential Moving Averages
        ema20_params = ta_params.get('ema20', {'length': 20})
        indicators['ema20'] = ta.ema(data['close'], **ema20_params)
        
        ema50_params = ta_params.get('ema50', {'length': 50})
        indicators['ema50'] = ta.ema(data['close'], **ema50_params)
        
        ema200_params = ta_params.get('ema200', {'length': 200})
        indicators['ema200'] = ta.ema(data['close'], **ema200_params)
        
        # Momentum Indicators
        rsi14_params = ta_params.get('rsi14', {'length': 14})
        indicators['rsi14'] = ta.rsi(data['close'], **rsi14_params)
        
        stoch_params = ta_params.get('stoch', {'k': 14, 'slow_k': 3, 'd': 3})
        stoch = ta.stoch(
            data['high'], data['low'], data['close'],
            k=stoch_params.get('k', 14),
            slow_k=stoch_params.get('slow_k', 3),
            d=stoch_params.get('d', 3)
        )
        indicators['stoch_k'] = stoch['STOCHk_14_3_3']
        indicators['stoch_d'] = stoch['STOCHd_14_3_3']
        
        macd_params = ta_params.get('macd', {'fast': 12, 'slow': 26, 'signal': 9})
        macd = ta.macd(
            data['close'],
            fast=macd_params.get('fast', 12),
            slow=macd_params.get('slow', 26),
            signal=macd_params.get('signal', 9)
        )
        indicators['macd'] = macd[f"MACD_{macd_params.get('fast', 12)}_{macd_params.get('slow', 26)}_{macd_params.get('signal', 9)}"]
        indicators['macd_signal'] = macd[f"MACDs_{macd_params.get('fast', 12)}_{macd_params.get('slow', 26)}_{macd_params.get('signal', 9)}"]
        indicators['macd_hist'] = macd[f"MACDh_{macd_params.get('fast', 12)}_{macd_params.get('slow', 26)}_{macd_params.get('signal', 9)}"]
        
        # Volatility Indicators
        atr14_params = ta_params.get('atr14', {'length': 14})
        indicators['atr14'] = ta.atr(data['high'], data['low'], data['close'], **atr14_params)
        
        bbands_params = ta_params.get('bbands', {'length': 20, 'std': 2})
        bbands = ta.bbands(
            data['close'],
            length=bbands_params.get('length', 20),
            std=bbands_params.get('std', 2)
        )
        # Get the column names from the bbands DataFrame for better compatibility
        bbands_cols = bbands.columns.tolist()
        
        # Find the appropriate columns for upper, middle and lower bands
        upper_col = next((col for col in bbands_cols if col.startswith('BBU_')), None)
        middle_col = next((col for col in bbands_cols if col.startswith('BBM_')), None)
        lower_col = next((col for col in bbands_cols if col.startswith('BBL_')), None)
        
        if upper_col:
            indicators['bbands_upper'] = bbands[upper_col]
        if middle_col:
            indicators['bbands_middle'] = bbands[middle_col]
        if lower_col:
            indicators['bbands_lower'] = bbands[lower_col]
        
        # Volume-based Indicators
        indicators['obv'] = ta.obv(data['close'], data['volume'])
        
        # Calculate VWAP manually since pandas_ta implementation requires DatetimeIndex
        # This avoids the 'RangeIndex' object has no attribute 'to_period' error
        try:
            # Calculate typical price
            typical_price = (data['high'] + data['low'] + data['close']) / 3
            # Calculate VWAP
            vwap_values = (typical_price * data['volume']).cumsum() / data['volume'].cumsum()
            indicators['vwap'] = vwap_values
        except Exception as e:
            logging.error(f"{Fore.RED}Error calculating VWAP: {e}{Style.RESET_ALL}")
        
        # Volume SMA
        vol_sma20_params = ta_params.get('volume_sma20', {'length': 20})
        indicators['volume_sma20'] = ta.sma(data['volume'], **vol_sma20_params)
        
        # Trend Strength
        adx14_params = ta_params.get('adx14', {'length': 14})
        adx = ta.adx(data['high'], data['low'], data['close'], **adx14_params)
        indicators['adx14'] = adx[f"ADX_{adx14_params.get('length', 14)}"]
        
    except Exception as e:
        logging.error(f"{Fore.RED}Error calculating indicators with pandas_ta: {e}{Style.RESET_ALL}")
        logging.error(traceback.format_exc())
    
    return indicators

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all technical indicators for the given OHLCV data.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with original data and all indicators
    """
    if df.empty:
        return pd.DataFrame()
    
    try:
        # Initialize results DataFrame with timestamp from input
        result_df = pd.DataFrame({'timestamp': df['timestamp']})
        
        # Calculate indicators based on available libraries
        indicators = {}
        if _check_talib():
            indicators = _calculate_indicators_talib(df)
        elif _check_pandas_ta():
            indicators = _calculate_indicators_pandas_ta(df)
        else:
            logging.error(f"{Fore.RED}No technical analysis library available. Cannot calculate indicators.{Style.RESET_ALL}")
            return pd.DataFrame()
        
        # Add each indicator to the results DataFrame
        for name, series in indicators.items():
            result_df[name] = series
        
        return result_df
    except Exception as e:
        logging.error(f"{Fore.RED}Error in calculate_indicators: {e}{Style.RESET_ALL}")
        logging.error(traceback.format_exc())
        return pd.DataFrame()

def _check_indicator_coverage(symbol: str, timeframe: str) -> bool:
    """
    Check if indicator coverage is adequate and trigger full recalculation if needed.
    
    Args:
        symbol: Cryptocurrency symbol
        timeframe: Timeframe (e.g., '1h')
        
    Returns:
        Boolean indicating if full recalculation is needed
    """
    table_name = f"data_{timeframe}"
    ta_table_name = f"ta_{timeframe}"
    
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            
            # Get counts
            cursor.execute(f"SELECT COUNT(*) FROM {ta_table_name} WHERE symbol = ?", (symbol,))
            existing_indicators = cursor.fetchone()[0]
            
            cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE symbol = ?", (symbol,))
            total_data = cursor.fetchone()[0]
            
            if total_data == 0:
                return False  # No data, no recalculation needed
            
            coverage_ratio = existing_indicators / total_data
            
            # If coverage is less than 10%, trigger full recalculation
            if coverage_ratio < 0.1:
                logging.info(f"Low indicator coverage ({coverage_ratio:.1%}) detected for {symbol} ({timeframe}) - triggering full recalculation")
                
                # Clear existing indicators for clean recalculation
                cursor.execute(f"DELETE FROM {ta_table_name} WHERE symbol = ?", (symbol,))
                conn.commit()
                
                return True
            
            return False
            
    except Exception as e:
        logging.error(f"{Fore.RED}Error checking indicator coverage for {symbol} ({timeframe}): {e}{Style.RESET_ALL}")
        return False

def save_indicators(symbol: str, timeframe: str, df: pd.DataFrame) -> bool:
    """
    Save calculated indicators to the database.
    
    Args:
        symbol: Cryptocurrency symbol
        timeframe: Timeframe (e.g., '5m')
        df: DataFrame with calculated indicators
        
    Returns:
        Boolean indicating success
    """
    if df.empty:
        return False
    
    table_name = f"ta_{timeframe}"
    
    try:
        with sqlite3.connect(DB_FILE) as conn:
            # Format timestamp for SQLite
            df_to_save = df.copy()
            df_to_save['symbol'] = symbol
            df_to_save['timestamp'] = df_to_save['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S')
            
            # Create SQL placeholders for the columns
            columns = df_to_save.columns
            placeholders = ', '.join(['?' for _ in columns])
            columns_str = ', '.join(columns)
            
            # Insert data using INSERT OR REPLACE
            cursor = conn.cursor()
            sql = f"INSERT OR REPLACE INTO {table_name} ({columns_str}) VALUES ({placeholders})"
            
            # Convert DataFrame to list of tuples
            records = [tuple(x) for x in df_to_save.values]
            cursor.executemany(sql, records)
            
            conn.commit()
            logging.info(f"Saved {Fore.GREEN}{len(records)}{Style.RESET_ALL} indicator records for {Fore.YELLOW}{symbol}{Style.RESET_ALL} ({timeframe})")
            return True
    except Exception as e:
        logging.error(f"{Fore.RED}Error saving indicators for {symbol} ({timeframe}): {e}{Style.RESET_ALL}")
        logging.error(traceback.format_exc())
        return False

async def compute_and_save_indicators(symbol: str, timeframe: str) -> bool:
    """
    Compute and save indicators for a specific symbol and timeframe.
    
    Args:
        symbol: Cryptocurrency symbol
        timeframe: Timeframe (e.g., '5m')
        
    Returns:
        Boolean indicating success
    """
    try:
        # Skip if no TA libraries are available
        if not (_check_talib() or _check_pandas_ta()):
            logging.error(f"{Fore.RED}No technical analysis libraries available. Skipping indicator calculation.{Style.RESET_ALL}")
            return False
        
        logging.info(f"Computing indicators for {Fore.YELLOW}{symbol}{Style.RESET_ALL} ({timeframe})")
        
        # 1. Check indicator coverage and determine if full recalculation is needed
        needs_full_recalc = _check_indicator_coverage(symbol, timeframe)
        
        # 2. Load OHLCV data from database (now coverage-aware)
        ohlcv_df = load_ohlcv_data(symbol, timeframe)
        
        if ohlcv_df.empty:
            if not needs_full_recalc:
                logging.info(f"No new data for {symbol} ({timeframe}) that needs indicators")
                return True  # Nothing to do, but not an error
            else:
                logging.warning(f"Low coverage detected but no data available for {symbol} ({timeframe})")
                return False
        
        update_type = "FULL RECALC" if needs_full_recalc else "INCREMENTAL"
        logging.info(f"Processing {Fore.GREEN}{len(ohlcv_df)}{Style.RESET_ALL} ({update_type}) candles for {Fore.YELLOW}{symbol}{Style.RESET_ALL}")
        
        # 2. Calculate indicators
        indicators_df = calculate_indicators(ohlcv_df)
        
        if indicators_df.empty:
            logging.warning(f"No indicators calculated for {symbol} ({timeframe})")
            return False
        
        # Validate indicators
        non_null_percentage = _validate_indicators(indicators_df)
        
        # 3. Save indicators to database
        success = save_indicators(symbol, timeframe, indicators_df)
        
        if success:
            logging.info(f"Successfully saved indicators for {Fore.YELLOW}{symbol}{Style.RESET_ALL} ({timeframe})")
            logging.info(f"Indicators non-null percentage: {Fore.CYAN}{non_null_percentage:.1f}%{Style.RESET_ALL}")
            return True
        else:
            return False
    
    except Exception as e:
        logging.error(f"{Fore.RED}Error in indicator processing pipeline for {symbol} ({timeframe}): {e}{Style.RESET_ALL}")
        logging.error(traceback.format_exc())
        return False

def _validate_indicators(df: pd.DataFrame) -> float:
    """
    Validate calculated indicators and log any issues.
    
    Args:
        df: DataFrame with calculated indicators
        
    Returns:
        Percentage of non-null values across all indicators
    """
    if df.empty:
        return 0.0
    
    try:
        # Skip the timestamp column when calculating null percentage
        indicator_columns = [col for col in df.columns if col != 'timestamp']
        
        if not indicator_columns:
            return 0.0
        
        # Count non-null values
        total_values = len(df) * len(indicator_columns)
        non_null_values = df[indicator_columns].count().sum()
        
        if total_values == 0:
            return 0.0
        
        non_null_percentage = (non_null_values / total_values) * 100
        
        # Check for values outside expected ranges
        range_checks = {
            'rsi14': (0, 100),
            'stoch_k': (0, 100),
            'stoch_d': (0, 100),
            'adx14': (0, 100)
        }
        
        for indicator, (min_val, max_val) in range_checks.items():
            if indicator in df:
                out_of_range = df[(df[indicator] < min_val) | (df[indicator] > max_val)].shape[0]
                if out_of_range > 0:
                    logging.warning(f"{Fore.YELLOW}Found {out_of_range} values outside range [{min_val}, {max_val}] for {indicator}{Style.RESET_ALL}")
        
        return non_null_percentage
    
    except Exception as e:
        logging.error(f"{Fore.RED}Error validating indicators: {e}{Style.RESET_ALL}")
        return 0.0
