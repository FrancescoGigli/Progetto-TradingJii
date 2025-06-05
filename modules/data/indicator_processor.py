#!/usr/bin/env python3
"""
Technical Indicator Processor Module for TradingJii

This module computes and manages technical indicators for cryptocurrency price data:
- Extracts OHLCV data from the SQLite database
- Computes various technical indicators (SMA, EMA, RSI, etc.)
- Stores the results in dedicated indicator tables
- Implements automatic warmup filtering for clean, NULL-free data

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
from modules.utils.config import DB_FILE, TA_PARAMS, DESIRED_ANALYSIS_DAYS, calculate_indicator_warmup_period

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
    Initialize indicator columns in the unified market data tables.
    DEPRECATED: Technical indicators are now stored in market_data_{timeframe} tables.
    
    Args:
        timeframes: List of timeframes to create tables for
    """
    logging.warning(f"{Fore.YELLOW}DEPRECATED: init_indicator_tables is deprecated. "
                   f"Technical indicators are now stored in market_data_* tables.{Style.RESET_ALL}")
    
    # No need to create separate tables anymore - the unified tables already have indicator columns
    pass

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
    table_name = f"market_data_{timeframe}"
    
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            
            # Check indicator coverage first - now using same table
            cursor.execute(f"""
                SELECT 
                    COUNT(*) as total_records,
                    SUM(CASE WHEN rsi14 IS NOT NULL THEN 1 ELSE 0 END) as records_with_indicators
                FROM {table_name} 
                WHERE symbol = ?
            """, (symbol,))
            
            result = cursor.fetchone()
            total_data = result[0] if result else 0
            existing_indicators = result[1] if result else 0
            
            if total_data == 0:
                logging.info(f"No data available for {Fore.YELLOW}{symbol}{Style.RESET_ALL} ({timeframe})")
                return pd.DataFrame()
            
            coverage_ratio = existing_indicators / total_data if total_data > 0 else 0
            
            # If coverage is very low, trigger full recalculation
            if coverage_ratio < 0.1:
                logging.info(f"🔧 Low indicator coverage ({coverage_ratio:.1%}) detected for {symbol} ({timeframe}) - loading ALL data for recalculation")
                
                # Clear existing indicators
                cursor.execute(f"""
                    UPDATE {table_name}
                    SET sma9 = NULL, sma20 = NULL, sma50 = NULL, 
                        ema20 = NULL, ema50 = NULL, ema200 = NULL,
                        rsi14 = NULL, stoch_k = NULL, stoch_d = NULL,
                        macd = NULL, macd_signal = NULL, macd_hist = NULL,
                        atr14 = NULL, bbands_upper = NULL, bbands_middle = NULL, bbands_lower = NULL,
                        obv = NULL, vwap = NULL, volume_sma20 = NULL,
                        adx14 = NULL
                    WHERE symbol = ?
                """, (symbol,))
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
                cursor.execute(f"""
                    SELECT MAX(timestamp) FROM {table_name} 
                    WHERE symbol = ? AND rsi14 IS NOT NULL
                """, (symbol,))
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
                        SELECT timestamp, open, high, low, close, volume
                        FROM {table_name}
                        WHERE symbol = ? AND rsi14 IS NULL
                        ORDER BY timestamp ASC
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
    # Use the new config function for automatic calculation
    return calculate_indicator_warmup_period()

def filter_warmup_period(df: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
    """
    Rimuove il periodo di warmup per avere solo indicatori non-null.
    Mantiene solo i dati corrispondenti al periodo di analisi desiderato.
    
    Args:
        df: DataFrame con indicatori calcolati
        symbol: Simbolo della criptovaluta  
        timeframe: Timeframe (es. '1h', '4h')
        
    Returns:
        DataFrame filtrato senza periodo di warmup
    """
    if df.empty:
        return df
    
    try:
        # Calcola il periodo di warmup necessario
        warmup_periods = calculate_indicator_warmup_period()
        
        # Se abbiamo meno dati del periodo di warmup, restituisci tutto
        if len(df) <= warmup_periods:
            logging.warning(f"Insufficient data for {symbol} ({timeframe}): {len(df)} records < {warmup_periods} warmup periods")
            return df
        
        # Mantieni solo gli ultimi DESIRED_ANALYSIS_DAYS di dati
        # Calcola quante candele corrispondono al periodo di analisi desiderato
        timeframe_multipliers = {
            '1m': 24 * 60,
            '5m': 24 * 12, 
            '15m': 24 * 4,
            '30m': 24 * 2,
            '1h': 24,
            '4h': 6,
            '1d': 1
        }
        
        multiplier = timeframe_multipliers.get(timeframe, 24)
        desired_candles = DESIRED_ANALYSIS_DAYS * multiplier
        
        # Prendi gli ultimi N candles per l'analisi (dopo il warmup)
        if len(df) > desired_candles:
            # Se abbiamo più dati del necessario, prendi solo gli ultimi desired_candles
            filtered_df = df.tail(desired_candles).copy()
            logging.info(f"📊 Filtered to latest {Fore.GREEN}{len(filtered_df)}{Style.RESET_ALL} candles for {Fore.YELLOW}{symbol}{Style.RESET_ALL} ({timeframe}) - {Fore.CYAN}100% clean data{Style.RESET_ALL}")
        else:
            # Rimuovi solo il periodo di warmup iniziale
            filtered_df = df.iloc[warmup_periods:].copy()
            clean_percentage = (len(filtered_df) / len(df)) * 100 if len(df) > 0 else 0
            logging.info(f"📊 Filtered warmup period for {Fore.YELLOW}{symbol}{Style.RESET_ALL} ({timeframe}): {Fore.GREEN}{len(filtered_df)}{Style.RESET_ALL}/{len(df)} records ({clean_percentage:.1f}% clean)")
        
        # Verifica che tutti gli indicatori abbiano valori validi
        indicator_columns = [col for col in filtered_df.columns if col != 'timestamp']
        
        if indicator_columns:
            null_counts = filtered_df[indicator_columns].isnull().sum()
            total_nulls = null_counts.sum()
            
            if total_nulls > 0:
                logging.warning(f"{Fore.YELLOW}Found {total_nulls} null values in filtered indicators for {symbol} ({timeframe}){Style.RESET_ALL}")
                # Log which indicators have nulls
                for col, null_count in null_counts[null_counts > 0].items():
                    logging.warning(f"  - {col}: {null_count} null values")
        
        return filtered_df
        
    except Exception as e:
        logging.error(f"{Fore.RED}Error filtering warmup period for {symbol} ({timeframe}): {e}{Style.RESET_ALL}")
        logging.error(traceback.format_exc())
        return df  # Return original data if filtering fails

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

def save_indicators(symbol: str, timeframe: str, indicators_df: pd.DataFrame) -> bool:
    """
    Save calculated indicators to the unified market data table.
    Applies warmup period filtering to ensure only clean data is saved.
    
    Args:
        symbol: Cryptocurrency symbol
        timeframe: Timeframe (e.g., '1h')
        indicators_df: DataFrame with calculated indicators
        
    Returns:
        Boolean indicating success
    """
    if indicators_df.empty:
        logging.warning(f"No indicators to save for {symbol} ({timeframe})")
        return False
    
    try:
        # Apply warmup period filtering to ensure clean data
        filtered_df = filter_warmup_period(indicators_df, symbol, timeframe)
        
        if filtered_df.empty:
            logging.warning(f"No data remaining after warmup filtering for {symbol} ({timeframe})")
            return False
        
        table_name = f"market_data_{timeframe}".replace('-', '_')
        
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            
            # Prepare data for UPDATE
            records_saved = 0
            for _, row in filtered_df.iterrows():
                try:
                    # UPDATE only the indicator columns in the unified table
                    cursor.execute(f"""
                        UPDATE {table_name}
                        SET 
                            sma9 = ?, sma20 = ?, sma50 = ?,
                            ema20 = ?, ema50 = ?, ema200 = ?,
                            rsi14 = ?, stoch_k = ?, stoch_d = ?,
                            macd = ?, macd_signal = ?, macd_hist = ?,
                            atr14 = ?, bbands_upper = ?, bbands_middle = ?, bbands_lower = ?,
                            obv = ?, vwap = ?, volume_sma20 = ?,
                            adx14 = ?
                        WHERE symbol = ? AND timestamp = ?
                    """, (
                        row.get('sma9'), row.get('sma20'), row.get('sma50'),
                        row.get('ema20'), row.get('ema50'), row.get('ema200'),
                        row.get('rsi14'), row.get('stoch_k'), row.get('stoch_d'),
                        row.get('macd'), row.get('macd_signal'), row.get('macd_hist'),
                        row.get('atr14'), row.get('bbands_upper'), row.get('bbands_middle'), row.get('bbands_lower'),
                        row.get('obv'), row.get('vwap'), row.get('volume_sma20'),
                        row.get('adx14'),
                        symbol, row['timestamp'].isoformat()
                    ))
                    # Check if the update affected any rows
                    if cursor.rowcount > 0:
                        records_saved += 1
                    else:
                        logging.debug(f"No matching record found for {symbol} at {row['timestamp']} - indicator update skipped")
                except Exception as e:
                    logging.error(f"Error updating indicator record for {symbol} at {row['timestamp']}: {e}")
                    continue
            
            conn.commit()
            logging.info(f"💾 Updated {Fore.GREEN}{records_saved}{Style.RESET_ALL} indicator records for {Fore.YELLOW}{symbol}{Style.RESET_ALL} ({timeframe})")
            return records_saved > 0
            
    except Exception as e:
        logging.error(f"{Fore.RED}Error saving indicators for {symbol} ({timeframe}): {e}{Style.RESET_ALL}")
        logging.error(traceback.format_exc())
        return False

def process_and_save_indicators(symbol: str, timeframe: str) -> bool:
    """
    Complete workflow to process and save technical indicators for a symbol and timeframe.
    
    Args:
        symbol: Cryptocurrency symbol
        timeframe: Timeframe (e.g., '1h')
        
    Returns:
        Boolean indicating success
    """
    try:
        logging.debug(f"🔧 Processing indicators for {Fore.YELLOW}{symbol}{Style.RESET_ALL} ({timeframe})")
        
        # Load OHLCV data
        ohlcv_df = load_ohlcv_data(symbol, timeframe)
        
        if ohlcv_df.empty:
            logging.debug(f"No OHLCV data available for {symbol} ({timeframe})")
            return False
        
        # Calculate indicators
        indicators_df = calculate_indicators(ohlcv_df)
        
        if indicators_df.empty:
            logging.warning(f"Failed to calculate indicators for {symbol} ({timeframe})")
            return False
        
        # Save indicators to database
        success = save_indicators(symbol, timeframe, indicators_df)
        
        if success:
            logging.debug(f"✅ Indicators processed successfully for {Fore.YELLOW}{symbol}{Style.RESET_ALL} ({timeframe})")
        else:
            logging.warning(f"❌ Failed to save indicators for {symbol} ({timeframe})")
        
        return success
        
    except Exception as e:
        logging.error(f"{Fore.RED}Error processing indicators for {symbol} ({timeframe}): {e}{Style.RESET_ALL}")
        logging.error(traceback.format_exc())
        return False

def compute_and_save_indicators(symbol: str, timeframe: str) -> bool:
    """
    Alias for process_and_save_indicators for backward compatibility.
    
    Args:
        symbol: Cryptocurrency symbol
        timeframe: Timeframe (e.g., '1h')
        
    Returns:
        Boolean indicating success
    """
    return process_and_save_indicators(symbol, timeframe)
