# volatility_utils.py

import pandas as pd
import numpy as np
import logging

def calculate_volatility_rate(df, window=20):
    """
    Calculate volatility rate from OHLCV data.
    Include historical volatility calculation in the same function
    to reduce DataFrame scanning.
    
    Formula: Vt = (Pt/Pt-1 - 1) * 100
    
    Args:
        df: DataFrame with OHLCV data, timestamp as index
        window: Window size for historical volatility calculation
        
    Returns:
        DataFrame with added volatility rates
    """
    if df.empty:
        logging.warning("Empty DataFrame provided to calculate_volatility_rate")
        return df
    
    df_vol = df.copy()
    
    # Calculate volatility rates for open, high, low, close
    for col in ['open', 'high', 'low', 'close']:
        if col in df_vol.columns:
            # Calculate percent change
            df_vol[f'{col}_volatility'] = (df_vol[col] / df_vol[col].shift(1) - 1) * 100
    
    # Calculate volume change
    if 'volume' in df_vol.columns:
        df_vol['volume_change'] = (df_vol['volume'] / df_vol['volume'].shift(1) - 1) * 100
    
    # Calculate historical volatility (integrato qui per efficienza)
    if 'close' in df_vol.columns:
        # Calcola i log return
        df_vol['log_return'] = np.log(df_vol['close'] / df_vol['close'].shift(1))
        
        # Calcola la volatilità storica
        df_vol['historical_volatility'] = df_vol['log_return'].rolling(window=window).std() * np.sqrt(252)
        
        # Rimuove la colonna temporanea
        df_vol.drop('log_return', axis=1, inplace=True, errors='ignore')
    
    return df_vol

def convert_to_volatility_series(df):
    """
    Convert a price series to a volatility series.
    
    Args:
        df: DataFrame with OHLCV data, timestamp as index
        
    Returns:
        DataFrame with only volatility rates (without original prices)
    """
    if df.empty:
        logging.warning("Empty DataFrame provided to convert_to_volatility_series")
        return df
    
    df_vol = calculate_volatility_rate(df)
    
    # Drop the original columns, keeping only volatility rates
    price_cols = ['open', 'high', 'low', 'close', 'volume']
    vol_cols = [col for col in df_vol.columns if 
                col.endswith('_volatility') or 
                col.endswith('_change') or
                col not in price_cols]
    
    # Make sure we have at least some columns
    if not vol_cols:
        logging.warning("No volatility columns were created")
        return pd.DataFrame()
    
    # Drop the first row which has NaN volatility values
    return df_vol[vol_cols].dropna()

def calculate_historical_volatility(df, window=20):
    """
    Calculate historical volatility based on close price.
    
    Args:
        df: DataFrame with close prices, timestamp as index
        window: The lookback period for volatility calculation
        
    Returns:
        DataFrame with added historical volatility
    """
    if df.empty or 'close' not in df.columns:
        logging.warning("DataFrame missing close column or empty")
        return df
    
    df_vol = df.copy()
    
    # Calculate log returns
    df_vol['log_return'] = np.log(df_vol['close'] / df_vol['close'].shift(1))
    
    # Calculate historical volatility (standard deviation of log returns)
    df_vol['historical_volatility'] = df_vol['log_return'].rolling(window=window).std() * np.sqrt(252)
    
    return df_vol
