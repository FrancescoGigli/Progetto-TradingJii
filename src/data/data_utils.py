# data_utils.py

import logging
import pandas as pd
import numpy as np
import ta
import math
from src.data.volatility_utils import calculate_volatility_rate, calculate_historical_volatility

def add_technical_indicators(df):
    """
    Add technical indicators to OHLCV data.
    
    Args:
        df: DataFrame with OHLCV data, timestamp as index
        
    Returns:
        DataFrame with added technical indicators
    """
    if df.empty:
        logging.warning("Empty DataFrame provided to add_technical_indicators")
        return df
    
    # Log initial DataFrame info
    logging.debug(f"Adding technical indicators to DataFrame with shape {df.shape}")
    
    try:
        # Make a copy to avoid modifying the original DataFrame
        df_copy = df.copy()
        
        # Ensure we have numeric data
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df_copy.columns:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
        
        # Fill any NaN values that might have been introduced
        df_copy.ffill(inplace=True)
        
        # ===== Trend Indicators =====
        
        # EMAs
        df_copy['ema5'] = ta.trend.ema_indicator(df_copy['close'], window=5)
        df_copy['ema10'] = ta.trend.ema_indicator(df_copy['close'], window=10)
        df_copy['ema20'] = ta.trend.ema_indicator(df_copy['close'], window=20)
        
        # MACD
        macd = ta.trend.MACD(df_copy['close'], window_fast=12, window_slow=26, window_sign=9)
        df_copy['macd'] = macd.macd()
        df_copy['macd_signal'] = macd.macd_signal()
        df_copy['macd_histogram'] = macd.macd_diff()
        
        # ===== Momentum Indicators =====
        
        # RSI
        df_copy['rsi_fast'] = ta.momentum.rsi(df_copy['close'], window=14)
        
        # Stochastic RSI
        stoch_rsi = ta.momentum.StochRSIIndicator(df_copy['close'], window=14, smooth1=3, smooth2=3)
        df_copy['stoch_rsi'] = stoch_rsi.stochrsi_k()
        
        # ===== Volatility Indicators =====
        
        # ATR
        df_copy['atr'] = ta.volatility.average_true_range(df_copy['high'], df_copy['low'], df_copy['close'], window=14)
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df_copy['close'], window=20, window_dev=2)
        df_copy['bollinger_hband'] = bollinger.bollinger_hband()
        df_copy['bollinger_lband'] = bollinger.bollinger_lband()
        df_copy['bollinger_pband'] = bollinger.bollinger_pband()
        
        # Calculate volatility rates (percent change)
        df_copy = calculate_volatility_rate(df_copy)
        
        # Calculate historical volatility (standard deviation of log returns)
        df_copy = calculate_historical_volatility(df_copy, window=20)
        
        # ===== Volume Indicators =====
        
        # VWAP (Volume Weighted Average Price)
        # Reset index to have timestamp as a column
        df_reset = df_copy.reset_index()
        # Add integer period column
        df_reset['period'] = (df_reset['timestamp'].dt.dayofyear * 24 + df_reset['timestamp'].dt.hour) // 24
        
        # Group by the period and calculate VWAP
        vwap_values = []
        
        # Get unique periods
        unique_periods = df_reset['period'].unique()
        
        for period in unique_periods:
            period_data = df_reset[df_reset['period'] == period]
            
            # Calculate VWAP for this period
            cum_vol = period_data['volume'].cumsum()
            cum_vol_price = (period_data['volume'] * period_data['close']).cumsum()
            period_vwap = cum_vol_price / cum_vol
            
            # Combine with the period data
            period_vwap_df = pd.DataFrame({
                'timestamp': period_data['timestamp'],
                'vwap': period_vwap
            })
            
            vwap_values.append(period_vwap_df)
        
        # Combine all VWAP values
        if vwap_values:
            all_vwap = pd.concat(vwap_values)
            # Sort by timestamp and set as index
            all_vwap.sort_values('timestamp', inplace=True)
            all_vwap.set_index('timestamp', inplace=True)
            
            # Merge VWAP values back to the original DataFrame
            df_copy['vwap'] = all_vwap['vwap']
        else:
            df_copy['vwap'] = np.nan
        
        # ===== Additional Indicators =====
        
        # ADX (Average Directional Index)
        df_copy['adx'] = ta.trend.adx(df_copy['high'], df_copy['low'], df_copy['close'], window=14)
        
        # ROC (Rate of Change)
        df_copy['roc'] = ta.momentum.roc(df_copy['close'], window=12)
        
        # Log Return
        df_copy['log_return'] = np.log(df_copy['close'] / df_copy['close'].shift(1))
        
        # Ichimoku Cloud
        ichimoku = ta.trend.IchimokuIndicator(df_copy['high'], df_copy['low'], window1=9, window2=26, window3=52)
        df_copy['tenkan_sen'] = ichimoku.ichimoku_conversion_line()
        df_copy['kijun_sen'] = ichimoku.ichimoku_base_line()
        df_copy['senkou_span_a'] = ichimoku.ichimoku_a()
        df_copy['senkou_span_b'] = ichimoku.ichimoku_b()
        df_copy['chikou_span'] = df_copy['close'].shift(-26)
        
        # Williams %R - correzione parametro (usa lbp invece di window)
        df_copy['williams_r'] = ta.momentum.williams_r(df_copy['high'], df_copy['low'], df_copy['close'], lbp=14)
        
        # OBV (On-Balance Volume)
        df_copy['obv'] = ta.volume.on_balance_volume(df_copy['close'], df_copy['volume'])
        
        # SMA Fast and Slow (for crossovers)
        df_copy['sma_fast'] = ta.trend.sma_indicator(df_copy['close'], window=50)
        df_copy['sma_slow'] = ta.trend.sma_indicator(df_copy['close'], window=200)
        
        # SMA Trends (1 for uptrend, -1 for downtrend, 0 for no trend)
        df_copy['sma_fast_trend'] = np.where(df_copy['sma_fast'] > df_copy['sma_fast'].shift(1), 1, 
                                       np.where(df_copy['sma_fast'] < df_copy['sma_fast'].shift(1), -1, 0))
        df_copy['sma_slow_trend'] = np.where(df_copy['sma_slow'] > df_copy['sma_slow'].shift(1), 1,
                                       np.where(df_copy['sma_slow'] < df_copy['sma_slow'].shift(1), -1, 0))
        
        # SMA Cross (1 for golden cross, -1 for death cross, 0 for no cross)
        df_copy['sma_cross'] = np.where((df_copy['sma_fast'] > df_copy['sma_slow']) & 
                                  (df_copy['sma_fast'].shift(1) <= df_copy['sma_slow'].shift(1)), 1,
                                  np.where((df_copy['sma_fast'] < df_copy['sma_slow']) & 
                                    (df_copy['sma_fast'].shift(1) >= df_copy['sma_slow'].shift(1)), -1, 0))
        
        # Lagged features (t-1)
        df_copy['close_lag_1'] = df_copy['close'].shift(1)
        df_copy['volume_lag_1'] = df_copy['volume'].shift(1)
        
        # Cyclical features using sin and cos transformations
        # Weekday (0=Monday, 6=Sunday)
        weekday = df_copy.index.weekday
        # Convert to radians and extract sin/cos components
        df_copy['weekday_sin'] = np.sin(weekday * (2 * np.pi / 7))
        df_copy['weekday_cos'] = np.cos(weekday * (2 * np.pi / 7))
        
        # Hour of day
        hour = df_copy.index.hour
        df_copy['hour_sin'] = np.sin(hour * (2 * np.pi / 24))
        df_copy['hour_cos'] = np.cos(hour * (2 * np.pi / 24))
        
        # MFI (Money Flow Index)
        df_copy['mfi'] = ta.volume.money_flow_index(df_copy['high'], df_copy['low'], 
                                        df_copy['close'], df_copy['volume'], window=14)
        
        # CCI (Commodity Channel Index)
        df_copy['cci'] = ta.trend.cci(df_copy['high'], df_copy['low'], df_copy['close'], window=20)
        
        # Fill any remaining NaN values with forward fill, backward fill, and then 0
        df_copy.ffill(inplace=True)
        df_copy.bfill(inplace=True)
        df_copy.fillna(0, inplace=True)
        
        # Log the number of NaN values in the final DataFrame
        nan_count = df_copy.isna().sum().sum()
        if nan_count > 0:
            logging.warning(f"Still have {nan_count} NaN values in the DataFrame after filling")
        
        return df_copy
        
    except Exception as e:
        logging.error(f"Error in add_technical_indicators: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return df  # Return original DataFrame on error
