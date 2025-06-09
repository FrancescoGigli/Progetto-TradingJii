"""
EMA Crossover Strategy

This strategy implements a classic moving average crossover approach using 
Exponential Moving Averages (EMA). It generates signals when a faster EMA 
crosses above or below a slower EMA.

Strategy Rules:
- LONG Entry: EMA20 crosses above EMA50 (Golden Cross)
- SHORT Entry: EMA20 crosses below EMA50 (Death Cross)

The strategy identifies trend changes by detecting when the faster moving average
(more responsive to recent price changes) crosses the slower moving average.
"""

import pandas as pd
import numpy as np


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate trading signals based on EMA crossover strategy.
    
    Args:
        df: DataFrame containing OHLCV data and technical indicators.
            Required columns: 'ema20', 'ema50'
    
    Returns:
        DataFrame with an additional 'signal' column containing:
        - 1: Long entry signal
        - -1: Short entry signal
        - 0: No signal
    
    Raises:
        KeyError: If required columns are missing from the DataFrame
    """
    # Create a copy to avoid modifying the original DataFrame
    result_df = df.copy()
    
    # Verify required columns exist
    required_columns = ['ema20', 'ema50']
    missing_columns = [col for col in required_columns if col not in result_df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")
    
    # Initialize signal column with zeros
    result_df['signal'] = 0
    
    # Get EMA values
    ema20 = result_df['ema20']
    ema50 = result_df['ema50']
    
    # Calculate previous values for crossover detection
    ema20_prev = ema20.shift(1)
    ema50_prev = ema50.shift(1)
    
    # Generate LONG signals (Golden Cross)
    # Condition: EMA20 crosses above EMA50
    # Current: EMA20 > EMA50, Previous: EMA20 <= EMA50
    long_condition = (ema20 > ema50) & (ema20_prev <= ema50_prev)
    result_df.loc[long_condition, 'signal'] = 1
    
    # Generate SHORT signals (Death Cross)
    # Condition: EMA20 crosses below EMA50
    # Current: EMA20 < EMA50, Previous: EMA20 >= EMA50
    short_condition = (ema20 < ema50) & (ema20_prev >= ema50_prev)
    result_df.loc[short_condition, 'signal'] = -1
    
    # Log signal statistics for debugging (optional)
    total_signals = result_df['signal'].abs().sum()
    long_signals = (result_df['signal'] == 1).sum()
    short_signals = (result_df['signal'] == -1).sum()
    
    # Optional: Print signal summary
    # print(f"EMA Crossover - Total signals: {total_signals} (Long: {long_signals}, Short: {short_signals})")
    
    return result_df
