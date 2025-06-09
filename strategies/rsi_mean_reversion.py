"""
RSI Mean Reversion Strategy

This strategy implements a mean reversion approach based on the Relative Strength Index (RSI).
The core concept is that when RSI reaches extreme levels (oversold or overbought), 
the price tends to revert to the mean.

Strategy Rules:
- LONG Entry: RSI < 30 (oversold) and RSI starts to rise (RSI > RSI_previous)
- SHORT Entry: RSI > 70 (overbought) and RSI starts to fall (RSI < RSI_previous)

The strategy looks for reversals at extreme RSI levels, entering positions when
momentum starts to shift back towards the mean.
"""

import pandas as pd
import numpy as np


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate trading signals based on RSI mean reversion strategy.
    
    Args:
        df: DataFrame containing OHLCV data and technical indicators.
            Required columns: 'rsi14'
    
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
    required_columns = ['rsi14']
    missing_columns = [col for col in required_columns if col not in result_df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")
    
    # Initialize signal column with zeros
    result_df['signal'] = 0
    
    # Get RSI values
    rsi = result_df['rsi14']
    
    # Calculate RSI shift (previous value)
    rsi_prev = rsi.shift(1)
    
    # Define oversold and overbought thresholds
    oversold_threshold = 30
    overbought_threshold = 70
    
    # Generate LONG signals
    # Condition: RSI < 30 (oversold) AND RSI is rising (current > previous)
    long_condition = (rsi < oversold_threshold) & (rsi > rsi_prev)
    result_df.loc[long_condition, 'signal'] = 1
    
    # Generate SHORT signals
    # Condition: RSI > 70 (overbought) AND RSI is falling (current < previous)
    short_condition = (rsi > overbought_threshold) & (rsi < rsi_prev)
    result_df.loc[short_condition, 'signal'] = -1
    
    # Log signal statistics for debugging (optional)
    total_signals = result_df['signal'].abs().sum()
    long_signals = (result_df['signal'] == 1).sum()
    short_signals = (result_df['signal'] == -1).sum()
    
    # Optional: Print signal summary
    # print(f"RSI Mean Reversion - Total signals: {total_signals} (Long: {long_signals}, Short: {short_signals})")
    
    return result_df
