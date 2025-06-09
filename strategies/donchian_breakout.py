"""
Donchian Breakout Strategy

This strategy implements the Donchian Channel breakout system, which identifies
breakouts from price channels formed by recent highs and lows. Named after
Richard Donchian, this is one of the oldest trend-following strategies.

Strategy Rules:
- LONG Entry: Close price breaks above the 20-period high (upper Donchian channel)
- SHORT Entry: Close price breaks below the 20-period low (lower Donchian channel)

The strategy captures strong directional moves when price breaks out of
established trading ranges.
"""

import pandas as pd
import numpy as np


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate trading signals based on Donchian channel breakout strategy.
    
    Args:
        df: DataFrame containing OHLCV data and technical indicators.
            Required columns: 'high', 'low', 'close'
    
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
    required_columns = ['high', 'low', 'close']
    missing_columns = [col for col in required_columns if col not in result_df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")
    
    # Initialize signal column with zeros
    result_df['signal'] = 0
    
    # Define Donchian channel period
    channel_period = 20
    
    # Calculate Donchian channels (20-period high and low)
    # Shift by 1 to ensure we're comparing with previous period's channel
    donchian_upper = result_df['high'].rolling(window=channel_period).max().shift(1)
    donchian_lower = result_df['low'].rolling(window=channel_period).min().shift(1)
    
    # Get current close price
    close = result_df['close']
    
    # Generate LONG signals
    # Condition: Close price breaks above the upper Donchian channel
    long_condition = close > donchian_upper
    result_df.loc[long_condition, 'signal'] = 1
    
    # Generate SHORT signals
    # Condition: Close price breaks below the lower Donchian channel
    short_condition = close < donchian_lower
    result_df.loc[short_condition, 'signal'] = -1
    
    # Log signal statistics for debugging (optional)
    total_signals = result_df['signal'].abs().sum()
    long_signals = (result_df['signal'] == 1).sum()
    short_signals = (result_df['signal'] == -1).sum()
    
    # Optional: Print signal summary
    # print(f"Donchian Breakout - Total signals: {total_signals} (Long: {long_signals}, Short: {short_signals})")
    
    return result_df
