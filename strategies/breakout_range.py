"""
Breakout Range Strategy

This strategy implements a price breakout approach that identifies when the price
breaks out of its recent trading range. It looks for breakouts above recent highs
or below recent lows as potential entry points.

Strategy Rules:
- LONG Entry: Close price breaks above the highest high of the last 20 periods
- SHORT Entry: Close price breaks below the lowest low of the last 20 periods

The strategy capitalizes on momentum when price breaks out of consolidation ranges,
assuming the breakout direction will continue.
"""

import pandas as pd
import numpy as np


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate trading signals based on breakout range strategy.
    
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
    
    # Define lookback period
    lookback_period = 20
    
    # Calculate rolling maximum high and minimum low
    # Use shift(1) to avoid look-ahead bias (we compare with previous periods)
    rolling_high = result_df['high'].rolling(window=lookback_period).max().shift(1)
    rolling_low = result_df['low'].rolling(window=lookback_period).min().shift(1)
    
    # Get current close price
    close = result_df['close']
    
    # Generate LONG signals
    # Condition: Close price breaks above the highest high of the last 20 periods
    long_condition = close > rolling_high
    result_df.loc[long_condition, 'signal'] = 1
    
    # Generate SHORT signals
    # Condition: Close price breaks below the lowest low of the last 20 periods
    short_condition = close < rolling_low
    result_df.loc[short_condition, 'signal'] = -1
    
    # Note: If price simultaneously breaks above high and below low (unlikely but possible
    # in very volatile conditions), the last assignment will take precedence (SHORT signal)
    
    # Log signal statistics for debugging (optional)
    total_signals = result_df['signal'].abs().sum()
    long_signals = (result_df['signal'] == 1).sum()
    short_signals = (result_df['signal'] == -1).sum()
    
    # Optional: Print signal summary
    # print(f"Breakout Range - Total signals: {total_signals} (Long: {long_signals}, Short: {short_signals})")
    
    return result_df
