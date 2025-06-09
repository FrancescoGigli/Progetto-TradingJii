"""
Bollinger Rebound Strategy

This strategy implements a mean reversion approach using Bollinger Bands. It looks
for price rebounds when the price touches or penetrates the Bollinger Bands and
then returns inside the bands.

Strategy Rules:
- LONG Entry: Price goes below lower band and then rebounds above it
- SHORT Entry: Price goes above upper band and then rebounds below it

The strategy assumes that extreme price movements outside the bands are temporary
and prices will revert back within the normal trading range.
"""

import pandas as pd
import numpy as np


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate trading signals based on Bollinger Bands rebound strategy.
    
    Args:
        df: DataFrame containing OHLCV data and technical indicators.
            Required columns: 'close', 'bbands_upper', 'bbands_lower'
    
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
    required_columns = ['close', 'bbands_upper', 'bbands_lower']
    missing_columns = [col for col in required_columns if col not in result_df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")
    
    # Initialize signal column with zeros
    result_df['signal'] = 0
    
    # Get price and Bollinger Bands values
    close = result_df['close']
    bb_upper = result_df['bbands_upper']
    bb_lower = result_df['bbands_lower']
    
    # Calculate previous values
    close_prev = close.shift(1)
    bb_upper_prev = bb_upper.shift(1)
    bb_lower_prev = bb_lower.shift(1)
    
    # Generate LONG signals
    # Condition: Previous close was below lower band AND current close is above lower band
    # This indicates a rebound from oversold conditions
    long_condition = (close_prev < bb_lower_prev) & (close > bb_lower)
    result_df.loc[long_condition, 'signal'] = 1
    
    # Generate SHORT signals
    # Condition: Previous close was above upper band AND current close is below upper band
    # This indicates a rebound from overbought conditions
    short_condition = (close_prev > bb_upper_prev) & (close < bb_upper)
    result_df.loc[short_condition, 'signal'] = -1
    
    # Log signal statistics for debugging (optional)
    total_signals = result_df['signal'].abs().sum()
    long_signals = (result_df['signal'] == 1).sum()
    short_signals = (result_df['signal'] == -1).sum()
    
    # Optional: Print signal summary
    # print(f"Bollinger Rebound - Total signals: {total_signals} (Long: {long_signals}, Short: {short_signals})")
    
    return result_df
