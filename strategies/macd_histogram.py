"""
MACD Histogram Strategy

This strategy uses the MACD (Moving Average Convergence Divergence) histogram
to identify momentum shifts. The histogram represents the difference between
the MACD line and its signal line.

Strategy Rules:
- LONG Entry: MACD histogram crosses from negative to positive (bullish momentum)
- SHORT Entry: MACD histogram crosses from positive to negative (bearish momentum)

The zero-line crossover of the histogram indicates a change in momentum direction
and potential trend reversal or continuation.
"""

import pandas as pd
import numpy as np


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate trading signals based on MACD histogram zero-line crossover.
    
    Args:
        df: DataFrame containing OHLCV data and technical indicators.
            Required columns: 'macd_hist'
    
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
    required_columns = ['macd_hist']
    missing_columns = [col for col in required_columns if col not in result_df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")
    
    # Initialize signal column with zeros
    result_df['signal'] = 0
    
    # Get MACD histogram values
    macd_hist = result_df['macd_hist']
    
    # Calculate previous histogram value
    macd_hist_prev = macd_hist.shift(1)
    
    # Generate LONG signals
    # Condition: MACD histogram crosses from negative to positive
    # Current histogram > 0 AND previous histogram <= 0
    long_condition = (macd_hist > 0) & (macd_hist_prev <= 0)
    result_df.loc[long_condition, 'signal'] = 1
    
    # Generate SHORT signals
    # Condition: MACD histogram crosses from positive to negative
    # Current histogram < 0 AND previous histogram >= 0
    short_condition = (macd_hist < 0) & (macd_hist_prev >= 0)
    result_df.loc[short_condition, 'signal'] = -1
    
    # Log signal statistics for debugging (optional)
    total_signals = result_df['signal'].abs().sum()
    long_signals = (result_df['signal'] == 1).sum()
    short_signals = (result_df['signal'] == -1).sum()
    
    # Optional: Print signal summary
    # print(f"MACD Histogram - Total signals: {total_signals} (Long: {long_signals}, Short: {short_signals})")
    
    return result_df
