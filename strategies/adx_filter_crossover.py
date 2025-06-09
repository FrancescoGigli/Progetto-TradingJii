"""
ADX Filter Crossover Strategy

This strategy combines the EMA crossover signals with an ADX (Average Directional Index)
filter. The ADX is used to confirm that the market is trending before taking
crossover signals, avoiding whipsaws in ranging markets.

Strategy Rules:
- LONG Entry: EMA20 crosses above EMA50 AND ADX > 20
- SHORT Entry: EMA20 crosses below EMA50 AND ADX > 20

The ADX filter ensures that trades are only taken when there's sufficient
directional movement in the market, improving the quality of signals.
"""

import pandas as pd
import numpy as np


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate trading signals based on EMA crossover filtered by ADX strength.
    
    Args:
        df: DataFrame containing OHLCV data and technical indicators.
            Required columns: 'ema20', 'ema50', 'adx14'
    
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
    required_columns = ['ema20', 'ema50', 'adx14']
    missing_columns = [col for col in required_columns if col not in result_df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")
    
    # Initialize signal column with zeros
    result_df['signal'] = 0
    
    # Get indicator values
    ema20 = result_df['ema20']
    ema50 = result_df['ema50']
    adx = result_df['adx14']
    
    # Calculate previous values for crossover detection
    ema20_prev = ema20.shift(1)
    ema50_prev = ema50.shift(1)
    
    # Define ADX threshold for trend strength
    adx_threshold = 20
    
    # Generate LONG signals (Golden Cross with ADX filter)
    # Conditions:
    # 1. EMA20 crosses above EMA50
    # 2. ADX > 20 (confirming trend strength)
    long_crossover = (ema20 > ema50) & (ema20_prev <= ema50_prev)
    long_condition = long_crossover & (adx > adx_threshold)
    result_df.loc[long_condition, 'signal'] = 1
    
    # Generate SHORT signals (Death Cross with ADX filter)
    # Conditions:
    # 1. EMA20 crosses below EMA50
    # 2. ADX > 20 (confirming trend strength)
    short_crossover = (ema20 < ema50) & (ema20_prev >= ema50_prev)
    short_condition = short_crossover & (adx > adx_threshold)
    result_df.loc[short_condition, 'signal'] = -1
    
    # Log signal statistics for debugging (optional)
    total_signals = result_df['signal'].abs().sum()
    long_signals = (result_df['signal'] == 1).sum()
    short_signals = (result_df['signal'] == -1).sum()
    
    # Count filtered signals (crossovers that were rejected due to low ADX)
    filtered_long = (long_crossover & (adx <= adx_threshold)).sum()
    filtered_short = (short_crossover & (adx <= adx_threshold)).sum()
    
    # Optional: Print signal summary
    # print(f"ADX Filter Crossover - Total signals: {total_signals} (Long: {long_signals}, Short: {short_signals})")
    # print(f"  Filtered out: {filtered_long} long and {filtered_short} short signals due to ADX < {adx_threshold}")
    
    return result_df
