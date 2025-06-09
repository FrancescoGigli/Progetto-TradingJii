"""
Trading Strategies for Web App
All strategies adapted for the new backtest engine
"""

import pandas as pd
import numpy as np


def rsi_mean_reversion(df: pd.DataFrame) -> pd.DataFrame:
    """
    RSI Mean Reversion Strategy
    - Long when RSI < 30 and starts rising
    - Short when RSI > 70 and starts falling
    """
    df = df.copy()
    
    # Calculate RSI if not present
    if 'rsi14' not in df.columns:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi14'] = 100 - (100 / (1 + rs))
    
    # Initialize signal column
    df['signal'] = 0
    
    # Long signals: RSI < 30 and starting to rise
    df.loc[(df['rsi14'] < 30) & (df['rsi14'] > df['rsi14'].shift(1)), 'signal'] = 1
    
    # Short signals: RSI > 70 and starting to fall
    df.loc[(df['rsi14'] > 70) & (df['rsi14'] < df['rsi14'].shift(1)), 'signal'] = -1
    
    return df


def ema_crossover(df: pd.DataFrame) -> pd.DataFrame:
    """
    EMA Crossover Strategy
    - Long when EMA20 crosses above EMA50
    - Short when EMA20 crosses below EMA50
    """
    df = df.copy()
    
    # Calculate EMAs if not present
    if 'ema20' not in df.columns:
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    if 'ema50' not in df.columns:
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # Initialize signal column
    df['signal'] = 0
    
    # Previous values
    ema20_prev = df['ema20'].shift(1)
    ema50_prev = df['ema50'].shift(1)
    
    # Long signal: EMA20 crosses above EMA50
    df.loc[(df['ema20'] > df['ema50']) & (ema20_prev <= ema50_prev), 'signal'] = 1
    
    # Short signal: EMA20 crosses below EMA50
    df.loc[(df['ema20'] < df['ema50']) & (ema20_prev >= ema50_prev), 'signal'] = -1
    
    return df


def breakout_range(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    Breakout Range Strategy
    - Long when close breaks above 20-period high
    - Short when close breaks below 20-period low
    """
    df = df.copy()
    
    # Calculate rolling high and low
    df['high_20'] = df['high'].rolling(window=period).max().shift(1)
    df['low_20'] = df['low'].rolling(window=period).min().shift(1)
    
    # Initialize signal column
    df['signal'] = 0
    
    # Long signal: close breaks above 20-period high
    df.loc[df['close'] > df['high_20'], 'signal'] = 1
    
    # Short signal: close breaks below 20-period low
    df.loc[df['close'] < df['low_20'], 'signal'] = -1
    
    return df


def bollinger_rebound(df: pd.DataFrame, period: int = 20, std_dev: int = 2) -> pd.DataFrame:
    """
    Bollinger Bands Rebound Strategy
    - Long when close bounces off lower band
    - Short when close bounces off upper band
    """
    df = df.copy()
    
    # Calculate Bollinger Bands if not present
    if 'bb_upper' not in df.columns or 'bb_lower' not in df.columns:
        df['bb_middle'] = df['close'].rolling(window=period).mean()
        df['bb_std'] = df['close'].rolling(window=period).std()
        df['bb_upper'] = df['bb_middle'] + (std_dev * df['bb_std'])
        df['bb_lower'] = df['bb_middle'] - (std_dev * df['bb_std'])
    
    # Initialize signal column
    df['signal'] = 0
    
    # Previous values
    close_prev = df['close'].shift(1)
    bb_lower_prev = df['bb_lower'].shift(1)
    bb_upper_prev = df['bb_upper'].shift(1)
    
    # Long signal: close was below lower band and now above it (bounce)
    df.loc[(close_prev < bb_lower_prev) & (df['close'] > df['bb_lower']), 'signal'] = 1
    
    # Short signal: close was above upper band and now below it (bounce)
    df.loc[(close_prev > bb_upper_prev) & (df['close'] < df['bb_upper']), 'signal'] = -1
    
    return df


def macd_histogram(df: pd.DataFrame) -> pd.DataFrame:
    """
    MACD Histogram Strategy
    - Long when MACD histogram crosses from negative to positive
    - Short when MACD histogram crosses from positive to negative
    """
    df = df.copy()
    
    # Calculate MACD if not present
    if 'macdhist' not in df.columns:
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macdsignal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macdhist'] = df['macd'] - df['macdsignal']
    
    # Initialize signal column
    df['signal'] = 0
    
    # Previous histogram value
    hist_prev = df['macdhist'].shift(1)
    
    # Long signal: histogram crosses from negative to positive
    df.loc[(df['macdhist'] > 0) & (hist_prev <= 0), 'signal'] = 1
    
    # Short signal: histogram crosses from positive to negative
    df.loc[(df['macdhist'] < 0) & (hist_prev >= 0), 'signal'] = -1
    
    return df


def donchian_breakout(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    Donchian Channel Breakout Strategy
    - Long when close breaks above upper channel
    - Short when close breaks below lower channel
    """
    df = df.copy()
    
    # Calculate Donchian channels
    df['dc_upper'] = df['high'].rolling(window=period).max().shift(1)
    df['dc_lower'] = df['low'].rolling(window=period).min().shift(1)
    
    # Initialize signal column
    df['signal'] = 0
    
    # Long signal: close breaks above upper channel
    df.loc[df['close'] > df['dc_upper'], 'signal'] = 1
    
    # Short signal: close breaks below lower channel
    df.loc[df['close'] < df['dc_lower'], 'signal'] = -1
    
    return df


def adx_filter_crossover(df: pd.DataFrame, adx_threshold: int = 20) -> pd.DataFrame:
    """
    ADX Filtered EMA Crossover Strategy
    - Same as EMA crossover but only when ADX > 20 (trending market)
    """
    df = df.copy()
    
    # Calculate EMAs if not present
    if 'ema20' not in df.columns:
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    if 'ema50' not in df.columns:
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # Calculate ADX if not present
    if 'adx' not in df.columns:
        # Simplified ADX calculation
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr14 = true_range.rolling(14).mean()
        
        # Directional indicators
        up_move = df['high'] - df['high'].shift()
        down_move = df['low'].shift() - df['low']
        
        pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        pos_di = 100 * pd.Series(pos_dm).rolling(14).mean() / atr14
        neg_di = 100 * pd.Series(neg_dm).rolling(14).mean() / atr14
        
        dx = 100 * np.abs(pos_di - neg_di) / (pos_di + neg_di)
        df['adx'] = dx.rolling(14).mean()
    
    # Initialize signal column
    df['signal'] = 0
    
    # Previous values
    ema20_prev = df['ema20'].shift(1)
    ema50_prev = df['ema50'].shift(1)
    
    # Long signal: EMA20 crosses above EMA50 AND ADX > threshold
    df.loc[(df['ema20'] > df['ema50']) & (ema20_prev <= ema50_prev) & (df['adx'] > adx_threshold), 'signal'] = 1
    
    # Short signal: EMA20 crosses below EMA50 AND ADX > threshold
    df.loc[(df['ema20'] < df['ema50']) & (ema20_prev >= ema50_prev) & (df['adx'] > adx_threshold), 'signal'] = -1
    
    return df


# Strategy mapping
STRATEGIES = {
    'rsi_mean_reversion': rsi_mean_reversion,
    'ema_crossover': ema_crossover,
    'breakout_range': breakout_range,
    'bollinger_rebound': bollinger_rebound,
    'macd_histogram': macd_histogram,
    'donchian_breakout': donchian_breakout,
    'adx_filter_crossover': adx_filter_crossover
}
