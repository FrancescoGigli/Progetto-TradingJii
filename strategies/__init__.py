"""
Trading Strategies Module for TradingJii

This module contains various trading strategies that generate buy/sell signals
based on technical indicators and price patterns.

Each strategy module exports a `generate_signals` function that takes a DataFrame
with OHLCV and indicator data and returns the same DataFrame with an additional
'signal' column containing:
- 1: Long entry signal
- -1: Short entry signal
- 0 or NaN: No signal

Available strategies:
- rsi_mean_reversion: Mean reversion based on RSI oversold/overbought levels
- ema_crossover: Moving average crossover strategy using EMA20 and EMA50
- breakout_range: Price breakout from recent high/low range
- bollinger_rebound: Price rebound from Bollinger Bands
- macd_histogram: MACD histogram zero-line crossover
- donchian_breakout: Donchian channel breakout strategy
- adx_filter_crossover: EMA crossover filtered by ADX strength
"""

__all__ = [
    'rsi_mean_reversion',
    'ema_crossover',
    'breakout_range',
    'bollinger_rebound',
    'macd_histogram',
    'donchian_breakout',
    'adx_filter_crossover'
]
