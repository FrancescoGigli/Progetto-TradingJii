#!/usr/bin/env python3
"""
Configuration module for TradingJii

Contains all configuration constants and settings used throughout the application.
"""

import os
from datetime import timedelta
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys for Bybit (from environment variables)
API_KEY = os.environ.get('BYBIT_API_KEY', '')
API_SECRET = os.environ.get('BYBIT_API_SECRET', '')

# Database configuration
DB_FILE = 'crypto_data.db'

# Default values
DEFAULT_TOP_SYMBOLS = 100
DEFAULT_DAYS = 300
DEFAULT_TIMEFRAMES = ['1h', '4h']
DEFAULT_BATCH_SIZE = 25
DEFAULT_CONCURRENCY = 8
DEFAULT_WINDOW_SIZE = 7  # Default window size for ML pattern generation

# ML classification thresholds
BUY_THRESHOLD = 0.5    # If y >= 0.5, then y_class = 1 (BUY)
SELL_THRESHOLD = -0.5  # If y <= -0.5, then y_class = 2 (SELL)

# Exchange configuration
EXCHANGE_CONFIG = {
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
    'timeout': 30000,
    'options': {
        'defaultType': 'swap',
        'adjustForTimeDifference': True
    }
}

# Timeframe configuration
TIMEFRAME_CONFIG = {
    '1m': {'max_age': timedelta(minutes=5), 'ms': 60 * 1000},
    '5m': {'max_age': timedelta(minutes=15), 'ms': 5 * 60 * 1000},
    '15m': {'max_age': timedelta(hours=1), 'ms': 15 * 60 * 1000},
    '30m': {'max_age': timedelta(hours=2), 'ms': 30 * 60 * 1000},
    '1h': {'max_age': timedelta(hours=4), 'ms': 60 * 60 * 1000},
    '4h': {'max_age': timedelta(hours=12), 'ms': 4 * 60 * 60 * 1000},
    '1d': {'max_age': timedelta(days=2), 'ms': 24 * 60 * 60 * 1000}
}

# Technical Analysis Parameters
# These parameters can be overridden by user configuration
TA_PARAMS = {
    # Simple Moving Averages
    'sma9': {'timeperiod': 9},
    'sma20': {'timeperiod': 20},
    'sma50': {'timeperiod': 50},
    
    # Exponential Moving Averages
    'ema20': {'timeperiod': 20},
    'ema50': {'timeperiod': 50},
    'ema200': {'timeperiod': 200},
    
    # Momentum Indicators
    'rsi14': {'timeperiod': 14},
    'stoch': {'fastk_period': 14, 'slowk_period': 3, 'slowd_period': 3},
    'macd': {'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9},
    
    # Volatility Indicators
    'atr14': {'timeperiod': 14},
    'bbands': {'timeperiod': 20, 'nbdevup': 2, 'nbdevdn': 2},
    
    # Volume-based Indicators
    'volume_sma20': {'timeperiod': 20},
    
    # Trend Strength
    'adx14': {'timeperiod': 14}
}
