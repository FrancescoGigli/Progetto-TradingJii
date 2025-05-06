# config.py

import os
import sys
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys for Bybit (from environment variables)
API_KEY = os.getenv('BYBIT_API_KEY')
API_SECRET = os.getenv('BYBIT_API_SECRET')

# Exchange configuration
exchange_config = {
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'swap',  # Use swap for linear and inverse contracts
        'adjustForTimeDifference': True
    }
}

# Database configuration
DB_FILE = 'crypto_data.db'

# Subseries configuration
SUBSERIES_LENGTH = 8  # Length of subseries (default: 8 time frames)
SUBSERIES_MIN_SAMPLES = 2  # Minimum number of samples per category (reduced for testing)

# Timeframes to analyze (adjust as needed)
ENABLED_TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '4h']

# Default timeframe
TIMEFRAME_DEFAULT = '1h'

# Number of top cryptocurrencies to analyze
TOP_ANALYSIS_CRYPTO = 100  # Top X cryptocurrencies by 24h volume

# Day range for data collection
DATA_LIMIT_DAYS = 100  # How many days of data to collect (approx)

# Symbols to exclude from analysis (regex patterns)
EXCLUDED_SYMBOLS = [
    r'USDC',    # Avoid stable coins or coins with unusual behavior
    r'BUSD',
    r'DAI',
    r'TUSD',
    r'USDT/USD',  # Avoid stable coin pairs
    r'[A-Z0-9]+USD/USDT',  # Avoid chains like ETHUSDT/USDT
    r'_PREMIUM'  # Avoid premium index contracts
]

# API call delay to avoid rate limits (in seconds)
API_CALL_DELAY = 0.1

# Process data in batches to avoid memory issues
BATCH_SIZE = 10

# Reset database on startup - CAUTION: will delete all data
RESET_DB_ON_STARTUP = False  # Set to True to reset database on startup

# Check if config is imported correctly
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Config imported successfully.")
    logging.info(f"Enabled timeframes: {ENABLED_TIMEFRAMES}")
    logging.info(f"Top cryptocurrencies to analyze: {TOP_ANALYSIS_CRYPTO}")
    logging.info(f"Data limit days: {DATA_LIMIT_DAYS}")
    logging.info(f"Database file: {DB_FILE}")
    logging.info(f"Reset database on startup: {RESET_DB_ON_STARTUP}")
    
    masked_key = API_KEY[:4] + '*' * (len(API_KEY) - 8) + API_KEY[-4:] if API_KEY and len(API_KEY) > 8 else 'Not set'
    masked_secret = API_SECRET[:4] + '*' * (len(API_SECRET) - 8) + API_SECRET[-4:] if API_SECRET and len(API_SECRET) > 8 else 'Not set'
    
    logging.info(f"API Key: {masked_key}")
    logging.info(f"API Secret: {masked_secret}")
