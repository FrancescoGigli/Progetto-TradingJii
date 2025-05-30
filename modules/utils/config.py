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
DEFAULT_TOP_SYMBOLS = 10
DEFAULT_DAYS = 600  # Optimized for technical indicators (EMA200 needs ~33 days for 4h timeframe)
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

# Real-Time System Configuration
REALTIME_CONFIG = {
    # Aggiornamento automatico
    'update_interval_seconds': 300,  # 5 minuti (300 secondi)
    'max_iterations': None,  # None = infinito, numero = limite iterazioni
    
    # Criptovalute da monitorare
    'num_symbols': 25,  # Numero di top simboli per volume
    'specific_symbols': [  # Lista specifica di simboli (opzionale)
        'BTC/USDT:USDT',
        'ETH/USDT:USDT', 
        'SOL/USDT:USDT',
        'ADA/USDT:USDT',
        'AVAX/USDT:USDT',
        'LINK/USDT:USDT',
        'UNI/USDT:USDT',
        'DOGE/USDT:USDT',
        'XRP/USDT:USDT',
        'SUI/USDT:USDT'
    ],
    'use_specific_symbols': False,  # True = usa specific_symbols, False = usa top N per volume
    
    # Timeframes da processare
    'timeframes': ['1h', '4h'],  # Lista dei timeframes da scaricare
    'days_back': 365,  # Giorni di storico da scaricare
    
    # Performance e concorrenza
    'batch_size': 15,  # Dimensione batch per download
    'concurrency': 6,  # Numero download paralleli per batch
    'sequential_mode': False,  # True = sequenziale, False = parallelo
    
    # Funzionalità avanzate
    'enable_technical_analysis': True,  # Calcola indicatori tecnici
    'enable_ml_datasets': True,  # Genera dataset ML automaticamente (DEFAULT)
    'force_ml_regeneration': False,  # Forza rigenerazione dataset ML
    'enable_data_validation': True,  # Validazione e riparazione dati
    'export_validation_reports': False,  # Export report validazione CSV
    'generate_validation_charts': False,  # Genera grafici validazione
    
    # Notifiche e logging
    'log_level': 'INFO',  # DEBUG, INFO, WARNING, ERROR
    'enable_startup_banner': True,  # Mostra banner di avvio
    'enable_progress_display': True,  # Mostra progress dettagliato
    
    # Controlli di sicurezza
    'max_download_retries': 3,  # Numero massimo retry per download falliti
    'api_rate_limit_delay': 1.0,  # Delay tra chiamate API (secondi)
    'emergency_stop_file': '.stop_realtime',  # File per stop di emergenza
}

# Training System Configuration  
TRAINING_CONFIG = {
    # Modelli disponibili
    'available_models': [
        'RandomForest',
        'XGBoost', 
        'LightGBM',
        'MLP'
    ],
    
    # Configurazione modelli di default
    'model_params': {
        'RandomForest': {
            'n_estimators': 100,
            'random_state': 42,
            'n_jobs': -1
        },
        'XGBoost': {
            'random_state': 42,
            'eval_metric': 'mlogloss',
            'verbosity': 0
        },
        'LightGBM': {
            'random_state': 42,
            'verbose': -1,
            'force_row_wise': True
        },
        'MLP': {
            'hidden_layer_sizes': (100, 50),
            'max_iter': 500,
            'random_state': 42,
            'early_stopping': True,
            'validation_fraction': 0.1
        }
    },
    
    # Configurazione training
    'train_test_split': 0.8,  # 80% training, 20% test
    'temporal_split': True,  # Split temporale (non random)
    'enable_preprocessing': True,  # Preprocessing avanzato
    'enable_feature_engineering': True,  # Feature engineering
    'log_training_to_file': True,  # Salva log training su file
    
    # Output e report
    'models_dir': 'models',
    'reports_dir': 'ml_system/reports/training',
    'enable_model_comparison': True,
    'save_feature_importance': True,
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
