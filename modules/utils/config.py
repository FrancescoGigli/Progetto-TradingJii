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
DEFAULT_TOP_SYMBOLS = 5  # Top 5 crypto per volume
EXCLUDED_SYMBOLS = []  # Simboli da escludere dalla selezione automatica
# Configurazione giorni di dati con warmup automatico per indicatori
DESIRED_ANALYSIS_DAYS = 365  # Giorni di dati puliti che vogliamo per l'analisi
INDICATOR_WARMUP_BUFFER = 10  # Buffer aggiuntivo per sicurezza

# Il valore effettivo di DEFAULT_DAYS verrà calcolato dinamicamente
# Include automaticamente il periodo di warmup necessario per gli indicatori
DEFAULT_DAYS = 365  # Verrà aggiornato dalla funzione calculate_total_days_needed()
DEFAULT_TIMEFRAMES = ['1h', '4h']
DEFAULT_BATCH_SIZE = 15  # Batch size moderato per stabilità
DEFAULT_CONCURRENCY = 6  # Concorrenza bilanciata per evitare rate limits

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

# OHLCV Fetch Retry Configuration
OHLCV_FETCH_CONFIG = {
    'max_retries': 3,           # Numero massimo di tentativi per fetch OHLCV
    'backoff_seconds': 2,       # Tempo base per backoff esponenziale (secondi)
    'max_backoff_seconds': 30,  # Tempo massimo di attesa tra tentativi
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
    'num_symbols': 5,  # Numero di top simboli per volume
    'specific_symbols': [  # Lista specifica di simboli (opzionale)
        'BTC/USDT:USDT',
        'ETH/USDT:USDT', 
        'SOL/USDT:USDT'
    ],
    'use_specific_symbols': True,  # True = usa specific_symbols, False = usa top N per volume
    
    # Timeframes da processare
    'timeframes': ['1h', '4h'],  # Lista dei timeframes da scaricare
    'days_back': 365,  # Giorni di storico da scaricare
    
    # Performance e concorrenza
    'batch_size': 15,  # Dimensione batch per download
    'concurrency': 6,  # Numero download paralleli per batch
    'sequential_mode': False,  # True = sequenziale, False = parallelo
    
    # Funzionalità avanzate
    'enable_technical_analysis': True,  # Calcola indicatori tecnici
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

# Funzioni per il calcolo automatico del warmup degli indicatori
def calculate_indicator_warmup_period() -> int:
    """
    Calcola automaticamente il periodo di warmup necessario per gli indicatori
    basandosi sui parametri configurati in TA_PARAMS.
    
    Returns:
        Il numero di periodi necessari per il warmup (corrispondente all'indicatore più lento)
    """
    # Trova il periodo più lungo tra tutti gli indicatori configurati
    max_periods = []
    
    # Simple Moving Averages
    max_periods.extend([
        TA_PARAMS.get('sma9', {}).get('timeperiod', 9),
        TA_PARAMS.get('sma20', {}).get('timeperiod', 20),
        TA_PARAMS.get('sma50', {}).get('timeperiod', 50)
    ])
    
    # Exponential Moving Averages
    max_periods.extend([
        TA_PARAMS.get('ema20', {}).get('timeperiod', 20),
        TA_PARAMS.get('ema50', {}).get('timeperiod', 50),
        TA_PARAMS.get('ema200', {}).get('timeperiod', 200)  # Questo è tipicamente il più lungo
    ])
    
    # Momentum Indicators
    max_periods.extend([
        TA_PARAMS.get('rsi14', {}).get('timeperiod', 14),
        TA_PARAMS.get('stoch', {}).get('fastk_period', 14),
        TA_PARAMS.get('macd', {}).get('slowperiod', 26)
    ])
    
    # Volatility Indicators
    max_periods.extend([
        TA_PARAMS.get('atr14', {}).get('timeperiod', 14),
        TA_PARAMS.get('bbands', {}).get('timeperiod', 20)
    ])
    
    # Volume-based Indicators
    max_periods.append(TA_PARAMS.get('volume_sma20', {}).get('timeperiod', 20))
    
    # Trend Strength
    max_periods.append(TA_PARAMS.get('adx14', {}).get('timeperiod', 14))
    
    # Trova il periodo massimo
    warmup_periods = max(max_periods) if max_periods else 200
    
    # Aggiungi il buffer di sicurezza
    return warmup_periods + INDICATOR_WARMUP_BUFFER

def calculate_total_days_needed() -> int:
    """
    Calcola il numero totale di giorni di dati da scaricare,
    includendo il periodo di warmup per gli indicatori.
    
    Returns:
        Numero totale di giorni da scaricare (analisi + warmup)
    """
    warmup_periods = calculate_indicator_warmup_period()
    total_days = DESIRED_ANALYSIS_DAYS + warmup_periods
    
    return total_days

def update_default_days():
    """
    Aggiorna la variabile globale DEFAULT_DAYS con il valore calcolato dinamicamente.
    Questa funzione dovrebbe essere chiamata all'avvio del sistema.
    """
    global DEFAULT_DAYS
    DEFAULT_DAYS = calculate_total_days_needed()
    return DEFAULT_DAYS

def get_effective_analysis_period(timeframe: str) -> dict:
    """
    Calcola informazioni dettagliate sul periodo di analisi effettivo
    per un dato timeframe.
    
    Args:
        timeframe: Il timeframe (es. '1h', '4h', '1d')
        
    Returns:
        Dictionary con informazioni sul periodo di analisi
    """
    warmup_periods = calculate_indicator_warmup_period()
    total_days = calculate_total_days_needed()
    
    # Calcola il numero di candele per timeframe
    timeframe_multipliers = {
        '1m': 24 * 60,
        '5m': 24 * 12,
        '15m': 24 * 4,
        '30m': 24 * 2,
        '1h': 24,
        '4h': 6,
        '1d': 1
    }
    
    multiplier = timeframe_multipliers.get(timeframe, 24)
    
    total_candles = total_days * multiplier
    warmup_candles = warmup_periods
    analysis_candles = total_candles - warmup_candles
    
    return {
        'total_days': total_days,
        'analysis_days': DESIRED_ANALYSIS_DAYS,
        'warmup_periods': warmup_periods,
        'total_candles': total_candles,
        'warmup_candles': warmup_candles,
        'analysis_candles': analysis_candles,
        'clean_data_percentage': (analysis_candles / total_candles) * 100 if total_candles > 0 else 0
    }

# Aggiorna DEFAULT_DAYS all'importazione del modulo
update_default_days()
