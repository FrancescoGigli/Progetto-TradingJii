#!/usr/bin/env python3
"""
Command-line argument parsing for TradingJii
"""

import argparse
from modules.utils.config import (
    DEFAULT_TOP_SYMBOLS, DEFAULT_DAYS, DEFAULT_TIMEFRAMES,
    DEFAULT_BATCH_SIZE, DEFAULT_CONCURRENCY, TIMEFRAME_CONFIG
)

def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        The parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Scarica dati OHLCV delle criptovalute da Bybit',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Parametri generali
    parser.add_argument(
        '-n', '--num-symbols',
        type=int,
        default=DEFAULT_TOP_SYMBOLS,
        help='Numero di criptovalute da scaricare'
    )
    
    parser.add_argument(
        '-d', '--days',
        type=int,
        default=DEFAULT_DAYS,
        help='Giorni di dati storici da scaricare'
    )
    
    parser.add_argument(
        '-t', '--timeframes',
        nargs='+',
        default=DEFAULT_TIMEFRAMES,
        choices=list(TIMEFRAME_CONFIG.keys()),
        help='Timeframes da scaricare'
    )
    
    # Parametri di ottimizzazione
    optimization_group = parser.add_argument_group('Parametri di ottimizzazione')
    
    optimization_group.add_argument(
        '-c', '--concurrency',
        type=int,
        default=DEFAULT_CONCURRENCY,
        help='Numero massimo di download paralleli per batch'
    )
    
    optimization_group.add_argument(
        '-b', '--batch-size',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help='Dimensione del batch per il download'
    )
    
    optimization_group.add_argument(
        '-s', '--sequential',
        action='store_true',
        default=True,
        help='Esegui in modalità sequenziale (DEFAULT)'
    )
    
    optimization_group.add_argument(
        '--parallel',
        action='store_false',
        dest='sequential',
        help='Disattiva modalità sequenziale, usa modalità parallela'
    )
    
    # Technical Analysis options
    ta_group = parser.add_argument_group('Analisi Tecnica')
    
    ta_group.add_argument(
        '--no-ta',
        action='store_true',
        help='Disabilita il calcolo degli indicatori di analisi tecnica'
    )
    
    # Data Validation options
    validation_group = parser.add_argument_group('Validazione Dati')
    
    validation_group.add_argument(
        '--skip-validation',
        action='store_true',
        help='Salta la validazione dati (utile per test o ambienti a basse risorse)'
    )
    
    validation_group.add_argument(
        '--export-validation-report',
        action='store_true',
        help='Esporta report di validazione in CSV con timestamp'
    )
    
    validation_group.add_argument(
        '--generate-validation-charts',
        action='store_true',
        help='Genera grafici/heatmap della qualità dati (richiede matplotlib)'
    )
    
    return parser.parse_args()
