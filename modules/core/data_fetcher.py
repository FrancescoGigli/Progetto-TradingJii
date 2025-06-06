#!/usr/bin/env python3
"""
Data fetcher module for TradingJii

Handles OHLCV data fetching from the exchange.
"""

import logging
import asyncio
import ccxt
from datetime import datetime, timedelta
from colorama import Fore, Style
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from modules.utils.config import TIMEFRAME_CONFIG, OHLCV_FETCH_CONFIG
from modules.data.db_manager import check_data_freshness, save_ohlcv_data

# Data fissa di inizio - 1° gennaio 2024
MIN_START_DATE = datetime(2024, 1, 1)

# Periodo di warmup per il calcolo degli indicatori tecnici
# Quante candele precedenti al 1° gennaio 2024 scaricare per timeframe
WARMUP_CANDLES = {
    '1m': 300,
    '5m': 300,
    '15m': 300,
    '30m': 250,
    '1h': 250,
    '4h': 250,
    '1d': 250
}

def estimated_iterations(since, now_ms, timeframe):
    """
    Calculate the estimated number of iterations for the progress bar.
    
    Args:
        since: Start timestamp in milliseconds
        now_ms: End timestamp in milliseconds
        timeframe: The timeframe being processed
        
    Returns:
        Estimated number of iterations
    """
    time_diff_ms = now_ms - since
    return max(1, int(time_diff_ms / TIMEFRAME_CONFIG[timeframe]['ms'] / 1000))

async def fetch_with_retry(exchange, symbol, timeframe, since, limit):
    """
    Fetch OHLCV data with exponential backoff retry mechanism using config parameters.
    
    Args:
        exchange: The exchange object
        symbol: The cryptocurrency symbol
        timeframe: The timeframe to fetch data for
        since: Start timestamp in milliseconds
        limit: Maximum number of candles to fetch
        
    Returns:
        OHLCV data chunk or raises exception after all retries fail
    """
    max_retries = OHLCV_FETCH_CONFIG['max_retries']
    backoff = OHLCV_FETCH_CONFIG['backoff_seconds']
    max_backoff = OHLCV_FETCH_CONFIG['max_backoff_seconds']
    
    for attempt in range(max_retries):
        try:
            return await exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            if attempt == max_retries - 1:  # Last attempt
                raise e
            wait_time = min(backoff * (2 ** attempt), max_backoff)
            logging.warning(f"Tentativo {attempt+1} fallito per {symbol} ({timeframe}): {e}. Ritento tra {wait_time}s...")
            await asyncio.sleep(wait_time)
    return None

async def fetch_ohlcv_data(exchange, symbol, timeframe, data_limit_days):
    """
    Fetch OHLCV data for a specific symbol and timeframe.
    Include dati di warmup prima del 1° gennaio 2024 per calcolo indicatori,
    ma salva solo i dati dal 1° gennaio 2024 in poi.
    
    Args:
        exchange: The exchange object
        symbol: The cryptocurrency symbol
        timeframe: The timeframe to fetch data for
        data_limit_days: Maximum days of historical data to fetch
        
    Returns:
        Tuple of (success, count) or None if data is already fresh
    """
    # Check if we already have fresh data
    is_fresh, last_date = check_data_freshness(symbol, timeframe)
    
    # Se l'ultima data è prima del 1° gennaio 2024, considera i dati non freschi
    if is_fresh and last_date and last_date < MIN_START_DATE:
        is_fresh = False
    
    if is_fresh:
        return None
        
    # Determine data fetch time range
    now = datetime.now()
    # Usa il massimo tra la data calcolata e il 1° gennaio 2024
    calculated_start_time = now - timedelta(days=data_limit_days)
    start_time = max(calculated_start_time, MIN_START_DATE)
    
    if last_date:
        # Se l'ultima data è prima del 1° gennaio 2024, parti dal 1° gennaio 2024
        if last_date < MIN_START_DATE:
            fetch_start_time = MIN_START_DATE
        else:
            # Se abbiamo dati ma non freschi, parti da un giorno prima dell'ultima data
            fetch_start_time = max(start_time, last_date - timedelta(days=1))
    else:
        fetch_start_time = start_time
    
    # Calcola il periodo di warmup per questo timeframe (per indicatori tecnici)
    # Ad esempio, per EMA200 abbiamo bisogno di almeno 200 candele prima
    warmup_candles = WARMUP_CANDLES.get(timeframe, 250)  # Default 250 candles
    
    # Converti il numero di candele in giorni in base al timeframe
    # Questo è un calcolo approssimativo
    if timeframe == '1d':
        warmup_days = warmup_candles
    elif timeframe == '4h':
        warmup_days = warmup_candles / 6  # circa 6 candele da 4h al giorno
    elif timeframe == '1h':
        warmup_days = warmup_candles / 24  # circa 24 candele da 1h al giorno
    elif timeframe == '30m':
        warmup_days = warmup_candles / 48  # circa 48 candele da 30m al giorno
    elif timeframe == '15m':
        warmup_days = warmup_candles / 96  # circa 96 candele da 15m al giorno
    elif timeframe == '5m':
        warmup_days = warmup_candles / 288  # circa 288 candele da 5m al giorno
    elif timeframe == '1m':
        warmup_days = warmup_candles / 1440  # circa 1440 candele da 1m al giorno
    else:
        warmup_days = 30  # valore di default conservativo
    
    # Se stiamo partendo dal 1° gennaio 2024, aggiungi il periodo di warmup
    if fetch_start_time == MIN_START_DATE:
        fetch_start_time = fetch_start_time - timedelta(days=int(warmup_days))
        logging.info(f"Aggiunti {warmup_days:.1f} giorni di warmup per {symbol} ({timeframe})")

    since = int(fetch_start_time.timestamp() * 1000)
    now_ms = int(now.timestamp() * 1000)
    ohlcv_data = []

    # Fetch data with progress bar
    with logging_redirect_tqdm():
        with tqdm(total=estimated_iterations(since, now_ms, timeframe), 
                 desc=f"{Fore.BLUE}Caricamento {symbol} ({timeframe}){Style.RESET_ALL}",
                 bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Style.RESET_ALL)) as pbar:
            current_since = since
            
            # Per salvare tutti i dati, inclusi quelli di warmup
            all_ohlcv_data = []
            
            while current_since < now_ms:
                try:
                    data_chunk = await fetch_with_retry(exchange, symbol, timeframe, current_since, 1000)
                    if not data_chunk:
                        break
                    
                    # Salva tutti i dati, inclusi quelli di warmup per il calcolo indicatori
                    all_ohlcv_data.extend(data_chunk)
                    
                    if data_chunk:
                        current_since = data_chunk[-1][0] + 1
                    pbar.update(1)
                except Exception as e:
                    logging.error(f"Fetch fallito dopo {OHLCV_FETCH_CONFIG['max_retries']} tentativi per {symbol} ({timeframe}): {e}")
                    break

    # Ora filtra per il salvataggio solo i dati dal 1° gennaio 2024 in poi
    min_timestamp_ms = int(MIN_START_DATE.timestamp() * 1000)
    
    # Dividi i dati in dati di warmup e dati da salvare
    warmup_data = [candle for candle in all_ohlcv_data if candle[0] < min_timestamp_ms]
    save_data = [candle for candle in all_ohlcv_data if candle[0] >= min_timestamp_ms]
    
    if warmup_data:
        logging.info(f"Scaricate {Fore.YELLOW}{len(warmup_data)}{Style.RESET_ALL} candele di warmup pre-2024 per {symbol} ({timeframe})")
    
    # Calcola indicatori usando TUTTI i dati (warmup + save)
    if all_ohlcv_data:
        # Prima salva temporaneamente tutti i dati (incluso warmup) per calcolo indicatori
        temp_result = save_ohlcv_data(symbol, timeframe, all_ohlcv_data, is_temp=True)
        if not temp_result[0]:
            logging.error(f"Errore nel salvataggio temporaneo dei dati per {symbol} ({timeframe})")
        
        # Ora salva definitivamente solo i dati dal 1° gennaio 2024
        if save_data:
            return save_ohlcv_data(symbol, timeframe, save_data)
        else:
            logging.warning(f"Nessun dato dal 1° gennaio 2024 per {symbol} ({timeframe})")
            return False, 0
    return False, 0
