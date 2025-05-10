#!/usr/bin/env python3
"""
Data fetcher module for TradingJii

Handles OHLCV data fetching from the exchange.
"""

import logging
import asyncio
from datetime import datetime, timedelta
from colorama import Fore, Style
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from modules.utils.config import TIMEFRAME_CONFIG
from modules.data.db_manager import check_data_freshness, save_ohlcv_data

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

async def fetch_ohlcv_data(exchange, symbol, timeframe, data_limit_days):
    """
    Fetch OHLCV data for a specific symbol and timeframe.
    
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
    if is_fresh:
        return None
        
    # Determine data fetch time range
    now = datetime.now()
    start_time = now - timedelta(days=data_limit_days)
    
    if last_date:
        # If we have data but it's not fresh, start from one day before last date
        fetch_start_time = max(start_time, last_date - timedelta(days=1))
    else:
        fetch_start_time = start_time

    since = int(fetch_start_time.timestamp() * 1000)
    now_ms = int(now.timestamp() * 1000)
    ohlcv_data = []

    # Fetch data with progress bar
    with logging_redirect_tqdm():
        with tqdm(total=estimated_iterations(since, now_ms, timeframe), 
                 desc=f"{Fore.BLUE}Caricamento {symbol} ({timeframe}){Style.RESET_ALL}",
                 bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Style.RESET_ALL)) as pbar:
            current_since = since
            while current_since < now_ms:
                try:
                    data_chunk = await exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=1000)
                    if not data_chunk:
                        break
                    ohlcv_data.extend(data_chunk)
                    if data_chunk:
                        current_since = data_chunk[-1][0] + 1
                    pbar.update(1)
                except Exception as e:
                    logging.error(f"Errore nel recupero dei dati OHLCV per {symbol} ({timeframe}): {e}")
                    break

    # Save data to database
    if ohlcv_data:
        return save_ohlcv_data(symbol, timeframe, ohlcv_data)
    return False, 0
