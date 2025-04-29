# fetcher.py - corrected
import asyncio
import logging
import pandas as pd
import random
import time
from datetime import datetime, timedelta
from config import TIMEFRAME_DEFAULT, DATA_LIMIT_DAYS, TOP_ANALYSIS_CRYPTO
from termcolor import colored

# Rate limiting e backoff esponenziale
MAX_RETRIES = 5
BASE_DELAY = 0.5  # Delay base in secondi
REQUEST_DELAY = 0.25  # Ritardo minimo tra richieste sequenziali
MAX_CONCURRENT_REQUESTS = 5  # Massimo numero di richieste concorrenti

# Semaforo per limitare richieste concorrenti
api_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

async def fetch_with_backoff(fetch_func, *args, **kwargs):
    """Wrapper per richieste API con backoff esponenziale."""
    retry = 0
    while retry < MAX_RETRIES:
        try:
            # Acquisisci il semaforo per limitare le richieste concorrenti
            async with api_semaphore:
                # Aggiungi un ritardo casuale per evitare burst di richieste
                await asyncio.sleep(REQUEST_DELAY + random.random() * 0.2)
                return await fetch_func(*args, **kwargs)
        except Exception as e:
            retry += 1
            if "throttle" in str(e).lower() or "rate limit" in str(e).lower():
                # Backoff esponenziale con jitter
                delay = BASE_DELAY * (2 ** retry) + random.random()
                logging.warning(f"Rate limit hit, retrying in {delay:.2f}s... ({retry}/{MAX_RETRIES})")
                await asyncio.sleep(delay)
            else:
                if retry >= MAX_RETRIES:
                    logging.error(f"Failed after {MAX_RETRIES} retries: {e}")
                    raise
                # Per altri errori, ritenta con un backoff più breve
                delay = BASE_DELAY * retry + random.random()
                logging.warning(f"Error: {e}, retrying in {delay:.2f}s... ({retry}/{MAX_RETRIES})")
                await asyncio.sleep(delay)
    
    raise Exception(f"Failed after {MAX_RETRIES} retries")

async def fetch_markets(exchange):
    return await fetch_with_backoff(exchange.load_markets)

async def fetch_ticker_volume(exchange, symbol):
    try:
        ticker = await fetch_with_backoff(exchange.fetch_ticker, symbol)
        return symbol, ticker.get('quoteVolume')
    except Exception as e:
        logging.error(f"Error fetching ticker volume for {symbol}: {e}")
        return symbol, None

async def get_top_symbols(exchange, symbols, top_n=TOP_ANALYSIS_CRYPTO):
    # Dividi i simboli in batch per evitare troppe richieste simultanee
    batch_size = MAX_CONCURRENT_REQUESTS * 2
    all_results = []
    
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        logging.info(f"Fetching batch {i//batch_size + 1}/{(len(symbols) + batch_size - 1)//batch_size} of symbols ({len(batch)} symbols)")
        
        tasks = [fetch_ticker_volume(exchange, symbol) for symbol in batch]
        batch_results = await asyncio.gather(*tasks)
        all_results.extend(batch_results)
        
        # Breve pausa tra i batch
        if i + batch_size < len(symbols):
            await asyncio.sleep(1)
    
    symbol_volumes = [(symbol, volume) for symbol, volume in all_results if volume is not None]
    symbol_volumes.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in symbol_volumes[:top_n]]

async def fetch_min_amounts(exchange, top_symbols, markets):
    min_amounts = {}
    for symbol in top_symbols:
        market = markets.get(symbol)
        min_amounts[symbol] = (
            market.get('limits', {}).get('amount', {}).get('min', 1) if market else 1
        )
    return min_amounts

async def get_data_async(exchange, symbol, timeframe=TIMEFRAME_DEFAULT, limit=1000):
    """Scarica OHLCV grezzi con gestione del rate limit."""
    ohlcv_all = []
    since_dt = datetime.utcnow() - timedelta(days=DATA_LIMIT_DAYS)
    since = int(since_dt.timestamp() * 1000)
    current_time = int(datetime.utcnow().timestamp() * 1000)

    while True:
        try:
            ohlcv = await fetch_with_backoff(
                exchange.fetch_ohlcv,
                symbol, 
                timeframe=timeframe, 
                limit=limit, 
                since=since
            )
        except Exception as e:
            logging.error(f"Error fetching ohlcv for {symbol}: {e}")
            # Se fallisce dopo tutti i tentativi, ritorna eventuali dati raccolti 
            # o None se non ce ne sono
            break
            
        if not ohlcv:
            break

        ohlcv_all.extend(ohlcv)
        last_timestamp = ohlcv[-1][0]
        if last_timestamp >= current_time:
            break
        new_since = last_timestamp + 1
        if new_since <= since:
            break
        since = new_since

    if not ohlcv_all:
        logging.info(f"Nessun dato disponibile per {symbol} negli ultimi {DATA_LIMIT_DAYS} giorni.")
        return None

    df = pd.DataFrame(
        ohlcv_all,
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    last_date = df.index[-1].strftime('%Y-%m-%d')
    logging.info(
        colored(
            f"Fetched {len(df)} candlestick for {symbol} (TF: {timeframe}) from {since_dt.strftime('%Y-%m-%d')} to {last_date}.",
            'cyan'
        )
    )
    return df

async def fetch_and_save_data(exchange, symbol, timeframe=TIMEFRAME_DEFAULT, limit=1000):
    """Ritorna DataFrame con indicatori tecnici."""
    df = await get_data_async(exchange, symbol, timeframe, limit)
    if df is None:
        return None
    from data_utils import add_technical_indicators
    return add_technical_indicators(df.copy())
