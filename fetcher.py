# fetcher.py

import asyncio
import logging
import pandas as pd
from datetime import datetime, timedelta
from config import TIMEFRAME_DEFAULT, DATA_LIMIT_DAYS, TOP_ANALYSIS_CRYPTO
from termcolor import colored
import re
import concurrent.futures
from functools import partial
from db_manager import save_data
from tqdm import tqdm

# Numero di thread per le operazioni CPU-intensive
MAX_WORKERS = 16  # Aumentato significativamente per sfruttare più core
# Numero massimo di richieste API concorrenti
MAX_API_CONCURRENCY = 50  # Aumentato per massimizzare l'utilizzo della rete

async def fetch_markets(exchange):
    return await exchange.load_markets()

async def fetch_ticker_volume(exchange, symbol, semaphore):
    async with semaphore:
        try:
            ticker = await exchange.fetch_ticker(symbol)
            return symbol, ticker.get('quoteVolume')
        except Exception:
            return symbol, None

async def get_top_symbols(exchange, symbols, top_n=TOP_ANALYSIS_CRYPTO):
    print(f"Recupero volumi per {len(symbols)} simboli...")
    
    # Aumentato il numero di richieste simultanee
    semaphore = asyncio.Semaphore(MAX_API_CONCURRENCY)
    
    # Ottimizzazione: dividiamo i simboli in batch per elaborazione parallela
    batch_size = 100  # Aumentato per elaborare più simboli contemporaneamente
    batches = [symbols[i:i+batch_size] for i in range(0, len(symbols), batch_size)]
    
    # Utilizziamo as_completed per elaborare i risultati man mano che arrivano
    all_symbol_volumes = []
    pbar = tqdm(total=len(symbols), desc="Recupero volumi", ncols=80)
    
    # Creiamo tutti i task una volta sola
    all_tasks = [fetch_ticker_volume(exchange, symbol, semaphore) for symbol in symbols]
    
    # Utilizziamo as_completed per elaborare i risultati più velocemente
    for completed_task in asyncio.as_completed(all_tasks):
        symbol, volume = await completed_task
        if volume is not None:
            all_symbol_volumes.append((symbol, volume))
        pbar.update(1)
    
    pbar.close()
    
    # Ordina per volume in ordine decrescente
    all_symbol_volumes.sort(key=lambda x: x[1], reverse=True)
    top_symbols = [x[0] for x in all_symbol_volumes[:top_n]]
    
    print(f"Selezionati i top {len(top_symbols)} simboli per volume")
    return top_symbols

# Funzione per elaborare più simboli in parallelo con maggiore concorrenza
async def fetch_data_for_multiple_symbols(exchange, symbols, timeframe=TIMEFRAME_DEFAULT, limit=1000):
    print(f"Recupero dati in parallelo per {len(symbols)} simboli (timeframe {timeframe})...")
    
    # Aumentiamo la concorrenza
    semaphore = asyncio.Semaphore(MAX_API_CONCURRENCY)
    
    # Ottimizzazione: funzione per elaborare un singolo simbolo
    async def process_single_symbol(symbol):
        async with semaphore:
            df = await get_data_async(exchange, symbol, timeframe, limit)
            if df is not None:
                # Utilizziamo un thread separato per elaborare gli indicatori
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    df_with_indicators = await asyncio.get_event_loop().run_in_executor(
                        None, process_technical_indicators, df.copy()
                    )
                save_data(symbol, df_with_indicators, timeframe)
            return symbol, df
    
    # Creiamo tutti i task in una volta
    tasks = [process_single_symbol(symbol) for symbol in symbols]
    results = {}
    
    # Utilizziamo una barra di avanzamento
    pbar = tqdm(total=len(symbols), desc=f"Recupero dati {timeframe}", ncols=80)
    
    # Elaboriamo i risultati man mano che arrivano
    for completed_task in asyncio.as_completed(tasks):
        symbol, df = await completed_task
        results[symbol] = df
        pbar.update(1)
    
    pbar.close()
    
    return results

async def fetch_min_amounts(exchange, top_symbols, markets):
    min_amounts = {}
    for symbol in top_symbols:
        market = markets.get(symbol)
        if market and 'limits' in market and 'amount' in market['limits'] and 'min' in market['limits']['amount']:
            min_amounts[symbol] = market['limits']['amount']['min']
        else:
            min_amounts[symbol] = 1
    return min_amounts

async def get_data_async(exchange, symbol, timeframe=TIMEFRAME_DEFAULT, limit=1000):
    ohlcv_all = []
    since_dt = datetime.utcnow() - timedelta(days=DATA_LIMIT_DAYS)
    since = int(since_dt.timestamp() * 1000)
    current_time = int(datetime.utcnow().timestamp() * 1000)
    
    # Incrementiamo il limite per ridurre il numero di chiamate necessarie
    fetch_limit = min(1000, limit)
    iterations = 0
    
    # Usiamo un timeout per evitare che si blocchi in attesa troppo a lungo
    timeout = 10  # secondi
    
    while True:
        iterations += 1
        try:
            # Aggiungiamo un timeout alla chiamata API
            ohlcv_task = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=fetch_limit, since=since)
            ohlcv = await asyncio.wait_for(ohlcv_task, timeout=timeout)
        except asyncio.TimeoutError:
            # Riprova una volta in caso di timeout
            try:
                ohlcv = await asyncio.wait_for(exchange.fetch_ohlcv(symbol, timeframe=timeframe, 
                                              limit=fetch_limit, since=since), 
                                              timeout=timeout*2)  # Timeout doppio per il retry
            except Exception:
                break
        except Exception:
            break
            
        if not ohlcv or len(ohlcv) == 0:
            break
        
        ohlcv_all.extend(ohlcv)
        
        # Ottimizzazione: se abbiamo ricevuto meno del limite, non ci sono più dati
        if len(ohlcv) < fetch_limit:
            break
            
        last_timestamp = ohlcv[-1][0]
        if last_timestamp >= current_time:
            break
            
        # Avanza al timestamp successivo
        new_since = last_timestamp + 1
        if new_since <= since:
            break
        since = new_since
        
        # Pausa breve per non sovraccaricare l'API
        if iterations % 3 == 0:
            await asyncio.sleep(0.1)  # Ridotta per aumentare velocità

    if ohlcv_all:
        df = pd.DataFrame(ohlcv_all, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Rimuove i duplicati che potrebbero esserci a causa delle richieste sovrapposte
        df = df[~df.index.duplicated(keep='first')]
        
        # Ordina per timestamp
        df = df.sort_index()
        
        return df
    else:
        return None

# Funzione per elaborare gli indicatori tecnici in un thread separato
def process_technical_indicators(df):
    from data_utils import add_technical_indicators
    return add_technical_indicators(df.copy())