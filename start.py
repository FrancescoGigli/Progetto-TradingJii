#!/usr/bin/env python3
"""
Cryptocurrency Data Fetcher
==========================

Questo script scarica i dati OHLCV delle criptovalute con maggior volume da Bybit.
Controlla se i dati sono aggiornati prima di scaricarli e scarica solo i dati necessari.

Timeframes: 5m, 15m, 30m, 1h
Periodo: 100 giorni di dati storici
"""

import os
import sys
import asyncio
import logging
import sqlite3
import argparse
from datetime import datetime, timedelta
import ccxt.async_support as ccxt_async
from colorama import init, Fore, Style, Back
from dotenv import load_dotenv
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# Carica le variabili d'ambiente dal file .env
load_dotenv()

# Inizializza colorama
init(autoreset=True)

# Imposta la policy di event loop per Windows se necessario
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Configurazione di default
DEFAULT_TOP_SYMBOLS = 100
DEFAULT_DAYS = 100
DEFAULT_TIMEFRAMES = ['5m', '15m']
DB_FILE = 'crypto_data.db'
BATCH_SIZE = 10

# Configurazione timeframe
TIMEFRAME_CONFIG = {
    '1m': {'max_age': timedelta(minutes=5), 'ms': 60 * 1000},
    '5m': {'max_age': timedelta(minutes=15), 'ms': 5 * 60 * 1000},
    '15m': {'max_age': timedelta(hours=1), 'ms': 15 * 60 * 1000},
    '30m': {'max_age': timedelta(hours=2), 'ms': 30 * 60 * 1000},
    '1h': {'max_age': timedelta(hours=4), 'ms': 60 * 60 * 1000},
    '4h': {'max_age': timedelta(hours=12), 'ms': 4 * 60 * 60 * 1000},
    '1d': {'max_age': timedelta(days=2), 'ms': 24 * 60 * 60 * 1000}
}

# Logger con colori personalizzati
class ColoredFormatter(logging.Formatter):
    FORMATS = {
        logging.DEBUG: Fore.CYAN + "%(asctime)s %(levelname)s: %(message)s" + Style.RESET_ALL,
        logging.INFO: Fore.GREEN + "%(asctime)s %(levelname)s: %(message)s" + Style.RESET_ALL,
        logging.WARNING: Fore.YELLOW + "%(asctime)s %(levelname)s: %(message)s" + Style.RESET_ALL,
        logging.ERROR: Fore.RED + "%(asctime)s %(levelname)s: %(message)s" + Style.RESET_ALL,
        logging.CRITICAL: Fore.RED + Back.WHITE + "%(asctime)s %(levelname)s: %(message)s" + Style.RESET_ALL
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)

# Inizializza il logging con colori
handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter())
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.addHandler(handler)

def parse_arguments():
    """Analizza gli argomenti da linea di comando."""
    parser = argparse.ArgumentParser(description='Scarica dati OHLCV delle criptovalute da Bybit')
    
    parser.add_argument(
        '-n', '--num-symbols',
        type=int,
        default=DEFAULT_TOP_SYMBOLS,
        help=f'Numero di criptovalute da scaricare (default: {DEFAULT_TOP_SYMBOLS})'
    )
    
    parser.add_argument(
        '-d', '--days',
        type=int,
        default=DEFAULT_DAYS,
        help=f'Giorni di dati storici da scaricare (default: {DEFAULT_DAYS})'
    )
    
    parser.add_argument(
        '-t', '--timeframes',
        nargs='+',
        default=DEFAULT_TIMEFRAMES,
        choices=list(TIMEFRAME_CONFIG.keys()),
        help=f'Timeframes da scaricare (default: {DEFAULT_TIMEFRAMES})'
    )
    
    return parser.parse_args()

async def create_exchange():
    """Crea e inizializza l'exchange Bybit."""
    exchange = ccxt_async.bybit({
        'apiKey': os.environ.get('BYBIT_API_KEY', ''),
        'secret': os.environ.get('BYBIT_API_SECRET', ''),
        'enableRateLimit': True,
        'timeout': 30000,
        'options': {
            'defaultType': 'swap',
            'adjustForTimeDifference': True
        }
    })
    await exchange.load_markets()
    return exchange

async def fetch_markets(exchange):
    """Recupera tutti i mercati disponibili dall'exchange."""
    try:
        return {
            market['symbol']: market
            for market in await exchange.fetch_markets()
            if market.get('quote') == 'USDT' and market.get('active') and market.get('type') == 'swap'
        }
    except Exception as e:
        logging.error(f"Errore nel recupero dei mercati: {e}")
        return {}

async def get_top_symbols(exchange, symbols, top_n=100):
    """Ottieni le prime n criptovalute per volume di trading."""
    try:
        logging.info(f"Recupero dati di volume per {len(symbols)} coppie USDT...")
        volumes = {}

        with logging_redirect_tqdm():
            with tqdm(total=len(symbols), desc=f"{Fore.BLUE}Ricerca delle coppie USDT con maggior volume{Style.RESET_ALL}", 
                     bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Style.RESET_ALL)) as pbar:
                for symbol in symbols:
                    try:
                        ticker = await exchange.fetch_ticker(symbol)
                        volumes[symbol] = ticker.get('quoteVolume', 0) if ticker else 0
                    except Exception as e:
                        logging.error(f"Errore nel recupero del volume per {symbol}: {e}")
                        volumes[symbol] = 0
                    pbar.update(1)

        top_symbols = [s[0] for s in sorted(volumes.items(), key=lambda x: x[1], reverse=True)[:top_n]]
        logging.info(f"Trovate {Fore.YELLOW}{len(top_symbols)}{Style.RESET_ALL} coppie USDT con maggior volume")

        # Mostra le prime 5 per riferimento
        for i, (symbol, volume) in enumerate(sorted(volumes.items(), key=lambda x: x[1], reverse=True)[:5]):
            logging.info(f"Top {i+1}: {Fore.YELLOW}{symbol}{Style.RESET_ALL} - Volume: {Fore.CYAN}{volume:,.2f}{Style.RESET_ALL} USDT")

        return top_symbols
    except Exception as e:
        logging.error(f"Errore nel recupero delle coppie con maggior volume: {e}")
        return []

def init_data_tables(timeframes):
    """Inizializza le tabelle del database per ogni timeframe."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        for timeframe in timeframes:
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS data_{timeframe} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    UNIQUE(symbol, timestamp)
                )
            """)
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{timeframe}_symbol_timestamp
                ON data_{timeframe} (symbol, timestamp)
            """)

def get_timestamp_range(cursor, table_name, symbol):
    """Ottieni il primo e l'ultimo timestamp disponibile per un simbolo."""
    cursor.execute(f"""
        SELECT MIN(timestamp), MAX(timestamp) 
        FROM {table_name}
        WHERE symbol = ?
    """, (symbol,))
    result = cursor.fetchone()
    if result and result[0] and result[1]:
        return (
            datetime.strptime(result[0], '%Y-%m-%dT%H:%M:%S'),
            datetime.strptime(result[1], '%Y-%m-%dT%H:%M:%S')
        )
    return None, None

async def fetch_and_save_data(exchange, symbol, timeframe, data_limit_days):
    """Recupera e salva i dati OHLCV per uno specifico simbolo e timeframe."""
    table_name = f"data_{timeframe}"
    now = datetime.now()
    start_time = now - timedelta(days=data_limit_days)

    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        first_date, last_date = get_timestamp_range(cursor, table_name, symbol)

        if last_date:
            time_diff = now - last_date
            if time_diff < TIMEFRAME_CONFIG[timeframe]['max_age']:
                logging.info(f"Saltato {Fore.YELLOW}{symbol}{Style.RESET_ALL} ({timeframe}): Dati recenti già esistenti")
                logging.info(f"  • Prima data: {Fore.CYAN}{first_date.strftime('%Y-%m-%d %H:%M')}{Style.RESET_ALL}")
                logging.info(f"  • Ultima data: {Fore.CYAN}{last_date.strftime('%Y-%m-%d %H:%M')}{Style.RESET_ALL}")
                return None
            fetch_start_time = max(start_time, last_date - timedelta(days=1))
        else:
            fetch_start_time = start_time

        since = int(fetch_start_time.timestamp() * 1000)
        now_ms = int(now.timestamp() * 1000)
        ohlcv_data = []

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

        if ohlcv_data:
            records = [(symbol, datetime.fromtimestamp(r[0]/1000).strftime('%Y-%m-%dT%H:%M:%S'), *r[1:]) for r in ohlcv_data]
            cursor.executemany(f"""
                INSERT OR REPLACE INTO {table_name}
                (symbol, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, records)
            conn.commit()

            first_date, last_date = get_timestamp_range(cursor, table_name, symbol)
            if first_date and last_date:
                logging.info(f"Completato {Fore.YELLOW}{symbol}{Style.RESET_ALL} ({timeframe}) - salvati {Fore.GREEN}{len(records)}{Style.RESET_ALL} record")
                logging.info(f"  • Prima data: {Fore.CYAN}{first_date.strftime('%Y-%m-%d %H:%M')}{Style.RESET_ALL}")
                logging.info(f"  • Ultima data: {Fore.CYAN}{last_date.strftime('%Y-%m-%d %H:%M')}{Style.RESET_ALL}")

            return True, len(records)
        return False, 0

def estimated_iterations(since, now_ms, timeframe):
    """Calcola il numero stimato di iterazioni per il progress bar."""
    time_diff_ms = now_ms - since
    return max(1, int(time_diff_ms / TIMEFRAME_CONFIG[timeframe]['ms'] / 1000))

async def fetch_data_sequential(symbols, timeframe, data_limit_days):
    """Recupera dati per più simboli sequenzialmente per uno specifico timeframe."""
    exchange = await create_exchange()
    batches = [symbols[i:i + BATCH_SIZE] for i in range(0, len(symbols), BATCH_SIZE)]

    for batch_idx, batch in enumerate(batches):
        logging.info(f"Elaborazione batch {Fore.YELLOW}{batch_idx+1}/{len(batches)}{Style.RESET_ALL} per timeframe {Fore.CYAN}{timeframe}{Style.RESET_ALL} ({len(batch)} simboli)...")
        
        for symbol in batch:
            try:
                result = await fetch_and_save_data(exchange, symbol, timeframe, data_limit_days)
                if result is None:
                    continue
                success, count = result
                if not success:
                    logging.warning(f"Impossibile recuperare dati per {Fore.YELLOW}{symbol}{Style.RESET_ALL} ({timeframe})")
            except Exception as e:
                logging.error(f"Errore nell'elaborazione di {symbol} ({timeframe}): {e}")

    await exchange.close()

async def main():
    """Punto di ingresso principale per il recupero dati delle criptovalute."""
    args = parse_arguments()
    
    print("\n" + "="*80)
    print(f"{Fore.CYAN}  RECUPERO DATI OHLCV CRIPTOVALUTE{Style.RESET_ALL}")
    print(f"  Recupero delle {Fore.YELLOW}{args.num_symbols}{Style.RESET_ALL} criptovalute principali con {Fore.YELLOW}{args.days}{Style.RESET_ALL} giorni di dati")
    print(f"  Timeframes: {Fore.GREEN}{', '.join(args.timeframes)}{Style.RESET_ALL}")
    print("="*80 + "\n")

    init_data_tables(args.timeframes)
    logging.info(f"Database inizializzato con tabelle per i timeframe: {Fore.GREEN}{', '.join(args.timeframes)}{Style.RESET_ALL}")

    try:
        async_exchange = await create_exchange()
        markets = await fetch_markets(async_exchange)
        
        if not markets:
            logging.error("Nessun mercato trovato. Controlla la tua connessione internet e le credenziali API.")
            return

        all_symbols = list(markets.keys())
        top_symbols = await get_top_symbols(async_exchange, all_symbols, top_n=args.num_symbols)
        await async_exchange.close()

        if not top_symbols:
            logging.error("Impossibile ottenere i simboli con maggior volume. Utilizzo di tutti i simboli disponibili.")
            top_symbols = all_symbols[:args.num_symbols]

        for i, timeframe in enumerate(args.timeframes):
            print("\n" + "-"*60)
            print(f"{Fore.CYAN}  TIMEFRAME {i+1}/{len(args.timeframes)}: {timeframe}{Style.RESET_ALL}")
            print("-"*60)
            await fetch_data_sequential(top_symbols, timeframe, args.days)

        print("\n" + "="*80)
        print(f"{Fore.CYAN}  RACCOLTA DATI COMPLETATA{Style.RESET_ALL}")
        print(f"  • Scaricati dati per {Fore.YELLOW}{len(top_symbols)}{Style.RESET_ALL} criptovalute")
        print(f"  • Timeframes: {Fore.GREEN}{', '.join(args.timeframes)}{Style.RESET_ALL}")
        print(f"  • Database: {Fore.BLUE}{os.path.abspath(DB_FILE)}{Style.RESET_ALL}")
        print("="*80 + "\n")

    except Exception as e:
        logging.error(f"Errore: {e}")
        import traceback
        logging.error(traceback.format_exc())
    finally:
        logging.info("Recupero dati completato.")

if __name__ == "__main__":
    asyncio.run(main())
