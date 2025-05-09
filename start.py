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
DEFAULT_BATCH_SIZE = 10
DEFAULT_CONCURRENCY = 5

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
    parser = argparse.ArgumentParser(
        description='Scarica dati OHLCV delle criptovalute da Bybit',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Parametri generali
    parser.add_argument(
        '-n', '--num-symbols',
        type=int,
        default=DEFAULT_TOP_SYMBOLS,
        help=f'Numero di criptovalute da scaricare'
    )
    
    parser.add_argument(
        '-d', '--days',
        type=int,
        default=DEFAULT_DAYS,
        help=f'Giorni di dati storici da scaricare'
    )
    
    parser.add_argument(
        '-t', '--timeframes',
        nargs='+',
        default=DEFAULT_TIMEFRAMES,
        choices=list(TIMEFRAME_CONFIG.keys()),
        help=f'Timeframes da scaricare'
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
        help='Esegui in modalità sequenziale invece che parallela'
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

        # Mostra le top criptovalute in una tabella formattata
        print("\n" + "="*80)
        print(f"{Back.BLUE}{Fore.WHITE}  TOP CRIPTOVALUTE PER VOLUME  {Style.RESET_ALL}")
        print("="*80)
        
        # Intestazione tabella
        print(f"{'#':4} {'Simbolo':20} {'Volume (USDT)':>25}")
        print("-"*60)
        
        # Mostra le prime 10 per riferimento
        for i, (symbol, volume) in enumerate(sorted(volumes.items(), key=lambda x: x[1], reverse=True)[:10]):
            # Alternare colori di sfondo per facilità di lettura
            bg_color = Back.BLACK if i % 2 == 0 else ""
            # Evidenzia in base alla posizione (TOP 3 in giallo, resto in bianco)
            symbol_color = Fore.YELLOW if i < 3 else Fore.WHITE
            volume_color = Fore.CYAN if i < 3 else Fore.WHITE
            
            print(f"{bg_color}{i+1:3} {symbol_color}{symbol:20} {volume_color}{volume:25,.2f}{Style.RESET_ALL}")
        
        print("="*80 + "\n")
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

async def fetch_data_parallel(symbols, timeframe, data_limit_days, max_concurrency=5, batch_size=DEFAULT_BATCH_SIZE):
    """Recupera dati per più simboli in parallelo per uno specifico timeframe."""
    exchange = await create_exchange()
    batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]
    
    # Dizionario per tracciare i risultati per simbolo
    results = {"completati": 0, "saltati": 0, "falliti": 0, "record_totali": 0}
    
    print(f"\n{Back.BLUE}{Fore.WHITE} DOWNLOAD PARALLELO: TIMEFRAME {timeframe} {Style.RESET_ALL}")
    print(f"Batch totali: {len(batches)}, Simboli: {len(symbols)}, Concorrenza: {max_concurrency}")
    print("-" * 60)

    try:
        for batch_idx, batch in enumerate(batches):
            print(f"\n{Fore.CYAN}Batch {batch_idx+1}/{len(batches)}{Style.RESET_ALL} - Timeframe {Fore.WHITE}{timeframe}{Style.RESET_ALL}")
            print(f"Simboli in questo batch: {', '.join([Fore.YELLOW + s + Style.RESET_ALL for s in batch])}")
            
            # Processa simboli in parallelo con limite di concorrenza
            tasks = []
            semaphore = asyncio.Semaphore(max_concurrency)
            
            # Utilizziamo una coda per organizzare i risultati
            result_queue = asyncio.Queue()
            
            async def process_symbol(sym):
                async with semaphore:
                    try:
                        result = await fetch_and_save_data(exchange, sym, timeframe, data_limit_days)
                        if result is None:
                            await result_queue.put((sym, "saltato", 0))
                            return
                            
                        success, count = result
                        if success:
                            await result_queue.put((sym, "completato", count))
                        else:
                            await result_queue.put((sym, "fallito", 0))
                    except Exception as e:
                        logging.error(f"Errore nell'elaborazione di {sym} ({timeframe}): {e}")
                        await result_queue.put((sym, "fallito", 0))
            
            # Crea task per ogni simbolo nel batch
            for symbol in batch:
                tasks.append(process_symbol(symbol))
            
            # Variabile per raccogliere i risultati del batch (spostata fuori dalla funzione interna)
            current_batch_results = []
            
            # Task per mostrare i risultati in tempo reale
            async def display_results():
                batch_count = len(batch)
                completed = 0
                
                while completed < batch_count:
                    sym, status, count = await result_queue.get()
                    completed += 1
                    
                    if status == "completato":
                        results["completati"] += 1
                        results["record_totali"] += count
                        status_str = f"{Fore.GREEN}✓ Completato{Style.RESET_ALL}"
                    elif status == "saltato":
                        results["saltati"] += 1
                        status_str = f"{Fore.BLUE}↷ Saltato{Style.RESET_ALL}"
                    else:
                        results["falliti"] += 1
                        status_str = f"{Fore.RED}✗ Fallito{Style.RESET_ALL}"
                    
                    current_batch_results.append((sym, status_str, count))
                    
                    # Mostra progressione ogni 5 simboli o alla fine
                    if completed % 5 == 0 or completed == batch_count:
                        print(f"\nProgresso batch: {completed}/{batch_count} simboli elaborati")
                        # Mostra gli ultimi 5 risultati elaborati
                        recent_results = current_batch_results[-min(5, len(current_batch_results)):]
                        for s, st, c in recent_results:
                            count_str = f"{c} record" if c > 0 else ""
                            print(f"  • {Fore.YELLOW}{s:<20}{Style.RESET_ALL} {st:<25} {count_str}")
            
            # Esecuzione parallela
            results_task = asyncio.create_task(display_results())
            await asyncio.gather(*tasks)
            await results_task
            
            # Mostra riepilogo batch
            print(f"\n{Fore.CYAN}Riepilogo batch {batch_idx+1}:{Style.RESET_ALL}")
            completati = sum(1 for _, st, _ in current_batch_results if 'Completato' in st)
            saltati = sum(1 for _, st, _ in current_batch_results if 'Saltato' in st)
            falliti = sum(1 for _, st, _ in current_batch_results if 'Fallito' in st)
            print(f"  • Completati: {Fore.GREEN}{completati}{Style.RESET_ALL}")
            print(f"  • Saltati: {Fore.BLUE}{saltati}{Style.RESET_ALL}")
            print(f"  • Falliti: {Fore.RED}{falliti}{Style.RESET_ALL}")
        
        # Riepilogo finale per timeframe
        print(f"\n{Back.BLUE}{Fore.WHITE} RIEPILOGO TIMEFRAME {timeframe} {Style.RESET_ALL}")
        print(f"  • Simboli completati: {Fore.GREEN}{results['completati']}{Style.RESET_ALL}")
        print(f"  • Simboli saltati: {Fore.BLUE}{results['saltati']}{Style.RESET_ALL}")
        print(f"  • Simboli falliti: {Fore.RED}{results['falliti']}{Style.RESET_ALL}")
        print(f"  • Record totali salvati: {Fore.CYAN}{results['record_totali']:,}{Style.RESET_ALL}")
        print("-" * 60)
    except Exception as e:
        logging.error(f"Errore durante il download parallelo per {timeframe}: {e}")
        import traceback
        logging.error(traceback.format_exc())
    finally:
        # Assicurati che l'exchange venga chiuso correttamente
        try:
            await exchange.close()
        except Exception as e:
            logging.error(f"Errore nella chiusura dell'exchange: {e}")

    return results

async def fetch_data_sequential(symbols, timeframe, data_limit_days, batch_size=DEFAULT_BATCH_SIZE):
    """Recupera dati per più simboli sequenzialmente per uno specifico timeframe."""
    exchange = await create_exchange()
    batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]
    
    # Dizionario per tracciare i risultati per simbolo
    results = {"completati": 0, "saltati": 0, "falliti": 0, "record_totali": 0}
    
    print(f"\n{Back.BLUE}{Fore.WHITE} DOWNLOAD SEQUENZIALE: TIMEFRAME {timeframe} {Style.RESET_ALL}")
    print(f"Batch totali: {len(batches)}, Simboli: {len(symbols)}")
    print("-" * 60)

    try:
        for batch_idx, batch in enumerate(batches):
            print(f"\n{Fore.CYAN}Batch {batch_idx+1}/{len(batches)}{Style.RESET_ALL} - Timeframe {Fore.WHITE}{timeframe}{Style.RESET_ALL}")
            print(f"Simboli in questo batch: {', '.join([Fore.YELLOW + s + Style.RESET_ALL for s in batch])}")
            
            batch_results = []
            for symbol in batch:
                try:
                    result = await fetch_and_save_data(exchange, symbol, timeframe, data_limit_days)
                    
                    if result is None:
                        results["saltati"] += 1
                        status_str = f"{Fore.BLUE}↷ Saltato{Style.RESET_ALL}"
                        batch_results.append((symbol, status_str, 0))
                        print(f"  • {Fore.YELLOW}{symbol:<20}{Style.RESET_ALL} {status_str}")
                        continue
                    
                    success, count = result
                    if success:
                        results["completati"] += 1
                        results["record_totali"] += count
                        status_str = f"{Fore.GREEN}✓ Completato{Style.RESET_ALL}"
                        batch_results.append((symbol, status_str, count))
                        print(f"  • {Fore.YELLOW}{symbol:<20}{Style.RESET_ALL} {status_str:<25} {count} record")
                    else:
                        results["falliti"] += 1
                        status_str = f"{Fore.RED}✗ Fallito{Style.RESET_ALL}"
                        batch_results.append((symbol, status_str, 0))
                        print(f"  • {Fore.YELLOW}{symbol:<20}{Style.RESET_ALL} {status_str}")
                        
                except Exception as e:
                    logging.error(f"Errore nell'elaborazione di {symbol} ({timeframe}): {e}")
                    results["falliti"] += 1
                    status_str = f"{Fore.RED}✗ Fallito (Errore){Style.RESET_ALL}"
                    batch_results.append((symbol, status_str, 0))
                    print(f"  • {Fore.YELLOW}{symbol:<20}{Style.RESET_ALL} {status_str}")
            
            # Mostra riepilogo batch
            print(f"\n{Fore.CYAN}Riepilogo batch {batch_idx+1}:{Style.RESET_ALL}")
            completati = sum(1 for _, st, _ in batch_results if 'Completato' in st)
            saltati = sum(1 for _, st, _ in batch_results if 'Saltato' in st)
            falliti = sum(1 for _, st, _ in batch_results if 'Fallito' in st)
            print(f"  • Completati: {Fore.GREEN}{completati}{Style.RESET_ALL}")
            print(f"  • Saltati: {Fore.BLUE}{saltati}{Style.RESET_ALL}")
            print(f"  • Falliti: {Fore.RED}{falliti}{Style.RESET_ALL}")
        
        # Riepilogo finale per timeframe
        print(f"\n{Back.BLUE}{Fore.WHITE} RIEPILOGO TIMEFRAME {timeframe} {Style.RESET_ALL}")
        print(f"  • Simboli completati: {Fore.GREEN}{results['completati']}{Style.RESET_ALL}")
        print(f"  • Simboli saltati: {Fore.BLUE}{results['saltati']}{Style.RESET_ALL}")
        print(f"  • Simboli falliti: {Fore.RED}{results['falliti']}{Style.RESET_ALL}")
        print(f"  • Record totali salvati: {Fore.CYAN}{results['record_totali']:,}{Style.RESET_ALL}")
        print("-" * 60)
    except Exception as e:
        logging.error(f"Errore durante il download sequenziale per {timeframe}: {e}")
        import traceback
        logging.error(traceback.format_exc())
    finally:
        # Assicurati che l'exchange venga chiuso correttamente
        try:
            await exchange.close()
        except Exception as e:
            logging.error(f"Errore nella chiusura dell'exchange: {e}")

    return results

async def process_timeframe(timeframe, top_symbols, days, concurrency, batch_size, use_sequential=False):
    """Processa un singolo timeframe."""
    print("\n" + "-"*80)
    print(f"{Back.CYAN}{Fore.WHITE}  ELABORAZIONE TIMEFRAME: {timeframe}  {Style.RESET_ALL}")
    print("-"*80)
    
    # Esegue il download in modalità sequenziale o parallela
    if use_sequential:
        results = await fetch_data_sequential(top_symbols, timeframe, days, batch_size=batch_size)
    else:
        results = await fetch_data_parallel(top_symbols, timeframe, days, max_concurrency=concurrency, batch_size=batch_size)
    
    logging.info(f"Completato timeframe {Fore.CYAN}{timeframe}{Style.RESET_ALL}")
    return timeframe, results

async def main():
    """Punto di ingresso principale per il recupero dati delle criptovalute."""
    # Registra il tempo di inizio
    start_time = datetime.now()
    
    args = parse_arguments()
    mode = "SEQUENZIALE" if args.sequential else "PARALLELA"
    
    # Header generale
    print("\n" + "="*80)
    print(f"{Back.BLUE}{Fore.WHITE}  RECUPERO DATI OHLCV CRIPTOVALUTE (MODALITÀ {mode})  {Style.RESET_ALL}")
    print("="*80)
    print(f"  • Criptovalute da elaborare: {Fore.YELLOW}{args.num_symbols}{Style.RESET_ALL}")
    print(f"  • Periodo storico: {Fore.GREEN}{args.days} giorni{Style.RESET_ALL}")
    print(f"  • Timeframes richiesti: {Fore.GREEN}{', '.join(args.timeframes)}{Style.RESET_ALL}")
    print(f"  • Batch size: {Fore.YELLOW}{args.batch_size}{Style.RESET_ALL}")
    if not args.sequential:
        print(f"  • Concorrenza: {Fore.YELLOW}{args.concurrency}{Style.RESET_ALL} download paralleli per batch")
    print(f"  • Data e ora inizio: {Fore.CYAN}{start_time.strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}")
    print("="*80 + "\n")

    # Inizializza il database
    init_data_tables(args.timeframes)
    logging.info(f"Database inizializzato con tabelle per i timeframe: {Fore.GREEN}{', '.join(args.timeframes)}{Style.RESET_ALL}")

    # Variabili per raccogliere statistiche complessive
    all_timeframe_results = {}
    total_records_saved = 0
    grand_total_symbols = {"completati": 0, "saltati": 0, "falliti": 0}
    
    try:
        # Fase 1: Recupero simboli
        print(f"\n{Back.BLUE}{Fore.WHITE}  FASE 1: RICERCA CRIPTOVALUTE  {Style.RESET_ALL}")
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
        
        # Fase 2: Download per tutti i timeframe
        print(f"\n{Back.BLUE}{Fore.WHITE}  FASE 2: DOWNLOAD DATI OHLCV  {Style.RESET_ALL}")
        if args.sequential:
            logging.info(f"{Fore.YELLOW}Modalità sequenziale attivata. Elaborazione timeframe uno alla volta.{Style.RESET_ALL}")
            for timeframe in args.timeframes:
                timeframe, results = await process_timeframe(timeframe, top_symbols, args.days, 
                                                         args.concurrency, args.batch_size, True)
                all_timeframe_results[timeframe] = results
                # Aggiorna statistiche totali
                grand_total_symbols["completati"] += results["completati"]
                grand_total_symbols["saltati"] += results["saltati"]
                grand_total_symbols["falliti"] += results["falliti"]
                total_records_saved += results["record_totali"]
        else:
            logging.info(f"{Fore.YELLOW}Modalità parallela attivata. Concorrenza massima per simbolo: {args.concurrency}{Style.RESET_ALL}")
            timeframe_tasks = []
            for timeframe in args.timeframes:
                timeframe_tasks.append(process_timeframe(timeframe, top_symbols, args.days, 
                                                       args.concurrency, args.batch_size))
            
            # Esegui tutti i timeframe in parallelo e raccoglie risultati
            results = await asyncio.gather(*timeframe_tasks)
            for tf, res in results:
                all_timeframe_results[tf] = res
                # Aggiorna statistiche totali
                grand_total_symbols["completati"] += res["completati"]
                grand_total_symbols["saltati"] += res["saltati"]
                grand_total_symbols["falliti"] += res["falliti"]
                total_records_saved += res["record_totali"]

        # Calcola il tempo totale di esecuzione
        end_time = datetime.now()
        execution_time = end_time - start_time
        hours, remainder = divmod(execution_time.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

        # Resoconto finale
        print("\n" + "="*80)
        print(f"{Back.GREEN}{Fore.BLACK}  RESOCONTO FINALE RACCOLTA DATI  {Style.RESET_ALL}")
        print("="*80)
        print(f"  • Criptovalute elaborate: {Fore.YELLOW}{len(top_symbols)}{Style.RESET_ALL}")
        print(f"  • Database: {Fore.BLUE}{os.path.abspath(DB_FILE)}{Style.RESET_ALL}")
        print(f"  • Tempo totale esecuzione: {Fore.CYAN}{time_str}{Style.RESET_ALL}")
        print()
        
        # Tabella dei risultati per timeframe
        print(f"{Back.WHITE}{Fore.BLACK}  STATISTICHE PER TIMEFRAME  {Style.RESET_ALL}")
        print(f"{'Timeframe':^10} | {'Completati':^12} | {'Saltati':^12} | {'Falliti':^12} | {'Record Salvati':^15}")
        print("-" * 70)
        for tf, res in sorted(all_timeframe_results.items()):
            print(f"{Fore.CYAN}{tf:^10}{Style.RESET_ALL} | " + 
                  f"{Fore.GREEN}{res['completati']:^12}{Style.RESET_ALL} | " + 
                  f"{Fore.BLUE}{res['saltati']:^12}{Style.RESET_ALL} | " + 
                  f"{Fore.RED}{res['falliti']:^12}{Style.RESET_ALL} | " + 
                  f"{Fore.YELLOW}{res['record_totali']:^15,}{Style.RESET_ALL}")
        
        # Totali
        print("-" * 70)
        average_per_timeframe = round(grand_total_symbols["completati"] / len(args.timeframes)) if args.timeframes else 0
        print(f"{'TOTALE':^10} | " + 
              f"{Fore.GREEN}{grand_total_symbols['completati']:^12}{Style.RESET_ALL} | " + 
              f"{Fore.BLUE}{grand_total_symbols['saltati']:^12}{Style.RESET_ALL} | " + 
              f"{Fore.RED}{grand_total_symbols['falliti']:^12}{Style.RESET_ALL} | " + 
              f"{Fore.YELLOW}{total_records_saved:^15,}{Style.RESET_ALL}")
        print()
        
        # Informazioni aggiuntive
        print(f"Media simboli completati per timeframe: {Fore.GREEN}{average_per_timeframe}{Style.RESET_ALL}")
        print(f"Record medi per simbolo: {Fore.YELLOW}{round(total_records_saved/max(1, grand_total_symbols['completati'])):,}{Style.RESET_ALL}")
        print(f"Inizio: {Fore.CYAN}{start_time.strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}")
        print(f"Fine:   {Fore.CYAN}{end_time.strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}")
        print("="*80 + "\n")

    except Exception as e:
        logging.error(f"Errore: {e}")
        import traceback
        logging.error(traceback.format_exc())
    finally:
        logging.info("Recupero dati completato.")

if __name__ == "__main__":
    asyncio.run(main())
