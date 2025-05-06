#!/usr/bin/env python3
# volatility_pipeline.py

import asyncio
import logging
import argparse
import os
import sys
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
from colorama import init, Fore, Style, Back
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(project_root)

# Import modules from the project
# Removed dependency on main.py to avoid import error
from src.core.fetcher import create_exchange, fetch_markets, get_top_symbols, fetch_and_save_data
from src.data.volatility_utils import calculate_volatility_rate
from src.data.subseries_utils import get_all_subseries_with_categories, binary_to_pattern
from src.data.db_manager import (
    init_data_tables, save_data, check_data_freshness, get_symbol_data, 
    get_top_categories, get_category_transitions, save_subseries_data
)
from src.data.model_selector import get_model_selector
from src.utils.config import (
    DB_FILE, ENABLED_TIMEFRAMES, TOP_ANALYSIS_CRYPTO, DATA_LIMIT_DAYS,
    SUBSERIES_LENGTH, SUBSERIES_MIN_SAMPLES
)

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    # Initialize colorama for cross-platform colored output
    init(autoreset=True)

def check_database_tables():
    """Check if database tables exist and create them if needed."""
    logging.info("Checking database tables...")
    init_data_tables()
    logging.info("Database tables ready.")

def add_volatility_columns(timeframe):
    """
    Add volatility columns to the data table for a specific timeframe.
    
    Args:
        timeframe: The timeframe to modify (e.g., '5m', '15m')
        
    Returns:
        True if successful, False otherwise
    """
    logging.info(f"Adding volatility columns to data_{timeframe} table...")
    
    # Connect to database
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    try:
        # Check if the columns already exist
        cursor.execute(f"PRAGMA table_info(data_{timeframe})")
        columns = [col[1] for col in cursor.fetchall()]
        
        # Columns to add if they don't exist
        volatility_columns = [
            "close_volatility REAL", 
            "open_volatility REAL", 
            "high_volatility REAL", 
            "low_volatility REAL",
            "volume_change REAL",
            "historical_volatility REAL"
        ]
        
        # Only add columns that don't exist
        for col in volatility_columns:
            col_name = col.split()[0]
            if col_name not in columns:
                cursor.execute(f"ALTER TABLE data_{timeframe} ADD COLUMN {col}")
                logging.info(f"Added column {col_name} to data_{timeframe} table")
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logging.error(f"Error adding volatility columns to data_{timeframe} table: {e}")
        conn.rollback()
        conn.close()
        return False

def process_symbol_volatility_sync(symbol, timeframe, lookback_days=100):
    """
    Versione sincrona di process_symbol_volatility per esecuzione parallela.
    
    Args:
        symbol: The symbol to process
        timeframe: The timeframe to process
        lookback_days: Number of days to look back
        
    Returns:
        Tuple of (DataFrame with volatility, Dictionary of categorized subseries)
    """
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(process_symbol_volatility(symbol, timeframe, lookback_days))
        loop.close()
        return result
    except Exception as e:
        logging.error(f"Error in parallel processing for {symbol}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None, None

async def process_symbol_volatility(symbol, timeframe, lookback_days=100):
    """
    Process volatility data for a specific symbol and timeframe.
    
    Args:
        symbol: The symbol to process
        timeframe: The timeframe to process
        lookback_days: Number of days to look back
        
    Returns:
        Tuple of (DataFrame with volatility, Dictionary of categorized subseries)
    """
    logging.info(f"Processing volatility for {symbol} ({timeframe})...")
    
    # Create a unique connection for each symbol to avoid database lock issues
    # Use timeout to avoid waiting indefinitely for locked database
    conn = sqlite3.connect(DB_FILE, timeout=30.0)
    
    try:
        # Implementazione della cache (punto 4)
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT COUNT(*) FROM data_{timeframe}
            WHERE symbol = ? AND close_volatility IS NOT NULL
            AND timestamp > datetime('now', '-30 minutes')
        """, (symbol,))
        recent_count = cursor.fetchone()[0]
        
        if recent_count > 10:  # Se ci sono più di 10 record con dati di volatilità recenti
            logging.info(f"Recent volatility data found for {symbol} ({timeframe}), skipping recalculation")
            # Raccogliamo categorized da dati esistenti
            return pd.DataFrame(), {}  # Restituisci DataFrame vuoto e dizionario vuoto
            
        # Calculate cutoff date
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%dT%H:%M:%S')
        
        # Query data for this symbol
        query = f"""
            SELECT timestamp, open, high, low, close, volume
            FROM data_{timeframe}
            WHERE symbol = ? AND timestamp > ?
            ORDER BY timestamp
        """
        
        df = pd.read_sql_query(query, conn, params=(symbol, cutoff_date))
        
        if df.empty:
            logging.warning(f"No data found for {symbol} ({timeframe}) after {cutoff_date}")
            conn.close()
            return None, None
        
        # Process timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Calculate volatility
        df_vol = calculate_volatility_rate(df)
        
        # Update volatility in database - BATCH UPDATE invece di riga per riga
        df_vol['timestamp_str'] = df_vol.index.strftime('%Y-%m-%dT%H:%M:%S')
        
        # Prepara un DataFrame con solo le colonne che ci servono per l'update del database
        update_cols = ['close_volatility', 'open_volatility', 'high_volatility', 
                      'low_volatility', 'volume_change', 'historical_volatility']
        
        # Filtra solo le colonne che esistono
        update_cols = [col for col in update_cols if col in df_vol.columns]
        
        if update_cols:
            # Aggiungi la colonna timestamp_str
            cols_to_extract = update_cols + ['timestamp_str']
            df_update = df_vol[cols_to_extract].dropna(subset=update_cols)
            
            if not df_update.empty:
                # Abilita il modalità WAL per prestazioni migliori
                conn.execute('PRAGMA journal_mode = WAL')
                
                # Crea lista di tuple per executemany
                update_data = []
                for idx, row in df_update.iterrows():
                    values = [row[col] for col in update_cols]
                    # Aggiungi symbol e timestamp in fondo
                    values.append(symbol)
                    values.append(row['timestamp_str'])
                    update_data.append(tuple(values))
                
                # Crea la query dinamica in base alle colonne disponibili
                set_clause = ", ".join([f"{col} = ?" for col in update_cols])
                query = f"""
                    UPDATE data_{timeframe}
                    SET {set_clause}
                    WHERE symbol = ? AND timestamp = ?
                """
                
                # Esegui l'update in un'unica transazione
                cursor = conn.cursor()
                cursor.executemany(query, update_data)
                conn.commit()
        
        # Get all subseries and categorize them
        categorized = None
        if len(df_vol) >= SUBSERIES_LENGTH:
            categorized = get_all_subseries_with_categories(df_vol)
            
            # Save subseries data
            if categorized:
                saved = save_subseries_data(symbol, timeframe, categorized)
                logging.info(f"Saved {saved} subseries for {symbol} ({timeframe})")
        
        conn.close()
        return df_vol, categorized
        
    except Exception as e:
        logging.error(f"Error processing volatility for {symbol} ({timeframe}): {e}")
        import traceback
        logging.error(traceback.format_exc())
        conn.close()
        return None, None

async def process_timeframe(timeframe, symbols, lookback_days=100):
    """
    Process all symbols for a specific timeframe using parallel execution.
    
    Args:
        timeframe: The timeframe to process
        symbols: List of symbols to process
        lookback_days: Number of days to look back
        
    Returns:
        Dictionary with statistics
    """
    logging.info(f"Processing {timeframe} data...")
    
    # First add volatility columns if needed
    add_volatility_columns(timeframe)
    
    # Process each symbol in parallel
    categories_count = 0
    subseries_count = 0
    symbols_processed = 0
    
    # Determina il numero di worker in base alle CPU disponibili
    # Riduciamo ulteriormente il numero di thread per evitare contese 
    max_workers = min(2, os.cpu_count())  # Massimo 2 thread per minimizzare contese
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Avvia le attività in parallelo
        futures = [executor.submit(process_symbol_volatility_sync, symbol, timeframe, lookback_days) 
                  for symbol in symbols]
        
        # Raccoglie i risultati con una barra di progresso
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), 
                          desc=f"Processing {timeframe} symbols"):
            df_vol, categorized = future.result()
            
            if df_vol is not None:
                symbols_processed += 1
                
                if categorized:
                    categories_count += len(categorized)
                    
                    # Count subseries
                    for category, subseries_list in categorized.items():
                        subseries_count += len(subseries_list)
    
    # Return statistics
    return {
        'timeframe': timeframe,
        'symbols_processed': symbols_processed,
        'categories': categories_count,
        'subseries': subseries_count
    }

def show_pattern_analysis():
    """Show analysis of patterns and transitions."""
    logging.info("Analyzing patterns and transitions...")
    
    # Get top categories
    top_categories = get_top_categories(limit=20)
    print("\n==== Top Pattern Categories ====")
    print(f"{'Category':<10} {'Pattern':<20} {'Count':<8}")
    print("-" * 40)
    for category_id, pattern, count in top_categories:
        print(f"{category_id:<10} {pattern:<20} {count:<8}")
    
    # Get top transitions
    transitions = get_category_transitions()
    print("\n==== Top Pattern Transitions ====")
    
    # Sort categories by the count of outgoing transitions
    sorted_categories = sorted(
        [(cat, len(trans)) for cat, trans in transitions.items()], 
        key=lambda x: x[1], 
        reverse=True
    )
    
    # Show top 10 categories with most transitions
    for i, (category, _) in enumerate(sorted_categories[:10]):
        pattern = binary_to_pattern(category)
        print(f"\nFrom {category} ({pattern}):")
        
        # Get top 3 transitions for this category
        cat_transitions = transitions[category]
        top_transitions = sorted(cat_transitions.items(), key=lambda x: x[1], reverse=True)[:3]
        
        for to_cat, prob in top_transitions:
            to_pattern = binary_to_pattern(to_cat)
            print(f"  → {to_cat} ({to_pattern}): {prob:.4f}")

def validate_symbol_predictions(symbols, timeframe, limit=100):
    """
    Validate predictions for multiple symbols and show statistics.
    
    Args:
        symbols: List of symbols to validate
        timeframe: Timeframe to use
        limit: Number of data points to use
    """
    print("\n==== Validating Predictions ====")
    
    selector = get_model_selector()
    
    prediction_stats = {
        'total': 0,
        'has_prediction': 0,
        'high_probability': 0  # >0.7
    }
    
    # Test each symbol
    for symbol in symbols[:10]:  # Limit to 10 symbols for brevity
        # Get data for this symbol
        df = get_symbol_data(symbol, timeframe, limit)
        
        if df.empty:
            continue
            
        # Get current category
        current_category, pattern = selector.categorize_current_data(df)
        if not current_category:
            continue
            
        prediction_stats['total'] += 1
        
        # Get prediction
        next_cat, next_pattern, prob = selector.predict_next_category(current_category)
        
        if next_cat:
            prediction_stats['has_prediction'] += 1
            
            if prob > 0.7:
                prediction_stats['high_probability'] += 1
    
    # Show statistics
    print(f"Tested {prediction_stats['total']} symbols")
    if prediction_stats['total'] > 0:
        print(f"Symbols with predictions: {prediction_stats['has_prediction']} ({prediction_stats['has_prediction']/prediction_stats['total']*100:.1f}%)")
        print(f"Symbols with high probability predictions (>0.7): {prediction_stats['high_probability']} ({prediction_stats['high_probability']/prediction_stats['total']*100:.1f}%)")
    
    # Show a detailed example if we have symbols
    if symbols:
        # Pick the first symbol for detailed example
        symbol = symbols[0]
        df = get_symbol_data(symbol, timeframe, limit)
        
        if not df.empty:
            current_category, pattern = selector.categorize_current_data(df)
            
            if current_category:
                print(f"\n==== Example Prediction for {symbol} ({timeframe}) ====")
                print(f"Current pattern: {pattern} ({current_category})")
                
                next_cat, next_pattern, prob = selector.predict_next_category(current_category)
                if next_cat:
                    print(f"Predicted next pattern: {next_pattern} ({next_cat})")
                    print(f"Probability: {prob:.4f}")
                    
                    # Show multi-step prediction
                    print("\nMulti-step prediction:")
                    predictions = selector.predict_multiple_steps(current_category, steps=5)
                    for i, (cat, pat, p) in enumerate(predictions):
                        print(f"Step {i+1}: {pat} ({cat}) - Probability: {p:.4f}")
                else:
                    print("No prediction available")

async def fetch_symbols_data(top_n=TOP_ANALYSIS_CRYPTO, days=DATA_LIMIT_DAYS, timeframes=None):
    """
    Fetch data for the top cryptocurrencies.
    
    Args:
        top_n: Number of top cryptocurrencies to fetch
        days: Number of days of history to fetch
        timeframes: List of timeframes to fetch
        
    Returns:
        List of symbols fetched
    """
    print(f"\n{'='*80}")
    print(f"  FASE 1: SCARICAMENTO DATI")
    print(f"  • Top {top_n} criptovalute")
    print(f"  • {days} giorni di dati storici")
    print(f"  • Timeframes: {', '.join(timeframes)}")
    print(f"{'='*80}\n")
    
    # Initialize exchange
    exchange = await create_exchange()
    
    try:
        # Fetch markets
        markets = await fetch_markets(exchange)
        if not markets:
            logging.error("No markets found. Check your internet connection and API credentials.")
            return []
        
        # Filter for USDT markets (linear perpetual futures)
        all_symbols = [m['symbol'] for m in markets.values() if m.get('quote') == 'USDT'
                      and m.get('active') and m.get('type') == 'swap']
        logging.info(f"Found {len(all_symbols)} USDT pairs on Bybit")
        
        # Get top cryptocurrencies by volume
        logging.info(f"Getting top {top_n} cryptocurrencies by USDT volume...")
        top_symbols = await get_top_symbols(exchange, all_symbols, top_n=top_n)
        
        if not top_symbols:
            logging.error("Failed to get top symbols.")
            return []
        
        # Process each timeframe
        for timeframe in timeframes:
            if timeframe not in ENABLED_TIMEFRAMES:
                logging.warning(f"Timeframe {timeframe} not enabled in config. Skipping.")
                continue
                
            print(f"\n{'-'*60}")
            print(f"  Scaricamento dati per timeframe: {timeframe}")
            print(f"{'-'*60}")
            
            # Process each symbol for this timeframe
            for i, symbol in enumerate(top_symbols):
                try:
                    # Check if data is fresh
                    is_fresh, last_timestamp = check_data_freshness(symbol, timeframe, max_age_days=1)
                    
                    if is_fresh:
                        # Get first timestamp to show data range
                        conn = sqlite3.connect(DB_FILE)
                        cursor = conn.cursor()
                        cursor.execute(f"""
                            SELECT MIN(timestamp) FROM data_{timeframe}
                            WHERE symbol = ?
                        """, (symbol,))
                        first_timestamp_str = cursor.fetchone()[0]
                        first_timestamp = datetime.strptime(first_timestamp_str, '%Y-%m-%dT%H:%M:%S') if first_timestamp_str else None
                        conn.close()
                        
                        # Calculate data span in days
                        if first_timestamp and last_timestamp:
                            days_span = (last_timestamp - first_timestamp).days
                            now = datetime.now()
                            days_ago = (now - last_timestamp).days
                            hours_ago = int((now - last_timestamp).total_seconds() / 3600)
                            
                            # Display colorful freshness info
                            time_ago = f"{hours_ago} ore" if hours_ago < 24 else f"{days_ago} giorni"
                            print(f"  {Fore.YELLOW}{symbol}{Style.RESET_ALL} ({timeframe}): " +
                                  f"{Fore.GREEN}Dati aggiornati{Style.RESET_ALL} - " +
                                  f"Ultimo: {Fore.CYAN}{last_timestamp.strftime('%Y-%m-%d %H:%M')}{Style.RESET_ALL} " +
                                  f"({Fore.MAGENTA}{time_ago} fa{Style.RESET_ALL}), " +
                                  f"Copertura: {Fore.BLUE}{days_span}{Style.RESET_ALL} giorni")
                        else:
                            logging.info(f"Data for {Fore.YELLOW}{symbol}{Style.RESET_ALL} ({timeframe}) is fresh. Skipping download.")
                    else:
                        # Fetch new data
                        df = await fetch_and_save_data(exchange, symbol, timeframe, limit=1000, max_age_hours=24)
                        if df is not None:
                            print(f"  {Fore.YELLOW}{symbol}{Style.RESET_ALL} ({timeframe}): " +
                                  f"{Fore.GREEN}Dati scaricati e salvati{Style.RESET_ALL}")
                        else:
                            print(f"  {Fore.YELLOW}{symbol}{Style.RESET_ALL} ({timeframe}): " +
                                  f"{Fore.RED}Nessun dato disponibile{Style.RESET_ALL}")
                    
                except Exception as e:
                    logging.error(f"Error processing {symbol} ({timeframe}): {e}")
        
        # Show download summary
        print(f"\n{'='*60}")
        print(f"  SCARICAMENTO DATI COMPLETATO")
        print(f"  • {len(top_symbols)} criptovalute processate")
        print(f"  • Timeframes: {', '.join(timeframes)}")
        print(f"{'='*60}\n")
        
        return top_symbols
        
    finally:
        # Close exchange connection
        await exchange.close()

async def process_volatility_data(symbols, timeframes, days=100):
    """
    Process volatility data for all symbols and timeframes.
    
    Args:
        symbols: List of symbols to process
        timeframes: List of timeframes to process
        days: Number of days of history to process
    """
    print(f"\n{'='*80}")
    print(f"  FASE 2: CALCOLO DELLA VOLATILITÀ E CATEGORIZZAZIONE")
    print(f"  • {len(symbols)} criptovalute da processare")
    print(f"  • Timeframes: {', '.join(timeframes)}")
    print(f"{'='*80}\n")
    
    # Process each timeframe
    results = []
    
    for tf in timeframes:
        if tf not in ENABLED_TIMEFRAMES:
            logging.warning(f"Timeframe {tf} not enabled in config. Skipping.")
            continue
            
        result = await process_timeframe(tf, symbols, days)
        results.append(result)
    
    # Print summary
    print("\n==== Riepilogo Processamento ====")
    for result in results:
        if 'error' in result:
            print(f"Error processing {result['timeframe']}: {result['error']}")
        else:
            print(f"Timeframe: {result['timeframe']}")
            print(f"  Symbols processed: {result['symbols_processed']}")
            print(f"  Categories: {result['categories']}")
            print(f"  Subseries: {result['subseries']}")

async def analyze_and_predict(symbols, timeframes):
    """
    Analyze patterns and make predictions.
    
    Args:
        symbols: List of symbols to analyze
        timeframes: List of timeframes to analyze
    """
    print(f"\n{'='*80}")
    print(f"  FASE 3: ANALISI DEI PATTERN E PREDIZIONI")
    print(f"{'='*80}\n")
    
    # Show pattern analysis
    show_pattern_analysis()
    
    # Validate predictions for a selection of symbols
    if symbols and timeframes:
        validate_symbol_predictions(symbols, timeframes[0])

async def run_pipeline(top_n=100, days=100, timeframes=None):
    """
    Run the complete volatility analysis pipeline.
    
    Args:
        top_n: Number of top cryptocurrencies to analyze
        days: Number of days of history to analyze
        timeframes: List of timeframes to analyze
    """
    if timeframes is None:
        timeframes = ["5m", "15m"]
        
    print(f"\n{'='*80}")
    print(f"  PIPELINE DI ANALISI DELLA VOLATILITÀ")
    print(f"  • Top {top_n} criptovalute")
    print(f"  • {days} giorni di dati storici")
    print(f"  • Timeframes: {', '.join(timeframes)}")
    print(f"{'='*80}\n")
    
    # Step 1: Set up logging and database
    setup_logging()
    check_database_tables()
    
    # Step 2: Fetch data for top symbols
    symbols = await fetch_symbols_data(top_n, days, timeframes)
    
    if not symbols:
        logging.error("No symbols fetched. Pipeline cannot continue.")
        return
    
    # Step 3: Process volatility data
    await process_volatility_data(symbols, timeframes, days)
    
    # Step 4: Analyze patterns and make predictions
    await analyze_and_predict(symbols, timeframes)
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"{Fore.CYAN}  PIPELINE COMPLETATO{Style.RESET_ALL}")
    print(f"  • {Fore.YELLOW}{len(symbols)}{Style.RESET_ALL} criptovalute analizzate")
    print(f"  • Timeframes: {Fore.GREEN}{', '.join(timeframes)}{Style.RESET_ALL}")
    print(f"  • Database: {Fore.BLUE}{os.path.abspath(DB_FILE)}{Style.RESET_ALL}")
    
    # Show data span information
    print(f"\n  {Fore.MAGENTA}INFORMAZIONI DATI:{Style.RESET_ALL}")
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    for tf in timeframes:
        try:
            # Find oldest and newest data
            cursor.execute(f"""
                SELECT MIN(timestamp), MAX(timestamp) FROM data_{tf}
                WHERE symbol IN ({','.join(['?']*len(symbols))})
            """, symbols[:min(len(symbols), 10)])  # Limit to first 10 symbols
            
            min_ts_str, max_ts_str = cursor.fetchone()
            
            if min_ts_str and max_ts_str:
                min_ts = datetime.strptime(min_ts_str, '%Y-%m-%dT%H:%M:%S')
                max_ts = datetime.strptime(max_ts_str, '%Y-%m-%dT%H:%M:%S')
                days_span = (max_ts - min_ts).days
                now = datetime.now()
                hours_ago = int((now - max_ts).total_seconds() / 3600)
                
                print(f"  • {Fore.CYAN}Timeframe {tf}{Style.RESET_ALL}:")
                print(f"    - Dal: {Fore.YELLOW}{min_ts.strftime('%Y-%m-%d %H:%M')}{Style.RESET_ALL}")
                print(f"    - Al: {Fore.YELLOW}{max_ts.strftime('%Y-%m-%d %H:%M')}{Style.RESET_ALL} ({Fore.MAGENTA}{hours_ago} ore fa{Style.RESET_ALL})")
                print(f"    - Copertura: {Fore.GREEN}{days_span} giorni{Style.RESET_ALL}")
        except Exception as e:
            print(f"  • {Fore.CYAN}Timeframe {tf}{Style.RESET_ALL}: {Fore.RED}Errore nel recupero informazioni{Style.RESET_ALL}")
    
    conn.close()
    print(f"{'='*80}\n")

if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Pipeline completo di analisi della volatilità")
    parser.add_argument("--top", type=int, default=TOP_ANALYSIS_CRYPTO, help="Numero di criptovalute da analizzare")
    parser.add_argument("--days", type=int, default=DATA_LIMIT_DAYS, help="Giorni di dati da considerare")
    parser.add_argument("--timeframes", type=str, default="5m,15m", help="Timeframes da analizzare (separati da virgola)")
    
    args = parser.parse_args()
    
    # Convert timeframes string to list
    timeframes_list = args.timeframes.split(",")
    
    # Fix event loop policy for Windows
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Run the pipeline
    asyncio.run(run_pipeline(args.top, args.days, timeframes_list))
