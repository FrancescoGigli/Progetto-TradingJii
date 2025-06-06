#!/usr/bin/env python3
"""
Database management module for TradingJii

Handles database initialization, queries, and data management.
"""

import sqlite3
from datetime import datetime
import logging
from colorama import Fore, Style
from modules.utils.config import DB_FILE, TIMEFRAME_CONFIG

# Data fissa di inizio - 1° gennaio 2024
MIN_START_DATE = datetime(2024, 1, 1)

def add_warmup_column_if_needed(timeframes):
    """
    Aggiunge la colonna warmup_data alle tabelle esistenti se non esiste.
    
    Args:
        timeframes: Lista dei timeframe da controllare
    """
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            
            for timeframe in timeframes:
                table_name = f"market_data_{timeframe}"
                table_name = table_name.replace('-', '_')
                
                # Verifica se la tabella esiste
                cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
                if not cursor.fetchone():
                    continue  # La tabella non esiste, salta
                
                # Verifica se la colonna warmup_data esiste già
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = [row[1] for row in cursor.fetchall()]
                
                if 'warmup_data' not in columns:
                    logging.info(f"Aggiunta colonna warmup_data alla tabella {table_name}")
                    try:
                        # Aggiungi la colonna warmup_data
                        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN warmup_data INTEGER")
                        conn.commit()
                    except Exception as e:
                        logging.error(f"Errore nell'aggiunta della colonna warmup_data a {table_name}: {e}")
                else:
                    logging.debug(f"La colonna warmup_data esiste già nella tabella {table_name}")
                    
    except Exception as e:
        logging.error(f"Errore nell'aggiornamento delle tabelle: {e}")
        import traceback
        logging.error(traceback.format_exc())

def init_market_data_tables(timeframes):
    """
    Initialize unified market data tables for each timeframe.
    These tables contain OHLCV data, technical indicators, and volatility metrics.
    
    Args:
        timeframes: List of timeframes to create tables for
    """
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            tables_created = []
            
            for timeframe in timeframes:
                table_name = f"market_data_{timeframe}"
                # Ensure the table name is valid - remove any problematic characters
                table_name = table_name.replace('-', '_')
                
                try:
                    logging.info(f"Creating unified table {table_name} if it doesn't exist...")
                    cursor.execute(f"""
                        CREATE TABLE IF NOT EXISTS {table_name} (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            symbol TEXT NOT NULL,
                            timestamp TEXT NOT NULL,
                            
                            /* OHLCV data */
                            open REAL,
                            high REAL,
                            low REAL,
                            close REAL,
                            volume REAL,
                            
                            /* Technical indicators */
                            sma9 REAL,
                            sma20 REAL,
                            sma50 REAL,
                            ema20 REAL,
                            ema50 REAL,
                            ema200 REAL,
                            rsi14 REAL,
                            stoch_k REAL,
                            stoch_d REAL,
                            macd REAL,
                            macd_signal REAL,
                            macd_hist REAL,
                            atr14 REAL,
                            bbands_upper REAL,
                            bbands_middle REAL,
                            bbands_lower REAL,
                            obv REAL,
                            vwap REAL,
                            volume_sma20 REAL,
                            adx14 REAL,
                            
                            /* Volatility data */
                            volatility REAL,
                            
                            UNIQUE(symbol, timestamp)
                        )
                    """)
                    cursor.execute(f"""
                        CREATE INDEX IF NOT EXISTS idx_{timeframe}_mkt_symbol_timestamp
                        ON {table_name} (symbol, timestamp)
                    """)
                    tables_created.append(table_name)
                    
                    # Verify table was created successfully
                    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
                    if cursor.fetchone():
                        logging.info(f"{Fore.GREEN}Unified table {table_name} created or already exists{Style.RESET_ALL}")
                    else:
                        logging.error(f"{Fore.RED}Failed to verify unified table {table_name} after creation{Style.RESET_ALL}")
                except Exception as e:
                    logging.error(f"{Fore.RED}Error creating unified table {table_name}: {e}{Style.RESET_ALL}")
                    raise
            
            conn.commit()
            
            # Verify all tables were created
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            all_tables = [row[0] for row in cursor.fetchall()]
            logging.info(f"{Fore.GREEN}Database initialization complete. Tables in database: {', '.join(all_tables)}{Style.RESET_ALL}")
            
            # Aggiungi la colonna warmup_data alle tabelle esistenti se necessario
            add_warmup_column_if_needed(timeframes)
            
    except Exception as e:
        logging.error(f"{Fore.RED}Database initialization error: {e}{Style.RESET_ALL}")
        import traceback
        logging.error(traceback.format_exc())
        raise

def init_data_tables(timeframes):
    """
    Initialize legacy database tables for each timeframe.
    DEPRECATED: Use init_market_data_tables instead.
    
    Args:
        timeframes: List of timeframes to create tables for
    """
    # For backward compatibility, this calls the new unified table creation function
    init_market_data_tables(timeframes)
    
    # Log a deprecation warning
    logging.warning(f"{Fore.YELLOW}DEPRECATED: init_data_tables is deprecated. Use init_market_data_tables instead.{Style.RESET_ALL}")

def get_timestamp_range(symbol, timeframe):
    """
    Get the first and last timestamp available for a symbol.
    
    Args:
        symbol: The cryptocurrency symbol
        timeframe: The timeframe to check
        
    Returns:
        Tuple of (first_date, last_date) as datetime objects or (None, None) if no data
    """
    table_name = f"market_data_{timeframe}"
    
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
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

def check_data_freshness(symbol, timeframe):
    """
    Check if we already have fresh data for a symbol and timeframe.
    
    Args:
        symbol: The cryptocurrency symbol
        timeframe: The timeframe to check
        
    Returns:
        Tuple of (is_fresh, last_timestamp)
    """
    try:
        now = datetime.now()
        table_name = f"market_data_{timeframe}"
        
        # First, verify the table exists
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
            if not cursor.fetchone():
                logging.warning(f"{Fore.YELLOW}Table {table_name} does not exist. Will create it in the download process.{Style.RESET_ALL}")
                return False, None
        
        # If the table exists, check for data freshness
        first_date, last_date = get_timestamp_range(symbol, timeframe)

        if last_date:
            time_diff = now - last_date
            if time_diff < TIMEFRAME_CONFIG[timeframe]['max_age']:
                logging.info(f"Saltato {Fore.YELLOW}{symbol}{Style.RESET_ALL} ({timeframe}): Dati recenti già esistenti")
                logging.info(f"  • Prima data: {Fore.CYAN}{first_date.strftime('%Y-%m-%d %H:%M')}{Style.RESET_ALL}")
                logging.info(f"  • Ultima data: {Fore.CYAN}{last_date.strftime('%Y-%m-%d %H:%M')}{Style.RESET_ALL}")
                return True, last_date
                
        return False, last_date
    except Exception as e:
        logging.error(f"{Fore.RED}Error checking data freshness for {symbol} ({timeframe}): {e}{Style.RESET_ALL}")
        return False, None

def cleanup_old_data():
    """
    Elimina tutti i dati precedenti al 1° gennaio 2024 dal database.
    Mantiene i dati temporanei di warmup contrassegnati come 'temp_warmup'.
    """
    min_date_str = MIN_START_DATE.strftime('%Y-%m-%dT%H:%M:%S')
    logging.info(f"Pulizia dati: rimozione record precedenti al {Fore.YELLOW}{min_date_str}{Style.RESET_ALL}")
    
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            
            # Ottieni tutte le tabelle di dati di mercato
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name LIKE 'market_data_%'
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
            total_deleted = 0
            
            for table in tables:
                try:
                    # Verifica se la colonna warmup_data esiste
                    cursor.execute(f"PRAGMA table_info({table})")
                    columns = [row[1] for row in cursor.fetchall()]
                    
                    if 'warmup_data' in columns:
                        # Elimina solo i dati definitivi pre-2024, mantenendo i dati di warmup
                        cursor.execute(f"""
                            DELETE FROM {table}
                            WHERE timestamp < ? AND (warmup_data IS NULL OR warmup_data = 0)
                        """, (min_date_str,))
                    else:
                        # Se la colonna non esiste, elimina tutti i dati pre-2024
                        cursor.execute(f"""
                            DELETE FROM {table}
                            WHERE timestamp < ?
                        """, (min_date_str,))
                    
                    deleted_count = cursor.rowcount
                    total_deleted += deleted_count
                    
                    if deleted_count > 0:
                        logging.info(f"Rimossi {Fore.RED}{deleted_count}{Style.RESET_ALL} record obsoleti dalla tabella {table}")
                except Exception as e:
                    logging.error(f"Errore durante la pulizia della tabella {table}: {e}")
            
            conn.commit()
            
            if total_deleted > 0:
                logging.info(f"Pulizia completata: rimossi {Fore.RED}{total_deleted}{Style.RESET_ALL} record obsoleti in totale")
            else:
                logging.info(f"Nessun record obsoleto trovato nel database")
            
    except Exception as e:
        logging.error(f"Errore durante la pulizia dei dati obsoleti: {e}")
        import traceback
        logging.error(traceback.format_exc())

def delete_warmup_data():
    """
    Elimina tutti i dati di warmup (warmup_data = 1) dal database.
    Questa funzione viene chiamata alla fine del ciclo di elaborazione.
    """
    logging.info(f"Eliminazione dati di warmup dal database...")
    
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            
            # Ottieni tutte le tabelle di dati di mercato
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name LIKE 'market_data_%'
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
            total_deleted = 0
            
            for table in tables:
                try:
                    # Verifica se la colonna warmup_data esiste
                    cursor.execute(f"PRAGMA table_info({table})")
                    columns = [row[1] for row in cursor.fetchall()]
                    
                    if 'warmup_data' in columns:
                        # Elimina i dati di warmup
                        cursor.execute(f"""
                            DELETE FROM {table}
                            WHERE warmup_data = 1
                        """)
                        
                        deleted_count = cursor.rowcount
                        total_deleted += deleted_count
                        
                        if deleted_count > 0:
                            logging.info(f"Rimossi {Fore.RED}{deleted_count}{Style.RESET_ALL} record di warmup dalla tabella {table}")
                except Exception as e:
                    logging.error(f"Errore durante l'eliminazione dei dati di warmup dalla tabella {table}: {e}")
            
            conn.commit()
            
            if total_deleted > 0:
                logging.info(f"Pulizia warmup completata: rimossi {Fore.RED}{total_deleted}{Style.RESET_ALL} record di warmup in totale")
            else:
                logging.info(f"Nessun record di warmup trovato nel database")
            
    except Exception as e:
        logging.error(f"Errore durante l'eliminazione dei dati di warmup: {e}")
        import traceback
        logging.error(traceback.format_exc())

def save_ohlcv_data(symbol, timeframe, ohlcv_data, is_temp=False):
    """
    Save OHLCV data to the unified market data table.
    
    Args:
        symbol: The cryptocurrency symbol
        timeframe: The timeframe of the data
        ohlcv_data: List of OHLCV data from the exchange
        is_temp: Se True, i dati sono temporanei per il calcolo degli indicatori
        
    Returns:
        Tuple of (success, num_records_saved)
    """
    if not ohlcv_data:
        return False, 0
        
    table_name = f"market_data_{timeframe}"
    
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            
            # First, ensure the table exists
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
            if not cursor.fetchone():
                logging.warning(f"{Fore.YELLOW}Table {table_name} does not exist. Creating it now...{Style.RESET_ALL}")
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        
                        /* OHLCV data */
                        open REAL,
                        high REAL,
                        low REAL,
                        close REAL,
                        volume REAL,
                        
                        /* Technical indicators */
                        sma9 REAL,
                        sma20 REAL,
                        sma50 REAL,
                        ema20 REAL,
                        ema50 REAL,
                        ema200 REAL,
                        rsi14 REAL,
                        stoch_k REAL,
                        stoch_d REAL,
                        macd REAL,
                        macd_signal REAL,
                        macd_hist REAL,
                        atr14 REAL,
                        bbands_upper REAL,
                        bbands_middle REAL,
                        bbands_lower REAL,
                        obv REAL,
                        vwap REAL,
                        volume_sma20 REAL,
                        adx14 REAL,
                        
                        /* Volatility data */
                        volatility REAL,
                        
                        /* Flag per dati di warmup */
                        warmup_data INTEGER,
                        
                        UNIQUE(symbol, timestamp)
                    )
                """)
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{timeframe.replace('-', '_')}_mkt_symbol_timestamp
                    ON {table_name} (symbol, timestamp)
                """)
                conn.commit()
                logging.info(f"{Fore.GREEN}Table {table_name} created successfully{Style.RESET_ALL}")
            
            # Verifica se la colonna warmup_data esiste
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [row[1] for row in cursor.fetchall()]
            has_warmup_column = 'warmup_data' in columns
            
            # Se la colonna non esiste, prova ad aggiungerla
            if not has_warmup_column and is_temp:
                try:
                    logging.info(f"Aggiunta colonna warmup_data alla tabella {table_name}")
                    cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN warmup_data INTEGER")
                    conn.commit()
                    has_warmup_column = True
                except Exception as e:
                    logging.warning(f"Impossibile aggiungere la colonna warmup_data: {e}")
                    # Continua comunque, useremo una query diversa
            
            # Convert OHLCV data to records format
            min_timestamp_ms = int(MIN_START_DATE.timestamp() * 1000)
            records = []
            
            for r in ohlcv_data:
                # Controlla se è un dato di warmup (prima del 1° gennaio 2024)
                is_warmup = 1 if r[0] < min_timestamp_ms and is_temp else 0
                
                # Crea il record con o senza il flag warmup_data
                if has_warmup_column:
                    record = (
                        symbol, 
                        datetime.fromtimestamp(r[0]/1000).strftime('%Y-%m-%dT%H:%M:%S'), 
                        *r[1:], 
                        is_warmup
                    )
                else:
                    record = (
                        symbol, 
                        datetime.fromtimestamp(r[0]/1000).strftime('%Y-%m-%dT%H:%M:%S'), 
                        *r[1:]
                    )
                records.append(record)
            
            # Insert data using INSERT OR REPLACE - only update OHLCV fields
            if has_warmup_column:
                cursor.executemany(f"""
                    INSERT OR REPLACE INTO {table_name}
                    (symbol, timestamp, open, high, low, close, volume, warmup_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, records)
            else:
                cursor.executemany(f"""
                    INSERT OR REPLACE INTO {table_name}
                    (symbol, timestamp, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, records)
            
            conn.commit()
            
            # Log data range after saving
            first_date, last_date = get_timestamp_range(symbol, timeframe)
            if first_date and last_date:
                logging.info(f"Completato {Fore.YELLOW}{symbol}{Style.RESET_ALL} ({timeframe}) - salvati {Fore.GREEN}{len(records)}{Style.RESET_ALL} record")
                logging.info(f"  • Prima data: {Fore.CYAN}{first_date.strftime('%Y-%m-%d %H:%M')}{Style.RESET_ALL}")
                logging.info(f"  • Ultima data: {Fore.CYAN}{last_date.strftime('%Y-%m-%d %H:%M')}{Style.RESET_ALL}")
            
            return True, len(records)
    except Exception as e:
        logging.error(f"{Fore.RED}Errore nel salvataggio dei dati per {symbol} ({timeframe}): {e}{Style.RESET_ALL}")
        import traceback
        logging.error(traceback.format_exc())
        return False, 0
