#!/usr/bin/env python3
"""
Technical Indicator Processor Module for TradingJii - Versione Semplificata

Calcola indicatori tecnici usando solo numpy e pandas, senza dipendenze esterne.
"""

import sqlite3
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from colorama import Fore, Style
from datetime import datetime
import traceback
from modules.utils.config import DB_FILE

def init_indicator_tables(timeframes):
    """
    Funzione dummy per compatibilità con il codice esistente.
    Gli indicatori sono ora salvati direttamente nelle tabelle market_data_{timeframe}.
    """
    logging.info(f"{Fore.GREEN}Inizializzazione indicatori tecnici - Usando implementazione semplificata{Style.RESET_ALL}")
    pass

def load_ohlcv_data(symbol: str, timeframe: str) -> pd.DataFrame:
    """
    Carica i dati OHLCV dal database per un simbolo e timeframe specifico.
    Include sia i dati regolari che quelli di warmup per il calcolo accurato degli indicatori.
    
    Args:
        symbol: Simbolo della criptovaluta
        timeframe: Timeframe (es. '1h')
        
    Returns:
        DataFrame con i dati OHLCV
    """
    table_name = f"market_data_{timeframe}"
    
    try:
        with sqlite3.connect(DB_FILE) as conn:
            # Verifica se la tabella ha la colonna warmup_data
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [row[1] for row in cursor.fetchall()]
            
            # Prepara la query in base alle colonne disponibili
            if 'warmup_data' in columns:
                query = f"""
                    SELECT timestamp, open, high, low, close, volume, warmup_data
                    FROM {table_name}
                    WHERE symbol = ?
                    ORDER BY timestamp ASC
                """
            else:
                query = f"""
                    SELECT timestamp, open, high, low, close, volume
                    FROM {table_name}
                    WHERE symbol = ?
                    ORDER BY timestamp ASC
                """
                
            df = pd.read_sql_query(query, conn, params=(symbol,))
            
            if df.empty:
                logging.info(f"Nessun dato disponibile per {Fore.YELLOW}{symbol}{Style.RESET_ALL} ({timeframe})")
                return pd.DataFrame()
            
            # Assicurati che il timestamp sia formattato correttamente
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            logging.info(f"Caricati {Fore.GREEN}{len(df)}{Style.RESET_ALL} record OHLCV per {Fore.YELLOW}{symbol}{Style.RESET_ALL} ({timeframe})")
            return df
            
    except Exception as e:
        logging.error(f"{Fore.RED}Errore durante il caricamento dei dati OHLCV per {symbol} ({timeframe}): {e}{Style.RESET_ALL}")
        logging.error(traceback.format_exc())
        return pd.DataFrame()

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola tutti gli indicatori tecnici utilizzando solo numpy e pandas.
    
    Args:
        df: DataFrame con i dati OHLCV
        
    Returns:
        DataFrame con i dati originali e tutti gli indicatori
    """
    if df.empty:
        return pd.DataFrame()
    
    try:
        # Inizializza il DataFrame risultante con il timestamp dall'input
        result_df = pd.DataFrame({'timestamp': df['timestamp']})
        
        # Se la colonna warmup_data esiste, copiala nel risultato
        if 'warmup_data' in df.columns:
            result_df['warmup_data'] = df['warmup_data']
        
        # Estrai le serie di dati
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # ----- MEDIE MOBILI SEMPLICI -----
        # SMA 9
        result_df['sma9'] = close.rolling(window=9).mean()
        
        # SMA 20
        result_df['sma20'] = close.rolling(window=20).mean()
        
        # SMA 50
        result_df['sma50'] = close.rolling(window=50).mean()
        
        # ----- MEDIE MOBILI ESPONENZIALI -----
        # EMA 20
        result_df['ema20'] = close.ewm(span=20, adjust=False).mean()
        
        # EMA 50
        result_df['ema50'] = close.ewm(span=50, adjust=False).mean()
        
        # EMA 200
        result_df['ema200'] = close.ewm(span=200, adjust=False).mean()
        
        # ----- INDICATORI DI MOMENTUM -----
        # RSI 14
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        # Per i periodi successivi ai primi 14
        for i in range(14, len(gain)):
            avg_gain.iloc[i] = (avg_gain.iloc[i-1] * 13 + gain.iloc[i]) / 14
            avg_loss.iloc[i] = (avg_loss.iloc[i-1] * 13 + loss.iloc[i]) / 14
        
        rs = avg_gain / avg_loss
        result_df['rsi14'] = 100 - (100 / (1 + rs))
        
        # Stocastico
        # %K = (Prezzo corrente - Minimo(n)) / (Massimo(n) - Minimo(n)) * 100
        # %D = SMA di %K
        period = 14
        low_min = low.rolling(window=period).min()
        high_max = high.rolling(window=period).max()
        
        # %K
        result_df['stoch_k'] = 100 * ((close - low_min) / (high_max - low_min))
        
        # %D (media mobile di %K su 3 periodi)
        result_df['stoch_d'] = result_df['stoch_k'].rolling(window=3).mean()
        
        # MACD
        # MACD Line = EMA(12) - EMA(26)
        # Signal Line = EMA(9) del MACD Line
        # Histogram = MACD Line - Signal Line
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - signal_line
        
        result_df['macd'] = macd_line
        result_df['macd_signal'] = signal_line
        result_df['macd_hist'] = macd_hist
        
        # ----- INDICATORI DI VOLATILITÀ -----
        # ATR 14
        # TR = max(High - Low, |High - Close_prev|, |Low - Close_prev|)
        # ATR = Media mobile di TR
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        result_df['atr14'] = tr.rolling(window=14).mean()
        
        # Bollinger Bands
        # Middle Band = SMA(20)
        # Upper Band = Middle Band + (2 * Deviazione standard di 20 periodi)
        # Lower Band = Middle Band - (2 * Deviazione standard di 20 periodi)
        result_df['bbands_middle'] = close.rolling(window=20).mean()
        std_dev = close.rolling(window=20).std()
        
        result_df['bbands_upper'] = result_df['bbands_middle'] + (2 * std_dev)
        result_df['bbands_lower'] = result_df['bbands_middle'] - (2 * std_dev)
        
        # ----- INDICATORI BASATI SUL VOLUME -----
        # OBV (On-Balance Volume)
        obv = [0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.append(obv[-1] + volume.iloc[i])
            elif close.iloc[i] < close.iloc[i-1]:
                obv.append(obv[-1] - volume.iloc[i])
            else:
                obv.append(obv[-1])
        
        result_df['obv'] = obv
        
        # VWAP (Volume-Weighted Average Price)
        typical_price = (high + low + close) / 3
        result_df['vwap'] = (typical_price * volume).cumsum() / volume.cumsum()
        
        # Volume SMA 20
        result_df['volume_sma20'] = volume.rolling(window=20).mean()
        
        # ----- INDICATORI DI TREND -----
        # ADX (Average Directional Index) - Versione semplificata
        # Per una versione completa servirebbe un calcolo più elaborato
        # Qui usiamo un semplice placeholder basato sulla volatilità
        result_df['adx14'] = tr.rolling(window=14).mean() / close * 100
        
        return result_df
    except Exception as e:
        logging.error(f"{Fore.RED}Errore nel calcolo degli indicatori: {e}{Style.RESET_ALL}")
        logging.error(traceback.format_exc())
        return pd.DataFrame()

def save_indicators(symbol: str, timeframe: str, indicators_df: pd.DataFrame) -> bool:
    """
    Salva gli indicatori calcolati nella tabella unificata dei dati di mercato.
    Aggiorna solo i dati dal 1° gennaio 2024 in poi, ignorando i dati di warmup.
    
    Args:
        symbol: Simbolo della criptovaluta
        timeframe: Timeframe (es. '1h')
        indicators_df: DataFrame con gli indicatori calcolati
        
    Returns:
        Boolean che indica il successo dell'operazione
    """
    if indicators_df.empty:
        logging.warning(f"Nessun indicatore da salvare per {symbol} ({timeframe})")
        return False
    
    try:
        table_name = f"market_data_{timeframe}".replace('-', '_')
        min_date = datetime(2024, 1, 1).strftime('%Y-%m-%dT%H:%M:%S')
        
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            
            # Verifica se la tabella ha la colonna warmup_data
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [row[1] for row in cursor.fetchall()]
            has_warmup_column = 'warmup_data' in columns
            
            # Prepara la query di base per l'UPDATE
            if has_warmup_column:
                query_base = f"""
                    UPDATE {table_name}
                    SET 
                        sma9 = ?, sma20 = ?, sma50 = ?,
                        ema20 = ?, ema50 = ?, ema200 = ?,
                        rsi14 = ?, stoch_k = ?, stoch_d = ?,
                        macd = ?, macd_signal = ?, macd_hist = ?,
                        atr14 = ?, bbands_upper = ?, bbands_middle = ?, bbands_lower = ?,
                        obv = ?, vwap = ?, volume_sma20 = ?,
                        adx14 = ?
                    WHERE symbol = ? AND timestamp = ? AND (warmup_data IS NULL OR warmup_data = 0)
                """
            else:
                query_base = f"""
                    UPDATE {table_name}
                    SET 
                        sma9 = ?, sma20 = ?, sma50 = ?,
                        ema20 = ?, ema50 = ?, ema200 = ?,
                        rsi14 = ?, stoch_k = ?, stoch_d = ?,
                        macd = ?, macd_signal = ?, macd_hist = ?,
                        atr14 = ?, bbands_upper = ?, bbands_middle = ?, bbands_lower = ?,
                        obv = ?, vwap = ?, volume_sma20 = ?,
                        adx14 = ?
                    WHERE symbol = ? AND timestamp = ? AND timestamp >= ?
                """
            
            # Filtra i record e aggiorna
            records_saved = 0
            for _, row in indicators_df.iterrows():
                try:
                    # Controlla se questo record è un dato di warmup o se è prima del 1° gennaio 2024
                    is_warmup = False
                    
                    # Se ha la colonna warmup_data, usa quella
                    if has_warmup_column and 'warmup_data' in row and row['warmup_data'] == 1:
                        is_warmup = True
                    
                    # Controlla anche la data
                    timestamp_str = row['timestamp'].strftime('%Y-%m-%dT%H:%M:%S')
                    if timestamp_str < min_date:
                        is_warmup = True
                    
                    # Salta i dati di warmup
                    if is_warmup:
                        continue
                    
                    # Prepara i parametri
                    params = [
                        row.get('sma9'), row.get('sma20'), row.get('sma50'),
                        row.get('ema20'), row.get('ema50'), row.get('ema200'),
                        row.get('rsi14'), row.get('stoch_k'), row.get('stoch_d'),
                        row.get('macd'), row.get('macd_signal'), row.get('macd_hist'),
                        row.get('atr14'), row.get('bbands_upper'), row.get('bbands_middle'), row.get('bbands_lower'),
                        row.get('obv'), row.get('vwap'), row.get('volume_sma20'),
                        row.get('adx14'),
                        symbol, timestamp_str
                    ]
                    
                    # Aggiungi il parametro data minima se necessario
                    if not has_warmup_column:
                        params.append(min_date)
                    
                    # Esegui la query
                    cursor.execute(query_base, params)
                    
                    # Verifica se l'aggiornamento ha modificato righe
                    if cursor.rowcount > 0:
                        records_saved += 1
                except Exception as e:
                    logging.error(f"Errore nell'aggiornamento degli indicatori per {symbol} a {timestamp_str}: {e}")
                    continue
            
            conn.commit()
            
            # Controlla quanti dati totali abbiamo
            if has_warmup_column:
                cursor.execute(f"""
                    SELECT COUNT(*) FROM {table_name}
                    WHERE symbol = ? AND (warmup_data IS NULL OR warmup_data = 0)
                """, (symbol,))
            else:
                cursor.execute(f"""
                    SELECT COUNT(*) FROM {table_name}
                    WHERE symbol = ? AND timestamp >= ?
                """, (symbol, min_date))
                
            total_records = cursor.fetchone()[0]
            
            percentage = (records_saved / total_records * 100) if total_records > 0 else 0
            
            logging.info(f"💾 Aggiornati {Fore.GREEN}{records_saved}{Style.RESET_ALL} record di indicatori " +
                       f"({percentage:.1f}% del totale) per {Fore.YELLOW}{symbol}{Style.RESET_ALL} ({timeframe})")
            return records_saved > 0
            
    except Exception as e:
        logging.error(f"{Fore.RED}Errore nel salvataggio degli indicatori per {symbol} ({timeframe}): {e}{Style.RESET_ALL}")
        logging.error(traceback.format_exc())
        return False

def process_and_save_indicators(symbol: str, timeframe: str) -> bool:
    """
    Flusso di lavoro completo per elaborare e salvare gli indicatori tecnici.
    
    Args:
        symbol: Simbolo della criptovaluta
        timeframe: Timeframe (es. '1h')
        
    Returns:
        Boolean che indica il successo dell'operazione
    """
    try:
        logging.info(f"🔧 Elaborazione indicatori per {Fore.YELLOW}{symbol}{Style.RESET_ALL} ({timeframe})")
        
        # Carica i dati OHLCV
        ohlcv_df = load_ohlcv_data(symbol, timeframe)
        
        if ohlcv_df.empty:
            logging.warning(f"Nessun dato OHLCV disponibile per {symbol} ({timeframe})")
            return False
        
        # Calcola gli indicatori
        indicators_df = calculate_indicators(ohlcv_df)
        
        if indicators_df.empty:
            logging.warning(f"Impossibile calcolare gli indicatori per {symbol} ({timeframe})")
            return False
        
        # Salva gli indicatori nel database
        success = save_indicators(symbol, timeframe, indicators_df)
        
        if success:
            logging.info(f"✅ Indicatori elaborati con successo per {Fore.YELLOW}{symbol}{Style.RESET_ALL} ({timeframe})")
        else:
            logging.warning(f"❌ Impossibile salvare gli indicatori per {symbol} ({timeframe})")
        
        return success
        
    except Exception as e:
        logging.error(f"{Fore.RED}Errore nell'elaborazione degli indicatori per {symbol} ({timeframe}): {e}{Style.RESET_ALL}")
        logging.error(traceback.format_exc())
        return False

def compute_and_save_indicators(symbol: str, timeframe: str) -> bool:
    """
    Alias per process_and_save_indicators per compatibilità.
    
    Args:
        symbol: Simbolo della criptovaluta
        timeframe: Timeframe (es. '1h')
        
    Returns:
        Boolean che indica il successo dell'operazione
    """
    return process_and_save_indicators(symbol, timeframe)
