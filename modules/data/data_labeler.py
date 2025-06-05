#!/usr/bin/env python3
"""
Data Labeler Module for TradingJii ML

Genera etichette di trading (BUY/SELL/HOLD) basate su rendimenti reali di mercato
anziché su euristiche predefinite.
"""

import sqlite3
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
from colorama import Fore, Style

from modules.data.db_manager import DB_FILE

def load_price_data(symbol: str, timeframe: str, lookback_periods: int = 500) -> pd.DataFrame:
    """
    Carica dati di prezzo dal database per generare etichette.
    
    Args:
        symbol: Simbolo della criptovaluta
        timeframe: Timeframe temporale
        lookback_periods: Numero di periodi da caricare
        
    Returns:
        DataFrame con prezzi OHLCV e timestamp
    """
    table_name = f"data_{timeframe}"
    
    try:
        with sqlite3.connect(DB_FILE) as conn:
            query = f"""
                SELECT timestamp, open, high, low, close, volume
                FROM {table_name}
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """
            
            df = pd.read_sql_query(query, conn, params=(symbol, lookback_periods))
            
            if df.empty:
                logging.warning(f"Nessun dato di prezzo trovato per {symbol} ({timeframe})")
                return pd.DataFrame()
            
            # Converti timestamp in datetime e ordina in ordine cronologico
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            return df
    
    except Exception as e:
        logging.error(f"Errore nel caricamento dei dati di prezzo per {symbol} ({timeframe}): {e}")
        return pd.DataFrame()

def generate_data_driven_labels(
    price_df: pd.DataFrame, 
    forward_periods: int = 10, 
    buy_threshold: float = 0.02, 
    sell_threshold: float = -0.02
) -> pd.DataFrame:
    """
    Genera etichette di trading basate sui rendimenti futuri.
    
    Args:
        price_df: DataFrame con dati OHLCV
        forward_periods: Periodi futuri da considerare
        buy_threshold: Soglia di rendimento per generare segnale BUY
        sell_threshold: Soglia di rendimento per generare segnale SELL
        
    Returns:
        DataFrame con etichette generate (1=BUY, -1=SELL, 0=HOLD)
    """
    if price_df.empty or len(price_df) < forward_periods + 1:
        logging.warning(f"Dati insufficienti per generare etichette")
        return pd.DataFrame()
    
    # Crea copia per non modificare il dataframe originale
    df = price_df.copy()
    
    # Calcola rendimenti percentuali futuri
    df['future_return'] = df['close'].pct_change(periods=forward_periods).shift(-forward_periods)
    
    # Calcola volatilità per normalizzare le soglie (opzionale)
    df['volatility'] = df['close'].pct_change().rolling(20).std()
    
    # Inizializza colonna delle etichette (0 = HOLD)
    df['signal'] = 0
    
    # Genera segnali BUY dove il rendimento futuro supera la soglia
    df.loc[df['future_return'] > buy_threshold, 'signal'] = 1
    
    # Genera segnali SELL dove il rendimento futuro è inferiore alla soglia
    df.loc[df['future_return'] < sell_threshold, 'signal'] = -1
    
    # Rimuovi le righe senza etichette (alla fine del dataframe)
    df = df.dropna(subset=['future_return', 'signal'])
    
    # Log della distribuzione delle etichette
    buy_count = (df['signal'] == 1).sum()
    sell_count = (df['signal'] == -1).sum()
    hold_count = (df['signal'] == 0).sum()
    total = len(df)
    
    logging.info(f"Etichette generate: {total} totali")
    logging.info(f"BUY: {buy_count} ({buy_count/total:.1%}), "
                f"SELL: {sell_count} ({sell_count/total:.1%}), "
                f"HOLD: {hold_count} ({hold_count/total:.1%})")
    
    return df

def merge_volatility_with_labels(
    volatility_df: pd.DataFrame, 
    labeled_price_df: pd.DataFrame,
    window_size: int = 7
) -> pd.DataFrame:
    """
    Unisce dati di volatilità con etichette basate su prezzo.
    
    Args:
        volatility_df: DataFrame con dati di volatilità
        labeled_price_df: DataFrame con etichette generate
        window_size: Dimensione della finestra di volatilità
        
    Returns:
        DataFrame unito con feature di volatilità ed etichette
    """
    # Assicurati che entrambi i dataframe abbiano timestamp come datetime
    volatility_df['timestamp'] = pd.to_datetime(volatility_df['timestamp'])
    labeled_price_df['timestamp'] = pd.to_datetime(labeled_price_df['timestamp'])
    
    # Prepara finestre di volatilità
    feature_rows = []
    
    for i in range(len(volatility_df) - window_size + 1):
        window = volatility_df['volatility'].iloc[i:i+window_size].values
        timestamp = volatility_df['timestamp'].iloc[i+window_size-1]
        
        # Crea riga di features con valori della finestra
        feature_row = {f'x_{j+1}': window[j] for j in range(window_size)}
        feature_row['timestamp'] = timestamp
        
        feature_rows.append(feature_row)
    
    if not feature_rows:
        logging.warning("Nessuna finestra di volatilità generata")
        return pd.DataFrame()
    
    # Crea dataframe di features
    features_df = pd.DataFrame(feature_rows)
    
    # Unisci con le etichette attraverso timestamp
    merged_df = pd.merge_asof(
        features_df.sort_values('timestamp'),
        labeled_price_df[['timestamp', 'signal']].sort_values('timestamp'),
        on='timestamp', 
        direction='nearest',
        tolerance=pd.Timedelta(minutes=30)  # Tollera piccole differenze di timestamp
    )
    
    # Rinomina la colonna del segnale come target
    merged_df = merged_df.rename(columns={'signal': 'y'})
    
    # Elimina le righe senza etichette
    merged_df = merged_df.dropna(subset=['y'])
    
    return merged_df

def calculate_adaptive_thresholds(
    price_df: pd.DataFrame,
    lookback_window: int = 50,
    volatility_scale: float = 1.0
) -> Tuple[float, float]:
    """
    Calcola soglie di rendimento adattive basate sulla volatilità recente.
    
    Args:
        price_df: DataFrame con dati di prezzo
        lookback_window: Finestra di osservazione per calcolare la volatilità
        volatility_scale: Fattore di scala per rendere le soglie più o meno sensibili
        
    Returns:
        Tupla di (soglia_acquisto, soglia_vendita)
    """
    if price_df.empty or len(price_df) < lookback_window:
        # Valori di default se non ci sono abbastanza dati
        return 0.02, -0.02
    
    # Calcola volatilità recente (deviazione standard dei rendimenti giornalieri)
    recent_returns = price_df['close'].pct_change().dropna()
    if len(recent_returns) > lookback_window:
        recent_returns = recent_returns.iloc[-lookback_window:]
    
    volatility = recent_returns.std()
    
    # Calcola soglie basate sulla volatilità
    # Una regola comune è utilizzare 1-2 deviazioni standard
    buy_threshold = volatility * volatility_scale * 1.5  # Segnali di acquisto leggermente più conservativi
    sell_threshold = -volatility * volatility_scale * 1.0  # Segnali di vendita più sensibili
    
    # Limita le soglie per evitare valori estremi
    buy_threshold = min(max(buy_threshold, 0.005), 0.05)  # Tra 0.5% e 5%
    sell_threshold = max(min(sell_threshold, -0.005), -0.05)  # Tra -0.5% e -5%
    
    return buy_threshold, sell_threshold
