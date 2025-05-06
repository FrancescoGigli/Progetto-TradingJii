#!/usr/bin/env python3
"""
Trading Dashboard
================

Una semplice dashboard web per visualizzare dati di trading dalle criptovalute
presenti nel database, inclusi pattern e previsioni.
"""

import os
import sqlite3
import json
import pandas as pd
import numpy as np
from flask import Flask, render_template, jsonify, request
from datetime import datetime, timedelta
import logging
from src.utils.config import DB_FILE, ENABLED_TIMEFRAMES
from src.data.model_selector import get_model_selector
from src.data.subseries_utils import binary_to_pattern
from src.data.db_manager import get_symbol_data, get_category_transitions, get_top_categories

# Configura logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Inizializza l'applicazione Flask
app = Flask(__name__)

# Verifica che il database esista
if not os.path.exists(DB_FILE):
    logging.error(f"Database file {DB_FILE} not found. Please run the volatility pipeline first.")

# Funzioni di utilità
def get_connection():
    """Crea una connessione al database SQLite"""
    return sqlite3.connect(DB_FILE)

def get_available_timeframes():
    """Recupera i timeframe disponibili nel database"""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'data_%'")
        tables = [row[0].replace('data_', '') for row in cursor.fetchall()]
        return tables
    except Exception as e:
        logging.error(f"Error retrieving timeframes: {e}")
        return []
    finally:
        conn.close()

def get_all_symbols(timeframe):
    """Recupera tutti i simboli disponibili per un determinato timeframe"""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(f"""
            SELECT DISTINCT symbol FROM data_{timeframe}
            ORDER BY symbol
        """)
        symbols = [row[0] for row in cursor.fetchall()]
        return symbols
    except Exception as e:
        logging.error(f"Error retrieving symbols: {e}")
        return []
    finally:
        conn.close()

def get_crypto_data(symbol, timeframe, limit=100):
    """Recupera i dati di una criptovaluta dal database"""
    conn = get_connection()
    
    try:
        query = f"""
            SELECT 
                timestamp, 
                open, high, low, close, volume,
                close_volatility, open_volatility, high_volatility, low_volatility,
                volume_change, historical_volatility
            FROM data_{timeframe}
            WHERE symbol = ?
            ORDER BY timestamp DESC
            LIMIT {limit}
        """
        
        df = pd.read_sql_query(query, conn, params=(symbol,))
        
        if not df.empty:
            # Converti timestamp in formato leggibile
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['formatted_time'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        
        return df
    except Exception as e:
        logging.error(f"Error retrieving data for {symbol} ({timeframe}): {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def get_current_pattern_prediction(symbol, timeframe):
    """Ottieni il pattern corrente e la previsione per un simbolo"""
    df = get_symbol_data(symbol, timeframe, limit=50)
    
    if df is None or df.empty or 'close_volatility' not in df.columns:
        return {
            "status": "error",
            "message": f"Insufficient data for {symbol} ({timeframe})"
        }
        
    # Inizializza il modello di previsione
    try:
        selector = get_model_selector()
        
        # Identifica il pattern corrente
        current_category, pattern = selector.categorize_current_data(df)
        
        if current_category is None:
            return {
                "status": "error",
                "message": f"Could not identify current pattern for {symbol}"
            }
            
        # Predici il prossimo pattern
        next_cat, next_pattern, probability = selector.predict_next_category(current_category)
        
        # Ottieni previsioni multi-step
        multi_step = []
        if next_cat:
            predictions = selector.predict_multiple_steps(current_category, steps=3)
            for cat, pat, prob in predictions:
                multi_step.append({
                    "category": cat,
                    "pattern": pat,
                    "probability": round(prob * 100, 1)
                })
        
        # Genera un consiglio di trading basato sulla previsione
        trading_signal = "HOLD"
        confidence = 0
        
        if next_pattern and probability:
            # Conta le frecce su e giù nel pattern previsto
            up_count = next_pattern.count('↑')
            down_count = next_pattern.count('↓')
            
            # Calcola la confidenza basata sulla probabilità e la direzione dominante
            if probability > 0.6:  # Soglia minima di probabilità
                if up_count > down_count:
                    trading_signal = "BUY"
                    confidence = (probability * 0.7 + (up_count / len(next_pattern)) * 0.3) * 100
                elif down_count > up_count:
                    trading_signal = "SELL"
                    confidence = (probability * 0.7 + (down_count / len(next_pattern)) * 0.3) * 100
                else:
                    trading_signal = "HOLD"
                    confidence = probability * 50  # Confidenza ridotta per segnale neutro
        
        return {
            "status": "success",
            "current_category": current_category,
            "current_pattern": pattern,
            "next_category": next_cat,
            "next_pattern": next_pattern,
            "probability": round(probability * 100, 1) if probability else 0,
            "multi_step": multi_step,
            "trading_signal": trading_signal,
            "confidence": round(confidence, 1),
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    except Exception as e:
        logging.error(f"Error predicting pattern for {symbol}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {
            "status": "error",
            "message": f"Error in pattern prediction: {str(e)}"
        }

def get_crypto_summary(symbol, timeframe):
    """Ottieni un riepilogo dei dati recenti per una criptovaluta"""
    df = get_crypto_data(symbol, timeframe, limit=50)
    
    if df.empty:
        return {
            "symbol": symbol,
            "status": "error",
            "message": f"No data for {symbol} ({timeframe})"
        }
    
    try:
        # Prendi i dati più recenti
        latest = df.iloc[0]
        prev = df.iloc[1] if len(df) > 1 else latest
        
        # Calcola la variazione percentuale
        price_change = (latest['close'] - prev['close']) / prev['close'] * 100
        
        # Calcola la volatilità media delle ultime 24 ore
        avg_volatility = df['close_volatility'].abs().mean() if 'close_volatility' in df else 0
        
        # Ottieni pattern e previsione
        pattern_data = get_current_pattern_prediction(symbol, timeframe)
        
        # Costruisci il riepilogo
        summary = {
            "symbol": symbol,
            "price": round(latest['close'], 6),
            "price_change_percent": round(price_change, 2),
            "volume": latest['volume'],
            "avg_volatility": round(avg_volatility, 2),
            "last_updated": latest['formatted_time'],
            "pattern": pattern_data.get("current_pattern", "N/A"),
            "next_pattern": pattern_data.get("next_pattern", "N/A"),
            "probability": pattern_data.get("probability", 0),
            "trading_signal": pattern_data.get("trading_signal", "HOLD"),
            "confidence": pattern_data.get("confidence", 0),
            "status": "success"
        }
        
        return summary
    
    except Exception as e:
        logging.error(f"Error creating summary for {symbol}: {e}")
        return {
            "symbol": symbol,
            "status": "error",
            "message": f"Error: {str(e)}"
        }

# Definizione delle route
@app.route('/')
def index():
    """Pagina principale della dashboard"""
    # Verifica la disponibilità del database
    if not os.path.exists(DB_FILE):
        return render_template('error.html', 
                              message="Database non trovato. Eseguire prima il pipeline di volatilità.")
    
    # Recupera i timeframe disponibili
    timeframes = get_available_timeframes()
    if not timeframes:
        return render_template('error.html', 
                              message="Nessun timeframe trovato nel database. Eseguire prima il pipeline di volatilità.")
    
    # Usa il primo timeframe disponibile
    default_timeframe = timeframes[0]
    
    return render_template('index.html', 
                          timeframes=timeframes, 
                          default_timeframe=default_timeframe)

@app.route('/api/timeframes')
def api_timeframes():
    """API: Restituisce i timeframe disponibili"""
    timeframes = get_available_timeframes()
    return jsonify({"timeframes": timeframes})

@app.route('/api/symbols/<timeframe>')
def api_symbols(timeframe):
    """API: Restituisce i simboli disponibili per un timeframe"""
    symbols = get_all_symbols(timeframe)
    return jsonify({"symbols": symbols})

@app.route('/api/data/<symbol>/<timeframe>')
def api_data(symbol, timeframe):
    """API: Restituisce i dati per un simbolo e timeframe"""
    limit = request.args.get('limit', 100, type=int)
    df = get_crypto_data(symbol, timeframe, limit)
    
    if df.empty:
        return jsonify({"status": "error", "message": f"No data found for {symbol} ({timeframe})"})
    
    # Converti DataFrame in formato JSON
    data = df.to_dict(orient='records')
    return jsonify({"status": "success", "data": data})

@app.route('/api/summary/<symbol>/<timeframe>')
def api_summary(symbol, timeframe):
    """API: Restituisce un riepilogo dei dati per un simbolo"""
    summary = get_crypto_summary(symbol, timeframe)
    return jsonify(summary)

@app.route('/api/pattern/<symbol>/<timeframe>')
def api_pattern(symbol, timeframe):
    """API: Restituisce il pattern corrente e la previsione per un simbolo"""
    pattern_data = get_current_pattern_prediction(symbol, timeframe)
    return jsonify(pattern_data)

@app.route('/api/market/<timeframe>')
def api_market(timeframe):
    """API: Restituisce un riepilogo del mercato per tutti i simboli"""
    symbols = get_all_symbols(timeframe)
    limit = request.args.get('limit', 20, type=int)
    
    # Limita il numero di simboli per prestazioni
    symbols = symbols[:limit]
    
    market_data = []
    for symbol in symbols:
        summary = get_crypto_summary(symbol, timeframe)
        market_data.append(summary)
    
    # Ordina per volume decrescente
    market_data.sort(key=lambda x: x.get('volume', 0), reverse=True)
    
    return jsonify({"status": "success", "market": market_data})

# Avvio dell'applicazione
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
