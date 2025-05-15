#!/usr/bin/env python3
"""
TradingJii Web Application

This is the main entry point for the TradingJii web application.
It initializes both the Flask API backend and serves the frontend.
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import sqlite3
import pandas as pd
import os
import json

# Import modules
from modules.utils.config import DB_FILE
from modules.data.series_segmenter import categorize_series, build_categorized_dataset

# Initialize Flask app
app = Flask(__name__, static_folder='frontend')
CORS(app)  # Enable Cross-Origin Resource Sharing

# Utility functions
def get_db_connection():
    """Create a connection to the SQLite database"""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row  # Return rows as dictionaries
    return conn

def get_available_symbols():
    """Get a list of all available cryptocurrency symbols in the database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Query to get all unique symbols from data_5m table
    cursor.execute("SELECT DISTINCT symbol FROM data_5m ORDER BY symbol")
    
    symbols = [row['symbol'] for row in cursor.fetchall()]
    conn.close()
    
    return symbols

def get_ohlcv_data(symbol, timeframe, limit=100):
    """Get OHLCV data for a specific cryptocurrency and timeframe"""
    conn = get_db_connection()
    
    query = f"""
    SELECT timestamp, open, high, low, close, volume
    FROM data_{timeframe}
    WHERE symbol = ?
    ORDER BY timestamp DESC
    LIMIT ?
    """
    
    df = pd.read_sql_query(query, conn, params=(symbol, limit))
    conn.close()
    
    # Convert to list of dictionaries for JSON serialization
    data = df.to_dict(orient='records')
    return data

def get_volatility_data(symbol, timeframe, limit=100):
    """Get volatility data for a specific cryptocurrency and timeframe"""
    conn = get_db_connection()
    
    query = f"""
    SELECT timestamp, volatility
    FROM volatility_{timeframe}
    WHERE symbol = ?
    ORDER BY timestamp DESC
    LIMIT ?
    """
    
    df = pd.read_sql_query(query, conn, params=(symbol, limit))
    conn.close()
    
    # Convert to list of dictionaries for JSON serialization
    data = df.to_dict(orient='records')
    return data

def get_indicator_data(symbol, timeframe, limit=100):
    """Get technical indicator data for a specific cryptocurrency and timeframe"""
    conn = get_db_connection()
    
    query = f"""
    SELECT timestamp, sma9, sma20, sma50, ema20, ema50, ema200, 
           rsi14, stoch_k, stoch_d, macd, macd_signal, macd_hist,
           atr14, bbands_upper, bbands_middle, bbands_lower,
           obv, vwap, volume_sma20, adx14
    FROM ta_{timeframe}
    WHERE symbol = ?
    ORDER BY timestamp DESC
    LIMIT ?
    """
    
    df = pd.read_sql_query(query, conn, params=(symbol, limit))
    conn.close()
    
    # Convert to list of dictionaries for JSON serialization
    data = df.to_dict(orient='records')
    return data

def get_pattern_data(symbol, timeframe, window_size=7):
    """Get pattern data for a specific cryptocurrency and timeframe"""
    try:
        # Use the build_categorized_dataset function to get the patterns
        categories = build_categorized_dataset(symbol, timeframe, window_size)
        
        # Extract categories and count of samples in each
        pattern_data = {}
        for pattern, samples in categories.items():
            pattern_data[pattern] = len(samples)
        
        return pattern_data
    except Exception as e:
        # Log the error and return an empty dictionary
        print(f"Error generating patterns for {symbol} in {timeframe}: {str(e)}")
        return {}

# API Routes
@app.route('/api/symbols')
def api_symbols():
    """API endpoint to get all available cryptocurrency symbols"""
    try:
        symbols = get_available_symbols()
        return jsonify({"status": "success", "data": symbols})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/ohlcv/<path:symbol>/<timeframe>')
def api_ohlcv(symbol, timeframe):
    """API endpoint to get OHLCV data for a specific cryptocurrency and timeframe"""
    try:
        limit = request.args.get('limit', default=100, type=int)
        data = get_ohlcv_data(symbol, timeframe, limit)
        return jsonify({"status": "success", "data": data})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/volatility/<path:symbol>/<timeframe>')
def api_volatility(symbol, timeframe):
    """API endpoint to get volatility data for a specific cryptocurrency and timeframe"""
    try:
        limit = request.args.get('limit', default=100, type=int)
        data = get_volatility_data(symbol, timeframe, limit)
        return jsonify({"status": "success", "data": data})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/patterns/<path:symbol>/<timeframe>')
def api_patterns(symbol, timeframe):
    """API endpoint to get pattern data for a specific cryptocurrency and timeframe"""
    try:
        window_size = request.args.get('window_size', default=7, type=int)
        data = get_pattern_data(symbol, timeframe, window_size)
        return jsonify({"status": "success", "data": data})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/indicators/<path:symbol>/<timeframe>')
def api_indicators(symbol, timeframe):
    """API endpoint to get technical indicator data for a specific cryptocurrency and timeframe"""
    try:
        limit = request.args.get('limit', default=100, type=int)
        data = get_indicator_data(symbol, timeframe, limit)
        return jsonify({"status": "success", "data": data})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Frontend routes
@app.route('/')
def index():
    """Serve the main frontend page"""
    return send_from_directory('frontend', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files like CSS, JS, etc."""
    return send_from_directory('frontend', path)

if __name__ == '__main__':
    # Make sure the frontend directory exists
    os.makedirs('frontend', exist_ok=True)
    
    # Run the Flask app
    app.run(debug=True, port=5000)
