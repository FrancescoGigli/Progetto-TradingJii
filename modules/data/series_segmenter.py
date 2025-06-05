#!/usr/bin/env python3
"""
Series Segmenter Module for TradingJii

This module processes volatility time series to:
- Extract sliding windows of volatility values
- Label the values with binary up/down categorization
- Group the data into category-specific datasets for training models
"""

import sqlite3
import logging
import pandas as pd
from typing import List, Tuple, Dict
from colorama import Fore, Style
from modules.utils.config import DB_FILE

def load_volatility_series(symbol: str, timeframe: str) -> pd.DataFrame:
    """
    Load timestamp, volatility for a given symbol and timeframe from the market data table.
    
    Args:
        symbol: The cryptocurrency symbol (e.g., "BTC/USDT")
        timeframe: The timeframe (e.g., "5m")
        
    Returns:
        A pandas DataFrame with timestamp and volatility columns, sorted chronologically
    """
    table_name = f"market_data_{timeframe}"
    
    try:
        with sqlite3.connect(DB_FILE) as conn:
            query = f"""
                SELECT timestamp, volatility
                FROM {table_name}
                WHERE symbol = ?
                ORDER BY timestamp ASC
            """
            
            df = pd.read_sql_query(query, conn, params=(symbol,))
            
            if df.empty:
                logging.warning(f"No volatility data found for {symbol} in timeframe {timeframe}")
                return pd.DataFrame(columns=['timestamp', 'volatility'])
                
            return df
            
    except Exception as e:
        logging.error(f"Error loading volatility series for {symbol} ({timeframe}): {e}")
        return pd.DataFrame(columns=['timestamp', 'volatility'])

def generate_subseries(df: pd.DataFrame, window_size: int = 7) -> List[Tuple[List[float], float]]:
    """
    Split the volatility series into overlapping subseries of window_size + 1.
    
    Args:
        df: DataFrame with volatility time series
        window_size: Size of the sliding window (default: 7)
        
    Returns:
        List of tuples: ([v1, v2, ..., vL], v_target)
    """
    if df.empty or len(df) < window_size + 1:
        logging.warning(f"Insufficient data to generate subseries (need at least {window_size + 1} data points)")
        return []
    
    # Extract volatility as a list
    volatility_values = df['volatility'].tolist()
    
    # Generate subseries
    subseries = []
    for i in range(len(volatility_values) - window_size):
        window = volatility_values[i:i + window_size]  # First L values
        target = volatility_values[i + window_size]    # Next value (L+1)
        subseries.append((window, target))
    
    return subseries

def categorize_series(sequence: List[float], threshold: float = 0.0) -> str:
    """
    Convert a list of volatility values into a binary pattern string.
    
    Args:
        sequence: List of volatility values
        threshold: Value for comparison (default: 0.0)
        
    Returns:
        String of '1' for values > threshold, '0' otherwise
    """
    if not sequence:
        return ""
    
    # Generate binary pattern
    pattern = ''.join('1' if v > threshold else '0' for v in sequence)
    return pattern

def build_categorized_dataset(
    symbol: str,
    timeframe: str,
    window_size: int = 7,
    threshold: float = 0.0
) -> Dict[str, List[Tuple[List[float], float]]]:
    """
    Produce a mapping of behavior categories to corresponding training samples.
    
    Args:
        symbol: The cryptocurrency symbol
        timeframe: The timeframe
        window_size: Size of the sliding window (default: 7)
        threshold: Value for categorization (default: 0.0)
        
    Returns:
        Dictionary mapping category patterns to lists of (window, target) tuples
    """
    # Load volatility data from database
    df = load_volatility_series(symbol, timeframe)
    if df.empty:
        logging.warning(f"No data available for {symbol} ({timeframe})")
        return {}
    
    # Generate subseries
    subseries = generate_subseries(df, window_size)
    if not subseries:
        logging.warning(f"Could not generate subseries for {symbol} ({timeframe})")
        return {}
    
    # Categorize subseries and organize by category
    categorized_data = {}
    for window, target in subseries:
        category = categorize_series(window, threshold)
        
        if category not in categorized_data:
            categorized_data[category] = []
        
        categorized_data[category].append((window, target))
    
    # Log summary
    total_samples = len(subseries)
    total_categories = len(categorized_data)
    logging.info(f"For {Fore.YELLOW}{symbol}{Style.RESET_ALL} ({timeframe}): "
                f"Created {Fore.CYAN}{total_categories}{Style.RESET_ALL} categories "
                f"from {Fore.GREEN}{total_samples}{Style.RESET_ALL} samples")
    
    return categorized_data

def process_all_categories(
    symbols: List[str],
    timeframes: List[str],
    window_size: int = 7,
    threshold: float = 0.0
) -> List[Dict]:
    """
    Orchestrate processing of multiple symbols and timeframes into categorized datasets.
    
    Args:
        symbols: List of cryptocurrency symbols
        timeframes: List of timeframes
        window_size: Size of the sliding window (default: 7)
        threshold: Value for categorization (default: 0.0)
        
    Returns:
        List of dictionaries with symbol, timeframe, and categories
    """
    results = []
    
    for symbol in symbols:
        for timeframe in timeframes:
            try:
                logging.info(f"Processing {Fore.YELLOW}{symbol}{Style.RESET_ALL} ({timeframe})")
                
                # Build categorized dataset
                categories = build_categorized_dataset(symbol, timeframe, window_size, threshold)
                
                if categories:
                    # Add to results
                    results.append({
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "categories": categories
                    })
                    
                    # Count total samples
                    total_samples = sum(len(samples) for samples in categories.values())
                    logging.info(f"Added {Fore.GREEN}{total_samples}{Style.RESET_ALL} samples "
                               f"in {Fore.CYAN}{len(categories)}{Style.RESET_ALL} categories "
                               f"for {Fore.YELLOW}{symbol}{Style.RESET_ALL} ({timeframe})")
            except Exception as e:
                logging.error(f"Error processing {symbol} ({timeframe}): {e}")
                import traceback
                logging.error(traceback.format_exc())
    
    # Log summary
    if results:
        total_symbols = len(set(r["symbol"] for r in results))
        total_timeframes = len(set(r["timeframe"] for r in results))
        total_datasets = len(results)
        
        logging.info(f"\n{Fore.GREEN}=== SEGMENTATION SUMMARY ==={Style.RESET_ALL}")
        logging.info(f"Processed {Fore.YELLOW}{total_symbols}{Style.RESET_ALL} symbols "
                   f"x {Fore.YELLOW}{total_timeframes}{Style.RESET_ALL} timeframes")
        logging.info(f"Created {Fore.GREEN}{total_datasets}{Style.RESET_ALL} categorized datasets")
    
    return results
