#!/usr/bin/env python3
"""
Dataset Generator Module for TradingJii

This module generates supervised learning datasets from volatility time series:
- Extracts sliding windows of consecutively timestamped volatility values
- Creates labeled datasets with X (features) and y (target)
- Organizes data by pattern categories
- Exports to CSV files for training supervised ML models
"""

import os
import logging
import pandas as pd
from typing import Dict, List, Tuple, Optional
from colorama import Fore, Style
from modules.data.series_segmenter import load_volatility_series, generate_subseries, categorize_series

def export_supervised_training_data(
    symbol: str,
    timeframe: str,
    output_dir: str,
    window_size: int = 7,
    threshold: float = 0.0
) -> Dict[str, int]:
    """
    Generate and export supervised training data from volatility time series.
    
    Args:
        symbol: The cryptocurrency symbol (e.g., "BTC/USDT")
        timeframe: The timeframe (e.g., "5m")
        output_dir: Directory where datasets will be saved
        window_size: Size of the sliding window (default: 7)
        threshold: Value for categorization (default: 0.0)
        
    Returns:
        Dictionary mapping pattern categories to number of records exported
    """
    # Load volatility data from database
    logging.info(f"Loading volatility data for {Fore.YELLOW}{symbol}{Style.RESET_ALL} ({timeframe})")
    df = load_volatility_series(symbol, timeframe)
    
    if df.empty:
        logging.warning(f"No volatility data available for {symbol} ({timeframe})")
        return {}
    
    # Generate subseries (windows with targets)
    logging.info(f"Generating subseries with window_size={window_size}")
    subseries = generate_subseries(df, window_size)
    
    if not subseries:
        logging.warning(f"Could not generate subseries for {symbol} ({timeframe})")
        return {}
    
    # Organize by categories
    categorized_data: Dict[str, List[Tuple[List[float], float]]] = {}
    
    for window, target in subseries:
        # Get the binary pattern for this window
        pattern = categorize_series(window, threshold)
        
        if pattern not in categorized_data:
            categorized_data[pattern] = []
        
        categorized_data[pattern].append((window, target))
    
    # Create output directory structure if it doesn't exist
    # Format: datasets/{symbol}/{timeframe}/
    # Sanitize symbol for safe directory name (remove invalid characters)
    symbol_safe = symbol.replace('/', '_').replace(':', '').replace('\\', '').replace('*', '')
    symbol_safe = symbol_safe.replace('?', '').replace('"', '').replace('<', '').replace('>', '')
    symbol_safe = symbol_safe.replace('|', '')
    dataset_path = os.path.join(output_dir, symbol_safe, timeframe)
    os.makedirs(dataset_path, exist_ok=True)
    
    # Export each category to a separate CSV file
    pattern_record_counts = {}
    total_records = 0
    
    for pattern, data in categorized_data.items():
        # Create DataFrame with x_1, x_2, ..., x_n columns and y column
        rows = []
        for window, target in data:
            row = {f'x_{i+1}': val for i, val in enumerate(window)}
            row['y'] = target
            row['pattern'] = pattern
            rows.append(row)
        
        # Convert to DataFrame
        cat_df = pd.DataFrame(rows)
        
        # Set filename
        filename = os.path.join(dataset_path, f"cat_{pattern}.csv")
        
        # Save to CSV
        cat_df.to_csv(filename, index=False)
        
        # Track counts
        num_records = len(cat_df)
        pattern_record_counts[pattern] = num_records
        total_records += num_records
        
        logging.info(f"Category {Fore.CYAN}{pattern}{Style.RESET_ALL}: "
                    f"Exported {Fore.GREEN}{num_records}{Style.RESET_ALL} records "
                    f"to {filename}")
    
    # Log summary
    num_categories = len(pattern_record_counts)
    logging.info(f"\n{Fore.GREEN}=== DATASET EXPORT SUMMARY ==={Style.RESET_ALL}")
    logging.info(f"Symbol: {Fore.YELLOW}{symbol}{Style.RESET_ALL}, "
               f"Timeframe: {Fore.YELLOW}{timeframe}{Style.RESET_ALL}")
    logging.info(f"Total records: {Fore.GREEN}{total_records}{Style.RESET_ALL}, "
               f"Categories: {Fore.CYAN}{num_categories}{Style.RESET_ALL}")
    logging.info(f"Datasets saved to: {Fore.MAGENTA}{dataset_path}{Style.RESET_ALL}")
    
    return pattern_record_counts

def export_all_supervised_data(
    symbols: List[str],
    timeframes: List[str],
    output_dir: str,
    window_size: int = 7,
    threshold: float = 0.0
) -> List[Dict]:
    """
    Export supervised datasets for multiple symbols and timeframes.
    
    Args:
        symbols: List of cryptocurrency symbols
        timeframes: List of timeframes
        output_dir: Base directory for saving datasets
        window_size: Size of the sliding window (default: 7)
        threshold: Value for categorization (default: 0.0)
        
    Returns:
        List of dictionaries with symbol, timeframe, and export statistics
    """
    results = []
    
    # Create base output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for symbol in symbols:
        for timeframe in timeframes:
            try:
                logging.info(f"Processing {Fore.YELLOW}{symbol}{Style.RESET_ALL} ({timeframe})")
                
                # Export supervised training data
                pattern_counts = export_supervised_training_data(
                    symbol,
                    timeframe,
                    output_dir,
                    window_size,
                    threshold
                )
                
                if pattern_counts:
                    # Add to results
                    results.append({
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "pattern_counts": pattern_counts,
                        "total_records": sum(pattern_counts.values()),
                        "num_categories": len(pattern_counts)
                    })
            except Exception as e:
                logging.error(f"Error exporting data for {symbol} ({timeframe}): {e}")
                import traceback
                logging.error(traceback.format_exc())
    
    # Log overall summary
    if results:
        total_symbols = len(set(r["symbol"] for r in results))
        total_timeframes = len(set(r["timeframe"] for r in results))
        total_datasets = len(results)
        total_records = sum(r["total_records"] for r in results)
        
        logging.info(f"\n{Fore.GREEN}=== OVERALL EXPORT SUMMARY ==={Style.RESET_ALL}")
        logging.info(f"Processed {Fore.YELLOW}{total_symbols}{Style.RESET_ALL} symbols "
                   f"× {Fore.YELLOW}{total_timeframes}{Style.RESET_ALL} timeframes")
        logging.info(f"Created {Fore.GREEN}{total_datasets}{Style.RESET_ALL} datasets "
                   f"with {Fore.CYAN}{total_records}{Style.RESET_ALL} total records")
        logging.info(f"All datasets saved to: {Fore.MAGENTA}{output_dir}{Style.RESET_ALL}")
    
    return results

if __name__ == "__main__":
    from modules.utils.logging_setup import setup_logging
    
    # Set up logging
    setup_logging(level=logging.INFO)
    
    # Example usage
    symbols = ["BTC/USDT"]
    timeframes = ["5m"]
    output_dir = "datasets"
    
    export_all_supervised_data(symbols, timeframes, output_dir)
