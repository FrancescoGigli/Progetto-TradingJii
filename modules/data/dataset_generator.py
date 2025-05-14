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
    threshold: float = 0.0,
    force_regeneration: bool = False,
    min_samples_per_pattern: int = 40,  # Minimum samples needed for proper train/validation split
    oversample_factor: int = 3  # How many times to oversample small datasets
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
    
    # Check if datasets already exist for this symbol/timeframe
    if not force_regeneration and os.path.exists(dataset_path):
        # Look for at least one CSV file
        csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
        if csv_files:
            logging.info(f"{Fore.BLUE}Datasets already exist for {symbol} ({timeframe}). Use --force-ml to regenerate.{Style.RESET_ALL}")
            # Read one of the files to get pattern counts
            pattern_record_counts = {}
            for csv_file in csv_files:
                pattern = csv_file[4:-4]  # Extract pattern from "cat_PATTERN.csv"
                df = pd.read_csv(os.path.join(dataset_path, csv_file))
                pattern_record_counts[pattern] = len(df)
            
            # Log summary of existing data
            total_records = sum(pattern_record_counts.values())
            num_categories = len(pattern_record_counts)
            logging.info(f"Found existing data: {Fore.GREEN}{total_records}{Style.RESET_ALL} records in "
                       f"{Fore.CYAN}{num_categories}{Style.RESET_ALL} categories")
            return pattern_record_counts
    
    # Create the directory if it doesn't exist or we're regenerating
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
        
        # Check if we need to augment data to ensure enough samples for training and validation
        original_size = len(cat_df)
        if original_size < min_samples_per_pattern:
            logging.info(f"{Fore.YELLOW}Pattern {pattern} has only {original_size} samples, which is below the minimum of {min_samples_per_pattern}. Augmenting data...{Style.RESET_ALL}")
            
            # Calculate how many additional samples we need to reach the minimum
            augmented_df = cat_df.copy()
            
            # Add small random noise to feature columns to create additional samples
            import numpy as np
            
            # Determine how many times to duplicate and augment
            times_to_duplicate = min(oversample_factor, max(1, min_samples_per_pattern // original_size + 1))
            
            for _ in range(times_to_duplicate):
                # Copy the original data
                new_samples = cat_df.copy()
                
                # Add small random noise to features (x_1 to x_n)
                for col in [f'x_{i+1}' for i in range(window_size)]:
                    # Add random noise (0.5-1.5% of the original value)
                    noise_factor = 0.01 * np.random.uniform(0.5, 1.5, size=len(new_samples))
                    # Apply noise while preserving sign
                    new_samples[col] = new_samples[col] * (1 + noise_factor * np.sign(new_samples[col]))
                
                # Also add small noise to target (y)
                noise_factor = 0.005 * np.random.uniform(0.5, 1.5, size=len(new_samples))
                new_samples['y'] = new_samples['y'] * (1 + noise_factor * np.sign(new_samples['y']))
                
                # Append the augmented samples
                augmented_df = pd.concat([augmented_df, new_samples], ignore_index=True)
            
            # Use the augmented dataframe
            cat_df = augmented_df
            logging.info(f"{Fore.GREEN}Augmented dataset size: {len(cat_df)} samples (original: {original_size}){Style.RESET_ALL}")
        
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
    threshold: float = 0.0,
    force_regeneration: bool = False
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
                    threshold,
                    force_regeneration,
                    min_samples_per_pattern=40,  # Ensure enough samples for training and validation
                    oversample_factor=3  # Augment small datasets by up to 3x
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

def generate_ml_dataset(
    db_path: str,
    output_dir: str,
    symbols: List[str],
    timeframes: List[str],
    segment_len: int = 7,
    force_regeneration: bool = False
) -> Dict[str, int]:
    """
    Generate ML dataset from database data for all provided symbols and timeframes.
    
    Args:
        db_path: Path to the SQLite database file
        output_dir: Directory where the dataset will be saved
        symbols: List of cryptocurrency symbols
        timeframes: List of timeframes to include
        segment_len: Length of segments for pattern generation (default: 7)
        force_regeneration: Whether to force regeneration of datasets even if they exist
    
    Returns:
        Dictionary with statistics about the generation process
    """
    logging.info(f"{Fore.GREEN}=== GENERATING ML DATASET ==={Style.RESET_ALL}")
    logging.info(f"Database: {Fore.BLUE}{db_path}{Style.RESET_ALL}")
    logging.info(f"Output directory: {Fore.BLUE}{output_dir}{Style.RESET_ALL}")
    logging.info(f"Symbols: {Fore.YELLOW}{', '.join(symbols)}{Style.RESET_ALL}")
    logging.info(f"Timeframes: {Fore.CYAN}{', '.join(timeframes)}{Style.RESET_ALL}")
    logging.info(f"Segment length: {Fore.MAGENTA}{segment_len}{Style.RESET_ALL}")
    logging.info(f"Force regeneration: {Fore.YELLOW}{force_regeneration}{Style.RESET_ALL}\n")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Stats collection
    stats = {
        "total_symbols": len(symbols),
        "total_timeframes": len(timeframes),
        "symbols_processed": 0,
        "empty_data_series": 0,
        "patterns_generated": 0,
        "csv_files_written": 0,
        "total_records": 0,
    }
    
    # Generate datasets for all combinations of symbols and timeframes
    results = export_all_supervised_data(
        symbols=symbols,
        timeframes=timeframes,
        output_dir=output_dir,
        window_size=segment_len,
        threshold=0.0,  # Default threshold
        force_regeneration=force_regeneration
    )
    
    # Process results for detailed statistics
    if results:
        stats["symbols_processed"] = len(set(r["symbol"] for r in results))
        stats["total_records"] = sum(r["total_records"] for r in results)
        stats["patterns_generated"] = sum(len(r["pattern_counts"]) for r in results if "pattern_counts" in r)
        stats["csv_files_written"] = sum(len(r["pattern_counts"]) for r in results if "pattern_counts" in r)
        stats["empty_data_series"] = len(symbols) * len(timeframes) - len(results)
    
    # Log detailed statistics
    logging.info(f"\n{Fore.GREEN}=== ML DATASET GENERATION COMPLETE ==={Style.RESET_ALL}")
    
    if stats["empty_data_series"] > 0:
        logging.warning(f"{Fore.YELLOW}Empty data series: {stats['empty_data_series']}{Style.RESET_ALL}")
    
    if stats["patterns_generated"] > 0:
        logging.info(f"{Fore.GREEN}Patterns generated: {stats['patterns_generated']}{Style.RESET_ALL}")
    else:
        logging.warning(f"{Fore.YELLOW}No patterns were generated. Check if there is enough data.{Style.RESET_ALL}")
    
    if stats["csv_files_written"] > 0:
        logging.info(f"{Fore.GREEN}CSV files written: {stats['csv_files_written']}{Style.RESET_ALL}")
    else:
        logging.warning(f"{Fore.YELLOW}No CSV files were written. Check output directory permissions.{Style.RESET_ALL}")
    
    logging.info(f"Processed {Fore.YELLOW}{stats['symbols_processed']}{Style.RESET_ALL} symbols × "
               f"{Fore.CYAN}{len(timeframes)}{Style.RESET_ALL} timeframes")
    logging.info(f"Total records: {Fore.GREEN}{stats['total_records']}{Style.RESET_ALL}")
    logging.info(f"Datasets saved to: {Fore.BLUE}{os.path.abspath(output_dir)}{Style.RESET_ALL}")
    
    return stats

if __name__ == "__main__":
    from modules.utils.logging_setup import setup_logging
    
    # Set up logging
    setup_logging(level=logging.INFO)
    
    # Example usage
    symbols = ["BTC/USDT"]
    timeframes = ["5m"]
    output_dir = "datasets"
    
    export_all_supervised_data(symbols, timeframes, output_dir)
