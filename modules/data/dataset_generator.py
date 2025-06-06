#!/usr/bin/env python3
"""
Dataset Generator Module for TradingJii

This module generates supervised learning datasets from volatility time series:
- Extracts sliding windows of consecutively timestamped volatility values
- Creates labeled datasets with X (features) and y (target)
- Organizes data by pattern categories or data-driven labels
- Exports to CSV files for training supervised ML models
"""

import os
import logging
import pandas as pd
from typing import Dict, List, Tuple, Optional
from colorama import Fore, Style
from modules.data.series_segmenter import load_volatility_series, generate_subseries, categorize_series
from modules.data.data_labeler import load_price_data, generate_data_driven_labels, merge_volatility_with_labels, calculate_adaptive_thresholds
from datetime import datetime

def export_supervised_training_data(
    symbol: str,
    timeframe: str,
    output_dir: str,
    window_size: int = 7,
    threshold: float = 0.0,
    force_regeneration: bool = False,
    min_samples_per_pattern: int = 40,  # Minimum samples needed for proper train/validation split
    oversample_factor: int = 3,  # How many times to oversample small datasets
    include_timestamps: bool = True,  # Include timestamps in the output to enable joining with TA data
    use_data_driven_labels: bool = True,  # Use real price returns for labeling
    forward_periods: int = 10,  # How many periods ahead to look for returns
    buy_threshold: float = 0.02,  # Threshold for BUY signals (% return)
    sell_threshold: float = -0.02,  # Threshold for SELL signals (% return)
    adaptive_thresholds: bool = True  # Calculate thresholds based on volatility
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
    vol_df = load_volatility_series(symbol, timeframe)
    
    if vol_df.empty:
        logging.warning(f"No volatility data available for {symbol} ({timeframe})")
        return {}
    
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
            
            # Check file naming pattern to determine if they are pattern-based or class-based
            is_class_based = any(f.startswith('class_') for f in csv_files)
            
            # Read files to get record counts
            record_counts = {}
            for csv_file in csv_files:
                if is_class_based:
                    class_name = csv_file[6:-4]  # Extract name from "class_NAME.csv"
                else:
                    class_name = csv_file[4:-4]  # Extract name from "cat_PATTERN.csv"
                    
                df = pd.read_csv(os.path.join(dataset_path, csv_file))
                record_counts[class_name] = len(df)
            
            # Log summary of existing data
            total_records = sum(record_counts.values())
            num_categories = len(record_counts)
            logging.info(f"Found existing data: {Fore.GREEN}{total_records}{Style.RESET_ALL} records in "
                       f"{Fore.CYAN}{num_categories}{Style.RESET_ALL} categories")
            return record_counts
    
    # Create the directory if it doesn't exist or we're regenerating
    os.makedirs(dataset_path, exist_ok=True)
    
    # Determina quale approccio di etichettatura usare
    use_pattern_fallback = False
    
    if use_data_driven_labels:
        logging.info(f"{Fore.CYAN}Utilizzando etichettatura data-driven basata su rendimenti reali{Style.RESET_ALL}")
        
        # Carica dati di prezzo
        price_df = load_price_data(symbol, timeframe, lookback_periods=500)
        
        if price_df.empty:
            logging.warning(f"{Fore.YELLOW}Impossibile generare etichette data-driven: nessun dato di prezzo per {symbol} ({timeframe}){Style.RESET_ALL}")
            logging.warning(f"{Fore.YELLOW}Fallback automatico all'etichettatura basata su pattern{Style.RESET_ALL}")
            use_pattern_fallback = True
    
    # APPROCCIO 1: ETICHETTATURA DATA-DRIVEN (BASATA SU RENDIMENTI REALI)
    if use_data_driven_labels and not use_pattern_fallback:
        try:
            # Calcola soglie adattive se richiesto
            if adaptive_thresholds:
                buy_thresh, sell_thresh = calculate_adaptive_thresholds(price_df)
                logging.info(f"Soglie adattive calcolate: BUY > {buy_thresh:.1%}, SELL < {sell_thresh:.1%}")
            else:
                buy_thresh, sell_thresh = buy_threshold, sell_threshold
                
            # Genera etichette basate su rendimenti futuri
            labeled_df = generate_data_driven_labels(
                price_df, 
                forward_periods=forward_periods,
                buy_threshold=buy_thresh,
                sell_threshold=sell_thresh
            )
            
            if labeled_df.empty:
                logging.warning(f"{Fore.YELLOW}Nessuna etichetta generata per {symbol} ({timeframe}){Style.RESET_ALL}")
                logging.warning(f"{Fore.YELLOW}Fallback all'etichettatura basata su pattern{Style.RESET_ALL}")
                use_pattern_fallback = True
            else:
                # Unisci volatilità con etichette
                merged_df = merge_volatility_with_labels(vol_df, labeled_df, window_size)
                
                if merged_df.empty:
                    logging.warning(f"{Fore.YELLOW}Impossibile unire volatilità ed etichette per {symbol} ({timeframe}){Style.RESET_ALL}")
                    logging.warning(f"{Fore.YELLOW}Fallback all'etichettatura basata su pattern{Style.RESET_ALL}")
                    use_pattern_fallback = True
                else:
                    # Organizza per classe di segnale
                    class_dfs = {
                        "buy": merged_df[merged_df['y'] == 1],
                        "sell": merged_df[merged_df['y'] == -1],
                        "hold": merged_df[merged_df['y'] == 0]
                    }
                    
                    # Gestisci classi sbilanciate
                    class_sizes = {k: len(v) for k, v in class_dfs.items() if not v.empty}
                    if not class_sizes:
                        logging.error(f"Nessuna classe valida generata per {symbol} ({timeframe})")
                        return {}
                        
                    min_class_size = min(class_sizes.values())
                    
                    if min_class_size < min_samples_per_pattern:
                        logging.info(f"{Fore.YELLOW}Classe minima ha solo {min_class_size} campioni, "
                                   f"che è sotto il minimo di {min_samples_per_pattern}. Bilanciamento classi...{Style.RESET_ALL}")
                        
                        # Sovracampiona classi minoritarie
                        balanced_dfs = {}
                        for class_name, df in class_dfs.items():
                            if df.empty:
                                continue
                                
                            if len(df) < min_samples_per_pattern:
                                # Sovracampiona con rumore
                                import numpy as np
                                
                                # Determina quante volte duplicare
                                times_to_duplicate = min(oversample_factor, max(1, min_samples_per_pattern // len(df) + 1))
                                
                                augmented_df = df.copy()
                                for _ in range(times_to_duplicate):
                                    # Copia i dati originali
                                    new_samples = df.copy()
                                    
                                    # Aggiungi piccolo rumore alle features
                                    for col in [f'x_{i+1}' for i in range(window_size)]:
                                        noise_factor = 0.01 * np.random.uniform(0.5, 1.5, size=len(new_samples))
                                        new_samples[col] = new_samples[col] * (1 + noise_factor * np.sign(new_samples[col]))
                                    
                                    # Aggiungi i campioni aumentati
                                    augmented_df = pd.concat([augmented_df, new_samples], ignore_index=True)
                                
                                balanced_dfs[class_name] = augmented_df
                            else:
                                balanced_dfs[class_name] = df
                        
                        class_dfs = balanced_dfs
                    
                    # Esporta ciascuna classe in un file CSV separato
                    class_record_counts = {}
                    total_records = 0
                    
                    for class_name, df in class_dfs.items():
                        if df.empty:
                            continue
                            
                        # Imposta filename
                        filename = os.path.join(dataset_path, f"class_{class_name}.csv")
                        
                        # Salva in CSV
                        df.to_csv(filename, index=False)
                        
                        # Traccia conteggi
                        num_records = len(df)
                        class_record_counts[class_name] = num_records
                        total_records += num_records
                        
                        logging.info(f"Classe {Fore.CYAN}{class_name}{Style.RESET_ALL}: "
                                   f"Esportati {Fore.GREEN}{num_records}{Style.RESET_ALL} record "
                                   f"in {filename}")
                    
                    # Log sommario
                    num_classes = len(class_record_counts)
                    logging.info(f"\n{Fore.GREEN}=== RIEPILOGO ESPORTAZIONE DATASET ==={Style.RESET_ALL}")
                    logging.info(f"Simbolo: {Fore.YELLOW}{symbol}{Style.RESET_ALL}, "
                               f"Timeframe: {Fore.YELLOW}{timeframe}{Style.RESET_ALL}")
                    logging.info(f"Totale record: {Fore.GREEN}{total_records}{Style.RESET_ALL}, "
                               f"Classi: {Fore.CYAN}{num_classes}{Style.RESET_ALL}")
                    logging.info(f"Dataset salvati in: {Fore.MAGENTA}{dataset_path}{Style.RESET_ALL}")
                    
                    return class_record_counts
        except Exception as e:
            logging.error(f"Errore nell'etichettatura data-driven: {e}")
            logging.warning(f"{Fore.YELLOW}Fallback all'etichettatura basata su pattern{Style.RESET_ALL}")
            use_pattern_fallback = True
    
    # APPROCCIO 2: ETICHETTATURA BASATA SU PATTERN
    if use_pattern_fallback or not use_data_driven_labels:
        logging.info(f"{Fore.YELLOW}Utilizzando etichettatura basata su pattern di volatilità{Style.RESET_ALL}")
        
        # Generate subseries (windows with targets)
        logging.info(f"Generating subseries with window_size={window_size}")
        subseries = generate_subseries(vol_df, window_size)
        
        if not subseries:
            logging.warning(f"Could not generate subseries for {symbol} ({timeframe})")
            return {}
        
        # Organize by categories
        categorized_data: Dict[str, List[Tuple[List[float], float, str]]] = {}
        
        for i in range(len(vol_df) - window_size):
            window = vol_df['volatility'].iloc[i:i+window_size].tolist()
            target = vol_df['volatility'].iloc[i+window_size]  # Next value is the target
            timestamp = vol_df['timestamp'].iloc[i+window_size]  # Timestamp corresponding to the target
            
            # Get the binary pattern for this window
            pattern = categorize_series(window, threshold)
            
            if pattern not in categorized_data:
                categorized_data[pattern] = []
            
            categorized_data[pattern].append((window, target, timestamp))
        
        # Export each category to a separate CSV file
        pattern_record_counts = {}
        total_records = 0
        
        for pattern, data in categorized_data.items():
            # Create DataFrame with x_1, x_2, ..., x_n columns, y column, timestamp, and pattern
            rows = []
            for window, target, timestamp in data:
                row = {f'x_{i+1}': val for i, val in enumerate(window)}
                row['y'] = target
                row['pattern'] = pattern
                # Convert timestamp to ISO 8601 format for joining with TA data
    
                if isinstance(timestamp, str):
                    row['timestamp'] = timestamp  # Already a string
                elif isinstance(timestamp, (int, float)):
                    # Assume timestamp is in UNIX seconds
                    row['timestamp'] = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                elif hasattr(timestamp, 'strftime'):
                    row['timestamp'] = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    logging.warning(f"Tipo timestamp non gestito: {type(timestamp)}. Lo converto a stringa grezza.")
                    row['timestamp'] = str(timestamp)
    
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
    
    # Non dovremmo mai arrivare qui, ma in caso di errori inaspettati
    logging.error(f"Nessun metodo di etichettatura è stato completato con successo per {symbol} ({timeframe})")
    return {}

def export_all_supervised_data(
    symbols: List[str],
    timeframes: List[str],
    output_dir: str,
    window_size: int = 7,
    threshold: float = 0.0,
    force_regeneration: bool = False,
    use_data_driven_labels: bool = True
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
                    oversample_factor=3,  # Augment small datasets by up to 3x
                    use_data_driven_labels=use_data_driven_labels,  # Use data-driven labels
                    forward_periods=10,  # Look ahead 10 periods for returns
                    adaptive_thresholds=True  # Calculate thresholds based on volatility
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
    force_regeneration: bool = False,
    use_data_driven_labels: bool = True
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
        force_regeneration=force_regeneration,
        use_data_driven_labels=use_data_driven_labels  # Use data-driven labels
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
