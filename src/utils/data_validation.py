# data_validation.py

import sqlite3
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime
import time
from tabulate import tabulate
from src.utils.config import DB_FILE, ENABLED_TIMEFRAMES

def create_validation_report_directory():
    """Create a directory for validation reports if it doesn't exist."""
    report_dir = "validation_reports"
    os.makedirs(report_dir, exist_ok=True)
    return report_dir

def validate_ohlcv_data(symbol, timeframe, report_dir=None):
    """
    Validate OHLCV data for a specific symbol and timeframe.
    
    Args:
        symbol: The trading symbol to validate
        timeframe: The timeframe to validate
        report_dir: Directory to save validation reports
        
    Returns:
        dict: Validation results
    """
    if report_dir is None:
        report_dir = create_validation_report_directory()
        
    table_name = f"data_{timeframe}"
    
    # Connect to the database
    conn = sqlite3.connect(DB_FILE)
    
    # Build the query for OHLCV data
    query = f"SELECT * FROM {table_name} WHERE symbol = ? ORDER BY timestamp ASC"
    
    try:
        # Load data into DataFrame
        df = pd.read_sql_query(query, conn, params=(symbol,))
        
        if df.empty:
            logging.warning(f"No data found for {symbol} in {table_name}")
            conn.close()
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "record_count": 0,
                "has_nan": False,
                "has_zero_volume": False,
                "invalid_pairs": 0,
                "large_price_changes": 0,
                "data_quality_score": 0,
                "error": "No data found"
            }
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # Count records
        record_count = len(df)
        
        # Check for NaN values
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        has_nan = df[numeric_columns].isna().any().any()
        nan_count = df[numeric_columns].isna().sum().sum()
        
        # OHLC data sanity checks
        invalid_pairs = 0
        
        # Check if high >= low
        invalid_high_low = df[df['high'] < df['low']].shape[0]
        invalid_pairs += invalid_high_low
        
        # Check if open is between high and low
        invalid_open = df[(df['open'] > df['high']) | (df['open'] < df['low'])].shape[0]
        invalid_pairs += invalid_open
        
        # Check if close is between high and low
        invalid_close = df[(df['close'] > df['high']) | (df['close'] < df['low'])].shape[0]
        invalid_pairs += invalid_close
        
        # Check for zero or negative volume
        zero_volume_count = df[df['volume'] <= 0].shape[0]
        has_zero_volume = zero_volume_count > 0
        
        # Check for large price changes
        df['close_pct_change'] = df['close'].pct_change() * 100  # Calculate percentage change
        large_price_changes = df[df['close_pct_change'].abs() > 20].shape[0]  # Count changes > 20%
        
        # Calculate a data quality score (0-100)
        # Perfect score starts at 100 and we subtract for issues
        quality_score = 100
        
        if has_nan:
            quality_score -= min(30, nan_count)  # Reduce score for NaN values
            
        if invalid_pairs > 0:
            quality_score -= min(30, invalid_pairs)  # Reduce score for invalid OHLC relationships
            
        if has_zero_volume:
            quality_score -= min(20, zero_volume_count)  # Reduce score for zero volumes
            
        if large_price_changes > 0:
            quality_score -= min(20, large_price_changes)  # Reduce score for large price changes
            
        # Ensure score doesn't go below 0
        quality_score = max(0, quality_score)
        
        # Create CSV report
        report_filename = f"{report_dir}/{symbol.replace('/', '_')}_{timeframe}_validation.csv"
        
        # Add validation flags to DataFrame for reporting
        df['is_high_below_low'] = df['high'] < df['low']
        df['is_open_outside_range'] = (df['open'] > df['high']) | (df['open'] < df['low'])
        df['is_close_outside_range'] = (df['close'] > df['high']) | (df['close'] < df['low'])
        df['is_zero_volume'] = df['volume'] <= 0
        df['is_large_price_change'] = df['close_pct_change'].abs() > 20
        
        # Save report
        df.to_csv(report_filename, index=False)
        
        # Return validation results
        results = {
            "symbol": symbol,
            "timeframe": timeframe,
            "record_count": record_count,
            "has_nan": has_nan,
            "nan_count": nan_count,
            "invalid_pairs": invalid_pairs,
            "has_zero_volume": has_zero_volume,
            "zero_volume_count": zero_volume_count,
            "large_price_changes": large_price_changes,
            "data_quality_score": quality_score,
            "report_file": report_filename
        }
        
        return results
        
    except Exception as e:
        logging.error(f"Error validating data for {symbol} ({timeframe}): {e}")
        conn.close()
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "record_count": 0,
            "has_nan": False,
            "has_zero_volume": False,
            "invalid_pairs": 0,
            "large_price_changes": 0,
            "data_quality_score": 0,
            "error": str(e)
        }
    finally:
        conn.close()

def validate_all_symbols(timeframe=None, symbols=None, top_n=None):
    """
    Validate OHLCV data for all symbols or a specific timeframe.
    
    Args:
        timeframe: The timeframe to validate (validate all timeframes if None)
        symbols: List of specific symbols to validate (validate all if None)
        top_n: Validate only top N symbols by record count
        
    Returns:
        list: Validation results for all symbols
    """
    # Create report directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = f"ohlcv_validation_reports/{timestamp}"
    os.makedirs(report_dir, exist_ok=True)
    
    # Connect to the database
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Get all timeframes to validate
    timeframes_to_validate = [timeframe] if timeframe else ENABLED_TIMEFRAMES
    
    all_results = []
    
    for tf in timeframes_to_validate:
        table_name = f"data_{tf}"
        
        # Check if table exists
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        if not cursor.fetchone():
            logging.warning(f"Table {table_name} does not exist")
            continue
        
        # Get all symbols for this timeframe
        if symbols:
            # Validate only specified symbols
            symbols_to_validate = symbols
        else:
            # Get all symbols from the database
            cursor.execute(f"SELECT DISTINCT symbol FROM {table_name}")
            all_symbols = [row[0] for row in cursor.fetchall()]
            
            if not all_symbols:
                logging.warning(f"No symbols found in {table_name}")
                continue
                
            # Sort by record count if specified top_n
            if top_n:
                # Get record count for each symbol
                symbol_counts = []
                for symbol in all_symbols:
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE symbol = ?", (symbol,))
                    count = cursor.fetchone()[0]
                    symbol_counts.append((symbol, count))
                
                # Sort by count (descending) and get top N
                symbol_counts.sort(key=lambda x: x[1], reverse=True)
                symbols_to_validate = [s[0] for s in symbol_counts[:top_n]]
            else:
                symbols_to_validate = all_symbols
        
        # Validate each symbol
        for symbol in symbols_to_validate:
            logging.info(f"Validating {symbol} ({tf})...")
            result = validate_ohlcv_data(symbol, tf, report_dir)
            all_results.append(result)
    
    conn.close()
    
    # Generate summary report
    summary_filename = f"{report_dir}/summary.csv"
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(summary_filename, index=False)
    
    # Print tabular summary to console
    print("\nOHLCV Data Validation Summary:")
    
    # Prepare table data
    table_data = []
    headers = ["Symbol", "Timeframe", "Records", "NaN?", "Invalid OHLC", "Zero Volume?", "Big Changes", "Quality Score"]
    
    for result in all_results:
        table_data.append([
            result['symbol'],
            result['timeframe'],
            result['record_count'],
            "Yes" if result.get('has_nan', False) else "No",
            result.get('invalid_pairs', 0),
            "Yes" if result.get('has_zero_volume', False) else "No",
            result.get('large_price_changes', 0),
            f"{result.get('data_quality_score', 0)}/100"
        ])
    
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Print statistics
    total_symbols = len(all_results)
    symbols_with_issues = sum(1 for r in all_results if r.get('data_quality_score', 100) < 100)
    avg_quality_score = sum(r.get('data_quality_score', 0) for r in all_results) / max(1, total_symbols)
    
    print(f"\nTotal symbols validated: {total_symbols}")
    print(f"Symbols with data quality issues: {symbols_with_issues} ({(symbols_with_issues/total_symbols)*100:.2f}%)")
    print(f"Average data quality score: {avg_quality_score:.2f}/100")
    print(f"Summary report saved to: {summary_filename}")
    
    return all_results

def check_timeframe_coverage(symbols=None, min_coverage_days=95):
    """
    Check if we have data covering at least min_coverage_days for each timeframe.
    
    Args:
        symbols: List of specific symbols to check (check all if None)
        min_coverage_days: Minimum required days of data coverage
        
    Returns:
        dict: Coverage results for each timeframe
    """
    # Connect to the database
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    coverage_results = {}
    
    for timeframe in ENABLED_TIMEFRAMES:
        table_name = f"data_{timeframe}"
        
        # Check if table exists
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        if not cursor.fetchone():
            logging.warning(f"Table {table_name} does not exist")
            coverage_results[timeframe] = {"exists": False, "symbols_coverage": {}}
            continue
            
        # Get symbols to check
        if symbols:
            symbols_to_check = symbols
        else:
            cursor.execute(f"SELECT DISTINCT symbol FROM {table_name}")
            symbols_to_check = [row[0] for row in cursor.fetchall()]
        
        symbol_coverage = {}
        
        for symbol in symbols_to_check:
            cursor.execute(f"""
                SELECT MIN(timestamp), MAX(timestamp), COUNT(*)
                FROM {table_name}
                WHERE symbol = ?
            """, (symbol,))
            
            min_date_str, max_date_str, count = cursor.fetchone()
            
            if min_date_str and max_date_str:
                try:
                    # Parse the timestamp strings
                    min_date = datetime.strptime(min_date_str, '%Y-%m-%dT%H:%M:%S')
                    max_date = datetime.strptime(max_date_str, '%Y-%m-%dT%H:%M:%S')
                    
                    # Calculate days difference
                    days_diff = (max_date - min_date).days + 1
                    
                    symbol_coverage[symbol] = {
                        "first_date": min_date.strftime("%Y-%m-%d"),
                        "last_date": max_date.strftime("%Y-%m-%d"),
                        "days_coverage": days_diff,
                        "record_count": count,
                        "sufficient": days_diff >= min_coverage_days
                    }
                except (ValueError, TypeError) as e:
                    logging.warning(f"Error parsing dates for {symbol} ({timeframe}): {e}")
                    symbol_coverage[symbol] = {
                        "first_date": min_date_str,
                        "last_date": max_date_str,
                        "days_coverage": 0,
                        "record_count": count,
                        "sufficient": False,
                        "error": str(e)
                    }
            else:
                symbol_coverage[symbol] = {
                    "first_date": None,
                    "last_date": None,
                    "days_coverage": 0,
                    "record_count": count,
                    "sufficient": False
                }
        
        # Calculate overall statistics for this timeframe
        symbols_with_sufficient_coverage = sum(1 for data in symbol_coverage.values() if data.get("sufficient", False))
        total_symbols = len(symbol_coverage)
        coverage_percentage = (symbols_with_sufficient_coverage / total_symbols) * 100 if total_symbols > 0 else 0
        
        coverage_results[timeframe] = {
            "exists": True,
            "symbols_coverage": symbol_coverage,
            "symbols_with_sufficient_coverage": symbols_with_sufficient_coverage,
            "total_symbols": total_symbols,
            "coverage_percentage": coverage_percentage
        }
    
    conn.close()
    return coverage_results

def print_timeframe_coverage_summary(coverage_results):
    """Print a summary of timeframe coverage."""
    print("\nTimeframe Coverage Summary:")
    
    for timeframe, data in coverage_results.items():
        if not data.get("exists", False):
            print(f"  {timeframe}: Table does not exist")
            continue
            
        symbols_with_sufficient = data.get("symbols_with_sufficient_coverage", 0)
        total_symbols = data.get("total_symbols", 0)
        coverage_pct = data.get("coverage_percentage", 0)
        
        print(f"  {timeframe}: {symbols_with_sufficient}/{total_symbols} symbols have sufficient coverage ({coverage_pct:.2f}%)")

def main():
    """Main function for data validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate cryptocurrency OHLCV data")
    parser.add_argument("--timeframe", help="Specific timeframe to validate")
    parser.add_argument("--symbol", help="Specific symbol to validate")
    parser.add_argument("--top", type=int, help="Validate only top N symbols by record count")
    
    args = parser.parse_args()
    
    symbols = [args.symbol] if args.symbol else None
    
    print("\n" + "="*80)
    print("  CRYPTOCURRENCY OHLCV DATA VALIDATION")
    print("  Checking data quality and coverage")
    print("="*80 + "\n")
    
    # Check timeframe coverage
    coverage_results = check_timeframe_coverage(symbols)
    print_timeframe_coverage_summary(coverage_results)
    
    # Validate data quality
    print("\n" + "="*80)
    print("  VALIDATING DATA QUALITY")
    print("  Checking for NaN values, OHLCV integrity, and price anomalies")
    print("="*80)
    
    start_time = time.time()
    validate_all_symbols(args.timeframe, symbols, args.top)
    end_time = time.time()
    
    print(f"\nValidation completed in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
