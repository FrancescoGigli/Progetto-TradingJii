#!/usr/bin/env python3
"""
OHLCV Data Checker
=================

This script validates the OHLCV data for missing values, gaps, and anomalies.
"""

import os
import sys
import logging
import argparse
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from tabulate import tabulate

from src.utils.config import DB_FILE, ENABLED_TIMEFRAMES
from src.utils.data_validation import validate_all_symbols, check_timeframe_coverage

def main():
    """Main entry point for OHLCV data checking."""
    parser = argparse.ArgumentParser(description="Check OHLCV data quality")
    parser.add_argument("--symbol", help="Specific symbol to check")
    parser.add_argument("--timeframe", help="Specific timeframe to check")
    parser.add_argument("--top", type=int, help="Check only top N symbols by data count")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    print("\n" + "="*80)
    print("  OHLCV DATA QUALITY CHECK")
    print("  Checking cryptocurrency data for quality issues")
    print("="*80 + "\n")
    
    # Get symbols for validation
    symbols = None
    if args.symbol:
        symbols = [args.symbol]
    
    # Check if database exists
    if not os.path.exists(DB_FILE):
        logging.error(f"Database file not found: {DB_FILE}")
        return
    
    # Check timeframe coverage first
    print("\n" + "="*80)
    print("  CHECKING DATA COVERAGE")
    print("="*80)
    
    coverage_results = check_timeframe_coverage(symbols)
    
    for timeframe, data in coverage_results.items():
        if not data.get("exists", False):
            continue
            
        total_symbols = data.get("total_symbols", 0)
        if total_symbols == 0:
            continue
            
        symbols_with_issues = sum(1 for s, info in data.get("symbols_coverage", {}).items() 
                                if not info.get("sufficient", False))
        
        percent_with_issues = (symbols_with_issues / total_symbols) * 100 if total_symbols > 0 else 0
        
        print(f"\n{timeframe} Timeframe:")
        print(f"  • Total symbols: {total_symbols}")
        print(f"  • Symbols with insufficient coverage: {symbols_with_issues} ({percent_with_issues:.1f}%)")
    
    # Validate data quality
    print("\n" + "="*80)
    print("  VALIDATING DATA QUALITY")
    print("="*80)
    
    validate_all_symbols(args.timeframe, symbols, args.top)
    
    print("\nData validation complete.")

if __name__ == "__main__":
    main()
