#!/usr/bin/env python3
"""
Check Data Dates
===============

This script analyzes the date coverage of cryptocurrency data to identify missing dates or gaps.
"""

import os
import sys
import logging
import argparse
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from tabulate import tabulate

from src.utils.config import DB_FILE, ENABLED_TIMEFRAMES

def get_date_coverage(symbol, timeframe):
    """
    Get date coverage information for a symbol/timeframe pair.
    
    Args:
        symbol: Trading symbol to check
        timeframe: Timeframe to check
        
    Returns:
        dict: Coverage information including start/end dates and gaps
    """
    # Connect to database
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Get data for this symbol/timeframe
    table_name = f"data_{timeframe}"
    
    try:
        # Check if table exists
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        if not cursor.fetchone():
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "exists": False,
                "error": f"Table {table_name} does not exist"
            }
        
        # Get min and max dates and count
        cursor.execute(f"""
            SELECT MIN(timestamp), MAX(timestamp), COUNT(*)
            FROM {table_name}
            WHERE symbol = ?
        """, (symbol,))
        
        min_date_str, max_date_str, count = cursor.fetchone()
        
        if not min_date_str or not max_date_str or count == 0:
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "exists": True,
                "count": 0,
                "start_date": None,
                "end_date": None,
                "days_covered": 0,
                "gaps": []
            }
        
        # Parse dates
        min_date = datetime.strptime(min_date_str, '%Y-%m-%dT%H:%M:%S')
        max_date = datetime.strptime(max_date_str, '%Y-%m-%dT%H:%M:%S')
        
        # Calculate total days covered
        days_covered = (max_date - min_date).days + 1
        
        # Find gaps by date
        query = f"""
            SELECT timestamp
            FROM {table_name}
            WHERE symbol = ?
            ORDER BY timestamp
        """
        
        cursor.execute(query, (symbol,))
        timestamps = [datetime.strptime(row[0], '%Y-%m-%dT%H:%M:%S') for row in cursor.fetchall()]
        
        # Group by date to find daily gaps
        dates = set(ts.date() for ts in timestamps)
        
        # Generate expected range of dates
        expected_dates = set()
        current_date = min_date.date()
        while current_date <= max_date.date():
            expected_dates.add(current_date)
            current_date += timedelta(days=1)
        
        # Find gaps
        missing_dates = sorted(expected_dates - dates)
        
        # Create groups of consecutive missing dates
        gaps = []
        
        if missing_dates:
            gap_start = missing_dates[0]
            gap_end = gap_start
            
            for i in range(1, len(missing_dates)):
                if (missing_dates[i] - gap_end).days == 1:
                    # Continue the current gap
                    gap_end = missing_dates[i]
                else:
                    # End the current gap and start a new one
                    gaps.append((gap_start, gap_end))
                    gap_start = missing_dates[i]
                    gap_end = gap_start
            
            # Add the last gap
            gaps.append((gap_start, gap_end))
        
        # Calculate completeness percentage
        completeness = (len(dates) / len(expected_dates)) * 100 if expected_dates else 0
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "exists": True,
            "count": count,
            "start_date": min_date.date(),
            "end_date": max_date.date(),
            "days_covered": days_covered,
            "days_complete": len(dates),
            "days_expected": len(expected_dates),
            "completeness": completeness,
            "gaps": gaps
        }
    except Exception as e:
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "exists": True,
            "error": str(e)
        }
    finally:
        conn.close()

def get_all_symbols():
    """Get all symbols from the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Get the first timeframe table
    first_timeframe = ENABLED_TIMEFRAMES[0] if ENABLED_TIMEFRAMES else "1h"
    table_name = f"data_{first_timeframe}"
    
    try:
        cursor.execute(f"SELECT DISTINCT symbol FROM {table_name}")
        symbols = [row[0] for row in cursor.fetchall()]
        return symbols
    except Exception as e:
        logging.error(f"Error getting symbols: {e}")
        return []
    finally:
        conn.close()

def main():
    """Main entry point for checking data dates."""
    parser = argparse.ArgumentParser(description="Check cryptocurrency data date coverage")
    parser.add_argument("--symbol", help="Specific symbol to check")
    parser.add_argument("--timeframe", help="Specific timeframe to check")
    parser.add_argument("--min-completeness", type=float, default=95.0, 
                        help="Minimum completeness percentage to consider acceptable (default: 95.0)")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    print("\n" + "="*80)
    print("  CRYPTOCURRENCY DATA DATE COVERAGE CHECK")
    print("  Analyzing data completeness and identifying gaps")
    print("="*80 + "\n")
    
    # Check if database exists
    if not os.path.exists(DB_FILE):
        logging.error(f"Database file not found: {DB_FILE}")
        return
    
    # Get the symbols to check
    symbols = [args.symbol] if args.symbol else get_all_symbols()
    
    if not symbols:
        logging.error("No symbols found in database")
        return
    
    # Get the timeframes to check
    timeframes = [args.timeframe] if args.timeframe else ENABLED_TIMEFRAMES
    
    # Check date coverage for each symbol and timeframe
    results = []
    
    for symbol in symbols:
        for timeframe in timeframes:
            print(f"Checking {symbol} ({timeframe})...")
            coverage = get_date_coverage(symbol, timeframe)
            results.append(coverage)
    
    # Print summary table
    table_data = []
    headers = ["Symbol", "Timeframe", "Start Date", "End Date", "Days", "Completeness", "Gaps"]
    
    for result in results:
        if not result.get("exists", False) or "error" in result:
            error_msg = result.get("error", "No data")
            table_data.append([
                result["symbol"],
                result["timeframe"],
                "-",
                "-",
                "0",
                "0%",
                error_msg
            ])
            continue
        
        # Format gaps for display
        gaps = result.get("gaps", [])
        if gaps:
            if len(gaps) <= 2:
                # Show all gaps
                gaps_str = ", ".join([f"{start} to {end}" for start, end in gaps])
            else:
                # Show count and first/last gaps
                first_gap = f"{gaps[0][0]} to {gaps[0][1]}"
                last_gap = f"{gaps[-1][0]} to {gaps[-1][1]}"
                gaps_str = f"{len(gaps)} gaps, first: {first_gap}, last: {last_gap}"
        else:
            gaps_str = "None"
        
        # Add to table data
        table_data.append([
            result["symbol"],
            result["timeframe"],
            result.get("start_date", "-"),
            result.get("end_date", "-"),
            result.get("days_covered", 0),
            f"{result.get('completeness', 0):.1f}%",
            gaps_str
        ])
    
    print("\nData Coverage Summary:")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Print symbols with poor coverage
    poor_coverage = [r for r in results if r.get("exists", False) and 
                    not "error" in r and 
                    r.get("completeness", 0) < args.min_completeness]
    
    if poor_coverage:
        print(f"\nSymbols with poor coverage (<{args.min_completeness}% complete):")
        for result in poor_coverage:
            print(f"  • {result['symbol']} ({result['timeframe']}): {result.get('completeness', 0):.1f}% complete")
    
    # Overall statistics
    valid_results = [r for r in results if r.get("exists", False) and not "error" in r]
    avg_completeness = sum(r.get("completeness", 0) for r in valid_results) / len(valid_results) if valid_results else 0
    
    print(f"\nOverall statistics:")
    print(f"  • Total symbol/timeframe pairs checked: {len(results)}")
    print(f"  • Average completeness: {avg_completeness:.1f}%")
    print(f"  • Pairs with gaps: {len([r for r in valid_results if r.get('gaps', [])])}")
    print(f"  • Pairs with poor coverage: {len(poor_coverage)}")

if __name__ == "__main__":
    main()
