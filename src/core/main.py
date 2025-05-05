#!/usr/bin/env python3
import sys
import os
import asyncio
import logging
import re
import sqlite3
import argparse
import json
from datetime import datetime
import ccxt.async_support as ccxt_async
from termcolor import colored
import subprocess

# Import config before other modules
from src.utils.config import (
    API_KEY,
    API_SECRET,
    exchange_config,
    ENABLED_TIMEFRAMES,
    EXCLUDED_SYMBOLS,
    TOP_ANALYSIS_CRYPTO,
    DB_FILE,
    DATA_LIMIT_DAYS,
    RESET_DB_ON_STARTUP,
    BATCH_SIZE
)
from src.utils.logging_config import *
from src.core.fetcher import fetch_markets, get_top_symbols, fetch_and_save_data, create_exchange
from src.data.db_manager import init_data_tables, get_symbol_data_info

from src.scripts.fix_progress_data import apply_progress_data_patch
# For Windows compatibility
if sys.platform.startswith('win'):
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

async def fetch_data_sequential(symbols, timeframe):
    """Fetch data for multiple symbols sequentially for a specific timeframe."""
    # Create one exchange instance
    exchange = await create_exchange()
    
    # Sort symbols alphabetically to ensure consistent processing order
    sorted_symbols = sorted(symbols)
    
    # Split symbols into batches to avoid overloading the API
    batches = [sorted_symbols[i:i + BATCH_SIZE] for i in range(0, len(sorted_symbols), BATCH_SIZE)]
    
    # Process each batch sequentially
    for batch_idx, batch in enumerate(batches):
        logging.info(f"Processing batch {batch_idx+1}/{len(batches)} for timeframe {timeframe} ({len(batch)} symbols)...")
        
        # Process each symbol in the batch sequentially
        for symbol in batch:
            try:
                # Check if data is fresh before fetching
                result = await fetch_and_save_data(exchange, symbol, timeframe)
                
                if result is None:
                    # Data was skipped because it was fresh
                    logging.info(f"Skipped {symbol} ({timeframe}) - data is fresh")
                else:
                    # Data was successfully fetched and saved
                    logging.info(f"Completed {symbol} ({timeframe})")
                    
            except Exception as e:
                logging.error(f"Error processing {symbol} ({timeframe}): {e}")
    
    # Close the exchange instance
    await exchange.close()

async def fetch_data():
    # Initialize progress_data from the fix module
    progress_data = apply_progress_data_patch()
    
    # Initialize progress tracking dictionary
    progress_file = "fetch_progress.json"
    
    # Load existing progress data if available, or create new
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logging.error(f"Error loading progress data: {e}")
            progress_data = {'symbols': {}, 'last_update': datetime.now().isoformat()}
    else:
        progress_data = {'symbols': {}, 'last_update': datetime.now().isoformat()}
    
    # Display header with clear information about what the program does
    print("\n" + "="*80)
    print(colored(f"  CRYPTOCURRENCY OHLCV DATA FETCHER", "yellow"))
    print(colored(f"  Fetching {TOP_ANALYSIS_CRYPTO} top cryptocurrencies with {DATA_LIMIT_DAYS} days of data", "yellow"))
    print(colored(f"  Timeframes: {', '.join(ENABLED_TIMEFRAMES)}", "yellow"))
    print(colored(f"  Sequential processing for all symbols", "yellow"))
    print(colored(f"  Processing in order by timeframe", "yellow"))
    print("="*80 + "\n")
    
    # Display API key information (partially masked for security)
    if API_KEY and API_SECRET:
        masked_key = API_KEY[:4] + '*' * (len(API_KEY) - 8) + API_KEY[-4:] if len(API_KEY) > 8 else '****'
        masked_secret = API_SECRET[:4] + '*' * (len(API_SECRET) - 8) + API_SECRET[-4:] if len(API_SECRET) > 8 else '****'
        logging.info(f"Using API Key: {masked_key}")
        logging.info(f"Using API Secret: {masked_secret}")
    else:
        logging.error("No API keys found. Check your .env file or environment variables.")
        
    # Check if .env file exists and was loaded
    env_path = os.path.join(os.getcwd(), '.env')
    if os.path.exists(env_path):
        logging.info(f".env file found at {env_path}")
    else:
        logging.warning(f".env file not found at {env_path}")

    # Reset database if configured
    if RESET_DB_ON_STARTUP and os.path.exists(DB_FILE):
        os.remove(DB_FILE)
        logging.info(f"Database file {DB_FILE} removed as per config; starting fresh.")
    
    # Initialize database tables
    init_data_tables()
    logging.info(f"Database initialized with tables for timeframes: {', '.join(ENABLED_TIMEFRAMES)}")
    
    try:
        # Initialize exchange for initial operations
        logging.info("Initializing Bybit exchange connection...")
        async_exchange = await create_exchange()
        
        try:
            # Test API connection by fetching account balance
            logging.info("Testing API connection...")
            balance = await async_exchange.fetch_balance()
            if balance:
                logging.info("API connection successful - authenticated mode")
                # Show USDT balance specifically
                usdt_balance = balance.get('free', {}).get('USDT', 0)
                if usdt_balance:
                    logging.info(f"USDT Balance: {usdt_balance:,.2f}")
            else:
                logging.warning("API connection returned empty response")
        except Exception as e:
            logging.error(f"API connection failed: {e}")
            logging.info("Attempting to continue in public API mode (some functionality may be limited)")
            
            # Try to test if public API works
            try:
                ticker = await async_exchange.fetch_ticker('BTC/USDT:USDT')
                if ticker:
                    logging.info("Public API connection successful")
                else:
                    logging.error("Public API returned empty response")
            except Exception as public_error:
                logging.error(f"Public API also failed: {public_error}")
                logging.error("Please check your internet connection and Bybit API status")
        
        # Fetch market data
        markets = await fetch_markets(async_exchange)
        if not markets:
            logging.error("No markets found. Check your internet connection and API credentials.")
            return []
        
        print("\n" + "-"*80)
        print(colored("  FINDING TOP CRYPTOCURRENCIES BY VOLUME", "cyan"))
        print("-"*80 + "\n")
            
        # Filter for USDT markets
        all_symbols = [m['symbol'] for m in markets.values() if m.get('quote') == 'USDT'
                       and m.get('active') and m.get('type') == 'swap']
        logging.info(f"Found {len(all_symbols)} USDT pairs on Bybit")
        
        # Apply exclusion filter
        if EXCLUDED_SYMBOLS:
            all_symbols_analysis = [s for s in all_symbols if not re.search('|'.join(EXCLUDED_SYMBOLS), s)]
            logging.info(f"After exclusions: {len(all_symbols_analysis)} USDT pairs")
        else:
            all_symbols_analysis = all_symbols
        
        if not all_symbols_analysis:
            logging.error("No symbols found after filtering. Check your exchange connection and filters.")
            return []
        
        # Get top cryptocurrencies by volume
        logging.info(f"Getting top {TOP_ANALYSIS_CRYPTO} cryptocurrencies by USDT volume...")
        top_symbols = await get_top_symbols(async_exchange, all_symbols_analysis, top_n=TOP_ANALYSIS_CRYPTO)
        
        if not top_symbols:
            logging.error("Failed to get top symbols. Using all available symbols instead.")
            top_symbols = all_symbols_analysis[:TOP_ANALYSIS_CRYPTO]
        
        # Close the initial exchange instance
        await async_exchange.close()
               
        # Display the list of timeframes we'll be processing
        print("\n" + "-"*80)
        print(colored(f"  FETCHING DATA FOR TOP {len(top_symbols)} CRYPTOCURRENCIES", "magenta"))
        print(colored(f"  Timeframes: {', '.join(ENABLED_TIMEFRAMES)}", "magenta"))
        print(colored(f"  Period: Last {DATA_LIMIT_DAYS} days", "magenta"))
        print(colored(f"  Processing sequentially in order: {', '.join(ENABLED_TIMEFRAMES)}", "magenta"))
        print("-"*80 + "\n")
        
        # Process one timeframe completely before moving to the next
        for i, timeframe in enumerate(ENABLED_TIMEFRAMES):
            print("\n" + "-"*60)
            print(colored(f"  TIMEFRAME {i+1}/{len(ENABLED_TIMEFRAMES)}: {timeframe}", "yellow"))
            print("-"*60)
            
            logging.info(f"Starting timeframe {timeframe} ({i+1}/{len(ENABLED_TIMEFRAMES)})")
            
            # Process all symbols for this timeframe sequentially
            await fetch_data_sequential(top_symbols, timeframe)
            
            logging.info(f"Completed timeframe {timeframe} ({i+1}/{len(ENABLED_TIMEFRAMES)})")
        
        # Final summary
        print("\n" + "="*80)
        print(colored("  DATA COLLECTION COMPLETE", "green"))
        print(colored(f"  • Downloaded data for {len(top_symbols)} cryptocurrencies", "green"))
        print(colored(f"  • Timeframes: {', '.join(ENABLED_TIMEFRAMES)}", "green"))
        print(colored(f"  • Database: {os.path.abspath(DB_FILE)}", "green"))
        print("="*80 + "\n")
        
        # Check if we should run validation (parser adds --no-validate flag)
        run_validation = True
        
        # Get the command line args to check for --no-validate
        parser_args = sys.argv
        if '--no-validate' in parser_args:
            run_validation = False
            logging.info("Skipping post-fetch validation (--no-validate flag used)")
        
        # Run OHLCV validation after fetching data if not disabled
        if run_validation:
            print("\n" + "="*80)
            print(colored("  RUNNING OHLCV DATA VALIDATION AFTER FETCH", "yellow"))
            print(colored("  Checking for NaN and empty values in cryptocurrency data", "yellow"))
            print("="*80 + "\n")
            
            try:
                # Create an argparse namespace with default values
                validation_args = argparse.Namespace()
                validation_args.symbol = None
                validation_args.timeframe = None
                validation_args.top = None
                
                # Run validation routine
                logging.info("Starting post-fetch OHLCV data validation...")
                run_ohlcv_validation(validation_args)
                logging.info("Post-fetch validation completed")
            except Exception as e:
                logging.error(f"Error during post-fetch validation: {e}")
            
        return top_symbols
        
    except Exception as e:
        logging.error(f"Error: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return []
    finally:
        logging.info("Data fetching completed.")

def get_symbols_from_db(limit=None):
    """Get a list of symbols from the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    try:
        # Get first timeframe table for checking
        first_timeframe = ENABLED_TIMEFRAMES[0] if ENABLED_TIMEFRAMES else "1h"
        table = f"data_{first_timeframe}"
        
        if limit:
            cursor.execute(f"SELECT DISTINCT symbol FROM {table} LIMIT ?", (limit,))
        else:
            cursor.execute(f"SELECT DISTINCT symbol FROM {table}")
            
        symbols = [row[0] for row in cursor.fetchall()]
        conn.close()
        return symbols
    except Exception as e:
        logging.error(f"Error getting symbols from database: {e}")
        conn.close()
        return []

# Helper function to get the smallest timeframe interval in seconds
def get_smallest_timeframe_interval():
    """Get the smallest timeframe interval in seconds."""
    intervals = {
        '1m': 60,
        '5m': 300,
        '15m': 900,
        '30m': 1800,
        '1h': 3600,
        '4h': 14400,
        '1d': 86400
    }
    
    smallest_interval = 3600  # Default to 1h
    for tf in ENABLED_TIMEFRAMES:
        if tf in intervals and intervals[tf] < smallest_interval:
            smallest_interval = intervals[tf]
    
    # Return interval in seconds
    return smallest_interval

async def continuous_fetch():
    """Continuously fetch data with intervals based on the smallest timeframe."""
    smallest_interval = get_smallest_timeframe_interval()
    
    logging.info(f"Starting continuous data fetching with {smallest_interval} second intervals")
    
    try:
        while True:
            # Fetch data once
            await fetch_data()
            
            # Wait for the smallest timeframe interval before fetching again
            logging.info(f"Data fetch complete. Sleeping for {smallest_interval} seconds before next update...")
            await asyncio.sleep(smallest_interval)
    except KeyboardInterrupt:
        logging.info("Continuous fetching stopped by user")
    except Exception as e:
        logging.error(f"Error in continuous fetching: {e}")
        import traceback
        logging.error(traceback.format_exc())

def run_ohlcv_validation(args):
    """Run the OHLCV data validation script with provided arguments."""
    print("\n" + "="*80)
    print(colored("  RUNNING OHLCV DATA VALIDATION", "yellow"))
    print(colored("  Checking for NaN, empty and invalid values in cryptocurrency data", "yellow"))
    print("="*80 + "\n")
    
    # Build command to run validation script
    cmd = ["python", "src/scripts/ohlcv_data_check.py"]
    
    # Add arguments if provided
    if args.symbol:
        cmd.extend(["--symbol", args.symbol])
        logging.info(f"Validating specific symbol: {args.symbol}")
    
    if args.timeframe:
        cmd.extend(["--timeframe", args.timeframe])
        logging.info(f"Validating specific timeframe: {args.timeframe}")
    
    if args.top:
        cmd.extend(["--top", str(args.top)])
        logging.info(f"Validating top {args.top} symbols by volume")
    
    try:
        # Run the validation script as a subprocess
        logging.info("Starting OHLCV data validation...")
        result = subprocess.run(cmd, check=True)
        
        # Check the return code
        if result.returncode == 0:
            logging.info("OHLCV data validation completed successfully.")
        else:
            logging.error(f"OHLCV data validation failed with return code {result.returncode}")
    
    except subprocess.CalledProcessError as e:
        logging.error(f"Error executing validation script: {e}")
    except FileNotFoundError:
        logging.error("Validation script not found. Make sure ohlcv_data_check.py exists in the current directory.")
    except Exception as e:
        logging.error(f"Unexpected error during validation: {e}")
        import traceback
        logging.error(traceback.format_exc())

async def main():
    """Main entry point with continuous data fetching as default."""
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Cryptocurrency OHLCV Data Fetcher and Validator")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Fetch data command
    fetch_parser = subparsers.add_parser("fetch", help="Fetch cryptocurrency data")
    fetch_parser.add_argument("--no-validate", action="store_true", help="Skip validation after fetching data")
    fetch_parser.add_argument("--once", action="store_true", help="Fetch data once instead of continuously")
    
    # Validate OHLCV data command
    validate_parser = subparsers.add_parser("validate", help="Validate OHLCV data for missing or NaN values")
    validate_parser.add_argument("--symbol", help="Specific symbol to validate (default: validate all)")
    validate_parser.add_argument("--timeframe", help="Specific timeframe to validate (default: validate all)")
    validate_parser.add_argument("--top", type=int, help="Validate only top N symbols by volume")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Default to fetch with continuous mode if no command specified
    if not args.command:
        args.command = "fetch"
        args.no_validate = False
        args.once = False
    
    # Handle each command
    if args.command == "fetch":
        # Determine if we should fetch once or continuously
        fetch_once = getattr(args, 'once', False)
        
        if fetch_once:
            # Just fetch once
            symbols = await fetch_data()
        else:
            # Continuous fetching
            await continuous_fetch()
    
    elif args.command == "validate":
        # Run OHLCV data validation
        run_ohlcv_validation(args)
    
    logging.info("Program terminated.")

if __name__ == "__main__":
    logging.info("Starting cryptocurrency data handler...")
    asyncio.run(main())
