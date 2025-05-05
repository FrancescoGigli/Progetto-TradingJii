# fetcher.py

import asyncio
import logging
import pandas as pd
from datetime import datetime, timedelta
from src.utils.config import TIMEFRAME_DEFAULT, DATA_LIMIT_DAYS, TOP_ANALYSIS_CRYPTO, API_CALL_DELAY, exchange_config
from termcolor import colored
import re
from src.data.db_manager import save_data, check_data_freshness, get_data_last_timestamp
from src.data.data_utils import add_technical_indicators
from tqdm import tqdm
import ccxt.async_support as ccxt_async

async def create_exchange():
    """Create a new exchange instance."""
    exchange = ccxt_async.bybit(exchange_config)
    exchange.enableRateLimit = True
    
    # Initialize exchange
    try:
        await exchange.load_markets(reload=True)
    except Exception as e:
        logging.error(f"Error initializing exchange: {e}")
    
    return exchange

async def fetch_markets(exchange):
    """Fetch all available markets from the exchange."""
    try:
        markets = await exchange.load_markets(reload=True)
        return markets
    except Exception as e:
        logging.error(f"Error fetching markets: {e}")
        return {}

async def fetch_ticker_volume(exchange, symbol):
    """Fetch volume data for a specific symbol."""
    try:
        ticker = await exchange.fetch_ticker(symbol)
        volume = ticker.get('quoteVolume')
        
        # Some exchanges return None or 0 for volume, try using 'baseVolume' in those cases
        if not volume:
            volume = ticker.get('baseVolume')
            
        # If still no volume data, use a small non-zero value to avoid sorting issues
        if not volume:
            logging.warning(f"No volume data for {symbol}, using default value")
            volume = 0.0001
            
        return symbol, volume
    except Exception as e:
        logging.error(f"Error fetching ticker volume for {symbol}: {e}")
        # Return minimal volume instead of None to keep the symbol in the list
        return symbol, 0.0001

async def get_top_symbols(exchange, symbols, top_n=TOP_ANALYSIS_CRYPTO):
    """Get top symbols by trading volume."""
    if not symbols:
        logging.warning("No symbols provided to get_top_symbols")
        return []
        
    logging.info(f"Fetching volume data for {len(symbols)} USDT pairs...")
    
    # Process symbols in smaller batches to avoid rate limits
    batch_size = 50
    all_results = []
    
    # Create a progress bar for the volume data fetching
    with tqdm(total=len(symbols), desc="Finding top USDT pairs by volume", ncols=100, colour="green") as pbar:
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            
            tasks = [fetch_ticker_volume(exchange, symbol) for symbol in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions
            valid_results = []
            for result in batch_results:
                if isinstance(result, Exception):
                    logging.error(f"Exception in volume fetch: {result}")
                elif isinstance(result, tuple) and len(result) == 2:
                    valid_results.append(result)
            
            all_results.extend(valid_results)
            pbar.update(len(batch))
            
            # Add a small delay between batches to avoid rate limits
            await asyncio.sleep(API_CALL_DELAY)
    
    # Sort by volume
    symbol_volumes = [(symbol, volume) for symbol, volume in all_results]
    symbol_volumes.sort(key=lambda x: x[1], reverse=True)
    
    # Get top N symbols
    top_symbols = [x[0] for x in symbol_volumes[:top_n]]
    
    if not top_symbols:
        logging.warning("No symbols with volume data found")
    else:
        logging.info(f"Found {len(top_symbols)} top USDT pairs by volume")
        # Display top symbols with their volumes
        for i, (symbol, volume) in enumerate(symbol_volumes[:min(5, len(symbol_volumes))], 1):
            logging.info(colored(f"Top {i}: {symbol} - Volume: {volume:,.2f} USDT", "cyan"))
        
    return top_symbols

async def get_data_async(exchange, symbol, timeframe=TIMEFRAME_DEFAULT, limit=1000):
    """Fetch OHLCV data for a symbol and timeframe with a progress bar showing days loaded."""
    # Calculate the time range for 100 days
    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=DATA_LIMIT_DAYS)
    
    since = int(start_dt.timestamp() * 1000)
    current_time = int(end_dt.timestamp() * 1000)
    
    logging.info(colored(f"Fetching data from {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}", "cyan"))
    
    ohlcv_all = []
    fetched_days = set()  # Track which days we've fetched data for
    
    # Create a progress bar to show days loaded
    total_days = DATA_LIMIT_DAYS
    with tqdm(total=total_days, desc=f"Loading {symbol} ({timeframe})", ncols=100, colour="cyan") as pbar:
        retry_count = 0
        max_retries = 3
        current_since = since
        
        while current_since < current_time:
            try:
                ohlcv = await exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, since=current_since)
                
                if not ohlcv:
                    # No more data to fetch
                    break
                
                # Add to our collection and update progress
                for candle in ohlcv:
                    timestamp = candle[0]
                    candle_dt = datetime.fromtimestamp(timestamp / 1000)
                    day_key = candle_dt.strftime('%Y-%m-%d')
                    
                    if day_key not in fetched_days:
                        fetched_days.add(day_key)
                        days_fetched = min(len(fetched_days), total_days)
                        pbar.update(days_fetched - pbar.n)
                
                ohlcv_all.extend(ohlcv)
                
                # Update since to the last timestamp + 1
                if ohlcv:
                    last_timestamp = ohlcv[-1][0]
                    
                    # Break if we've reached or passed current time
                    if last_timestamp >= current_time:
                        break
                        
                    current_since = last_timestamp + 1
                    if current_since <= since:
                        break
                else:
                    break
                    
                # Reset retry counter on success
                retry_count = 0
                
                # Add a small delay between API calls to avoid rate limits
                await asyncio.sleep(API_CALL_DELAY)
                
            except Exception as e:
                retry_count += 1
                if retry_count > max_retries:
                    logging.error(f"Error fetching OHLCV for {symbol} after {max_retries} retries: {e}")
                    break
                logging.warning(f"Retry {retry_count}/{max_retries} for {symbol}: {e}")
                await asyncio.sleep(2)  # Wait before retry
        
        # Ensure the progress bar is completed
        pbar.update(total_days - pbar.n)

    if ohlcv_all:
        df = pd.DataFrame(ohlcv_all, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Get actual time range in the data
        if not df.empty:
            actual_start = df.index.min().strftime('%Y-%m-%d')
            actual_end = df.index.max().strftime('%Y-%m-%d')
            days_covered = (df.index.max() - df.index.min()).days + 1
            
            logging.info(colored(
                f"{symbol}: {len(df)} candles from {actual_start} to {actual_end} ({days_covered} days)",
                "green"))
        return df
    else:
        logging.info(f"No data available for {symbol} in the last {DATA_LIMIT_DAYS} days.")
        return None

async def fetch_and_save_data(exchange, symbol, timeframe=TIMEFRAME_DEFAULT, limit=1000, max_age_hours=24):
    """
    Fetch OHLCV data for a symbol, add technical indicators, and save to database.
    Skips fetching if recent data already exists in the database.
    """
    # Check if we already have fresh data for this symbol and timeframe
    has_fresh_data, last_timestamp = check_data_freshness(symbol, timeframe, max_age_days=max_age_hours/24)
    
    if has_fresh_data:
        # Data is fresh, no need to fetch again
        logging.info(colored(
            f"Skipping {symbol} ({timeframe}): Recent data already exists (last record: {last_timestamp.strftime('%Y-%m-%d %H:%M')})",
            "yellow"))
        return None
    
    # If we have some data but it's not fresh, log what we're doing
    if last_timestamp:
        last_timestamp_str = last_timestamp.strftime('%Y-%m-%d %H:%M')
        logging.info(colored(
            f"Updating {symbol} ({timeframe}): Last data point from {last_timestamp_str}",
            "blue"))
    else:
        logging.info(colored(
            f"Fetching {symbol} ({timeframe}): No existing data",
            "magenta"))
    
    # Fetch new data
    df = await get_data_async(exchange, symbol, timeframe, limit)
    if df is not None and not df.empty:
        try:
            # Add technical indicators before saving
            df_indicators = add_technical_indicators(df.copy())
            # Save the data with indicators to the database
            save_data(symbol, df_indicators, timeframe)
            return df
        except Exception as e:
            logging.error(f"Error processing data for {symbol}: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return None
    return None
