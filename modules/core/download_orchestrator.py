#!/usr/bin/env python3
"""
Download orchestrator module for TradingJii

Handles parallel and sequential download of cryptocurrency data.
"""

import asyncio
import logging
import traceback
from datetime import datetime
from colorama import Fore, Style, Back
from modules.core.exchange import create_exchange
from modules.core.data_fetcher import fetch_ohlcv_data

async def process_timeframe(timeframe, top_symbols, days, concurrency, batch_size, use_sequential=False):
    """
    Process a single timeframe by downloading data for all symbols.
    
    Args:
        timeframe: The timeframe to process
        top_symbols: List of top symbols to download data for
        days: Number of days of historical data to download
        concurrency: Maximum number of concurrent downloads
        batch_size: Batch size for downloads
        use_sequential: Whether to use sequential mode (default: False)
        
    Returns:
        Tuple of (timeframe, results)
    """
    print("\n" + "-"*80)
    print(f"{Back.CYAN}{Fore.WHITE}  ELABORAZIONE TIMEFRAME: {timeframe}  {Style.RESET_ALL}")
    print("-"*80)
    
    # Use sequential or parallel download mode
    if use_sequential:
        results = await fetch_data_sequential(top_symbols, timeframe, days, batch_size)
    else:
        results = await fetch_data_parallel(top_symbols, timeframe, days, concurrency, batch_size)
    
    logging.info(f"Completato timeframe {Fore.CYAN}{timeframe}{Style.RESET_ALL}")
    return timeframe, results

async def fetch_data_parallel(symbols, timeframe, data_limit_days, max_concurrency=5, batch_size=10):
    """
    Fetch data for multiple symbols in parallel for a specific timeframe.
    
    Args:
        symbols: List of symbols to fetch data for
        timeframe: The timeframe to fetch data for
        data_limit_days: Maximum days of historical data to fetch
        max_concurrency: Maximum number of concurrent downloads (default: 5)
        batch_size: Batch size for downloads (default: 10)
        
    Returns:
        Dictionary of results
    """
    exchange = await create_exchange()
    batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]
    
    # Results tracking dictionary
    results = {"completati": 0, "saltati": 0, "falliti": 0, "record_totali": 0}
    
    print(f"\n{Back.BLUE}{Fore.WHITE} DOWNLOAD PARALLELO: TIMEFRAME {timeframe} {Style.RESET_ALL}")
    print(f"Batch totali: {len(batches)}, Simboli: {len(symbols)}, Concorrenza: {max_concurrency}")
    print("-" * 60)

    try:
        for batch_idx, batch in enumerate(batches):
            print(f"\n{Fore.CYAN}Batch {batch_idx+1}/{len(batches)}{Style.RESET_ALL} - Timeframe {Fore.WHITE}{timeframe}{Style.RESET_ALL}")
            print(f"Simboli in questo batch: {', '.join([Fore.YELLOW + s + Style.RESET_ALL for s in batch])}")
            
            # Process symbols in parallel with concurrency limit
            tasks = []
            semaphore = asyncio.Semaphore(max_concurrency)
            
            # Use a queue to organize results
            result_queue = asyncio.Queue()
            
            async def process_symbol(sym):
                async with semaphore:
                    try:
                        result = await fetch_ohlcv_data(exchange, sym, timeframe, data_limit_days)
                        if result is None:
                            await result_queue.put((sym, "saltato", 0))
                            return
                            
                        success, count = result
                        if success:
                            await result_queue.put((sym, "completato", count))
                        else:
                            await result_queue.put((sym, "fallito", 0))
                    except Exception as e:
                        logging.error(f"Errore nell'elaborazione di {sym} ({timeframe}): {e}")
                        await result_queue.put((sym, "fallito", 0))
            
            # Create task for each symbol in batch
            for symbol in batch:
                tasks.append(process_symbol(symbol))
            
            # Current batch results collector
            current_batch_results = []
            
            # Task to display results in real-time
            async def display_results():
                batch_count = len(batch)
                completed = 0
                
                while completed < batch_count:
                    sym, status, count = await result_queue.get()
                    completed += 1
                    
                    if status == "completato":
                        results["completati"] += 1
                        results["record_totali"] += count
                        status_str = f"{Fore.GREEN}✓ Completato{Style.RESET_ALL}"
                    elif status == "saltato":
                        results["saltati"] += 1
                        status_str = f"{Fore.BLUE}↷ Saltato{Style.RESET_ALL}"
                    else:
                        results["falliti"] += 1
                        status_str = f"{Fore.RED}✗ Fallito{Style.RESET_ALL}"
                    
                    current_batch_results.append((sym, status_str, count))
                    
                    # Show progress every 5 symbols or at the end
                    if completed % 5 == 0 or completed == batch_count:
                        print(f"\nProgresso batch: {completed}/{batch_count} simboli elaborati")
                        # Show the last 5 processed results
                        recent_results = current_batch_results[-min(5, len(current_batch_results)):]
                        for s, st, c in recent_results:
                            count_str = f"{c} record" if c > 0 else ""
                            print(f"  • {Fore.YELLOW}{s:<20}{Style.RESET_ALL} {st:<25} {count_str}")
            
            # Parallel execution
            results_task = asyncio.create_task(display_results())
            await asyncio.gather(*tasks)
            await results_task
            
            # Show batch summary
            print(f"\n{Fore.CYAN}Riepilogo batch {batch_idx+1}:{Style.RESET_ALL}")
            completati = sum(1 for _, st, _ in current_batch_results if 'Completato' in st)
            saltati = sum(1 for _, st, _ in current_batch_results if 'Saltato' in st)
            falliti = sum(1 for _, st, _ in current_batch_results if 'Fallito' in st)
            print(f"  • Completati: {Fore.GREEN}{completati}{Style.RESET_ALL}")
            print(f"  • Saltati: {Fore.BLUE}{saltati}{Style.RESET_ALL}")
            print(f"  • Falliti: {Fore.RED}{falliti}{Style.RESET_ALL}")
        
        # Final summary for timeframe
        print(f"\n{Back.BLUE}{Fore.WHITE} RIEPILOGO TIMEFRAME {timeframe} {Style.RESET_ALL}")
        print(f"  • Simboli completati: {Fore.GREEN}{results['completati']}{Style.RESET_ALL}")
        print(f"  • Simboli saltati: {Fore.BLUE}{results['saltati']}{Style.RESET_ALL}")
        print(f"  • Simboli falliti: {Fore.RED}{results['falliti']}{Style.RESET_ALL}")
        print(f"  • Record totali salvati: {Fore.CYAN}{results['record_totali']:,}{Style.RESET_ALL}")
        print("-" * 60)
    except Exception as e:
        logging.error(f"Errore durante il download parallelo per {timeframe}: {e}")
        logging.error(traceback.format_exc())
    finally:
        # Make sure exchange is closed properly
        try:
            await exchange.close()
        except Exception as e:
            logging.error(f"Errore nella chiusura dell'exchange: {e}")

    return results

async def fetch_data_sequential(symbols, timeframe, data_limit_days, batch_size=10):
    """
    Fetch data for multiple symbols sequentially for a specific timeframe.
    
    Args:
        symbols: List of symbols to fetch data for
        timeframe: The timeframe to fetch data for
        data_limit_days: Maximum days of historical data to fetch
        batch_size: Batch size for downloads (default: 10)
        
    Returns:
        Dictionary of results
    """
    exchange = await create_exchange()
    batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]
    
    # Results tracking dictionary
    results = {"completati": 0, "saltati": 0, "falliti": 0, "record_totali": 0}
    
    print(f"\n{Back.BLUE}{Fore.WHITE} DOWNLOAD SEQUENZIALE: TIMEFRAME {timeframe} {Style.RESET_ALL}")
    print(f"Batch totali: {len(batches)}, Simboli: {len(symbols)}")
    print("-" * 60)

    try:
        for batch_idx, batch in enumerate(batches):
            print(f"\n{Fore.CYAN}Batch {batch_idx+1}/{len(batches)}{Style.RESET_ALL} - Timeframe {Fore.WHITE}{timeframe}{Style.RESET_ALL}")
            print(f"Simboli in questo batch: {', '.join([Fore.YELLOW + s + Style.RESET_ALL for s in batch])}")
            
            batch_results = []
            for symbol in batch:
                try:
                    result = await fetch_ohlcv_data(exchange, symbol, timeframe, data_limit_days)
                    
                    if result is None:
                        results["saltati"] += 1
                        status_str = f"{Fore.BLUE}↷ Saltato{Style.RESET_ALL}"
                        batch_results.append((symbol, status_str, 0))
                        print(f"  • {Fore.YELLOW}{symbol:<20}{Style.RESET_ALL} {status_str}")
                        continue
                    
                    success, count = result
                    if success:
                        results["completati"] += 1
                        results["record_totali"] += count
                        status_str = f"{Fore.GREEN}✓ Completato{Style.RESET_ALL}"
                        batch_results.append((symbol, status_str, count))
                        print(f"  • {Fore.YELLOW}{symbol:<20}{Style.RESET_ALL} {status_str:<25} {count} record")
                    else:
                        results["falliti"] += 1
                        status_str = f"{Fore.RED}✗ Fallito{Style.RESET_ALL}"
                        batch_results.append((symbol, status_str, 0))
                        print(f"  • {Fore.YELLOW}{symbol:<20}{Style.RESET_ALL} {status_str}")
                        
                except Exception as e:
                    logging.error(f"Errore nell'elaborazione di {symbol} ({timeframe}): {e}")
                    results["falliti"] += 1
                    status_str = f"{Fore.RED}✗ Fallito (Errore){Style.RESET_ALL}"
                    batch_results.append((symbol, status_str, 0))
                    print(f"  • {Fore.YELLOW}{symbol:<20}{Style.RESET_ALL} {status_str}")
            
            # Show batch summary
            print(f"\n{Fore.CYAN}Riepilogo batch {batch_idx+1}:{Style.RESET_ALL}")
            completati = sum(1 for _, st, _ in batch_results if 'Completato' in st)
            saltati = sum(1 for _, st, _ in batch_results if 'Saltato' in st)
            falliti = sum(1 for _, st, _ in batch_results if 'Fallito' in st)
            print(f"  • Completati: {Fore.GREEN}{completati}{Style.RESET_ALL}")
            print(f"  • Saltati: {Fore.BLUE}{saltati}{Style.RESET_ALL}")
            print(f"  • Falliti: {Fore.RED}{falliti}{Style.RESET_ALL}")
        
        # Final summary for timeframe
        print(f"\n{Back.BLUE}{Fore.WHITE} RIEPILOGO TIMEFRAME {timeframe} {Style.RESET_ALL}")
        print(f"  • Simboli completati: {Fore.GREEN}{results['completati']}{Style.RESET_ALL}")
        print(f"  • Simboli saltati: {Fore.BLUE}{results['saltati']}{Style.RESET_ALL}")
        print(f"  • Simboli falliti: {Fore.RED}{results['falliti']}{Style.RESET_ALL}")
        print(f"  • Record totali salvati: {Fore.CYAN}{results['record_totali']:,}{Style.RESET_ALL}")
        print("-" * 60)
    except Exception as e:
        logging.error(f"Errore durante il download sequenziale per {timeframe}: {e}")
        logging.error(traceback.format_exc())
    finally:
        # Make sure exchange is closed properly
        try:
            await exchange.close()
        except Exception as e:
            logging.error(f"Errore nella chiusura dell'exchange: {e}")

    return results
