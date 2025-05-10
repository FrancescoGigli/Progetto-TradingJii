#!/usr/bin/env python3
"""
Exchange connection module for TradingJii

Handles connections to cryptocurrency exchanges and market data retrieval.
"""

import os
import logging
import ccxt.async_support as ccxt_async
from colorama import Fore, Style
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from modules.utils.config import EXCHANGE_CONFIG

async def create_exchange():
    """
    Create and initialize the Bybit exchange connection.
    
    Returns:
        Initialized exchange object
    """
    exchange = ccxt_async.bybit(EXCHANGE_CONFIG)
    await exchange.load_markets()
    return exchange

async def fetch_markets(exchange):
    """
    Retrieve all available markets from the exchange.
    
    Args:
        exchange: The exchange object
        
    Returns:
        Dictionary of markets filtered to USDT quote currency and active swap markets
    """
    try:
        return {
            market['symbol']: market
            for market in await exchange.fetch_markets()
            if market.get('quote') == 'USDT' and market.get('active') and market.get('type') == 'swap'
        }
    except Exception as e:
        logging.error(f"Errore nel recupero dei mercati: {e}")
        return {}

async def get_top_symbols(exchange, symbols, top_n=100):
    """
    Get the top N cryptocurrencies by trading volume.
    
    Args:
        exchange: The exchange object
        symbols: List of all symbols to check
        top_n: Number of top symbols to return (default: 100)
        
    Returns:
        List of top symbols by volume
    """
    try:
        logging.info(f"Recupero dati di volume per {len(symbols)} coppie USDT...")
        volumes = {}

        with logging_redirect_tqdm():
            with tqdm(total=len(symbols), desc=f"{Fore.BLUE}Ricerca delle coppie USDT con maggior volume{Style.RESET_ALL}", 
                     bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Style.RESET_ALL)) as pbar:
                for symbol in symbols:
                    try:
                        ticker = await exchange.fetch_ticker(symbol)
                        volumes[symbol] = ticker.get('quoteVolume', 0) if ticker else 0
                    except Exception as e:
                        logging.error(f"Errore nel recupero del volume per {symbol}: {e}")
                        volumes[symbol] = 0
                    pbar.update(1)

        top_symbols = [s[0] for s in sorted(volumes.items(), key=lambda x: x[1], reverse=True)[:top_n]]
        logging.info(f"Trovate {Fore.YELLOW}{len(top_symbols)}{Style.RESET_ALL} coppie USDT con maggior volume")

        # Mostra le top criptovalute in una tabella formattata
        print("\n" + "="*80)
        print(f"{Fore.WHITE}  TOP CRIPTOVALUTE PER VOLUME  {Style.RESET_ALL}")
        print("="*80)
        
        # Intestazione tabella
        print(f"{'#':4} {'Simbolo':20} {'Volume (USDT)':>25}")
        print("-"*60)
        
        # Mostra le prime 10 per riferimento
        for i, (symbol, volume) in enumerate(sorted(volumes.items(), key=lambda x: x[1], reverse=True)[:10]):
            # Alternare colori di sfondo per facilità di lettura
            bg_color = "" if i % 2 == 0 else ""
            # Evidenzia in base alla posizione (TOP 3 in giallo, resto in bianco)
            symbol_color = Fore.YELLOW if i < 3 else Fore.WHITE
            volume_color = Fore.CYAN if i < 3 else Fore.WHITE
            
            print(f"{bg_color}{i+1:3} {symbol_color}{symbol:20} {volume_color}{volume:25,.2f}{Style.RESET_ALL}")
        
        print("="*80 + "\n")
        return top_symbols
    except Exception as e:
        logging.error(f"Errore nel recupero delle coppie con maggior volume: {e}")
        return []
