#!/usr/bin/env python3
"""
Symbol Manager module for TradingJii

Centralizes all symbol selection and filtering logic.
"""

import logging
from colorama import Fore, Style
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from modules.utils.config import DEFAULT_TOP_SYMBOLS, EXCLUDED_SYMBOLS


async def get_top_symbols(exchange, limit=DEFAULT_TOP_SYMBOLS):
    """
    Restituisce i primi N simboli filtrati da quelli esclusi, con volume discendente.
    
    Args:
        exchange: L'oggetto exchange inizializzato
        limit: Numero di simboli da restituire (default: DEFAULT_TOP_SYMBOLS)
        
    Returns:
        List: Lista dei simboli ordinati per volume discendente
    """
    try:
        # Carica i mercati dall'exchange
        markets = await exchange.load_markets()
        
        # Filtra per coppie USDT attive e di tipo swap, escludendo quelli nella blacklist
        usdt_pairs = {
            symbol: data
            for symbol, data in markets.items()
            if (data['quote'] == 'USDT' and 
                symbol not in EXCLUDED_SYMBOLS and 
                data['active'] and 
                data.get('type') == 'swap')
        }
        
        if not usdt_pairs:
            logging.error("Nessun mercato USDT trovato dopo il filtraggio")
            return []
        
        logging.info(f"Recupero dati di volume per {len(usdt_pairs)} coppie USDT...")
        volumes = {}

        with logging_redirect_tqdm():
            with tqdm(total=len(usdt_pairs), desc=f"{Fore.BLUE}Ricerca delle coppie USDT con maggior volume{Style.RESET_ALL}", 
                     bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Style.RESET_ALL)) as pbar:
                for symbol in usdt_pairs.keys():
                    try:
                        ticker = await exchange.fetch_ticker(symbol)
                        volumes[symbol] = ticker.get('quoteVolume', 0) if ticker else 0
                    except Exception as e:
                        logging.error(f"Errore nel recupero del volume per {symbol}: {e}")
                        volumes[symbol] = 0
                    pbar.update(1)

        # Ordina per volume discendente e prendi i primi N
        sorted_pairs = sorted(volumes.items(), key=lambda x: x[1], reverse=True)
        top_symbols = [symbol for symbol, _ in sorted_pairs[:limit]]
        
        logging.info(f"Trovate {Fore.YELLOW}{len(top_symbols)}{Style.RESET_ALL} coppie USDT con maggior volume")

        # Mostra le top criptovalute in una tabella formattata
        print("\n" + "="*80)
        print(f"{Fore.WHITE}  TOP CRIPTOVALUTE PER VOLUME  {Style.RESET_ALL}")
        print("="*80)
        
        # Intestazione tabella
        print(f"{'#':4} {'Simbolo':20} {'Volume (USDT)':>25}")
        print("-"*60)
        
        # Mostra le prime 10 per riferimento
        for i, (symbol, volume) in enumerate(sorted_pairs[:min(10, len(sorted_pairs))]):
            # Evidenzia in base alla posizione (TOP 3 in giallo, resto in bianco)
            symbol_color = Fore.YELLOW if i < 3 else Fore.WHITE
            volume_color = Fore.CYAN if i < 3 else Fore.WHITE
            
            print(f"{i+1:3} {symbol_color}{symbol:20} {volume_color}{volume:25,.2f}{Style.RESET_ALL}")
        
        print("="*80 + "\n")
        
        return top_symbols
        
    except Exception as e:
        logging.error(f"Errore nel recupero delle coppie con maggior volume: {e}")
        return []


async def get_filtered_markets(exchange):
    """
    Restituisce tutti i mercati filtrati per USDT quote currency e attivi.
    
    Args:
        exchange: L'oggetto exchange inizializzato
        
    Returns:
        Dict: Dictionary dei mercati filtrati
    """
    try:
        markets = await exchange.load_markets()
        return {
            symbol: market
            for symbol, market in markets.items()
            if (market.get('quote') == 'USDT' and 
                market.get('active') and 
                market.get('type') == 'swap' and
                symbol not in EXCLUDED_SYMBOLS)
        }
    except Exception as e:
        logging.error(f"Errore nel recupero dei mercati filtrati: {e}")
        return {}
