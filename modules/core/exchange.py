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

# Note: fetch_markets and get_top_symbols functions have been moved to 
# modules/utils/symbol_manager.py for centralized symbol management.
# Import from there: from modules.utils.symbol_manager import get_top_symbols, get_filtered_markets
