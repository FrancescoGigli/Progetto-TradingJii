#!/usr/bin/env python3
import re
import time
from datetime import datetime
import logging
from termcolor import colored
import os
import asyncio
import uuid
import json

from config import (
    MARGIN_USDT, LEVERAGE, EXCLUDED_SYMBOLS,
    TOP_ANALYSIS_CRYPTO
)
from fetcher import get_top_symbols, get_data_async

def is_symbol_excluded(symbol):
    normalized = re.sub(r'[^A-Za-z0-9]', '', symbol).upper()
    return any(exc.upper() in normalized for exc in EXCLUDED_SYMBOLS)

async def get_real_balance(exchange):
    try:
        # Usa l'approccio diretto di CCXT per ottenere il saldo
        logging.info(colored("📊 Recupero saldo tramite CCXT", "cyan"))
        
        # Ottieni il saldo completo
        balance = await exchange.fetch_balance()
        
        # Inizializza il valore totale in USD
        total_value = 0
        
        # Controlla se esiste la struttura info e unified (per account unificato)
        if 'info' in balance and 'totalEquity' in balance['info']:
            # Account unificato - possiamo ottenere direttamente l'equity totale
            total_value = float(balance['info']['totalEquity'])
            logging.info(colored(f"💰 Valore totale account: {total_value} USD", "green"))
            return total_value
        
        # Calcola il valore totale sommando tutti i saldi in USD
        # Cerca prima nella sezione 'total' che contiene tutti i saldi
        if 'total' in balance:
            for currency, amount in balance['total'].items():
                if amount > 0:
                    # Stampa le informazioni per debug
                    logging.info(colored(f"💴 {currency}: {amount}", "green"))
                    
                    # Se è USDT, aggiungi direttamente
                    if currency == 'USDT':
                        total_value += amount
                    else:
                        # Per altre valute, prova a convertire in USD/USDT
                        try:
                            if currency != 'USD' and currency != 'USDT':
                                # Trova il ticker per la conversione
                                ticker_symbol = f"{currency}/USDT"
                                ticker = await exchange.fetch_ticker(ticker_symbol)
                                if ticker and 'last' in ticker:
                                    currency_value = amount * ticker['last']
                                    logging.info(colored(f"  → {amount} {currency} = {currency_value:.2f} USDT", "green"))
                                    total_value += currency_value
                            else:
                                # USD e USDT sono considerati equivalenti
                                total_value += amount
                        except Exception as conv_err:
                            logging.warning(colored(f"⚠️ Impossibile convertire {currency} in USDT: {conv_err}", "yellow"))
        
        # Se abbiamo trovato almeno qualche saldo
        if total_value > 0:
            logging.info(colored(f"💰 Valore totale stimato: {total_value:.2f} USDT", "green"))
            return total_value
        
        # Fallback: cerca il saldo USDT se non abbiamo trovato altro
        usdt_balance = balance.get('USDT', {}).get('free', 0)
        if isinstance(usdt_balance, dict) and 'free' in usdt_balance:
            usdt_balance = usdt_balance['free']
        
        # Se è ancora zero, prova a cercarlo nella struttura total
        if usdt_balance == 0 and 'total' in balance and 'USDT' in balance['total']:
            usdt_balance = balance['total']['USDT']
        
        if usdt_balance > 0:
            logging.info(colored(f"💰 Saldo USDT: {usdt_balance}", "green"))
        else:
            logging.warning(colored("⚠️ Nessun saldo trovato.", "yellow"))
        
        return usdt_balance or 0
    except Exception as e:
        logging.error(colored(f"❌ Errore nel recupero del saldo: {e}", "red"))
        logging.error(colored("Riprovare successivamente", "yellow"))
        return 0

async def get_open_positions(exchange):
    try:
        positions = await exchange.fetch_positions(None, {
            'limit': 100, 
            'category': 'linear',
            'settleCoin': 'USDT'
        })
        return len([p for p in positions if float(p.get('contracts', 0)) > 0])
    except Exception as e:
        logging.error(colored(f"❌ Errore nel recupero delle posizioni aperte: {e}", "red"))
        return 0

async def calculate_position_size(exchange, symbol, usdt_balance, min_amount=0, risk_factor=1.0):
    try:
        ticker = await exchange.fetch_ticker(symbol)
        current_price = ticker.get('last')
        if current_price is None or not isinstance(current_price, (int, float)):
            logging.error(colored(f"❌ Prezzo corrente per {symbol} non disponibile", "red"))
            return None
        margin = MARGIN_USDT
        leverage = LEVERAGE
        notional_value = margin * leverage
        position_size = notional_value / current_price
        position_size = float(exchange.amount_to_precision(symbol, position_size))
        logging.info(colored(f"📏 Dimensione posizione per {symbol}: {position_size} contratti (Margine = {margin})", "cyan"))
        if position_size < min_amount:
            logging.warning(colored(f"⚠️ Dimensione posizione {position_size} inferiore al minimo {min_amount} per {symbol}.", "yellow"))
            position_size = min_amount
        return position_size
    except Exception as e:
        logging.error(colored(f"❌ Errore nel calcolo della dimensione per {symbol}: {e}", "red"))
        return None

async def manage_position(exchange, symbol, signal, usdt_balance, min_amounts,
                          lstm_model, lstm_scaler, rf_model, rf_scaler, df, predictions=None):
    current_time = time.time()
    new_im = 30.0
    total_im = await get_total_initial_margin(exchange, symbol)
    if total_im + new_im > 35.0:
        logging.info(colored(f"{symbol}: Apertura non consentita (IM totale superiore al limite).", "yellow"))
        return
    margin = MARGIN_USDT
    logging.info(colored(f"{symbol} - Utilizzo margine USDT: {margin:.2f}", "magenta"))
    position_size = await calculate_position_size(exchange, symbol, usdt_balance, min_amount=min_amounts.get(symbol, 0.1))
    if not position_size or position_size < min_amounts.get(symbol, 0.1):
        return
    ticker = await exchange.fetch_ticker(symbol)
    price = ticker.get('last')
    if price is None:
        logging.error(colored(f"❌ Prezzo corrente non disponibile per {symbol}", "red"))
        return
    if usdt_balance < 30.0:
        logging.warning(colored(f"{symbol}: Saldo USDT insufficiente.", "yellow"))
        return "insufficient_balance"
    try:
        await exchange.set_leverage(LEVERAGE, symbol)
    except Exception as lev_err:
        logging.warning(colored(f"{symbol}: Leva non modificata: {lev_err}", "yellow"))
    side = "Buy" if signal == 1 else "Sell"
    logging.info(colored(f"{symbol}: Ordine eseguito: {side}", "blue"))
    new_trade = await execute_order(exchange, symbol, side, position_size, price, current_time, df, predictions)
    return new_trade

async def execute_order(exchange, symbol, side, position_size, price, current_time, df, predictions=None):
    try:
        if side == "Buy":
            order = await exchange.create_market_buy_order(symbol, position_size)
        else:
            order = await exchange.create_market_sell_order(symbol, position_size)
    except Exception as e:
        error_str = str(e)
        if "110007" in error_str or "not enough" in error_str:
            logging.warning(colored(f"⚠️ Errore ordine per {symbol}: {error_str}", "yellow"))
            return "insufficient_balance"
        else:
            logging.error(colored(f"❌ Errore eseguendo ordine {side} per {symbol}: {error_str}", "red"))
            return None
    entry_price = order.get('average') or price
    trade_id = order.get("id") or f"{symbol}-{datetime.utcnow().timestamp()}"
    
    from data_utils import add_technical_indicators
    df = add_technical_indicators(df)
    
    new_trade = {
        "trade_id": trade_id,
        "symbol": symbol,
        "side": side,
        "entry_price": entry_price,
        "exit_price": None,
        "trade_type": "Open",
        "closed_pnl": None,
        "result": None,
        "open_trade_volume": None,
        "closed_trade_volume": None,
        "opening_fee": None,
        "closing_fee": None,
        "funding_fee": None,
        "trade_time": datetime.utcnow().isoformat(),
        "timestamp": datetime.utcnow().isoformat(),
        "status": "open"
    }
    
    logging.info(colored(f"🔔 Trade aperto: {new_trade}", "green"))
    return new_trade

async def get_total_initial_margin(exchange, symbol):
    try:
        positions = await exchange.fetch_positions(None, {
            'limit': 100, 
            'category': 'linear',
            'settleCoin': 'USDT'
        })
        total_im = 0.0
        for pos in positions:
            if pos.get('symbol') == symbol and float(pos.get('contracts', 0)) > 0:
                im = pos.get('initialMargin') or 30.0
                total_im += float(im)
        return total_im
    except Exception as e:
        logging.error(colored(f"❌ Errore nel recupero del margine iniziale per {symbol}: {e}", "red"))
        return 0.0

async def update_orders_status(exchange):
    pass

async def save_orders_tracker():
    pass

async def fetch_closed_orders_for_symbol(exchange, symbol, since, limit, semaphore):
    async with semaphore:
        try:
            orders = await exchange.fetch_closed_orders(symbol, since, limit)
            logging.info(colored(f"Recuperati {len(orders)} ordini chiusi per {symbol}.", "cyan"))
            await asyncio.sleep(0.3)
            return symbol, orders
        except Exception as e:
            logging.error(colored(f"❌ Errore nel recupero degli ordini chiusi per {symbol}: {e}", "red"))
            await asyncio.sleep(0.3)
            return symbol, f"Errore: {e}"

def aggregate_closed_orders(orders):
    aggregated_trades = []
    orders_by_symbol = {}
    for order in orders:
        symbol = order.get("symbol")
        orders_by_symbol.setdefault(symbol, []).append(order)
    
    for symbol, ord_list in orders_by_symbol.items():
        open_orders = [o for o in ord_list if o.get("info", {}).get("createType", "").lower() not in ["createbyclosing", "createbystoploss"]]
        close_orders = [o for o in ord_list if o.get("info", {}).get("createType", "").lower() in ["createbyclosing", "createbystoploss"]]
        open_orders.sort(key=lambda o: o.get("datetime"))
        close_orders.sort(key=lambda o: o.get("datetime"))
        pairs = min(len(open_orders), len(close_orders))
        for i in range(pairs):
            o_order = open_orders[i]
            c_order = close_orders[i]
            agg_trade_type = "Close Short" if o_order.get("side", "").lower() == "sell" else "Close Long"
            o_price = float(o_order.get("average", o_order.get("price", 0)))
            c_price = float(c_order.get("average", c_order.get("price", 0)))
            quantity = float(o_order.get("amount", 0))
            aggregated_pnl = (o_price - c_price) * quantity if agg_trade_type == "Close Short" else (c_price - o_price) * quantity
            o_fee = float(o_order.get("fee", {}).get("cost", 0))
            c_fee = float(c_order.get("fee", {}).get("cost", 0))
            trade_time = c_order.get("datetime")
            aggregated_trade = {
                "trade_id": o_order.get("id") + "_" + c_order.get("id"),
                "symbol": symbol,
                "side": o_order.get("side"),
                "entry_price": o_price,
                "exit_price": c_price,
                "trade_type": agg_trade_type,
                "closed_pnl": aggregated_pnl,
                "result": "Win" if aggregated_pnl > 0 else "Loss",
                "open_trade_volume": float(o_order.get("cost", 0)),
                "closed_trade_volume": float(c_order.get("cost", 0)),
                "opening_fee": o_fee,
                "closing_fee": c_fee,
                "funding_fee": 0.0,
                "trade_time": trade_time,
                "timestamp": trade_time,
                "status": "closed"
            }
            aggregated_trades.append(aggregated_trade)
    return aggregated_trades

async def wait_and_update_closed_trades(exchange, wait_duration=600, interval=10):
    end_time = time.time() + wait_duration
    while time.time() < end_time:
        await asyncio.sleep(interval)

async def monitor_open_trades(exchange):
    while True:
        await asyncio.sleep(30)

async def load_existing_positions(exchange):
    try:
        positions = await exchange.fetch_positions(None, {
            'limit': 100, 
            'category': 'linear',
            'settleCoin': 'USDT'
        })
        for pos in positions:
            contracts = float(pos.get("contracts", 0))
            if contracts > 0:
                symbol = pos.get("symbol")
                side = parse_position_side(pos)
                entry_price = float(pos.get("entryPrice", 0))
                logging.info(colored(f"Posizione aperta: {symbol}, {side}, {entry_price}", "green"))
        logging.info(colored("✅ Posizioni aperte caricate/aggiornate.", "green"))
    except Exception as e:
        logging.error(colored(f"❌ Errore nel caricamento delle posizioni aperte: {e}", "red"))

def build_trade_key(pos):
    tid = pos.get("id")
    if tid:
        return tid
    info = pos.get("info", {})
    created_time = info.get("createdTime")
    symbol = pos.get("symbol", "unknown")
    return f"{symbol}-{created_time}" if created_time else f"{symbol}-{uuid.uuid4()}"

def parse_position_side(position):
    side_field = position.get("side")
    if side_field and side_field.lower() in ["sell", "short"]:
        return "Sell"
    if "positionSide" in position:
        return "Sell" if position["positionSide"].upper() == "SHORT" else "Buy"
    return "Buy"