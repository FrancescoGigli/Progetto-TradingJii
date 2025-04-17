#!/usr/bin/env python3
import sqlite3
import re
import time
from datetime import datetime, timedelta
import logging
from termcolor import colored
import os
import asyncio
import uuid
import json

from config import (
    MARGIN_USDT, LEVERAGE, EXCLUDED_SYMBOLS, DB_FILE,
    TOP_ANALYSIS_CRYPTO, USE_DATABASE
)
from fetcher import get_top_symbols, get_data_async

def is_symbol_excluded(symbol):
    normalized = re.sub(r'[^A-Za-z0-9]', '', symbol).upper()
    return any(exc.upper() in normalized for exc in EXCLUDED_SYMBOLS)

def init_db():
    if not USE_DATABASE:
        logging.info(colored("Uso del database disattivato. Inizializzazione del database saltato.", "cyan"))
        return
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            trade_id TEXT PRIMARY KEY,
            symbol TEXT,
            side TEXT,
            entry_price REAL,
            exit_price REAL,
            trade_type TEXT,
            closed_pnl REAL,
            result TEXT,
            open_trade_volume REAL,
            closed_trade_volume REAL,
            opening_fee REAL,
            closing_fee REAL,
            funding_fee REAL,
            trade_time TEXT,
            timestamp TEXT,
            status TEXT
        )
    ''')
    conn.commit()
    conn.close()
    logging.info(colored("Database inizializzato.", "green"))

init_db()

def clean_old_trades():
    if not USE_DATABASE:
        logging.info(colored("Uso del database disattivato. Pulizia delle vecchie tabelle sul database saltata.", "cyan"))
        return
    cutoff = datetime.utcnow() - timedelta(days=30)
    cutoff_iso = cutoff.isoformat()
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM trades WHERE timestamp < ?", (cutoff_iso,))
        conn.commit()
        conn.close()
        logging.info(colored(f"Puliti i trade antecedenti a {cutoff_iso}", "green"))
    except Exception as e:
        logging.error(colored(f"[ERRORE] Errore nella pulizia dei vecchi trade: {e}", "red"))

def save_trade_db(trade):
    if not USE_DATABASE:
        logging.info(colored("Uso del database disattivato. Salvataggio trade saltato.", "cyan"))
        return
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO trades (
                trade_id, symbol, side, entry_price, exit_price, trade_type, closed_pnl,
                result, open_trade_volume, closed_trade_volume,
                opening_fee, closing_fee, funding_fee, trade_time, timestamp, status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade.get("trade_id"),
            trade.get("symbol"),
            trade.get("side"),
            trade.get("entry_price"),
            trade.get("exit_price"),
            trade.get("trade_type"),
            trade.get("closed_pnl"),
            trade.get("result"),
            trade.get("open_trade_volume"),
            trade.get("closed_trade_volume"),
            trade.get("opening_fee"),
            trade.get("closing_fee"),
            trade.get("funding_fee"),
            trade.get("trade_time"),
            trade.get("timestamp"),
            trade.get("status")
        ))
        conn.commit()
    except Exception as e:
        logging.error(colored(f"[ERRORE] Errore nel salvataggio del trade: {e}", "red"))
    finally:
        if USE_DATABASE:
            conn.close()

def close_trade_record(trade_record, exit_price):
    side = trade_record["side"]
    entry_price = trade_record["entry_price"]
    if side.lower() == "buy":
        realizedpnl = (exit_price - entry_price)
        win = exit_price > entry_price
    else:
        realizedpnl = (entry_price - exit_price)
        win = exit_price < entry_price
    trade_record["exit_price"] = exit_price
    trade_record["closed_pnl"] = realizedpnl
    trade_record["result"] = "Win" if win else "Loss"
    trade_record["status"] = "closed"
    now_iso = datetime.utcnow().isoformat()
    trade_record["trade_time"] = now_iso
    trade_record["timestamp"] = now_iso
    save_trade_db(trade_record)
    logging.info(colored(f"[OK] Trade chiuso: {trade_record}", "green"))

async def get_real_balance(exchange):
    try:
        balance = await exchange.fetch_balance()
        usdt_balance = balance.get('USDT', {}).get('free', 0)
        if usdt_balance == 0:
            logging.warning(colored("[AVVISO] Il saldo USDT è zero o non trovato.", "yellow"))
        return usdt_balance
    except Exception as e:
        logging.error(colored(f"[ERRORE] Errore nel recupero del saldo: {e}", "red"))
        return None

async def get_open_positions(exchange):
    try:
        positions = await exchange.fetch_positions(None, {'limit': 100, 'type': 'swap'})
        return len([p for p in positions if float(p.get('contracts', 0)) > 0])
    except Exception as e:
        logging.error(colored(f"[ERRORE] Errore nel recupero delle posizioni aperte: {e}", "red"))
        return 0

async def calculate_position_size(exchange, symbol, usdt_balance, min_amount=0, risk_factor=1.0):
    try:
        ticker = await exchange.fetch_ticker(symbol)
        current_price = ticker.get('last')
        if current_price is None or not isinstance(current_price, (int, float)):
            logging.error(colored(f"[ERRORE] Prezzo corrente per {symbol} non disponibile", "red"))
            return None
        margin = MARGIN_USDT
        leverage = LEVERAGE
        notional_value = margin * leverage
        position_size = notional_value / current_price
        position_size = float(exchange.amount_to_precision(symbol, position_size))
        logging.info(colored(f"[DIMENSIONE] Dimensione posizione per {symbol}: {position_size} contratti (Margine = {margin})", "cyan"))
        if position_size < min_amount:
            logging.warning(colored(f"[AVVISO] Dimensione posizione {position_size} inferiore al minimo {min_amount} per {symbol}.", "yellow"))
            position_size = min_amount
        return position_size
    except Exception as e:
        logging.error(colored(f"[ERRORE] Errore nel calcolo della dimensione per {symbol}: {e}", "red"))
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
        logging.error(colored(f"[ERRORE] Prezzo corrente non disponibile per {symbol}", "red"))
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
            logging.warning(colored(f"[AVVISO] Errore ordine per {symbol}: {error_str}", "yellow"))
            return "insufficient_balance"
        else:
            logging.error(colored(f"[ERRORE] Errore eseguendo ordine {side} per {symbol}: {error_str}", "red"))
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
    save_trade_db(new_trade)
    logging.info(colored(f"[NOTIFICA] Trade aperto: {new_trade}", "green"))
    return new_trade

async def get_total_initial_margin(exchange, symbol):
    try:
        positions = await exchange.fetch_positions(None, {'limit': 100, 'type': 'swap'})
        total_im = 0.0
        for pos in positions:
            if pos.get('symbol') == symbol and float(pos.get('contracts', 0)) > 0:
                im = pos.get('initialMargin') or 30.0
                total_im += float(im)
        return total_im
    except Exception as e:
        logging.error(colored(f"[ERRORE] Errore nel recupero del margine iniziale per {symbol}: {e}", "red"))
        return 0.0

async def update_orders_status(exchange):
    await save_orders_tracker()

async def save_orders_tracker():
    if not USE_DATABASE:
        logging.info(colored("Uso del database disattivato. Salvataggio dello stato degli ordini saltato.", "cyan"))
        return
    try:
        import aiofiles
        async with aiofiles.open("orders_status.json", 'w') as f:
            await f.write(json.dumps([], indent=2))
    except Exception as e:
        logging.error(colored(f"[ERRORE] Errore nel salvataggio dello stato degli ordini: {e}", "red"))

async def fetch_closed_orders_for_symbol(exchange, symbol, since, limit, semaphore):
    async with semaphore:
        try:
            orders = await exchange.fetch_closed_orders(symbol, since, limit)
            logging.info(colored(f"Recuperati {len(orders)} ordini chiusi per {symbol}.", "cyan"))
            await asyncio.sleep(0.3)
            return symbol, orders
        except Exception as e:
            logging.error(colored(f"[ERRORE] Errore nel recupero degli ordini chiusi per {symbol}: {e}", "red"))
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

async def update_closed_orders_from_account(exchange, since_date=None, limit=100):
    if not USE_DATABASE:
        logging.info(colored("Uso del database disattivato. update_closed_orders_from_account() saltata.", "cyan"))
        return
    if not exchange.symbols:
        await exchange.load_markets()
    try:
        since_str = (datetime.utcnow() - timedelta(days=30)).isoformat() + "Z" if since_date is None else since_date
        since_ts = exchange.parse8601(since_str)
        logging.info(colored(f"Recupero ordini chiusi da {since_str} (timestamp: {since_ts})", "cyan"))
    except Exception as e:
        logging.error(colored(f"[ERRORE] Errore nel parsing della data 'since': {e}", "red"))
        return

    all_symbols = [s for s in exchange.symbols if s.endswith(":USDT") and not is_symbol_excluded(s)]
    top_symbols = await get_top_symbols(exchange, all_symbols, top_n=TOP_ANALYSIS_CRYPTO)
    logging.info(colored(f"Utilizzo di {len(top_symbols)} simboli per il recupero degli ordini chiusi.", "cyan"))

    semaphore = asyncio.Semaphore(2)
    tasks = [fetch_closed_orders_for_symbol(exchange, sym, since_ts, limit, semaphore) for sym in top_symbols]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_closed_orders = []
    for sym, orders in results:
        if isinstance(orders, list):
            logging.info(colored(f"Recuperati {len(orders)} ordini per {sym}", "cyan"))
            all_closed_orders.extend(orders)
        else:
            logging.warning(colored(f"Errore nel recupero degli ordini per {sym}: {orders}", "yellow"))
    logging.info(colored(f"Totale ordini chiusi recuperati: {len(all_closed_orders)}", "cyan"))

    aggregated_trades = aggregate_closed_orders(all_closed_orders)
    for trade in aggregated_trades:
        try:
            save_trade_db(trade)
        except Exception as e:
            logging.error(colored(f"[ERRORE] Errore nel salvataggio del trade aggregato per {trade.get('symbol')}: {e}", "red"))
    logging.info(colored("[OK] Ordini chiusi aggregati e salvati nel DB.", "green"))

async def update_closed_trades(exchange):
    if not USE_DATABASE:
        logging.info(colored("Uso del database disattivato. Nessun aggiornamento dei trade sul database.", "cyan"))
        return
    try:
        positions = await exchange.fetch_positions(None, {'limit': 100, 'type': 'swap'})
        # Logica per aggiornare i trade chiusi, se necessario.
    except Exception as e:
        logging.error(colored(f"[ERRORE] Errore nell'aggiornamento dei trade chiusi: {e}", "red"))

async def wait_and_update_closed_trades(exchange, wait_duration=600, interval=10):
    end_time = time.time() + wait_duration
    while time.time() < end_time:
        await update_closed_trades(exchange)
        await asyncio.sleep(interval)

async def monitor_open_trades(exchange):
    while True:
        await update_closed_trades(exchange)
        await asyncio.sleep(30)

async def load_existing_positions(exchange):
    if not USE_DATABASE:
        logging.info(colored("Uso del database disattivato. Non carico le vecchie posizioni.", "cyan"))
        return
    try:
        positions = await exchange.fetch_positions(None, {'limit': 100, 'type': 'swap'})
        for pos in positions:
            contracts = float(pos.get("contracts", 0))
            if contracts > 0:
                trade_key = build_trade_key(pos)
                symbol = pos.get("symbol")
                side = parse_position_side(pos)
                entry_price = float(pos.get("entryPrice", 0))
                new_trade = {
                    "trade_id": trade_key,
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
                save_trade_db(new_trade)
        logging.info(colored("[OK] Posizioni aperte caricate/aggiornate.", "green"))
    except Exception as e:
        logging.error(colored(f"[ERRORE] Errore nel caricamento delle posizioni aperte: {e}", "red"))

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