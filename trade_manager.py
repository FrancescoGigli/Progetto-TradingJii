#!/usr/bin/env python3
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
    MARGIN_USDT, LEVERAGE, EXCLUDED_SYMBOLS,
    TOP_ANALYSIS_CRYPTO
)
from fetcher import get_top_symbols, get_data_async

# Costante per il periodo di statistiche dei trade (in giorni)
TRADE_STATISTICS_DAYS = 30

# Strutture dati in memoria per sostituire il database
trades_db = []
trade_statistics_db = []

def is_symbol_excluded(symbol):
    normalized = re.sub(r'[^A-Za-z0-9]', '', symbol).upper()
    return any(exc.upper() in normalized for exc in EXCLUDED_SYMBOLS)

def init_db():
    logging.info(colored("Database in memoria inizializzato.", "green"))

init_db()

def clean_old_trades():
    cutoff = datetime.utcnow() - timedelta(days=TRADE_STATISTICS_DAYS)
    cutoff_iso = cutoff.isoformat()
    global trades_db
    trades_db = [trade for trade in trades_db if trade.get("timestamp", "") >= cutoff_iso]
    logging.info(colored(f"Puliti i trade antecedenti a {cutoff_iso}", "green"))

def save_trade_db(trade):
    global trades_db
    # Cerca se esiste già un trade con lo stesso ID
    for i, existing_trade in enumerate(trades_db):
        if existing_trade.get("trade_id") == trade.get("trade_id"):
            trades_db[i] = trade
            return
    # Se non esiste, aggiungi il nuovo trade
    trades_db.append(trade)
    logging.info(colored(f"Trade salvato in memoria: {trade.get('trade_id')}", "green"))

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
    logging.info(colored(f"Trade chiuso: {trade_record}", "green"))

async def get_real_balance(exchange):
    try:
        balance = await exchange.fetch_balance()
        usdt_balance = balance.get('USDT', {}).get('free', 0)
        if usdt_balance == 0:
            logging.warning(colored("Il saldo USDT è zero o non trovato.", "yellow"))
        return usdt_balance
    except Exception as e:
        logging.error(colored(f"Errore nel recupero del saldo: {e}", "red"))
        return None

async def get_open_positions(exchange):
    try:
        positions = await exchange.fetch_positions(None, {'limit': 100, 'type': 'swap'})
        return len([p for p in positions if float(p.get('contracts', 0)) > 0])
    except Exception as e:
        logging.error(colored(f"Errore nel recupero delle posizioni aperte: {e}", "red"))
        return 0

async def calculate_position_size(exchange, symbol, usdt_balance, min_amount=0, risk_factor=1.0):
    try:
        ticker = await exchange.fetch_ticker(symbol)
        current_price = ticker.get('last')
        if current_price is None or not isinstance(current_price, (int, float)):
            logging.error(colored(f"Prezzo corrente per {symbol} non disponibile", "red"))
            return None
        margin = MARGIN_USDT
        leverage = LEVERAGE
        notional_value = margin * leverage
        position_size = notional_value / current_price
        position_size = float(exchange.amount_to_precision(symbol, position_size))
        logging.info(colored(f"Dimensione posizione per {symbol}: {position_size} contratti (Margine = {margin})", "cyan"))
        if position_size < min_amount:
            logging.warning(colored(f"Dimensione posizione {position_size} inferiore al minimo {min_amount} per {symbol}.", "yellow"))
            position_size = min_amount
        return position_size
    except Exception as e:
        logging.error(colored(f"Errore nel calcolo della dimensione per {symbol}: {e}", "red"))
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
        logging.error(colored(f"Prezzo corrente non disponibile per {symbol}", "red"))
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
            logging.warning(colored(f"Errore ordine per {symbol}: {error_str}", "yellow"))
            return "insufficient_balance"
        else:
            logging.error(colored(f"Errore eseguendo ordine {side} per {symbol}: {error_str}", "red"))
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
    logging.info(colored(f"Trade aperto: {new_trade}", "green"))
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
        logging.error(colored(f"Errore nel recupero del margine iniziale per {symbol}: {e}", "red"))
        return 0.0

async def update_closed_trades(exchange):
    try:
        positions = await exchange.fetch_positions(None, {'limit': 100, 'type': 'swap'})
        # Logica per aggiornare i trade chiusi, se necessario.
    except Exception as e:
        logging.error(colored(f"Errore nell'aggiornamento dei trade chiusi: {e}", "red"))

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
        logging.info(colored("Posizioni aperte caricate/aggiornate.", "green"))
    except Exception as e:
        logging.error(colored(f"Errore nel caricamento delle posizioni aperte: {e}", "red"))

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

def compute_trade_statistics_for_trades(trades):
    total_closed = len(trades)
    total_wins = sum(1 for trade in trades if trade.get("result") == "Win")
    total_losses = sum(1 for trade in trades if trade.get("result") == "Loss")
    win_rate_percent = (total_wins / (total_wins + total_losses) * 100) if (total_wins + total_losses) > 0 else 0
    total_realizedpnl = sum((trade.get("closed_pnl") or 0) for trade in trades)
    return total_closed, total_wins, total_losses, win_rate_percent, total_realizedpnl

def compute_trade_statistics():
    global trades_db
    closed_trades = [trade for trade in trades_db if trade.get("status") == "closed"]
    return compute_trade_statistics_for_trades(closed_trades)

def compute_trade_statistics_for_period(period: timedelta):
    global trades_db
    now = datetime.utcnow()
    closed_trades = []
    for trade in trades_db:
        if trade.get("status") != "closed":
            continue
        try:
            ts = trade.get("timestamp", "")
            if ts.endswith("Z"):
                ts = ts[:-1]
            # Gestione di formati ISO diversi
            try:
                trade_time = datetime.fromisoformat(ts)
            except ValueError:
                # Prova un formato alternativo se il primo fallisce
                try:
                    trade_time = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%f")
                except ValueError:
                    try:
                        trade_time = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S")
                    except ValueError:
                        logging.warning(f"Impossibile analizzare il timestamp: {ts}")
                        continue
        except Exception as e:
            logging.warning(f"Errore nell'analisi del timestamp: {e}")
            continue
        if now - trade_time <= period:
            closed_trades.append(trade)
    return compute_trade_statistics_for_trades(closed_trades)

def get_trade_statistics_text():
    overall = compute_trade_statistics()
    last_24h = compute_trade_statistics_for_period(timedelta(hours=24))
    last_4h = compute_trade_statistics_for_period(timedelta(hours=4))
    last_1h = compute_trade_statistics_for_period(timedelta(hours=1))
    lines = []
    lines.append("============================================")
    lines.append("           STATISTICHE DEI TRADE")
    lines.append("============================================")
    lines.append(f"Ultimi {TRADE_STATISTICS_DAYS} giorni:")
    lines.append(f"   Trade chiusi : {overall[0]}")
    lines.append(f"   Vincite      : {overall[1]}")
    lines.append(f"   Perse        : {overall[2]}")
    lines.append(f"   Win Rate     : {overall[3]:.2f}%")
    lines.append(f"   PnL          : {overall[4]:.2f}")
    lines.append("--------------------------------------------")
    lines.append("Ultime 24h:")
    lines.append(f"   Trade chiusi : {last_24h[0]}")
    lines.append(f"   Vincite      : {last_24h[1]}")
    lines.append(f"   Perse        : {last_24h[2]}")
    lines.append(f"   Win Rate     : {last_24h[3]:.2f}%")
    lines.append(f"   PnL          : {last_24h[4]:.2f}")
    lines.append("--------------------------------------------")
    lines.append("Ultime 4h:")
    lines.append(f"   Trade chiusi : {last_4h[0]}")
    lines.append(f"   Vincite      : {last_4h[1]}")
    lines.append(f"   Perse        : {last_4h[2]}")
    lines.append(f"   Win Rate     : {last_4h[3]:.2f}%")
    lines.append(f"   PnL          : {last_4h[4]:.2f}")
    lines.append("--------------------------------------------")
    lines.append("Ultima 1h:")
    lines.append(f"   Trade chiusi : {last_1h[0]}")
    lines.append(f"   Vincite      : {last_1h[1]}")
    lines.append(f"   Perse        : {last_1h[2]}")
    lines.append(f"   Win Rate     : {last_1h[3]:.2f}%")
    lines.append(f"   PnL          : {last_1h[4]:.2f}")
    lines.append("============================================")
    return "\n".join(lines)

def save_trade_statistics():
    timestamp = datetime.utcnow().isoformat()
    overall = compute_trade_statistics()
    last_24h = compute_trade_statistics_for_period(timedelta(hours=24))
    last_4h = compute_trade_statistics_for_period(timedelta(hours=4))
    last_1h = compute_trade_statistics_for_period(timedelta(hours=1))
    
    global trade_statistics_db
    trade_statistics_db = [
        {
            "timestamp": timestamp,
            "period": f"Ultimi {TRADE_STATISTICS_DAYS} giorni",
            "total_closed_trades": overall[0],
            "total_wins": overall[1],
            "total_losses": overall[2],
            "win_rate_percent": overall[3],
            "total_realizedpnl": overall[4]
        },
        {
            "timestamp": timestamp,
            "period": "last_24h",
            "total_closed_trades": last_24h[0],
            "total_wins": last_24h[1],
            "total_losses": last_24h[2],
            "win_rate_percent": last_24h[3],
            "total_realizedpnl": last_24h[4]
        },
        {
            "timestamp": timestamp,
            "period": "last_4h",
            "total_closed_trades": last_4h[0],
            "total_wins": last_4h[1],
            "total_losses": last_4h[2],
            "win_rate_percent": last_4h[3],
            "total_realizedpnl": last_4h[4]
        },
        {
            "timestamp": timestamp,
            "period": "last_1h",
            "total_closed_trades": last_1h[0],
            "total_wins": last_1h[1],
            "total_losses": last_1h[2],
            "win_rate_percent": last_1h[3],
            "total_realizedpnl": last_1h[4]
        }
    ]
    logging.info(colored("Statistiche dei trade salvate in memoria.", "green"))

def print_trade_statistics():
    stats_text = get_trade_statistics_text()
    logging.info(colored(stats_text, "cyan"))