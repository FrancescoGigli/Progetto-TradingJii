# app.py
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import asyncio
import os
import sys
import json
import re
import time
import traceback
import logging
from datetime import datetime
from pathlib import Path

import ccxt.async_support as ccxt_async
import numpy as np
import uvicorn
from cryptography.fernet import Fernet
from dotenv import load_dotenv
import ccxt
import ta
import threading
import queue

# Moduli locali
from trainer import (
    train_lstm_model_for_timeframe,
    train_random_forest_model_wrapper,
    train_xgboost_model_wrapper,
    ensure_trained_models_dir
)
from main import main as run_bot
from trade_manager import get_real_balance, get_open_positions
from config import exchange_config, API_KEY, API_SECRET, EXCLUDED_SYMBOLS, TOP_ANALYSIS_CRYPTO
from fetcher import (
    fetch_markets, get_top_symbols, fetch_min_amounts,
    fetch_data_for_multiple_symbols, get_data_async
)

# === CONFIGURAZIONE FILE CHIAVI ===
API_KEY_FILE = Path("api_keys.enc")
FERNET_KEY_FILE = Path("fernet.key")

# === INIZIALIZZAZIONE FASTAPI ===
app = FastAPI(title="Trading Bot API", version="1.0")
logger = logging.getLogger("uvicorn")

# Configurazione CORS (in produzione è preferibile restringere i domini ammessi)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inizializza lo stato condiviso in app.state
app.state.bot_task = None
app.state.bot_running = False
app.state.async_exchange = None
app.state.current_config = None
app.state.initialized = False

# === MODELLI DI DATI ===
class ApiKeys(BaseModel):
    api_key: str
    secret_key: str

class BotConfig(BaseModel):
    models: List[str]
    timeframes: List[str]

class ClosePositionRequest(BaseModel):
    symbol: str
    side: str

class ApiKeyRequest(BaseModel):
    api_key: str
    api_secret: str

# === FUNZIONI DI UTILITÀ ===
def load_fernet():
    """Carica o genera la chiave Fernet per la crittografia."""
    if not FERNET_KEY_FILE.exists():
        key = Fernet.generate_key()
        FERNET_KEY_FILE.write_bytes(key)
    else:
        key = FERNET_KEY_FILE.read_bytes()
    return Fernet(key)

def save_api_keys_secure(keys: ApiKeys):
    """Cripta e salva le chiavi API in modo sicuro."""
    fernet = load_fernet()
    data = json.dumps(keys.dict()).encode()
    encrypted = fernet.encrypt(data)
    API_KEY_FILE.write_bytes(encrypted)

def load_api_keys_secure():
    """Legge le chiavi API criptate dal file."""
    if not API_KEY_FILE.exists():
        return None
    fernet = load_fernet()
    decrypted = fernet.decrypt(API_KEY_FILE.read_bytes())
    return json.loads(decrypted)

def verify_auth_token(api_key: str = Header(None, alias="api-key"), 
                        api_secret: str = Header(None, alias="api-secret")):
    """Verifica che le credenziali presenti nell'header siano corrette."""
    if api_key != API_KEY or api_secret != API_SECRET:
        raise HTTPException(status_code=401, detail="Credenziali non autorizzate")

# Dependency per lo stato
def get_state(request: Request):
    return request.app.state

# === ENDPOINTS API ===

@app.post("/initialize")
async def initialize_bot(config: BotConfig, 
                           auth: None = Depends(verify_auth_token),
                           state = Depends(get_state)):
    """
    Inizializza il bot con i modelli e i timeframe selezionati.
    Verifica i parametri e inizializza l'exchange.
    """
    try:
        # Validazione dei timeframe
        valid_timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '4h', '1d']
        if not all(tf in valid_timeframes for tf in config.timeframes):
            raise HTTPException(status_code=400, detail="Timeframe non valido")
        if not (1 <= len(config.timeframes) <= 3):
            raise HTTPException(status_code=400, detail="Numero di timeframe non valido (min: 1, max: 3)")
        
        # Validazione dei modelli
        valid_models = ['lstm', 'rf', 'xgb']
        if not all(model in valid_models for model in config.models):
            raise HTTPException(status_code=400, detail="Modello non valido")
        if not (1 <= len(config.models) <= 3):
            raise HTTPException(status_code=400, detail="Numero di modelli non valido (min: 1, max: 3)")
        
        # Inizializza l'exchange se non già presente
        if not state.async_exchange:
            state.async_exchange = ccxt_async.bybit(exchange_config)
            await state.async_exchange.load_markets()
            await state.async_exchange.load_time_difference()
        
        state.current_config = config

        # Aggiorna la configurazione nel modulo principale
        import main
        main.ENABLED_TIMEFRAMES = config.timeframes
        main.TIMEFRAME_DEFAULT = config.timeframes[0]
        main.selected_models = config.models
        
        state.initialized = True

        return {
            "status": "Bot inizializzato", 
            "config": config.dict(),
            "exchange": "Bybit",
            "initialized": state.initialized
        }
    except Exception as e:
        logging.error(f"Errore durante l'inizializzazione: {str(e)}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Errore durante l'inizializzazione: {str(e)}")

@app.post("/start")
async def start_bot(background_tasks: BackgroundTasks, 
                    auth: None = Depends(verify_auth_token),
                    state = Depends(get_state)):
    """
    Avvia il bot se non è già in esecuzione.
    """
    if state.bot_running:
        raise HTTPException(status_code=400, detail="Bot già in esecuzione")
    if not state.current_config:
        raise HTTPException(status_code=400, detail="Bot non inizializzato. Chiamare prima /initialize")
    
    state.bot_task = asyncio.create_task(run_bot())
    state.bot_running = True
    return {"status": "Bot avviato", "config": state.current_config.dict()}

@app.post("/stop")
async def stop_bot(auth: None = Depends(verify_auth_token),
                   state = Depends(get_state)):
    """
    Richiede l'arresto del bot se in esecuzione.
    """
    if state.bot_task and not state.bot_task.done():
        state.bot_task.cancel()
        state.bot_running = False
        return {"status": "Stop richiesto"}
    else:
        raise HTTPException(status_code=400, detail="Bot non in esecuzione")

@app.get("/status")
def status(auth: None = Depends(verify_auth_token),
           state = Depends(get_state)):
    """
    Restituisce lo stato corrente del bot.
    """
    return {
        "running": state.bot_running,
        "config": state.current_config.dict() if state.current_config else None
    }

@app.post("/set-keys")
def set_api_keys(keys: ApiKeys, 
                 auth: None = Depends(verify_auth_token)):
    """
    Salva le chiavi API in maniera sicura.
    """
    try:
        save_api_keys_secure(keys)
        return {"status": "Chiavi salvate in modo sicuro"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore salvataggio chiavi: {e}")

@app.get("/get-keys")
def get_api_keys(auth: None = Depends(verify_auth_token)):
    """
    Recupera le chiavi API salvate.
    """
    try:
        keys = load_api_keys_secure()
        return keys or {"status": "Chiavi non trovate"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore lettura chiavi: {e}")

@app.get("/balance")
async def get_balance(auth: None = Depends(verify_auth_token),
                      state = Depends(get_state)):
    """
    Recupera il bilancio completo dall'exchange e restituisce una struttura dettagliata.
    """
    if not state.async_exchange:
        state.async_exchange = ccxt_async.bybit(exchange_config)
        await state.async_exchange.load_markets()
    try:
        balance = await state.async_exchange.fetch_balance()
        detailed_balance = {
            "usdt_balance": 0,
            "total_wallet": 0,
            "available": 0,
            "used": 0,
            "pnl": 0
        }
        if 'info' in balance:
            detailed_balance["total_wallet"] = float(balance['info'].get('totalEquity', 0))
            detailed_balance["available"] = float(balance['info'].get('totalAvailableBalance', 0))
            detailed_balance["used"] = float(balance['info'].get('totalInitialMargin', 0))
            detailed_balance["pnl"] = float(balance['info'].get('totalUnrealizedProfit', 0))
        
        if detailed_balance["total_wallet"] == 0 and 'total' in balance and 'USDT' in balance['total']:
            detailed_balance["total_wallet"] = float(balance['total']['USDT'])
        if detailed_balance["available"] == 0 and 'free' in balance and 'USDT' in balance['free']:
            detailed_balance["available"] = float(balance['free']['USDT'])
        if detailed_balance["used"] == 0 and 'used' in balance and 'USDT' in balance['used']:
            detailed_balance["used"] = float(balance['used']['USDT'])
        
        unified_balance = float(balance['info'].get('totalEquity', 0)) if 'info' in balance else 0
        spot_balance = float(balance.get('total', {}).get('USDT', 0))
        control_balance = 0
        if 'accounts' in balance:
            for account in balance['accounts']:
                control_balance += float(account.get('total', {}).get('USDT', 0))
        total_usdt = unified_balance + spot_balance + control_balance
        if total_usdt == 0:
            total_usdt = get_real_balance(state.async_exchange)
        detailed_balance["usdt_balance"] = total_usdt
        
        if detailed_balance["pnl"] == 0:
            try:
                positions = await state.async_exchange.fetch_positions(None, {
                    'limit': 100, 
                    'category': 'linear',
                    'settleCoin': 'USDT'
                })
                total_pnl = 0.0
                for position in positions:
                    if float(position.get('contracts', 0)) > 0:
                        total_pnl += float(position.get('unrealizedPnl', 0))
                detailed_balance["pnl"] = total_pnl
            except Exception as e:
                logging.error(f"Errore nel recupero del PnL dalle posizioni: {e}")
                detailed_balance["pnl"] = 0.0
                
        return detailed_balance
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore recupero balance: {e}")

@app.get("/positions")
async def get_positions(auth: None = Depends(verify_auth_token),
                        state = Depends(get_state)):
    """
    Recupera il numero di posizioni aperte.
    """
    try:
        if not state.async_exchange:
            state.async_exchange = ccxt_async.bybit(exchange_config)
            await state.async_exchange.load_markets()
        count = await get_open_positions(state.async_exchange)
        return {"open_positions": count}
    except Exception as e:
        logging.error(f"Errore nel recupero delle posizioni: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/orders/open")
async def get_open_orders(auth: None = Depends(verify_auth_token),
                          state = Depends(get_state)):
    """
    Recupera e filtra gli ordini aperti e le posizioni attive.
    """
    if not state.async_exchange:
        state.async_exchange = ccxt_async.bybit(exchange_config)
        await state.async_exchange.load_markets()
    try:
        open_orders = await state.async_exchange.fetch_open_orders()
        positions = await state.async_exchange.fetch_positions(None, {
            'limit': 100, 
            'category': 'linear',
            'settleCoin': 'USDT'
        })
        from config import LEVERAGE, MARGIN_USDT
        leverage = LEVERAGE
        
        # Mappa le posizioni attive per simbolo e lato
        active_position_keys = {}
        for p in positions:
            if float(p.get("contracts", 0)) > 0:
                symbol = p.get("symbol")
                side = p.get("side")
                active_position_keys[f"{symbol}_{side}"] = True
        
        related_orders = {}
        stop_loss_orders = {}
        for order in open_orders:
            symbol = order.get('symbol')
            order_side = order.get('side', '').lower()
            is_related = False
            if order.get('info', {}).get('stopOrderType') == 'Stop':
                is_related = True
            elif 'stopPrice' in order.get('info', {}):
                is_related = True
            elif order.get('info', {}).get('reduceOnly') is True:
                is_related = True
            elif order.get('info', {}).get('closeOnTrigger') is True:
                is_related = True
            elif order.get('info', {}).get('positionIdx') is not None:
                is_related = True
                
            opposite_side = "long" if order_side == "sell" else "short"
            if f"{symbol}_{opposite_side}" in active_position_keys:
                is_related = True
            
            order_amount = float(order.get('amount', 0))
            for p in positions:
                if p.get("symbol") == symbol and abs(float(p.get("contracts", 0)) - order_amount) < 0.1:
                    is_related = True
                    break
            
            if is_related:
                related_orders.setdefault(symbol, []).append(order)
                if order.get('info', {}).get('stopOrderType') == 'Stop' or 'stopPrice' in order.get('info', {}):
                    stop_loss_orders[symbol] = order
        
        filtered_orders = [
            order for order in open_orders
            if order.get('symbol') not in related_orders or order not in related_orders[order.get('symbol')]
        ]
        
        active_positions = []
        for p in positions:
            if float(p.get("contracts", 0)) > 0:
                symbol = p.get("symbol")
                side = p.get("side")
                stop_loss_value = "N/A"
                potential_profit = "N/A"
                entry_price = p.get("entryPrice")
                contracts = float(p.get("contracts", 0))
                if symbol in stop_loss_orders:
                    sl_order = stop_loss_orders[symbol]
                    sl_price = sl_order.get('stopPrice', None) or sl_order.get('info', {}).get('stopPrice')
                else:
                    sl_price = p.get("stopLossPrice")
                if sl_price and sl_price != 0 and entry_price:
                    stop_loss_value = sl_price
                    sl_price = float(sl_price)
                    entry_price = float(entry_price)
                    potential_profit = (sl_price - entry_price) * contracts if side == "long" else (entry_price - sl_price) * contracts
                
                active_positions.append({
                    "symbol": symbol,
                    "type": "position",
                    "side": "Buy" if side == "long" else "Sell",
                    "amount": p.get("contracts"),
                    "price": entry_price,
                    "status": "open",
                    "pnl": p.get("unrealizedPnl", 0),
                    "margin": p.get("initialMargin", MARGIN_USDT),
                    "leverage": p.get("leverage", leverage),
                    "stop_loss": stop_loss_value,
                    "sl_profit": potential_profit
                })
        
        all_open_items = filtered_orders + active_positions
        
        return all_open_items
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore recupero open orders: {e}")

@app.get("/orders/closed")
async def get_closed_orders(auth: None = Depends(verify_auth_token),
                            state = Depends(get_state)):
    """
    Recupera gli ordini chiusi dall'exchange.
    """
    if not state.async_exchange:
        state.async_exchange = ccxt_async.bybit(exchange_config)
        await state.async_exchange.load_markets()
    try:
        closed_orders = await state.async_exchange.fetch_closed_orders()
        return closed_orders
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore recupero closed orders: {e}")

@app.get("/trades")
async def get_closed_trades(auth: None = Depends(verify_auth_token),
                            state = Depends(get_state)):
    """
    Recupera gli ultimi trade eseguiti e li formatta.
    """
    try:
        if not state.async_exchange:
            state.async_exchange = ccxt_async.bybit(exchange_config)
            await state.async_exchange.load_markets()
        trades = await state.async_exchange.fetch_my_trades(limit=50)
        formatted_trades = [{
            'symbol': trade['symbol'],
            'type': trade['type'],
            'side': trade['side'],
            'amount': trade['amount'],
            'price': trade['price'],
            'realized_pnl': trade.get('info', {}).get('closedPnl', 0),
            'timestamp': trade['timestamp']
        } for trade in trades]
        return formatted_trades
    except Exception as e:
        logging.error(f"Errore nel recupero dei trade: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """
    Verifica la connessione con l'exchange.
    """
    try:
        exchange = ccxt.bybit(exchange_config)
        exchange.load_markets()
        return {"status": "ok", "exchange": "bybit"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore connessione exchange: {e}")

@app.get("/chart-data/{symbol:path}")
async def get_chart_data(symbol: str, timeframe: str = "15m", limit: int = 100,
                         auth: None = Depends(verify_auth_token),
                         state = Depends(get_state)):
    """
    Recupera e formatta i dati OHLCV per il simbolo e timeframe specificati.
    """
    if not state.async_exchange:
        state.async_exchange = ccxt_async.bybit(exchange_config)
        await state.async_exchange.load_markets()
    try:
        symbol = symbol.replace("%3A", ":")
        logging.info(f"Recupero dati grafico per simbolo: {symbol}, timeframe: {timeframe}")
        ohlcv = await state.async_exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        chart_data = {
            "timestamps": [],
            "labels": [],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volumes": []
        }
        for candle in ohlcv:
            timestamp, open_price, high_price, low_price, close_price, volume = candle
            date = datetime.fromtimestamp(timestamp / 1000)
            formatted_date = date.strftime('%H:%M %d/%m')
            chart_data["timestamps"].append(timestamp)
            chart_data["labels"].append(formatted_date)
            chart_data["open"].append(open_price)
            chart_data["high"].append(high_price)
            chart_data["low"].append(low_price)
            chart_data["close"].append(close_price)
            chart_data["volumes"].append(volume)
        
        return chart_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore nel recupero dei dati del grafico: {e}")

@app.post("/close-position")
async def close_position_endpoint(request: ClosePositionRequest,
                                  auth: None = Depends(verify_auth_token),
                                  state = Depends(get_state)):
    """
    Chiude una posizione per il simbolo indicato, inviando un ordine di mercato
    con lato opposto a quello della posizione corrente.
    """
    if not state.async_exchange:
        state.async_exchange = ccxt_async.bybit(exchange_config)
        await state.async_exchange.load_markets()
    try:
        close_side = "sell" if request.side.lower() == "buy" else "buy"
        positions = await state.async_exchange.fetch_positions([request.symbol], {
            'limit': 100, 
            'category': 'linear',
            'settleCoin': 'USDT'
        })
        position = next((p for p in positions if p.get("symbol") == request.symbol and float(p.get("contracts", 0)) > 0), None)
        if not position:
            raise HTTPException(status_code=404, detail=f"Posizione {request.symbol} non trovata")
        amount = float(position.get("contracts", 0))
        order = await state.async_exchange.create_market_order(
            symbol=request.symbol,
            side=close_side,
            amount=amount,
            params={"reduceOnly": True}
        )
        return {"status": "success", "message": f"Posizione {request.symbol} chiusa", "order": order}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore nella chiusura della posizione: {str(e)}")

# === TRAINING MODEL ===
# La struttura training_status viene gestita internamente a questo file
training_status = {}

@app.post("/api/train-model")
async def train_model(request: Request,
                      auth: None = Depends(verify_auth_token)):
    """
    Avvia il processo di training per il modello specificato in un thread separato.
    """
    try:
        data = await request.json()
        model_type = data.get("model_type")
        timeframe = data.get("timeframe")
        data_limit_days = data.get("data_limit_days", 30)
        top_train_crypto = data.get("top_train_crypto", 5)
        
        training_key = f"{model_type}_{timeframe}"
        training_status[training_key] = {
            "status": "running",
            "progress": 0,
            "current_step": "Inizializzazione...",
            "error": None,
            "metrics": None,
            "start_time": time.time()
        }
        thread = threading.Thread(
            target=run_model_training,
            args=(model_type, timeframe, data_limit_days, top_train_crypto, training_key)
        )
        thread.daemon = True
        thread.start()
        
        return {"message": f"Training avviato per il modello {model_type} con timeframe {timeframe}"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/training-status/{model_type}/{timeframe}")
async def get_training_status(model_type: str, timeframe: str,
                              auth: None = Depends(verify_auth_token)):
    """
    Restituisce lo stato del training per il modello e timeframe richiesti.
    """
    training_key = f"{model_type}_{timeframe}"
    if training_key not in training_status:
        raise HTTPException(status_code=404, detail="Training non trovato")
    
    status_info = training_status[training_key]
    if status_info["status"] == "running":
        elapsed_time = time.time() - status_info["start_time"]
        if status_info["progress"] > 0:
            estimated_total = elapsed_time / (status_info["progress"] / 100)
            remaining_time = max(0, estimated_total - elapsed_time)
            status_info["estimated_time"] = f"{int(remaining_time / 60)}:{int(remaining_time % 60):02d}"
        else:
            status_info["estimated_time"] = "--:--"
    
    return status_info

def run_model_training(model_type, timeframe, data_limit_days, top_train_crypto, training_key):
    """
    Esegue il training del modello in un thread separato gestendo l'event loop.
    """
    try:
        if sys.platform.startswith('win'):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(train_process(model_type, timeframe, data_limit_days, top_train_crypto, training_key))
        finally:
            try:
                pending = asyncio.all_tasks(loop)
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception as e:
                logging.warning(f"Errore durante la chiusura dei task pendenti: {e}")
            finally:
                loop.close()
    except Exception as e:
        logging.error(f"Errore durante il training: {e}")
        logging.error(traceback.format_exc())
        if training_key in training_status:
            training_status[training_key]["status"] = "error"
            training_status[training_key]["error"] = str(e)
            training_status[training_key]["current_step"] = f"Errore durante il training: {str(e)}"
        raise

async def train_process(model_type, timeframe, data_limit_days, top_train_crypto, training_key):
    """
    Processo asincrono che esegue il training del modello utilizzando i dati e i simboli validati.
    """
    try:
        from config import (
            exchange_config, EXCLUDED_SYMBOLS, TIME_STEPS, 
            TOP_TRAIN_CRYPTO, TOP_ANALYSIS_CRYPTO, EXPECTED_COLUMNS,
            SYMBOLS_PER_ANALYSIS_CYCLE, SYMBOLS_FOR_VALIDATION,
            TRAIN_IF_NOT_FOUND, ENABLED_TIMEFRAMES, TIMEFRAME_DEFAULT, 
            SELECTED_MODELS as selected_models
        )
        from fetcher import (
            fetch_markets, get_top_symbols, fetch_min_amounts,
            fetch_data_for_multiple_symbols, get_data_async
        )
        training_status[training_key]["current_step"] = "Inizializzazione dell'exchange..."
        training_status[training_key]["progress"] = 5
        
        exchange = ccxt_async.bybit(exchange_config)
        try:
            await exchange.load_markets()
            await exchange.load_time_difference()
            
            training_status[training_key]["current_step"] = "Recupero dei mercati disponibili..."
            training_status[training_key]["progress"] = 10
            
            markets = await fetch_markets(exchange)
            all_symbols = [m['symbol'] for m in markets.values() if m.get('quote') == 'USDT'
                           and m.get('active') and m.get('type') == 'swap']
            
            all_symbols_analysis = [s for s in all_symbols if not re.search('|'.join(EXCLUDED_SYMBOLS), s)]
            
            training_status[training_key]["current_step"] = f"Recupero dei top {top_train_crypto} simboli per il training..."
            training_status[training_key]["progress"] = 20
            
            top_symbols_training = await get_top_symbols(exchange, all_symbols, top_n=top_train_crypto)
            
            training_status[training_key]["current_step"] = "Validazione dei dati..."
            training_status[training_key]["progress"] = 30
            
            valid_symbols = []
            for tf in ENABLED_TIMEFRAMES:
                result_dict = await fetch_data_for_multiple_symbols(
                    exchange, 
                    top_symbols_training[:SYMBOLS_FOR_VALIDATION],
                    timeframe=tf
                )
                valid_for_tf = []
                for symbol, df in result_dict.items():
                    if df is not None and not (df.isnull().any().any() or np.isinf(df).any().any()):
                        valid_for_tf.append(symbol)
                if not valid_symbols:
                    valid_symbols = valid_for_tf
                else:
                    valid_symbols = [s for s in valid_symbols if s in valid_for_tf]
            
            training_status[training_key]["current_step"] = "Preparazione dei dati per il training..."
            training_status[training_key]["progress"] = 40
            
            top_symbols_training = valid_symbols
            ensure_trained_models_dir()
            
            training_status[training_key]["current_step"] = f"Training del modello {model_type}..."
            training_status[training_key]["progress"] = 50
            
            if model_type == 'lstm':
                model, scaler, metrics = await train_lstm_model_for_timeframe(
                    exchange, top_symbols_training, timeframe=timeframe, timestep=TIME_STEPS)
            elif model_type == 'rf':
                model, scaler, metrics = await train_random_forest_model_wrapper(
                    top_symbols_training, exchange, timestep=TIME_STEPS, timeframe=timeframe)
            elif model_type == 'xgb':
                model, scaler, metrics = await train_xgboost_model_wrapper(
                    top_symbols_training, exchange, timestep=TIME_STEPS, timeframe=timeframe)
            else:
                raise ValueError(f"Tipo di modello non supportato: {model_type}")
            
            training_status[training_key]["status"] = "completed"
            training_status[training_key]["progress"] = 100
            training_status[training_key]["current_step"] = "Training completato con successo!"
            training_status[training_key]["metrics"] = metrics
            
            try:
                await exchange.close()
            except Exception as e:
                logging.warning(f"Errore durante la chiusura dell'exchange: {e}")
        except Exception as e:
            try:
                await exchange.close()
            except:
                pass
            raise e
        
    except Exception as e:
        training_status[training_key]["status"] = "error"
        training_status[training_key]["error"] = str(e)
        training_status[training_key]["current_step"] = f"Errore durante il training: {str(e)}"
        logging.error(f"Errore durante il training: {e}")
        logging.error(traceback.format_exc())
        raise

# === AVVIO MANUALE DELL'APPLICAZIONE ===
if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    # Rimuove eventuali handler esistenti e ne aggiunge uno per il terminale
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
