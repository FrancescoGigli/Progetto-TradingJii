from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Header, Request
from pydantic import BaseModel
from typing import Optional, Dict, Any
import asyncio
from asyncio import all_tasks
import os
import sys
import uvicorn
import logging
import json
import re
import numpy as np
from pathlib import Path
from cryptography.fernet import Fernet
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, time
import ccxt
import ccxt.async_support as ccxt_async
import ta
import threading
import queue
import traceback
from trainer import (
    train_lstm_model_for_timeframe,
    train_random_forest_model_wrapper,
    train_xgboost_model_wrapper,
    ensure_trained_models_dir
)
import time

# Importa la funzione principale del bot
from main import main as run_bot
from trade_manager import (
    get_real_balance, get_open_positions,
    fetch_closed_orders_for_symbol, aggregate_closed_orders
)
from config import exchange_config, API_KEY, API_SECRET

# === CONFIGURAZIONE ===
API_KEY_FILE = Path("api_keys.enc")
FERNET_KEY_FILE = Path("fernet.key")

# === INIZIALIZZAZIONE FASTAPI ===
app = FastAPI(title="Trading Bot API", version="1.0")
logger = logging.getLogger("uvicorn")

# === CONFIGURAZIONE CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In produzione specificare domini esatti
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

bot_task = None
bot_running = False
async_exchange = None

# === MODELLI ===
class ApiKeys(BaseModel):
    api_key: str
    secret_key: str

class ClosePositionRequest(BaseModel):
    symbol: str
    side: str

# === MODELLI DI RISPOSTA ===
class ApiKeyRequest(BaseModel):
    api_key: str
    api_secret: str

# === UTILITÀ ===
def load_fernet():
    if not FERNET_KEY_FILE.exists():
        key = Fernet.generate_key()
        FERNET_KEY_FILE.write_bytes(key)
    else:
        key = FERNET_KEY_FILE.read_bytes()
    return Fernet(key)

def save_api_keys_secure(keys: ApiKeys):
    fernet = load_fernet()
    data = json.dumps(keys.dict()).encode()
    encrypted = fernet.encrypt(data)
    API_KEY_FILE.write_bytes(encrypted)

def load_api_keys_secure():
    if not API_KEY_FILE.exists():
        return None
    fernet = load_fernet()
    decrypted = fernet.decrypt(API_KEY_FILE.read_bytes())
    return json.loads(decrypted)

def verify_auth_token(api_key: str = Header(None, alias="api-key"), api_secret: str = Header(None, alias="api-secret")):
    if api_key != API_KEY or api_secret != API_SECRET:
        raise HTTPException(status_code=401, detail="Credenziali non autorizzate")

# === ENDPOINT ===

@app.post("/start")
async def start_bot(background_tasks: BackgroundTasks, auth: None = Depends(verify_auth_token)):
    global bot_task, bot_running
    if bot_running:
        raise HTTPException(status_code=400, detail="Bot già in esecuzione")

    bot_task = asyncio.create_task(run_bot())
    bot_running = True
    return {"status": "Bot avviato"}

@app.post("/stop")
async def stop_bot(auth: None = Depends(verify_auth_token)):
    global bot_task, bot_running
    if bot_task and not bot_task.done():
        bot_task.cancel()
        bot_running = False
        return {"status": "Stop richiesto"}
    else:
        raise HTTPException(status_code=400, detail="Bot non in esecuzione")

@app.get("/status")
def status(auth: None = Depends(verify_auth_token)):
    return {"running": bot_running}

@app.post("/set-keys")
def set_api_keys(keys: ApiKeys, auth: None = Depends(verify_auth_token)):
    try:
        save_api_keys_secure(keys)
        return {"status": "Chiavi salvate in modo sicuro"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore salvataggio chiavi: {e}")

@app.get("/get-keys")
def get_api_keys(auth: None = Depends(verify_auth_token)):
    try:
        keys = load_api_keys_secure()
        return keys or {"status": "Chiavi non trovate"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore lettura chiavi: {e}")

@app.get("/balance")
async def get_balance(auth: None = Depends(verify_auth_token)):
    global async_exchange
    if not async_exchange:
        async_exchange = ccxt_async.bybit(exchange_config)
        await async_exchange.load_markets()
    try:
        # Recupera il saldo completo
        balance = await async_exchange.fetch_balance()
        
        # Estrai le informazioni dettagliate dal bilancio
        detailed_balance = {
            "usdt_balance": 0,  # Sarà calcolato come somma di tutti i tipi di account
            "total_wallet": 0,
            "available": 0,
            "used": 0,
            "pnl": 0
        }
        
        # Estrai informazioni da account unificato se disponibile
        if 'info' in balance:
            if 'totalEquity' in balance['info']:
                detailed_balance["total_wallet"] = float(balance['info']['totalEquity'])
            if 'totalAvailableBalance' in balance['info']:
                detailed_balance["available"] = float(balance['info']['totalAvailableBalance'])
            if 'totalInitialMargin' in balance['info']:
                detailed_balance["used"] = float(balance['info']['totalInitialMargin'])
            if 'totalUnrealizedProfit' in balance['info']:
                detailed_balance["pnl"] = float(balance['info']['totalUnrealizedProfit'])
        
        # Calcola i valori mancanti dai dati disponibili
        if detailed_balance["total_wallet"] == 0 and 'total' in balance and 'USDT' in balance['total']:
            detailed_balance["total_wallet"] = float(balance['total']['USDT'])
        
        if detailed_balance["available"] == 0 and 'free' in balance and 'USDT' in balance['free']:
            detailed_balance["available"] = float(balance['free']['USDT'])
        
        if detailed_balance["used"] == 0 and 'used' in balance and 'USDT' in balance['used']:
            detailed_balance["used"] = float(balance['used']['USDT'])
            
        # Calcola il bilancio USDT sommando tutti i tipi di account
        # Unified account
        unified_balance = 0
        if 'info' in balance and 'totalEquity' in balance['info']:
            unified_balance = float(balance['info']['totalEquity'])
        
        # Spot account e altri account
        spot_balance = 0
        if 'USDT' in balance.get('total', {}):
            spot_balance = float(balance['total']['USDT'])
        
        # Control account o altri bilanci
        control_balance = 0
        
        # Verifica se esistono altri account o wallet nella struttura
        if 'accounts' in balance:
            for account in balance['accounts']:
                if 'USDT' in account.get('total', {}):
                    control_balance += float(account['total']['USDT'])
        
        # Calcola il bilancio USDT totale
        total_usdt = unified_balance + spot_balance + control_balance
        
        # Se ancora non abbiamo un valore, usiamo il bilancio USDT standard
        if total_usdt == 0:
            total_usdt = await get_real_balance(async_exchange)
        
        detailed_balance["usdt_balance"] = total_usdt
        
        # Se il PnL è ancora 0, recuperalo direttamente dalle posizioni aperte
        if detailed_balance["pnl"] == 0:
            try:
                # Modifica i parametri per la chiamata API a Bybit
                # Utilizziamo parametri più specifici per la v5 API di Bybit
                positions = await async_exchange.fetch_positions(None, {
                    'limit': 100, 
                    'category': 'linear',
                    'settleCoin': 'USDT'
                })
                total_pnl = 0.0
                
                # Calcola il PnL totale dalle posizioni aperte
                for position in positions:
                    if float(position.get('contracts', 0)) > 0:
                        position_pnl = float(position.get('unrealizedPnl', 0))
                        total_pnl += position_pnl
                        
                detailed_balance["pnl"] = total_pnl
                
            except Exception as e:
                logging.error(f"Errore nel recupero del PnL dalle posizioni: {e}")
                # Aggiungiamo più dettagli all'errore per il debug
                logging.error(f"Dettagli errore: {str(e)}")
                # Impostiamo un valore di default per il PnL in caso di errore
                detailed_balance["pnl"] = 0.0
        
        return detailed_balance
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore recupero balance: {e}")

@app.get("/positions")
async def get_positions(auth: None = Depends(verify_auth_token)):
    global async_exchange
    if not async_exchange:
        async_exchange = ccxt_async.bybit(exchange_config)
        await async_exchange.load_markets()
    try:
        count = await get_open_positions(async_exchange)
        return {"open_positions": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore recupero posizioni: {e}")

@app.get("/orders/open")
async def get_open_orders(auth: None = Depends(verify_auth_token)):
    global async_exchange
    if not async_exchange:
        async_exchange = ccxt_async.bybit(exchange_config)
        await async_exchange.load_markets()
    try:
        # Recupera ordini aperti
        open_orders = await async_exchange.fetch_open_orders()
        
        # Recupera anche le posizioni aperte
        positions = await async_exchange.fetch_positions(None, {
            'limit': 100, 
            'category': 'linear',
            'settleCoin': 'USDT'
        })
        
        # Ottieni il leverage configurato
        from config import LEVERAGE, MARGIN_USDT
        leverage = LEVERAGE
        
        # Mappatura delle posizioni attive per simbolo e lato
        active_position_keys = {}
        for p in positions:
            if float(p.get("contracts", 0)) > 0:
                symbol = p.get("symbol")
                side = p.get("side")  # long o short
                active_position_keys[f"{symbol}_{side}"] = True
        
        # Identifica tutti i tipi di ordini correlati (stop loss, take profit, ecc.)
        related_orders = {}
        stop_loss_orders = {}
        
        for order in open_orders:
            symbol = order.get('symbol')
            order_side = order.get('side', '').lower()  # buy o sell
            
            # Guarda i campi che indicano ordini correlati
            is_related = False
            
            # Controlla i vari tipi di ordini correlati basati su campi Bybit
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
            
            # Verifica se l'ordine è opposto a una posizione attiva (potenziale take profit o stop loss)
            # Ad esempio: un ordine buy per una posizione short o un ordine sell per una posizione long
            opposite_side = "long" if order_side == "sell" else "short"
            if f"{symbol}_{opposite_side}" in active_position_keys:
                is_related = True
                
            # Controlla se c'è una corrispondenza di quantità con una posizione esistente
            order_amount = float(order.get('amount', 0))
            for p in positions:
                if p.get("symbol") == symbol and abs(float(p.get("contracts", 0)) - order_amount) < 0.1:
                    # La quantità è quasi uguale (con tolleranza)
                    is_related = True
                    break
            
            if is_related:
                # Salva come ordine correlato
                if symbol not in related_orders:
                    related_orders[symbol] = []
                related_orders[symbol].append(order)
                
                # Se è specificamente uno stop loss, salvalo nel dizionario stop_loss_orders
                if order.get('info', {}).get('stopOrderType') == 'Stop' or 'stopPrice' in order.get('info', {}):
                    stop_loss_orders[symbol] = order
        
        # Filtra gli ordini per escludere quelli correlati a posizioni
        filtered_orders = []
        for order in open_orders:
            symbol = order.get('symbol')
            
            # Salta gli ordini correlati a posizioni
            if symbol in related_orders and order in related_orders[symbol]:
                continue
                
            # Aggiungi l'ordine alla lista filtrata
            filtered_orders.append(order)
        
        # Filtra solo le posizioni con contratti > 0
        active_positions = []
        for p in positions:
            if float(p.get("contracts", 0)) > 0:
                # Tenta di recuperare lo stop loss se esiste
                stop_loss_value = "N/A"
                potential_profit = "N/A"
                
                # Verifica se la posizione ha stop loss
                symbol = p.get("symbol")
                sl_price = p.get("stopLossPrice")
                entry_price = p.get("entryPrice")
                contracts = float(p.get("contracts", 0))
                side = p.get("side")
                
                # Se esiste un ordine di stop loss per questo simbolo, usa quello
                if symbol in stop_loss_orders:
                    sl_order = stop_loss_orders[symbol]
                    sl_price = sl_order.get('stopPrice', None) or sl_order.get('info', {}).get('stopPrice')
                
                # Se abbiamo un stop loss e un prezzo di entrata, calcoliamo il profitto potenziale
                if sl_price and sl_price != 0 and entry_price:
                    stop_loss_value = sl_price
                    
                    # Calcola profitto/perdita potenziale
                    sl_price = float(sl_price)
                    entry_price = float(entry_price)
                    
                    if side == "long":
                        potential_profit = (sl_price - entry_price) * contracts
                    else:
                        potential_profit = (entry_price - sl_price) * contracts
                
                active_positions.append({
                    "symbol": symbol,
                    "type": "position",  # Indica che è una posizione e non un ordine
                    "side": "Buy" if side == "long" else "Sell",
                    "amount": p.get("contracts"),
                    "price": p.get("entryPrice"),
                    "status": "open",
                    "pnl": p.get("unrealizedPnl", 0),
                    "margin": p.get("initialMargin", MARGIN_USDT),  # Margine utilizzato per la posizione
                    "leverage": p.get("leverage", leverage),  # Leva utilizzata
                    "stop_loss": stop_loss_value,  # Valore di stop loss
                    "sl_profit": potential_profit  # Profitto potenziale se si attiva lo stop loss
                })
        
        # Unisci ordini (filtrati) e posizioni
        all_open_items = filtered_orders + active_positions
        
        return all_open_items
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore recupero open orders: {e}")

@app.get("/orders/closed")
async def get_closed_orders(auth: None = Depends(verify_auth_token)):
    global async_exchange
    if not async_exchange:
        async_exchange = ccxt_async.bybit(exchange_config)
        await async_exchange.load_markets()
    try:
        closed_orders = await async_exchange.fetch_closed_orders()
        return closed_orders
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore recupero closed orders: {e}")

@app.get("/trades")
async def get_closed_trades(auth: None = Depends(verify_auth_token)):
    try:
        global async_exchange
        if not async_exchange:
            async_exchange = ccxt_async.bybit(exchange_config)
            await async_exchange.load_markets()
        trades = await async_exchange.fetch_my_trades()
        return {"trades": trades}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    try:
        from config import exchange_config
        import ccxt
        exchange = ccxt.bybit(exchange_config)
        exchange.load_markets()
        return {"status": "ok", "exchange": "bybit"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore connessione exchange: {e}")

@app.get("/chart-data/{symbol:path}")
async def get_chart_data(symbol: str, timeframe: str = "15m", limit: int = 100, auth: None = Depends(verify_auth_token)):
    global async_exchange
    if not async_exchange:
        async_exchange = ccxt_async.bybit(exchange_config)
        await async_exchange.load_markets()
    try:
        # Recupera i dati OHLCV (Open, High, Low, Close, Volume)
        # Gestiamo correttamente il simbolo che può contenere caratteri speciali
        symbol = symbol.replace("%3A", ":")  # Sostituisci %3A con :
        
        logging.info(f"Recupero dati grafico per simbolo: {symbol}, timeframe: {timeframe}")
        
        ohlcv = await async_exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        # Formatta i dati per grafici a candele
        chart_data = {
            "timestamps": [],  # Timestamp Unix
            "labels": [],      # Timestamp formattato
            "open": [],        # Prezzi di apertura
            "high": [],        # Prezzi massimi
            "low": [],         # Prezzi minimi
            "close": [],       # Prezzi di chiusura
            "volumes": []      # Volumi
        }
        
        for candle in ohlcv:
            timestamp = candle[0]
            open_price = candle[1]
            high_price = candle[2]
            low_price = candle[3]
            close_price = candle[4]
            volume = candle[5]
            
            # Converti timestamp Unix in formato leggibile
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
async def close_position_endpoint(request: ClosePositionRequest, auth: None = Depends(verify_auth_token)):
    global async_exchange
    if not async_exchange:
        async_exchange = ccxt_async.bybit(exchange_config)
        await async_exchange.load_markets()
    try:
        # Convertiamo il side dal nostro formato al formato dell'exchange
        # Dobbiamo chiudere con l'ordine opposto
        close_side = "sell" if request.side.lower() == "buy" else "buy"
        
        # Recupera la posizione per ottenere la dimensione
        positions = await async_exchange.fetch_positions([request.symbol], {
            'limit': 100, 
            'category': 'linear',
            'settleCoin': 'USDT'
        })
        
        # Trova la posizione corrispondente
        position = None
        for p in positions:
            if p.get("symbol") == request.symbol and float(p.get("contracts", 0)) > 0:
                position = p
                break
                
        if not position:
            raise HTTPException(status_code=404, detail=f"Posizione {request.symbol} non trovata")
        
        # Ottieni l'importo della posizione
        amount = float(position.get("contracts", 0))
        
        # Chiudi la posizione con un ordine di mercato
        order = await async_exchange.create_market_order(
            symbol=request.symbol,
            side=close_side,
            amount=amount,
            params={"reduceOnly": True}
        )
        
        return {"status": "success", "message": f"Posizione {request.symbol} chiusa", "order": order}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore nella chiusura della posizione: {str(e)}")

# Dizionario per tenere traccia dello stato del training
training_status = {}

@app.post("/api/train-model")
async def train_model(request: Request, auth: None = Depends(verify_auth_token)):
    try:
        data = await request.json()
        model_type = data.get("model_type")
        timeframe = data.get("timeframe")
        data_limit_days = data.get("data_limit_days", 30)
        top_train_crypto = data.get("top_train_crypto", 5)
        
        # Inizializza lo stato del training
        training_key = f"{model_type}_{timeframe}"
        training_status[training_key] = {
            "status": "running",
            "progress": 0,
            "current_step": "Inizializzazione...",
            "error": None,
            "metrics": None,
            "start_time": time.time()
        }
        
        # Avvia il training in un thread separato
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
async def get_training_status(model_type: str, timeframe: str, auth: None = Depends(verify_auth_token)):
    training_key = f"{model_type}_{timeframe}"
    if training_key not in training_status:
        raise HTTPException(status_code=404, detail="Training non trovato")
    
    status = training_status[training_key]
    
    # Calcola il tempo stimato rimanente
    if status["status"] == "running":
        elapsed_time = time.time() - status["start_time"]
        if status["progress"] > 0:
            estimated_total = elapsed_time / (status["progress"] / 100)
            remaining_time = max(0, estimated_total - elapsed_time)
            status["estimated_time"] = f"{int(remaining_time / 60)}:{int(remaining_time % 60):02d}"
        else:
            status["estimated_time"] = "--:--"
    
    return status

# Funzione per eseguire il processo di training come in main.py
def run_model_training(model_type, timeframe, data_limit_days, top_train_crypto, training_key):
    try:
        # Per Windows, utilizziamo una policy più sicura per l'event loop
        if sys.platform.startswith('win'):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        # Crea un nuovo event loop per questo thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Esecuzione del processo asincrono
            loop.run_until_complete(train_process(model_type, timeframe, data_limit_days, top_train_crypto, training_key))
        finally:
            # Assicurati di chiudere il loop correttamente
            try:
                # Chiudi eventuali task pendenti
                pending = asyncio.all_tasks(loop)
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception as e:
                logging.warning(f"Errore durante la chiusura dei task pendenti: {e}")
            finally:
                loop.close()
    except Exception as e:
        logging.error(f"Errore durante il training: {e}")
        import traceback
        logging.error(traceback.format_exc())
        # Assicurati che lo stato sia aggiornato anche in caso di errore
        if training_key in training_status:
            training_status[training_key]["status"] = "error"
            training_status[training_key]["error"] = str(e)
            training_status[training_key]["current_step"] = f"Errore durante il training: {str(e)}"
        raise

# Funzione asincrona per eseguire il processo di training
async def train_process(model_type, timeframe, data_limit_days, top_train_crypto, training_key):
    try:
        import sys
        import os
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
        
        # Aggiorna stato: Inizializzazione
        training_status[training_key]["current_step"] = "Inizializzazione dell'exchange..."
        training_status[training_key]["progress"] = 5
        
        # Inizializzazione dell'exchange
        exchange = ccxt_async.bybit(exchange_config)
        try:
            await exchange.load_markets()
            await exchange.load_time_difference()
            
            # Aggiorna stato: Recupero mercati
            training_status[training_key]["current_step"] = "Recupero dei mercati disponibili..."
            training_status[training_key]["progress"] = 10
            
            # Recupero dei mercati disponibili
            markets = await fetch_markets(exchange)
            all_symbols = [m['symbol'] for m in markets.values() if m.get('quote') == 'USDT'
                           and m.get('active') and m.get('type') == 'swap']
            
            all_symbols_analysis = [s for s in all_symbols if not re.search('|'.join(EXCLUDED_SYMBOLS), s)]
            
            # Aggiorna stato: Recupero simboli per training
            training_status[training_key]["current_step"] = f"Recupero dei top {top_train_crypto} simboli per il training..."
            training_status[training_key]["progress"] = 20
            
            # Recupero dei simboli per il training
            top_symbols_training = await get_top_symbols(exchange, all_symbols, top_n=top_train_crypto)
            
            # Aggiorna stato: Validazione dati
            training_status[training_key]["current_step"] = "Validazione dei dati..."
            training_status[training_key]["progress"] = 30
            
            # Validazione dei dati
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
            
            # Aggiorna stato: Preparazione dati
            training_status[training_key]["current_step"] = "Preparazione dei dati per il training..."
            training_status[training_key]["progress"] = 40
            
            # Aggiornamento della lista di simboli validi per il training
            top_symbols_training = valid_symbols
            
            # Verifica directory dei modelli
            ensure_trained_models_dir()
            
            # Aggiorna stato: Training modello
            training_status[training_key]["current_step"] = f"Training del modello {model_type}..."
            training_status[training_key]["progress"] = 50
            
            # Training del modello specificato
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
            
            # Aggiorna stato: Completato
            training_status[training_key]["status"] = "completed"
            training_status[training_key]["progress"] = 100
            training_status[training_key]["current_step"] = "Training completato con successo!"
            training_status[training_key]["metrics"] = metrics
            
            # Chiudi l'exchange in modo sicuro
            try:
                await exchange.close()
            except Exception as e:
                logging.warning(f"Errore durante la chiusura dell'exchange: {e}")
        except Exception as e:
            # Assicurati di chiudere l'exchange anche in caso di errore
            try:
                await exchange.close()
            except:
                pass
            raise e
        
    except Exception as e:
        # Aggiorna stato: Errore
        training_status[training_key]["status"] = "error"
        training_status[training_key]["error"] = str(e)
        training_status[training_key]["current_step"] = f"Errore durante il training: {str(e)}"
        logging.error(f"Errore durante il training: {e}")
        import traceback
        logging.error(traceback.format_exc())
        raise

@app.get("/check-model-exists/{file_name}")
async def check_model_exists(file_name: str, auth: None = Depends(verify_auth_token)):
    """Verifica se un modello specifico esiste nella cartella trained_models."""
    try:
        trained_models_dir = os.path.join(os.path.dirname(__file__), 'trained_models')
        file_path = os.path.join(trained_models_dir, file_name)
        
        if os.path.exists(file_path):
            return {"exists": True, "file_name": file_name}
        else:
            raise HTTPException(status_code=404, detail=f"Modello {file_name} non trovato")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list-models")
async def list_models(auth: None = Depends(verify_auth_token)):
    """Restituisce la lista di tutti i modelli disponibili nella cartella trained_models."""
    try:
        trained_models_dir = os.path.join(os.path.dirname(__file__), 'trained_models')
        
        # Crea la directory se non esiste
        os.makedirs(trained_models_dir, exist_ok=True)
        
        # Ottieni la lista dei file
        model_files = [f for f in os.listdir(trained_models_dir) 
                      if os.path.isfile(os.path.join(trained_models_dir, f)) 
                      and (f.endswith('.h5') or f.endswith('.pkl'))]
        
        # Organizza i modelli per tipo
        models_by_type = {
            "lstm": [f for f in model_files if f.startswith('lstm_model_')],
            "rf": [f for f in model_files if f.startswith('rf_model_')],
            "xgb": [f for f in model_files if f.startswith('xgb_model_')]
        }
        
        return {
            "models": model_files,
            "models_by_type": models_by_type
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trained_models/{filename}")
async def get_trained_model_file(filename: str, auth: None = Depends(verify_auth_token)):
    """Restituisce il contenuto di un file dalla cartella trained_models."""
    try:
        trained_models_dir = os.path.join(os.path.dirname(__file__), 'trained_models')
        file_path = os.path.join(trained_models_dir, filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File {filename} non trovato")
        
        # Leggi il contenuto del file
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Se è un JSON, restituiscilo come dizionario
        if filename.endswith(".json"):
            return json.loads(content)
        else:
            # Per altri tipi di file, restituisci una risposta testuale
            return content
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"Errore nel parsing del file JSON {filename}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predictions")
async def get_predictions(auth: None = Depends(verify_auth_token)):
    """Restituisce le predizioni correnti per i top simboli in analisi."""
    try:
        from predictor import predict_signal_ensemble, get_color_normal
        from config import (
            ENABLED_TIMEFRAMES, TIME_STEPS, TOP_ANALYSIS_CRYPTO,
            TIMEFRAME_DEFAULT, MODEL_RATES
        )
        from model_loader import (
            load_lstm_model_func, load_random_forest_model_func, load_xgboost_model_func
        )
        from fetcher import fetch_markets, get_top_symbols, fetch_data_for_multiple_symbols
        import re
        
        # Inizializza l'exchange
        global async_exchange
        if not async_exchange:
            async_exchange = ccxt_async.bybit(exchange_config)
            await async_exchange.load_markets()
        
        # Carica i modelli se non sono già caricati
        lstm_models, lstm_scalers = {}, {}
        rf_models, rf_scalers = {}, {}
        xgb_models, xgb_scalers = {}, {}
        
        # Carica i modelli in parallelo per tutti i timeframe abilitati
        model_load_tasks = []
        
        for tf in ENABLED_TIMEFRAMES:
            model_load_tasks.append(('lstm', tf, asyncio.create_task(
                asyncio.to_thread(load_lstm_model_func, tf))))
            model_load_tasks.append(('rf', tf, asyncio.create_task(
                asyncio.to_thread(load_random_forest_model_func, tf))))
            model_load_tasks.append(('xgb', tf, asyncio.create_task(
                asyncio.to_thread(load_xgboost_model_func, tf))))
        
        for model_type, tf, task in model_load_tasks:
            try:
                result = await task
                if model_type == 'lstm':
                    lstm_models[tf], lstm_scalers[tf] = result
                elif model_type == 'rf':
                    rf_models[tf], rf_scalers[tf] = result
                elif model_type == 'xgb':
                    xgb_models[tf], xgb_scalers[tf] = result
            except Exception as e:
                logging.error(f"Errore nel caricamento del modello {model_type} per {tf}: {e}")
        
        # Recupera i mercati disponibili
        markets = await fetch_markets(async_exchange)
        all_symbols = [m['symbol'] for m in markets.values() if m.get('quote') == 'USDT'
                       and m.get('active') and m.get('type') == 'swap']
        
        # Filtra i simboli esclusi
        from config import EXCLUDED_SYMBOLS
        all_symbols_analysis = [s for s in all_symbols if not re.search('|'.join(EXCLUDED_SYMBOLS), s)]
        
        # Ottieni i top simboli per analisi
        top_symbols_analysis = await get_top_symbols(async_exchange, all_symbols_analysis, top_n=TOP_ANALYSIS_CRYPTO)
        
        # Limita a massimo 15 simboli per una risposta API veloce
        top_symbols_analysis = top_symbols_analysis[:15]
        
        # Calcola i pesi normalizzati per i modelli
        normalized_weights = {}
        for tf in ENABLED_TIMEFRAMES:
            normalized_weights[tf] = {
                'lstm': MODEL_RATES['lstm'],
                'rf': MODEL_RATES['rf'],
                'xgb': MODEL_RATES['xgb']
            }
        
        # Recupera i dati per ogni simbolo e genera le predizioni
        predictions = []
        
        # Esegui il recupero dati per tutti i timeframe in parallelo
        timeframe_tasks = {}
        for tf in ENABLED_TIMEFRAMES:
            task = asyncio.create_task(fetch_data_for_multiple_symbols(
                async_exchange, top_symbols_analysis, timeframe=tf
            ))
            timeframe_tasks[tf] = task
        
        # Attendi tutti i risultati
        dataframes_by_symbol = {}
        for tf, task in timeframe_tasks.items():
            result_dict = await task
            for symbol, df in result_dict.items():
                if df is not None:
                    if symbol not in dataframes_by_symbol:
                        dataframes_by_symbol[symbol] = {}
                    dataframes_by_symbol[symbol][tf] = df
        
        # Genera le predizioni per ogni simbolo
        for symbol in top_symbols_analysis:
            if symbol in dataframes_by_symbol and len(dataframes_by_symbol[symbol]) == len(ENABLED_TIMEFRAMES):
                try:
                    ensemble_value, final_signal, prediction_details = predict_signal_ensemble(
                        dataframes_by_symbol[symbol],
                        lstm_models, lstm_scalers,
                        rf_models, rf_scalers,
                        xgb_models, xgb_scalers,
                        symbol, TIME_STEPS,
                        {tf: normalized_weights[tf]['lstm'] for tf in ENABLED_TIMEFRAMES},
                        {tf: normalized_weights[tf]['rf'] for tf in ENABLED_TIMEFRAMES},
                        {tf: normalized_weights[tf]['xgb'] for tf in ENABLED_TIMEFRAMES}
                    )
                    
                    if ensemble_value is not None:
                        # Estrai i dettagli delle predizioni
                        lstm_preds, rf_preds, xgb_preds, rsi_value = prediction_details
                        
                        # Calcola il colore in base all'ensemble value
                        color = get_color_normal(ensemble_value)
                        
                        # Ottieni la direzione del segnale in base al valore finale
                        direction = "Neutrale"
                        if final_signal == 1:
                            direction = "Buy"
                        elif final_signal == 0:
                            direction = "Sell"
                        
                        # Aggiungi il risultato alle predizioni
                        predictions.append({
                            "symbol": symbol,
                            "ensemble_value": float(ensemble_value),
                            "direction": direction,
                            "signal": final_signal,
                            "color": color,
                            "rsi_value": float(rsi_value),
                            "models": {
                                "lstm": {tf: float(lstm_preds[tf]) for tf in ENABLED_TIMEFRAMES},
                                "rf": {tf: float(rf_preds[tf]) for tf in ENABLED_TIMEFRAMES},
                                "xgb": {tf: float(xgb_preds[tf]) for tf in ENABLED_TIMEFRAMES}
                            }
                        })
                except Exception as e:
                    logging.error(f"Errore nella predizione per {symbol}: {e}")
        
        # Ordina le predizioni per ensemble_value
        predictions.sort(key=lambda x: x["ensemble_value"], reverse=True)
        
        return {
            "predictions": predictions,
            "timeframes": ENABLED_TIMEFRAMES,
            "default_timeframe": TIMEFRAME_DEFAULT
        }
    except Exception as e:
        logging.error(f"Errore generale nell'endpoint delle predizioni: {e}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Errore nel generare le predizioni: {e}")

# === AVVIO MANUALE ===
if __name__ == "__main__":
    # Configura il logger per inviare i log solo al terminale
    logger.setLevel(logging.DEBUG)

    # Rimuovi tutti gli handler esistenti
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Aggiungi uno StreamHandler per inviare i log al terminale
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)