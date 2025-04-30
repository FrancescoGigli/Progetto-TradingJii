# app.py – TradingJii API
# Versione rivista 28/04/2025

from __future__ import annotations

# ───────────────────────────────────────────────────────────────────────────────
# Caricamento variabili d'ambiente PRIMA di ogni altro import
# ───────────────────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

import os
print(f"DEBUG: API_KEY={'SET' if os.getenv('API_KEY') else 'MISSING'}, "
      f"API_SECRET={'SET' if os.getenv('API_SECRET') else 'MISSING'}")

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    Request,
    BackgroundTasks,
    Body,
    APIRouter,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict
from cryptography.fernet import Fernet

# CCXT asincrono (compatibilità con versioni 3.x e PRO)
try:
    import ccxt.async_support as ccxt_async  # ccxt < 4.x
except ModuleNotFoundError:
    import ccxt.pro as ccxt_async            # ccxt-pro (licenza)
import ccxt

import numpy as np
import ta
import uvicorn
import time

from model_manager import (
    train_lstm_model_for_timeframe,
    train_random_forest_model_wrapper,
    train_xgboost_model_wrapper,
    ensure_trained_models_dir,
)
from main import main as run_bot
from trade_manager import (
    get_real_balance,
    get_open_positions,
)
from config import (
    exchange_config,
    API_KEY,
    API_SECRET,
    EXCLUDED_SYMBOLS,
    TOP_ANALYSIS_CRYPTO,
    LEVERAGE,
    MARGIN_USDT,
)
from fetcher import (
    fetch_markets,
    get_top_symbols,
    fetch_min_amounts,
    fetch_and_save_data,
    get_data_async,
)
from state import app_state
from dependencies import verify_auth_token, generate_auth_token  # import centralizzato per autenticazione

# ───────────────────────────────────────────────────────────────────────────────
# Configurazione dell'exchange (chiavi + rate limit)
# ───────────────────────────────────────────────────────────────────────────────
exchange_config.update({
    'apiKey':          API_KEY,
    'secret':          API_SECRET,
    'enableRateLimit': True,
})

# ───────────────────────────────────────────────────────────────────────────────
# Logger e file per chiavi cifrate
# ───────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger("uvicorn")
API_KEY_FILE    = Path("api_keys.enc")
FERNET_KEY_FILE = Path("fernet.key")

# ───────────────────────────────────────────────────────────────────────────────
# Utility di crittografia
# ───────────────────────────────────────────────────────────────────────────────
def _load_fernet() -> Fernet:
    env_key = os.getenv("FERNET_SECRET_KEY")
    if env_key:
        return Fernet(env_key.encode())
    if not FERNET_KEY_FILE.exists():
        FERNET_KEY_FILE.write_bytes(Fernet.generate_key())
    return Fernet(FERNET_KEY_FILE.read_bytes())

def save_api_keys_secure(keys: ApiKeys) -> None:
    f = _load_fernet()
    API_KEY_FILE.write_bytes(f.encrypt(json.dumps(keys.dict()).encode()))

def load_api_keys_secure() -> Optional[dict]:
    if not API_KEY_FILE.exists():
        return None
    f = _load_fernet()
    return json.loads(f.decrypt(API_KEY_FILE.read_bytes()))

# ───────────────────────────────────────────────────────────────────────────────
# Schemi Pydantic
# ───────────────────────────────────────────────────────────────────────────────
class ApiKeys(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    api_key:    str
    secret_key: str

class BotConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    models:         List[str]
    timeframes:     List[str]
    trading_params: Optional[Dict[str, Any]] = None

class ClosePositionRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    symbol: str
    side:   str  # "Buy" o "Sell"

# ───────────────────────────────────────────────────────────────────────────────
# Lifespan: bootstrap e teardown dell'exchange
# ───────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: inizializza ccxt_async.bybit e carica i mercati
    ex = ccxt_async.bybit(exchange_config)
    await ex.load_markets()
    app_state.async_exchange = ex

    yield

    # Shutdown: chiudi exchange e task pendenti
    if app_state.async_exchange:
        await app_state.async_exchange.close()
    if (task := app.state.bot_task) and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    app_state.bot_running = False

# ───────────────────────────────────────────────────────────────────────────────
# Creazione app FastAPI
# ───────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Trading Bot API", version="2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.state.bot_task: Optional[asyncio.Task] = None
app.mount("/trained_models", StaticFiles(directory="trained_models"), name="trained_models")

def get_state(request: Request):
    return request.app.state

root = APIRouter()
api  = APIRouter(prefix="/api")

# ───────────────────────────────────────────────────────────────────────────────
# ENDPOINT: initialize
# ───────────────────────────────────────────────────────────────────────────────
@root.post("/initialize")
@api.post("/initialize")
async def initialize_bot(
    config: BotConfig,
    state=Depends(get_state),
    _=Depends(verify_auth_token),
):
    valid_tfs    = {'5m','15m','30m','1h','4h'}
    valid_models = {'lstm','rf','xgb'}

    if not set(config.timeframes) <= valid_tfs:
        raise HTTPException(400, "Timeframe non valido.")
    if not (1 <= len(config.timeframes) <= 3):
        raise HTTPException(400, "Numero di timeframe non valido (min 1, max 3).")
    if not set(config.models) <= valid_models:
        raise HTTPException(400, "Modello non valido.")
    if not (1 <= len(config.models) <= 3):
        raise HTTPException(400, "Numero di modelli non valido (min 1, max 3).")

    # Ricrea l'exchange se già esistente
    if app_state.async_exchange:
        await app_state.async_exchange.close()

    app_state.async_exchange = ccxt_async.bybit(exchange_config)
    await app_state.async_exchange.load_markets()

    app_state.initialized = True
    app_state.update_from_config(config.dict())

    return {
        "status":      "Bot inizializzato",
        "config":      config.dict(),
        "exchange":    "Bybit",
        "initialized": True,
    }

# ───────────────────────────────────────────────────────────────────────────────
# START / STOP / STATUS
# ───────────────────────────────────────────────────────────────────────────────
@root.post("/start")
@api.post("/start")
async def start_bot(
    background_tasks: BackgroundTasks,
    state=Depends(get_state),
    _=Depends(verify_auth_token),
):
    if app_state.bot_running:
        raise HTTPException(400, "Bot già in esecuzione.")
    if not app_state.initialized:
        raise HTTPException(400, "Bot non inizializzato. Richiamare /initialize.")

    async def _runner():
        try:
            logger.info("Avvio del bot in corso…")
            await run_bot()
        except asyncio.CancelledError:
            logger.info("Task bot annullata.")
            raise
        except Exception as exc:
            logger.error("Errore bot: %s", exc, exc_info=True)
        finally:
            app_state.bot_running = False

    state.bot_task        = asyncio.create_task(_runner())
    app_state.bot_running = True
    return {"status": "Bot avviato"}

@root.post("/stop")
@api.post("/stop")
async def stop_bot(state=Depends(get_state), _=Depends(verify_auth_token)):
    if not app_state.bot_running or not state.bot_task:
        return {"status": "Bot già fermo"}
    state.bot_task.cancel()
    app_state.bot_running = False
    return {"status": "Bot fermato"}

@root.get("/status")
@api.get("/status")
def bot_status(state=Depends(get_state), _=Depends(verify_auth_token)):
    running = app_state.bot_running and state.bot_task and not state.bot_task.done()
    if not running:
        app_state.bot_running = False
    return {
        "running":     running,
        "config":      app_state.current_config.dict() if app_state.current_config else None,
        "task_status": getattr(state.bot_task, "_state", "not_created") if state.bot_task else "not_created",
    }

# ───────────────────────────────────────────────────────────────────────────────
# HEALTH-CHECK
# ───────────────────────────────────────────────────────────────────────────────
@root.get("/health")
@api.get("/health")
def health_check():
    try:
        ex = ccxt.bybit({**exchange_config, "apiKey": None, "secret": None})
        ex.load_markets()
        ex.close()
        return {"status": "ok", "exchange": "bybit"}
    except Exception as exc:
        return {"status": "ko", "error": str(exc)}

# ───────────────────────────────────────────────────────────────────────────────
# BALANCE
# ───────────────────────────────────────────────────────────────────────────────
@root.get("/balance")
@api.get("/balance")
async def get_balance(state=Depends(get_state), _=Depends(verify_auth_token)):
    try:
        if not app_state.async_exchange:
            app_state.async_exchange = ccxt_async.bybit(exchange_config)
            await app_state.async_exchange.load_markets()

        balance = await app_state.async_exchange.fetch_balance()
        logger.info(f"[DEBUG BALANCE] {balance}")  # Log dettagliato
        det = {
            "usdt_balance": 0.0,
            "total_wallet": 0.0,
            "available": 0.0,
            "used": 0.0,
            "pnl": 0.0,
        }

        try:
            # --- Estraggo i dati dalla struttura coin se Bybit Unified ---
            coins = None
            if (
                "info" in balance and
                isinstance(balance["info"], dict) and
                "result" in balance["info"] and
                "list" in balance["info"]["result"] and
                isinstance(balance["info"]["result"]["list"], list) and
                len(balance["info"]["result"]["list"]) > 0
            ):
                coins = balance["info"]["result"]["list"][0].get("coin", [])
                usdt_coin = next((c for c in coins if c.get("coin") == "USDT"), None)
                if usdt_coin:
                    equity = float(usdt_coin.get("equity", 0) or 0)
                    total_position_im = float(usdt_coin.get("totalPositionIM", 0) or 0)
                    det["available"] = max(equity - total_position_im, 0.0)
                    det["used"] = total_position_im
                    det["total_wallet"] = float(usdt_coin.get("walletBalance", 0) or 0)
                    det["usdt_balance"] = equity

            # Fallback: vecchia logica se non Bybit Unified
            if det["total_wallet"] == 0 and "total" in balance and "USDT" in balance["total"]:
                det["total_wallet"] = float(balance["total"]["USDT"] or 0)
            if det["available"] == 0 and "free" in balance and "USDT" in balance["free"]:
                det["available"] = float(balance["free"]["USDT"] or 0)
            if det["used"] == 0 and "used" in balance and "USDT" in balance["used"]:
                det["used"] = float(balance["used"]["USDT"] or 0)

            # Calcolo del totale USDT
            unified = float(balance["info"].get("totalEquity", 0) or 0) if "info" in balance else 0.0
            spot = float(balance.get("total", {}).get("USDT", 0) or 0)
            control = sum(
                float(acc.get("total", {}).get("USDT", 0) or 0)
                for acc in balance.get("accounts", [])
            )
            total_usdt = unified + spot + control
            
            # Se il totale è 0, prova a ottenere il bilancio reale
            if total_usdt == 0:
                try:
                    total_usdt = await get_real_balance(app_state.async_exchange)
                except Exception as e:
                    logger.error(f"Errore nel recupero del bilancio reale: {e}")
                    total_usdt = 0.0
            det["usdt_balance"] = total_usdt if det["usdt_balance"] == 0 else det["usdt_balance"]

            # Calcolo del PnL
            if det["pnl"] == 0:
                try:
                    positions = await app_state.async_exchange.fetch_positions(None, {
                        "limit": 100,
                        "category": "linear",
                        "settleCoin": "USDT",
                    })
                    det["pnl"] = sum(
                        float(p.get("unrealizedPnl", 0) or 0)
                        for p in positions
                        if float(p.get("contracts", 0) or 0) > 0
                    )
                except Exception as e:
                    logger.error(f"Errore nel recupero delle posizioni: {e}")
                    det["pnl"] = 0.0

            # Calcolo manuale se ancora non disponibili
            if det["available"] == 0.0 or det["used"] == 0.0:
                try:
                    positions = await app_state.async_exchange.fetch_positions(None, {
                        "limit": 100,
                        "category": "linear",
                        "settleCoin": "USDT",
                    })
                    used = sum(float(p.get("initialMargin", 0) or 0) for p in positions if float(p.get("contracts", 0) or 0) > 0)
                    det["used"] = used
                    det["available"] = max(det["total_wallet"] - used, 0.0)
                except Exception as e:
                    logger.error(f"Errore nel calcolo manuale di available/used: {e}")
        except Exception as e:
            logger.error(f"Errore nell'elaborazione dei dati del bilancio: {e}")

        return det

    except Exception as exc:
        logger.error(f"Errore recupero balance: {exc}")
        return {
            "usdt_balance": 0.0,
            "total_wallet": 0.0,
            "available": 0.0,
            "used": 0.0,
            "pnl": 0.0,
            "error": str(exc)
        }

# ───────────────────────────────────────────────────────────────────────────────
# POSITIONS
# ───────────────────────────────────────────────────────────────────────────────
@root.get("/positions")
@api.get("/positions")
async def get_positions(state=Depends(get_state), _=Depends(verify_auth_token)):
    try:
        if not app_state.async_exchange:
            app_state.async_exchange = ccxt_async.bybit(exchange_config)
            await app_state.async_exchange.load_markets()
        count = await get_open_positions(app_state.async_exchange)
        return {"open_positions": count}
    except Exception as exc:
        raise HTTPException(500, str(exc))

# ───────────────────────────────────────────────────────────────────────────────
# ORDERS OPEN
# ───────────────────────────────────────────────────────────────────────────────
@root.get("/orders/open")
@api.get("/orders/open")
async def get_open_orders(state=Depends(get_state), _=Depends(verify_auth_token)):
    try:
        if not app_state.async_exchange:
            app_state.async_exchange = ccxt_async.bybit(exchange_config)
            await app_state.async_exchange.load_markets()

        open_orders = await app_state.async_exchange.fetch_open_orders()
        positions   = await app_state.async_exchange.fetch_positions(None, {
            "limit":      100,
            "category":   "linear",
            "settleCoin": "USDT",
        })

        active_positions: Dict[str, Any] = {
            f"{p['symbol']}_{p['side']}": p
            for p in positions
            if float(p.get("contracts", 0)) > 0
        }

        stop_orders: Dict[str, dict] = {}
        related_orders: Dict[str, List[dict]] = {}

        for order in open_orders:
            sym   = order["symbol"]
            side  = order.get("side", "").lower()
            info  = order.get("info", {})
            rel   = False

            if (info.get("stopOrderType") == "Stop"
                    or "stopPrice" in info
                    or info.get("reduceOnly")
                    or info.get("closeOnTrigger")):
                rel = True

            opposite = "long" if side == "sell" else "short"
            if f"{sym}_{opposite}" in active_positions:
                rel = True

            if rel:
                related_orders.setdefault(sym, []).append(order)
                if (info.get("stopOrderType") == "Stop"
                        or "stopPrice" in info):
                    stop_orders[sym] = order

        filtered_orders = [
            o for o in open_orders
            if o["symbol"] not in related_orders or o not in related_orders[o["symbol"]]
        ]

        enriched_positions: List[dict] = []
        for p in positions:
            if float(p.get("contracts", 0)) == 0:
                continue
            sym        = p["symbol"]
            side       = p["side"]
            entry      = float(p["entryPrice"])
            amount     = float(p["contracts"])
            stop_price = (stop_orders.get(sym) or {}).get("stopPrice") or p.get("stopLossPrice")
            stop_val   = float(stop_price) if stop_price else None
            potential  = ((stop_val - entry) * amount if (stop_val and side == "long")
                          else ((entry - stop_val) * amount if stop_val else None))

            enriched_positions.append({
                "symbol":    sym,
                "type":      "position",
                "side":      "Buy" if side == "long" else "Sell",
                "amount":    amount,
                "price":     entry,
                "status":    "open",
                "pnl":       float(p.get("unrealizedPnl", 0)),
                "margin":    float(p.get("initialMargin", MARGIN_USDT)),
                "leverage":  float(p.get("leverage", LEVERAGE)),
                "stop_loss": stop_val,
                "sl_profit": potential,
            })

        return filtered_orders + enriched_positions

    except Exception as exc:
        logger.error("Errore open orders: %s", exc, exc_info=True)
        raise HTTPException(500, f"Errore recupero open orders: {exc}")

# ───────────────────────────────────────────────────────────────────────────────
# ORDERS CLOSED
# ───────────────────────────────────────────────────────────────────────────────
@root.get("/orders/closed")
@api.get("/orders/closed")
async def get_closed_orders(state=Depends(get_state), _=Depends(verify_auth_token)):
    try:
        if not app_state.async_exchange:
            app_state.async_exchange = ccxt_async.bybit(exchange_config)
            await app_state.async_exchange.load_markets()
        closed_orders = await app_state.async_exchange.fetch_closed_orders()
        return closed_orders
    except Exception as exc:
        raise HTTPException(500, f"Errore recupero closed orders: {exc}")

# ───────────────────────────────────────────────────────────────────────────────
# TRADES
# ───────────────────────────────────────────────────────────────────────────────
@root.get("/trades")
@api.get("/trades")
async def get_trades(state=Depends(get_state), _=Depends(verify_auth_token)):
    try:
        if not app_state.async_exchange:
            app_state.async_exchange = ccxt_async.bybit(exchange_config)
            await app_state.async_exchange.load_markets()
        trades = await app_state.async_exchange.fetch_my_trades(limit=50)
        return [{
            "symbol": t["symbol"],
            "type": t["type"],
            "side": t["side"],
            "amount": t["amount"],
            "price": t["price"],
            "realized_pnl": t.get("info", {}).get("closedPnl", 0),
            "timestamp": t["timestamp"],
        } for t in trades]
    except Exception as exc:
        raise HTTPException(500, f"Errore recupero trades: {exc}")

# ───────────────────────────────────────────────────────────────────────────────
# CHART DATA
# ───────────────────────────────────────────────────────────────────────────────
@root.get("/chart-data/{symbol:path}")
@api.get("/chart-data/{symbol:path}")
async def get_chart_data(
    symbol: str,
    timeframe: str = "15m",
    limit: int = 100,
    state = Depends(get_state),
    _ = Depends(verify_auth_token),
):
    try:
        if not app_state.async_exchange:
            app_state.async_exchange = ccxt_async.bybit(exchange_config)
            await app_state.async_exchange.load_markets()

        # Normalizza il simbolo
        symbol = symbol.replace(":", "/").upper()
        if symbol.endswith("/USDT/USDT"):
            symbol = symbol[:-5]

        # Carica i mercati
        markets = await app_state.async_exchange.load_markets()

        # Prova a normalizzare per Bybit (es. 10000SATSUSDT)
        if symbol not in markets:
            alt_symbol = symbol.replace("/", "")
            if alt_symbol in markets:
                symbol = alt_symbol

        # Se ancora non trovato, mostra solo il messaggio senza esempi
        if symbol not in markets:
            logger.error(f"Simbolo non trovato nel mercato: {symbol}.")
            return {
                "timestamps": [], "labels": [], "open": [],
                "high": [], "low": [], "close": [],
                "volumes": [], "error": f"Simbolo {symbol} non trovato nel mercato."
            }

        # Verifica che il timeframe sia valido
        valid_timeframes = ["5m", "15m", "30m", "1h", "4h"]
        if timeframe not in valid_timeframes:
            logger.error(f"Timeframe non valido: {timeframe}")
            return {
                "timestamps": [], "labels": [], "open": [],
                "high": [], "low": [], "close": [],
                "volumes": [], "error": "Timeframe non valido"
            }

        if limit > 1000:
            limit = 1000

        try:
            ohlcv = await app_state.async_exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        except Exception as e:
            logger.error(f"Errore nel recupero dei dati OHLCV per {symbol}: {e}")
            return {
                "timestamps": [], "labels": [], "open": [],
                "high": [], "low": [], "close": [],
                "volumes": [], "error": f"Errore nel recupero dei dati: {str(e)}"
            }

        out = {
            "timestamps": [], "labels": [], "open": [],
            "high": [], "low": [], "close": [],
            "volumes": [],
        }
        for ts, o, h, l, c, v in ohlcv:
            dt = datetime.fromtimestamp(ts / 1000)
            out["timestamps"].append(ts)
            out["labels"].append(dt.strftime("%H:%M %d/%m"))
            out["open"].append(o)
            out["high"].append(h)
            out["low"].append(l)
            out["close"].append(c)
            out["volumes"].append(v)
        return out
    except Exception as exc:
        logger.error(f"Errore nel recupero dati grafico: {exc}")
        return {
            "timestamps": [], "labels": [], "open": [],
            "high": [], "low": [], "close": [],
            "volumes": [], "error": f"Errore interno: {str(exc)}"
        }

# ───────────────────────────────────────────────────────────────────────────────
# CANCEL ORDER
# ───────────────────────────────────────────────────────────────────────────────
@root.post("/cancel-order")
@api.post("/cancel-order")
async def cancel_order(
    order_id: str = Body(..., embed=True, alias="order_id"),
    state = Depends(get_state),
    _ = Depends(verify_auth_token),
):
    try:
        if not app_state.async_exchange:
            app_state.async_exchange = ccxt_async.bybit(exchange_config)
            await app_state.async_exchange.load_markets()

        open_orders = await app_state.async_exchange.fetch_open_orders()
        target = next((o for o in open_orders if str(o.get("id")) == str(order_id)), None)
        if not target:
            raise HTTPException(404, f"Ordine {order_id} non trovato o già chiuso.")

        symbol = target.get("symbol")
        await app_state.async_exchange.cancel_order(order_id, symbol)
        return {"status": "success", "message": f"Ordine {order_id} cancellato."}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Errore annullamento ordine: {exc}")

# ───────────────────────────────────────────────────────────────────────────────
# SAVE / GET API-KEYS
# ───────────────────────────────────────────────────────────────────────────────
@root.post("/set-keys")
@api.post("/set-keys")
def set_keys(keys: ApiKeys, _=Depends(verify_auth_token)):
    try:
        save_api_keys_secure(keys)
        return {"status": "Chiavi salvate."}
    except Exception as exc:
        raise HTTPException(500, f"Errore salvataggio chiavi: {exc}")

@root.get("/get-keys")
@api.get("/get-keys")
def get_keys(_=Depends(verify_auth_token)):
    try:
        keys = load_api_keys_secure()
        return keys or {"status": "Chiavi non trovate."}
    except Exception as exc:
        raise HTTPException(500, f"Errore lettura chiavi: {exc}")

# ───────────────────────────────────────────────────────────────────────────────
# TRAIN-MODEL (via Celery)
# ───────────────────────────────────────────────────────────────────────────────
@root.post("/train-model")
@api.post("/train-model")
async def train_model(request: Request, _=Depends(verify_auth_token)):
    try:
        data       = await request.json()
        model_type = data.get("model_type")
        timeframe  = data.get("timeframe")
        limit_days = data.get("data_limit_days", 30)
        top_train  = data.get("top_train_crypto")

        from tasks import train_model_task
        task = train_model_task.delay(model_type, timeframe, limit_days, top_train)
        return {"message": f"Training avviato ({task.id}).", "task_id": task.id}
    except Exception as exc:
        raise HTTPException(500, f"Errore avvio training: {exc}")

@root.get("/training-status/{task_id}")
@api.get("/training-status/{task_id}")
async def training_status(task_id: str, _=Depends(verify_auth_token)):
    from tasks import TASK_REGISTRY
    
    task_info = TASK_REGISTRY.get(task_id)
    if not task_info:
        return {"status": "pending"}
    
    if task_info["state"] == "FAILURE":
        raise HTTPException(500, str(task_info["result"]))
    
    # Usa il progresso e lo step corrente dal registro dei task
    if task_info["state"] == "RUNNING":
        return {
            "status": "running", 
            "progress": task_info.get("progress", 0),
            "current_step": task_info.get("current_step", "In esecuzione")
        }
    
    # Per i task completati
    return {
        "status": task_info["state"].lower(), 
        "result": task_info["result"],
        "progress": 100,
        "current_step": "Completato"
    }

# ───────────────────────────────────────────────────────────────────────────────
# EXECUTE-TRADE
# ───────────────────────────────────────────────────────────────────────────────
@root.post("/execute-trade")
@api.post("/execute-trade")
async def execute_trade(
    request: Request,
    state = Depends(get_state),
    _ = Depends(verify_auth_token),
):
    try:
        data = await request.json()
        symbol = data.get("symbol")
        side = data.get("side")
        leverage = data.get("leverage", LEVERAGE)
        margin = data.get("margin", MARGIN_USDT)

        if not symbol or side not in {"Buy", "Sell"}:
            raise HTTPException(400, "Parametri 'symbol' o 'side' errati.")

        if not app_state.async_exchange:
            app_state.async_exchange = ccxt_async.bybit(exchange_config)
            await app_state.async_exchange.load_markets()

        usdt_balance = await get_real_balance(app_state.async_exchange)
        if usdt_balance is None or usdt_balance < 10:
            raise HTTPException(400, "Saldo USDT insufficiente.")

        markets = await fetch_markets(app_state.async_exchange)
        min_amounts = await fetch_min_amounts(app_state.async_exchange, [symbol], markets)

        try:
            await app_state.async_exchange.set_leverage(leverage, symbol)
        except Exception as exc:
            logger.warning("Impossibile impostare leva: %s", exc)

        ticker = await app_state.async_exchange.fetch_ticker(symbol)
        price = ticker.get("last")
        notional = margin * leverage
        size_raw = notional / price if price else None
        size_prec = (app_state.async_exchange.amount_to_precision(symbol, size_raw)
                    if size_raw else None)
        size = float(size_prec) if size_prec else None

        if size is None or size < min_amounts.get(symbol, 0.1):
            raise HTTPException(400, "Dimensione posizione inferiore al minimo consentito.")

        order = (
            await app_state.async_exchange.create_market_buy_order(symbol, size)
            if side == "Buy"
            else await app_state.async_exchange.create_market_sell_order(symbol, size)
        )

        entry    = order.get("average") or price
        trade_id = order.get("id") or f"{symbol}-{datetime.utcnow().timestamp()}"

        return {
            "status":        "success",
            "message":       f"Ordine {side} eseguito per {symbol} a {entry}",
            "order_id":      trade_id,
            "entry_price":   entry,
            "position_size": size,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Errore execute-trade: %s", exc, exc_info=True)
        raise HTTPException(500, f"Errore execute-trade: {exc}")

# ───────────────────────────────────────────────────────────────────────────────
# CALCOLA SIZE POSIZIONE
# ───────────────────────────────────────────────────────────────────────────────
async def calculate_position_size(
    exchange,
    symbol: str,
    usdt_balance: float,
    min_amount: float = 0,
    margin: Optional[float] = None,
) -> Optional[float]:
    try:
        ticker = await exchange.fetch_ticker(symbol)
        price  = ticker.get("last")
        if price is None:
            return None
        margin_val = margin or MARGIN_USDT
        leverage   = LEVERAGE
        notional   = margin_val * leverage
        size_raw   = notional / price
        size_prec  = exchange.amount_to_precision(symbol, size_raw)
        return float(size_prec)
    except Exception as exc:
        logger.error("Errore calcolo size: %s", exc)
        return None

# ───────────────────────────────────────────────────────────────────────────────
# MODELS MANAGEMENT
# ───────────────────────────────────────────────────────────────────────────────
@root.get("/list-models")
@api.get("/list-models")
async def list_models(_=Depends(verify_auth_token)):
    try:
        tgt = Path(__file__).parent / "trained_models"
        if not tgt.exists():
            return {"models": []}
        return {"models": [f.name for f in tgt.iterdir() if f.suffix in (".h5", ".pkl")]}
    except Exception as exc:
        raise HTTPException(500, f"Errore lista modelli: {exc}")

@root.get("/check-model-exists/{model_file}")
@api.get("/check-model-exists/{model_file}")
async def check_model(model_file: str, _=Depends(verify_auth_token)):
    try:
        path = Path(__file__).parent / "trained_models" / model_file
        exists = path.exists() and model_file.endswith((".h5", ".pkl"))
        return {"exists": exists}
    except Exception as exc:
        raise HTTPException(500, f"Errore verifica modello: {exc}")

# ───────────────────────────────────────────────────────────────────────────────
# AUTH TOKEN
# ───────────────────────────────────────────────────────────────────────────────
@root.post("/auth/token")
@api.post("/auth/token")
async def get_auth_token(keys: Optional[ApiKeys] = None):
    try:
        logger.info("Richiesta token con credenziali dal file .env")
        
        if not API_KEY or not API_SECRET:
            logger.error("Credenziali API mancanti nel file .env")
            raise HTTPException(500, "Credenziali API mancanti nel file .env")
        
        token = generate_auth_token(API_KEY, API_SECRET)
        logger.info("Token generato con successo")
        return {"token": token}
    except Exception as exc:
        logger.error(f"Errore generazione token: {exc}", exc_info=True)
        raise HTTPException(500, f"Errore generazione token: {exc}")

# ───────────────────────────────────────────────────────────────────────────────
# Include router alias /api e avvio manuale
# ───────────────────────────────────────────────────────────────────────────────
app.include_router(root)
app.include_router(api)

if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
