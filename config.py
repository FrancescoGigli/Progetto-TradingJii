# config.py

from typing import Dict, List, Optional
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
from state import app_state

# Carica le variabili d'ambiente
load_dotenv()

# Ottieni le API key dalle variabili d'ambiente
API_KEY = os.getenv("API_KEY", "")
API_SECRET = os.getenv("API_SECRET", "")

class ExchangeConfig(BaseModel):
    api_key: str = API_KEY
    api_secret: str = API_SECRET
    enable_rate_limit: bool = True
    options: Dict = {
        "adjustForTimeDifference": True,
        "recvWindow": 60000
    }

class ModelConfig(BaseModel):
    model_config = {"protected_namespaces": ()}
    time_steps: int = app_state.time_steps
    expected_columns: List[str] = app_state.expected_columns
    model_rates: Dict[str, float] = app_state.model_rates

class TradingConfig(BaseModel):
    excluded_symbols: List[str] = app_state.excluded_symbols
    top_train_crypto: int = app_state.top_train_crypto
    top_analysis_crypto: int = app_state.top_analysis_crypto
    trade_cycle_interval: int = app_state.trade_cycle_interval
    train_if_not_found: bool = app_state.train_if_not_found
    enabled_timeframes: List[str] = app_state.enabled_timeframes
    timeframe_default: str = app_state.timeframe_default
    leverage: int = app_state.leverage
    margin_usdt: float = app_state.margin_usdt

class Config(BaseModel):
    exchange: ExchangeConfig = ExchangeConfig()
    model: ModelConfig = ModelConfig()
    trading: TradingConfig = TradingConfig()

# Configurazione di default
config = Config()

# Esporta le configurazioni
EXCHANGE_CONFIG = config.exchange.dict()
MODEL_CONFIG = config.model.dict()
TRADING_CONFIG = config.trading.dict()

# Costanti
TIME_STEPS = MODEL_CONFIG["time_steps"]
EXPECTED_COLUMNS = MODEL_CONFIG["expected_columns"]
MODEL_RATES = MODEL_CONFIG["model_rates"]
EXCLUDED_SYMBOLS = TRADING_CONFIG["excluded_symbols"]
TOP_TRAIN_CRYPTO = TRADING_CONFIG["top_train_crypto"]
TOP_ANALYSIS_CRYPTO = TRADING_CONFIG["top_analysis_crypto"]
TRADE_CYCLE_INTERVAL = TRADING_CONFIG["trade_cycle_interval"]
TRAIN_IF_NOT_FOUND = TRADING_CONFIG["train_if_not_found"]
ENABLED_TIMEFRAMES = TRADING_CONFIG["enabled_timeframes"]
TIMEFRAME_DEFAULT = TRADING_CONFIG["timeframe_default"]
LEVERAGE = TRADING_CONFIG["leverage"]
MARGIN_USDT = TRADING_CONFIG["margin_usdt"]

# Configurazione dell'exchange per l'uso diretto
# ccxt richiede le chiavi "apiKey" e "secret" (camelCase).
# Manteniamo EXCHANGE_CONFIG (snake_case) per eventuali altri usi interni,
# ma per l'inizializzazione dell'exchange usiamo un dict con le chiavi corrette.
exchange_config = {
    "apiKey":  API_KEY,
    "secret":  API_SECRET,
    "enableRateLimit": True,
    "options": {
        "adjustForTimeDifference": True,
        "recvWindow": 60000,
    },
}

# Funzioni helper per i percorsi dei modelli
def get_model_path(timeframe: str, model_type: str) -> str:
    base_dir = os.path.join(os.path.dirname(__file__), "trained_models")
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, f"{model_type}_{timeframe}.h5")

def get_scaler_path(timeframe: str, model_type: str) -> str:
    base_dir = os.path.join(os.path.dirname(__file__), "trained_models")
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, f"{model_type}_scaler_{timeframe}.pkl")

# ==== funzioni percorso modelli ==============================================
def _mp(model, tf, ext):            # helper interno
    return f"trained_models/{model}_model_{tf}{ext}"

def get_lstm_model_file(tf):
    return _mp("lstm", tf, ".h5")
def get_lstm_scaler_file(tf):
    return _mp("lstm", tf, "_scaler.pkl")
def get_rf_model_file(tf):
    return _mp("rf", tf, ".pkl")
def get_rf_scaler_file(tf):
    return _mp("rf", tf, "_scaler.pkl")
def get_xgb_model_file(tf):
    return _mp("xgb", tf, ".json")
def get_xgb_scaler_file(tf):
    return _mp("xgb", tf, "_scaler.pkl")

# ==== trading params di default ==============================================
NEUTRAL_LOWER_THRESHOLD = 0.40
NEUTRAL_UPPER_THRESHOLD = 0.60
COLOR_THRESHOLD_GREEN   = 0.65
COLOR_THRESHOLD_RED     = 0.35
RSI_BUY_THRESHOLD       = 30
RSI_SELL_THRESHOLD      = 70

DATA_LIMIT_DAYS = 50

RSI_THRESHOLDS = {
    "sideways": {"oversold": 30, "overbought": 70}
}

THRESHOLDS = {
    -60: [-0.50, 30],
    -40: [-0.38, 30],
    15:  [0.11, 30],
    21:  [0.19, 60],
    35:  [0.28, 60],
    50:  [0.46, 150],
    70:  [0.64, 150],
    85:  [0.78, 150],
    100: [0.85, 150],
    150: [1.20, 500],
    180: [1.50, 600],
    200: [1.80, 120],
    250: [2.30, 120],
} 