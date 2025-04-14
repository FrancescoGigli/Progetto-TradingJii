# config.py
# ------------------------------
import os
from dotenv import load_dotenv

# Carica le variabili dal file .env
load_dotenv()

# Ottieni le chiavi API dalle variabili d'ambiente
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

# Exchange configuration
exchange_config = {
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
    'options': {
        'adjustForTimeDifference': True,  # Add this line to handle time synchronization
        'recvWindow': 60000  # Increase the receive window to accommodate time differences
    }
}

MARGIN_USDT = 40.0
LEVERAGE = 10

ENABLED_TIMEFRAMES = ["15m", "30m", "1h"]
SELECTED_MODELS = ["lstm", "rf", "xgb"]
TIMEFRAME_DEFAULT = "15m"

TIME_STEPS = 10

MODEL_RATES = {
    'lstm': 0.6,
    'rf': 0.2,
    'xgb': 0.2
}

NEUTRAL_LOWER_THRESHOLD = 0.40
NEUTRAL_UPPER_THRESHOLD = 0.60
COLOR_THRESHOLD_GREEN   = 0.65
COLOR_THRESHOLD_RED     = 0.35
RSI_BUY_THRESHOLD       = 30  # Updated to match RSI_THRESHOLDS
RSI_SELL_THRESHOLD      = 70  # Updated to match RSI_THRESHOLDS

TRADE_CYCLE_INTERVAL    = 300
DATA_LIMIT_DAYS = 50

# File paths: ora definiti come funzioni, in modo da essere sempre calcolati in base al timeframe attuale
def _file_name(base, tf):
    return f"{base}_{tf}"

def get_lstm_model_file(timeframe):
    """Return the path to the LSTM model file for the given timeframe."""
    return f"trained_models/lstm_model_{timeframe}.h5"

def get_lstm_scaler_file(tf):
    return f"trained_models/lstm_scaler_{tf}.pkl"

def get_rf_model_file(tf):
    return f"trained_models/rf_model_{tf}.pkl"  # Updated to match trainer.py

def get_rf_scaler_file(tf):
    return f"trained_models/rf_scaler_{tf}.pkl"  # Updated to match trainer.py

def get_xgb_model_file(tf):
    return f"trained_models/xgb_model_{tf}.pkl"  # Updated to match trainer.py

def get_xgb_scaler_file(tf):
    return f"trained_models/xgb_scaler_{tf}.pkl"  # Updated to match trainer.py

EXCLUDED_SYMBOLS = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']
TOP_TRAIN_CRYPTO = 50          # Numero totale di criptovalute da considerare per il training
TOP_ANALYSIS_CRYPTO = 150      # Numero totale di criptovalute da considerare per l'analisi

# Numero di simboli da processare per ciclo
SYMBOLS_PER_ANALYSIS_CYCLE = 60  # Numero di simboli da analizzare in ciascun ciclo di trading
SYMBOLS_FOR_VALIDATION = 10     # Numero di simboli da validare prima del training

EXPECTED_COLUMNS = [
    'open', 'high', 'low', 'close', 'volume',
    'ema5', 'ema10', 'ema20',
    'macd', 'macd_signal', 'macd_histogram',
    'rsi_fast', 'stoch_rsi',
    'atr',
    'bollinger_hband', 'bollinger_lband', 'bollinger_pband',
    'vwap',
    'adx',
    'roc', 'log_return',
    'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span',
    'williams_r', 'obv',
    'sma_fast', 'sma_slow', 'sma_fast_trend', 'sma_slow_trend', 'sma_cross',
    'close_lag_1', 'volume_lag_1',
    'weekday_sin', 'weekday_cos', 'hour_sin', 'hour_cos',
    'mfi', 'cci'
]

RSI_THRESHOLDS = {
    'sideways': {
         'oversold': 30,
         'overbought': 70
    }
}

THRESHOLDS = {
    -60: [-0.50, 30],
    -40: [-0.38, 30],
    15: [0.11, 30],
    21: [0.19, 60],
    35: [0.28, 60],
    50: [0.46, 150],
    70: [0.64, 150],
    85: [0.78, 150],
    100: [0.85, 150],
    150: [1.2, 500],
    180: [1.5, 600],
    200: [1.8, 120],
    250: [2.3, 120]
}

TRAIN_IF_NOT_FOUND = True