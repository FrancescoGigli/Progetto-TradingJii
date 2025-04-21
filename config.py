# config.py
# ------------------------------
# Le API key sono ora gestite tramite il file .env

# Exchange configuration
# exchange_config è ora definito in main.py

import os
from dotenv import load_dotenv

# Carica le variabili d'ambiente
load_dotenv()

# Configurazione API
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

# Configurazione dell'exchange
exchange_config = {
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
    'options': {
        'adjustForTimeDifference': True,
        'recvWindow': 60000
    }
}

# Configurazione dei timeframes
ENABLED_TIMEFRAMES = ['5m', '15m', '30m', '1h', '4h']
TIMEFRAME_DEFAULT = '15m'

# Configurazione dei modelli
TIME_STEPS = 60  # Numero di candele per la predizione
TRADE_CYCLE_INTERVAL = 300  # Intervallo tra i cicli di trading in secondi

# Pesi dei modelli (la somma deve essere 1)
MODEL_RATES = {
    'lstm': 0.4,
    'rf': 0.3,
    'xgb': 0.3
}

# Configurazione del training
TOP_TRAIN_CRYPTO = 100  # Numero di criptovalute per il training
TOP_ANALYSIS_CRYPTO = 150  # Numero di criptovalute per l'analisi
TRAIN_IF_NOT_FOUND = True  # Se True, allena i modelli se non trovati

# Simboli da escludere
EXCLUDED_SYMBOLS = [
    'USDC', 'BUSD', 'USDD', 'TUSD', 'USDN',  # Stablecoins
    'DOWN', 'UP', 'BULL', 'BEAR',  # Tokens leveraged
    'EUR', 'GBP', 'AUD', 'BRL',  # Fiat pairs
]

# Colonne attese nei dati
EXPECTED_COLUMNS = [
    'timestamp', 'open', 'high', 'low', 'close', 'volume'
]

# Funzioni per ottenere i percorsi dei file dei modelli
def get_model_path(model_type, timeframe):
    return f'trained_models/{model_type}_model_{timeframe}'

def get_lstm_model_file(timeframe):
    return get_model_path('lstm', timeframe) + '.h5'

def get_lstm_scaler_file(timeframe):
    return get_model_path('lstm', timeframe) + '_scaler.pkl'

def get_rf_model_file(timeframe):
    return get_model_path('rf', timeframe) + '.pkl'

def get_rf_scaler_file(timeframe):
    return get_model_path('rf', timeframe) + '_scaler.pkl'

def get_xgb_model_file(timeframe):
    return get_model_path('xgb', timeframe) + '.pkl'

def get_xgb_scaler_file(timeframe):
    return get_model_path('xgb', timeframe) + '_scaler.pkl'

MARGIN_USDT = 40.0
LEVERAGE = 10

TIME_STEPS = 10

NEUTRAL_LOWER_THRESHOLD = 0.40
NEUTRAL_UPPER_THRESHOLD = 0.60
COLOR_THRESHOLD_GREEN   = 0.65
COLOR_THRESHOLD_RED     = 0.35
RSI_BUY_THRESHOLD       = 30  # Updated to match RSI_THRESHOLDS
RSI_SELL_THRESHOLD      = 70  # Updated to match RSI_THRESHOLDS

DATA_LIMIT_DAYS = 50

# File paths: ora definiti come funzioni, in modo da essere sempre calcolati in base al timeframe attuale
def _file_name(base, tf):
    return f"{base}_{tf}"

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