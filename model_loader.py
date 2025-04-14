import os
import logging
from termcolor import colored
import numpy as np
import joblib
from keras.models import load_model

# Import delle funzioni per ricavare i percorsi dei file di modelli e scaler
from config import (
    get_lstm_model_file, get_lstm_scaler_file,
    get_rf_model_file, get_rf_scaler_file,
    get_xgb_model_file, get_xgb_scaler_file
)

# Import della custom loss (assicurati che FocalLoss sia definita in models.py)
from models import FocalLoss

def load_lstm_model_func(timeframe):
    """Load LSTM model and scaler for a specific timeframe, handling custom_objects."""
    try:
        model_file = get_lstm_model_file(timeframe)
        scaler_file = get_lstm_scaler_file(timeframe)
        
        if os.path.exists(model_file) and os.path.exists(scaler_file):
            # Carica il modello specificando la custom loss FocalLoss
            model = load_model(model_file, custom_objects={'FocalLoss': FocalLoss})
            scaler = joblib.load(scaler_file)
            logging.info(f"{colored('LSTM model loaded for', 'green')} {timeframe}")
            return model, scaler
        else:
            logging.warning(f"{colored('LSTM model or scaler not found for', 'yellow')} {timeframe}")
            return None, None
    except Exception as e:
        logging.error(f"{colored('Errore nel caricamento del modello LSTM per', 'red')} {timeframe}: {e}")
        return None, None

def load_random_forest_model_func(timeframe):
    """Load Random Forest model and scaler for a specific timeframe."""
    try:
        model_file = get_rf_model_file(timeframe)
        scaler_file = get_rf_scaler_file(timeframe)
        
        if os.path.exists(model_file) and os.path.exists(scaler_file):
            model = joblib.load(model_file)
            scaler = joblib.load(scaler_file)
            logging.info(f"{colored('Random Forest model loaded for', 'green')} {timeframe}")
            return model, scaler
        else:
            logging.warning(f"{colored('Random Forest model or scaler not found for', 'yellow')} {timeframe}")
            return None, None
    except Exception as e:
        logging.error(f"{colored('Errore nel caricamento del modello Random Forest per', 'red')} {timeframe}: {e}")
        return None, None

def load_xgboost_model_func(timeframe):
    """Load XGBoost model and scaler for a specific timeframe."""
    try:
        model_file = get_xgb_model_file(timeframe)
        scaler_file = get_xgb_scaler_file(timeframe)
        
        if os.path.exists(model_file) and os.path.exists(scaler_file):
            model = joblib.load(model_file)
            scaler = joblib.load(scaler_file)
            logging.info(f"{colored('XGBoost model loaded for', 'green')} {timeframe}")
            return model, scaler
        else:
            logging.warning(f"{colored('XGBoost model or scaler not found for', 'yellow')} {timeframe}")
            return None, None
    except Exception as e:
        logging.error(f"{colored('Errore nel caricamento del modello XGBoost per', 'red')} {timeframe}: {e}")
        return None, None