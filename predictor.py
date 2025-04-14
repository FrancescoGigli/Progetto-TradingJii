import logging
from termcolor import colored
import numpy as np
import config
from config import (
    EXPECTED_COLUMNS,
    NEUTRAL_UPPER_THRESHOLD,
    NEUTRAL_LOWER_THRESHOLD,
    RSI_THRESHOLDS,
    COLOR_THRESHOLD_GREEN,
    COLOR_THRESHOLD_RED
)
from data_utils import prepare_data

def get_color_normal(value):
    if value > COLOR_THRESHOLD_GREEN:
        return "green"
    elif value < COLOR_THRESHOLD_RED:
        return "red"
    else:
        return "yellow"

def get_color_rsi(rsi_value):
    if rsi_value < RSI_THRESHOLDS['sideways']['oversold']:
        return "green"
    elif rsi_value > RSI_THRESHOLDS['sideways']['overbought']:
        return "red"
    else:
        return "yellow"

def predict_signal_for_model(df, model, scaler, symbol, time_steps, expected_features=None, timeframe=""):
    if expected_features is None:
        expected_features = len(EXPECTED_COLUMNS)
    if model is None or scaler is None:
        logging.error(f"{symbol} [{timeframe}]: modello o scaler non disponibili.")
        return None
    try:
        # Se il numero di colonne non corrisponde, rielabora i dati
        if df.shape[1] != expected_features:
            data = prepare_data(df)
        else:
            data = df.values
            # Assicurati che non ci siano NaN o infiniti nei dati
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        if len(data) < time_steps:
            logging.error(f"{symbol} [{timeframe}]: dati insufficienti ({len(data)} elementi, richiesti {time_steps}).")
            return None

        X = data[-time_steps:]
        # Verifica che non ci siano NaN o infiniti prima dello scaling
        if np.isnan(X).any() or np.isinf(X).any():
            logging.warning(f"{symbol} [{timeframe}]: valori NaN o infiniti trovati nei dati, sostituiti con 0.0")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
        X_scaled = scaler.transform(X)
        if X_scaled.shape[1] != expected_features:
            logging.error(f"{symbol} [{timeframe}]: numero di feature non corretto dopo scaling ({X_scaled.shape[1]} invece di {expected_features}).")
            return None
        X_scaled = X_scaled.reshape((1, time_steps, expected_features))
        pred = model.predict(X_scaled)[0][0]
        return pred
    except Exception as e:
        logging.error(f"Errore in prediction per {symbol} [{timeframe}]: {e}")
        return None

def predict_signal_ensemble(dataframes,
                            lstm_models, lstm_scalers,
                            rf_models, rf_scalers,
                            xgb_models, xgb_scalers,
                            symbol, time_steps,
                            weight_lstm, weight_rf, weight_xgb):
    lstm_preds = {}
    rf_preds = {}
    xgb_preds = {}
    
    for tf, df in dataframes.items():
        # Predizione LSTM
        lstm_pred = predict_signal_for_model(df, lstm_models.get(tf), lstm_scalers.get(tf), symbol, time_steps, timeframe=tf)
        if lstm_pred is None:
            logging.error(f"{symbol} [{tf}]: predizione LSTM fallita.")
            return None, None, None
        # Predizione RF
        try:
            data_rf = prepare_data(df)
            if len(data_rf) < time_steps:
                logging.error(f"{symbol} [{tf}]: dati insufficienti per RF ({len(data_rf)} elementi, richiesti {time_steps}).")
                return None, None, None
                
            # Assicurati che non ci siano NaN o infiniti nei dati
            if np.isnan(data_rf).any() or np.isinf(data_rf).any():
                logging.warning(f"{symbol} [{tf}]: valori NaN o infiniti trovati nei dati RF, sostituiti con 0.0")
                data_rf = np.nan_to_num(data_rf, nan=0.0, posinf=0.0, neginf=0.0)
                
            X_rf = data_rf[-time_steps:].flatten().reshape(1, -1)
            X_rf_scaled = rf_scalers.get(tf).transform(X_rf)
            expected_rf_features = time_steps * len(EXPECTED_COLUMNS)
            if X_rf_scaled.shape[1] != expected_rf_features:
                logging.error(f"{symbol} [{tf}]: numero di feature RF non corretto ({X_rf_scaled.shape[1]} invece di {expected_rf_features}).")
                return None, None, None
            rf_pred = float(rf_models.get(tf).predict(X_rf_scaled)[0])
        except Exception as e:
            logging.error(f"{symbol} [{tf}]: errore nella predizione RF: {e}")
            return None, None, None
        
        # Predizione XGBoost
        try:
            data_xgb = prepare_data(df)
            if len(data_xgb) < time_steps:
                logging.error(f"{symbol} [{tf}]: dati insufficienti per XGB ({len(data_xgb)} elementi, richiesti {time_steps}).")
                return None, None, None
                
            # Assicurati che non ci siano NaN o infiniti nei dati
            if np.isnan(data_xgb).any() or np.isinf(data_xgb).any():
                logging.warning(f"{symbol} [{tf}]: valori NaN o infiniti trovati nei dati XGB, sostituiti con 0.0")
                data_xgb = np.nan_to_num(data_xgb, nan=0.0, posinf=0.0, neginf=0.0)
                
            X_xgb = data_xgb[-time_steps:].flatten().reshape(1, -1)
            X_xgb_scaled = xgb_scalers.get(tf).transform(X_xgb)
            expected_xgb_features = time_steps * len(EXPECTED_COLUMNS)
            if X_xgb_scaled.shape[1] != expected_xgb_features:
                logging.error(f"{symbol} [{tf}]: numero di feature XGB non corretto ({X_xgb_scaled.shape[1]} invece di {expected_xgb_features}).")
                return None, None, None
            xgb_pred = float(xgb_models.get(tf).predict(X_xgb_scaled)[0])
        except Exception as e:
            logging.error(f"{symbol} [{tf}]: errore nella predizione XGB: {e}")
            return None, None, None

        lstm_preds[tf] = lstm_pred
        rf_preds[tf] = rf_pred
        xgb_preds[tf] = xgb_pred

    # Calcola l'RSI utilizzando il timeframe predefinito da config
    try:
        rsi_value = dataframes[config.TIMEFRAME_DEFAULT]['rsi_fast'].iloc[-1]
    except Exception as e:
        logging.error(f"{symbol}: errore nel calcolo dell'RSI per {config.TIMEFRAME_DEFAULT}: {e}")
        rsi_value = 50

    # Calcolo dell'ensemble value: somma dei contributi pesati e normalizzazione per il numero dei timeframe
    ensemble_value = 0.0
    for tf in dataframes.keys():
        ensemble_value += (weight_lstm.get(tf, 0) * lstm_preds[tf] +
                           weight_rf.get(tf, 0) * rf_preds[tf] +
                           weight_xgb.get(tf, 0) * xgb_preds[tf])
    # Normalizza dividendo per il numero dei timeframe
    ensemble_value /= len(dataframes)

    final_signal = None
    if ensemble_value > NEUTRAL_UPPER_THRESHOLD:
        if rsi_value < RSI_THRESHOLDS['sideways']['oversold']:
            final_signal = 1
        else:
            logging.info(f"{symbol}: segnale BUY non confermato (RSI = {rsi_value}).")
    elif ensemble_value < NEUTRAL_LOWER_THRESHOLD:
        if rsi_value > RSI_THRESHOLDS['sideways']['overbought']:
            final_signal = 0
        else:
            logging.info(f"{symbol}: segnale SELL non confermato (RSI = {rsi_value}).")
    else:
        logging.info(f"{symbol}: segnale neutro per ensemble_value = {ensemble_value:.4f}")

    for tf in dataframes.keys():
        logging.info(f"{symbol} [{tf}] - LSTM: {lstm_preds[tf]:.4f}, RF: {rf_preds[tf]:.4f}, XGB: {xgb_preds[tf]:.4f}")
    logging.info(f"{symbol} - RSI ({config.TIMEFRAME_DEFAULT}): {rsi_value:.0f}")
    
    return ensemble_value, final_signal, (lstm_preds, rf_preds, xgb_preds, rsi_value)
