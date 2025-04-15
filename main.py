#!/usr/bin/env python3
import sys
import os
import numpy as np
from datetime import timedelta
import asyncio
import logging
import re
import ccxt.async_support as ccxt_async
from termcolor import colored
from tqdm import tqdm  # Import per la progress bar
import time

from config import (
    exchange_config,
    EXCLUDED_SYMBOLS, TIME_STEPS, TRADE_CYCLE_INTERVAL,
    MODEL_RATES,  # I rate definiti in config; la somma DEVE essere pari a 1
    RESET_DB_ON_STARTUP, DB_FILE,
    TOP_TRAIN_CRYPTO, TOP_ANALYSIS_CRYPTO, EXPECTED_COLUMNS,
    TRAIN_IF_NOT_FOUND,  # Variabile di controllo per il training
    LEVERAGE, MARGIN_USDT  # Parametri di trading configurabili
)
from logging_config import *
from fetcher import fetch_markets, get_top_symbols, fetch_min_amounts, fetch_and_save_data, fetch_ohlcv_data
from model_loader import (
    load_lstm_model_func,
    load_random_forest_model_func,
    load_xgboost_model_func
)
from trainer import (
    train_lstm_model_for_timeframe,
    train_random_forest_model_wrapper,
    train_xgboost_model_wrapper
)
from predictor import predict_signal_ensemble, get_color_normal
from data_utils import prepare_data, add_technical_indicators
from db_manager import init_data_tables
from trainer import ensure_trained_models_dir
from trade_manager import (
    get_real_balance, manage_position, get_open_positions,
    update_orders_status, load_existing_positions, monitor_open_trades,
    print_trade_statistics, save_trade_statistics, compute_trade_statistics_for_period,
    init_db
)

# Rendi disponibili le variabili a livello globale per poterle modificare dall'API
LEVERAGE = LEVERAGE
MARGIN_USDT = MARGIN_USDT
TOP_ANALYSIS_CRYPTO = TOP_ANALYSIS_CRYPTO

if sys.platform.startswith('win'):
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

first_cycle = True

# --- Sezione: Configurazione interattiva ---
def select_config():
    # Utilizza i valori predefiniti dal file config.py
    import config
    default_timeframes = config.ENABLED_TIMEFRAMES
    default_models = config.SELECTED_MODELS  # Utilizza i modelli predefiniti dal config.py
    return default_timeframes, default_models

# Esegui la selezione e aggiorna la configurazione
selected_timeframes, selected_models = select_config()
import config
config.ENABLED_TIMEFRAMES = selected_timeframes
config.TIMEFRAME_DEFAULT = selected_timeframes[0]

# Aggiorna le variabili locali per comodità
ENABLED_TIMEFRAMES = selected_timeframes
TIMEFRAME_DEFAULT = ENABLED_TIMEFRAMES[0]

# --- Calcolo dei pesi raw e normalizzati per i modelli ---
raw_weights = {}
for tf in ENABLED_TIMEFRAMES:
    raw_weights[tf] = {}
    for model in selected_models:
        raw_weights[tf][model] = MODEL_RATES.get(model, 0)
def normalize_weights(raw_weights):
    normalized = {}
    for tf, weights in raw_weights.items():
        total = sum(weights.values())
        if total > 0:
            normalized[tf] = {model: weight / total for model, weight in weights.items()}
        else:
            normalized[tf] = weights
    return normalized
normalized_weights = normalize_weights(raw_weights)

# --- Funzioni ausiliarie ---
async def track_orders():
    while True:
        await update_orders_status(async_exchange)
        await asyncio.sleep(60)

async def countdown_timer(duration):
    for remaining in tqdm(range(duration, 0, -1), desc="Attesa ciclo successivo", ncols=80, ascii=True):
        await asyncio.sleep(1)
    print()

async def trade_signals():
    global async_exchange, lstm_models, lstm_scalers, rf_models, rf_scalers, xgb_models, xgb_scalers, min_amounts

    while True:
        try:
            predicted_buys = []
            predicted_sells = []
            predicted_neutrals = []

            logging.info(colored("Statistiche iniziali (DB):", "cyan"))
            print_trade_statistics()
            await load_existing_positions(async_exchange)

            markets = await fetch_markets(async_exchange)
            all_symbols_analysis = [m['symbol'] for m in markets.values() if m.get('quote') == 'USDT'
                                    and m.get('active') and m.get('type') == 'swap'
                                    and not re.search('|'.join(EXCLUDED_SYMBOLS), m['symbol'])]
            top_symbols_analysis = await get_top_symbols(async_exchange, all_symbols_analysis, top_n=TOP_ANALYSIS_CRYPTO)
            logging.info(f"{colored('Simboli per analisi:', 'cyan')} {', '.join(top_symbols_analysis)}")

            reference_counts = {}
            first_symbol = top_symbols_analysis[0]
            for tf in ENABLED_TIMEFRAMES:
                df = await fetch_and_save_data(async_exchange, first_symbol, tf)
                if df is not None:
                    reference_counts[tf] = len(df)
                    logging.info(f"Reference candle count for {tf}: {reference_counts[tf]} from {first_symbol}")

            usdt_balance = await get_real_balance(async_exchange)
            if usdt_balance is None:
                logging.warning(colored("AVVISO: Impossibile ottenere il saldo USDT. Nuovo tentativo tra 5 secondi.", "yellow"))
                await asyncio.sleep(5)
                return
            open_positions_count = await get_open_positions(async_exchange)
            logging.info(f"{colored('USDT Balance:', 'cyan')} {colored(f'{usdt_balance:.2f}', 'yellow')} | {colored('Open Positions:', 'cyan')} {colored(str(open_positions_count), 'yellow')}")

            for index, symbol in enumerate(top_symbols_analysis, start=1):
                logging.info(colored("-" * 60, "white"))
                try:
                    logging.info(f"{colored(f'[{index}/{len(top_symbols_analysis)}] Analizzo', 'magenta')} {colored(symbol, 'yellow')}...")
                    dataframes = {}
                    skip_symbol = False

                    for tf in ENABLED_TIMEFRAMES:
                        df = await fetch_and_save_data(async_exchange, symbol, tf)
                        if df is None or len(df) < reference_counts[tf] * 0.9:  # accetta anche il 90% dei dati attesi
                            logging.warning(colored(f"AVVISO: Saltando {symbol}: Dati candlestick insufficienti per {tf} (Ottenuti: {len(df) if df is not None else 0}, Attesi: {reference_counts[tf]})", "yellow"))
                            skip_symbol = True
                            break
                        dataframes[tf] = df

                    if skip_symbol:
                        continue

                    ensemble_value, final_signal, predictions = predict_signal_ensemble(
                        dataframes,
                        lstm_models, lstm_scalers,
                        rf_models, rf_scalers,
                        xgb_models, xgb_scalers,
                        symbol, TIME_STEPS,
                        {tf: normalized_weights[tf]['lstm'] for tf in dataframes.keys()},
                        {tf: normalized_weights[tf]['rf'] for tf in dataframes.keys()},
                        {tf: normalized_weights[tf]['xgb'] for tf in dataframes.keys()}
                    )
                    if ensemble_value is None:
                        continue
                    logging.info(f"{colored('Ensemble value:', 'blue')} {colored(f'{ensemble_value:.4f}', get_color_normal(ensemble_value))}")
                    logging.info(f"{colored('Predizioni:', 'blue')} {colored(str(predictions), 'magenta')}")
                    if final_signal is None:
                        logging.info(colored("SEGNALE: Zona di indecisione.", "yellow"))
                        continue
                    else:
                        if final_signal == 1:
                            predicted_buys.append(symbol)
                        elif final_signal == 0:
                            predicted_sells.append(symbol)
                    logging.info(f"{colored('Decisione Finale di Trading:', 'green')} {colored('COMPRA' if final_signal==1 else 'VENDI', 'cyan')}")
                    result = await manage_position(
                        async_exchange,
                        symbol,
                        final_signal,
                        usdt_balance,
                        min_amounts,
                        None,
                        None,
                        None,
                        None,
                        dataframes[TIMEFRAME_DEFAULT]
                    )
                    if result == "insufficient_balance":
                        logging.info(f"{colored(symbol, 'yellow')}: Trade non eseguito per mancanza di balance.")
                        break
                except Exception as e:
                    logging.error(f"{colored('ERRORE durante elaborazione di', 'red')} {symbol}: {e}")
                logging.info(colored("-" * 60, "white"))

            logging.info(colored("Fine ciclo: aggiornamento statistiche.", "cyan"))
            print_trade_statistics()
            save_trade_statistics()
            logging.info(colored("Bot is running", "green"))

            await countdown_timer(TRADE_CYCLE_INTERVAL)

        except Exception as e:
            logging.error(f"{colored('Error in trade cycle:', 'red')} {e}")
            await asyncio.sleep(60)

# Funzione missing per le predizioni
async def get_data_for_symbol(exchange, symbol, timeframe):
    """
    Recupera i dati per un singolo simbolo e timeframe direttamente dall'exchange.
    Versione semplificata senza uso del database.
    """
    try:
        # Prova a recuperare direttamente i dati dall'exchange con più campioni
        # TIME_STEPS * 10 garantisce abbastanza dati per gli indicatori tecnici
        limit = TIME_STEPS * 10
        
        # Recupero diretto dall'exchange 
        from fetcher import fetch_ohlcv_data
        ohlcv_data = await fetch_ohlcv_data(exchange, symbol, timeframe, limit=limit)
        
        if ohlcv_data is None or len(ohlcv_data) == 0:
            logging.warning(f"Nessun dato disponibile per {symbol} (timeframe {timeframe})")
            return None
            
        import pandas as pd
        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        if len(df) < TIME_STEPS * 2:  # Minimo 2x TIME_STEPS per calcolare gli indicatori
            logging.warning(f"Dati insufficienti per {symbol} (timeframe {timeframe}): {len(df)} < {TIME_STEPS*2}")
            return None
            
        logging.info(f"Dati recuperati per {symbol} ({timeframe}): {len(df)} campioni")
        
        # Calcola tutti gli indicatori tecnici necessari
        from data_utils import add_technical_indicators
        df = add_technical_indicators(df)
        
        # Verifica che tutte le colonne EXPECTED_COLUMNS siano presenti
        missing_columns = set(EXPECTED_COLUMNS) - set(df.columns)
        if missing_columns:
            # Aggiungi le colonne mancanti con valori 0
            for col in missing_columns:
                df[col] = 0.0
        
        return df
    except Exception as e:
        logging.error(f"Errore nel recupero dati per {symbol} (timeframe {timeframe}): {e}")
        return None

async def predict_with_model(model, scaler, data, model_type='lstm'):
    """
    Genera una predizione utilizzando il modello specificato.
    Utilizzata dall'endpoint /predictions.
    """
    from data_utils import prepare_data
    try:
        # Assicurati che i dati abbiano tutte le colonne richieste
        required_columns = set(EXPECTED_COLUMNS)
        existing_columns = set(data.columns)
        missing_columns = required_columns - existing_columns
        
        if missing_columns:
            logging.warning(f"Mancano {len(missing_columns)} colonne nei dati: {missing_columns}")
            # Aggiungi colonne mancanti con valori 0
            for col in missing_columns:
                data[col] = 0.0
        
        # Riordina le colonne nello stesso ordine usato durante l'addestramento
        try:
            data = data[EXPECTED_COLUMNS]
        except KeyError as e:
            logging.error(f"Errore nel riordinamento colonne: {e}")
            # Fallback: usa le colonne esistenti
            logging.warning("Utilizzando prepare_data per normalizzare i dati...")
        
        if model_type == 'lstm':
            # Prepara i dati per LSTM
            X = prepare_data(data)
            if len(X) < TIME_STEPS:
                logging.error(f"Dati insufficienti per LSTM: {len(X)} < {TIME_STEPS}")
                return 0.5  # Valore neutrale in caso di dati insufficienti
                
            X = X[-TIME_STEPS:].copy()  # Usa .copy() per evitare avvertimenti SettingWithCopyWarning
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            try:
                X_scaled = scaler.transform(X)
                X_reshaped = X_scaled.reshape((1, TIME_STEPS, len(EXPECTED_COLUMNS)))
                prediction = float(model.predict(X_reshaped)[0][0])
            except Exception as e:
                logging.error(f"Errore nella predizione LSTM: {e}")
                return 0.5
        else:
            # Prepara i dati per RF e XGB
            X = prepare_data(data)
            if len(X) < TIME_STEPS:
                logging.error(f"Dati insufficienti per {model_type}: {len(X)} < {TIME_STEPS}")
                return 0.5
                
            X = X[-TIME_STEPS:].copy()
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            try:
                X_flat = X.flatten().reshape(1, -1)
                X_scaled = scaler.transform(X_flat)
                prediction = float(model.predict(X_scaled)[0])
            except Exception as e:
                logging.error(f"Errore nella predizione {model_type}: {e}")
                return 0.5
        
        # Limita la predizione nell'intervallo [0,1]
        prediction = max(0.0, min(1.0, prediction))
        return prediction
    except Exception as e:
        logging.error(f"Errore generale nella predizione con modello {model_type}: {e}")
        return 0.5  # Valore neutro in caso di errore

async def load_model(model_type, timeframe):
    """
    Carica un modello dal disco.
    Utilizzata dall'endpoint /predictions.
    """
    try:
        if model_type == 'lstm':
            model, scaler = await asyncio.to_thread(load_lstm_model_func, timeframe)
        elif model_type == 'rf':
            model, scaler = await asyncio.to_thread(load_random_forest_model_func, timeframe)
        elif model_type == 'xgb':
            model, scaler = await asyncio.to_thread(load_xgboost_model_func, timeframe)
        else:
            logging.error(f"Tipo di modello {model_type} non supportato")
            return None, None
            
        return model, scaler
    except Exception as e:
        logging.error(f"Errore nel caricamento del modello {model_type} per timeframe {timeframe}: {e}")
        return None, None

# --- Funzione Main ---
async def main():
    global async_exchange, lstm_models, lstm_scalers, rf_models, rf_scalers, xgb_models, xgb_scalers, min_amounts

    # Inizializza l'exchange
    async_exchange = ccxt_async.bybit(exchange_config)
    await async_exchange.load_markets()
    await async_exchange.load_time_difference()

    try:
        markets = await fetch_markets(async_exchange)
        all_symbols = [m['symbol'] for m in markets.values() if m.get('quote') == 'USDT'
                       and m.get('active') and m.get('type') == 'swap']
        all_symbols_analysis = [s for s in all_symbols if not re.search('|'.join(EXCLUDED_SYMBOLS), s)]

        top_symbols_analysis = await get_top_symbols(async_exchange, all_symbols_analysis, top_n=TOP_ANALYSIS_CRYPTO)
        logging.info(f"{colored('Simboli per analisi:', 'cyan')} {', '.join(top_symbols_analysis)}")

        # Ensure trained_models directory exists
        ensure_trained_models_dir()
        
        # Initialize models and scalers
        lstm_models = {}
        lstm_scalers = {}
        rf_models = {}
        rf_scalers = {}
        xgb_models = {}
        xgb_scalers = {}
        
        for tf in ENABLED_TIMEFRAMES:
            lstm_models[tf], lstm_scalers[tf] = await asyncio.to_thread(load_lstm_model_func, tf)
            rf_models[tf], rf_scalers[tf] = await asyncio.to_thread(load_random_forest_model_func, tf)
            xgb_models[tf], xgb_scalers[tf] = await asyncio.to_thread(load_xgboost_model_func, tf)
        
        # Entra in un loop per fare predizioni sulle monete
        while True:
            for symbol in top_symbols_analysis:
                logging.info(colored("-" * 60, "white"))
                logging.info(f"{colored('Analizzo', 'magenta')} {colored(symbol, 'yellow')}...")
                
                dataframes = {}
                for tf in ENABLED_TIMEFRAMES:
                    data = await get_data_for_symbol(async_exchange, symbol, tf)
                    if data is not None:
                        dataframes[tf] = data
                
                if not dataframes:
                    logging.warning(f"Nessun dato valido per {symbol}, salta predizione")
                    continue
                
                # Calcola predizioni usando tutti i modelli disponibili
                for tf, df in dataframes.items():
                    for model_type in ['lstm', 'rf', 'xgb']:
                        if model_type in selected_models:
                            model = None
                            scaler = None
                            
                            if model_type == 'lstm' and tf in lstm_models:
                                model = lstm_models[tf]
                                scaler = lstm_scalers[tf]
                            elif model_type == 'rf' and tf in rf_models:
                                model = rf_models[tf]
                                scaler = rf_scalers[tf]
                            elif model_type == 'xgb' and tf in xgb_models:
                                model = xgb_models[tf]
                                scaler = xgb_scalers[tf]
                            
                            if model is not None and scaler is not None:
                                prediction = await predict_with_model(model, scaler, df, model_type)
                                logging.info(f"{colored(f'Predizione {model_type.upper()} {tf}:', 'blue')} {colored(f'{prediction:.4f}', get_color_normal(prediction))}")
                
                logging.info(colored("-" * 60, "white"))
            
            # Attendi prima del prossimo ciclo
            logging.info(colored("Attesa 5 minuti per il prossimo ciclo...", "cyan"))
            await asyncio.sleep(300)
            
    except KeyboardInterrupt:
        logging.info(colored("Interrupt signal received. Shutting down...", "red"))
    except Exception as e:
        error_msg = str(e)
        logging.error(f"{colored('Error in main loop:', 'red')} {error_msg}")
    finally:
        await async_exchange.close()
        logging.info(colored("Program terminated.", "red"))

if __name__ == "__main__":
    asyncio.run(main())
