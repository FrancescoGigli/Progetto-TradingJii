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
import json
import signal
import platform

# Importazione per il supporto del colore su Windows
try:
    import colorama
    colorama.init(autoreset=True)
    COLORED_OUTPUT = True
except ImportError:
    COLORED_OUTPUT = False
    print("Per avere output colorato, installa colorama: pip install colorama")

# Configurazione specifica per Windows
if platform.system() == 'Windows':
    # Configura asyncio per Windows
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Importa la configurazione di logging
import logging_config

from config import (
    exchange_config,
    EXCLUDED_SYMBOLS, TIME_STEPS, TRADE_CYCLE_INTERVAL,
    MODEL_RATES,  # I rate definiti in config; la somma DEVE essere pari a 1
    DB_FILE,
    TOP_TRAIN_CRYPTO, TOP_ANALYSIS_CRYPTO, EXPECTED_COLUMNS,
    TRAIN_IF_NOT_FOUND,  # Variabile di controllo per il training
    ENABLED_TIMEFRAMES, TIMEFRAME_DEFAULT, get_lstm_model_file  # Importazione dei timeframe predefiniti da config
)
from logging_config import *
from fetcher import fetch_markets, get_top_symbols, fetch_min_amounts, fetch_and_save_data
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
from trade_manager import (
    get_real_balance, manage_position, get_open_positions,
    update_orders_status, load_existing_positions, monitor_open_trades,
    print_trade_statistics, save_trade_statistics, compute_trade_statistics_for_period,
    wait_and_update_closed_trades
)
from data_utils import prepare_data
from trainer import ensure_trained_models_dir

first_cycle = True

# Modelli predefiniti da utilizzare
selected_models = ['lstm', 'rf', 'xgb']

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
                logging.warning(colored("[AVVISO] Failed to get USDT balance. Retrying in 5 seconds.", "yellow"))
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
                        if df is None or len(df) < reference_counts[tf] * 0.95:
                            logging.warning(colored(f"[AVVISO] Skipping {symbol}: Insufficient candles for {tf} (Got: {len(df) if df is not None else 0}, Expected: {reference_counts[tf]})", "yellow"))
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
                        logging.info(colored("[NOTIFICA] Segnale neutro: zona di indecisione.", "yellow"))
                        continue
                    else:
                        if final_signal == 1:
                            predicted_buys.append(symbol)
                        elif final_signal == 0:
                            predicted_sells.append(symbol)
                    logging.info(f"{colored('[DECISIONE] Final Trading Decision:', 'green')} {colored('BUY' if final_signal==1 else 'SELL', 'cyan')}")
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
                    logging.error(f"{colored('[ERRORE] Error processing', 'red')} {symbol}: {e}")
                logging.info(colored("-" * 60, "white"))

            logging.info(colored("Fine ciclo: aggiornamento statistiche.", "cyan"))
            print_trade_statistics()
            save_trade_statistics()
            logging.info(colored("Bot is running", "green"))

            await countdown_timer(TRADE_CYCLE_INTERVAL)

        except Exception as e:
            logging.error(f"{colored('Error in trade cycle:', 'red')} {e}")
            await asyncio.sleep(60)

# --- Funzione Main ---
async def main():
    global async_exchange, lstm_models, lstm_scalers, rf_models, rf_scalers, xgb_models, xgb_scalers, min_amounts

    async_exchange = ccxt_async.bybit(exchange_config)
    await async_exchange.load_markets()
    await async_exchange.load_time_difference()

    await load_existing_positions(async_exchange)

    try:
        markets = await fetch_markets(async_exchange)
        all_symbols = [m['symbol'] for m in markets.values() if m.get('quote') == 'USDT'
                       and m.get('active') and m.get('type') == 'swap']
        all_symbols_analysis = [s for s in all_symbols if not re.search('|'.join(EXCLUDED_SYMBOLS), s)]

        top_symbols_analysis = await get_top_symbols(async_exchange, all_symbols_analysis, top_n=TOP_ANALYSIS_CRYPTO)
        top_symbols_training = await get_top_symbols(async_exchange, all_symbols, top_n=TOP_TRAIN_CRYPTO)

        # Validazione dei dati prima del training
        for symbol in top_symbols_training[:]:
            for tf in ENABLED_TIMEFRAMES:
                df = await fetch_and_save_data(async_exchange, symbol, tf)
                if df is not None and (df.isnull().any().any() or np.isinf(df).any().any()):
                    logging.warning(f"Removing {symbol} from training set due to invalid data")
                    top_symbols_training.remove(symbol)
                    break

        logging.info(f"{colored('Numero di monete per il training:', 'cyan')} {colored(str(len(top_symbols_training)), 'yellow')}")
        logging.info(f"{colored('Numero di monete per analisi operativa:', 'cyan')} {colored(str(len(top_symbols_analysis)), 'yellow')}")

        min_amounts = await fetch_min_amounts(async_exchange, top_symbols_analysis, markets)

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
        
            # Training models if not found
            if 'lstm' in selected_models and not lstm_models[tf]:
                # Add additional check to see if the model file exists
                model_file = get_lstm_model_file(tf)
                if os.path.exists(model_file) and os.path.getsize(model_file) > 0:
                    logging.warning(f"Model file exists but couldn't be loaded: {model_file}")
                    
                if TRAIN_IF_NOT_FOUND:
                    logging.info(f"Training new LSTM model for timeframe {tf}")
                    lstm_models[tf], lstm_scalers[tf], _ = await train_lstm_model_for_timeframe(
                        async_exchange, top_symbols_training, timeframe=tf, timestep=TIME_STEPS)
                else:
                    raise Exception(f"LSTM model for timeframe {tf} not available. Train models first.")
                    
            if 'rf' in selected_models and not rf_models[tf]:
                if TRAIN_IF_NOT_FOUND:
                    rf_models[tf], rf_scalers[tf], _ = await train_random_forest_model_wrapper(
                        top_symbols_training, async_exchange, timestep=TIME_STEPS, timeframe=tf)
                else:
                    raise Exception(f"RF model for timeframe {tf} not available. Train models first.")
                    
            if 'xgb' in selected_models and not xgb_models[tf]:
                if TRAIN_IF_NOT_FOUND:
                    xgb_models[tf], xgb_scalers[tf], _ = await train_xgboost_model_wrapper(
                        top_symbols_training, async_exchange, timestep=TIME_STEPS, timeframe=tf)
                else:
                    raise Exception(f"XGBoost model for timeframe {tf} not available. Train models first.")

        save_trade_statistics()
        logging.info(colored("Modelli caricati da disco o allenati per tutti i timeframe abilitati.", "magenta"))
        await load_existing_positions(async_exchange)

        trade_count = len(top_symbols_analysis)
        logging.info(f"{colored('Numero totale di trade stimati (basato sui simboli per analisi):', 'cyan')} {colored(str(trade_count), 'yellow')}")

        await asyncio.gather(
            trade_signals(),
            monitor_open_trades(async_exchange),
            track_orders()
        )
    except KeyboardInterrupt:
        logging.info(colored("Interrupt signal received. Shutting down...", "red"))
    except Exception as e:
        error_msg = str(e)
        logging.error(f"{colored('Error in main loop:', 'red')} {error_msg}")
        if "invalid request, please check your server timestamp" in error_msg:
            logging.info(colored("Timestamp error detected. Restarting script...", "red"))
            os.execv(sys.executable, [sys.executable] + sys.argv)
    finally:
        await async_exchange.close()
        logging.info(colored("Program terminated.", "red"))

if __name__ == "__main__":
    asyncio.run(main())
