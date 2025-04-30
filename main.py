#!/usr/bin/env python3
import sys
import os
import numpy as np
from datetime import timedelta, datetime
import asyncio
import logging
import re
import ccxt.async_support as ccxt_async
from termcolor import colored
from tqdm import tqdm  # Import per la progress bar
from dotenv import load_dotenv
from state import app_state

# Carica le variabili d'ambiente dal file .env
load_dotenv()

from config import (
    exchange_config,
    EXCLUDED_SYMBOLS, TIME_STEPS, TRADE_CYCLE_INTERVAL,
    MODEL_RATES,  # I rate definiti in config; la somma DEVE essere pari a 1
    TOP_TRAIN_CRYPTO, TOP_ANALYSIS_CRYPTO, EXPECTED_COLUMNS,
    TRAIN_IF_NOT_FOUND,  # Variabile di controllo per il training
    ENABLED_TIMEFRAMES, TIMEFRAME_DEFAULT,  # Importazione dei timeframe predefiniti da config
    get_lstm_model_file, get_lstm_scaler_file,
    get_rf_model_file, get_rf_scaler_file,
    get_xgb_model_file, get_xgb_scaler_file
)
from logging_utils import *
from fetcher import fetch_markets, get_top_symbols, fetch_min_amounts, fetch_and_save_data
from model_manager import (
    load_lstm_model,
    load_rf_model,
    load_xgb_model,
    train_lstm_model_for_timeframe,
    train_random_forest_model_wrapper,
    train_xgboost_model_wrapper,
    ensure_trained_models_dir
)
from predictor import predict_signal_ensemble, get_color_normal
from trade_manager import (
    get_real_balance, manage_position, get_open_positions,
    load_existing_positions, monitor_open_trades,
    wait_and_update_closed_trades
)
from data_utils import prepare_data

if sys.platform.startswith('win'):
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

first_cycle = True

# Modelli predefiniti da utilizzare
selected_models = ['lstm', 'rf', 'xgb']

# --- Calcolo dei pesi raw e normalizzati per i modelli ---
raw_weights = {}
for tf in app_state.enabled_timeframes:
    raw_weights[tf] = {}
    for model in app_state.selected_models:
        raw_weights[tf][model] = app_state.model_rates.get(model, 0)

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

# Funzione per verificare quali modelli mancano
def check_missing_models(timeframes, models):
    missing_models = {}
    for tf in timeframes:
        missing_for_tf = []
        
        if 'lstm' in models:
            lstm_file = get_lstm_model_file(tf)
            if not os.path.exists(lstm_file):
                print(colored(f"File LSTM non trovato: {lstm_file}", "yellow"))
                missing_for_tf.append('lstm')
            else:
                print(colored(f"File LSTM trovato: {lstm_file}", "green"))
            
        if 'rf' in models:
            rf_file = get_rf_model_file(tf)
            if not os.path.exists(rf_file):
                print(colored(f"File RF non trovato: {rf_file}", "yellow")) 
                missing_for_tf.append('rf')
            else:
                print(colored(f"File RF trovato: {rf_file}", "green"))
            
        if 'xgb' in models:
            xgb_file = get_xgb_model_file(tf)
            print(colored(f"Cercando file XGB: {xgb_file}", "cyan"))
            
            # Controllo anche l'estensione alternativa .model per XGBoost
            alt_xgb_file = f"trained_models/xgb_model_{tf}.model"
            
            if not os.path.exists(xgb_file) and not os.path.exists(alt_xgb_file):
                print(colored(f"File XGB non trovato (cercato: {xgb_file} e {alt_xgb_file})", "yellow"))
                missing_for_tf.append('xgb')
            else:
                if os.path.exists(xgb_file):
                    print(colored(f"File XGB trovato: {xgb_file}", "green"))
                else:
                    print(colored(f"File XGB trovato con estensione alternativa: {alt_xgb_file}", "green"))
            
        if missing_for_tf:
            missing_models[tf] = missing_for_tf
            
    return missing_models

# Funzione per selezionare modelli da trainare interattivamente
def select_models_to_train(missing_models):
    if not missing_models:
        print(colored("Tutti i modelli sono già addestrati!", "green"))
        return {}
    
    print(colored("\n=== MODELLI MANCANTI ===", "yellow"))
    timeframes = list(missing_models.keys())
    
    for i, tf in enumerate(timeframes):
        print(f"{i+1}. Timeframe {tf}: {', '.join(missing_models[tf])}")
    
    print("\nOpzioni:")
    print("A. Addestra tutti i modelli mancanti")
    print("S. Seleziona specifici modelli da addestrare")
    print("N. Non addestrare nessun modello (avvia solo il bot)")
    
    choice = input("\nScelta [A/S/N]: ").strip().upper()
    
    if choice == 'N':
        return {}
        
    if choice == 'A':
        return missing_models
        
    if choice == 'S':
        selected = {}
        
        for tf in timeframes:
            print(f"\nTimeframe {tf}:")
            for model in missing_models[tf]:
                train_it = input(f"Addestrare il modello {model.upper()}? [s/n]: ").strip().lower() == 's'
                if train_it:
                    selected.setdefault(tf, []).append(model)
        
        return selected
        
    # Default: addestra tutto se input non valido
    print(colored("Input non valido, verranno addestrati tutti i modelli mancanti.", "yellow"))
    return missing_models

# --- Funzioni ausiliarie ---
async def countdown_timer(duration):
    for remaining in tqdm(range(duration, 0, -1), desc="Attesa ciclo successivo", ncols=80, ascii=True):
        await asyncio.sleep(1)
    print()

async def trade_signals():
    global async_exchange, lstm_models, lstm_scalers, rf_models, rf_scalers, xgb_models, xgb_scalers, min_amounts
    
    # Flag per controllare lo stato del bot
    is_running = True
    error_count = 0
    MAX_ERRORS = 3  # Numero massimo di errori consecutivi prima di fermare il bot
    
    while is_running:
        try:
            predicted_buys = []
            predicted_sells = []
            predicted_neutrals = []

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
                logging.warning(colored("Failed to get USDT balance. Retrying in 5 seconds.", "yellow"))
                await asyncio.sleep(5)
                continue

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
                            logging.warning(colored(f"Skipping {symbol}: Insufficient candles for {tf} (Got: {len(df) if df is not None else 0}, Expected: {reference_counts[tf]})", "yellow"))
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
                        logging.info(colored("Segnale neutro: zona di indecisione.", "yellow"))
                        continue
                    else:
                        if final_signal == 1:
                            predicted_buys.append(symbol)
                        elif final_signal == 0:
                            predicted_sells.append(symbol)
                    logging.info(f"{colored('Final Trading Decision:', 'green')} {colored('BUY' if final_signal==1 else 'SELL', 'cyan')}")
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
                    logging.error(f"{colored('Error processing', 'red')} {symbol}: {e}")
                logging.info(colored("-" * 60, "white"))

            logging.info(colored("Bot is running", "green"))
            error_count = 0  # Reset error count on successful iteration
            await countdown_timer(TRADE_CYCLE_INTERVAL)

        except Exception as e:
            error_count += 1
            logging.error(f"{colored('Error in trade cycle:', 'red')} {e}")
            
            if error_count >= MAX_ERRORS:
                logging.error(colored(f"Troppi errori consecutivi ({error_count}). Fermando il bot...", "red"))
                is_running = False
                break
                
            # Aumenta il tempo di attesa in base al numero di errori
            wait_time = min(60 * error_count, 300)  # Max 5 minuti
            logging.info(colored(f"Aspettando {wait_time} secondi prima di riprovare...", "yellow"))
            await asyncio.sleep(wait_time)

# Funzione asincrona per addestrare un singolo modello
async def train_model(exchange, symbols, model_type, timeframe):
    logging.info(colored(f"Avvio training {model_type.upper()} per timeframe {timeframe}...", "cyan"))
    
    try:
        if model_type == 'lstm':
            model, scaler, _ = await train_lstm_model_for_timeframe(
                exchange, symbols, timeframe=timeframe, timestep=TIME_STEPS)
            model_path = get_lstm_model_file(timeframe)
            
        elif model_type == 'rf':
            model, scaler, _ = await train_random_forest_model_wrapper(
                symbols, exchange, timestep=TIME_STEPS, timeframe=timeframe)
            model_path = get_rf_model_file(timeframe)
            
        elif model_type == 'xgb':
            model, scaler, _ = await train_xgboost_model_wrapper(
                symbols, exchange, timestep=TIME_STEPS, timeframe=timeframe)
            model_path = get_xgb_model_file(timeframe)
            
        logging.info(colored(f"Training {model_type.upper()} per {timeframe} completato! Modello salvato in {model_path}", "green"))
        return model, scaler
    except Exception as e:
        logging.error(colored(f"Errore durante il training {model_type.upper()} per {timeframe}: {e}", "red"))
        return None, None

# Funzione asincrona per addestrare tutti i modelli in parallelo
async def train_models_in_parallel(exchange, symbols, model_types, timeframes):
    """Addestra tutti i modelli per tutti i timeframe in parallelo.
    
    Args:
        exchange: L'exchange da utilizzare per i dati
        symbols: Lista di simboli da utilizzare per il training
        model_types: Lista di tipi di modelli da addestrare ('lstm', 'rf', 'xgb')
        timeframes: Lista di timeframe da utilizzare
    
    Returns:
        Un dizionario contenente tutti i modelli e gli scaler addestrati
    """
    
    print(colored(f"Avvio training parallelo di {len(model_types) * len(timeframes)} modelli...", "cyan"))
    
    # Preparazione per i task paralleli
    tasks = []
    model_results = {}
    
    # Inizializza il dizionario dei risultati
    for model_type in model_types:
        model_results[model_type] = {'models': {}, 'scalers': {}}
    
    # Crea i task per ogni combinazione di modello e timeframe
    for tf in timeframes:
        for model_type in model_types:
            print(colored(f"Aggiunto task per training {model_type.upper()} - {tf}", "cyan"))
            tasks.append(train_model(exchange, symbols, model_type, tf))
    
    # Esegui tutti i task in parallelo
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Processa i risultati
    task_index = 0
    for tf in timeframes:
        for model_type in model_types:
            result = results[task_index]
            task_index += 1
            
            # Gestione degli errori
            if isinstance(result, Exception):
                logging.error(f"Errore nel training {model_type} per {tf}: {result}")
                continue
                
            # Estrai modello e scaler
            model, scaler = result
            
            if model is not None and scaler is not None:
                # Salva nel dizionario dei risultati
                model_results[model_type]['models'][tf] = model
                model_results[model_type]['scalers'][tf] = scaler
                print(colored(f"Training {model_type.upper()} per {tf} completato con successo", "green"))
            else:
                logging.warning(f"Training {model_type.upper()} per {tf} fallito o prodotto risultato None")
    
    # Riassunto dei risultati
    print(colored("\n=== RIEPILOGO TRAINING ===", "cyan"))
    for model_type in model_types:
        successful = len(model_results[model_type]['models'])
        print(colored(f"{model_type.upper()}: {successful}/{len(timeframes)} modelli addestrati", 
                     "green" if successful == len(timeframes) else "yellow"))
    
    return model_results

# Nuova funzione per gestire il training-only
async def execute_training_only(timeframes=None, models=None, num_symbols=None):
    print(colored("\n=== MODALITÀ TRAINING MODELLI ===", "cyan"))
    
    # Se non è specificato, usa tutte le timeframe disponibili
    if not timeframes:
        timeframes = ENABLED_TIMEFRAMES
    
    # Se non è specificato, usa tutti i modelli disponibili
    if not models:
        models = ['lstm', 'rf', 'xgb']
    
    # Numero di simboli per training
    if not num_symbols:
        num_symbols = TOP_TRAIN_CRYPTO
    
    print(colored(f"Timeframes selezionati: {', '.join(timeframes)}", "yellow"))
    print(colored(f"Modelli selezionati: {', '.join(models)}", "yellow"))
    print(colored(f"Numero di simboli: {num_symbols}", "yellow"))
    
    # Inizializza lo scambio
    async_exchange = ccxt_async.bybit(exchange_config)
    await async_exchange.load_markets()
    
    try:
        # Prepara i dati per il training
        markets = await fetch_markets(async_exchange)
        all_symbols = [m['symbol'] for m in markets.values() if m.get('quote') == 'USDT'
                      and m.get('active') and m.get('type') == 'swap']
        all_symbols = [s for s in all_symbols if not re.search('|'.join(EXCLUDED_SYMBOLS), s)]
        
        # Ottieni i top simboli per il training
        top_symbols_training = await get_top_symbols(async_exchange, all_symbols, top_n=num_symbols)
        
        # Validazione dei dati
        validated_symbols = []
        print(colored("Validazione dei dati per il training...", "cyan"))
        
        for i, symbol in enumerate(top_symbols_training):
            valid = True
            print(f"Verificando {symbol} ({i+1}/{len(top_symbols_training)})...", end="\r")
            
            for tf in timeframes:
                df = await fetch_and_save_data(async_exchange, symbol, tf)
                if df is None or df.isnull().any().any() or np.isinf(df).any().any():
                    valid = False
                    break
            
            if valid:
                validated_symbols.append(symbol)
        
        print(" " * 80, end="\r")  # Clear the line
        print(colored(f"Dati validati: {len(validated_symbols)}/{len(top_symbols_training)} simboli utilizzabili", "green"))
        
        # Verifica se ci sono abbastanza dati
        if len(validated_symbols) < 10:
            proceed = input(colored(f"ATTENZIONE: Solo {len(validated_symbols)} simboli validi trovati. Continuare? [s/n]: ", "red")).strip().lower() == 's'
            if not proceed:
                print(colored("Training annullato dall'utente.", "red"))
                return
        
        # Crea directory per i modelli
        ensure_trained_models_dir()
        
        # Misura il tempo di addestramento
        start_time = datetime.now()
        
        # Training parallelo dei modelli
        print(colored("\nAvvio training parallelo di tutti i modelli...", "cyan"))
        model_results = await train_models_in_parallel(
            async_exchange, 
            validated_symbols, 
            models, 
            timeframes
        )
        
        # Calcola e mostra il tempo totale
        total_time = datetime.now() - start_time
        total_models = len(models) * len(timeframes)
        
        print(colored(f"\nTraining completato in {total_time}!", "green"))
        print(colored(f"Tempo medio per modello: {total_time/total_models}", "cyan"))
        
    except Exception as e:
        print(colored(f"Errore durante il training: {e}", "red"))
    
    finally:
        await async_exchange.close()
        print(colored("Sessione di training terminata.", "yellow"))

# --- Funzione Main ---
async def main():
    global async_exchange, lstm_models, lstm_scalers, rf_models, rf_scalers, xgb_models, xgb_scalers, min_amounts

    # Gestione degli argomenti da linea di comando
    import argparse
    parser = argparse.ArgumentParser(description="Trading Bot con modelli di ML")
    parser.add_argument('--train-only', action='store_true', help='Esegui solo il training dei modelli senza avviare il bot')
    parser.add_argument('--timeframes', type=str, help='Lista di timeframes separati da virgola (es. 1h,4h,1d)')
    parser.add_argument('--models', type=str, help='Lista di modelli separati da virgola (es. lstm,rf,xgb)')
    parser.add_argument('--symbols', type=int, help='Numero di simboli da utilizzare per il training')
    args = parser.parse_args()
    
    # Modalità training-only
    if args.train_only:
        timeframes = args.timeframes.split(',') if args.timeframes else None
        models = args.models.split(',') if args.models else None
        symbols = args.symbols if args.symbols else None
        await execute_training_only(timeframes, models, symbols)
        return
    
    print(colored("\n=== TRADING JII - SISTEMA DI TRADING ALGORITMICO ===", "cyan"))
    print(colored("Inizializzazione...", "yellow"))

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
        
        # Controlla quali modelli mancano
        missing_models = check_missing_models(ENABLED_TIMEFRAMES, selected_models)
        
        if missing_models:
            print(colored(f"Alcuni modelli non sono stati trovati per i timeframe: {', '.join(missing_models.keys())}", "yellow"))
            
            # Chiedi all'utente quali modelli vuole trainare
            models_to_train = select_models_to_train(missing_models)
            
            if models_to_train:
                print(colored("\nPreparazione per il training dei modelli selezionati...", "cyan"))
                
                # Prepara i dati per il training solo se necessario
                top_symbols_training = await get_top_symbols(async_exchange, all_symbols, top_n=TOP_TRAIN_CRYPTO)
                
                # Validazione dei dati prima del training
                validated_symbols = []
                print(colored("Validazione dei dati per il training...", "cyan"))
                
                for i, symbol in enumerate(top_symbols_training):
                    valid = True
                    print(f"Verificando {symbol} ({i+1}/{len(top_symbols_training)})...", end="\r")
                    
                    for tf in ENABLED_TIMEFRAMES:
                        df = await fetch_and_save_data(async_exchange, symbol, tf)
                        if df is None or df.isnull().any().any() or np.isinf(df).any().any():
                            valid = False
                            break
                    
                    if valid:
                        validated_symbols.append(symbol)
                
                print(" " * 80, end="\r")  # Clear the line
                print(colored(f"Dati validati: {len(validated_symbols)}/{len(top_symbols_training)} simboli utilizzabili", "green"))
                
                # Se non ci sono abbastanza simboli validi, avvisa l'utente
                if len(validated_symbols) < 10:
                    proceed = input(colored(f"ATTENZIONE: Solo {len(validated_symbols)} simboli validi trovati. Continuare? [s/n]: ", "red")).strip().lower() == 's'
                    if not proceed:
                        print(colored("Training annullato dall'utente.", "red"))
                        models_to_train = {}
                
                # Training dei modelli selezionati
                if models_to_train:
                    print(colored("\nInizio training dei modelli...", "cyan"))
                    ensure_trained_models_dir()
                    
                    # Initialize models and scalers
                    lstm_models = {}
                    lstm_scalers = {}
                    rf_models = {}
                    rf_scalers = {}
                    xgb_models = {}
                    xgb_scalers = {}
                    
                    # Prepara lista di modelli e timeframe da addestrare
                    tf_to_train = list(models_to_train.keys())
                    
                    # Prepara dizionario per modelli da addestrare per ogni timeframe
                    models_by_tf = {}
                    for tf in tf_to_train:
                        models_by_tf[tf] = models_to_train[tf]
                    
                    # Misura il tempo di training
                    start_time = datetime.now()
                    
                    try:
                        # Converti il dizionario di modelli_to_train in liste piatte per il training parallelo
                        all_tfs = []
                        all_models = []
                        
                        for tf, model_list in models_to_train.items():
                            for model_type in model_list:
                                all_tfs.append(tf)
                                all_models.append(model_type)
                        
                        print(colored(f"Avvio training parallelo di {len(all_tfs)} modelli...", "cyan"))
                        
                        # Crea task per ogni combinazione (tf, model)
                        tasks = [train_model(async_exchange, validated_symbols, model_type, tf) 
                                for tf, model_type in zip(all_tfs, all_models)]
                        
                        # Esegui tutti i training in parallelo
                        train_results = await asyncio.gather(*tasks, return_exceptions=True)
                        
                        # Processa i risultati
                        for i, result in enumerate(train_results):
                            tf = all_tfs[i]
                            model_type = all_models[i]
                            
                            # Gestione eccezioni
                            if isinstance(result, Exception):
                                logging.error(f"Errore nel training di {model_type} per {tf}: {str(result)}")
                                continue
                            
                            model, scaler = result
                            
                            if model is not None:
                                if model_type == 'lstm':
                                    lstm_models[tf] = model
                                    lstm_scalers[tf] = scaler
                                elif model_type == 'rf':
                                    rf_models[tf] = model
                                    rf_scalers[tf] = scaler
                                elif model_type == 'xgb':
                                    xgb_models[tf] = model
                                    xgb_scalers[tf] = scaler
                                
                                print(colored(f"Training di {model_type} per {tf} completato con successo", "green"))
                        
                        total_time = datetime.now() - start_time
                        print(colored(f"Training completato in {total_time}", "green"))
                    
                    except Exception as e:
                        print(colored(f"Errore durante il training parallelo: {str(e)}", "red"))
                        # Se fallisce il parallelo, prova il metodo sequenziale
                        print(colored("Tentativo con training sequenziale...", "yellow"))
                        
                        # Train models in sequence (fallback)
                        for tf in models_to_train:
                            for model_type in models_to_train[tf]:
                                model, scaler = await train_model(async_exchange, validated_symbols, model_type, tf)
                                
                                if model is not None:
                                    if model_type == 'lstm':
                                        lstm_models[tf] = model
                                        lstm_scalers[tf] = scaler
                                    elif model_type == 'rf':
                                        rf_models[tf] = model
                                        rf_scalers[tf] = scaler
                                    elif model_type == 'xgb':
                                        xgb_models[tf] = model
                                        xgb_scalers[tf] = scaler
            else:
                print(colored("Nessun modello selezionato per il training.", "yellow"))
                # Imposta TRAIN_IF_NOT_FOUND a False per evitare errori
                app_state.train_if_not_found = False
        else:
            print(colored("Tutti i modelli sono già presenti sul disco.", "green"))
            top_symbols_training = []  # Nessuna moneta per il training se i modelli esistono

        print(colored("\nCaricamento modelli...", "cyan"))
        
        # Load models
        lstm_models = {}
        lstm_scalers = {}
        rf_models = {}
        rf_scalers = {}
        xgb_models = {}
        xgb_scalers = {}
        
        for tf in ENABLED_TIMEFRAMES:
            lstm_models[tf], lstm_scalers[tf] = await asyncio.to_thread(load_lstm_model, tf)
            rf_models[tf], rf_scalers[tf] = await asyncio.to_thread(load_rf_model, tf)
            xgb_models[tf], xgb_scalers[tf] = await asyncio.to_thread(load_xgb_model, tf)
        
            # Training models if not found
            if 'lstm' in selected_models and not lstm_models[tf]:
                # Add additional check to see if the model file exists
                model_file = get_lstm_model_file(tf)
                if os.path.exists(model_file) and os.path.getsize(model_file) > 0:
                    logging.warning(f"Model file exists but couldn't be loaded: {model_file}")
                    
                if TRAIN_IF_NOT_FOUND and top_symbols_training:
                    logging.info(f"Training new LSTM model for timeframe {tf}")
                    lstm_models[tf], lstm_scalers[tf], _ = await train_lstm_model_for_timeframe(
                        async_exchange, top_symbols_training, timeframe=tf, timestep=TIME_STEPS)
                else:
                    raise Exception(f"LSTM model for timeframe {tf} not available. Train models first.")
                    
            if 'rf' in selected_models and not rf_models[tf]:
                if TRAIN_IF_NOT_FOUND and top_symbols_training:
                    rf_models[tf], rf_scalers[tf], _ = await train_random_forest_model_wrapper(
                        top_symbols_training, async_exchange, timestep=TIME_STEPS, timeframe=tf)
                else:
                    raise Exception(f"RF model for timeframe {tf} not available. Train models first.")
                    
            if 'xgb' in selected_models and not xgb_models[tf]:
                if TRAIN_IF_NOT_FOUND and top_symbols_training:
                    xgb_models[tf], xgb_scalers[tf], _ = await train_xgboost_model_wrapper(
                        top_symbols_training, async_exchange, timestep=TIME_STEPS, timeframe=tf)
                else:
                    raise Exception(f"XGBoost model for timeframe {tf} not available. Train models first.")

        print(colored("Modelli caricati con successo!", "green"))
        
        min_amounts = await fetch_min_amounts(async_exchange, top_symbols_analysis, markets)
        await load_existing_positions(async_exchange)

        trade_count = len(top_symbols_analysis)
        print(colored(f"\n=== TRADING BOT PRONTO ===", "cyan"))
        print(f"• Timeframes attivi: {', '.join(ENABLED_TIMEFRAMES)}")
        print(f"• Modelli utilizzati: {', '.join(selected_models)}")
        print(f"• Simboli per analisi: {trade_count}")
        print(f"• Intervallo ciclo trading: {TRADE_CYCLE_INTERVAL} secondi")
        
        print(colored("\nAvvio del bot di trading...", "green"))
        await asyncio.gather(
            trade_signals(),
            monitor_open_trades(async_exchange)
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
