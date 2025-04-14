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
import concurrent.futures

from config import (
    exchange_config,
    EXCLUDED_SYMBOLS, TIME_STEPS, TRADE_CYCLE_INTERVAL,
    MODEL_RATES,  # I rate definiti in config; la somma DEVE essere pari a 1
    TOP_TRAIN_CRYPTO, TOP_ANALYSIS_CRYPTO, EXPECTED_COLUMNS,
    SYMBOLS_PER_ANALYSIS_CYCLE, SYMBOLS_FOR_VALIDATION,  # Nuove variabili di configurazione
    TRAIN_IF_NOT_FOUND,  # Variabile di controllo per il training
    ENABLED_TIMEFRAMES, TIMEFRAME_DEFAULT, SELECTED_MODELS as selected_models
)
from logging_config import *
from fetcher import (
    fetch_markets, get_top_symbols, fetch_min_amounts,
    fetch_data_for_multiple_symbols, get_data_async
)
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
    update_orders_status, load_existing_positions, monitor_open_trades
)
from data_utils import prepare_data
from db_manager import init_data_tables
from trainer import ensure_trained_models_dir

if sys.platform.startswith('win'):
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

first_cycle = True

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
            start_time = datetime.now()
            predicted_buys = []
            predicted_sells = []
            predicted_neutrals = []

            await load_existing_positions(async_exchange)

            markets = await fetch_markets(async_exchange)
            all_symbols_analysis = [m['symbol'] for m in markets.values() if m.get('quote') == 'USDT'
                                    and m.get('active') and m.get('type') == 'swap'
                                    and not re.search('|'.join(EXCLUDED_SYMBOLS), m['symbol'])]
            
            # Utilizziamo una quantità maggiore di simboli per analisi
            top_symbols_analysis = await get_top_symbols(async_exchange, all_symbols_analysis, top_n=TOP_ANALYSIS_CRYPTO)

            # Raccogliamo informazioni di riferimento
            reference_counts = {}
            first_symbol = top_symbols_analysis[0]
            for tf in ENABLED_TIMEFRAMES:
                df = await get_data_async(exchange=async_exchange, symbol=first_symbol, timeframe=tf)
                if df is not None:
                    reference_counts[tf] = len(df)

            # Otteniamo il saldo e le posizioni aperte
            usdt_balance = await get_real_balance(async_exchange)
            if usdt_balance is None:
                await asyncio.sleep(5)
                return
            open_positions_count = await get_open_positions(async_exchange)
            print(f"USDT Balance: {usdt_balance:.2f} | Open Positions: {open_positions_count}")

            # Recupera dati per tutti i simboli in parallelo per ogni timeframe
            dataframes_by_symbol = {}
            
            # Eseguiamo il recupero dati per tutti i timeframe in parallelo
            # Utilizziamo il numero configurato di simboli da analizzare per ciclo
            timeframe_tasks = []
            for tf in ENABLED_TIMEFRAMES:
                task = asyncio.create_task(fetch_data_for_multiple_symbols(
                    async_exchange, top_symbols_analysis[:SYMBOLS_PER_ANALYSIS_CYCLE], timeframe=tf
                ))
                timeframe_tasks.append((tf, task))
            
            # Attendiamo tutti i risultati
            for tf, task in timeframe_tasks:
                result_dict = await task
                for symbol, df in result_dict.items():
                    if df is not None:
                        if symbol not in dataframes_by_symbol:
                            dataframes_by_symbol[symbol] = {}
                        dataframes_by_symbol[symbol][tf] = df
            
            # Filtra i simboli che hanno dati validi per tutti i timeframes
            valid_symbols = []
            for symbol, dfs in dataframes_by_symbol.items():
                if all(tf in dfs for tf in ENABLED_TIMEFRAMES):
                    valid_count = all(len(dfs[tf]) >= reference_counts[tf] * 0.95 for tf in ENABLED_TIMEFRAMES)
                    if valid_count:
                        valid_symbols.append(symbol)
            
            print(f"Analisi di {len(valid_symbols)} simboli validi...")
            
            # Analizziamo i simboli validi in batch con maggiore parallelismo
            # Utilizziamo un numero di thread ancora maggiore (12)
            def analyze_with_models(symbol_data):
                symbol, dataframes = symbol_data
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
                return symbol, ensemble_value, final_signal, predictions
            
            # Eseguiamo l'analisi modelli in parallelo utilizzando thread
            signal_results = []
            with tqdm(total=len(valid_symbols), desc="Analisi segnali", ncols=80) as pbar:
                with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
                    futures = []
                    for symbol in valid_symbols:
                        futures.append(executor.submit(
                            analyze_with_models, (symbol, dataframes_by_symbol[symbol])
                        ))
                    
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            symbol, ensemble_value, final_signal, predictions = future.result()
                            if ensemble_value is not None and final_signal is not None:
                                signal_results.append({
                                    "symbol": symbol,
                                    "ensemble_value": ensemble_value,
                                    "signal": final_signal,
                                    "predictions": predictions
                                })
                                if final_signal == 1:
                                    predicted_buys.append(symbol)
                                elif final_signal == 0:
                                    predicted_sells.append(symbol)
                            pbar.update(1)
                        except Exception as e:
                            pbar.update(1)
            
            # Eseguiamo gli ordini in parallelo per i segnali validi
            # Aumentiamo la concorrenza a 10
            semaphore = asyncio.Semaphore(10)
            
            async def execute_order_with_semaphore(signal_data):
                async with semaphore:
                    symbol = signal_data["symbol"]
                    signal = signal_data["signal"]
                    try:
                        result = await manage_position(
                            async_exchange,
                            symbol,
                            signal,
                            usdt_balance,
                            min_amounts,
                            None, None, None, None,
                            dataframes_by_symbol[symbol][TIMEFRAME_DEFAULT]
                        )
                        return {"symbol": symbol, "signal": signal, "result": result}
                    except Exception as e:
                        return {"symbol": symbol, "signal": signal, "result": "error", "error": str(e)}
            
            # Eseguiamo gli ordini in parallelo
            order_tasks = []
            for signal_data in signal_results:
                task = asyncio.create_task(execute_order_with_semaphore(signal_data))
                order_tasks.append(task)
            
            with tqdm(total=len(order_tasks), desc="Esecuzione ordini", ncols=80) as pbar:
                for i, task in enumerate(asyncio.as_completed(order_tasks)):
                    result = await task
                    pbar.update(1)
            
            # Calcoliamo e mostriamo il tempo totale impiegato
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"Generati segnali: {len(signal_results)} (Buy: {len(predicted_buys)}, Sell: {len(predicted_sells)})")
            print(f"Completato in {elapsed:.2f} secondi")
            print("Fine ciclo. Bot in esecuzione.")

            await countdown_timer(TRADE_CYCLE_INTERVAL)

        except Exception as e:
            print(f"Error in trade cycle: {e}")
            await asyncio.sleep(60)

# --- Funzione Main ---
async def main():
    global async_exchange, lstm_models, lstm_scalers, rf_models, rf_scalers, xgb_models, xgb_scalers, min_amounts

    logging.info(colored("Avvio del programma...", "green"))
    init_data_tables()
    logging.info(colored("Inizializzazione delle tabelle di dati completata", "green"))

    logging.info(colored("Connessione all'exchange in corso...", "cyan"))
    async_exchange = ccxt_async.bybit(exchange_config)
    logging.info(colored("Caricamento dei mercati in corso...", "cyan"))
    await async_exchange.load_markets()
    logging.info(colored("Sincronizzazione orario in corso...", "cyan"))
    await async_exchange.load_time_difference()
    logging.info(colored("Connessione all'exchange completata", "green"))

    logging.info(colored("Caricamento delle posizioni esistenti...", "cyan"))
    await load_existing_positions(async_exchange)
    logging.info(colored("Caricamento posizioni completato", "green"))

    try:
        logging.info(colored("Recupero dei mercati disponibili...", "cyan"))
        markets = await fetch_markets(async_exchange)
        logging.info(colored(f"Recuperati {len(markets)} mercati", "cyan"))
        
        all_symbols = [m['symbol'] for m in markets.values() if m.get('quote') == 'USDT'
                       and m.get('active') and m.get('type') == 'swap']
        logging.info(colored(f"Trovati {len(all_symbols)} simboli USDT attivi", "cyan"))
        
        all_symbols_analysis = [s for s in all_symbols if not re.search('|'.join(EXCLUDED_SYMBOLS), s)]
        logging.info(colored(f"Filtrati {len(all_symbols_analysis)} simboli dopo esclusioni", "cyan"))

        logging.info(colored(f"Recupero top {TOP_ANALYSIS_CRYPTO} simboli per analisi...", "cyan"))
        top_symbols_analysis = await get_top_symbols(async_exchange, all_symbols_analysis, top_n=TOP_ANALYSIS_CRYPTO)
        
        logging.info(colored(f"Recupero top {TOP_TRAIN_CRYPTO} simboli per training...", "cyan"))
        top_symbols_training = await get_top_symbols(async_exchange, all_symbols, top_n=TOP_TRAIN_CRYPTO)

        # Validazione dei dati in parallelo
        logging.info(colored("Validazione dei dati prima del training...", "cyan"))
        
        valid_symbols = []
        for timeframe in ENABLED_TIMEFRAMES:
            logging.info(colored(f"Recupero dati in parallelo per timeframe {timeframe}...", "cyan"))
            # Utilizziamo la nuova funzione per recuperare i dati in parallelo
            result_dict = await fetch_data_for_multiple_symbols(
                async_exchange, 
                top_symbols_training[:SYMBOLS_FOR_VALIDATION],  # Utilizziamo il numero configurato per validazione
                timeframe=timeframe
            )
            
            # Filtriamo solo simboli con dati validi
            valid_for_tf = []
            for symbol, df in result_dict.items():
                if df is not None and not (df.isnull().any().any() or np.isinf(df).any().any()):
                    valid_for_tf.append(symbol)
                else:
                    logging.warning(f"Removing {symbol} from training set due to invalid data")
            
            # Se è il primo timeframe, inizializziamo la lista, altrimenti manteniamo solo quelli validi in tutti i timeframe
            if not valid_symbols:
                valid_symbols = valid_for_tf
            else:
                valid_symbols = [s for s in valid_symbols if s in valid_for_tf]
        
        # Aggiorniamo la lista di simboli validi per il training
        top_symbols_training = valid_symbols
        logging.info(f"{colored('Numero di monete per il training dopo validazione:', 'cyan')} {colored(str(len(top_symbols_training)), 'yellow')}")
        logging.info(f"{colored('Numero di monete per analisi operativa:', 'cyan')} {colored(str(len(top_symbols_analysis)), 'yellow')}")

        logging.info(colored("Recupero importi minimi per trading...", "cyan"))
        min_amounts = await fetch_min_amounts(async_exchange, top_symbols_analysis, markets)
        logging.info(colored("Importi minimi recuperati", "green"))

        # Ensure trained_models directory exists
        logging.info(colored("Verifica directory dei modelli...", "cyan"))
        ensure_trained_models_dir()
        
        # Initialize models and scalers
        logging.info(colored("Inizializzazione modelli e scalers...", "cyan"))
        
        # Precarichiamo tutti i modelli in memoria per massimizzare la velocità
        print("Precaricamento dei modelli per massimizzare le prestazioni...")
        lstm_models = {}
        lstm_scalers = {}
        rf_models = {}
        rf_scalers = {}
        xgb_models = {}
        xgb_scalers = {}
        
        # Caricamento parallelo dei modelli per tutti i timeframe
        model_load_tasks = []
        
        for tf in ENABLED_TIMEFRAMES:
            if 'lstm' in selected_models:
                model_load_tasks.append(('lstm', tf, asyncio.create_task(
                    asyncio.to_thread(load_lstm_model_func, tf))))
            
            if 'rf' in selected_models:
                model_load_tasks.append(('rf', tf, asyncio.create_task(
                    asyncio.to_thread(load_random_forest_model_func, tf))))
            
            if 'xgb' in selected_models:
                model_load_tasks.append(('xgb', tf, asyncio.create_task(
                    asyncio.to_thread(load_xgboost_model_func, tf))))
        
        # Utilizziamo una barra di progresso per il caricamento dei modelli
        with tqdm(total=len(model_load_tasks), desc="Caricamento modelli", ncols=80) as pbar:
            for model_type, tf, task in model_load_tasks:
                try:
                    result = await task
                    
                    if model_type == 'lstm':
                        lstm_models[tf], lstm_scalers[tf] = result
                        if not lstm_models[tf] and TRAIN_IF_NOT_FOUND:
                            print(f"Training LSTM model for {tf}...")
                            lstm_models[tf], lstm_scalers[tf], _ = await train_lstm_model_for_timeframe(
                                async_exchange, top_symbols_training, timeframe=tf, timestep=TIME_STEPS)
                    
                    elif model_type == 'rf':
                        rf_models[tf], rf_scalers[tf] = result
                        if not rf_models[tf] and TRAIN_IF_NOT_FOUND:
                            print(f"Training RF model for {tf}...")
                            rf_models[tf], rf_scalers[tf], _ = await train_random_forest_model_wrapper(
                                top_symbols_training, async_exchange, timestep=TIME_STEPS, timeframe=tf)
                    
                    elif model_type == 'xgb':
                        xgb_models[tf], xgb_scalers[tf] = result
                        if not xgb_models[tf] and TRAIN_IF_NOT_FOUND:
                            print(f"Training XGB model for {tf}...")
                            xgb_models[tf], xgb_scalers[tf], _ = await train_xgboost_model_wrapper(
                                top_symbols_training, async_exchange, timestep=TIME_STEPS, timeframe=tf)
                    
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"Error loading {model_type} model for {tf}: {e}")
                    pbar.update(1)
        
        # Verifica e addestramento di modelli mancanti
        for tf in ENABLED_TIMEFRAMES:
            # Stampa messaggio riepilogativo
            print(f"Modelli per {tf}: " + 
                  f"LSTM: {'✓' if 'lstm' in selected_models and lstm_models.get(tf) else '✗'}, " +
                  f"RF: {'✓' if 'rf' in selected_models and rf_models.get(tf) else '✗'}, " +
                  f"XGB: {'✓' if 'xgb' in selected_models and xgb_models.get(tf) else '✗'}")
        
        print("Modelli precaricati in memoria. Ottimizzazione completata.")

        logging.info(colored("Modelli caricati da disco o allenati per tutti i timeframe abilitati.", "magenta"))
        await load_existing_positions(async_exchange)

        trade_count = len(top_symbols_analysis)
        logging.info(f"{colored('Numero totale di trade stimati (basato sui simboli per analisi):', 'cyan')} {colored(str(trade_count), 'yellow')}")

        logging.info(colored("Avvio del ciclo principale di trading...", "green"))
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
