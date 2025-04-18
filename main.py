#!/usr/bin/env python3
import sys
import os
import numpy as np
from datetime import timedelta
import asyncio
import logging
import re
import ccxt
import ccxt.async_support as ccxt_async
from termcolor import colored
from tqdm import tqdm  # Import per la progress bar
import time
import json
import signal
import platform
import traceback
import ta

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
    ENABLED_TIMEFRAMES, TIMEFRAME_DEFAULT, get_lstm_model_file, LEVERAGE, MARGIN_USDT  # Importazione dei timeframe predefiniti da config
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

            logging.info(colored("Inizio ciclo di trading", "cyan"))
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

            logging.info(colored("Fine ciclo di trading", "cyan"))
            logging.info(colored("Bot is running", "green"))

            await countdown_timer(TRADE_CYCLE_INTERVAL)

        except Exception as e:
            logging.error(f"{colored('Error in trade cycle:', 'red')} {e}")
            await asyncio.sleep(60)

# --- Funzione Main ---
async def main():
    """
    Funzione principale che gestisce l'esecuzione del bot.
    """
    start_time = time.time()
    try:
        print("Inizializzazione bot...")
        logging.info("Inizializzazione bot...")
        
        # Accesso all'exchange tramite la libreria CCXT
        exchange = ccxt.bybit(exchange_config)
        # Usa la versione asincrona per operazioni parallele
        async_exchange = ccxt_async.bybit(exchange_config)
        
        # Log di configurazione del bot
        logging.info(f"Bot configurato per: Modelli={selected_models}, Timeframes={ENABLED_TIMEFRAMES}")
        logging.info(f"Parametri trading: Leva={LEVERAGE}x, Margine={MARGIN_USDT} USDT")
        
        # Inizializza gli ultimi timestamp per ogni timeframe
        last_process_time = {}
        for tf in ENABLED_TIMEFRAMES:
            last_process_time[tf] = time.time()
        
        # Connessione all'exchange
        try:
            await async_exchange.load_markets()
            await async_exchange.load_time_difference()
            logging.info(f"Connessione a {exchange.name} stabilita con successo")
        except Exception as e:
            logging.error(f"Errore durante la connessione a {exchange.name}: {e}")
            raise
        
        # Carica i mercati
        markets = await fetch_markets(async_exchange)
        
        # Filtriamo solo i mercati USDT e derivati
        usdt_markets = [
            symbol for symbol in markets 
            if symbol.endswith(':USDT') and not any(excluded in symbol for excluded in EXCLUDED_SYMBOLS)
        ]
        
        logging.info(f"Trovati {len(usdt_markets)} mercati USDT validi")
        
        # Ottieni i top symbols per volume
        # Correggo l'ordine dei parametri: prima l'exchange, poi la lista di simboli
        top_symbols = await get_top_symbols(async_exchange, usdt_markets, TOP_ANALYSIS_CRYPTO)
        
        # Ottieni i minimi importi per ogni simbolo
        min_amounts = await fetch_min_amounts(async_exchange, top_symbols, markets)
        
        # Log di avvio con configurazione
        logging.info(f"Bot avviato con {len(top_symbols)}/{len(usdt_markets)} simboli")
        
        # Controllo di validità dei modelli selezionati
        valid_models = validate_models()
        if not valid_models:
            logging.error("Nessun modello valido selezionato. Il bot non può continuare.")
            return
        
        # Inizializza dizionari per i modelli e gli scaler
        models = {model_type: {} for model_type in selected_models}
        scalers = {model_type: {} for model_type in selected_models}
        
        # Carica tutti i modelli e gli scaler all'inizio
        for model_type in selected_models:
            for timeframe in ENABLED_TIMEFRAMES:
                try:
                    model, scaler = await load_model(model_type, timeframe)
                    if model and scaler:
                        models[model_type][timeframe] = model
                        scalers[model_type][timeframe] = scaler
                        logging.info(f"Modello {model_type} per timeframe {timeframe} caricato con successo")
                    else:
                        logging.warning(f"Modello {model_type} per timeframe {timeframe} non disponibile")
                except Exception as e:
                    logging.error(f"Errore nel caricamento del modello {model_type} per timeframe {timeframe}: {e}")
        
        # Controllo di validità dei modelli caricati
        model_count = sum(len(models[model_type]) for model_type in models)
        if model_count == 0:
            logging.error("Nessun modello caricato correttamente. Il bot non può continuare.")
            return
        
        # Inizia il loop principale
        running = True
        iteration = 0
        
        while running:
            iteration += 1
            loop_start_time = time.time()
            
            try:
                # Controlla se bisogna processare ciascun timeframe
                for timeframe in ENABLED_TIMEFRAMES:
                    # Determina l'intervallo di tempo per ogni timeframe
                    interval_seconds = timeframe_to_seconds(timeframe)
                    elapsed = time.time() - last_process_time[timeframe]
                    
                    # Se è passato abbastanza tempo, processa questo timeframe
                    # Usiamo un intervallo ridotto per assicurarci di processare ogni candle
                    if elapsed >= interval_seconds * 0.8:
                        logging.info(f"Elaborazione {timeframe} (iter. {iteration})")
                        
                        # Aggiorna il timestamp dell'ultimo processo
                        last_process_time[timeframe] = time.time()
                        
                        # Processa tutti i simboli per questo timeframe
                        for symbol in top_symbols:
                            try:
                                # Ottieni i dati per questo simbolo e timeframe
                                data = await get_data_for_symbol(async_exchange, symbol, timeframe)
                                if data is None or len(data) < TIME_STEPS * 2:
                                    continue
                                
                                # Calcola RSI
                                rsi = ta.momentum.RSIIndicator(data['close']).rsi()
                                rsi_value = float(rsi.iloc[-1]) if not rsi.empty else 50.0
                                
                                # Predizioni per ogni modello per questo simbolo e timeframe
                                predictions = {}
                                
                                # Ottieni predizioni da ciascun modello
                                for model_type in selected_models:
                                    if timeframe in models[model_type] and timeframe in scalers[model_type]:
                                        # Ottieni predizione con quel modello
                                        prediction = await predict_with_model(
                                            models[model_type][timeframe],
                                            scalers[model_type][timeframe],
                                            data,
                                            model_type
                                        )
                                        predictions[model_type] = prediction
                                
                                # Calcola e registra il segnale aggregato solo se abbiamo predizioni
                                if predictions:
                                    # Logica di trading basata sulle predizioni e sul RSI
                                    signal = await process_trading_signal(
                                        symbol, timeframe, predictions, rsi_value,
                                        async_exchange, min_amounts.get(symbol, 0.1)
                                    )
                                    
                                    if signal:
                                        logging.info(f"Segnale generato per {symbol} [{timeframe}]: {signal}")
                            except Exception as symbol_error:
                                logging.error(f"Errore nell'elaborazione di {symbol} [{timeframe}]: {symbol_error}")
                                # Continuiamo con il prossimo simbolo, non interrompiamo il loop
                
                # Rallenta il loop in base al numero di timeframe
                sleep_time = max(5, 60 / len(ENABLED_TIMEFRAMES))
                await asyncio.sleep(sleep_time)
                
                # Log di stato ad ogni 5 iterazioni
                if iteration % 5 == 0:
                    elapsed_time = time.time() - start_time
                    hours, remainder = divmod(elapsed_time, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    logging.info(f"Bot in esecuzione da {int(hours)}h {int(minutes)}m {int(seconds)}s - Iterazione {iteration}")
                
                # Controlla se il loop è durato troppo a lungo
                loop_duration = time.time() - loop_start_time
                if loop_duration > 120:  # Se il loop ha impiegato più di 2 minuti
                    logging.warning(f"Loop iter. {iteration} ha impiegato {loop_duration:.1f}s, possibile sovraccarico")
            
            except asyncio.CancelledError:
                logging.info("Bot task cancellata")
                running = False
                break
            except Exception as e:
                logging.error(f"Errore nell'iterazione {iteration}: {e}")
                logging.error(traceback.format_exc())
                # Continuiamo l'esecuzione dopo un errore, non interrompiamo il bot
                await asyncio.sleep(10)  # Breve pausa prima di riprovare
        
        # Pulizia finale
        logging.info("Chiusura connessione all'exchange...")
        await async_exchange.close()
        logging.info("Bot terminato correttamente")
        
    except asyncio.CancelledError:
        logging.info("Il bot è stato cancellato dall'esterno")
        # Propaghiamo l'eccezione per segnalare la cancellazione
        raise
    except Exception as e:
        logging.error(f"Errore fatale nell'esecuzione del bot: {e}")
        logging.error(traceback.format_exc())
        # Assicuriamo che la funzione termini correttamente nonostante l'errore
    finally:
        # Chiusura e pulizia
        try:
            if 'async_exchange' in locals() and async_exchange:
                await async_exchange.close()
                logging.info("Connessione all'exchange chiusa")
        except Exception as close_error:
            logging.error(f"Errore nella chiusura dell'exchange: {close_error}")
            
    # Calcola il tempo totale di esecuzione
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logging.info(f"Tempo totale di esecuzione: {int(hours)}h {int(minutes)}m {int(seconds)}s")

# Funzioni che potrebbero mancare
async def get_data_for_symbol(exchange, symbol, timeframe):
    """
    Ottiene e prepara i dati per un simbolo e timeframe specifici.
    """
    try:
        # Ottieni i dati OHLCV
        limit = TIME_STEPS * 3  # Ottieni più dati del necessario per calcolare indicatori
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        if not ohlcv or len(ohlcv) < TIME_STEPS * 2:
            logging.warning(f"Dati insufficienti per {symbol} sul timeframe {timeframe}")
            return None
            
        # Converti in DataFrame
        import pandas as pd
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Calcola indicatori tecnici aggiuntivi
        # RSI
        df['rsi_fast'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'])
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_middle'] = bollinger.bollinger_mavg()
        df['bb_lower'] = bollinger.bollinger_lband()
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Media mobile esponenziale
        df['ema_9'] = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
        df['ema_21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
        
        # Rimuovi righe con NaN dopo aver aggiunto gli indicatori
        df.dropna(inplace=True)
        
        return df
    except Exception as e:
        logging.error(f"Errore nel recupero/preparazione dati per {symbol} [{timeframe}]: {e}")
        return None

async def process_trading_signal(symbol, timeframe, predictions, rsi_value, exchange, min_amount):
    """
    Elabora le predizioni dei vari modelli e determina se generare un segnale di trading.
    Ritorna il segnale generato (se presente).
    """
    try:
        # Calcola il valore medio delle predizioni
        prediction_values = list(predictions.values())
        if not prediction_values:
            return None
            
        avg_prediction = sum(prediction_values) / len(prediction_values)
        
        # Definisci le soglie per i segnali
        buy_threshold = 0.65  # Forte segnale di acquisto
        weak_buy = 0.55       # Segnale debole di acquisto
        sell_threshold = 0.35 # Forte segnale di vendita
        weak_sell = 0.45      # Segnale debole di vendita
        
        # Incorpora anche il RSI nella decisione
        rsi_buy_signal = rsi_value < 30  # Condizione di ipervenduto
        rsi_sell_signal = rsi_value > 70 # Condizione di ipercomprato
        
        # Logica di decisione combinata
        if avg_prediction >= buy_threshold or (avg_prediction >= weak_buy and rsi_buy_signal):
            # Genera segnale di acquisto
            return {
                "signal": "buy",
                "symbol": symbol,
                "timeframe": timeframe,
                "confidence": avg_prediction,
                "rsi": rsi_value
            }
        elif avg_prediction <= sell_threshold or (avg_prediction <= weak_sell and rsi_sell_signal):
            # Genera segnale di vendita
            return {
                "signal": "sell",
                "symbol": symbol,
                "timeframe": timeframe,
                "confidence": 1 - avg_prediction,
                "rsi": rsi_value
            }
            
        # Nessun segnale chiaro
        return None
        
    except Exception as e:
        logging.error(f"Errore nell'elaborazione del segnale per {symbol} [{timeframe}]: {e}")
        return None
        
# Funzione mancante per caricare i modelli
async def load_model(model_type, timeframe):
    """
    Carica un modello e il relativo scaler.
    """
    try:
        if model_type == 'lstm':
            return await asyncio.to_thread(load_lstm_model_func, timeframe)
        elif model_type == 'rf':
            return await asyncio.to_thread(load_random_forest_model_func, timeframe)
        elif model_type == 'xgb':
            return await asyncio.to_thread(load_xgboost_model_func, timeframe)
        else:
            logging.error(f"Tipo di modello non supportato: {model_type}")
            return None, None
    except Exception as e:
        logging.error(f"Errore nel caricamento del modello {model_type} per {timeframe}: {e}")
        return None, None

async def predict_with_model(model, scaler, data, model_type):
    """
    Effettua una predizione con un modello specifico.
    """
    try:
        from predictor import predict_signal_for_model
        return predict_signal_for_model(data, model, scaler, data.index[0], TIME_STEPS, timeframe=data.index.name)
    except Exception as e:
        logging.error(f"Errore nella predizione con {model_type}: {e}")
        return 0.5  # Valore neutro in caso di errore

def validate_models():
    """
    Verifica che i modelli selezionati siano validi.
    """
    valid_model_types = ['lstm', 'rf', 'xgb']
    valid = [model for model in selected_models if model in valid_model_types]
    return len(valid) > 0

def timeframe_to_seconds(timeframe):
    """
    Converte un timeframe in secondi.
    """
    unit = timeframe[-1]
    value = int(timeframe[:-1])
    
    if unit == 'm':
        return value * 60
    elif unit == 'h':
        return value * 3600
    elif unit == 'd':
        return value * 86400
    else:
        return 3600  # Default: 1 ora

if __name__ == "__main__":
    asyncio.run(main())
