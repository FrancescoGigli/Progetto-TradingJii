# tasks.py
import time
import asyncio
import logging
from celery_worker import celery_app
import ccxt.async_support as ccxt_async
from trainer import (
    train_lstm_model_for_timeframe,
    train_random_forest_model_wrapper,
    train_xgboost_model_wrapper,
    ensure_trained_models_dir
)
from config import exchange_config, TIME_STEPS, TOP_TRAIN_CRYPTO
from fetcher import get_top_symbols, fetch_markets

@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def train_model_task(self, model_type, timeframe, data_limit_days=30, top_train_crypto=TOP_TRAIN_CRYPTO):
    """
    Task Celery per eseguire il training del modello.
    model_type: 'lstm', 'rf' o 'xgb'
    timeframe: ad es. '15m'
    data_limit_days e top_train_crypto possono avere valori predefiniti.
    """
    start_time = time.time()
    try:
        # Imposta la policy dell'event loop per Windows, se necessario
        if hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Crea l'exchange e carica i mercati
        exchange = ccxt_async.bybit(exchange_config)
        loop.run_until_complete(exchange.load_markets())
        loop.run_until_complete(exchange.load_time_difference())
        
        # Recupera i mercati disponibili
        markets = loop.run_until_complete(fetch_markets(exchange))
        
        # Filtra i simboli per ottenere solo quelli USDT
        all_symbols = [m['symbol'] for m in markets.values() if m.get('quote') == 'USDT'
                       and m.get('active') and m.get('type') == 'swap']
        
        # Ottieni i simboli con il maggior volume
        top_symbols = loop.run_until_complete(get_top_symbols(exchange, all_symbols, top_n=top_train_crypto))
        
        # Determina la directory dove salvare i modelli (se necessario)
        trained_models_dir = ensure_trained_models_dir()
        
        # Esegui il training in base al tipo di modello richiesto
        if model_type == 'lstm':
            model, scaler, metrics = loop.run_until_complete(
                train_lstm_model_for_timeframe(exchange, top_symbols, timeframe, TIME_STEPS)
            )
        elif model_type == 'rf':
            model, scaler, metrics = loop.run_until_complete(
                train_random_forest_model_wrapper(top_symbols, exchange, timestep=TIME_STEPS, timeframe=timeframe)
            )
        elif model_type == 'xgb':
            model, scaler, metrics = loop.run_until_complete(
                train_xgboost_model_wrapper(top_symbols, exchange, timestep=TIME_STEPS, timeframe=timeframe)
            )
        else:
            raise ValueError(f"Tipo di modello non supportato: {model_type}")
        
        # Chiudi l'exchange e calcola il tempo impiegato
        loop.run_until_complete(exchange.close())
        duration = time.time() - start_time
        result = {
            "status": "completed",
            "model_type": model_type,
            "timeframe": timeframe,
            "metrics": metrics,
            "training_time_sec": duration,
            "symbols_used": top_symbols
        }
        logging.info("Training completato: %s", result)
        return result
    except Exception as exc:
        logging.error("Errore nel training: %s", exc)
        try:
            self.retry(exc=exc)
        except Exception as retry_exc:
            logging.error("Retry fallito: %s", retry_exc)
            raise exc
