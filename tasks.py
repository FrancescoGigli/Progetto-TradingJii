import uuid
import threading
import asyncio
import logging
import os
import ccxt.async_support as ccxt_async
from typing import Dict, Any, List

# Importa le funzioni dai moduli esistenti
from model_manager import (
    ensure_trained_models_dir,
    train_lstm_model_for_timeframe,
    train_random_forest_model_wrapper,
    train_xgboost_model_wrapper,
    TIME_STEPS
)
from config import (
    API_KEY,
    API_SECRET,
    TOP_TRAIN_CRYPTO
)
from fetcher import (
    fetch_markets,
    get_top_symbols
)

# Registro globale per lo stato dei task
TASK_REGISTRY: Dict[str, Dict[str, Any]] = {}

class DummyTask:
    """Oggetto che imita il risultato di Celery."""
    def __init__(self, task_id: str):
        self.id = task_id

class TrainModelTask:
    """Gestore del training dei modelli."""

    @staticmethod
    def delay(model_type: str, timeframe: str, limit_days: int, top_train_crypto: int):
        task_id = str(uuid.uuid4())
        TASK_REGISTRY[task_id] = {
            "state": "PENDING",
            "result": None,
            "progress": 0,
            "current_step": "Inizializzazione"
        }

        def _runner():
            async def _async_train():
                try:
                    # Aggiorna stato a RUNNING
                    TASK_REGISTRY[task_id]["state"] = "RUNNING"
                    TASK_REGISTRY[task_id]["progress"] = 5
                    TASK_REGISTRY[task_id]["current_step"] = "Inizializzazione exchange"
                    
                    # Step 1: Configura l'exchange
                    exchange_config = {
                        'apiKey': API_KEY,
                        'secret': API_SECRET,
                        'enableRateLimit': True,
                    }
                    exchange = ccxt_async.bybit(exchange_config)
                    await exchange.load_markets()
                    
                    # Step 2: Ottieni i simboli più scambiati
                    TASK_REGISTRY[task_id]["progress"] = 10
                    TASK_REGISTRY[task_id]["current_step"] = "Recupero lista simboli"
                    
                    markets = await fetch_markets(exchange)
                    
                    # Aggiungi un ritardo per evitare troppe richieste simultanee
                    await asyncio.sleep(1)
                    
                    all_symbols = [m['symbol'] for m in markets.values() 
                                  if m.get('quote') == 'USDT' and m.get('active')]
                    
                    TASK_REGISTRY[task_id]["progress"] = 20
                    TASK_REGISTRY[task_id]["current_step"] = f"Recupero top {top_train_crypto} simboli"
                    
                    top_symbols = await get_top_symbols(exchange, all_symbols, top_n=TOP_TRAIN_CRYPTO)
                    
                    # Aggiungi un ritardo per evitare throttling
                    await asyncio.sleep(2)
                    
                    logging.info(f"Recuperati {len(top_symbols)} simboli per training: {', '.join(top_symbols[:5])}...")
                    
                    # Step 3: Esegui il training del modello
                    TASK_REGISTRY[task_id]["progress"] = 30
                    TASK_REGISTRY[task_id]["current_step"] = f"Avvio training {model_type.upper()} per {timeframe}"
                    
                    # Crea la directory per i modelli addestrati
                    ensure_trained_models_dir()
                    
                    # Avvia il training
                    if model_type == 'lstm':
                        TASK_REGISTRY[task_id]["current_step"] = f"Training LSTM su {len(top_symbols)} simboli"
                        model, scaler, metrics = await train_lstm_model_for_timeframe(
                            exchange, top_symbols, timeframe, TIME_STEPS)
                    elif model_type == 'rf':
                        TASK_REGISTRY[task_id]["current_step"] = f"Training Random Forest su {len(top_symbols)} simboli"
                        model, scaler, metrics = await train_random_forest_model_wrapper(
                            top_symbols, exchange, TIME_STEPS, timeframe)
                    elif model_type == 'xgb':
                        TASK_REGISTRY[task_id]["current_step"] = f"Training XGBoost su {len(top_symbols)} simboli"
                        model, scaler, metrics = await train_xgboost_model_wrapper(
                            top_symbols, exchange, TIME_STEPS, timeframe)
                    else:
                        raise ValueError(f"Tipo di modello non supportato: {model_type}")
                    
                    # Step 4: Chiudi la sessione dell'exchange
                    await exchange.close()
                    
                    # Aggiorna le metriche finali
                    val_metrics = metrics.get('val', {}) if metrics and isinstance(metrics, dict) else {}
                    
                    # Genera i risultati
                    result = {
                        "model_type": model_type,
                        "timeframe": timeframe,
                        "training_complete": True,
                        "model_saved": True if model else False,
                        "status": "completed",
                        "symbols_used": len(top_symbols),
                        "metrics": val_metrics
                    }
                    
                    TASK_REGISTRY[task_id]["progress"] = 100
                    TASK_REGISTRY[task_id]["state"] = "SUCCESS" 
                    TASK_REGISTRY[task_id]["result"] = result
                    TASK_REGISTRY[task_id]["current_step"] = "Training completato"
                    
                    logging.info(f"Training completato: {model_type} - {timeframe}")
                    
                except Exception as exc:
                    logging.error(f"Errore nel training {model_type}-{timeframe}: {exc}", exc_info=True)
                    TASK_REGISTRY[task_id]["state"] = "FAILURE"
                    TASK_REGISTRY[task_id]["result"] = str(exc)
            
            try:
                # Crea un nuovo event loop per ogni thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(_async_train())
            except Exception as e:
                logging.error(f"Errore nel thread di training: {e}")
                TASK_REGISTRY[task_id]["state"] = "FAILURE"
                TASK_REGISTRY[task_id]["result"] = str(e)
            finally:
                loop.close()

        # Avvia il thread per il training
        threading.Thread(target=_runner, daemon=True).start()
        return DummyTask(task_id)

# Esporta l'istanza utilizzata dall'applicazione
train_model_task = TrainModelTask() 