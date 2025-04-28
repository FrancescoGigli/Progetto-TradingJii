# model_manager.py
# Gestione completa dei modelli di machine learning: definizione, training e caricamento

import os
import json
import logging
import asyncio
import numpy as np
import joblib
import xgboost as xgb
import tensorflow as tf
from datetime import datetime, timedelta
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from keras.losses import Loss
from keras import backend as K
from keras.callbacks import EarlyStopping, TensorBoard, Callback
from keras_tuner import HyperModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from celery import shared_task

from config import (
    TIME_STEPS, EXPECTED_COLUMNS,
    get_lstm_model_file, get_lstm_scaler_file,
    get_rf_model_file, get_rf_scaler_file,
    get_xgb_model_file, get_xgb_scaler_file
)
from fetcher import fetch_markets, get_top_symbols, get_data_async
from data_utils import prepare_data

# === Helpers ===
def ensure_trained_models_dir():
    """Create the trained_models directory if it doesn't exist."""
    d = os.path.join(os.path.dirname(__file__), 'trained_models')
    os.makedirs(d, exist_ok=True)
    return d

def save_training_history_plot(history, timeframe):
    """Save loss/accuracy plots to logs/plots."""
    plot_dir = os.path.join("logs", "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plt.figure(figsize=(12,5))
    # Loss
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    # Accuracy
    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'training_history_{timeframe}.png'))
    plt.close()

def augment_jitter(X, sigma=0.03):
    """Aggiunge rumore gaussiano ai dati per aumentare il dataset."""
    noise = np.random.normal(0, sigma, X.shape)
    return X + noise

# === Custom Loss ===
class FocalLoss(Loss):
    def __init__(self, gamma=2., alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        eps = K.epsilon()
        y_pred = K.clip(y_pred, eps, 1. - eps)
        ce = -y_true * K.log(y_pred) - (1. - y_true) * K.log(1. - y_pred)
        weight = (
            self.alpha * y_true * K.pow((1. - y_pred), self.gamma) +
            (1. - self.alpha) * (1. - y_true) * K.pow(y_pred, self.gamma)
        )
        return K.mean(weight * ce)

# === Model Builders ===
def create_lstm_model(input_shape):
    """Crea un modello LSTM bidirezionale."""
    model = Sequential([
        Input(shape=input_shape),
        Bidirectional(LSTM(100, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(100)),
        Dropout(0.3),
        Dense(50, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss=FocalLoss(), metrics=['accuracy'])
    return model

class LSTMHyperModel(HyperModel):
    """Modello iperparametrizzato per LSTM."""
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        model = Sequential()
        model.add(Input(shape=self.input_shape))
        
        # Primo layer LSTM bidirezionale
        units_1 = hp.Int('units_1', min_value=50, max_value=200, step=50)
        model.add(Bidirectional(LSTM(units_1, return_sequences=True)))
        model.add(Dropout(hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)))
        
        # Secondo layer LSTM bidirezionale
        units_2 = hp.Int('units_2', min_value=50, max_value=200, step=50)
        model.add(Bidirectional(LSTM(units_2)))
        model.add(Dropout(hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)))
        
        # Layer densi
        model.add(Dense(hp.Int('dense_1', min_value=25, max_value=100, step=25), activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(
            optimizer='adam',
            loss=FocalLoss(),
            metrics=['accuracy']
        )
        return model

def create_rf_model():
    """Crea un modello Random Forest."""
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )

# === Model Training ===
def train_rf_sync(X, y):
    """Addestra un modello Random Forest in modo sincrono."""
    model = create_rf_model()
    
    # Calcola i pesi delle classi
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y),
        y=y
    )
    class_weight_dict = dict(zip(np.unique(y), class_weights))
    
    # Addestra il modello
    model.fit(X, y, class_weight=class_weight_dict)
    
    return model

def train_xgb_model(X, y):
    """Addestra un modello XGBoost."""
    # Configurazione del modello
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'scale_pos_weight': sum(y == 0) / sum(y == 1)  # Bilancia le classi
    }
    
    # Crea il dataset DMatrix
    dtrain = xgb.DMatrix(X, label=y)
    
    # Addestra il modello
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=[(dtrain, 'train')],
        verbose_eval=False
    )
    
    return model

async def train_lstm_model_for_timeframe(exchange, symbols, timeframe, timestep):
    """Addestra un modello LSTM per un timeframe specifico."""
    logging.info(f"Training LSTM model for timeframe {timeframe}")
    
    # Raccogli i dati di training
    all_data = []
    all_labels = []
    
    for symbol in symbols:
        try:
            df = await get_data_async(exchange, symbol, timeframe=timeframe)
            if df is None or len(df) < timestep + 1:
                continue
                
            # Prepara i dati
            data = prepare_data(df)
            
            # Crea le sequenze
            for i in range(len(data) - timestep):
                X = data[i:i+timestep]
                y = 1 if data[i+timestep, 3] > data[i+timestep-1, 3] else 0  # 1 se il prezzo è salito
                all_data.append(X)
                all_labels.append(y)
                
        except Exception as e:
            logging.error(f"Error processing {symbol}: {e}")
    
    if not all_data:
        logging.error(f"No data collected for timeframe {timeframe}")
        return None, None, None
    
    # Converti in array numpy
    X = np.array(all_data)
    y = np.array(all_labels)
    
    # Crea e addestra il modello
    model = create_lstm_model(input_shape=(timestep, X.shape[2]))
    
    # Calcola i pesi delle classi
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y),
        y=y
    )
    class_weight_dict = dict(zip(np.unique(y), class_weights))
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        TensorBoard(log_dir=f'logs/lstm_{timeframe}')
    ]
    
    # Addestra il modello
    history = model.fit(
        X, y,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    # Salva il grafico della storia di training
    save_training_history_plot(history, timeframe)
    
    # Crea e addestra lo scaler
    scaler = StandardScaler()
    scaler.fit(X.reshape(-1, X.shape[2]))
    
    # Salva il modello e lo scaler
    ensure_trained_models_dir()
    model.save(get_lstm_model_file(timeframe))
    joblib.dump(scaler, get_lstm_scaler_file(timeframe))
    
    logging.info(f"LSTM model trained and saved for timeframe {timeframe}")
    return model, scaler, history

async def train_random_forest_model_wrapper(symbols, exchange, timestep, timeframe):
    """Wrapper per l'addestramento del modello Random Forest."""
    logging.info(f"Training Random Forest model for timeframe {timeframe}")
    
    # Raccogli i dati di training
    all_data = []
    all_labels = []
    
    for symbol in symbols:
        try:
            df = await get_data_async(exchange, symbol, timeframe=timeframe)
            if df is None or len(df) < timestep + 1:
                continue
                
            # Prepara i dati
            data = prepare_data(df)
            
            # Crea le sequenze
            for i in range(len(data) - timestep):
                X = data[i:i+timestep].flatten()  # Appiattisci per RF
                y = 1 if data[i+timestep, 3] > data[i+timestep-1, 3] else 0  # 1 se il prezzo è salito
                all_data.append(X)
                all_labels.append(y)
                
        except Exception as e:
            logging.error(f"Error processing {symbol}: {e}")
    
    if not all_data:
        logging.error(f"No data collected for timeframe {timeframe}")
        return None, None, None
    
    # Converti in array numpy
    X = np.array(all_data)
    y = np.array(all_labels)
    
    # Addestra il modello
    model = train_rf_sync(X, y)
    
    # Crea e addestra lo scaler
    scaler = StandardScaler()
    scaler.fit(X)
    
    # Salva il modello e lo scaler
    ensure_trained_models_dir()
    joblib.dump(model, get_rf_model_file(timeframe))
    joblib.dump(scaler, get_rf_scaler_file(timeframe))
    
    logging.info(f"Random Forest model trained and saved for timeframe {timeframe}")
    return model, scaler, None

async def train_xgboost_model_wrapper(symbols, exchange, timestep, timeframe):
    """Wrapper per l'addestramento del modello XGBoost."""
    logging.info(f"Training XGBoost model for timeframe {timeframe}")
    
    # Raccogli i dati di training
    all_data = []
    all_labels = []
    
    for symbol in symbols:
        try:
            df = await get_data_async(exchange, symbol, timeframe=timeframe)
            if df is None or len(df) < timestep + 1:
                continue
                
            # Prepara i dati
            data = prepare_data(df)
            
            # Crea le sequenze
            for i in range(len(data) - timestep):
                X = data[i:i+timestep].flatten()  # Appiattisci per XGB
                y = 1 if data[i+timestep, 3] > data[i+timestep-1, 3] else 0  # 1 se il prezzo è salito
                all_data.append(X)
                all_labels.append(y)
                
        except Exception as e:
            logging.error(f"Error processing {symbol}: {e}")
    
    if not all_data:
        logging.error(f"No data collected for timeframe {timeframe}")
        return None, None, None
    
    # Converti in array numpy
    X = np.array(all_data)
    y = np.array(all_labels)
    
    # Addestra il modello
    model = train_xgb_model(X, y)
    
    # Crea e addestra lo scaler
    scaler = StandardScaler()
    scaler.fit(X)
    
    # Salva il modello e lo scaler
    ensure_trained_models_dir()
    model.save_model(get_xgb_model_file(timeframe))
    joblib.dump(scaler, get_xgb_scaler_file(timeframe))
    
    logging.info(f"XGBoost model trained and saved for timeframe {timeframe}")
    return model, scaler, None

# === Model Loading ===
def load_lstm_model(timeframe: str):
    """Carica un modello LSTM e il suo scaler."""
    model_path = get_lstm_model_file(timeframe)
    scaler_path = get_lstm_scaler_file(timeframe)
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = load_model(model_path, custom_objects={'FocalLoss': FocalLoss})
        scaler = joblib.load(scaler_path)
        logging.info(f"LSTM loaded for {timeframe}")
        return model, scaler
    logging.warning(f"LSTM or scaler missing for {timeframe}")
    return None, None

def load_rf_model(timeframe: str):
    """Carica un modello Random Forest e il suo scaler."""
    model_path = get_rf_model_file(timeframe)
    scaler_path = get_rf_scaler_file(timeframe)
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        logging.info(f"RF loaded for {timeframe}")
        return model, scaler
    logging.warning(f"RF or scaler missing for {timeframe}")
    return None, None

def load_xgb_model(timeframe: str):
    """Carica un modello XGBoost e il suo scaler."""
    model_path = get_xgb_model_file(timeframe)
    scaler_path = get_xgb_scaler_file(timeframe)
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = xgb.Booster()
        model.load_model(model_path)
        scaler = joblib.load(scaler_path)
        logging.info(f"XGB loaded for {timeframe}")
        return model, scaler
    logging.warning(f"XGB or scaler missing for {timeframe}")
    return None, None

# === Celery Tasks ===
@shared_task(bind=True)
def train_model_task(self, model_type, timeframe, data_limit_days, top_train_crypto):
    """Task Celery per l'addestramento dei modelli."""
    async def _do_train():
        exchange = None  # Inizializza l'exchange qui
        symbols = []  # Ottieni i simboli qui
        
        if model_type == 'lstm':
            return await train_lstm_model_for_timeframe(exchange, symbols, timeframe, TIME_STEPS)
        elif model_type == 'rf':
            return await train_random_forest_model_wrapper(symbols, exchange, TIME_STEPS, timeframe)
        elif model_type == 'xgb':
            return await train_xgboost_model_wrapper(symbols, exchange, TIME_STEPS, timeframe)
        else:
            logging.error(f"Model type {model_type} not supported")
            return None, None, None
    
    # Esegui la funzione asincrona
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(_do_train()) 