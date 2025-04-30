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
from termcolor import colored

from config import (
    TIME_STEPS, EXPECTED_COLUMNS,
    get_lstm_model_file, get_lstm_scaler_file,
    get_rf_model_file, get_rf_scaler_file,
    get_xgb_model_file, get_xgb_scaler_file
)
from fetcher import fetch_markets, get_top_symbols, get_data_async
from data_utils import prepare_data

# Configurazione per l'utilizzo della GPU e ottimizzazione TensorFlow
def configure_tensorflow():
    """Configura TensorFlow per le prestazioni ottimali."""
    try:
        # Disabilita i messaggi di avviso di oneDNN
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Silenzia i messaggi di info e warning
        
        # Verifica se la GPU è disponibile
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            logging.info(f"GPU disponibile: {len(gpus)} device trovati")
            for gpu in gpus:
                # Abilita la memoria dinamica su GPU
                tf.config.experimental.set_memory_growth(gpu, True)
            logging.info("GPU configurata per il training")
        else:
            logging.info("Nessuna GPU rilevata, utilizzando CPU")

        # Ottimizzazioni TensorFlow
        tf.config.threading.set_inter_op_parallelism_threads(4)
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.optimizer.set_jit(True)  # Abilita XLA
        
        # Disattiva i warning di AutoGraph
        tf.autograph.set_verbosity(0)
        
        return True
    except Exception as e:
        logging.warning(f"Errore nella configurazione di TensorFlow: {str(e)}")
        return False

# Configura TensorFlow all'avvio del modulo
configure_tensorflow()

# === Helpers ===
def ensure_trained_models_dir():
    """Create the trained_models directory if it doesn't exist."""
    d = os.path.join(os.path.dirname(__file__), 'trained_models')
    os.makedirs(d, exist_ok=True)
    return d

def save_training_history_plot(history, timeframe):
    """Save loss/accuracy plots to logs/plots."""
    # Crea le directory se non esistono
    plot_dir = os.path.join("logs", "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    # Calcola metriche aggiuntive
    epochs = len(history.history['loss'])
    best_epoch = np.argmin(history.history['val_loss'])
    best_val_loss = min(history.history['val_loss'])
    best_val_acc = max(history.history['val_accuracy'])
    train_loss_at_best = history.history['loss'][best_epoch]
    train_acc_at_best = history.history['accuracy'][best_epoch]
    
    last_train_loss = history.history['loss'][-1]
    last_train_acc = history.history['accuracy'][-1]
    last_val_loss = history.history['val_loss'][-1]
    last_val_acc = history.history['val_accuracy'][-1]
    
    # Calcola overfitting/underfitting
    loss_diff = last_train_loss - last_val_loss
    acc_diff = last_train_acc - last_val_acc
    
    # Determina condizione del modello
    model_condition = "balanced"
    if last_train_acc > 0.95 and last_val_acc < 0.7:
        model_condition = "overfit"
    elif last_train_acc < 0.7 and last_val_acc < 0.7:
        model_condition = "underfit"
    
    # Salva le metriche dettagliate in formato JSON
    metrics_file = os.path.join(plot_dir, f'training_metrics_{timeframe}.json')
    try:
        detailed_metrics = {
            # Dati grezzi per ogni epoca
            "raw_metrics": {
                "loss": history.history['loss'],
                "val_loss": history.history['val_loss'],
                "accuracy": history.history['accuracy'],
                "val_accuracy": history.history['val_accuracy']
            },
            # Metriche riassuntive
            "summary": {
                "epochs_trained": epochs,
                "best_epoch": int(best_epoch),
                "training_time": None,  # Sarebbe meglio tracciare il tempo di training
                "model_condition": model_condition
            },
            # Performance sul set di training
            "training": {
                "final_loss": float(last_train_loss),
                "final_accuracy": float(last_train_acc),
                "best_loss": float(min(history.history['loss'])),
                "best_accuracy": float(max(history.history['accuracy']))
            },
            # Performance sul set di validazione
            "validation": {
                "final_loss": float(last_val_loss),
                "final_accuracy": float(last_val_acc),
                "best_loss": float(best_val_loss),
                "best_accuracy": float(best_val_acc)
            },
            # Metriche alla migliore epoca
            "best_epoch_metrics": {
                "training_loss": float(train_loss_at_best),
                "training_accuracy": float(train_acc_at_best),
                "validation_loss": float(best_val_loss),
                "validation_accuracy": float(history.history['val_accuracy'][best_epoch])
            },
            # Analisi overfitting/underfitting
            "model_analysis": {
                "train_val_loss_gap": float(loss_diff),
                "train_val_accuracy_gap": float(acc_diff),
                "recommendations": get_model_recommendations(model_condition)
            },
            # Metadati
            "metadata": {
                "timeframe": timeframe,
                "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(detailed_metrics, f, indent=4)
        print(f"Metriche dettagliate salvate in {metrics_file}")
    except Exception as e:
        print(f"Errore nel salvataggio delle metriche dettagliate: {e}")
    
    # Crea e salva i grafici
    try:
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
        
        # Salva il grafico
        plot_file = os.path.join(plot_dir, f'training_history_{timeframe}.png')
        plt.savefig(plot_file)
        plt.close()
        print(f"Grafico salvato in {plot_file}")
        
    except Exception as e:
        print(f"Errore nella creazione del grafico: {e}")
        # In caso di errore con matplotlib, almeno salviamo i dati delle metriche
        print("Loss finale:")
        print(f"  Training: {history.history['loss'][-1]:.4f}")
        print(f"  Validation: {history.history['val_loss'][-1]:.4f}")
        print("Accuracy finale:")
        print(f"  Training: {history.history['accuracy'][-1]:.4f}")
        print(f"  Validation: {history.history['val_accuracy'][-1]:.4f}")

def get_model_recommendations(condition):
    """Restituisce raccomandazioni in base alla condizione del modello."""
    if condition == "overfit":
        return [
            "Aumentare il dropout",
            "Aggiungere regolarizzazione L1/L2",
            "Ridurre la complessità del modello",
            "Aumentare i dati di training",
            "Utilizzare data augmentation"
        ]
    elif condition == "underfit":
        return [
            "Aumentare la complessità del modello",
            "Ridurre il dropout",
            "Aumentare il numero di epoche",
            "Utilizzare un learning rate più alto"
        ]
    else:  # balanced
        return [
            "Il modello sembra ben bilanciato",
            "Si può provare ad aumentare leggermente la complessità per migliorare",
            "Considerare tecniche di ensemble per risultati migliori"
        ]

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
        
        # Converti esplicitamente y_true in float32
        y_true = K.cast(y_true, 'float32')
        
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
    start_time = datetime.now()
    
    # Raccogli i dati di training
    all_data = []
    all_labels = []
    
    for symbol in symbols:
        try:
            df = await get_data_async(exchange, symbol, timeframe=timeframe)
            if df is None or len(df) < timestep + 1:
                continue
                
            # Verifica se il DataFrame contiene una colonna timestamp
            if 'timestamp' in df.columns:
                # Rimuovi la colonna timestamp prima di preparare i dati
                df_temp = df.drop(columns=['timestamp'], errors='ignore')
            else:
                df_temp = df
                
            # Prepara i dati
            data = prepare_data(df_temp)
            
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
    
    # Log delle dimensioni del dataset per debug
    logging.info(f"Dataset dimensioni: {X.shape}, etichette: {y.shape}")
    
    # Determina batch_size ottimale in base alla dimensione del dataset
    # Batch più grandi sono più efficienti per GPU e per training parallelo
    batch_size = min(128, X.shape[0] // 10)  # max 128 o 1/10 della dimensione del dataset
    batch_size = max(32, batch_size)  # minimo 32
    logging.info(f"Batch size impostato a: {batch_size}")
    
    # Crea e addestra il modello
    with tf.device('/GPU:0' if len(tf.config.list_physical_devices('GPU')) > 0 else '/CPU:0'):
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
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            TensorBoard(log_dir=f'logs/lstm_{timeframe}')
        ]
        
        @tf.autograph.experimental.do_not_convert
        def train_model():
            return model.fit(
                X, y,
                epochs=50,
                batch_size=batch_size,  # Batch size aumentato
                validation_split=0.2,
                class_weight=class_weight_dict,
                callbacks=callbacks,
                verbose=1,
                use_multiprocessing=True,  # Abilita il multiprocessing
                workers=4  # Utilizza 4 worker per il caricamento parallelo dei dati
            )
        
        # Addestra il modello
        history = train_model()
    
    # Salva il grafico della storia di training
    save_training_history_plot(history, timeframe)
    
    # Crea e addestra lo scaler
    scaler = StandardScaler()
    scaler.fit(X.reshape(-1, X.shape[2]))
    
    # Salva il modello e lo scaler
    ensure_trained_models_dir()
    model.save(get_lstm_model_file(timeframe))
    joblib.dump(scaler, get_lstm_scaler_file(timeframe))
    
    # Calcola e registra il tempo di training
    training_time = datetime.now() - start_time
    logging.info(f"LSTM model trained and saved for timeframe {timeframe} in {training_time}")
    return model, scaler, history

async def train_random_forest_model_wrapper(symbols, exchange, timestep, timeframe):
    """Wrapper per l'addestramento del modello Random Forest."""
    logging.info(f"Training Random Forest model for timeframe {timeframe}")
    start_time = datetime.now()
    
    # Raccogli i dati di training
    all_data = []
    all_labels = []
    
    for symbol in symbols:
        try:
            df = await get_data_async(exchange, symbol, timeframe=timeframe)
            if df is None or len(df) < timestep + 1:
                continue
                
            # Verifica se il DataFrame contiene una colonna timestamp
            if 'timestamp' in df.columns:
                # Rimuovi la colonna timestamp prima di preparare i dati
                df_temp = df.drop(columns=['timestamp'], errors='ignore')
            else:
                df_temp = df
                
            # Prepara i dati
            data = prepare_data(df_temp)
            
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
    
    # Log delle dimensioni del dataset
    logging.info(f"Random Forest dataset dimensioni: {X.shape}, etichette: {y.shape}")
    
    # Crea un RandomForestClassifier ottimizzato con più parallelismo
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42, 
        class_weight='balanced',
        n_jobs=-1,  # Utilizza tutti i core disponibili
        verbose=1  # Mostra progresso
    )
    
    # Calcola i pesi delle classi
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y),
        y=y
    )
    class_weight_dict = dict(zip(np.unique(y), class_weights))
    
    # Addestra il modello in modo asincrono su un thread separato
    def train_rf_with_logging():
        logging.info(f"Avvio training RF per {timeframe}")
        model.fit(X, y, class_weight=class_weight_dict)
        logging.info(f"Training RF per {timeframe} completato")
        return model
    
    # Esegui il training in modo asincrono su un altro thread
    model = await asyncio.to_thread(train_rf_with_logging)
    
    # Crea e addestra lo scaler
    scaler = StandardScaler()
    scaler.fit(X)
    
    # Salva il modello e lo scaler
    ensure_trained_models_dir()
    joblib.dump(model, get_rf_model_file(timeframe))
    joblib.dump(scaler, get_rf_scaler_file(timeframe))
    
    # Calcola e registra il tempo di training
    training_time = datetime.now() - start_time
    logging.info(f"Random Forest model trained and saved for timeframe {timeframe} in {training_time}")
    return model, scaler, None

async def train_xgboost_model_wrapper(symbols, exchange, timestep, timeframe):
    """Wrapper per l'addestramento del modello XGBoost."""
    logging.info(f"Training XGBoost model for timeframe {timeframe}")
    start_time = datetime.now()
    
    # Raccogli i dati di training
    all_data = []
    all_labels = []
    
    for symbol in symbols:
        try:
            df = await get_data_async(exchange, symbol, timeframe=timeframe)
            if df is None or len(df) < timestep + 1:
                continue
                
            # Verifica se il DataFrame contiene una colonna timestamp
            if 'timestamp' in df.columns:
                # Rimuovi la colonna timestamp prima di preparare i dati
                df_temp = df.drop(columns=['timestamp'], errors='ignore')
            else:
                df_temp = df
                
            # Prepara i dati
            data = prepare_data(df_temp)
            
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
    
    # Log delle dimensioni del dataset
    logging.info(f"XGBoost dataset dimensioni: {X.shape}, etichette: {y.shape}")
    
    # Configurazione del modello ottimizzata
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'scale_pos_weight': sum(y == 0) / sum(y == 1),  # Bilancia le classi
        'tree_method': 'hist',  # Usa l'algoritmo più veloce
        'nthread': -1  # Usa tutti i core disponibili
    }
    
    # Crea il dataset DMatrix
    dtrain = xgb.DMatrix(X, label=y)
    
    # Funzione per l'addestramento in modo asincrono
    def train_xgb_with_logging():
        logging.info(f"Avvio training XGBoost per {timeframe}")
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=100,
            evals=[(dtrain, 'train')],
            verbose_eval=10  # Log ogni 10 iterazioni
        )
        logging.info(f"Training XGBoost per {timeframe} completato")
        return model
    
    # Esegui il training in modo asincrono su un altro thread
    model = await asyncio.to_thread(train_xgb_with_logging)
    
    # Crea e addestra lo scaler
    scaler = StandardScaler()
    scaler.fit(X)
    
    # Salva il modello e lo scaler
    ensure_trained_models_dir()
    # Salva il modello con estensione .model (formato nativo di XGBoost)
    model_path = f"trained_models/xgb_model_{timeframe}.model"
    model.save_model(model_path)
    joblib.dump(scaler, get_xgb_scaler_file(timeframe))
    
    # Calcola e registra il tempo di training
    training_time = datetime.now() - start_time
    logging.info(f"XGBoost model trained and saved for timeframe {timeframe} in {training_time}")
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
    alt_model_path = f"trained_models/xgb_model_{timeframe}.model"
    scaler_path = get_xgb_scaler_file(timeframe)
    
    # Controllo se il modello esiste con estensione .json
    json_exists = os.path.exists(model_path)
    # Controllo se il modello esiste con estensione .model
    model_exists = os.path.exists(alt_model_path)
    # Controllo se lo scaler esiste
    scaler_exists = os.path.exists(scaler_path)
    
    if (json_exists or model_exists) and scaler_exists:
        model = xgb.Booster()
        try:
            # Prova a caricare prima con estensione .json
            if json_exists:
                logging.info(f"Caricamento modello XGB da {model_path}")
                model.load_model(model_path)
            # Altrimenti prova con estensione .model
            else:
                logging.info(f"Caricamento modello XGB da {alt_model_path}")
                model.load_model(alt_model_path)
                
            scaler = joblib.load(scaler_path)
            logging.info(f"XGB loaded for {timeframe}")
            return model, scaler
        except Exception as e:
            logging.error(f"Errore nel caricamento del modello XGB: {e}")
            return None, None
            
    logging.warning(f"XGB o scaler mancanti per {timeframe}")
    logging.warning(f"Model (.json): {json_exists}, Model (.model): {model_exists}, Scaler: {scaler_exists}")
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