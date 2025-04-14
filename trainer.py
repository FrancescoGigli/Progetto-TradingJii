import os
import time
import logging
import joblib
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from keras.losses import Loss
from keras import backend as K
from keras_tuner import HyperModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm

import config
from models import FocalLoss, create_lstm_model, create_rf_model
from data_utils import prepare_data

def save_training_history_plot(history, timeframe):
    plot_dir = os.path.join("logs", "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    # Plot della Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss durante il Training')
    plt.xlabel('Epoca')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot dell'Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy durante il Training')
    plt.xlabel('Epoca')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plot_file = os.path.join(plot_dir, f"training_history_{timeframe}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.png")
    plt.savefig(plot_file)
    plt.close()
    logging.info("Grafici di training salvati in: %s", plot_file)

def augment_jitter(X, sigma=0.03):
    noise = np.random.normal(loc=0.0, scale=sigma, size=X.shape)
    return X + noise

# Add at the top of the file with other imports
import os

def ensure_trained_models_dir():
    """Create trained_models directory if it doesn't exist"""
    trained_models_dir = os.path.join(os.path.dirname(__file__), 'trained_models')
    os.makedirs(trained_models_dir, exist_ok=True)
    return trained_models_dir

# Trainer per il modello LSTM
async def train_lstm_model_for_timeframe(exchange, symbols, timeframe, timestep):
    X_list = []
    y_list = []
    from fetcher import get_data_async  # Funzione per il recupero dati
    for symbol in symbols:
        df = await get_data_async(exchange, symbol, timeframe)
        if df is None:
            continue
        data = prepare_data(df)
        # Verifica la presenza di valori non finiti
        if not np.all(np.isfinite(data)):
            logging.warning(f"Il dataset per {symbol} nel timeframe {timeframe} contiene NaN o infiniti. Saltato.")
            continue
        if len(data) < timestep + 1:
            continue
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        X, y = [], []
        for i in range(timestep, len(data_scaled) - 1):
            X.append(data_scaled[i - timestep:i])
            y.append(1 if df['close'].iloc[i + 1] > df['close'].iloc[i] else 0)
        if X:
            X_list.append(np.array(X))
            y_list.append(np.array(y))
    if not X_list:
        logging.error(f"Failed to collect data for LSTM training at timeframe {timeframe}")
        return None, None, None

    X_all = np.concatenate(X_list)
    y_all = np.concatenate(y_list)
    
    tscv = TimeSeriesSplit(n_splits=4)
    splits = list(tscv.split(X_all))
    train_index, val_index = splits[-1]
    X_train, X_val = X_all[train_index], X_all[val_index]
    y_train, y_val = y_all[train_index], y_all[val_index]
    
    # Augmentazione per bilanciare le classi, se necessario
    unique, counts = np.unique(y_train, return_counts=True)
    if len(unique) == 2 and counts[0] != counts[1]:
        minority_class = unique[np.argmin(counts)]
        indices_minority = np.where(y_train == minority_class)[0]
        augmented_X = []
        augmented_y = []
        for i in indices_minority:
            augmented_sample = augment_jitter(X_train[i])
            augmented_X.append(augmented_sample)
            augmented_y.append(y_train[i])
        if augmented_X:
            X_train = np.concatenate([X_train, np.array(augmented_X)], axis=0)
            y_train = np.concatenate([y_train, np.array(augmented_y)], axis=0)
            logging.info("Augmentazione applicata per bilanciare le classi nel training set.")

    class_weights_arr = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {i: class_weights_arr[i] for i in range(len(class_weights_arr))}
    
    num_features = len(config.EXPECTED_COLUMNS)
    input_shape = (timestep, num_features)
    model = create_lstm_model(input_shape)
    model.compile(
        optimizer='adam',
        loss=model.loss,
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    epochs = 100
    class TrainingProgressBar(tf.keras.callbacks.Callback):
        def __init__(self, total_epochs):
            super().__init__()
            self.pbar = tqdm(total=total_epochs, desc=f"Training LSTM {timeframe}", ncols=80, leave=False)
        def on_epoch_end(self, epoch, logs=None):
            self.pbar.update(1)
            self.pbar.set_postfix({
                'loss': f"{logs['loss']:.4f}",
                'acc': f"{logs['accuracy']:.4f}"
            })
        def on_train_end(self, logs=None):
            self.pbar.close()
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=[early_stop, tensorboard_callback, TrainingProgressBar(epochs)],
        verbose=0
    )
    print()
    save_training_history_plot(history, timeframe)
    
    train_metrics = model.evaluate(X_train, y_train, verbose=0)
    val_metrics = model.evaluate(X_val, y_val, verbose=0)
    logging.info(
        "Training metrics -> Loss: %.4f, Accuracy: %.4f, Precision: %.4f, Recall: %.4f, AUC: %.4f",
        train_metrics[0], train_metrics[1], train_metrics[2], train_metrics[3], train_metrics[4]
    )
    logging.info(
        "Validation metrics -> Loss: %.4f, Accuracy: %.4f, Precision: %.4f, Recall: %.4f, AUC: %.4f",
        val_metrics[0], val_metrics[1], val_metrics[2], val_metrics[3], val_metrics[4]
    )
    
    model_file = config.get_lstm_model_file(timeframe)
    scaler_file = config.get_lstm_scaler_file(timeframe)
    
    # Update model and scaler file paths to use trained_models directory
    trained_models_dir = ensure_trained_models_dir()
    model_file = os.path.join(trained_models_dir, f'lstm_model_{timeframe}.h5')
    scaler_file = os.path.join(trained_models_dir, f'lstm_scaler_{timeframe}.pkl')
    metrics_file = os.path.join(trained_models_dir, f'lstm_model_{timeframe}_metrics.json')
    
    model.save(model_file)
    scaler_final = StandardScaler()
    scaler_final.fit(X_train.reshape(-1, X_train.shape[2]))
    joblib.dump(scaler_final, scaler_file)
    logging.info("LSTM model trained and saved for timeframe %s in '%s' and '%s'.", timeframe, model_file, scaler_file)
    
    metrics_dict = {
        "train": {
            "loss": train_metrics[0],
            "accuracy": train_metrics[1],
            "precision": train_metrics[2],
            "recall": train_metrics[3],
            "auc": train_metrics[4],
        },
        "validation": {
            "loss": val_metrics[0],
            "accuracy": val_metrics[1],
            "precision": val_metrics[2],
            "recall": val_metrics[3],
            "auc": val_metrics[4],
        }
    }
    
    metrics_file = model_file.replace(".h5", "_metrics.json")
    try:
        with open(metrics_file, "w") as f:
            json.dump(metrics_dict, f, indent=4)
        logging.info("Metriche di training salvate in '%s'.", metrics_file)
    except Exception as e:
        logging.error("Errore nel salvataggio delle metriche: %s", e)
    
    return model, scaler_final, metrics_dict

# Trainer per Random Forest
def train_rf_sync(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    tscv = TimeSeriesSplit(n_splits=4)
    splits = list(tscv.split(X_scaled))
    train_index, val_index = splits[-1]
    X_train, X_val = X_scaled[train_index], X_scaled[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    rf = create_rf_model()
    logging.info("Training Random Forest model...")
    rf.fit(X_train, y_train)
    logging.info("Random Forest training completed")
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    y_pred = rf.predict(X_val)
    metrics = {
        'validation_accuracy': accuracy_score(y_val, y_pred),
        'validation_precision': precision_score(y_val, y_pred, average='binary'),
        'validation_recall': recall_score(y_val, y_pred, average='binary'),
        'validation_f1': f1_score(y_val, y_pred, average='binary')
    }
    logging.info(f"Random Forest Metrics: {metrics}")
    return rf, scaler, metrics

async def train_random_forest_model_wrapper(top_symbols, exchange, timestep, timeframe):
    X_combined = []
    y_combined = []
    from fetcher import get_data_async
    for symbol in top_symbols:
        df = await get_data_async(exchange, symbol, timeframe)
        if df is None:
            continue
        data = prepare_data(df)
        if not np.all(np.isfinite(data)):
            logging.warning(f"Il dataset per {symbol} nel timeframe {timeframe} contiene NaN o infiniti. Saltato.")
            continue
        if len(data) < timestep + 1:
            continue
        scaler_local = StandardScaler()
        data_scaled = scaler_local.fit_transform(data)
        X, y_data = [], []
        for i in range(timestep, len(data_scaled) - 1):
            X.append(data_scaled[i - timestep:i].flatten())
            y_data.append(1 if df['close'].iloc[i+1] > df['close'].iloc[i] else 0)
        if X:
            X_combined.extend(X)
            y_combined.extend(y_data)
    if X_combined and y_combined:
        X_all = np.array(X_combined)
        y_all = np.array(y_combined)
        rf_model, rf_scaler, metrics = train_rf_sync(X_all, y_all)
        trained_models_dir = ensure_trained_models_dir()
        model_file = os.path.join(trained_models_dir, f'rf_model_{timeframe}.pkl')
        scaler_file = os.path.join(trained_models_dir, f'rf_scaler_{timeframe}.pkl')
        metrics_file = os.path.join(trained_models_dir, f'rf_model_{timeframe}_metrics.json')
        
        joblib.dump(rf_model, model_file)
        joblib.dump(rf_scaler, scaler_file)
        
        try:
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=4)
            logging.info("RF model, scaler and metrics saved in trained_models directory")
        except Exception as e:
            logging.error("Error saving RF metrics: %s", e)
        
        return rf_model, rf_scaler, metrics
    else:
        logging.error("Failed to collect data for Random Forest training")
        return None, None, None

# Trainer per XGBoost
def train_xgb_model(X, y):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    tscv = TimeSeriesSplit(n_splits=4)
    splits = list(tscv.split(X_scaled))
    train_index, val_index = splits[-1]
    X_train, X_val = X_scaled[train_index], X_scaled[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    import xgboost as xgb
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    logging.info("Training XGBoost model...")
    model.fit(X_train, y_train)
    logging.info("XGBoost training completed")
    y_pred = model.predict(X_val)
    metrics = {
        'validation_accuracy': accuracy_score(y_val, y_pred),
        'validation_precision': precision_score(y_val, y_pred, average='binary'),
        'validation_recall': recall_score(y_val, y_pred, average='binary'),
        'validation_f1': f1_score(y_val, y_pred, average='binary')
    }
    logging.info(f"XGBoost Metrics: {metrics}")
    return model, scaler, metrics

async def train_xgboost_model_wrapper(top_symbols, exchange, timestep, timeframe):
    X_combined = []
    y_combined = []
    from fetcher import get_data_async
    for symbol in top_symbols:
        df = await get_data_async(exchange, symbol, timeframe)
        if df is None:
            continue
        data = prepare_data(df)
        if not np.all(np.isfinite(data)):
            logging.warning(f"Il dataset per {symbol} nel timeframe {timeframe} contiene NaN o infiniti. Saltato.")
            continue
        if len(data) < timestep + 1:
            continue
        X, y_data = [], []
        for i in range(timestep, len(data) - 1):
            X.append(data[i - timestep:i].flatten())
            y_data.append(1 if df['close'].iloc[i+1] > df['close'].iloc[i] else 0)
        if X:
            X_combined.extend(X)
            y_combined.extend(y_data)
    if X_combined and y_combined:
        X_all = np.array(X_combined)
        y_all = np.array(y_combined)
        xgb_model, xgb_scaler, metrics = train_xgb_model(X_all, y_all)
        trained_models_dir = ensure_trained_models_dir()
        model_file = os.path.join(trained_models_dir, f'xgb_model_{timeframe}.pkl')
        scaler_file = os.path.join(trained_models_dir, f'xgb_scaler_{timeframe}.pkl')
        metrics_file = os.path.join(trained_models_dir, f'xgb_model_{timeframe}_metrics.json')
        
        joblib.dump(xgb_model, model_file)
        joblib.dump(xgb_scaler, scaler_file)
        
        try:
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=4)
            logging.info("XGBoost model, scaler and metrics saved in trained_models directory")
        except Exception as e:
            logging.error("Error saving XGBoost metrics: %s", e)
        
        return xgb_model, xgb_scaler, metrics

def train_lstm_model(X_train, y_train, X_val, y_val, timeframe):
    # Save model and metrics
    model_path = os.path.join('trained_models', f'lstm_model_{timeframe}')
    metrics_path = os.path.join('trained_models', f'lstm_model_{timeframe}_metrics.json')
    
    # Save model
    model.save(model_path)
    
    # Save metrics
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
