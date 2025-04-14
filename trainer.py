import os
import time
import logging
import joblib
import json
import datetime
import numpy as np
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
    """
    Funzione vuota per retrocompatibilità. Non salva più log né file.
    """
    pass

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
    
    # Raccogli informazioni sul periodo di training
    training_info = {
        "num_cryptocurrencies": len(symbols),
        "symbols_used": symbols,
        "start_date": None,
        "end_date": None,
        "timeframe": timeframe,
        "timestep": timestep,
        "training_started": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    for symbol in symbols:
        df = await get_data_async(exchange, symbol, timeframe)
        if df is None:
            continue
        
        # Raccogli informazioni sulla data di inizio e fine
        if df.index.size > 0:
            symbol_start = df.index[0].strftime("%Y-%m-%d %H:%M:%S")
            symbol_end = df.index[-1].strftime("%Y-%m-%d %H:%M:%S")
            
            # Aggiorna le date di inizio e fine generali
            if training_info["start_date"] is None or symbol_start < training_info["start_date"]:
                training_info["start_date"] = symbol_start
            if training_info["end_date"] is None or symbol_end > training_info["end_date"]:
                training_info["end_date"] = symbol_end
        
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
    
    # Calcola durata in giorni del periodo di training
    if training_info["start_date"] and training_info["end_date"]:
        start = datetime.datetime.strptime(training_info["start_date"], "%Y-%m-%d %H:%M:%S")
        end = datetime.datetime.strptime(training_info["end_date"], "%Y-%m-%d %H:%M:%S")
        training_info["days_covered"] = (end - start).days + 1
    else:
        training_info["days_covered"] = "N/A"
    
    # Salva informazioni sui dati di training
    training_info["total_samples"] = len(X_all)
    training_info["class_distribution"] = {
        "class_0": int(np.sum(y_all == 0)),
        "class_1": int(np.sum(y_all == 1))
    }
    
    tscv = TimeSeriesSplit(n_splits=4)
    splits = list(tscv.split(X_all))
    train_index, val_index = splits[-1]
    X_train, X_val = X_all[train_index], X_all[val_index]
    y_train, y_val = y_all[train_index], y_all[val_index]
    
    # Aggiorna le informazioni di training con i dettagli del split
    training_info["training_samples"] = len(X_train)
    training_info["validation_samples"] = len(X_val)
    
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
            
            # Aggiorna le informazioni di training dopo l'augmentazione
            training_info["augmented_samples"] = len(augmented_X)
            training_info["post_augmentation_training_samples"] = len(X_train)

    class_weights_arr = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {i: class_weights_arr[i] for i in range(len(class_weights_arr))}
    
    # Aggiungi i pesi delle classi alle informazioni di training
    training_info["class_weights"] = {str(i): float(class_weights_arr[i]) for i in range(len(class_weights_arr))}
    
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
    
    # Aggiunta configurazione del modello alle informazioni di training
    training_info["model_config"] = {
        "type": "LSTM",
        "input_shape": input_shape,
        "optimizer": "adam",
        "loss": model.loss.__class__.__name__ if hasattr(model.loss, '__class__') else str(model.loss)
    }
    
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    epochs = 100
    class TrainingProgressBar(tf.keras.callbacks.Callback):
        def __init__(self, total_epochs):
            super().__init__()
            self.total_epochs = total_epochs
            self.current_epoch = 0
            
        def on_epoch_end(self, epoch, logs=None):
            self.current_epoch += 1
            # Utilizziamo logging invece di tqdm per evitare problemi di thread
            logging.info(
                f"Epoch {self.current_epoch}/{self.total_epochs} - "
                f"loss: {logs['loss']:.4f}, accuracy: {logs['accuracy']:.4f}, "
                f"val_loss: {logs['val_loss']:.4f}, val_accuracy: {logs['val_accuracy']:.4f}"
            )
            
        def on_train_end(self, logs=None):
            logging.info(f"Training completato dopo {self.current_epoch} epoche")
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=[early_stop, TrainingProgressBar(epochs)],
        verbose=0
    )
    
    # Aggiorna le informazioni con i dettagli del training
    training_info["epochs_trained"] = len(history.history['loss'])
    training_info["early_stopping"] = training_info["epochs_trained"] < epochs
    training_info["training_completed"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
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
            "loss": float(train_metrics[0]),
            "accuracy": float(train_metrics[1]),
            "precision": float(train_metrics[2]),
            "recall": float(train_metrics[3]),
            "auc": float(train_metrics[4]),
        },
        "validation": {
            "loss": float(val_metrics[0]),
            "accuracy": float(val_metrics[1]),
            "precision": float(val_metrics[2]),
            "recall": float(val_metrics[3]),
            "auc": float(val_metrics[4]),
        },
        "history": {
            key: [float(val) for val in history.history[key]]
            for key in history.history.keys()
        },
        "training_info": training_info
    }
    
    # Utilizzo della nuova funzione per salvare le metriche in modo uniforme
    save_model_metrics(metrics_dict, "lstm", timeframe, trained_models_dir)
    
    return model, scaler_final, metrics_dict

# Trainer per Random Forest
async def train_random_forest_model_wrapper(top_symbols, exchange, timestep, timeframe):
    X_combined = []
    y_combined = []
    from fetcher import get_data_async
    
    # Raccogli informazioni sul periodo di training
    training_info = {
        "num_cryptocurrencies": len(top_symbols),
        "symbols_used": top_symbols,
        "start_date": None,
        "end_date": None,
        "timeframe": timeframe,
        "timestep": timestep,
        "training_started": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    for symbol in top_symbols:
        df = await get_data_async(exchange, symbol, timeframe)
        if df is None:
            continue
            
        # Raccogli informazioni sulla data di inizio e fine
        if df.index.size > 0:
            symbol_start = df.index[0].strftime("%Y-%m-%d %H:%M:%S")
            symbol_end = df.index[-1].strftime("%Y-%m-%d %H:%M:%S")
            
            # Aggiorna le date di inizio e fine generali
            if training_info["start_date"] is None or symbol_start < training_info["start_date"]:
                training_info["start_date"] = symbol_start
            if training_info["end_date"] is None or symbol_end > training_info["end_date"]:
                training_info["end_date"] = symbol_end
                
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
        
        # Calcola durata in giorni del periodo di training
        if training_info["start_date"] and training_info["end_date"]:
            start = datetime.datetime.strptime(training_info["start_date"], "%Y-%m-%d %H:%M:%S")
            end = datetime.datetime.strptime(training_info["end_date"], "%Y-%m-%d %H:%M:%S")
            training_info["days_covered"] = (end - start).days + 1
        else:
            training_info["days_covered"] = "N/A"
            
        # Salva informazioni sui dati di training
        training_info["total_samples"] = len(X_all)
        training_info["class_distribution"] = {
            "class_0": int(np.sum(y_all == 0)),
            "class_1": int(np.sum(y_all == 1))
        }
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_all)
        tscv = TimeSeriesSplit(n_splits=4)
        splits = list(tscv.split(X_scaled))
        train_index, val_index = splits[-1]
        X_train, X_val = X_scaled[train_index], X_scaled[val_index]
        y_train, y_val = y_all[train_index], y_all[val_index]
        
        # Aggiorna le informazioni di training con i dettagli del split
        training_info["training_samples"] = len(X_train)
        training_info["validation_samples"] = len(X_val)
        training_info["train_val_split_ratio"] = float(len(X_train) / (len(X_train) + len(X_val)))
        
        rf = create_rf_model()
        logging.info("Training Random Forest model...")
        
        # Registra il tempo di inizio training
        start_time = time.time()
        
        rf.fit(X_train, y_train)
        
        # Calcola il tempo di training
        training_time = time.time() - start_time
        training_info["training_time_seconds"] = float(training_time)
        training_info["training_completed"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Aggiungi configurazione del modello
        rf_params = {
            "type": "RandomForest",
            "n_estimators": rf.n_estimators,
            "max_depth": rf.max_depth,
            "random_state": 42
        }
        training_info["model_config"] = rf_params
        
        logging.info("Random Forest training completed")
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
        
        # Valutazione sul training set
        y_train_pred = rf.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_precision = precision_score(y_train, y_train_pred, average='binary', zero_division=0)
        train_recall = recall_score(y_train, y_train_pred, average='binary', zero_division=0)
        train_f1 = f1_score(y_train, y_train_pred, average='binary', zero_division=0)
        
        # Valutazione sul validation set
        y_pred = rf.predict(X_val)
        y_proba = rf.predict_proba(X_val)[:, 1] if hasattr(rf, 'predict_proba') else None
        
        val_accuracy = accuracy_score(y_val, y_pred)
        val_precision = precision_score(y_val, y_pred, average='binary', zero_division=0)
        val_recall = recall_score(y_val, y_pred, average='binary', zero_division=0)
        val_f1 = f1_score(y_val, y_pred, average='binary', zero_division=0)
        
        # Calcola matrice di confusione
        cm = confusion_matrix(y_val, y_pred).tolist()
        
        # Calcola curva ROC e AUC se possibile
        fpr, tpr, _ = roc_curve(y_val, y_proba) if y_proba is not None else ([0], [0], [0])
        roc_auc = auc(fpr, tpr) if len(fpr) > 1 else 0
        
        metrics = {
            "train": {
                "accuracy": float(train_accuracy),
                "precision": float(train_precision),
                "recall": float(train_recall),
                "f1": float(train_f1)
            },
            "validation": {
                "accuracy": float(val_accuracy),
                "precision": float(val_precision),
                "recall": float(val_recall),
                "f1": float(val_f1)
            },
            "confusion_matrix": cm,
            "roc_auc": float(roc_auc),
            "feature_importances": [float(imp) for imp in rf.feature_importances_] if hasattr(rf, 'feature_importances_') else [],
            "training_info": training_info
        }
        
        # Converti i valori fpr e tpr in liste di float
        if y_proba is not None:
            metrics["fpr"] = [float(f) for f in fpr]
            metrics["tpr"] = [float(t) for t in tpr]
            
        logging.info(f"Random Forest Metrics: {metrics}")
        
        trained_models_dir = ensure_trained_models_dir()
        model_file = os.path.join(trained_models_dir, f'rf_model_{timeframe}.pkl')
        scaler_file = os.path.join(trained_models_dir, f'rf_scaler_{timeframe}.pkl')
        metrics_file = os.path.join(trained_models_dir, f'rf_model_{timeframe}_metrics.json')
        
        joblib.dump(rf, model_file)
        joblib.dump(scaler, scaler_file)
        
        # Utilizzo della nuova funzione per salvare le metriche in modo uniforme
        save_model_metrics(metrics, "rf", timeframe, trained_models_dir)
        
        return rf, scaler, metrics
    else:
        logging.error("Failed to collect data for Random Forest training")
        return None, None, None

# Trainer per XGBoost
async def train_xgboost_model_wrapper(top_symbols, exchange, timestep, timeframe):
    X_combined = []
    y_combined = []
    from fetcher import get_data_async
    
    # Raccogli informazioni sul periodo di training
    training_info = {
        "num_cryptocurrencies": len(top_symbols),
        "symbols_used": top_symbols,
        "start_date": None,
        "end_date": None,
        "timeframe": timeframe,
        "timestep": timestep,
        "training_started": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    for symbol in top_symbols:
        df = await get_data_async(exchange, symbol, timeframe)
        if df is None:
            continue
            
        # Raccogli informazioni sulla data di inizio e fine
        if df.index.size > 0:
            symbol_start = df.index[0].strftime("%Y-%m-%d %H:%M:%S")
            symbol_end = df.index[-1].strftime("%Y-%m-%d %H:%M:%S")
            
            # Aggiorna le date di inizio e fine generali
            if training_info["start_date"] is None or symbol_start < training_info["start_date"]:
                training_info["start_date"] = symbol_start
            if training_info["end_date"] is None or symbol_end > training_info["end_date"]:
                training_info["end_date"] = symbol_end
                
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
        
        # Calcola durata in giorni del periodo di training
        if training_info["start_date"] and training_info["end_date"]:
            start = datetime.datetime.strptime(training_info["start_date"], "%Y-%m-%d %H:%M:%S")
            end = datetime.datetime.strptime(training_info["end_date"], "%Y-%m-%d %H:%M:%S")
            training_info["days_covered"] = (end - start).days + 1
        else:
            training_info["days_covered"] = "N/A"
            
        # Salva informazioni sui dati di training
        training_info["total_samples"] = len(X_all)
        training_info["class_distribution"] = {
            "class_0": int(np.sum(y_all == 0)),
            "class_1": int(np.sum(y_all == 1))
        }
        
        # Prepariamo un training migliorato con metriche dettagliate
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_all)
        tscv = TimeSeriesSplit(n_splits=4)
        splits = list(tscv.split(X_scaled))
        train_index, val_index = splits[-1]
        X_train, X_val = X_scaled[train_index], X_scaled[val_index]
        y_train, y_val = y_all[train_index], y_all[val_index]
        
        # Aggiorna le informazioni di training con i dettagli del split
        training_info["training_samples"] = len(X_train)
        training_info["validation_samples"] = len(X_val)
        training_info["train_val_split_ratio"] = float(len(X_train) / (len(X_train) + len(X_val)))
        
        import xgboost as xgb
        xgb_params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "use_label_encoder": False,
            "eval_metric": 'logloss'
        }
        model = xgb.XGBClassifier(**xgb_params)
        
        # Aggiungi configurazione del modello alle informazioni di training
        training_info["model_config"] = {
            "type": "XGBoost",
            "params": xgb_params
        }
        
        logging.info("Training XGBoost model...")
        
        # Registra il tempo di inizio training
        start_time = time.time()
        
        # Train with early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=10,
            verbose=False
        )
        
        # Calcola il tempo di training
        training_time = time.time() - start_time
        training_info["training_time_seconds"] = float(training_time)
        training_info["training_completed"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        training_info["best_iteration"] = model.best_iteration if hasattr(model, 'best_iteration') else None
        
        logging.info("XGBoost training completed")
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
        
        # Valutazione sul training set
        y_train_pred = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_precision = precision_score(y_train, y_train_pred, average='binary', zero_division=0)
        train_recall = recall_score(y_train, y_train_pred, average='binary', zero_division=0)
        train_f1 = f1_score(y_train, y_train_pred, average='binary', zero_division=0)
        
        # Valutazione sul validation set
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
        
        val_accuracy = accuracy_score(y_val, y_pred)
        val_precision = precision_score(y_val, y_pred, average='binary', zero_division=0)
        val_recall = recall_score(y_val, y_pred, average='binary', zero_division=0)
        val_f1 = f1_score(y_val, y_pred, average='binary', zero_division=0)
        
        # Calcola matrice di confusione
        cm = confusion_matrix(y_val, y_pred).tolist()
        
        # Calcola curva ROC e AUC se possibile
        fpr, tpr, _ = roc_curve(y_val, y_proba) if y_proba is not None else ([0], [0], [0])
        roc_auc = auc(fpr, tpr) if len(fpr) > 1 else 0
        
        # Metriche complete
        metrics = {
            "train": {
                "accuracy": float(train_accuracy),
                "precision": float(train_precision),
                "recall": float(train_recall),
                "f1": float(train_f1)
            },
            "validation": {
                "accuracy": float(val_accuracy),
                "precision": float(val_precision),
                "recall": float(val_recall),
                "f1": float(val_f1)
            },
            "confusion_matrix": cm,
            "roc_auc": float(roc_auc),
            "feature_importances": [float(imp) for imp in model.feature_importances_] if hasattr(model, 'feature_importances_') else [],
            "training_info": training_info
        }
        
        # Converti i valori fpr e tpr in liste di float
        if y_proba is not None:
            metrics["fpr"] = [float(f) for f in fpr]
            metrics["tpr"] = [float(t) for t in tpr]
            
        logging.info(f"XGBoost Metrics: {metrics}")
        
        trained_models_dir = ensure_trained_models_dir()
        model_file = os.path.join(trained_models_dir, f'xgb_model_{timeframe}.pkl')
        scaler_file = os.path.join(trained_models_dir, f'xgb_scaler_{timeframe}.pkl')
        metrics_file = os.path.join(trained_models_dir, f'xgb_model_{timeframe}_metrics.json')
        
        joblib.dump(model, model_file)
        joblib.dump(scaler, scaler_file)
        
        # Utilizzo della nuova funzione per salvare le metriche in modo uniforme  
        save_model_metrics(metrics, "xgb", timeframe, trained_models_dir)
        
        return model, scaler, metrics

# Aggiungo una funzione per centralizzare il salvataggio delle metriche
def save_model_metrics(metrics, model_type, timeframe, trained_models_dir):
    """
    Funzione centralizzata per salvare le metriche dei modelli.
    
    Args:
        metrics (dict): Dizionario contenente tutte le metriche del modello
        model_type (str): Tipo di modello ('lstm', 'rf', o 'xgb')
        timeframe (str): Timeframe utilizzato per il training
        trained_models_dir (str): Directory dove salvare le metriche
    
    Returns:
        bool: True se il salvataggio è avvenuto con successo, False altrimenti
    """
    metrics_file = os.path.join(trained_models_dir, f'{model_type}_model_{timeframe}_metrics.json')
    
    # Aggiungi un timestamp al salvataggio delle metriche
    metrics["saved_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Aggiungi una versione del formato delle metriche
    metrics["metrics_version"] = "1.0"
    
    try:
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)
        logging.info(f"{model_type.upper()} metrics saved to {metrics_file}")
        return True
    except Exception as e:
        logging.error(f"Error saving {model_type.upper()} metrics: {e}")
        return False
