#!/usr/bin/env python3
"""
Binary Model Training Script
Allena modelli binari per predizioni BUY/SELL usando i dataset generati da real_time.py
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import joblib
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

# XGBoost and LightGBM
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️ XGBoost non installato")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("⚠️ LightGBM non installato")

class BinaryModelTrainer:
    def __init__(self, config_path="binary_system_config.json"):
        """Inizializza il trainer per modelli binari"""
        self.config = self.load_config(config_path)
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.scaler = StandardScaler()
        
        # Crea directory per modelli e report
        self.setup_directories()
    
    def load_config(self, config_path):
        """Carica configurazione da file JSON"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️ Config file {config_path} non trovato, uso config default")
            return self.get_default_config()
    
    def get_default_config(self):
        """Configurazione di default se il file non esiste"""
        return {
            "model_training_config": {
                "remove_hold_labels": True,
                "label_mapping": {"BUY": 1, "SELL": 0},
                "balance_dataset": True,
                "use_smote": True,
                "test_size": 0.2,
                "random_state": 42,
                "models_to_train": ["RandomForest", "XGBoost", "LightGBM", "LogisticRegression"],
                "enable_hyperparameter_tuning": True,
                "cv_folds": 5
            },
            "binary_prediction_config": {
                "model_path": "ml_system/models/binary_models/best_binary_model.pkl",
                "prediction_symbols": ["BTC_USDTUSDT", "ETH_USDTUSDT", "DOGE_USDTUSDT", "SOL_USDTUSDT"],
                "prediction_timeframes": ["1h", "4h"]
            }
        }
    
    def setup_directories(self):
        """Crea directory necessarie"""
        dirs = [
            "ml_system",
            "ml_system/models",
            "ml_system/models/binary_models",
            "ml_system/reports",
            "ml_system/reports/binary_training"
        ]
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def load_datasets(self):
        """Carica tutti i dataset disponibili"""
        datasets = []
        ml_datasets_path = Path("ml_datasets")
        
        if not ml_datasets_path.exists():
            raise FileNotFoundError("Directory ml_datasets non trovata. Esegui prima real_time.py")
        
        symbols = self.config["binary_prediction_config"]["prediction_symbols"]
        timeframes = self.config["binary_prediction_config"]["prediction_timeframes"]
        
        print("📊 Caricamento dataset...")
        
        for symbol in symbols:
            for timeframe in timeframes:
                dataset_path = ml_datasets_path / symbol / timeframe / "merged.csv"
                if dataset_path.exists():
                    try:
                        df = pd.read_csv(dataset_path)
                        if not df.empty and 'label' in df.columns:
                            print(f"  ✅ {symbol} ({timeframe}): {len(df)} records")
                            datasets.append(df)
                        else:
                            print(f"  ⚠️ {symbol} ({timeframe}): dataset vuoto o senza label")
                    except Exception as e:
                        print(f"  ❌ {symbol} ({timeframe}): errore caricamento - {e}")
                else:
                    print(f"  ⚠️ {symbol} ({timeframe}): file non trovato")
        
        if not datasets:
            raise ValueError("Nessun dataset valido trovato")
        
        # Combina tutti i dataset
        combined_df = pd.concat(datasets, ignore_index=True)
        print(f"\n📈 Dataset combinato: {len(combined_df)} records totali")
        
        return combined_df
    
    def preprocess_data(self, df):
        """Preprocessa i dati per il training"""
        print("🔧 Preprocessing dati...")
        
        # Rimuovi colonne non necessarie
        # IMPORTANTE: escludere 'y' e 'pattern' per evitare data leakage!
        exclude_cols = ['timestamp', 'symbol', 'timeframe', 'y', 'pattern']
        feature_cols = [col for col in df.columns if col not in exclude_cols + ['label']]
        
        X = df[feature_cols].copy()
        y = df['label'].copy()
        
        # Gestisci valori mancanti
        X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Rimuovi feature con varianza zero
        zero_var_cols = X.columns[X.var() == 0]
        if len(zero_var_cols) > 0:
            print(f"  📊 Rimosse {len(zero_var_cols)} colonne con varianza zero")
            X = X.drop(columns=zero_var_cols)
        
        # Mapping delle label
        if self.config["model_training_config"]["remove_hold_labels"]:
            # Rimuovi HOLD se presenti
            mask = y.isin([0, 1])
            X = X[mask]
            y = y[mask]
        
        print(f"  📊 Features finali: {X.shape[1]}")
        print(f"  📊 Samples finali: {len(X)}")
        print(f"  📊 Distribuzione label: {y.value_counts().to_dict()}")
        
        return X, y
    
    def balance_dataset(self, X, y):
        """Bilancia il dataset usando SMOTE"""
        if not self.config["model_training_config"]["balance_dataset"]:
            return X, y
        
        print("⚖️ Bilanciamento dataset...")
        
        if self.config["model_training_config"]["use_smote"]:
            smote = SMOTETomek(random_state=self.config["model_training_config"]["random_state"])
            X_balanced, y_balanced = smote.fit_resample(X, y)
            print(f"  📊 Dopo SMOTE: {len(X_balanced)} samples")
            print(f"  📊 Nuova distribuzione: {pd.Series(y_balanced).value_counts().to_dict()}")
            return X_balanced, y_balanced
        
        return X, y
    
    def get_model_configs(self):
        """Configurazioni dei modelli"""
        configs = {}
        
        if "RandomForest" in self.config["model_training_config"]["models_to_train"]:
            configs["RandomForest"] = {
                "model": RandomForestClassifier(random_state=42),
                "params": {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
            }
        
        if "LogisticRegression" in self.config["model_training_config"]["models_to_train"]:
            configs["LogisticRegression"] = {
                "model": LogisticRegression(random_state=42, max_iter=1000),
                "params": {
                    'C': [0.1, 1, 10],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                }
            }
        
        if HAS_XGBOOST and "XGBoost" in self.config["model_training_config"]["models_to_train"]:
            configs["XGBoost"] = {
                "model": xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
                "params": {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.1, 0.01],
                    'subsample': [0.8, 1.0]
                }
            }
        
        if HAS_LIGHTGBM and "LightGBM" in self.config["model_training_config"]["models_to_train"]:
            configs["LightGBM"] = {
                "model": lgb.LGBMClassifier(random_state=42, verbose=-1),
                "params": {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.1, 0.01],
                    'num_leaves': [31, 50]
                }
            }
        
        return configs
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Allena tutti i modelli configurati"""
        print(f"🤖 Training modelli...")
        
        model_configs = self.get_model_configs()
        results = {}
        
        for name, config in model_configs.items():
            print(f"\n🔧 Training {name}...")
            
            try:
                if self.config["model_training_config"]["enable_hyperparameter_tuning"]:
                    # Grid search per hyperparameter tuning
                    grid_search = GridSearchCV(
                        config["model"],
                        config["params"],
                        cv=self.config["model_training_config"]["cv_folds"],
                        scoring='accuracy',
                        n_jobs=-1,
                        verbose=0
                    )
                    grid_search.fit(X_train, y_train)
                    model = grid_search.best_estimator_
                    print(f"  📊 Best params: {grid_search.best_params_}")
                else:
                    # Training standard
                    model = config["model"]
                    model.fit(X_train, y_train)
                
                # Predizioni
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Metriche
                accuracy = accuracy_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
                
                # Cross validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
                
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'auc': auc,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'predictions': y_pred,
                    'probabilities': y_proba
                }
                
                print(f"  ✅ Accuracy: {accuracy:.4f}")
                print(f"  ✅ AUC: {auc:.4f}" if auc else "  ⚠️ AUC: N/A")
                print(f"  ✅ CV Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
                
                # Aggiorna il miglior modello
                if accuracy > self.best_score:
                    self.best_score = accuracy
                    self.best_model = model
                    self.best_model_name = name
                
            except Exception as e:
                print(f"  ❌ Errore training {name}: {e}")
                continue
        
        return results
    
    def save_best_model(self):
        """Salva il miglior modello"""
        if self.best_model is None:
            print("❌ Nessun modello da salvare")
            return
        
        model_path = self.config["binary_prediction_config"]["model_path"]
        
        # Salva modello e scaler
        model_data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'model_name': self.best_model_name,
            'accuracy': self.best_score,
            'trained_at': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, model_path)
        print(f"💾 Miglior modello salvato: {model_path}")
        print(f"   📊 Modello: {self.best_model_name}")
        print(f"   📊 Accuracy: {self.best_score:.4f}")
    
    def generate_report(self, results, y_test):
        """Genera report dettagliato"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"ml_system/reports/binary_training/training_report_{timestamp}.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("BINARY MODEL TRAINING REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write(f"Test set size: {len(y_test)}\n\n")
            
            for name, result in results.items():
                f.write(f"\n{name.upper()}\n")
                f.write("-" * 40 + "\n")
                f.write(f"Accuracy: {result['accuracy']:.4f}\n")
                f.write(f"AUC: {result['auc']:.4f}\n" if result['auc'] else "AUC: N/A\n")
                f.write(f"CV Score: {result['cv_mean']:.4f} (±{result['cv_std']:.4f})\n")
                f.write("\nClassification Report:\n")
                f.write(classification_report(y_test, result['predictions']))
                f.write("\n")
            
            f.write(f"\n🏆 BEST MODEL: {self.best_model_name} (Accuracy: {self.best_score:.4f})\n")
        
        print(f"📋 Report salvato: {report_path}")
    
    def run_training(self):
        """Esegue l'intero processo di training"""
        print("🚀 AVVIO BINARY MODEL TRAINING")
        print("="*60)
        
        try:
            # 1. Carica dataset
            df = self.load_datasets()
            
            # 2. Preprocessa dati
            X, y = self.preprocess_data(df)
            
            # 3. Split train/test
            test_size = self.config["model_training_config"]["test_size"]
            random_state = self.config["model_training_config"]["random_state"]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # 4. Scala features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # 5. Bilancia dataset
            X_train_balanced, y_train_balanced = self.balance_dataset(X_train_scaled, y_train)
            
            # 6. Allena modelli
            results = self.train_models(X_train_balanced, X_test_scaled, y_train_balanced, y_test)
            
            # 7. Salva miglior modello
            self.save_best_model()
            
            # 8. Genera report
            self.generate_report(results, y_test)
            
            print("\n" + "="*60)
            print("✅ TRAINING COMPLETATO CON SUCCESSO!")
            print(f"🏆 Miglior modello: {self.best_model_name}")
            print(f"📊 Accuracy: {self.best_score:.4f}")
            print("="*60)
            
        except Exception as e:
            print(f"\n❌ ERRORE DURANTE IL TRAINING: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Funzione principale"""
    trainer = BinaryModelTrainer()
    trainer.run_training()

if __name__ == "__main__":
    main()
