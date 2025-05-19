#!/usr/bin/env python3
"""
Utility functions for ML data processing
"""

import os
import logging
import pandas as pd
from colorama import Fore, Style

def assign_y_class_multiclass(csv_path: str, buy_threshold: float = 0.5, sell_threshold: float = -0.5):
    """
    Assegna etichette multiclasse al dataset.
    
    Args:
        csv_path: Percorso del file CSV da elaborare
        buy_threshold: Soglia per classificare come BUY (default: 0.5)
        sell_threshold: Soglia per classificare come SELL (default: -0.5)
        
    La funzione aggiunge una nuova colonna y_class al dataset con i seguenti valori:
    - Se y >= buy_threshold, allora y_class = 1 (BUY)
    - Se y <= sell_threshold, allora y_class = 2 (SELL)
    - Altrimenti, y_class = 0 (HOLD)
    """
    try:
        # Verifica se il file esiste
        if not os.path.exists(csv_path):
            logging.warning(f"File CSV non trovato: {csv_path}")
            return
            
        # Carica il CSV usando pandas
        df = pd.read_csv(csv_path)
        
        # Verifica se la colonna 'y' esiste
        if 'y' not in df.columns:
            logging.warning(f"La colonna 'y' non è presente nel file: {csv_path}")
            return
            
        # Controlla se ci sono valori in 'y'
        if df['y'].isna().all():
            logging.warning(f"La colonna 'y' è vuota nel file: {csv_path}")
            return
            
        # Applica la logica di classificazione multiclasse
        def classify(y):
            if pd.isna(y):
                return None
            elif y >= buy_threshold:
                return 1  # BUY
            elif y <= sell_threshold:
                return 2  # SELL
            else:
                return 0  # HOLD
                
        # Crea la nuova colonna y_class
        df['y_class'] = df['y'].apply(classify)
        
        # Conta la distribuzione delle classi
        class_counts = df['y_class'].value_counts().to_dict()
        
        # Prepara log con distribuzione delle classi
        total_records = len(df)
        buy_count = class_counts.get(1, 0)
        sell_count = class_counts.get(2, 0)
        hold_count = class_counts.get(0, 0)
        
        buy_pct = (buy_count / total_records) * 100 if total_records > 0 else 0
        sell_pct = (sell_count / total_records) * 100 if total_records > 0 else 0
        hold_pct = (hold_count / total_records) * 100 if total_records > 0 else 0
        
        logging.info(f"Distribuzione classi per {csv_path}:")
        logging.info(f"  • {Fore.GREEN}BUY (1): {buy_count} record ({buy_pct:.1f}%){Style.RESET_ALL}")
        logging.info(f"  • {Fore.RED}SELL (2): {sell_count} record ({sell_pct:.1f}%){Style.RESET_ALL}")
        logging.info(f"  • {Fore.BLUE}HOLD (0): {hold_count} record ({hold_pct:.1f}%){Style.RESET_ALL}")
        
        # Salva il file sovrascrivendo l'originale
        df.to_csv(csv_path, index=False)
        logging.info(f"File CSV aggiornato con successo: {csv_path}")
        
    except Exception as e:
        logging.error(f"Errore durante l'assegnazione delle classi multiclasse: {e}")
        import traceback
        logging.error(traceback.format_exc())
