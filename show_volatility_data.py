#!/usr/bin/env python3
"""
Visualizzatore Dati di Volatilità
=================================

Questo script mostra i dati di volatilità salvati nel database per permettere
un'ispezione rapida delle metriche calcolate.
"""

import os
import sys
import sqlite3
import pandas as pd
import argparse
from datetime import datetime, timedelta
from colorama import init, Fore, Style, Back
from tabulate import tabulate

# Inizializza colorama per colori cross-platform
init(autoreset=True)

# Configurazione
DB_FILE = 'crypto_data.db'

def print_header():
    """Stampa l'intestazione del programma"""
    print("\n" + "="*80)
    print(f"{Fore.CYAN}  VISUALIZZATORE DATI DI VOLATILITÀ{Style.RESET_ALL}")
    print("="*80 + "\n")

def get_available_symbols(timeframe):
    """Ottiene tutti i simboli disponibili per un timeframe specifico"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    try:
        cursor.execute(f"""
            SELECT DISTINCT symbol FROM data_{timeframe}
            WHERE close_volatility IS NOT NULL
            ORDER BY symbol
        """)
        symbols = [row[0] for row in cursor.fetchall()]
        return symbols
    except Exception as e:
        print(f"{Fore.RED}Errore nel recupero dei simboli: {e}{Style.RESET_ALL}")
        return []
    finally:
        conn.close()

def get_available_timeframes():
    """Ottiene tutti i timeframe disponibili nel database"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'data_%'")
        tables = [row[0].replace('data_', '') for row in cursor.fetchall()]
        return tables
    except Exception as e:
        print(f"{Fore.RED}Errore nel recupero dei timeframe: {e}{Style.RESET_ALL}")
        return []
    finally:
        conn.close()

def fetch_volatility_data(symbol, timeframe, limit=10):
    """Recupera i dati di volatilità dal database"""
    conn = sqlite3.connect(DB_FILE)
    
    try:
        query = f"""
            SELECT 
                timestamp, 
                open, high, low, close, volume,
                close_volatility, open_volatility, high_volatility, low_volatility,
                volume_change, historical_volatility
            FROM data_{timeframe}
            WHERE symbol = ? AND close_volatility IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT {limit}
        """
        
        df = pd.read_sql_query(query, conn, params=(symbol,))
        
        if df.empty:
            print(f"{Fore.YELLOW}Nessun dato di volatilità trovato per {symbol} ({timeframe}){Style.RESET_ALL}")
            return None
            
        # Converti timestamp in oggetti datetime più leggibili
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        
        # Rinomina le colonne per la visualizzazione
        df.rename(columns={
            'timestamp': 'Data/Ora',
            'open': 'Apertura',
            'high': 'Massimo',
            'low': 'Minimo',
            'close': 'Chiusura',
            'volume': 'Volume',
            'close_volatility': 'Vol. Chiusura',
            'open_volatility': 'Vol. Apertura',
            'high_volatility': 'Vol. Massimo',
            'low_volatility': 'Vol. Minimo',
            'volume_change': 'Var. Volume',
            'historical_volatility': 'Vol. Storica'
        }, inplace=True)
        
        return df
    except Exception as e:
        print(f"{Fore.RED}Errore nel recupero dei dati: {e}{Style.RESET_ALL}")
        return None
    finally:
        conn.close()

def show_volatility_stats(symbol, timeframe):
    """Visualizza statistiche di volatilità per un simbolo e timeframe"""
    conn = sqlite3.connect(DB_FILE)
    
    try:
        query = f"""
            SELECT 
                COUNT(*) as total_records,
                MIN(timestamp) as first_date,
                MAX(timestamp) as last_date,
                AVG(close_volatility) as avg_volatility,
                MAX(close_volatility) as max_volatility
            FROM data_{timeframe}
            WHERE symbol = ? AND close_volatility IS NOT NULL
        """
        
        df = pd.read_sql_query(query, conn, params=(symbol,))
        
        if df.empty or df['total_records'].iloc[0] == 0:
            print(f"{Fore.YELLOW}Nessuna statistica di volatilità disponibile per {symbol} ({timeframe}){Style.RESET_ALL}")
            return
            
        # Converti date in formato leggibile
        first_date = pd.to_datetime(df['first_date'].iloc[0]).strftime('%Y-%m-%d %H:%M')
        last_date = pd.to_datetime(df['last_date'].iloc[0]).strftime('%Y-%m-%d %H:%M')
        days_span = (pd.to_datetime(df['last_date'].iloc[0]) - pd.to_datetime(df['first_date'].iloc[0])).days
        
        # Mostra statistiche
        print(f"\n{Fore.CYAN}Statistiche per {Fore.YELLOW}{symbol}{Fore.CYAN} ({timeframe}){Style.RESET_ALL}")
        print(f"  • Periodo: {Fore.GREEN}{first_date}{Style.RESET_ALL} - {Fore.GREEN}{last_date}{Style.RESET_ALL} ({days_span} giorni)")
        print(f"  • Totale record: {Fore.GREEN}{df['total_records'].iloc[0]}{Style.RESET_ALL}")
        print(f"  • Volatilità media chiusura: {Fore.MAGENTA}{df['avg_volatility'].iloc[0]:.6f}{Style.RESET_ALL}")
        print(f"  • Volatilità massima chiusura: {Fore.RED}{df['max_volatility'].iloc[0]:.6f}{Style.RESET_ALL}")
        
    except Exception as e:
        print(f"{Fore.RED}Errore nel recupero delle statistiche: {e}{Style.RESET_ALL}")
    finally:
        conn.close()

def main():
    parser = argparse.ArgumentParser(description="Visualizza dati di volatilità dal database")
    parser.add_argument("--symbol", type=str, help="Simbolo da visualizzare (es. BTC/USDT:USDT)")
    parser.add_argument("--timeframe", type=str, help="Timeframe da visualizzare (es. 5m, 15m)")
    parser.add_argument("--limit", type=int, default=10, help="Numero di record da visualizzare")
    parser.add_argument("--list", action="store_true", help="Mostra solo la lista dei simboli disponibili")
    
    args = parser.parse_args()
    
    print_header()
    
    # Recupera timeframes disponibili
    timeframes = get_available_timeframes()
    if not timeframes:
        print(f"{Fore.RED}Nessun timeframe trovato nel database. Eseguire prima il pipeline di volatilità.{Style.RESET_ALL}")
        return
    
    print(f"{Fore.GREEN}Timeframes disponibili: {', '.join(timeframes)}{Style.RESET_ALL}")
    
    # Se non è specificato un timeframe, usa il primo disponibile
    timeframe = args.timeframe if args.timeframe else timeframes[0]
    if timeframe not in timeframes:
        print(f"{Fore.RED}Timeframe {timeframe} non trovato. Utilizzare uno dei seguenti: {', '.join(timeframes)}{Style.RESET_ALL}")
        return
    
    # Recupera simboli disponibili
    symbols = get_available_symbols(timeframe)
    if not symbols:
        print(f"{Fore.RED}Nessun simbolo con dati di volatilità trovato per il timeframe {timeframe}.{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Eseguire prima il pipeline di volatilità con 'python main.py' o 'python quick_test.py'.{Style.RESET_ALL}")
        return
    
    # Mostra solo la lista dei simboli se richiesto
    if args.list:
        print(f"\n{Fore.CYAN}Simboli disponibili per timeframe {timeframe}:{Style.RESET_ALL}")
        for i, sym in enumerate(symbols):
            print(f"  {i+1}. {Fore.YELLOW}{sym}{Style.RESET_ALL}")
        return
    
    # Se non è specificato un simbolo, usa il primo disponibile
    symbol = args.symbol if args.symbol else symbols[0]
    if symbol not in symbols:
        print(f"{Fore.RED}Simbolo {symbol} non trovato con dati di volatilità. Utilizzare uno dei seguenti: {', '.join(symbols[:5])}...{Style.RESET_ALL}")
        return
    
    # Mostra statistiche
    show_volatility_stats(symbol, timeframe)
    
    # Recupera e mostra dati
    df = fetch_volatility_data(symbol, timeframe, args.limit)
    if df is not None:
        print(f"\n{Fore.CYAN}Ultimi {args.limit} record per {Fore.YELLOW}{symbol}{Fore.CYAN} ({timeframe}):{Style.RESET_ALL}")
        
        # Formato per visualizzazione migliore
        format_dict = {
            'Apertura': '{:.2f}',
            'Massimo': '{:.2f}',
            'Minimo': '{:.2f}',
            'Chiusura': '{:.2f}',
            'Volume': '{:,.2f}',
            'Vol. Chiusura': '{:.6f}',
            'Vol. Apertura': '{:.6f}',
            'Vol. Massimo': '{:.6f}',
            'Vol. Minimo': '{:.6f}',
            'Var. Volume': '{:.6f}',
            'Vol. Storica': '{:.6f}'
        }
        
        # Visualizza tabella con tabulate per output formattato
        try:
            # Importazione condizionale di tabulate
            from tabulate import tabulate
            print(tabulate(df, headers='keys', tablefmt='grid', floatfmt='.6f', showindex=False))
        except ImportError:
            # Fallback a formato pandas standard se tabulate non è disponibile
            for col, fmt in format_dict.items():
                if col in df.columns:
                    df[col] = df[col].map(lambda x: fmt.format(x) if pd.notnull(x) else 'N/A')
            print(df.to_string(index=False))
        
        print(f"\n{Fore.GREEN}Nota: La volatilità è rappresentata come variazione relativa tra i periodi{Style.RESET_ALL}")
        print(f"{Fore.GREEN}      I valori più alti indicano cambiamenti più significativi nei prezzi{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
