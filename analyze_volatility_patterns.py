#!/usr/bin/env python3
"""
Analizzatore dei Pattern di Volatilità
=====================================

Questo script permette di visualizzare e analizzare i pattern di volatilità
nelle criptovalute, evidenziando correlazioni tra volatilità dei prezzi e
altre metriche come volume, movimento direzionale, e volatilità storica.
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns
import argparse
from datetime import datetime, timedelta
from colorama import init, Fore, Style

# Inizializza colorama per i colori cross-platform
init(autoreset=True)

# Configurazione
DB_FILE = 'crypto_data.db'
DEFAULT_TIMEFRAME = '5m'
DEFAULT_DAYS = 7
DEFAULT_LIMIT = 1000

def print_header():
    """Stampa l'intestazione del programma"""
    print("\n" + "="*80)
    print(f"{Fore.CYAN}  ANALIZZATORE DEI PATTERN DI VOLATILITÀ{Style.RESET_ALL}")
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

def fetch_volatility_data(symbol, timeframe, days=DEFAULT_DAYS):
    """Recupera i dati di volatilità dal database"""
    conn = sqlite3.connect(DB_FILE)
    
    try:
        # Calcola data di inizio
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%dT%H:%M:%S')
        
        query = f"""
            SELECT 
                timestamp, 
                open, high, low, close, volume,
                close_volatility, open_volatility, high_volatility, low_volatility,
                volume_change, historical_volatility
            FROM data_{timeframe}
            WHERE symbol = ? AND timestamp > ?
            ORDER BY timestamp ASC
        """
        
        df = pd.read_sql_query(query, conn, params=(symbol, cutoff_date))
        
        if df.empty:
            print(f"{Fore.YELLOW}Nessun dato di volatilità trovato per {symbol} ({timeframe}){Style.RESET_ALL}")
            return None
            
        # Converti timestamp in oggetti datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        conn.close()
        return df
    except Exception as e:
        print(f"{Fore.RED}Errore nel recupero dei dati: {e}{Style.RESET_ALL}")
        conn.close()
        return None

def plot_volatility_analysis(df, symbol, timeframe):
    """
    Visualizza grafici dettagliati sulla volatilità.
    
    Args:
        df: DataFrame con dati OHLCV e volatilità
        symbol: Simbolo della criptovaluta
        timeframe: Timeframe dei dati
    """
    if df is None or df.empty:
        print(f"{Fore.RED}Nessun dato disponibile per il grafico{Style.RESET_ALL}")
        return
    
    # Filtra i dati per rimuovere valori estremi che potrebbero distorcere il grafico
    df_filtered = df.copy()
    for col in ['close_volatility', 'open_volatility', 'high_volatility', 'low_volatility', 'volume_change']:
        if col in df_filtered.columns:
            # Rimuovi outlier (oltre 3 deviazioni standard)
            mean = df_filtered[col].mean()
            std = df_filtered[col].std()
            df_filtered = df_filtered[(df_filtered[col] >= mean - 3*std) & 
                                     (df_filtered[col] <= mean + 3*std)]
    
    # Crea una figura con layout con griglie
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(4, 2, figure=fig)
    
    # Grafico dei prezzi
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df_filtered.index, df_filtered['close'], label='Prezzo di chiusura', color='blue')
    ax1.set_title(f'Prezzo di {symbol} ({timeframe})', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Prezzo', fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    # Grafico della volatilità di chiusura
    ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
    ax2.plot(df_filtered.index, df_filtered['close_volatility'], label='Volatilità chiusura', color='red')
    ax2.set_title('Volatilità del prezzo di chiusura', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Volatilità (%)', fontweight='bold')
    ax2.legend(loc='upper left')
    
    # Media mobile della volatilità (per trend)
    if len(df_filtered) > 20:
        volatility_ma = df_filtered['close_volatility'].rolling(window=20).mean()
        ax2.plot(df_filtered.index, volatility_ma, label='Media mobile (20 periodi)', 
                color='orange', linestyle='--')
        ax2.legend(loc='upper left')
    
    # Grafico del volume
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    ax3.bar(df_filtered.index, df_filtered['volume'], label='Volume', color='purple', alpha=0.6)
    ax3.set_title('Volume di trading', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Volume', fontweight='bold')
    ax3.legend(loc='upper left')
    
    # Grafico della volatilità del volume
    ax4 = fig.add_subplot(gs[2, 1], sharex=ax1)
    ax4.plot(df_filtered.index, df_filtered['volume_change'], label='Volatilità volume', 
             color='green')
    ax4.set_title('Volatilità del volume', fontweight='bold', fontsize=12)
    ax4.set_ylabel('Variazione (%)', fontweight='bold')
    ax4.legend(loc='upper left')
    
    # Correlazione tra volatilità e altre metriche
    ax5 = fig.add_subplot(gs[3, 0])
    
    # Prepara dati per matrice di correlazione
    corr_columns = [col for col in ['close', 'volume', 'close_volatility', 
                                     'volume_change', 'historical_volatility'] 
                    if col in df_filtered.columns and not df_filtered[col].isna().all()]
    
    if len(corr_columns) > 1:
        corr = df_filtered[corr_columns].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', cbar=False, ax=ax5)
        ax5.set_title('Matrice di correlazione', fontweight='bold', fontsize=12)
    else:
        ax5.text(0.5, 0.5, "Dati insufficienti per la correlazione", 
                ha='center', va='center', fontsize=12)
    
    # Istogramma della volatilità di chiusura
    ax6 = fig.add_subplot(gs[3, 1])
    if 'close_volatility' in df_filtered and not df_filtered['close_volatility'].isna().all():
        sns.histplot(df_filtered['close_volatility'], kde=True, ax=ax6, color='red')
        ax6.set_title('Distribuzione della volatilità', fontweight='bold', fontsize=12)
        ax6.set_xlabel('Volatilità (%)')
    else:
        ax6.text(0.5, 0.5, "Dati di volatilità non disponibili", 
                ha='center', va='center', fontsize=12)
    
    # Aggiusta layout
    plt.tight_layout()
    fig.subplots_adjust(top=0.92)
    fig.suptitle(f'Analisi della Volatilità per {symbol} ({timeframe})', 
                fontsize=16, fontweight='bold')
    
    # Salva la figura
    output_dir = 'volatility_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'{symbol.replace("/", "_")}_{timeframe}_{current_time}.png')
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n{Fore.GREEN}Grafico salvato in: {output_file}{Style.RESET_ALL}")
    
    # Mostra il grafico
    plt.show()

def analyze_extreme_volatility(df, symbol, timeframe, threshold=1.0):
    """
    Analizza i periodi di volatilità estrema.
    
    Args:
        df: DataFrame con dati OHLCV e volatilità
        symbol: Simbolo della criptovaluta
        timeframe: Timeframe dei dati
        threshold: Soglia percentuale per considerare la volatilità estrema
    """
    if df is None or df.empty:
        print(f"{Fore.RED}Nessun dato disponibile per l'analisi{Style.RESET_ALL}")
        return
    
    if 'close_volatility' not in df.columns:
        print(f"{Fore.RED}Dati di volatilità non disponibili{Style.RESET_ALL}")
        return
    
    # Trova periodi di alta volatilità
    high_volatility = df[abs(df['close_volatility']) > threshold].copy()
    
    if high_volatility.empty:
        print(f"{Fore.YELLOW}Nessun periodo di volatilità estrema trovato (soglia: {threshold}%){Style.RESET_ALL}")
        
        # Suggerisci una soglia più bassa
        max_vol = abs(df['close_volatility']).max()
        if max_vol > 0:
            suggested_threshold = max_vol * 0.75  # 75% del valore massimo
            print(f"{Fore.GREEN}Prova con una soglia più bassa: {suggested_threshold:.2f}%{Style.RESET_ALL}")
        return
    
    # Conta eventi di volatilità positiva e negativa
    positive_events = high_volatility[high_volatility['close_volatility'] > 0]
    negative_events = high_volatility[high_volatility['close_volatility'] < 0]
    
    # Stampa statistiche
    print(f"\n{Fore.CYAN}Analisi della Volatilità Estrema per {symbol} ({timeframe}){Style.RESET_ALL}")
    print(f"Soglia: {Fore.YELLOW}{threshold}%{Style.RESET_ALL}")
    print(f"Totale eventi di volatilità estrema: {Fore.GREEN}{len(high_volatility)}{Style.RESET_ALL}")
    print(f"  • Eventi positivi: {Fore.GREEN}{len(positive_events)}{Style.RESET_ALL}")
    print(f"  • Eventi negativi: {Fore.RED}{len(negative_events)}{Style.RESET_ALL}")
    
    # Statistiche sulla volatilità
    if not high_volatility.empty:
        avg_vol = high_volatility['close_volatility'].mean()
        max_vol = high_volatility['close_volatility'].max()
        min_vol = high_volatility['close_volatility'].min()
        
        print(f"\n{Fore.CYAN}Statistiche di Volatilità:{Style.RESET_ALL}")
        print(f"  • Volatilità media: {Fore.YELLOW}{avg_vol:.4f}%{Style.RESET_ALL}")
        print(f"  • Volatilità massima positiva: {Fore.GREEN}{max_vol:.4f}%{Style.RESET_ALL}")
        print(f"  • Volatilità massima negativa: {Fore.RED}{min_vol:.4f}%{Style.RESET_ALL}")
    
    # Statistiche di volume durante alta volatilità
    if 'volume' in high_volatility.columns and 'volume' in df.columns:
        avg_vol_high = high_volatility['volume'].mean()
        avg_vol_normal = df['volume'].mean()
        vol_ratio = avg_vol_high / avg_vol_normal if avg_vol_normal > 0 else 0
        
        print(f"\n{Fore.CYAN}Relazione con il Volume:{Style.RESET_ALL}")
        print(f"  • Volume medio durante volatilità estrema: {Fore.YELLOW}{avg_vol_high:.2f}{Style.RESET_ALL}")
        print(f"  • Volume medio normale: {Fore.YELLOW}{avg_vol_normal:.2f}{Style.RESET_ALL}")
        print(f"  • Rapporto: {Fore.GREEN}{vol_ratio:.2f}x{Style.RESET_ALL} " +
             f"({Fore.GREEN if vol_ratio > 1 else Fore.RED}{(vol_ratio-1)*100:.2f}%{Style.RESET_ALL})")
    
    # Mostra i top 5 eventi di volatilità (positivi e negativi)
    print(f"\n{Fore.CYAN}Top 5 Eventi di Volatilità Positiva:{Style.RESET_ALL}")
    if not positive_events.empty:
        top_positive = positive_events.sort_values('close_volatility', ascending=False).head(5)
        for idx, row in top_positive.iterrows():
            timestamp = idx.strftime('%Y-%m-%d %H:%M')
            print(f"  • {Fore.GREEN}{timestamp}{Style.RESET_ALL}: {Fore.YELLOW}{row['close_volatility']:.4f}%{Style.RESET_ALL}, " +
                 f"Prezzo: {row['close']:.4f}, Volume: {row['volume']:.2f}")
    else:
        print(f"  {Fore.YELLOW}Nessun evento positivo trovato{Style.RESET_ALL}")
    
    print(f"\n{Fore.CYAN}Top 5 Eventi di Volatilità Negativa:{Style.RESET_ALL}")
    if not negative_events.empty:
        top_negative = negative_events.sort_values('close_volatility', ascending=True).head(5)
        for idx, row in top_negative.iterrows():
            timestamp = idx.strftime('%Y-%m-%d %H:%M')
            print(f"  • {Fore.RED}{timestamp}{Style.RESET_ALL}: {Fore.YELLOW}{row['close_volatility']:.4f}%{Style.RESET_ALL}, " +
                 f"Prezzo: {row['close']:.4f}, Volume: {row['volume']:.2f}")
    else:
        print(f"  {Fore.YELLOW}Nessun evento negativo trovato{Style.RESET_ALL}")

def main():
    parser = argparse.ArgumentParser(description="Analizza i pattern di volatilità delle criptovalute")
    parser.add_argument("--symbol", type=str, help="Simbolo da analizzare (es. BTC/USDT:USDT)")
    parser.add_argument("--timeframe", type=str, default=DEFAULT_TIMEFRAME, help="Timeframe da analizzare (es. 5m, 15m)")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS, help="Numero di giorni di storia da analizzare")
    parser.add_argument("--list", action="store_true", help="Mostra solo la lista dei simboli disponibili")
    parser.add_argument("--threshold", type=float, default=1.0, help="Soglia per la volatilità estrema (percentuale)")
    parser.add_argument("--no-plot", action="store_true", help="Non generare grafici")
    
    args = parser.parse_args()
    
    print_header()
    
    # Recupera timeframes disponibili
    timeframes = get_available_timeframes()
    if not timeframes:
        print(f"{Fore.RED}Nessun timeframe trovato nel database. Eseguire prima il pipeline di volatilità.{Style.RESET_ALL}")
        return
    
    print(f"{Fore.GREEN}Timeframes disponibili: {', '.join(timeframes)}{Style.RESET_ALL}")
    
    # Se non è specificato un timeframe, usa il primo disponibile
    timeframe = args.timeframe if args.timeframe in timeframes else timeframes[0]
    print(f"Usando timeframe: {Fore.GREEN}{timeframe}{Style.RESET_ALL}")
    
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
    symbol = args.symbol if args.symbol in symbols else symbols[0]
    print(f"Analizzando: {Fore.YELLOW}{symbol}{Style.RESET_ALL} (ultimi {Fore.GREEN}{args.days}{Style.RESET_ALL} giorni)")
    
    # Fetch and plot data
    df = fetch_volatility_data(symbol, timeframe, args.days)
    
    if df is not None:
        # Analizza la volatilità estrema
        analyze_extreme_volatility(df, symbol, timeframe, args.threshold)
        
        # Plot volatility analysis
        if not args.no_plot:
            plot_volatility_analysis(df, symbol, timeframe)
    else:
        print(f"{Fore.RED}Impossibile recuperare i dati di volatilità per {symbol} ({timeframe}).{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
