"""
Script principale per eseguire le strategie di trading TradingJii
Analizza i segnali in tempo reale e genera report
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from colorama import init, Fore, Style
import argparse

# Inizializza colorama
init()

# Import delle strategie
from strategies.rsi_mean_reversion import generate_signals as rsi_signals
from strategies.ema_crossover import generate_signals as ema_signals
from strategies.breakout_range import generate_signals as breakout_signals
from strategies.bollinger_rebound import generate_signals as bollinger_signals
from strategies.macd_histogram import generate_signals as macd_signals
from strategies.donchian_breakout import generate_signals as donchian_signals


def analyze_symbol(symbol, timeframe='1h', lookback_days=30):
    """Analizza un simbolo con tutte le strategie"""
    
    print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Analisi {symbol} - Timeframe: {timeframe}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
    
    # Connessione al database
    conn = sqlite3.connect("crypto_data.db")
    
    # Carica dati recenti
    table_name = f"market_data_{timeframe}"
    cutoff_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    
    query = f"""
    SELECT * FROM {table_name}
    WHERE symbol = ? AND timestamp >= ?
    ORDER BY timestamp
    """
    
    df = pd.read_sql_query(query, conn, params=(symbol, cutoff_date))
    conn.close()
    
    if df.empty:
        print(f"{Fore.RED}Nessun dato trovato per {symbol}{Style.RESET_ALL}")
        return None
        
    print(f"{Fore.GREEN}Dati caricati: {len(df)} record (ultimi {lookback_days} giorni){Style.RESET_ALL}")
    
    # Applica tutte le strategie
    strategies = {
        'RSI Mean Reversion': rsi_signals,
        'EMA Crossover': ema_signals,
        'Breakout Range': breakout_signals,
        'Bollinger Rebound': bollinger_signals,
        'MACD Histogram': macd_signals,
        'Donchian Breakout': donchian_signals
    }
    
    # Raccogli i segnali più recenti
    latest_signals = []
    
    for name, strategy_func in strategies.items():
        try:
            df_signals = strategy_func(df)
            
            # Trova l'ultimo segnale
            last_signal_idx = df_signals[df_signals['signal'] != 0].index
            if len(last_signal_idx) > 0:
                last_idx = last_signal_idx[-1]
                signal = df_signals.loc[last_idx, 'signal']
                timestamp = df_signals.loc[last_idx, 'timestamp']
                price = df_signals.loc[last_idx, 'close']
                
                latest_signals.append({
                    'strategy': name,
                    'signal': signal,
                    'timestamp': timestamp,
                    'price': price
                })
        except Exception as e:
            print(f"{Fore.RED}Errore in {name}: {e}{Style.RESET_ALL}")
    
    # Mostra segnali recenti
    if latest_signals:
        print(f"\n{Fore.CYAN}Ultimi segnali per strategia:{Style.RESET_ALL}")
        print("-" * 70)
        
        # Ordina per timestamp
        latest_signals.sort(key=lambda x: x['timestamp'], reverse=True)
        
        for sig in latest_signals[:10]:  # Mostra solo i 10 più recenti
            signal_type = "LONG" if sig['signal'] == 1 else "SHORT"
            color = Fore.GREEN if sig['signal'] == 1 else Fore.RED
            print(f"{sig['timestamp']} | {sig['strategy']:<20} | {color}{signal_type:<5}{Style.RESET_ALL} | ${sig['price']:,.2f}")
    
    # Analisi consensus attuale
    current_price = df.iloc[-1]['close']
    current_time = df.iloc[-1]['timestamp']
    
    print(f"\n{Fore.CYAN}Situazione attuale:{Style.RESET_ALL}")
    print(f"Prezzo: ${current_price:,.2f}")
    print(f"Timestamp: {current_time}")
    
    # Conta segnali nell'ultima candela
    consensus = {'long': 0, 'short': 0}
    
    for name, strategy_func in strategies.items():
        try:
            df_signals = strategy_func(df)
            last_signal = df_signals.iloc[-1]['signal']
            if last_signal == 1:
                consensus['long'] += 1
            elif last_signal == -1:
                consensus['short'] += 1
        except:
            pass
    
    print(f"\n{Fore.CYAN}Consensus attuale:{Style.RESET_ALL}")
    print(f"Strategie LONG: {Fore.GREEN}{consensus['long']}{Style.RESET_ALL}")
    print(f"Strategie SHORT: {Fore.RED}{consensus['short']}{Style.RESET_ALL}")
    
    if consensus['long'] >= 3:
        print(f"\n{Fore.GREEN}⚠️  FORTE SEGNALE LONG - {consensus['long']} strategie concordano!{Style.RESET_ALL}")
    elif consensus['short'] >= 3:
        print(f"\n{Fore.RED}⚠️  FORTE SEGNALE SHORT - {consensus['short']} strategie concordano!{Style.RESET_ALL}")
    
    return df


def main():
    """Funzione principale"""
    parser = argparse.ArgumentParser(description='Esegui strategie di trading TradingJii')
    parser.add_argument('--symbols', nargs='+', default=['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT'],
                        help='Simboli da analizzare')
    parser.add_argument('--timeframe', default='1h', help='Timeframe (1h, 4h, 1d)')
    parser.add_argument('--lookback', type=int, default=30, help='Giorni di lookback')
    
    args = parser.parse_args()
    
    print(f"\n{Fore.GREEN}=== TradingJii Strategy Scanner ==={Style.RESET_ALL}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Lookback: {args.lookback} giorni")
    print(f"Simboli: {', '.join(args.symbols)}")
    
    # Analizza ogni simbolo
    for symbol in args.symbols:
        analyze_symbol(symbol, args.timeframe, args.lookback)
    
    print(f"\n{Fore.GREEN}Analisi completata!{Style.RESET_ALL}")
    print(f"\n{Fore.YELLOW}Suggerimenti:{Style.RESET_ALL}")
    print("1. Per analizzare altri simboli: python run_strategies.py --symbols BTC/USDT:USDT ETH/USDT:USDT")
    print("2. Per cambiare timeframe: python run_strategies.py --timeframe 4h")
    print("3. Per vedere più storia: python run_strategies.py --lookback 60")


if __name__ == "__main__":
    main()
