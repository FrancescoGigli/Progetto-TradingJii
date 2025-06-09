"""
Test delle strategie di trading con dati reali dal database TradingJii
"""

import sqlite3
import pandas as pd
from colorama import init, Fore, Style

# Inizializza colorama per Windows
init()

# Import delle strategie
from strategies.rsi_mean_reversion import generate_signals as rsi_signals
from strategies.ema_crossover import generate_signals as ema_signals
from strategies.breakout_range import generate_signals as breakout_signals
from strategies.bollinger_rebound import generate_signals as bollinger_signals
from strategies.macd_histogram import generate_signals as macd_signals
from strategies.donchian_breakout import generate_signals as donchian_signals
from strategies.adx_filter_crossover import generate_signals as adx_filter_signals


def test_strategy_on_symbol(symbol, timeframe='1h'):
    """Testa tutte le strategie su un simbolo specifico"""
    
    print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Testing {symbol} on {timeframe} timeframe{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    
    # Connessione al database
    conn = sqlite3.connect("crypto_data.db")
    
    # Carica i dati
    table_name = f"market_data_{timeframe}"
    query = f"""
    SELECT * FROM {table_name}
    WHERE symbol = ?
    ORDER BY timestamp
    """
    
    try:
        df = pd.read_sql_query(query, conn, params=(symbol,))
        conn.close()
        
        if df.empty:
            print(f"{Fore.RED}Nessun dato trovato per {symbol}{Style.RESET_ALL}")
            return
            
        print(f"{Fore.GREEN}Dati caricati: {len(df)} record{Style.RESET_ALL}")
        print(f"Periodo: {df['timestamp'].iloc[0]} - {df['timestamp'].iloc[-1]}")
        
        # Testa ogni strategia
        strategies = {
            'RSI Mean Reversion': rsi_signals,
            'EMA Crossover': ema_signals,
            'Breakout Range': breakout_signals,
            'Bollinger Rebound': bollinger_signals,
            'MACD Histogram': macd_signals,
            'Donchian Breakout': donchian_signals,
            'ADX Filter Crossover': adx_filter_signals
        }
        
        print(f"\n{Fore.CYAN}Risultati delle strategie:{Style.RESET_ALL}")
        print("-" * 40)
        
        for name, strategy_func in strategies.items():
            try:
                # Applica la strategia
                df_with_signals = strategy_func(df)
                
                # Conta i segnali
                long_signals = (df_with_signals['signal'] == 1).sum()
                short_signals = (df_with_signals['signal'] == -1).sum()
                total_signals = long_signals + short_signals
                
                print(f"\n{Fore.YELLOW}{name}:{Style.RESET_ALL}")
                print(f"  Segnali LONG:  {Fore.GREEN}{long_signals}{Style.RESET_ALL}")
                print(f"  Segnali SHORT: {Fore.RED}{short_signals}{Style.RESET_ALL}")
                print(f"  Totale:        {total_signals}")
                
                # Mostra gli ultimi 3 segnali
                recent_signals = df_with_signals[df_with_signals['signal'] != 0].tail(3)
                if not recent_signals.empty:
                    print(f"  {Fore.CYAN}Ultimi segnali:{Style.RESET_ALL}")
                    for _, row in recent_signals.iterrows():
                        signal_type = "LONG" if row['signal'] == 1 else "SHORT"
                        color = Fore.GREEN if row['signal'] == 1 else Fore.RED
                        print(f"    {row['timestamp']} - {color}{signal_type}{Style.RESET_ALL} @ ${row['close']:.2f}")
                        
            except Exception as e:
                print(f"\n{Fore.YELLOW}{name}:{Style.RESET_ALL}")
                print(f"  {Fore.RED}Errore: {e}{Style.RESET_ALL}")
                
    except Exception as e:
        print(f"{Fore.RED}Errore nel caricamento dati: {e}{Style.RESET_ALL}")
        conn.close()


def check_available_symbols():
    """Verifica quali simboli sono disponibili nel database"""
    conn = sqlite3.connect("crypto_data.db")
    
    # Controlla la tabella 1h come riferimento
    query = """
    SELECT DISTINCT symbol, COUNT(*) as records
    FROM market_data_1h
    GROUP BY symbol
    ORDER BY records DESC
    """
    
    try:
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        print(f"\n{Fore.CYAN}Simboli disponibili nel database:{Style.RESET_ALL}")
        print("-" * 40)
        
        for _, row in df.iterrows():
            print(f"{Fore.YELLOW}{row['symbol']}{Style.RESET_ALL}: {row['records']} record")
            
        return df['symbol'].tolist()
        
    except Exception as e:
        print(f"{Fore.RED}Errore: {e}{Style.RESET_ALL}")
        conn.close()
        return []


def main():
    """Funzione principale"""
    print(f"\n{Fore.GREEN}=== Test Strategie TradingJii ==={Style.RESET_ALL}")
    
    # Verifica simboli disponibili
    symbols = check_available_symbols()
    
    if not symbols:
        print(f"{Fore.RED}Nessun dato trovato nel database!{Style.RESET_ALL}")
        return
    
    # Testa le strategie sui primi 3 simboli
    test_symbols = symbols[:3]
    
    for symbol in test_symbols:
        test_strategy_on_symbol(symbol)
    
    print(f"\n{Fore.GREEN}Test completato!{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
