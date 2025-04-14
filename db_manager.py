import sqlite3
import config

DB_FILE = config.DB_FILE

def init_data_tables():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Usa i timeframe aggiornati dalla configurazione
    enabled_timeframes = config.ENABLED_TIMEFRAMES

    # Se RESET_DB_ON_STARTUP è True, elimina le tabelle esistenti per i timeframe abilitati
    if config.RESET_DB_ON_STARTUP:
        for tf in enabled_timeframes:
            table_name = f"data_{tf}"
            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
    
    # Crea le tabelle per ogni timeframe abilitato, se non esistono già
    for tf in enabled_timeframes:
        table_name = f"data_{tf}"
        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {table_name} (
                timestamp TEXT,
                symbol TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                ema5 REAL,
                ema10 REAL,
                ema20 REAL,
                macd REAL,
                macd_signal REAL,
                rsi_fast REAL,
                stoch_rsi REAL,
                atr REAL,
                bollinger_hband REAL,
                bollinger_lband REAL,
                vwap REAL,
                adx REAL,
                roc REAL,
                weekday INTEGER,
                hour INTEGER,
                PRIMARY KEY (timestamp, symbol)
            )
        ''')
    conn.commit()
    conn.close()

def save_data(symbol, df, timeframe):
    table = f"data_{timeframe}"
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    for timestamp, row in df.iterrows():
        cursor.execute(f'''
            INSERT OR REPLACE INTO {table} (
                timestamp, symbol, open, high, low, close, volume,
                ema5, ema10, ema20, macd, macd_signal,
                rsi_fast, stoch_rsi, atr, bollinger_hband, bollinger_lband,
                vwap, adx, roc, weekday, hour
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            timestamp.isoformat(),
            symbol,
            row.get('open', 0),
            row.get('high', 0),
            row.get('low', 0),
            row.get('close', 0),
            row.get('volume', 0),
            row.get('ema5', 0),
            row.get('ema10', 0),
            row.get('ema20', 0),
            row.get('macd', 0),
            row.get('macd_signal', 0),
            row.get('rsi_fast', 0),
            row.get('stoch_rsi', 0),
            row.get('atr', 0),
            row.get('bollinger_hband', 0),
            row.get('bollinger_lband', 0),
            row.get('vwap', 0),
            row.get('adx', 0),
            row.get('roc', 0),
            row.get('weekday', 0),
            row.get('hour', 0)
        ))
    conn.commit()
    conn.close()
