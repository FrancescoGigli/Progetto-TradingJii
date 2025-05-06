# Ottimizzazioni della Pipeline di Analisi della Volatilità

Questo documento descrive le ottimizzazioni implementate nel sistema di analisi della volatilità per migliorare le prestazioni e ridurre i tempi di elaborazione.

## Ottimizzazioni Implementate

### 1. Parallelizzazione del Processo

La versione precedente processava un simbolo alla volta sequenzialmente. La nuova implementazione:

- Utilizza `ThreadPoolExecutor` per elaborare più simboli contemporaneamente
- Limita il numero di thread paralleli a 2 per evitare conflitti di accesso al database
- Fornisce una versione sincrona dell'elaborazione simboli per uso parallelo

```python
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(process_symbol_volatility_sync, symbol, timeframe, lookback_days) 
              for symbol in symbols]
    
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        # elaborazione risultati
```

### 2. Ottimizzazione delle Query SQL

L'aggiornamento delle colonne di volatilità ora utilizza operazioni batch invece di riga per riga:

- Raccoglie tutti gli aggiornamenti in un'unica lista di tuple
- Utilizza `executemany` per aggiornare più righe con un'unica chiamata al database
- Costruisce dinamicamente le query SQL in base alle colonne disponibili

```python
# Crea la query dinamica in base alle colonne disponibili
set_clause = ", ".join([f"{col} = ?" for col in update_cols])
query = f"""
    UPDATE data_{timeframe}
    SET {set_clause}
    WHERE symbol = ? AND timestamp = ?
"""

# Esegue l'update in un'unica transazione
cursor.executemany(query, update_data)
```

### 3. Calcolo Vettoriale delle Volatilità

Il calcolo della volatilità è stato ottimizzato:

- Integrazione del calcolo di `historical_volatility` direttamente nella funzione `calculate_volatility_rate`
- Riduzione delle scansioni multiple del DataFrame
- Uso di operazioni vettoriali di pandas più efficienti

```python
# Calculate historical volatility (integrato nella stessa funzione)
if 'close' in df_vol.columns:
    # Calcola i log return
    df_vol['log_return'] = np.log(df_vol['close'] / df_vol['close'].shift(1))
    
    # Calcola la volatilità storica
    df_vol['historical_volatility'] = df_vol['log_return'].rolling(window=window).std() * np.sqrt(252)
    
    # Rimuove la colonna temporanea
    df_vol.drop('log_return', axis=1, inplace=True, errors='ignore')
```

### 4. Caching dei Risultati Intermedi

Implementato un sistema di cache per evitare ricalcoli non necessari:

- Verifica se esistono già dati di volatilità calcolati negli ultimi 30 minuti
- Salta il ricalcolo se sono disponibili dati recenti
- Riduce significativamente il carico per esecuzioni ripetute in breve tempo

```python
# Implementazione della cache
cursor.execute(f"""
    SELECT COUNT(*) FROM data_{timeframe}
    WHERE symbol = ? AND close_volatility IS NOT NULL
    AND timestamp > datetime('now', '-30 minutes')
""", (symbol,))
recent_count = cursor.fetchone()[0]

if recent_count > 10:  # Se ci sono più di 10 record con dati di volatilità recenti
    logging.info(f"Recent volatility data found for {symbol} ({timeframe}), skipping recalculation")
    return pd.DataFrame(), {}  # Salta calcolo
```

### 5. Riduzione del Database I/O

Ottimizzazioni specifiche per SQLite:

- Utilizzo della modalità WAL (Write-Ahead Logging) per migliorare le prestazioni di scrittura
- Aggiunta di un indice composto su (symbol, timestamp) per query più veloci
- Configurazione di un timeout per le connessioni al database per gestire i conflitti
- Utilizzo di connessioni separate per ogni thread di elaborazione

```python
# Abilita WAL mode per prestazioni superiori
conn.execute('PRAGMA journal_mode = WAL')

# Timeout per evitare attese indefinite
conn = sqlite3.connect(DB_FILE, timeout=30.0)

# Indici ottimizzati
cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_symbol_timestamp ON {table_name} (symbol, timestamp)")
```

## Risultati

Le ottimizzazioni hanno portato a:

1. **Velocità di elaborazione**: Tempi di elaborazione significativamente ridotti, soprattutto per dataset più grandi
2. **Utilizzo delle risorse**: Migliore utilizzo della CPU grazie alla parallelizzazione
3. **Efficienza del database**: Minor numero di operazioni I/O e query più veloci
4. **Resistenza agli errori**: Migliore gestione dei conflitti di accesso al database

## Test delle Prestazioni

È stato sviluppato uno script di test (`test_optimized_volatility.py`) per misurare e confrontare le prestazioni:

```
python test_optimized_volatility.py --symbols 5 --days 30 --timeframe 5m
```

Questo script aiuta a misurare il tempo di elaborazione e verificare l'efficacia delle ottimizzazioni.

## Ulteriori Ottimizzazioni Possibili

- Implementazione di una cache più sofisticata utilizzando Redis o un altro sistema di cache
- Ottimizzazione della gestione della memoria per dataset molto grandi
- Distribuzione dell'elaborazione su più processi invece dei thread per aggirare il GIL di Python
