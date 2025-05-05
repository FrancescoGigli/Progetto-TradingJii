# Bybit Crypto Data Downloader

Questo strumento scarica dati OHLCV delle criptovalute con maggior volume di trading da Bybit per analisi e backtesting.

## Caratteristiche

- **Download automatico** dei dati OHLCV delle criptovalute con maggior volume su Bybit
- **Supporto per multipli timeframe**: 5m, 15m, 30m, 1h
- **Scaricamento intelligente** che verifica se i dati sono già aggiornati prima di effettuare nuovi download
- **Supporto per API autenticata** per un accesso ottimale ai dati
- **Database SQLite** per lo storage locale dei dati

## Requisiti

- Python 3.7+
- Dipendenze Python (vedi `requirements.txt`)
- Chiavi API Bybit (opzionali ma consigliate)

## Installazione

1. Clona questo repository:
   ```
   git clone https://github.com/yourusername/bybit-crypto-data-downloader.git
   cd bybit-crypto-data-downloader
   ```

2. Installa le dipendenze richieste:
   ```
   pip install -r requirements.txt
   ```

3. Crea un file `.env` nella directory principale con le tue chiavi API Bybit:
   ```
   BYBIT_API_KEY=la_tua_api_key
   BYBIT_API_SECRET=il_tuo_api_secret
   ```

## Utilizzo

Esegui lo script principale:

```
python start.py
```

Lo script:
1. Si connetterà a Bybit
2. Identificherà le top 100 criptovalute per volume di trading
3. Scaricherà i dati OHLCV per gli ultimi 100 giorni nei timeframe specificati
4. Salverà i dati in un database SQLite (`crypto_data.db`)

## Configurazione

È possibile configurare diversi parametri modificando le variabili all'inizio dello script `start.py`:

- `TOP_SYMBOLS_COUNT`: Numero di criptovalute con maggior volume da scaricare (default: 100)
- `DATA_LIMIT_DAYS`: Giorni di dati storici da scaricare (default: 100)
- `TIMEFRAMES`: Lista dei timeframe per cui scaricare i dati (default: ['5m', '15m', '30m', '1h'])
- `DB_FILE`: Nome del file database SQLite (default: 'crypto_data.db')

## Struttura del Database

I dati sono organizzati in tabelle separate per ogni timeframe (`data_5m`, `data_15m`, ecc.), con la seguente struttura:

- `id`: ID univoco del record
- `symbol`: Simbolo della criptovaluta (es. "BTC/USDT:USDT")
- `timestamp`: Data e ora della candela nel formato ISO
- `open`: Prezzo di apertura
- `high`: Prezzo massimo
- `low`: Prezzo minimo
- `close`: Prezzo di chiusura
- `volume`: Volume di trading nel periodo

## Licenza

[MIT](LICENSE)
