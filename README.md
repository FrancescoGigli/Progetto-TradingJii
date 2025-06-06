# TradingJii - Crypto Data Collector

Sistema di raccolta dati crypto configurato per scaricare dati storici di BTC, ETH e SOL negli ultimi 365 giorni.

## ⚡ Quick Start

### Comando per Raccogliere Dati BTC, ETH e SOL
```bash
python data_collector.py
```

Il sistema è **preconfigurato** per scaricare automaticamente:
- **BTC/USDT:USDT**, **ETH/USDT:USDT**, **SOL/USDT:USDT**
- **365 giorni** di dati storici
- **Timeframes**: 1h e 4h
- **Con retry automatico** per errori di rete

## 📋 Configurazione Attuale

Il sistema è preconfigurato per:
- **Simboli**: BTC/USDT:USDT, ETH/USDT:USDT, SOL/USDT:USDT
- **Periodo**: 365 giorni di dati storici (più buffer per indicatori tecnici)
- **Timeframes**: 1h e 4h
- **Exchange**: Bybit

## 🚀 Installazione e Setup

### 1. Installare le Dipendenze
```bash
pip install -r requirements.txt
```

### 2. Configurare le API Keys
Crea un file `.env` nella directory principale del progetto:
```
BYBIT_API_KEY=your_bybit_api_key_here
BYBIT_API_SECRET=your_bybit_api_secret_here
```

**Nota**: Le API keys sono necessarie solo per evitare rate limits. Il sistema può funzionare anche senza credenziali per download limitati.

### 3. Eseguire il Data Collector
```bash
python data_collector.py
```

## ⚙️ Configurazioni Principali

### Simboli Specifici (Configurazione Attuale)
Nel file `modules/utils/config.py`:
```python
REALTIME_CONFIG = {
    'specific_symbols': [
        'BTC/USDT:USDT',
        'ETH/USDT:USDT', 
        'SOL/USDT:USDT'
    ],
    'use_specific_symbols': True,  # Usa simboli specifici
}
```

### Cambiar Simboli
Per modificare i simboli da scaricare, edita la lista `specific_symbols` in `modules/utils/config.py`:
```python
'specific_symbols': [
    'BTC/USDT:USDT',
    'ETH/USDT:USDT', 
    'SOL/USDT:USDT',
    'ADA/USDT:USDT',  # Aggiungi altri simboli
],
```

### Usare Top Simboli per Volume
Per usare i top N simboli per volume invece di simboli specifici:
```python
REALTIME_CONFIG = {
    'num_symbols': 10,  # Numero di top simboli
    'use_specific_symbols': False,  # Usa top per volume
}
```

## 🛠️ Opzioni da Linea di Comando

### Opzioni Base
```bash
# Scarica solo 3 simboli per 180 giorni
python data_collector.py -n 3 -d 180

# Usa timeframes specifici
python data_collector.py -t 1h 4h 1d

# Modalità sequenziale (default)
python data_collector.py --sequential

# Modalità parallela (più veloce)
python data_collector.py --parallel
```

### Opzioni Avanzate
```bash
# Disabilita indicatori tecnici (più veloce)
python data_collector.py --no-ta

# Salta validazione dati
python data_collector.py --skip-validation

# Esporta report di validazione
python data_collector.py --export-validation-report

# Aumenta concorrenza (attenzione ai rate limits)
python data_collector.py -c 10 -b 20
```

## 📊 Output

### Database
I dati vengono salvati nel file `crypto_data.db` (SQLite).

### Tabelle Principali
- `ohlcv_[timeframe]`: Dati OHLCV (Open, High, Low, Close, Volume)
- `indicators_[timeframe]`: Indicatori tecnici (RSI, MACD, EMA, etc.)
- `volatility_[timeframe]`: Dati di volatilità

### Log
I log vengono mostrati a console con colori per facilità di lettura.

## 🔧 Configurazioni Avanzate

### Modificare Timeframes
Nel file `modules/utils/config.py`:
```python
DEFAULT_TIMEFRAMES = ['1h', '4h', '1d']  # Aggiungi/rimuovi timeframes
```

### Modificare Periodo di Analisi
```python
DESIRED_ANALYSIS_DAYS = 365  # Giorni di dati per analisi
```

### Indicatori Tecnici
Per modificare i parametri degli indicatori, edita `TA_PARAMS` in `config.py`:
```python
TA_PARAMS = {
    'rsi14': {'timeperiod': 14},
    'ema20': {'timeperiod': 20},
    'sma50': {'timeperiod': 50},
    # ... altri indicatori
}
```

## 📈 Utilizzo Tipico

### Download Singolo
```bash
# Download immediato con configurazione attuale
python data_collector.py
```

### Monitoraggio Continuo
Il sistema include un loop automatico che aggiorna i dati ogni 5 minuti. Per interrompere, premi `Ctrl+C`.

### Test Rapido
```bash
# Test veloce: solo BTC, 30 giorni, senza indicatori tecnici
python data_collector.py -d 30 --no-ta --skip-validation
```

## 🚨 Note Importanti

1. **Rate Limits**: Bybit ha limiti di richieste. Usa le API keys per limiti più alti.
2. **Spazio Disco**: 365 giorni di dati per 3 simboli su 2 timeframes = ~50MB.
3. **Tempo Download**: ~2-5 minuti per la configurazione attuale.
4. **Interruzioni**: Il sistema gestisce interruzioni di rete e retry automatici.

## 🔍 Troubleshooting

### Errore API Keys
```
Errore: Unauthorized
```
Verifica che le API keys nel file `.env` siano corrette.

### Errore Rate Limit
```
Errore: Rate limit exceeded
```
Riduci la concorrenza con `-c 3` o aggiungi delay.

### Database Locked
```
Errore: Database is locked
```
Chiudi altre istanze del programma che potrebbero usare il database.
