# TradingJii Data Collector - Documentazione Dettagliata

## Struttura del Progetto

```
/
├── data_collector.py         # Script principale del data collector
├── config.json               # File di configurazione generale
├── requirements.txt          # Dipendenze Python
├── ml_system/                # Sistema di machine learning
│   ├── backtesting/          # Moduli per il backtesting delle strategie
│   ├── data_labeling/        # Etichettatura dei dati per ML
│   ├── feature_engineering/  # Estrazione e preparazione feature
│   ├── logs/                 # Directory per i log
│   │   └── predictions/      # Log delle predizioni
│   ├── models/               # Modelli di machine learning
│   │   ├── __init__.py       # Inizializzazione package
│   │   ├── trainer.py        # Addestramento modelli
│   │   ├── predictor.py      # Predizione usando modelli addestrati
│   │   └── ensemble.py       # Modelli ensemble
│   ├── signal_scanner/       # Scanner per segnali di trading
│   └── utils/                # Utilità per il ML
└── modules/                  # Moduli principali del sistema
    ├── __init__.py           # Inizializzazione package
    ├── core/                 # Componenti core
    │   ├── __init__.py       # Inizializzazione package
    │   ├── data_fetcher.py   # Scaricamento dati OHLCV
    │   ├── download_orchestrator.py  # Orchestrazione dei download
    │   └── exchange.py       # Interfaccia con gli exchange
    ├── data/                 # Elaborazione dati
    │   ├── __init__.py       # Inizializzazione package
    │   ├── data_integrity_checker.py  # Verifica integrità dati
    │   ├── data_labeler.py   # Etichettatura dei dati
    │   ├── data_validator.py # Validazione dei dati
    │   ├── dataset_generator.py  # Generazione dataset
    │   ├── db_manager.py     # Gestione database SQLite
    │   ├── indicator_processor.py  # Calcolo indicatori tecnici
    │   ├── series_segmenter.py  # Segmentazione serie temporali
    │   └── volatility_processor.py  # Analisi volatilità
    └── utils/                # Utilità generali
        ├── __init__.py       # Inizializzazione package
        ├── command_args.py   # Gestione argomenti linea comando
        ├── config.py         # Gestione configurazione
        ├── logging_setup.py  # Setup del sistema di logging
        └── symbol_manager.py # Gestione simboli criptovalute
```

## Descrizione dei File Principali

### File di Base

- **data_collector.py**: Script principale che avvia e gestisce l'intero processo di raccolta dati. Implementa il ciclo di aggiornamento continuo e coordina tutte le operazioni.
- **config.json**: Contiene la configurazione globale del sistema, inclusi parametri per gli exchange, database, e comportamento del data collector.
- **requirements.txt**: Elenco delle dipendenze Python necessarie per eseguire il sistema.

### Moduli Core

- **modules/core/exchange.py**: Gestisce la connessione con gli exchange di criptovalute, implementando un'interfaccia unificata basata su ccxt.
- **modules/core/data_fetcher.py**: Responsabile del download effettivo dei dati OHLCV, con gestione degli errori e meccanismo di retry.
- **modules/core/download_orchestrator.py**: Coordina i download paralleli o sequenziali, gestendo la concorrenza e monitorando lo stato.

### Gestione Dati

- **modules/data/db_manager.py**: Gestisce tutte le operazioni sul database SQLite, inclusi creazione tabelle, query e ottimizzazioni.
- **modules/data/indicator_processor.py**: Calcola tutti gli indicatori tecnici (SMA, EMA, RSI, MACD, etc.) utilizzando numpy e pandas.
- **modules/data/data_integrity_checker.py**: Verifica l'integrità e la qualità dei dati scaricati, identificando gap e anomalie.
- **modules/data/data_validator.py**: Convalida i dati prima del salvataggio, assicurando che rispettino i vincoli definiti.
- **modules/data/data_labeler.py**: Etichetta i dati per l'addestramento di modelli di machine learning.
- **modules/data/volatility_processor.py**: Analizza e calcola metriche di volatilità per le serie temporali.
- **modules/data/series_segmenter.py**: Segmenta le serie temporali in pattern significativi.
- **modules/data/dataset_generator.py**: Genera dataset strutturati per l'addestramento dei modelli ML.

### Utilità

- **modules/utils/config.py**: Carica e gestisce le configurazioni da config.json.
- **modules/utils/command_args.py**: Gestisce il parsing degli argomenti da linea di comando.
- **modules/utils/logging_setup.py**: Configura il sistema di logging con formattazione colorata.
- **modules/utils/symbol_manager.py**: Gestisce la selezione e il filtraggio dei simboli delle criptovalute.

### Sistema ML

- **ml_system/models/trainer.py**: Implementa l'addestramento dei modelli di machine learning.
- **ml_system/models/predictor.py**: Utilizza i modelli addestrati per generare predizioni.
- **ml_system/models/ensemble.py**: Implementa strategie ensemble per combinare multiple predizioni.

## Interazione tra i Componenti

```
┌─────────────────────────┐
│    data_collector.py    │
└─────────────┬───────────┘
              │
              ▼
┌─────────────────────────┐
│    command_args.py      │◄───┐
└─────────────┬───────────┘    │
              │                │
              ▼                │
┌─────────────────────────┐    │
│      config.py          │    │
└─────────────┬───────────┘    │
              │                │
              ▼                │
┌─────────────────────────┐    │
│    download_orchestrator.py  │
└─────────────┬───────────┘    │
              │                │
    ┌─────────┴─────────┐      │
    │                   │      │
    ▼                   ▼      │
┌─────────┐      ┌─────────┐   │
│exchange.py│      │data_fetcher.py│
└─────┬───┘      └────┬────┘   │
      │               │        │
      │               ▼        │
      │         ┌──────────┐   │
      │         │db_manager.py│
      │         └─────┬────┘   │
      │               │        │
      └───────┐       │        │
              ▼       ▼        │
      ┌─────────────────────┐  │
      │ indicator_processor.py │
      └─────────┬───────────┘  │
                │              │
                ▼              │
      ┌─────────────────────┐  │
      │data_integrity_checker.py│
      └─────────┬───────────┘  │
                │              │
                └──────────────┘
```

## Panoramica del Sistema

Il Data Collector è un componente fondamentale del sistema TradingJii, progettato per raccogliere, elaborare e archiviare dati di mercato delle criptovalute. Questo sistema automatizzato esegue le seguenti operazioni principali:

1. Download continuo di dati OHLCV (Open, High, Low, Close, Volume)
2. Calcolo di indicatori tecnici avanzati
3. Salvataggio persistente su database SQLite
4. Controlli di integrità dei dati
5. Aggiornamento automatico a intervalli configurabili

Il sistema supporta il monitoraggio di multiple criptovalute su diversi timeframe simultaneamente, con un'architettura scalabile che consente elaborazioni sia parallele che sequenziali.

## Flusso di Lavoro Dettagliato

### 1. Inizializzazione e Configurazione

Il processo inizia con l'inizializzazione del sistema:

```
┌─────────────────────────┐
│ Inizializzazione Sistema │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Configurazione Logger   │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│    Parsing Argomenti     │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ Inizializzazione DB      │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│    Pulizia Dati Vecchi   │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Init Tabelle Indicatori │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│     Inizio Ciclo di      │
│      Aggiornamento       │
└───────────┬─────────────┘
```

Durante questa fase:
- Viene configurato il sistema di logging
- Vengono analizzati gli argomenti da linea di comando per determinare i parametri operativi
- Vengono inizializzate le tabelle del database
- Vengono eliminati i dati obsoleti (precedenti al 1° gennaio 2024)
- Vengono inizializzate le tabelle per gli indicatori tecnici
- Se richiesto, vengono ricalcolati gli indicatori tecnici per i dati esistenti

### 2. Selezione delle Criptovalute

Il sistema supporta due modalità per la selezione delle criptovalute da monitorare:

**Modalità 1: Simboli Specifici Configurati**
- Utilizza una lista predefinita di simboli nel file di configurazione
- Ideale per monitorare criptovalute specifiche di interesse

**Modalità 2: Top Criptovalute per Volume**
- Interroga l'exchange per ottenere le criptovalute con il maggior volume di scambi
- Dinamico e si adatta alle condizioni di mercato
- Il numero di criptovalute è configurabile tramite argomento `--num-symbols`

```
┌─────────────────────────┐
│   Creazione Exchange    │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ Configurazione Specifica │
│        di Simboli?       │
└───────────┬─────────────┘
            │
      ┌─────┴─────┐
      │           │
      ▼           ▼
┌──────────┐ ┌──────────────┐
│   Sì     │ │      No      │
└────┬─────┘ └───────┬──────┘
     │               │
     ▼               ▼
┌──────────┐ ┌──────────────┐
│Usa Simboli│ │ Ottieni Top  │
│Configurati│ │  Simboli per │
│           │ │    Volume    │
└────┬─────┘ └───────┬──────┘
     │               │
     └───────┬───────┘
             ▼
┌─────────────────────────┐
│ Chiusura Connection     │
│      Exchange           │
└───────────┬─────────────┘
```

### 3. Elaborazione dei Timeframe

Il sistema supporta l'elaborazione di molteplici timeframe (ad es. 1m, 5m, 15m, 30m, 1h, 4h, 1d). L'elaborazione può avvenire in due modalità:

**Modalità Sequenziale**:
- I timeframe vengono elaborati uno alla volta
- Minore utilizzo delle risorse del sistema
- Esecuzione più lenta ma più stabile

**Modalità Parallela**:
- Tutti i timeframe vengono elaborati simultaneamente
- Utilizzo più intensivo delle risorse del sistema
- Esecuzione più rapida

```
┌─────────────────────────┐
│ Modalità Sequenziale?   │
└───────────┬─────────────┘
            │
      ┌─────┴─────┐
      │           │
      ▼           ▼
┌──────────┐ ┌──────────────┐
│   Sì     │ │      No      │
└────┬─────┘ └───────┬──────┘
     │               │
     ▼               ▼
┌──────────────┐ ┌──────────────┐
│Elabora ogni  │ │Crea Task per │
│timeframe     │ │ogni timeframe│
│sequenzialmente│ │              │
└──────┬───────┘ └───────┬──────┘
       │                 │
       │                 ▼
       │        ┌──────────────┐
       │        │Esegui tutti i│
       │        │task in       │
       │        │parallelo     │
       │        └───────┬──────┘
       │                │
       └────────┬───────┘
                ▼
┌─────────────────────────┐
│ Calcolo Indicatori      │
│     Tecnici             │
└───────────┬─────────────┘
```

### 4. Download dei Dati OHLCV

Per ogni timeframe, il sistema scarica i dati OHLCV per ciascun simbolo selezionato:

```
┌─────────────────────────┐
│ Elaborazione Timeframe  │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Suddivisione in Batch  │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Elaborazione Batch     │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ Elaborazione Simboli    │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Controllo Freshness    │
└───────────┬─────────────┘
            │
      ┌─────┴─────┐
      │           │
      ▼           ▼
┌──────────┐ ┌──────────────┐
│ Dati già │ │ Dati non     │
│ freschi  │ │ freschi      │
└────┬─────┘ └───────┬──────┘
     │               │
     ▼               ▼
┌──────────┐ ┌──────────────┐
│  Salta   │ │ Determina    │
│ Simbolo  │ │ Range Date   │
└──────────┘ └───────┬──────┘
                     │
                     ▼
             ┌──────────────┐
             │ Aggiungi     │
             │ Periodo      │
             │ Warmup       │
             └───────┬──────┘
                     │
                     ▼
             ┌──────────────┐
             │ Fetch OHLCV  │
             │ con Retry    │
             └───────┬──────┘
                     │
                     ▼
             ┌──────────────┐
             │ Filtra Dati  │
             │ per Periodo  │
             └───────┬──────┘
                     │
                     ▼
             ┌──────────────┐
             │ Salva Dati   │
             │ nel Database │
             └───────┬──────┘
                     │
                     ▼
             ┌──────────────┐
             │ Verifica     │
             │ Integrità    │
             └───────┬──────┘
```

**Dettagli importanti del processo**:

1. **Controllo Freshness**: 
   - Verifica se i dati sono già aggiornati nel database
   - Salta il download se i dati sono già freschi
   - Effettua il download solo dei dati mancanti se i dati sono parzialmente freschi

2. **Periodo di Warmup**:
   - Prima del 1° gennaio 2024, viene scaricato un numero configurabile di candele aggiuntive
   - Queste candele "warmup" sono necessarie per calcolare accuratamente indicatori tecnici che richiedono dati storici (come EMA200)
   - Le candele di warmup vengono usate per il calcolo degli indicatori ma non vengono considerate nei dati finali

3. **Retry Mechanism**:
   - Implementa un meccanismo di backoff esponenziale per gestire errori di rete
   - Ritenta il download con tempi di attesa crescenti tra i tentativi
   - Configurabile tramite parametri nel file di configurazione

4. **Filtraggio Dati**:
   - I dati vengono filtrati per mantenere solo quelli a partire dal 1° gennaio 2024
   - I dati di warmup vengono temporaneamente salvati ma marcati specificamente

5. **Verifica Integrità**:
   - Dopo il download, viene eseguito un controllo di integrità dei dati
   - Verifica la qualità dei dati, la presenza di gap, e altre anomalie
   - Fornisce un punteggio di qualità per ogni serie di dati

### 5. Elaborazione Parallela e Concorrenza

Il sistema implementa un sofisticato meccanismo di elaborazione parallela:

```
┌─────────────────────────┐
│  Download Parallelo     │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│   Creazione Semaforo    │
│     di Concorrenza      │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Creazione Task Queue   │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ Creazione Task per ogni │
│        Simbolo          │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ Creazione Task Display  │
│      Risultati          │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│   Esecuzione Gather     │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Visualizzazione        │
│  Riepilogo Batch        │
└───────────┬─────────────┘
```

**Dettagli del meccanismo di concorrenza**:

1. **Suddivisione in Batch**:
   - I simboli vengono divisi in batch di dimensione configurabile
   - Ogni batch viene elaborato separatamente

2. **Controllo della Concorrenza**:
   - Utilizza un semaforo asyncio per limitare il numero di download simultanei
   - Configurabile tramite il parametro `--concurrency`
   - Previene il sovraccarico dell'exchange e della rete

3. **Task Queue**:
   - Implementa una coda asincrona per gestire i risultati in tempo reale
   - Consente la visualizzazione progressiva dei risultati durante l'elaborazione

4. **Visualizzazione Progresso**:
   - Mostra il progresso in tempo reale durante il download
   - Fornisce statistiche immediate per ogni batch completato

### 6. Calcolo degli Indicatori Tecnici

Dopo il download dei dati OHLCV, il sistema calcola automaticamente una serie di indicatori tecnici:

```
┌─────────────────────────┐
│ Processo Indicatori     │
│       Tecnici           │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Caricamento Dati OHLCV │
│      dal Database       │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ Calcolo Indicatori      │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ Salvataggio Indicatori  │
│      nel Database       │
└───────────┬─────────────┘
```

**Indicatori calcolati**:

1. **Medie Mobili**:
   - SMA 9, 20, 50 (Simple Moving Average)
   - EMA 20, 50, 200 (Exponential Moving Average)

2. **Indicatori di Momentum**:
   - RSI 14 (Relative Strength Index)
   - Stocastico (Stochastic %K e %D)
   - MACD (Moving Average Convergence Divergence)
     - MACD Line
     - Signal Line
     - Histogram

3. **Indicatori di Volatilità**:
   - ATR 14 (Average True Range)
   - Volatilità (deviazione standard dei rendimenti)
   - Bande di Bollinger (Upper, Middle, Lower)

4. **Indicatori Basati sul Volume**:
   - OBV (On-Balance Volume)
   - VWAP (Volume-Weighted Average Price)
   - Volume SMA 20

5. **Indicatori di Trend**:
   - ADX (Average Directional Index)

Il sistema utilizza un'implementazione ottimizzata basata su numpy e pandas per il calcolo efficiente degli indicatori tecnici, senza dipendenze esterne.

### 7. Persistenza dei Dati e Gestione Database

Il sistema utilizza SQLite come database per la persistenza dei dati:

```
┌─────────────────────────┐
│  Struttura Database     │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ Tabelle market_data_TF  │
│ (una per timeframe)     │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ Schema Unificato        │
│ OHLCV + Indicatori      │
└───────────┬─────────────┘
```

**Caratteristiche della gestione dati**:

1. **Struttura delle Tabelle**:
   - Una tabella separata per ogni timeframe (es. market_data_1h, market_data_4h)
   - Schema unificato che include sia i dati OHLCV che gli indicatori tecnici
   - Ottimizzato per query rapide e analisi

2. **Gestione dei Dati di Warmup**:
   - I dati di warmup (pre-2024) vengono marcati specificamente
   - Utilizzati solo per il calcolo degli indicatori, non inclusi nelle analisi finali
   - Eliminati automaticamente al termine del ciclo di aggiornamento

3. **Pulizia Dati**:
   - Funzionalità di pulizia automatica dei dati obsoleti
   - Ottimizzazione dello spazio di archiviazione

4. **Controllo Integrità**:
   - Verifica dell'integrità dei dati salvati
   - Prevenzione di duplicati e inconsistenze

### 8. Visualizzazione Risultati e Monitoraggio

Il sistema fornisce un'interfaccia ricca per il monitoraggio e la visualizzazione dei risultati:

```
┌─────────────────────────┐
│  Display Risultati      │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ Statistiche Per         │
│     Timeframe           │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ Statistiche Complessive │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Visualizzazione Giorni │
│        Salvati          │
└───────────┬─────────────┘
```

**Caratteristiche della visualizzazione**:

1. **Resoconto per Timeframe**:
   - Numero di simboli completati, saltati e falliti
   - Numero totale di record salvati
   - Tempo di esecuzione

2. **Statistiche Complessive**:
   - Aggregazione dei risultati di tutti i timeframe
   - Visualizzazione del tempo totale di esecuzione

3. **Visualizzazione Giorni Salvati**:
   - Funzionalità per visualizzare il numero di giorni di dati salvati per ogni simbolo
   - Informazioni su primo e ultimo giorno disponibile
   - Conteggio candele totali

4. **Interfaccia Colorata**:
   - Utilizzo di colori per evidenziare informazioni importanti
   - Formattazione avanzata per migliorare la leggibilità

### 9. Ciclo di Aggiornamento Continuo

Il sistema opera in un ciclo continuo di aggiornamento:

```
┌─────────────────────────┐
│ Ciclo di Aggiornamento  │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ Esecuzione Aggiornamento│
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ Visualizzazione         │
│     Risultati           │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ Eliminazione Dati       │
│      Warmup             │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ Calcolo Prossimo        │
│    Aggiornamento        │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│     Attesa              │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│   Nuovo Ciclo           │
└───────────┬─────────────┘
```

**Caratteristiche del ciclo di aggiornamento**:

1. **Intervallo Configurabile**:
   - L'intervallo tra gli aggiornamenti è configurabile nel file di configurazione
   - Default: 5 minuti

2. **Gestione delle Interruzioni**:
   - Gestione corretta delle interruzioni manuali (Ctrl+C)
   - Pulizia delle risorse prima della terminazione

3. **Logging Avanzato**:
   - Registrazione dettagliata di ogni fase del processo
   - Informazioni su errori e anomalie

4. **Eliminazione Dati Warmup**:
   - I dati di warmup vengono eliminati dopo ogni ciclo completo
   - Ottimizzazione dello spazio di archiviazione

## Parametri da Linea di Comando

Il Data Collector supporta numerosi parametri da linea di comando per personalizzare il suo comportamento:

- `--timeframes`: Specifica i timeframe da monitorare (default: 1h,4h,1d)
- `--num-symbols`: Numero di criptovalute da monitorare (default: 10)
- `--days`: Numero di giorni di dati storici da scaricare (default: 30)
- `--concurrency`: Numero massimo di download paralleli (default: 5)
- `--batch-size`: Dimensione del batch per i download (default: 10)
- `--sequential`: Attiva la modalità sequenziale invece di quella parallela
- `--no-ta`: Disabilita il calcolo degli indicatori tecnici
- `--show-days`: Visualizza il numero di giorni di dati salvati nel database

## Funzionalità Avanzate

### Controllo di Integrità dei Dati

Il sistema implementa controlli avanzati di integrità per garantire la qualità dei dati:

- Verifica dell'integrità temporale (assenza di gap tra le candele)
- Controllo della completezza dei dati
- Rilevamento di anomalie nei volumi o nei prezzi
- Assegnazione di un punteggio di qualità per ogni serie di dati

### Meccanismo di Retry

Per gestire errori di rete e limitazioni degli exchange:

- Retry automatico con backoff esponenziale
- Parametri configurabili per numero massimo di tentativi
- Tempo di attesa crescente tra i tentativi
- Limite massimo per il tempo di attesa

### Elaborazione Parallela Ottimizzata

Il sistema implementa un'elaborazione parallela altamente ottimizzata:

- Utilizzo di asyncio per operazioni asincrone
- Semafori per limitare la concorrenza
- Code asincrone per la gestione dei risultati
- Visualizzazione in tempo reale del progresso

## Conclusioni

Il Data Collector di TradingJii è un sistema avanzato e robusto per la raccolta, l'elaborazione e l'archiviazione di dati di mercato delle criptovalute. La sua architettura modulare, le funzionalità di elaborazione parallela e il sistema completo di calcolo degli indicatori tecnici lo rendono uno strumento potente per l'analisi tecnica e lo sviluppo di strategie di trading.

Il sistema è progettato per essere affidabile, efficiente e facile da configurare, offrendo un'ampia gamma di opzioni per personalizzare il suo comportamento in base alle esigenze specifiche.
