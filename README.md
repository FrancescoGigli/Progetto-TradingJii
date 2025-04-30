# TradingJii - Trading Bot Platform

TradingJii è una piattaforma di trading automatizzato che utilizza modelli di machine learning e analisi tecnica per generare previsioni sui mercati delle criptovalute.

## Struttura del Progetto

### File Principali Backend

- **app.py**: Punto di ingresso principale dell'applicazione. Configura e avvia il server web.
- **main.py**: Contiene la logica principale per l'esecuzione del bot di trading.
- **server.py**: Implementa il server web che espone le API REST per interagire con il bot.
- **config.py**: Contiene le configurazioni globali dell'applicazione, come parametri di connessione e impostazioni predefinite.
- **state.py**: Gestisce lo stato globale dell'applicazione, incluso lo stato del bot e le sessioni attive.

### Gestione Dati e Trading

- **data_utils.py**: Utility per la manipolazione e la preparazione dei dati per i modelli di previsione.
- **fetcher.py**: Responsabile del recupero dei dati di mercato dalle API degli exchange.
- **trade_manager.py**: Gestisce l'esecuzione degli ordini di trading e il monitoraggio delle posizioni aperte.
- **predictor.py**: Implementa la logica per generare previsioni utilizzando i modelli addestrati.
- **model_manager.py**: Gestisce il caricamento, l'addestramento e la valutazione dei modelli di machine learning.

### Task Asincroni

- **celery_worker.py**: Configurazione del worker Celery per l'esecuzione di task asincroni.
- **tasks.py**: Definisce i task asincroni che vengono eseguiti in background, come l'addestramento dei modelli e l'aggiornamento dei dati.

### Utility

- **dependencies.py**: Gestisce le dipendenze dell'applicazione e l'iniezione delle dipendenze.
- **exceptions.py**: Definisce le eccezioni personalizzate utilizzate nell'applicazione.
- **logging_utils.py**: Utility per la configurazione e l'utilizzo dei log nell'applicazione.

### Frontend (Static)

#### HTML

- **static/index.html**: Pagina principale dell'interfaccia utente, che include il dashboard di trading.
- **static/models.html**: Pagina per la gestione e il monitoraggio dei modelli di previsione.
- **static/model_metrics.html**: Pagina che mostra le metriche di performance dei modelli.

#### CSS

- **static/css/styles.css**: Stili principali dell'applicazione.
- **static/css/main.css**: Stili aggiuntivi per il layout principale.
- **static/css/style.css**: Stili specifici per componenti UI.

#### JavaScript

- **static/js/app.js**: Script principale che inizializza l'applicazione frontend.
- **static/js/main.js**: Funzionalità generali dell'interfaccia utente.
- **static/js/training.js**: Gestisce l'interfaccia per l'addestramento dei modelli.
- **static/js/auto-status.js**: Aggiorna automaticamente lo stato del bot nell'interfaccia.

#### Moduli JavaScript

- **static/js/modules/api.js**: Gestisce le chiamate API al backend.
- **static/js/modules/charts.js**: Crea e aggiorna i grafici nell'interfaccia utente.
- **static/js/modules/dashboard.js**: Gestisce la dashboard principale con le informazioni di trading.
- **static/js/modules/models.js**: Gestisce l'interfaccia per la visualizzazione e la configurazione dei modelli.
- **static/js/modules/ui.js**: Utility generali per l'interfaccia utente.

##### Moduli di Previsione

- **static/js/modules/predictions.js**: Punto di ingresso centrale per il sistema di previsioni.
- **static/js/modules/predictionAPI.js**: Gestisce le chiamate API specifiche per le previsioni.
- **static/js/modules/predictionCore.js**: Implementa la logica di base per le previsioni.
- **static/js/modules/predictionData.js**: Gestisce il caricamento e l'elaborazione dei dati per le previsioni.
- **static/js/modules/predictionDisplay.js**: Visualizza i risultati delle previsioni nell'interfaccia.
- **static/js/modules/predictionModels.js**: Gestisce i modelli di previsione e i timeframes.
- **static/js/modules/predictionParams.js**: Gestisce i parametri configurabili per le previsioni.
- **static/js/modules/predictionTrading.js**: Collega le previsioni alle operazioni di trading.
- **static/js/modules/predictionUI.js**: Gestisce l'interfaccia utente specifica per le previsioni.

### Templates

- **templates/training.html**: Template per la pagina di addestramento dei modelli.

### Modelli Addestrati e Log

- **trained_models/**: Contiene i modelli addestrati salvati in formato JSON o altri formati.
- **logs/**: Contiene i log di addestramento e le metriche dei modelli.
  - **logs/lstm_**/: Log specifici per i modelli LSTM con diversi timeframes.
  - **logs/plots/**: Grafici e visualizzazioni delle performance dei modelli.

## Funzionalità Principali

1. **Dashboard di Trading**: Visualizza lo stato attuale del portafoglio, le posizioni aperte e le statistiche di trading.
2. **Previsioni in Tempo Reale**: Genera previsioni sui movimenti di prezzo utilizzando diversi modelli e timeframes.
3. **Gestione Modelli**: Interfaccia per addestrare, testare e configurare i modelli di previsione.
4. **Trading Automatizzato**: Esegue automaticamente operazioni di trading basate sulle previsioni generate.
5. **Monitoraggio Performance**: Traccia e visualizza le performance dei modelli e delle strategie di trading.

## Utilizzo

1. Configura le API keys degli exchange nella sezione apposita.
2. Seleziona i modelli e i timeframes desiderati nella sezione Predizioni.
3. Avvia il bot utilizzando il pulsante "Avvia" nella dashboard.
4. Monitora le previsioni e le operazioni di trading nella dashboard principale.

## Note Tecniche

- L'applicazione utilizza un'architettura modulare con separazione tra backend (Python) e frontend (HTML/CSS/JavaScript).
- I modelli di machine learning supportati includono LSTM, Random Forest e XGBoost.
- Il sistema supporta diversi timeframes per le previsioni: 5m, 15m, 30m, 1h e 4h.
- Le operazioni di trading vengono eseguite tramite API degli exchange di criptovalute.
