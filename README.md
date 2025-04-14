# TradingJii Bot

Un bot di trading per criptovalute con funzionalità di machine learning e un'interfaccia frontend basata su web.

## Interfaccia Frontend

Il TradingJii Bot include ora un'interfaccia web moderna che consente di:

- Monitorare le statistiche di trading e le performance
- Visualizzare le posizioni aperte e i segnali di trading recenti
- Configurare i parametri del bot attraverso un'interfaccia user-friendly
- Avviare e fermare il bot di trading
- Visualizzare i log in tempo reale

## Iniziare

### Prerequisiti

Assicurati di avere tutte le dipendenze richieste installate:

```
pip install -r requirements.txt
```

### Avvio del Frontend

Per avviare l'interfaccia web, esegui:

```
python start_frontend.py
```

Questo avvierà il server API e aprirà automaticamente il browser predefinito all'indirizzo http://localhost:5000.

## Utilizzare il Frontend

### Dashboard

La dashboard fornisce una panoramica della tua attività di trading:

- **Bilancio**: Mostra il tuo attuale bilancio in USDT
- **Posizioni Aperte**: Visualizza il numero di posizioni di trading attualmente aperte
- **Stato Bot**: Indica se il bot è in esecuzione o fermo
- **Salute Sistema**: Mostra lo stato di connessione con l'exchange
- **Tabella Ordini Aperti**: Elenca tutti gli ordini di trading attualmente aperti
- **Tabella Trades Recenti**: Mostra i trade recenti completati dal bot

### Modelli

La sezione Modelli ti permette di:

- Visualizzare lo stato dei modelli addestrati per ciascun timeframe
- Addestrare nuovi modelli selezionando il tipo di modello e il timeframe

### API Keys

Nella sezione API Keys puoi impostare le tue chiavi API dell'exchange per consentire al bot di operare sul tuo account.

### Avvio/Arresto del Bot

Usa il pulsante "Avvia Bot" nell'angolo in alto a destra per avviare il bot. Una volta in esecuzione, il pulsante diventerà "Ferma Bot" che potrai cliccare per interrompere l'operazione del bot.

## Architettura

Il frontend consiste in:

- **HTML/CSS/JavaScript**: L'interfaccia utente
- **Flask Server**: Serve l'interfaccia frontend
- **FastAPI Server**: Fornisce le API per il bot di trading
- **Core del Bot di Trading**: Il sistema di trading Python esistente

## File

- `static/index.html`: Il file HTML principale del frontend
- `static/css/styles.css`: Gli stili CSS per il frontend
- `static/js/app.js`: JavaScript per la funzionalità del frontend
- `frontend_server.py`: Server Flask che serve l'interfaccia frontend
- `start_frontend.py`: Script di avvio per lanciare sia il backend che il frontend

## Nota

Questo frontend è progettato per funzionare con il bot di trading TradingJii esistente. Assicurati che tutte le dipendenze richieste per il bot di trading siano installate prima di utilizzare il frontend.