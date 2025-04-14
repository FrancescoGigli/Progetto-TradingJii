// Costanti
const API_BASE_URL = 'http://localhost:8000';
const LOCAL_STORAGE_API_KEY = 'trae_api_key';
const LOCAL_STORAGE_API_SECRET = 'trae_api_secret';
const DEFAULT_API_KEY = 'hRI4q8EB3ryaURdyBm';
const DEFAULT_API_SECRET = 'xQpYxVtEinsD6yqa84PGbYVsgYrT9O3k0MRf';

// Elementi DOM
const botStatusBtn = document.getElementById('botStatusBtn');
const logContent = document.getElementById('log-content');
const navLinks = document.querySelectorAll('.nav-link');
const sections = document.querySelectorAll('.section');

// Stato dell'applicazione
let botRunning = false;
let apiKey = localStorage.getItem(LOCAL_STORAGE_API_KEY) || DEFAULT_API_KEY;
let apiSecret = localStorage.getItem(LOCAL_STORAGE_API_SECRET) || DEFAULT_API_SECRET;

// Dichiarazione variabili globali per i grafici
let positionChart = null;
let metricsChart = null;
let comparisonChart = null;
let modelMetricsData = {};
let currentMetric = 'accuracy';

// Variabili globali per il controllo delle predizioni
let predictionsInterval = null;
let isPredictionsRunning = false;

// Inizializzazione dell'applicazione
document.addEventListener('DOMContentLoaded', () => {
    // Imposta gli stati iniziali
    document.getElementById('health-status').textContent = 'In attesa...';
    document.getElementById('health-status').className = 'card-text text-warning';
    
    // Imposta gli event listener
    setupEventListeners();
    
    // Aggiungo un listener diretto per il form di training (oltre a quello in setupEventListeners)
    const trainingForm = document.getElementById('model-training-form');
    if (trainingForm) {
        console.log('Form di training trovato, aggiungo event listener');
        trainingForm.addEventListener('submit', handleTrainingFormSubmit);
    } else {
        console.error('Form di training non trovato!');
    }
    
    // Carica i dati iniziali
    checkStatus();
    checkHealth();
    loadBalance();
    loadPositions();
    loadOpenOrders();
    loadTrades();
    
    // Imposta gli header per le richieste API
    updateApiHeaders();
    
    // Se abbiamo le API keys, le precompiliamo nel form
    if (apiKey && apiSecret) {
        document.getElementById('api-key').value = apiKey;
        document.getElementById('secret-key').value = apiSecret;
    }
    
    // Mostra la dashboard all'avvio
    showSection('dashboard');
    
    // Carica i simboli per il grafico
    loadChartSymbols();
    
    // Controlla lo stato dei modelli
    checkAllModelsStatus();
    
    // Carica le metriche dei modelli esistenti
    loadModelMetrics();
    
    // Inizializza il controllo delle predizioni
    initializePredictionsControl();
    
    // Aggiorna periodicamente i dati
    setInterval(() => {
        if (document.getElementById('dashboard-section').classList.contains('d-none') === false) {
            checkStatus();
            checkHealth();
            loadBalance();
            loadPositions();
            loadOpenOrders();
        }
    }, 10000); // Aggiorna ogni 10 secondi
});

// Funzione per impostare gli event listener
function setupEventListeners() {
    // Event listener per il pulsante di stato del bot
    if (botStatusBtn) {
        botStatusBtn.addEventListener('click', toggleBotStatus);
    }
    
    // Event listener per i link di navigazione
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const sectionId = link.getAttribute('href').substring(1);
            showSection(sectionId);
        });
    });
    
    // Event listener per il pulsante di salvataggio delle API keys
    const saveApiKeysBtn = document.getElementById('save-api-keys-btn');
    if (saveApiKeysBtn) {
        saveApiKeysBtn.addEventListener('click', saveApiKeys);
    }
    
    // Event listener per il pulsante di test della connessione
    const testConnectionBtn = document.getElementById('test-connection-btn');
    if (testConnectionBtn) {
        testConnectionBtn.addEventListener('click', testConnection);
    }
    
    // Event listener per il pulsante di aggiornamento predizioni
    const refreshPredictionsBtn = document.getElementById('refresh-predictions-btn');
    if (refreshPredictionsBtn) {
        refreshPredictionsBtn.addEventListener('click', loadPredictions);
    }
    
    // Event listener per il selettore del timeframe del grafico
    const chartTimeframeSelect = document.getElementById('chart-timeframe-select');
    if (chartTimeframeSelect) {
        chartTimeframeSelect.addEventListener('change', () => {
            const symbol = document.getElementById('chart-symbol-select').value;
            if (symbol) {
                loadChartData(symbol, chartTimeframeSelect.value);
            }
        });
    }
    
    // Event listener per il selettore del simbolo del grafico
    const chartSymbolSelect = document.getElementById('chart-symbol-select');
    if (chartSymbolSelect) {
        chartSymbolSelect.addEventListener('change', () => {
            const symbol = chartSymbolSelect.value;
            if (symbol) {
                const timeframe = document.getElementById('chart-timeframe-select').value;
                loadChartData(symbol, timeframe);
            }
        });
    }
    
    // Event listener per i selettori delle metriche
    const metricsTimeframeSelect = document.getElementById('metrics-timeframe-select');
    if (metricsTimeframeSelect) {
        metricsTimeframeSelect.addEventListener('change', function() {
            loadModelMetrics();
        });
    }
    
    // Event listener per i pulsanti delle metriche
    const metricButtons = document.querySelectorAll('[data-metric]');
    if (metricButtons.length > 0) {
        metricButtons.forEach(button => {
            button.addEventListener('click', function() {
                const metric = this.getAttribute('data-metric');
                currentMetric = metric;
                
                // Aggiorna la classe active
                metricButtons.forEach(btn => btn.classList.remove('active'));
                this.classList.add('active');
                
                // Aggiorna il grafico con la nuova metrica
                displayMetricsChart();
            });
        });
    }
    
    // Event listener per il pulsante di training di tutti i modelli
    const trainAllModelsBtn = document.getElementById('train-all-models-btn');
    if (trainAllModelsBtn) {
        trainAllModelsBtn.addEventListener('click', function() {
            startTrainingAllModels();
        });
    }
    
    // Event listener per il pulsante di training dei modelli mancanti
    const trainMissingModelsBtn = document.getElementById('train-missing-models-btn');
    if (trainMissingModelsBtn) {
        trainMissingModelsBtn.addEventListener('click', function() {
            startTrainingMissingModels();
        });
    }
}

// Funzioni di gestione UI
function showSection(sectionId) {
    // Nascondi tutte le sezioni
    sections.forEach(section => {
        section.classList.add('d-none');
    });
    
    // Rimuovi la classe active da tutti i link
    navLinks.forEach(link => {
        link.classList.remove('active');
    });
    
    // Mostra la sezione selezionata
    const selectedSection = document.getElementById(`${sectionId}-section`);
    if (selectedSection) {
        selectedSection.classList.remove('d-none');
    }
    
    // Aggiungi la classe active al link selezionato
    const selectedLink = document.querySelector(`.nav-link[href="#${sectionId}"]`);
    if (selectedLink) {
        selectedLink.classList.add('active');
    }
    
    // Carica i dati specifici in base alla sezione
    if (sectionId === 'dashboard') {
        loadBalance();
        loadPositions();
        loadOpenOrders();
        loadTrades();
    } else if (sectionId === 'models') {
        // Aggiorna lo stato dei modelli quando si visualizza la sezione modelli
        checkAllModelsStatus();
    }
}

function appendToLog(message) {
    // Verifica se l'elemento logContent esiste
    if (!logContent) {
        console.log(`[${new Date().toLocaleTimeString()}] ${message}`);
        return;
    }
    
    const timestamp = new Date().toLocaleTimeString();
    const logLine = `[${timestamp}] ${message}\n`;
    
    // Limita il numero di righe nel log (mantieni solo le ultime 20 righe)
    const currentLog = logContent.innerHTML;
    const lines = currentLog.split('\n');
    if (lines.length > 20) {
        lines.splice(0, lines.length - 20);
        logContent.innerHTML = lines.join('\n');
    }
    
    // Rimuovi i messaggi di debug API_KEY ricevuta/attesa
    if (message.includes('API_KEY ricevuta') || 
        message.includes('API_SECRET ricevuta') ||
        message.includes('API_KEY attesa') ||
        message.includes('API_SECRET attesa')) {
        return;
    }
    
    // Aggiungi il nuovo messaggio
    logContent.innerHTML += logLine;
    logContent.scrollTop = logContent.scrollHeight;
}

// Funzioni per le chiamate API
async function makeApiRequest(endpoint, method = 'GET', data = null) {
    try {
        const headers = {
            'Content-Type': 'application/json',
            'api-key': apiKey,
            'api-secret': apiSecret
        };
        
        const options = {
            method,
            headers
        };
        
        if (data && (method === 'POST' || method === 'PUT')) {
            options.body = JSON.stringify(data);
        }

        // Log solo per operazioni importanti o POST/PUT
        if (method !== 'GET' || 
            endpoint === '/status' || 
            endpoint === '/balance' || 
            endpoint === '/health') {
            appendToLog(`Richiesta ${method} a ${endpoint}...`);
        }
        
        const response = await fetch(`${API_BASE_URL}${endpoint}`, options);
        
        if (!response.ok) {
            let errorMessage = `Errore ${response.status}: `;
            try {
                const errorData = await response.json();
                errorMessage += errorData.detail || 'Errore sconosciuto';
            } catch (e) {
                errorMessage += 'Impossibile leggere dettaglio errore';
            }
            
            throw new Error(errorMessage);
        }
        
        const result = await response.json();
        
        // Log solo per operazioni importanti o POST/PUT
        if (method !== 'GET' || endpoint === '/status') {
            appendToLog(`Risposta ricevuta da ${endpoint}`);
        }
        
        return result;
    } catch (error) {
        appendToLog(`Errore: ${error.message}`);
        console.error('API Error:', error);
        return null;
    }
}

function updateApiHeaders() {
    apiKey = localStorage.getItem(LOCAL_STORAGE_API_KEY) || DEFAULT_API_KEY;
    apiSecret = localStorage.getItem(LOCAL_STORAGE_API_SECRET) || DEFAULT_API_SECRET;
}

// Funzioni per interagire con le API
async function checkStatus() {
    const result = await makeApiRequest('/status');
    if (result) {
        botRunning = result.running;
        updateBotStatusDisplay();
    }
}

async function checkHealth() {
    document.getElementById('health-status').textContent = 'In attesa...';
    document.getElementById('health-status').className = 'card-text text-warning';
    
    try {
        // Tentativo diretto senza passare attraverso makeApiRequest per evitare gestione degli errori
        const response = await fetch(`${API_BASE_URL}/health`);
        
        if (response.ok) {
            const result = await response.json();
            document.getElementById('health-status').textContent = 'Online';
            document.getElementById('health-status').className = 'card-text status-online';
            return true;
        }
    } catch (error) {
        console.error("Errore controllo salute:", error);
    }
    
    // Se arriviamo qui, c'è stato un errore o la risposta non è ok
    document.getElementById('health-status').textContent = 'Offline';
    document.getElementById('health-status').className = 'card-text status-offline';
    return false;
}

async function toggleBotStatus() {
    if (botRunning) {
        const result = await makeApiRequest('/stop', 'POST');
        if (result) {
            appendToLog('Richiesta di arresto del bot inviata.');
            botRunning = false;
        }
    } else {
        const result = await makeApiRequest('/start', 'POST');
        if (result) {
            appendToLog('Bot avviato con successo.');
            botRunning = true;
        }
    }
    updateBotStatusDisplay();
}

function updateBotStatusDisplay() {
    const statusElement = document.getElementById('bot-status');
    if (botRunning) {
        statusElement.textContent = 'In esecuzione';
        statusElement.className = 'card-text status-online';
        botStatusBtn.textContent = 'Ferma Bot';
        botStatusBtn.className = 'btn btn-sm btn-danger';
    } else {
        statusElement.textContent = 'Fermo';
        statusElement.className = 'card-text status-offline';
        botStatusBtn.textContent = 'Avvia Bot';
        botStatusBtn.className = 'btn btn-sm btn-success';
    }
}

async function loadBalance() {
    const result = await makeApiRequest('/balance');
    if (result) {
        // I campi rinominati
        const totalWallet = document.getElementById('total-wallet');
        const availableBalance = document.getElementById('available-balance');
        const usedBalance = document.getElementById('used-balance');
        const unrealizedPnl = document.getElementById('unrealized-pnl');
        const equity = document.getElementById('equity');
        
        // Aggiorniamo i valori
        if (totalWallet) totalWallet.textContent = `${result.total_wallet.toFixed(2)} USDT`;
        if (availableBalance) availableBalance.textContent = `${result.available.toFixed(2)} USDT`;
        if (usedBalance) usedBalance.textContent = `${result.used.toFixed(2)} USDT`;
        
        // Formatta il PnL con colore a seconda se è positivo o negativo
        if (unrealizedPnl) {
            const pnl = result.pnl;
            const pnlClass = pnl > 0 ? 'text-success' : (pnl < 0 ? 'text-danger' : '');
            unrealizedPnl.textContent = `${pnl.toFixed(2)} USDT`;
            unrealizedPnl.className = `card-text ${pnlClass}`;
        }
        
        // Calcola e formatta il Totale non realizzato (ex Equity)
        if (equity) {
            const total = result.total_wallet || 0;
            const pnl = result.pnl || 0;
            const equityValue = total + pnl;
            const equityClass = pnl > 0 ? 'text-success' : (pnl < 0 ? 'text-danger' : '');
            equity.textContent = `${equityValue.toFixed(2)} USDT`;
            equity.className = `card-text ${equityClass}`;
        }
    }
}

async function loadPositions() {
    const result = await makeApiRequest('/positions');
    if (result) {
        document.getElementById('positions').textContent = result.open_positions;
    }
}

async function loadOpenOrders() {
    const result = await makeApiRequest('/orders/open');
    if (result) {
        const tableBody = document.getElementById('open-orders-table');
        tableBody.innerHTML = '';
        
        // Aggiorniamo le intestazioni della tabella per includere le nuove colonne
        const tableHeader = document.querySelector('#open-orders-table-header tr');
        if (tableHeader) {
            // Verifica se abbiamo già le nuove colonne
            if (!document.getElementById('sl-header')) {
                const newHeader = `
                    <th id="symbol-header">Simbolo</th>
                    <th id="side-header">Direzione</th>
                    <th id="amount-header">Quantità</th>
                    <th id="price-header">Prezzo</th>
                    <th id="pnl-header">P/L</th>
                    <th id="usdt-header">USDT Usati</th>
                    <th id="sl-header">Stop Loss</th>
                    <th id="sl-profit-header">P/L SL</th>
                    <th id="status-header">Stato</th>
                    <th id="action-header">Azione</th>
                `;
                tableHeader.innerHTML = newHeader;
            }
        }
        
        if (result.length === 0) {
            const row = document.createElement('tr');
            row.innerHTML = '<td colspan="10" class="text-center">Nessun ordine aperto</td>';
            tableBody.appendChild(row);
        } else {
            result.forEach(item => {
                const row = document.createElement('tr');
                
                // Verifica se è una posizione o un ordine
                const isPosition = item.type === 'position';
                
                // Formatta il PnL se presente (solo per posizioni)
                let pnlDisplay = '-';
                if (isPosition && item.pnl !== undefined) {
                    const pnl = parseFloat(item.pnl);
                    const pnlClass = pnl > 0 ? 'text-success' : (pnl < 0 ? 'text-danger' : '');
                    pnlDisplay = `<span class="${pnlClass}">${pnl.toFixed(2)} USDT</span>`;
                }
                
                // Formatta il lato (Buy/Sell)
                const sideClass = item.side === 'Buy' ? 'text-success' : 'text-danger';
                
                // Formatta lo stato
                let statusDisplay = item.status || '';
                if (isPosition) {
                    statusDisplay = 'Aperta';
                }
                
                // Mostra gli USDT utilizzati per la posizione (margine)
                let usedUSDT = '-';
                if (isPosition && item.margin) {
                    usedUSDT = `${parseFloat(item.margin).toFixed(2)} USDT`;
                }
                
                // Formatta il valore di stop loss e profitto potenziale
                let stopLossDisplay = 'N/A';
                let slProfitDisplay = 'N/A';
                
                if (isPosition) {
                    // Visualizza stop loss
                    if (item.stop_loss && item.stop_loss !== 'N/A') {
                        stopLossDisplay = parseFloat(item.stop_loss).toFixed(4);
                    }
                    
                    // Visualizza profitto potenziale
                    if (item.sl_profit && item.sl_profit !== 'N/A') {
                        const slProfit = parseFloat(item.sl_profit);
                        const slProfitClass = slProfit > 0 ? 'text-success' : 'text-danger';
                        slProfitDisplay = `<span class="${slProfitClass}">${slProfit.toFixed(2)} USDT</span>`;
                    }
                }
                
                // Crea pulsante di azione per chiudere posizioni/annullare ordini
                let actionButton = '';
                if (isPosition) {
                    actionButton = `<button class="btn btn-sm btn-danger" onclick="closePosition('${item.symbol}', '${item.side}')">Chiudi</button>`;
                } else {
                    actionButton = `<button class="btn btn-sm btn-secondary" onclick="cancelOrder('${item.id}')">Annulla</button>`;
                }
                
                row.innerHTML = `
                    <td>${item.symbol}</td>
                    <td class="${sideClass}">${item.side}</td>
                    <td>${item.amount}</td>
                    <td>${typeof item.price === 'number' ? item.price.toFixed(4) : item.price}</td>
                    <td>${pnlDisplay}</td>
                    <td>${usedUSDT}</td>
                    <td>${stopLossDisplay}</td>
                    <td>${slProfitDisplay}</td>
                    <td>${statusDisplay}</td>
                    <td>${actionButton}</td>
                `;
                tableBody.appendChild(row);
            });
        }
        
        // Ricarichiamo i simboli per il grafico quando vengono caricate le posizioni
        loadChartSymbols();
    }
}

// Funzioni per gestire le azioni sugli ordini
async function closePosition(symbol, side) {
    if (!confirm(`Confermi la chiusura della posizione ${symbol}?`)) return;
    
    const closeSide = side === 'Buy' ? 'Sell' : 'Buy';
    appendToLog(`Chiusura posizione ${symbol} (${closeSide})...`);
    
    try {
        // Chiamata all'API per chiudere la posizione
        const response = await makeApiRequest('/close-position', 'POST', {
            symbol: symbol,
            side: side
        });
        
        if (response && response.status === 'success') {
            appendToLog(`✅ Posizione ${symbol} chiusa con successo`);
            // Aggiorna le posizioni e gli ordini
            loadPositions();
            loadOpenOrders();
            loadTrades();
        } else {
            appendToLog(`❌ Errore nella chiusura della posizione ${symbol}: ${response.message || 'Errore sconosciuto'}`);
        }
    } catch (error) {
        appendToLog(`❌ Errore nella chiusura della posizione ${symbol}: ${error.message || error}`);
    }
}

async function cancelOrder(orderId) {
    if (!confirm(`Confermi l'annullamento dell'ordine?`)) return;
    
    appendToLog(`Annullamento ordine ${orderId}...`);
    // Qui aggiungi il codice per chiamare l'API per annullare l'ordine
    // Implementazione futura
}

async function loadTrades() {
    const result = await makeApiRequest('/trades');
    if (result) {
        const tableBody = document.getElementById('trades-table');
        tableBody.innerHTML = '';
        
        if (result.length === 0) {
            const row = document.createElement('tr');
            row.innerHTML = '<td colspan="7" class="text-center">Nessun trade recente</td>';
            tableBody.appendChild(row);
        } else {
            result.forEach(trade => {
                const timestamp = new Date(trade.timestamp).toLocaleString();
                const pnl = parseFloat(trade.realized_pnl || 0);
                const pnlClass = pnl > 0 ? 'text-success' : (pnl < 0 ? 'text-danger' : '');
                
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${trade.symbol}</td>
                    <td>${trade.type}</td>
                    <td>${trade.side}</td>
                    <td>${trade.amount}</td>
                    <td>${trade.price}</td>
                    <td class="${pnlClass}">${pnl.toFixed(2)} USDT</td>
                    <td>${timestamp}</td>
                `;
                tableBody.appendChild(row);
            });
        }
    }
}

// Funzione per salvare le API keys
async function saveApiKeys(e) {
    e.preventDefault();
    
    const apiKeyValue = document.getElementById('api-key').value.trim();
    const secretKeyValue = document.getElementById('secret-key').value.trim();
    
    if (!apiKeyValue || !secretKeyValue) {
        appendToLog('Errore: Entrambe le chiavi sono richieste.');
        return;
    }
    
    const result = await makeApiRequest('/set-keys', 'POST', { 
        api_key: apiKeyValue,
        secret_key: secretKeyValue
    });
    
    if (result) {
        localStorage.setItem(LOCAL_STORAGE_API_KEY, apiKeyValue);
        localStorage.setItem(LOCAL_STORAGE_API_SECRET, secretKeyValue);
        updateApiHeaders();
        appendToLog('Chiavi API salvate con successo!');
    }
}

// Funzione per testare la connessione
async function testConnection() {
    appendToLog('Test di connessione in corso...');
    
    // Aggiorniamo prima gli header con i valori correnti nei campi
    const tmpApiKey = document.getElementById('api-key').value.trim();
    const tmpApiSecret = document.getElementById('secret-key').value.trim();
    
    if (tmpApiKey && tmpApiSecret) {
        apiKey = tmpApiKey;
        apiSecret = tmpApiSecret;
    }
    
    try {
        // Test dell'endpoint di health senza autenticazione
        const healthResponse = await fetch(`${API_BASE_URL}/health`);
        
        if (!healthResponse.ok) {
            appendToLog(`Connessione al server API: FALLITA - Errore ${healthResponse.status}`);
            return false;
        }
        
        const healthResult = await healthResponse.json();
        
        if (healthResult && healthResult.status === 'ok') {
            appendToLog('Connessione al server API: OK');
            
            // Test endpoint autenticato
            const statusResponse = await fetch(`${API_BASE_URL}/status`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    'api-key': apiKey,
                    'api-secret': apiSecret
                }
            });
            
            // Se riceviamo 401 Unauthorized, le chiavi API sono errate
            if (statusResponse.status === 401) {
                appendToLog('Autenticazione API: FALLITA - Chiavi API non valide');
                return false;
            }
            
            if (!statusResponse.ok) {
                appendToLog(`Autenticazione API: FALLITA - Errore ${statusResponse.status}`);
                return false;
            }
            
            const statusResult = await statusResponse.json();
            appendToLog('Autenticazione API: OK');
            return true;
        } else {
            appendToLog('Connessione al server API: FALLITA - Risposta non valida');
            return false;
        }
    } catch (error) {
        appendToLog(`Connessione al server API: FALLITA - ${error.message}`);
        console.error('Error in test connection:', error);
        return false;
    }
}

// Funzione per visualizzare le metriche di training
function displayMetrics(metrics, modelType, timeframe) {
    // Mostra l'area delle metriche
    const metricsContent = document.getElementById('metrics-content');
    if (!metricsContent) {
        console.error('Elemento metrics-content non trovato');
        return;
    }
    
    // Svuota il contenitore delle metriche
    metricsContent.innerHTML = '';
    
    // Aggiunge il titolo
    const titleElement = document.createElement('h5');
    titleElement.innerHTML = `Risultati Training ${modelType.toUpperCase()} - ${timeframe}`;
    metricsContent.appendChild(titleElement);
    
    // Crea una tabella per le metriche
    const table = document.createElement('table');
    table.className = 'table table-sm';
    
    // Intestazione tabella
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    headerRow.innerHTML = `<th>Metrica</th><th>Valore</th>`;
    thead.appendChild(headerRow);
    table.appendChild(thead);
    
    // Corpo tabella
    const tbody = document.createElement('tbody');
    
    // Aggiungi ogni metrica come riga della tabella
    Object.entries(metrics).forEach(([key, value]) => {
        const row = document.createElement('tr');
        
        // Formatta la chiave per la visualizzazione
        const formattedKey = key
            .replace(/_/g, ' ')
            .replace(/\b\w/g, l => l.toUpperCase());
            
        // Formatta il valore se è un numero
        let formattedValue = value;
        if (!isNaN(value) && typeof value !== 'boolean') {
            formattedValue = Number(value).toFixed(4);
        }
        
        row.innerHTML = `
            <td>${formattedKey}</td>
            <td>${formattedValue}</td>
        `;
        tbody.appendChild(row);
    });
    
    table.appendChild(tbody);
    metricsContent.appendChild(table);
    
    // Mostra l'area delle metriche
    document.getElementById('training-metrics').classList.remove('d-none');
}

function createCharts(metrics, model, timeframe) {
    // Cancella i grafici esistenti
    clearCharts();
    
    // Assicurati che timeframe sia definito
    if (!timeframe) {
        timeframe = document.getElementById('metric-timeframe-select').value || '15m';
        console.warn(`Timeframe non specificato, utilizzo valore dal selettore: ${timeframe}`);
    }
    
    // Crea il grafico dell'accuratezza
    createAccuracyChart(metrics, model, timeframe);
    
    // Crea la matrice di confusione
    createConfusionMatrixChart(metrics, model, timeframe);
    
    // Crea la curva ROC se disponibile
    createROCCurveChart(metrics, model, timeframe);
    
    // Crea il grafico della storia dell'addestramento
    createTrainingHistoryChart(metrics, model, timeframe);
}

function clearCharts() {
    // Cancella i grafici esistenti
    if (accuracyChart) {
        accuracyChart.destroy();
        accuracyChart = null;
    }
    
    if (confusionMatrixChart) {
        confusionMatrixChart.destroy();
        confusionMatrixChart = null;
    }
    
    if (rocCurveChart) {
        rocCurveChart.destroy();
        rocCurveChart = null;
    }
    
    if (trainingHistoryChart) {
        trainingHistoryChart.destroy();
        trainingHistoryChart = null;
    }
    
    if (comparisonChart) {
        comparisonChart.destroy();
        comparisonChart = null;
    }
}

function createAccuracyChart(metrics, model, timeframe) {
    const ctx = document.getElementById('accuracy-chart').getContext('2d');
    
    // Assicurati che timeframe sia definito
    if (!timeframe) {
        timeframe = document.getElementById('metric-timeframe-select').value || '15m';
        console.warn(`createAccuracyChart: timeframe non specificato, utilizzo valore dal selettore: ${timeframe}`);
    }
    
    // Dati predefiniti se non ci sono dati di accuratezza per epoche
    let labels = [];
    let accuracyData = [];
    let valAccuracyData = [];
    
    // Per LSTM, potrebbero esserci dati di accuratezza per epoche
    if (model === 'lstm' && metrics.accuracy_history && metrics.val_accuracy_history) {
        labels = Array.from({length: metrics.accuracy_history.length}, (_, i) => i + 1);
        accuracyData = metrics.accuracy_history;
        valAccuracyData = metrics.val_accuracy_history;
    } else {
        // Per RF e XGB, mostra solo il valore finale
        labels = ['Finale'];
        accuracyData = [metrics.accuracy || 0];
        valAccuracyData = [metrics.test_accuracy || metrics.accuracy || 0];
    }
    
    accuracyChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Accuratezza Training',
                    data: accuracyData,
                    borderColor: 'rgb(75, 192, 192)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    tension: 0.1
                },
                {
                    label: 'Accuratezza Validation',
                    data: valAccuracyData,
                    borderColor: 'rgb(255, 99, 132)',
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: `Accuratezza del Modello ${model.toUpperCase()} (${timeframe})`
                }
            },
            scales: {
                y: {
                    min: 0,
                    max: 1,
                    ticks: {
                        callback: function(value) {
                            return (value * 100).toFixed(0) + '%';
                        }
                    }
                }
            }
        }
    });
}

function createConfusionMatrixChart(metrics, model, timeframe) {
    // Assicurati che timeframe sia definito
    if (!timeframe) {
        timeframe = document.getElementById('metric-timeframe-select').value || '15m';
        console.warn(`createConfusionMatrixChart: timeframe non specificato, utilizzo valore dal selettore: ${timeframe}`);
    }
    
    // Verifica se ci sono dati per la matrice di confusione
    if (!metrics.confusion_matrix) {
        document.getElementById('confusion-matrix-chart').parentNode.parentNode.style.display = 'none';
        return;
    }
    
    document.getElementById('confusion-matrix-chart').parentNode.parentNode.style.display = 'block';
    
    try {
        const ctx = document.getElementById('confusion-matrix-chart').getContext('2d');
        const cm = metrics.confusion_matrix;
        
        // Controlla se il tipo di grafico 'matrix' è disponibile
        if (typeof Chart.controllers.matrix === 'undefined') {
            // Se il plugin matrix non è disponibile, usa un grafico alternativo (doughnut)
            confusionMatrixChart = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: ['Veri Negativi', 'Falsi Positivi', 'Falsi Negativi', 'Veri Positivi'],
                    datasets: [{
                        data: [cm[0][0], cm[0][1], cm[1][0], cm[1][1]],
                        backgroundColor: [
                            'rgba(75, 192, 192, 0.6)',  // Veri Negativi
                            'rgba(255, 99, 132, 0.6)',  // Falsi Positivi
                            'rgba(255, 206, 86, 0.6)',  // Falsi Negativi
                            'rgba(54, 162, 235, 0.6)'   // Veri Positivi
                        ]
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        title: {
                            display: true,
                            text: `Matrice di Confusione (${model.toUpperCase()}) - ${timeframe}`
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    const value = context.raw;
                                    const label = context.label;
                                    return `${label}: ${value}`;
                                }
                            }
                        }
                    }
                }
            });
        } else {
            // Usa il grafico matrice se disponibile
            confusionMatrixChart = new Chart(ctx, {
                type: 'matrix',
                data: {
                    datasets: [{
                        label: 'Matrice di Confusione',
                        data: [
                            { x: 'Negativo', y: 'Negativo', v: cm[0][0] },
                            { x: 'Positivo', y: 'Negativo', v: cm[0][1] },
                            { x: 'Negativo', y: 'Positivo', v: cm[1][0] },
                            { x: 'Positivo', y: 'Positivo', v: cm[1][1] }
                        ],
                        backgroundColor: function(context) {
                            const value = context.dataset.data[context.dataIndex].v;
                            const max = Math.max(...cm.flat());
                            const alpha = value / max;
                            return value > 0 ? `rgba(54, 162, 235, ${alpha})` : 'rgba(255, 99, 132, 0.2)';
                        },
                        borderWidth: 1,
                        borderColor: 'rgba(0, 0, 0, 0.1)'
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        title: {
                            display: true,
                            text: `Matrice di Confusione (${model.toUpperCase()}) - ${timeframe}`
                        },
                        legend: {
                            display: false
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    const v = context.dataset.data[context.dataIndex].v;
                                    return `Valore: ${v}`;
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Predetto'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Reale'
                            }
                        }
                    }
                }
            });
        }
    } catch (error) {
        console.error('Errore nella creazione della matrice di confusione:', error);
        // Mostra un messaggio di errore al posto del grafico
        document.getElementById('confusion-matrix-chart').parentNode.innerHTML = `
            <div class="alert alert-warning">
                Non è stato possibile creare la matrice di confusione: ${error.message}
            </div>
        `;
    }
}

function createROCCurveChart(metrics, model, timeframe) {
    // Assicurati che timeframe sia definito
    if (!timeframe) {
        timeframe = document.getElementById('metric-timeframe-select').value || '15m';
        console.warn(`createROCCurveChart: timeframe non specificato, utilizzo valore dal selettore: ${timeframe}`);
    }
    
    // Verifica se ci sono dati per la curva ROC
    if (!metrics.fpr || !metrics.tpr) {
        document.getElementById('roc-curve-chart').parentNode.parentNode.style.display = 'none';
        return;
    }
    
    document.getElementById('roc-curve-chart').parentNode.parentNode.style.display = 'block';
    
    const ctx = document.getElementById('roc-curve-chart').getContext('2d');
    
    rocCurveChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: metrics.fpr,
            datasets: [
                {
                    label: 'ROC Curve',
                    data: metrics.tpr,
                    borderColor: 'rgb(75, 192, 192)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    fill: false,
                    pointRadius: 0
                },
                {
                    label: 'Linea di riferimento',
                    data: metrics.fpr,
                    borderColor: 'rgb(200, 200, 200)',
                    borderDash: [5, 5],
                    fill: false,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: `Curva ROC (${model.toUpperCase()}) - ${timeframe} - AUC: ${metrics.auc ? metrics.auc.toFixed(4) : 'N/A'}`
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'False Positive Rate'
                    },
                    min: 0,
                    max: 1
                },
                y: {
                    title: {
                        display: true,
                        text: 'True Positive Rate'
                    },
                    min: 0,
                    max: 1
                }
            }
        }
    });
}

function createTrainingHistoryChart(metrics, model, timeframe) {
    // Verifica se esiste l'history nell'oggetto metrics
    if (!metrics.history || Object.keys(metrics.history).length === 0) {
        document.getElementById('training-history-chart').parentNode.parentNode.style.display = 'none';
        return;
    }
    
    document.getElementById('training-history-chart').parentNode.parentNode.style.display = 'block';
    
    try {
        const ctx = document.getElementById('training-history-chart').getContext('2d');
        const history = metrics.history;
        
        // Controlla che ci siano dati storia validi
        const valuesArray = Object.values(history);
        if (valuesArray.length === 0 || !Array.isArray(valuesArray[0]) || valuesArray[0].length === 0) {
            throw new Error('Dati di storia non validi');
        }
        
        // Estrai le epoche come asse x
        const epochs = Array.from({ length: valuesArray[0].length }, (_, i) => i + 1);
        
        // Prepara i dataset per il grafico
        const datasets = [];
        const colors = {
            'loss': 'rgba(255, 99, 132, 1)',
            'val_loss': 'rgba(255, 99, 132, 0.5)',
            'accuracy': 'rgba(54, 162, 235, 1)',
            'val_accuracy': 'rgba(54, 162, 235, 0.5)',
            'precision': 'rgba(75, 192, 192, 1)',
            'val_precision': 'rgba(75, 192, 192, 0.5)',
            'recall': 'rgba(255, 206, 86, 1)',
            'val_recall': 'rgba(255, 206, 86, 0.5)',
            'f1_score': 'rgba(153, 102, 255, 1)',
            'val_f1_score': 'rgba(153, 102, 255, 0.5)'
        };
        
        // Crea un dataset per ogni metrica nell'history
        Object.keys(history).forEach(metric => {
            // Salta metriche con dati non validi
            if (!Array.isArray(history[metric]) || history[metric].some(val => val === undefined || val === null)) {
                console.warn(`Metrica ${metric} contiene valori non validi e verrà saltata`);
                return;
            }
            
            const color = colors[metric] || `rgba(${Math.floor(Math.random() * 255)}, ${Math.floor(Math.random() * 255)}, ${Math.floor(Math.random() * 255)}, 0.8)`;
            
            datasets.push({
                label: metric,
                data: history[metric],
                borderColor: color,
                backgroundColor: color.replace('1)', '0.1)'),
                borderWidth: 2,
                pointRadius: 3,
                pointHoverRadius: 5,
                fill: false,
                tension: 0.1
            });
        });
        
        // Se non ci sono dataset validi, nascondi il grafico
        if (datasets.length === 0) {
            throw new Error('Nessun dataset valido trovato nella storia');
        }
        
        // Crea il grafico
        trainingHistoryChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: epochs,
                datasets: datasets
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: `Storia dell'Addestramento (${model.toUpperCase()}) - ${timeframe}`
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    },
                    legend: {
                        position: 'top',
                    }
                },
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Epoca'
                        }
                    },
                    y: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Valore'
                        }
                    }
                }
            }
        });
    } catch (error) {
        console.error('Errore nella creazione del grafico della storia dell\'addestramento:', error);
        document.getElementById('training-history-chart').parentNode.innerHTML = `
            <div class="alert alert-warning">
                Non è stato possibile creare il grafico: ${error.message}
            </div>
        `;
    }
}

function loadModelDetails(model) {
    // ... existing code ...
    
    fetch(`/api/model_metrics?model=${model}`)
        .then(response => response.json())
        .then(metrics => {
            // Aggiorna le metriche del modello nella pagina
            if (metrics.error) {
                document.getElementById('model-metrics').innerHTML = `<div class="alert alert-danger">${metrics.error}</div>`;
                return;
            }
            
            // Nascondi il messaggio di caricamento
            document.getElementById('loading-metrics').style.display = 'none';
            
            // Aggiorna la tabella delle metriche
            const metricsTable = document.getElementById('metrics-table');
            metricsTable.innerHTML = '';
            
            // Aggiungi intestazioni
            const headerRow = document.createElement('tr');
            headerRow.innerHTML = '<th>Metrica</th><th>Valore</th>';
            metricsTable.appendChild(headerRow);
            
            // Aggiungi righe per ogni metrica
            const metricsToShow = [
                { key: 'accuracy', label: 'Accuratezza' },
                { key: 'precision', label: 'Precisione' },
                { key: 'recall', label: 'Richiamo' },
                { key: 'f1_score', label: 'F1-Score' },
                { key: 'roc_auc', label: 'ROC AUC' }
            ];
            
            metricsToShow.forEach(metricInfo => {
                if (metrics[metricInfo.key] !== undefined) {
                    const row = document.createElement('tr');
                    const formattedValue = typeof metrics[metricInfo.key] === 'number' 
                        ? metrics[metricInfo.key].toFixed(4)
                        : metrics[metricInfo.key];
                    row.innerHTML = `<td>${metricInfo.label}</td><td>${formattedValue}</td>`;
                    metricsTable.appendChild(row);
                }
            });
            
            // Ottieni il timeframe corrente
            const timeframe = document.getElementById('metric-timeframe-select').value || '15m';
            
            // Crea grafici
            createRocCurveChart(metrics, model, timeframe);
            createConfusionMatrixChart(metrics, model, timeframe);
            createTrainingHistoryChart(metrics, model, timeframe);
        })
        .catch(error => {
            console.error('Errore nel caricamento delle metriche:', error);
            document.getElementById('model-metrics').innerHTML = `
                <div class="alert alert-danger">
                    Si è verificato un errore durante il caricamento delle metriche: ${error.message}
                </div>
            `;
        });
}

// Funzione per caricare le metriche di tutti i modelli per un dato timeframe
async function loadComparisonMetrics(timeframe) {
    try {
        appendToLog(`Caricamento comparativo metriche per timeframe ${timeframe}...`);
        
        // Carica le metriche per ogni tipo di modello
        const modelTypes = ['lstm', 'rf', 'xgb'];
        const metricsPromises = modelTypes.map(model => 
            fetch(`${API_BASE_URL}/model/metrics/${model}/${timeframe}`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    'api-key': apiKey,
                    'api-secret': apiSecret
                }
            })
            .then(response => {
                if (!response.ok) {
                    if (response.status === 404) {
                        console.warn(`Metriche per ${model}_${timeframe} non trovate`);
                        return null;
                    }
                    throw new Error(`Errore HTTP ${response.status} per ${model}`);
                }
                return response.json();
            })
            .then(data => {
                if (!data) return null;
                
                console.log(`Dati ricevuti per ${model}:`, data);
                
                // Adatta le metriche LSTM al formato standard
                if (model === 'lstm' && data.train && data.validation) {
                    return {
                        model: 'LSTM',
                        accuracy: data.validation.accuracy,
                        precision: data.validation.precision,
                        recall: data.validation.recall,
                        f1_score: data.validation.auc // Approssimazione
                    };
                } else if ((model === 'rf' || model === 'xgb') && data.validation_accuracy) {
                    // Per Random Forest e XGBoost, il formato è come visto nei file JSON
                    return {
                        model: model === 'rf' ? 'Random Forest' : 'XGBoost',
                        accuracy: data.validation_accuracy,
                        precision: data.validation_precision,
                        recall: data.validation_recall,
                        f1_score: data.validation_f1
                    };
                } else if (model === 'rf' || model === 'xgb') {
                    // Formato alternativo per RF e XGB
                    return {
                        model: model === 'rf' ? 'Random Forest' : 'XGBoost',
                        accuracy: data.accuracy || (data.test_accuracy || data.train_accuracy || data.val_accuracy || 0),
                        precision: data.precision || (data.test_precision || data.train_precision || data.val_precision || 0),
                        recall: data.recall || (data.test_recall || data.train_recall || data.val_recall || 0),
                        f1_score: data.f1_score || data.f1 || (data.test_f1_score || data.train_f1_score || data.val_f1_score || 0)
                    };
                }
                
                // Formato generico per qualsiasi altro modello
                return {
                    model: model.toUpperCase(),
                    accuracy: data.accuracy || 0,
                    precision: data.precision || 0,
                    recall: data.recall || 0,
                    f1_score: data.f1_score || data.f1 || 0
                };
            })
            .catch(error => {
                console.error(`Errore caricamento ${model}:`, error);
                return null;
            })
        );
        
        // Attendi che tutte le richieste siano completate
        const metricsResults = await Promise.all(metricsPromises);
        
        // Filtra i risultati nulli
        const validMetrics = metricsResults.filter(m => m !== null);
        
        console.log('Metriche valide:', validMetrics);
        
        if (validMetrics.length === 0) {
            throw new Error(`Nessuna metrica trovata per il timeframe ${timeframe}`);
        }
        
        appendToLog(`Confronto metriche caricato per timeframe ${timeframe}`);
        
        // Visualizza le metriche in una tabella comparativa
        displayComparisonTable(validMetrics);
        
        // Crea il grafico comparativo
        createComparisonChart(validMetrics, timeframe);
        
    } catch (error) {
        appendToLog(`Errore nel caricamento comparativo: ${error.message}`);
        console.error('Error loading comparison metrics:', error);
        
        document.getElementById('comparison-metrics').innerHTML = `
            <div class="alert alert-danger">
                Errore nel caricamento delle metriche comparative: ${error.message}
            </div>
        `;
    }
}

// Funzione per visualizzare la tabella comparativa
function displayComparisonTable(metricsArray) {
    const container = document.getElementById('comparison-metrics');
    
    // Crea la tabella
    let tableHtml = `
        <table class="table table-striped">
            <thead>
                <tr>
                    <th>Metrica</th>
                    ${metricsArray.map(m => `<th>${m.model}</th>`).join('')}
                </tr>
            </thead>
            <tbody>
    `;
    
    // Aggiungi righe per ogni metrica
    const metricNames = [
        { key: 'accuracy', label: 'Accuratezza' },
        { key: 'precision', label: 'Precisione' },
        { key: 'recall', label: 'Richiamo' },
        { key: 'f1_score', label: 'F1-Score' }
    ];
    
    metricNames.forEach(({ key, label }) => {
        tableHtml += `<tr><td>${label}</td>`;
        
        metricsArray.forEach(metrics => {
            const value = metrics[key];
            const formattedValue = value !== undefined ? 
                (value * 100).toFixed(2) + '%' : 'N/A';
            
            tableHtml += `<td>${formattedValue}</td>`;
        });
        
        tableHtml += `</tr>`;
    });
    
    tableHtml += `</tbody></table>`;
    
    container.innerHTML = tableHtml;
}

// Funzione per creare il grafico a barre comparativo
function createComparisonChart(metricsArray, timeframe) {
    // Cancella il grafico esistente se presente
    if (comparisonChart) {
        comparisonChart.destroy();
        comparisonChart = null;
    }
    
    const ctx = document.getElementById('comparison-chart').getContext('2d');
    
    // Prepara i dati per il grafico
    const labels = ['Accuratezza', 'Precisione', 'Richiamo', 'F1-Score'];
    const datasets = metricsArray.map((metrics, index) => {
        // Colori per ogni modello
        const colors = [
            'rgba(54, 162, 235, 0.7)',  // LSTM - blu
            'rgba(75, 192, 192, 0.7)',  // RF - verde
            'rgba(255, 159, 64, 0.7)'   // XGB - arancione
        ];
        
        // Estrai i valori delle metriche
        return {
            label: metrics.model,
            data: [
                metrics.accuracy !== undefined ? metrics.accuracy * 100 : 0,
                metrics.precision !== undefined ? metrics.precision * 100 : 0,
                metrics.recall !== undefined ? metrics.recall * 100 : 0,
                metrics.f1_score !== undefined ? metrics.f1_score * 100 : 0
            ],
            backgroundColor: colors[index % colors.length],
            borderColor: colors[index % colors.length].replace('0.7', '1'),
            borderWidth: 1
        };
    });
    
    // Crea il grafico a barre
    comparisonChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: `Confronto Modelli - Timeframe ${timeframe}`
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: ${context.raw.toFixed(2)}%`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Percentuale'
                    },
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                }
            }
        }
    });
}

// Funzione per caricare i simboli per il grafico
async function loadChartSymbols() {
    const symbolSelect = document.getElementById('chart-symbol-select');
    const timeframeSelect = document.getElementById('chart-timeframe-select');
    
    if (!symbolSelect || !timeframeSelect) return;
    
    try {
        // Ottieni le posizioni aperte per popolare il selettore
        const positions = await makeApiRequest('/orders/open');
        
        // Pulisci il selettore dei simboli e aggiungi l'opzione predefinita
        symbolSelect.innerHTML = '<option value="">Seleziona simbolo</option>';
        
        // Set per evitare duplicati
        const uniqueSymbols = new Set();
        
        // Aggiungi simboli delle posizioni aperte
        if (positions && positions.length > 0) {
            positions.forEach(position => {
                if (position.symbol && !uniqueSymbols.has(position.symbol)) {
                    uniqueSymbols.add(position.symbol);
                    const option = document.createElement('option');
                    option.value = position.symbol;
                    option.textContent = position.symbol;
                    symbolSelect.appendChild(option);
                }
            });
        }
        
        // Usa timeframe predefiniti invece di chiamare l'API
        const defaultTimeframes = [
            { value: '15m', text: '15 minuti' },
            { value: '30m', text: '30 minuti' },
            { value: '1h', text: '1 ora' }
        ];
        
        // Pulisci il selettore dei timeframe
        timeframeSelect.innerHTML = '';
        
        // Aggiungi i timeframe predefiniti
        defaultTimeframes.forEach(tf => {
            const option = document.createElement('option');
            option.value = tf.value;
            option.textContent = tf.text;
            timeframeSelect.appendChild(option);
        });
        
        // Event listener per il cambio di simbolo
        symbolSelect.addEventListener('change', (e) => {
            const selectedSymbol = e.target.value;
            if (selectedSymbol) {
                const selectedTimeframe = timeframeSelect.value;
                loadChartData(selectedSymbol, selectedTimeframe);
            } else {
                // Se nessun simbolo è selezionato, puliamo il grafico
                clearChart();
            }
        });
        
        // Event listener per il cambio di timeframe
        timeframeSelect.addEventListener('change', (e) => {
            const selectedTimeframe = e.target.value;
            const selectedSymbol = symbolSelect.value;
            if (selectedSymbol) {
                loadChartData(selectedSymbol, selectedTimeframe);
            }
        });
        
        // Se ci sono simboli, carichiamo automaticamente il primo
        if (uniqueSymbols.size > 0) {
            const firstSymbol = [...uniqueSymbols][0];
            symbolSelect.value = firstSymbol;
            loadChartData(firstSymbol, timeframeSelect.value);
        }
    } catch (error) {
        console.error('Errore nel caricamento dei simboli:', error);
        appendToLog(`Errore nel caricamento dei simboli: ${error.message}`);
    }
}

// Funzione per caricare i dati del grafico
async function loadChartData(symbol, timeframe = '15m', limit = 100) {
    try {
        appendToLog(`Caricamento grafico per ${symbol} (${timeframe})...`);
        
        // Codifica il simbolo per l'URL
        const encodedSymbol = encodeURIComponent(symbol);
        
        const chartData = await makeApiRequest(`/chart-data/${encodedSymbol}?timeframe=${timeframe}&limit=${limit}`);
        
        if (chartData && chartData.labels && chartData.open) {
            // Crea o aggiorna il grafico
            createOrUpdateChart(chartData, symbol, timeframe);
        }
    } catch (error) {
        console.error('Errore nel caricamento dei dati del grafico:', error);
        appendToLog(`Errore nel caricamento del grafico: ${error.message}`);
    }
}

// Funzione per creare o aggiornare il grafico
function createOrUpdateChart(data, symbol, timeframe) {
    const ctx = document.getElementById('position-chart').getContext('2d');
    
    // Se esiste già un grafico, distruggilo
    if (positionChart) {
        positionChart.destroy();
    }
    
    // Prepara i dati per il grafico a candele
    const candleData = [];
    const volumeData = [];
    
    for (let i = 0; i < data.timestamps.length; i++) {
        candleData.push({
            x: data.timestamps[i],
            o: data.open[i],
            h: data.high[i],
            l: data.low[i],
            c: data.close[i]
        });
        
        // Prepara i dati dei volumi
        const isGreen = data.close[i] >= data.open[i];
        volumeData.push({
            x: data.timestamps[i],
            y: data.volumes[i],
            color: isGreen ? 'rgba(75, 192, 192, 0.5)' : 'rgba(255, 99, 132, 0.5)'
        });
    }
    
    // Calcola min e max per la scala del grafico
    const minPrice = Math.min(...data.low);
    const maxPrice = Math.max(...data.high);
    const padding = (maxPrice - minPrice) * 0.1;
    
    // Calcola max volume per la scala
    const maxVolume = Math.max(...data.volumes);
    
    // Crea un nuovo grafico a candele con volumi
    positionChart = new Chart(ctx, {
        data: {
            datasets: [
                {
                    type: 'candlestick',
                    label: `${symbol} (${timeframe})`,
                    data: candleData,
                    color: {
                        up: 'rgba(75, 192, 192, 1)',
                        down: 'rgba(255, 99, 132, 1)',
                        unchanged: 'rgba(110, 110, 110, 1)',
                    },
                    yAxisID: 'y-price'
                },
                {
                    type: 'bar',
                    label: 'Volume',
                    data: volumeData,
                    backgroundColor: volumeData.map(v => v.color),
                    yAxisID: 'y-volume'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function(context) {
                            const dataset = context.dataset;
                            if (dataset.type === 'bar') {
                                return `Volume: ${parseFloat(context.raw.y).toLocaleString()}`;
                            } else {
                                const point = context.raw;
                                return [
                                    `Apertura: ${point.o.toFixed(4)}`,
                                    `Massimo: ${point.h.toFixed(4)}`,
                                    `Minimo: ${point.l.toFixed(4)}`,
                                    `Chiusura: ${point.c.toFixed(4)}`
                                ];
                            }
                        }
                    }
                },
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: `${symbol} - Timeframe: ${timeframe}`
                }
            },
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'minute',
                        displayFormats: {
                            minute: 'HH:mm',
                            hour: 'dd HH:mm',
                            day: 'MMM dd'
                        },
                        tooltipFormat: 'dd MMM yyyy HH:mm'
                    },
                    ticks: {
                        maxRotation: 0,
                        autoSkip: true,
                        maxTicksLimit: 10
                    }
                },
                'y-price': {
                    position: 'right',
                    beginAtZero: false,
                    min: minPrice - padding,
                    max: maxPrice + padding,
                    ticks: {
                        callback: function(value) {
                            return value.toFixed(4);
                        }
                    }
                },
                'y-volume': {
                    position: 'left',
                    display: true,
                    beginAtZero: true,
                    max: maxVolume * 3,
                    grid: {
                        drawOnChartArea: false
                    },
                    ticks: {
                        callback: function(value) {
                            if (value >= 1000000) return (value / 1000000).toFixed(1) + 'M';
                            if (value >= 1000) return (value / 1000).toFixed(1) + 'K';
                            return value;
                        }
                    }
                }
            },
            interaction: {
                mode: 'index',
                intersect: false
            }
        }
    });
    
    appendToLog(`Grafico a candele con volumi per ${symbol} (${timeframe}) aggiornato.`);
}

// Funzione per pulire il grafico
function clearChart() {
    if (positionChart) {
        positionChart.destroy();
        positionChart = null;
    }
}

// Funzione per aggiornare la visualizzazione delle fasi del training con la nuova UI
function updateTrainingPhase(phase, progress) {
    // Seleziona gli elementi delle fasi
    const initStep = document.getElementById('step-init');
    const dataStep = document.getElementById('step-data');
    const prepStep = document.getElementById('step-prep');
    const trainStep = document.getElementById('step-train');
    const completeStep = document.getElementById('step-complete');
    
    // Seleziona la barra di progresso
    const progressBar = document.getElementById('progress-bar-fill');
    
    // Nascondi lo step se siamo ancora in fase "non iniziato"
    if (progress === 0 && phase !== 'running') {
        initStep.classList.remove('active');
        dataStep.classList.remove('active');
        prepStep.classList.remove('active');
        trainStep.classList.remove('active');
        completeStep.classList.remove('active');
        
        initStep.classList.remove('completed');
        dataStep.classList.remove('completed');
        prepStep.classList.remove('completed');
        trainStep.classList.remove('completed');
        
        // Nascondi completamente la barra di progresso
        progressBar.style.width = '0%';
        return;
    }
    
    // Reimposta il progresso della barra
    progressBar.style.width = `${progress}%`;
    
    if (phase === 'completed') {
        // Tutte le fasi sono completate
        initStep.classList.remove('active');
        dataStep.classList.remove('active');
        prepStep.classList.remove('active');
        trainStep.classList.remove('active');
        completeStep.classList.add('active');
        
        initStep.classList.add('completed');
        dataStep.classList.add('completed');
        prepStep.classList.add('completed');
        trainStep.classList.add('completed');
        
        // Se il training è completato e ci sono metriche, mostrale
        document.getElementById('training-metrics').classList.remove('d-none');
    } else if (phase === 'running') {
        // In base al progresso, determina quale fase è attiva
        if (progress < 10) {
            // Fase di inizializzazione
            initStep.classList.add('active');
            dataStep.classList.remove('active');
            prepStep.classList.remove('active');
            trainStep.classList.remove('active');
            completeStep.classList.remove('active');
            
            initStep.classList.remove('completed');
            dataStep.classList.remove('completed');
            prepStep.classList.remove('completed');
            trainStep.classList.remove('completed');
        } else if (progress < 40) {
            // Fase di recupero dati
            initStep.classList.remove('active');
            dataStep.classList.add('active');
            prepStep.classList.remove('active');
            trainStep.classList.remove('active');
            completeStep.classList.remove('active');
            
            initStep.classList.add('completed');
            dataStep.classList.remove('completed');
            prepStep.classList.remove('completed');
            trainStep.classList.remove('completed');
        } else if (progress < 60) {
            // Fase di preparazione
            initStep.classList.remove('active');
            dataStep.classList.remove('active');
            prepStep.classList.add('active');
            trainStep.classList.remove('active');
            completeStep.classList.remove('active');
            
            initStep.classList.add('completed');
            dataStep.classList.add('completed');
            prepStep.classList.remove('completed');
            trainStep.classList.remove('completed');
        } else {
            // Fase di training
            initStep.classList.remove('active');
            dataStep.classList.remove('active');
            prepStep.classList.remove('active');
            trainStep.classList.add('active');
            completeStep.classList.remove('active');
            
            initStep.classList.add('completed');
            dataStep.classList.add('completed');
            prepStep.classList.add('completed');
            trainStep.classList.remove('completed');
        }
        
        // Nascondi le metriche durante il training
        document.getElementById('training-metrics').classList.add('d-none');
    }
}

// Funzione per aggiornare i dettagli di una fase specifica
function updateStepDetails(step, details) {
    const detailsElement = document.getElementById(`step-${step}-details`);
    if (detailsElement) {
        detailsElement.textContent = details;
    }
}

// Aggiorna il terminale con informazioni più semplici
function updateTerminalProgress(progress, currentStep, totalItems = 0, currentItem = 0) {
    const progressElement = document.getElementById('terminal-progress-text');
    if (!progressElement) return;
    
    // Nascondi completamente la visualizzazione se siamo all'inizio del processo (0%)
    if (progress === 0) {
        progressElement.parentElement.style.display = 'none';
        return;
    } else {
        progressElement.parentElement.style.display = 'block';
    }
    
    // Formatta il messaggio in modo più semplice, senza percentuali o tempi
    let stepDescription = currentStep || "Elaborazione";
    
    // Determina la fase attiva e una descrizione più dettagliata
    if (progress < 10) {
        stepDescription = "Inizializzazione";
    } else if (progress < 40) {
        stepDescription = "Recupero dati";
        updateStepDetails('data', 'Recupero dati in corso');
    } else if (progress < 60) {
        stepDescription = "Preparazione dati";
        updateStepDetails('prep', 'Elaborazione features');
    } else if (progress < 100) {
        stepDescription = "Training modello";
        updateStepDetails('train', 'Addestramento in corso');
    } else {
        stepDescription = "Completato";
        updateStepDetails('complete', 'Modello addestrato');
    }
    
    // Crea il messaggio finale semplificato
    progressElement.textContent = stepDescription;
}

// Aggiorna la funzione updateTrainingProgress per includere l'aggiornamento delle fasi
function updateTrainingProgress(progress) {
    // Aggiorna la visualizzazione delle fasi
    updateTrainingPhase('running', progress);
}

// Aggiorna la funzione updateTrainingStatus per includere l'aggiornamento delle fasi
function updateTrainingStatus(status, type) {
    const statusElement = document.getElementById('training-status');
    statusElement.textContent = status;
    statusElement.className = `badge bg-${type}`;
    
    // Aggiorna la visualizzazione delle fasi basata sullo stato
    if (status === 'Completato') {
        updateTrainingPhase('completed', 100);
    } else if (status === 'Errore') {
        // In caso di errore, interrompi il progresso della fase
        // ma mantieni la fase corrente attiva
    }
}

function addTrainingLog(message, type = 'info') {
    const logsContainer = document.getElementById('training-logs');
    if (logsContainer) {
        const timestamp = new Date().toLocaleTimeString();
        const logEntry = document.createElement('div');
        logEntry.className = `log-entry ${type}`;
        logEntry.innerHTML = `<span class="log-timestamp">${timestamp}</span> ${message}`;
        logsContainer.appendChild(logEntry);
        logsContainer.scrollTop = logsContainer.scrollHeight;
    }
}

function updateTrainingTime(time) {
    const timeElement = document.getElementById('training-time');
    const timeContainer = document.querySelector('.training-time-container');
    
    if (!timeElement || !timeContainer) return;
    
    // Nascondi il container del tempo se il tempo è --:-- o non definito
    if (time === '--:--' || !time) {
        timeContainer.classList.add('d-none');
    } else {
        timeContainer.classList.remove('d-none');
        timeElement.textContent = `Tempo stimato: ${time}`;
    }
}

// Funzione dedicata per gestire il submit del form
async function handleTrainingFormSubmit(e) {
    e.preventDefault();
    console.log('Form di training inviato');
    
    try {
        const modelType = document.getElementById('model-type')?.value;
        const timeframe = document.getElementById('timeframe')?.value;
        // Utilizziamo valori fissi invece di quelli dinamici
        const dataLimitDays = 50;  // Valore fisso: 50 giorni
        const topTrainCrypto = 100; // Valore fisso: 100 criptovalute
        
        console.log('Parametri di training:', { modelType, timeframe, dataLimitDays, topTrainCrypto });
        
        if (!modelType || !timeframe) {
            console.error('Parametri mancanti', { modelType, timeframe });
            alert('Errore: Parametri mancanti per il training.');
            return;
        }
        
        // Reset UI e nascondo gli elementi iniziali
        updateTrainingProgress(0);
        const statusElement = document.getElementById('training-status');
        if (statusElement) {
            statusElement.style.display = 'none';
        }
        document.getElementById('training-logs').innerHTML = '';
        
        // Aggiorna l'informazione sul modello in training
        updateCurrentTrainingModel(modelType, timeframe);
        
        // Inizializza il timestamp di avvio
        window.trainingStartTime = Date.now();
        updateTerminalProgress(0, "Inizializzazione");
        
        console.log(`Invio richiesta a ${API_BASE_URL}/api/train-model`);
        
        const response = await fetch(`${API_BASE_URL}/api/train-model`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'api-key': apiKey,
                'api-secret': apiSecret
            },
            body: JSON.stringify({
                model_type: modelType,
                timeframe: timeframe,
                data_limit_days: dataLimitDays,
                top_train_crypto: topTrainCrypto
            })
        });
        
        console.log('Risposta ricevuta:', response.status);
        
        let result;
        const contentType = response.headers.get('content-type');
        if (contentType && contentType.includes('application/json')) {
            result = await response.json();
        } else {
            const text = await response.text();
            throw new Error(`Risposta non valida dal server: ${text.substring(0, 100)}...`);
        }

        if (!response.ok) {
            throw new Error(result.error || 'Errore durante il training');
        }

        // Polling per aggiornamenti di stato
        let progress = 0;
        const pollInterval = setInterval(async () => {
            try {
                // Codifica i parametri per l'URL
                const encodedModelType = encodeURIComponent(modelType);
                const encodedTimeframe = encodeURIComponent(timeframe);
                
                const statusResponse = await fetch(`${API_BASE_URL}/api/training-status/${encodedModelType}/${encodedTimeframe}`, {
                    headers: {
                        'Accept': 'application/json',
                        'api-key': apiKey,
                        'api-secret': apiSecret
                    }
                });
                
                if (!statusResponse.ok) {
                    const errorText = await statusResponse.text();
                    throw new Error(`Errore nel recupero dello stato del training: ${errorText}`);
                }

                const statusData = await statusResponse.json();
                
                if (statusData.status === 'completed') {
                    clearInterval(pollInterval);
                    updateTrainingProgress(100);
                    updateTrainingStatus('Completato', 'success');
                    updateTerminalProgress(100, "Completato");
                    
                    // Aggiorna l'informazione sul modello completato
                    resetCurrentTrainingModel('completed');
                    
                    // Aggiorna le metriche se disponibili
                    if (statusData.metrics) {
                        displayMetrics(statusData.metrics, modelType, timeframe);
                    }
                    
                    // Aggiorna lo stato del modello appena addestrato
                    const modelCell = document.getElementById(`${modelType}-${timeframe}`);
                    if (modelCell) {
                        modelCell.innerHTML = '<i class="fas fa-check-circle text-success me-2"></i> Disponibile';
                        modelCell.className = "model-status model-available";
                    }
                    
                    // Aggiorna il grafico delle metriche dopo l'addestramento
                    loadModelMetrics();
                } else if (statusData.status === 'error') {
                    clearInterval(pollInterval);
                    updateTrainingStatus('Errore', 'danger');
                    updateTerminalProgress(0, "Errore");
                    
                    // Aggiorna l'informazione sul modello in errore
                    resetCurrentTrainingModel('error');
                } else {
                    // Aggiorna progresso
                    progress = statusData.progress || progress;
                    updateTrainingProgress(progress);
                    
                    // Aggiorna la barra di progresso testuale
                    let currentStep = "Elaborazione";
                    let totalItems = 0;
                    let currentItem = 0;
                    
                    // Estrai informazioni dal current_step se disponibile
                    if (statusData.current_step) {
                        currentStep = statusData.current_step.split(':')[0].trim();
                        
                        // Prova a estrarre informazioni sul conteggio
                        const countMatch = statusData.current_step.match(/(\d+)\/(\d+)/);
                        if (countMatch) {
                            currentItem = parseInt(countMatch[1], 10);
                            totalItems = parseInt(countMatch[2], 10);
                        }
                    }
                    
                    updateTerminalProgress(progress, currentStep, totalItems, currentItem);
                }
            } catch (error) {
                clearInterval(pollInterval);
                updateTrainingStatus('Errore', 'danger');
                updateTerminalProgress(0, "Errore");
                console.error('Errore nel monitoraggio del training:', error);
            }
        }, 2000); // Polling ogni 2 secondi

    } catch (error) {
        console.error('Errore durante il training:', error);
        updateTrainingStatus('Errore', 'danger');
        updateTerminalProgress(0, "Errore");
    }
}

// Funzione per controllare lo stato di un modello
async function checkModelStatus(modelType, timeframe) {
    try {
        const cellId = `${modelType}-${timeframe}`;
        const cell = document.getElementById(cellId);
        
        if (!cell) {
            console.error(`Cella ${cellId} non trovata`);
            return;
        }
        
        // Imposta lo stato iniziale
        cell.innerHTML = '<i class="fas fa-spinner fa-spin text-warning me-2"></i> Verifica in corso...';
        cell.className = "model-status model-checking";
        
        let fileName;
        switch(modelType) {
            case 'lstm':
                fileName = `lstm_model_${timeframe}.h5`;
                break;
            case 'rf':
                fileName = `rf_model_${timeframe}.pkl`;
                break;
            case 'xgb':
                fileName = `xgb_model_${timeframe}.pkl`;
                break;
            default:
                console.error(`Tipo di modello sconosciuto: ${modelType}`);
                return;
        }
        
        try {
            // Verifica se il file del modello esiste usando una richiesta HEAD
            const response = await fetch(`${API_BASE_URL}/check-model-exists/${fileName}`, {
                method: 'HEAD',
                headers: {
                    'api-key': apiKey,
                    'api-secret': apiSecret
                }
            });
            
            if (response.ok) {
                cell.innerHTML = '<i class="fas fa-check-circle text-success me-2"></i> Disponibile';
                cell.className = "model-status model-available";
            } else {
                cell.innerHTML = '<i class="fas fa-times-circle text-danger me-2"></i> Non disponibile';
                cell.className = "model-status model-unavailable";
            }
        } catch (error) {
            // In caso di errore, assumiamo che il modello non esista
            cell.innerHTML = '<i class="fas fa-times-circle text-danger me-2"></i> Non disponibile';
            cell.className = "model-status model-unavailable";
        }
    } catch (error) {
        console.error(`Errore verifica modello ${modelType} ${timeframe}:`, error);
    }
}

// Funzione per controllare lo stato di tutti i modelli
async function checkAllModelsStatus() {
    const modelTypes = ['lstm', 'rf', 'xgb'];
    const timeframes = ['5m', '15m', '30m', '1h', '4h'];
    
    // Utilizziamo fetch per recuperare la lista dei modelli disponibili
    try {
        const response = await fetch(`${API_BASE_URL}/list-models`, {
            headers: {
                'api-key': apiKey,
                'api-secret': apiSecret
            }
        });
        
        if (response.ok) {
            const data = await response.json();
            const modelFiles = data.models || [];
            
            // Aggiorniamo lo stato dei modelli nella tabella
            for (const modelType of modelTypes) {
                for (const timeframe of timeframes) {
                    const cellId = `${modelType}-${timeframe}`;
                    const cell = document.getElementById(cellId);
                    
                    if (cell) {
                        let fileName;
                        switch(modelType) {
                            case 'lstm':
                                fileName = `lstm_model_${timeframe}.h5`;
                                break;
                            case 'rf':
                                fileName = `rf_model_${timeframe}.pkl`;
                                break;
                            case 'xgb':
                                fileName = `xgb_model_${timeframe}.pkl`;
                                break;
                            default:
                                console.error(`Tipo di modello sconosciuto: ${modelType}`);
                                return;
                        }
                        
                        if (modelFiles.includes(fileName)) {
                            cell.innerHTML = '<i class="fas fa-check-circle text-success me-2"></i> Disponibile';
                            cell.className = "model-status model-available";
                        } else {
                            cell.innerHTML = '<i class="fas fa-times-circle text-danger me-2"></i> Non disponibile';
                            cell.className = "model-status model-unavailable";
                        }
                    }
                }
            }
            
            // Carica le metriche dopo aver verificato i modelli disponibili
            loadModelMetrics();
        } else {
            // Se non riusciamo a ottenere la lista dei modelli, utilizziamo il metodo fallback
            fallbackCheckModels(modelTypes, timeframes);
        }
    } catch (error) {
        console.error("Errore recupero lista modelli:", error);
        // In caso di errore, utilizziamo il metodo fallback
        fallbackCheckModels(modelTypes, timeframes);
    }
}

// Metodo fallback per controllare lo stato dei modelli
function fallbackCheckModels(modelTypes, timeframes) {
    // Imposta tutti i modelli come "Non disponibile" poiché non possiamo verificare effettivamente
    for (const modelType of modelTypes) {
        for (const timeframe of timeframes) {
            const cellId = `${modelType}-${timeframe}`;
            const cell = document.getElementById(cellId);
            
            if (cell) {
                cell.innerHTML = '<i class="fas fa-times-circle text-danger me-2"></i> Non disponibile';
                cell.className = "model-status model-unavailable";
            }
        }
    }
}

// Funzione per caricare e visualizzare le metriche dei modelli
function loadModelMetrics() {
    const timeframeSelect = document.getElementById('metrics-timeframe-select');
    if (!timeframeSelect) return;
    
    const timeframe = timeframeSelect.value;
    
    // Disabilita tutti i pulsanti delle metriche durante il caricamento
    document.querySelectorAll('[data-metric]').forEach(btn => {
        btn.disabled = true;
    });
    
    // Mostra un loader
    document.getElementById('metrics-comparison-data').innerHTML = `
        <div class="d-flex justify-content-center my-4">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Caricamento...</span>
            </div>
        </div>
    `;
    
    // Recupera le metriche per questo timeframe
    fetchModelMetrics(timeframe)
        .then(metrics => {
            console.log('Metriche caricate:', metrics);
            modelMetricsData = metrics;
            
            // Visualizza il grafico con i dati caricati
            displayMetricsChart();
            
            // Riabilita i pulsanti delle metriche
            document.querySelectorAll('[data-metric]').forEach(btn => {
                btn.disabled = false;
            });
        })
        .catch(error => {
            console.error('Errore nel caricamento delle metriche:', error);
            document.getElementById('metrics-comparison-data').innerHTML = `
                <div class="alert alert-danger">
                    Errore nel caricamento delle metriche: ${error.message}
                </div>
            `;
            
            // Riabilita l'interazione
            document.querySelectorAll('[data-metric]').forEach(btn => {
                btn.disabled = false;
            });
        });
}

async function fetchModelMetrics(timeframe) {
    // Lista dei tipi di modelli da controllare
    const modelTypes = ['lstm', 'rf', 'xgb'];
    const modelNames = {
        'lstm': 'LSTM',
        'rf': 'Random Forest',
        'xgb': 'XGBoost'
    };
    
    // Recupera il file metrics.json per ogni modello
    const metrics = {};
    
    for (const type of modelTypes) {
        try {
            // Formatta il nome del file delle metriche in base al tipo di modello
            const metricsFile = `${type}_model_${timeframe}_metrics.json`;
            
            // Controlla prima se il modello esiste
            const modelFile = type === 'lstm' ? `lstm_model_${timeframe}.h5` : `${type}_model_${timeframe}.pkl`;
            const modelResponse = await fetch(`${API_BASE_URL}/check-model-exists/${modelFile}`, {
                headers: {
                    'api-key': apiKey,
                    'api-secret': apiSecret
                }
            });
            
            if (modelResponse.ok) {
                // Ora ottieni il contenuto effettivo delle metriche
                const metricDetailsResponse = await fetch(`${API_BASE_URL}/trained_models/${metricsFile}`, {
                    headers: {
                        'api-key': apiKey,
                        'api-secret': apiSecret
                    }
                });
                
                if (metricDetailsResponse.ok) {
                    const data = await metricDetailsResponse.json();
                    
                    // Estrai le metriche rilevanti e standardizzale
                    metrics[type] = {
                        name: modelNames[type],
                        type: type,
                        accuracy: extractMetric(data, 'accuracy'),
                        precision: extractMetric(data, 'precision'),
                        recall: extractMetric(data, 'recall'),
                        f1: extractMetric(data, 'f1'),
                        auc: extractMetric(data, 'auc'),
                        // Salva anche i dati grezzi per i dettagli
                        rawMetrics: data
                    };
                }
            }
        } catch (error) {
            console.warn(`Errore nel recupero delle metriche per ${type} ${timeframe}:`, error);
        }
    }
    
    return metrics;
}

// Funzione ausiliaria per estrarre una metrica standardizzata dai dati
function extractMetric(data, metricName) {
    // LSTM models
    if (data.validation && data.validation[metricName] !== undefined) {
        return data.validation[metricName];
    }
    
    // RF and XGBoost models
    if (data.validation && data.validation[metricName] !== undefined) {
        return data.validation[metricName];
    }
    
    if (data.validation_accuracy && metricName === 'accuracy') {
        return data.validation_accuracy;
    }
    
    if (data.validation_precision && metricName === 'precision') {
        return data.validation_precision;
    }
    
    if (data.validation_recall && metricName === 'recall') {
        return data.validation_recall;
    }
    
    if ((data.validation_f1 || data.validation_f1_score) && metricName === 'f1') {
        return data.validation_f1 || data.validation_f1_score;
    }
    
    if (data.roc_auc && metricName === 'auc') {
        return data.roc_auc;
    }
    
    // Prova a cercare anche altrove nei dati
    if (data[metricName] !== undefined) {
        return data[metricName];
    }
    
    return 0; // Valore predefinito se non trovato
}

function displayMetricsChart() {
    const ctx = document.getElementById('metrics-comparison-chart').getContext('2d');
    
    // Distruggi il grafico esistente, se presente
    if (metricsChart) {
        metricsChart.destroy();
    }
    
    // Prepara i dati per il grafico
    const models = Object.keys(modelMetricsData);
    
    if (models.length === 0) {
        document.getElementById('metrics-comparison-data').innerHTML = `
            <div class="alert alert-warning">
                <i class="fas fa-info-circle me-2"></i>
                Nessun modello disponibile per questo timeframe.
            </div>
        `;
        return;
    }
    
    // Ottieni le etichette (nomi dei modelli)
    const labels = models.map(model => modelMetricsData[model].name);
    
    // Prepara i dati per la metrica selezionata
    const data = models.map(model => (modelMetricsData[model][currentMetric] || 0) * 100);
    
    // Colori per ogni tipo di modello
    const backgroundColors = models.map(model => {
        switch(model) {
            case 'lstm': return 'rgba(54, 162, 235, 0.7)';
            case 'rf': return 'rgba(75, 192, 192, 0.7)';
            case 'xgb': return 'rgba(255, 159, 64, 0.7)';
            default: return 'rgba(201, 203, 207, 0.7)';
        }
    });
    
    const borderColors = backgroundColors.map(color => color.replace('0.7', '1.0'));
    
    // Crea il grafico
    metricsChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: getMetricLabel(currentMetric),
                data: data,
                backgroundColor: backgroundColors,
                borderColor: borderColors,
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Percentuale'
                    },
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: `${getMetricLabel(currentMetric)} per Modello (Timeframe: ${document.getElementById('metrics-timeframe-select').value})`,
                    font: {
                        size: 16
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: ${context.raw.toFixed(2)}%`;
                        }
                    }
                }
            }
        }
    });
    
    // Visualizza una tabella con i dati
    displayMetricsTable();
    
    // Visualizza i dettagli completi dei modelli
    displayDetailedMetrics();
}

function displayMetricsTable() {
    const models = Object.keys(modelMetricsData);
    
    if (models.length === 0) return;
    
    // Crea la tabella HTML
    let tableHtml = `
        <div class="table-responsive">
            <table class="table table-sm table-bordered">
                <thead>
                    <tr>
                        <th>Modello</th>
                        <th>Accuratezza</th>
                        <th>Precisione</th>
                        <th>Richiamo</th>
                        <th>F1-Score</th>
                        <th>AUC</th>
                    </tr>
                </thead>
                <tbody>
    `;
    
    // Aggiungi una riga per ogni modello
    models.forEach(model => {
        const metrics = modelMetricsData[model];
        const accuracy = (metrics.accuracy || 0) * 100;
        const precision = (metrics.precision || 0) * 100;
        const recall = (metrics.recall || 0) * 100;
        const f1 = (metrics.f1 || 0) * 100;
        const auc = (metrics.auc || 0) * 100;
        
        // Evidenzia la metrica attualmente selezionata
        const accuracyClass = currentMetric === 'accuracy' ? 'table-primary' : '';
        const precisionClass = currentMetric === 'precision' ? 'table-primary' : '';
        const recallClass = currentMetric === 'recall' ? 'table-primary' : '';
        const f1Class = currentMetric === 'f1' ? 'table-primary' : '';
        const aucClass = currentMetric === 'auc' ? 'table-primary' : '';
        
        tableHtml += `
            <tr>
                <td><strong>${metrics.name}</strong></td>
                <td class="${accuracyClass}">${accuracy.toFixed(2)}%</td>
                <td class="${precisionClass}">${precision.toFixed(2)}%</td>
                <td class="${recallClass}">${recall.toFixed(2)}%</td>
                <td class="${f1Class}">${f1.toFixed(2)}%</td>
                <td class="${aucClass}">${auc.toFixed(2)}%</td>
            </tr>
        `;
    });
    
    tableHtml += `
                </tbody>
            </table>
        </div>
    `;
    
    document.getElementById('metrics-comparison-data').innerHTML = tableHtml;
}

// Funzione per visualizzare i dettagli completi delle metriche di ogni modello
function displayDetailedMetrics() {
    const models = Object.keys(modelMetricsData);
    const timeframe = document.getElementById('metrics-timeframe-select').value;
    
    if (models.length === 0) return;
    
    // Crea una sezione accordion per i dettagli estesi
    let detailsHtml = `
        <div class="accordion mt-4" id="modelsDetailsAccordion">
    `;
    
    // Aggiungi una sezione per ogni modello
    models.forEach((model, index) => {
        const metrics = modelMetricsData[model].rawMetrics || {};
        const trainingInfo = metrics.training_info || {};
        
        detailsHtml += `
            <div class="accordion-item">
                <h2 class="accordion-header" id="heading${index}">
                    <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" 
                        data-bs-target="#collapse${index}" aria-expanded="false" aria-controls="collapse${index}">
                        <strong>${modelMetricsData[model].name}</strong> - Complete Details
                    </button>
                </h2>
                <div id="collapse${index}" class="accordion-collapse collapse" 
                    aria-labelledby="heading${index}" data-bs-parent="#modelsDetailsAccordion">
                    <div class="accordion-body">
                        <div class="row">
                            <div class="col-md-6">
                                <div class="card mb-3">
                                    <div class="card-header">
                                        <h5 class="card-title mb-0">Training Information</h5>
                                    </div>
                                    <div class="card-body">
                                        <table class="table table-sm">
                                            <tr>
                                                <td>Timeframe:</td>
                                                <td><span class="badge bg-info">${trainingInfo.timeframe || timeframe}</span></td>
                                            </tr>
                                            <tr>
                                                <td>Number of Cryptocurrencies:</td>
                                                <td>${trainingInfo.num_cryptocurrencies || 'N/A'}</td>
                                            </tr>
                                            <tr>
                                                <td>Training Period:</td>
                                                <td>${trainingInfo.days_covered ? trainingInfo.days_covered + ' days' : '50 days'}</td>
                                            </tr>
                                            <tr>
                                                <td>Start Date:</td>
                                                <td>${trainingInfo.start_date || 'N/A'}</td>
                                            </tr>
                                            <tr>
                                                <td>End Date:</td>
                                                <td>${trainingInfo.end_date || 'N/A'}</td>
                                            </tr>
                                            <tr>
                                                <td>Total Samples:</td>
                                                <td>${trainingInfo.total_samples || 'N/A'}</td>
                                            </tr>
                                            <tr>
                                                <td>Training Started:</td>
                                                <td>${trainingInfo.training_started || 'N/A'}</td>
                                            </tr>
                                            <tr>
                                                <td>Training Completed:</td>
                                                <td>${trainingInfo.training_completed || 'N/A'}</td>
                                            </tr>
                                        </table>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="card mb-3">
                                    <div class="card-header">
                                        <h5 class="card-title mb-0">Detailed Metrics</h5>
                                    </div>
                                    <div class="card-body">
                                        <table class="table table-sm">
                                            <tr>
                                                <th>Metric</th>
                                                <th>Training</th>
                                                <th>Validation</th>
                                            </tr>
        `;
        
        // Aggiungi le metriche di training e validation in confronto
        if (metrics.train && metrics.validation) {
            const metricsToShow = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'loss'];
            
            metricsToShow.forEach(metric => {
                const trainValue = metrics.train[metric];
                const valValue = metrics.validation[metric];
                
                if (trainValue !== undefined || valValue !== undefined) {
                    const isPercentage = metric !== 'loss';
                    
                    detailsHtml += `
                        <tr>
                            <td>${getMetricLabel(metric)}</td>
                            <td>${trainValue !== undefined ? 
                                (isPercentage ? (trainValue * 100).toFixed(2) + '%' : trainValue.toFixed(4)) 
                                : 'N/A'}</td>
                            <td>${valValue !== undefined ? 
                                (isPercentage ? (valValue * 100).toFixed(2) + '%' : valValue.toFixed(4)) 
                                : 'N/A'}</td>
                        </tr>
                    `;
                }
            });
        }
        
        detailsHtml += `
                                        </table>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    });
    
    detailsHtml += `</div>`;
    
    // Aggiungi la sezione dei dettagli dopo la tabella
    document.getElementById('metrics-comparison-data').innerHTML += detailsHtml;
}

// Funzione di utilità per ottenere l'etichetta di una metrica
function getMetricLabel(metric) {
    switch(metric) {
        case 'accuracy': return 'Accuracy';
        case 'precision': return 'Precision';
        case 'recall': return 'Recall';
        case 'f1': return 'F1-Score';
        case 'auc': return 'AUC';
        case 'loss': return 'Loss';
        default: return metric;
    }
}

// ... existing code ...

// Funzione per avviare il training di tutti i modelli
function startTrainingAllModels() {
    // Definisco i tipi di modelli e i timeframe
    const modelTypes = ['lstm', 'rf', 'xgb'];
    const timeframes = ['5m', '15m', '30m', '1h', '4h'];
    
    // Preparo la lista dei modelli da addestrare
    const trainingQueue = [];
    
    modelTypes.forEach(modelType => {
        timeframes.forEach(timeframe => {
            trainingQueue.push({
                modelType,
                timeframe
            });
        });
    });
    
    // Mostro la barra di progresso
    document.getElementById('train-all-progress').classList.remove('d-none');
    document.getElementById('train-all-title').textContent = 'Training in sequenza...';
    document.getElementById('total-models').textContent = trainingQueue.length;
    document.getElementById('completed-models').textContent = '0';
    document.getElementById('train-all-progress-bar').style.width = '0%';
    
    // Imposta lo stato globale per il training
    window.trainingQueue = trainingQueue;
    window.trainingIndex = 0;
    window.completedModels = 0;
    
    // Avvia il training in sequenza
    processNextTraining();
}

// Funzione per avviare il training dei modelli mancanti
function startTrainingMissingModels() {
    // Definisco i tipi di modelli e i timeframe
    const modelTypes = ['lstm', 'rf', 'xgb'];
    const timeframes = ['5m', '15m', '30m', '1h', '4h'];
    
    // Visualizza un dialogo con i modelli mancanti
    showCheckingModelsDialog();
    
    // Verifica quali modelli sono mancanti
    checkMissingModels(modelTypes, timeframes)
        .then(missingModels => {
            // Nascondi il dialogo di verifica
            hideCheckingModelsDialog();
            
            if (missingModels.length === 0) {
                // Se non ci sono modelli mancanti, mostra un messaggio
                showNoMissingModelsDialog();
                return;
            }
            
            // Mostra i modelli mancanti all'utente e chiedi conferma
            showMissingModelsDialog(missingModels);
        })
        .catch(error => {
            // In caso di errore, nascondi il dialogo e mostra un messaggio di errore
            hideCheckingModelsDialog();
            showErrorDialog('Errore durante la verifica dei modelli: ' + error.message);
        });
}

// Funzione per verificare quali modelli sono mancanti
async function checkMissingModels(modelTypes, timeframes) {
    const missingModels = [];
    
    // Per ogni combinazione di tipo di modello e timeframe
    for (const modelType of modelTypes) {
        for (const timeframe of timeframes) {
            // Controlla se il modello esiste
            const exists = await checkIfModelExists(modelType, timeframe);
            
            if (!exists) {
                // Se il modello non esiste, aggiungilo alla lista dei modelli mancanti
                missingModels.push({
                    modelType,
                    timeframe
                });
            }
        }
    }
    
    return missingModels;
}

// Funzione per controllare se un modello esiste
async function checkIfModelExists(modelType, timeframe) {
    try {
        // Formato del nome del file del modello
        let modelFile;
        if (modelType === 'lstm') {
            modelFile = `lstm_model_${timeframe}.h5`;
        } else {
            modelFile = `${modelType}_model_${timeframe}.pkl`;
        }
        
        // Invia la richiesta per verificare se il modello esiste
        const response = await fetch(`${API_BASE_URL}/check-model-exists/${modelFile}`, {
            headers: {
                'api-key': apiKey,
                'api-secret': apiSecret
            }
        });
        
        return response.ok;
    } catch (error) {
        console.error(`Errore durante la verifica del modello ${modelType}_${timeframe}:`, error);
        return false;
    }
}

// Funzione per mostrare un dialogo di verifica dei modelli
function showCheckingModelsDialog() {
    // Crea un elemento div per il dialogo
    const dialog = document.createElement('div');
    dialog.id = 'checking-models-dialog';
    dialog.className = 'modal fade show';
    dialog.style.display = 'block';
    dialog.setAttribute('tabindex', '-1');
    dialog.setAttribute('aria-modal', 'true');
    dialog.setAttribute('role', 'dialog');
    
    // Aggiungi il contenuto del dialogo
    dialog.innerHTML = `
        <div class="modal-dialog modal-dialog-centered">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">Verifica modelli</h5>
                </div>
                <div class="modal-body text-center">
                    <div class="spinner-border text-primary mb-3" role="status">
                        <span class="visually-hidden">Caricamento...</span>
                    </div>
                    <p>Verificando quali modelli sono mancanti...</p>
                </div>
            </div>
        </div>
    `;
    
    // Aggiungi il backdrop
    const backdrop = document.createElement('div');
    backdrop.className = 'modal-backdrop fade show';
    
    // Aggiungi gli elementi al body
    document.body.appendChild(dialog);
    document.body.appendChild(backdrop);
    document.body.classList.add('modal-open');
}

// Funzione per nascondere il dialogo di verifica
function hideCheckingModelsDialog() {
    // Rimuovi il dialogo e il backdrop
    const dialog = document.getElementById('checking-models-dialog');
    if (dialog) {
        dialog.remove();
    }
    
    // Rimuovi il backdrop
    const backdrop = document.querySelector('.modal-backdrop');
    if (backdrop) {
        backdrop.remove();
    }
    
    document.body.classList.remove('modal-open');
}

// Funzione per mostrare che non ci sono modelli mancanti
function showNoMissingModelsDialog() {
    // Crea un elemento div per il dialogo
    const dialog = document.createElement('div');
    dialog.id = 'no-missing-models-dialog';
    dialog.className = 'modal fade show';
    dialog.style.display = 'block';
    dialog.setAttribute('tabindex', '-1');
    dialog.setAttribute('aria-modal', 'true');
    dialog.setAttribute('role', 'dialog');
    
    // Aggiungi il contenuto del dialogo
    dialog.innerHTML = `
        <div class="modal-dialog modal-dialog-centered">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">Nessun modello mancante</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body text-center">
                    <i class="fas fa-check-circle text-success fa-4x mb-3"></i>
                    <p>Tutti i modelli sono già stati addestrati!</p>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-primary" data-bs-dismiss="modal">OK</button>
                </div>
            </div>
        </div>
    `;
    
    // Aggiungi il backdrop
    const backdrop = document.createElement('div');
    backdrop.className = 'modal-backdrop fade show';
    
    // Aggiungi gli elementi al body
    document.body.appendChild(dialog);
    document.body.appendChild(backdrop);
    document.body.classList.add('modal-open');
    
    // Aggiungi event listener per chiudere il dialogo
    const closeButtons = dialog.querySelectorAll('[data-bs-dismiss="modal"]');
    closeButtons.forEach(button => {
        button.addEventListener('click', () => {
            dialog.remove();
            backdrop.remove();
            document.body.classList.remove('modal-open');
        });
    });
}

// Funzione per mostrare un dialogo di errore
function showErrorDialog(message) {
    // Crea un elemento div per il dialogo
    const dialog = document.createElement('div');
    dialog.id = 'error-dialog';
    dialog.className = 'modal fade show';
    dialog.style.display = 'block';
    dialog.setAttribute('tabindex', '-1');
    dialog.setAttribute('aria-modal', 'true');
    dialog.setAttribute('role', 'dialog');
    
    // Aggiungi il contenuto del dialogo
    dialog.innerHTML = `
        <div class="modal-dialog modal-dialog-centered">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">Errore</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body">
                    <div class="alert alert-danger">
                        ${message}
                    </div>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-primary" data-bs-dismiss="modal">OK</button>
                </div>
            </div>
        </div>
    `;
    
    // Aggiungi il backdrop
    const backdrop = document.createElement('div');
    backdrop.className = 'modal-backdrop fade show';
    
    // Aggiungi gli elementi al body
    document.body.appendChild(dialog);
    document.body.appendChild(backdrop);
    document.body.classList.add('modal-open');
    
    // Aggiungi event listener per chiudere il dialogo
    const closeButtons = dialog.querySelectorAll('[data-bs-dismiss="modal"]');
    closeButtons.forEach(button => {
        button.addEventListener('click', () => {
            dialog.remove();
            backdrop.remove();
            document.body.classList.remove('modal-open');
        });
    });
}

// Funzione per mostrare i modelli mancanti e chiedere conferma
function showMissingModelsDialog(missingModels) {
    // Crea un elemento div per il dialogo
    const dialog = document.createElement('div');
    dialog.id = 'missing-models-dialog';
    dialog.className = 'modal fade show';
    dialog.style.display = 'block';
    dialog.setAttribute('tabindex', '-1');
    dialog.setAttribute('aria-modal', 'true');
    dialog.setAttribute('role', 'dialog');
    
    // Crea la lista dei modelli mancanti
    let modelsList = '';
    missingModels.forEach(model => {
        const modelTypeName = getModelTypeName(model.modelType);
        modelsList += `<li>${modelTypeName} - Timeframe ${model.timeframe}</li>`;
    });
    
    // Aggiungi il contenuto del dialogo
    dialog.innerHTML = `
        <div class="modal-dialog modal-dialog-centered">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">Modelli Mancanti</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body">
                    <p>I seguenti modelli non sono ancora stati addestrati:</p>
                    <ul>
                        ${modelsList}
                    </ul>
                    <p>Vuoi procedere con l'addestramento?</p>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Annulla</button>
                    <button type="button" id="confirm-training-btn" class="btn btn-primary">Procedi</button>
                </div>
            </div>
        </div>
    `;
    
    // Aggiungi il backdrop
    const backdrop = document.createElement('div');
    backdrop.className = 'modal-backdrop fade show';
    
    // Aggiungi gli elementi al body
    document.body.appendChild(dialog);
    document.body.appendChild(backdrop);
    document.body.classList.add('modal-open');
    
    // Aggiungi event listener per chiudere il dialogo
    const closeButtons = dialog.querySelectorAll('[data-bs-dismiss="modal"]');
    closeButtons.forEach(button => {
        button.addEventListener('click', () => {
            dialog.remove();
            backdrop.remove();
            document.body.classList.remove('modal-open');
        });
    });
    
    // Aggiungi event listener per il pulsante di conferma
    const confirmButton = dialog.querySelector('#confirm-training-btn');
    confirmButton.addEventListener('click', () => {
        // Chiudi il dialogo
        dialog.remove();
        backdrop.remove();
        document.body.classList.remove('modal-open');
        
        // Mostra la barra di progresso
        document.getElementById('train-all-progress').classList.remove('d-none');
        document.getElementById('train-all-title').textContent = 'Training dei modelli mancanti...';
        document.getElementById('total-models').textContent = missingModels.length;
        document.getElementById('completed-models').textContent = '0';
        document.getElementById('train-all-progress-bar').style.width = '0%';
        
        // Imposta lo stato globale per il training
        window.trainingQueue = missingModels;
        window.trainingIndex = 0;
        window.completedModels = 0;
        
        // Avvia il training in sequenza
        processNextTraining();
    });
}

// Funzione per processare il prossimo training nella coda
function processNextTraining() {
    // Controlla se abbiamo finito
    if (window.trainingIndex >= window.trainingQueue.length) {
        finishAllTraining();
        return;
    }
    
    // Ottieni il prossimo modello da addestrare
    const model = window.trainingQueue[window.trainingIndex];
    
    // Aggiorna l'interfaccia
    document.getElementById('current-model-training').textContent = 
        `${getModelTypeName(model.modelType)} - ${model.timeframe}`;
    
    // Aggiorna la barra di progresso
    const progressPercent = (window.trainingIndex / window.trainingQueue.length) * 100;
    document.getElementById('train-all-progress-bar').style.width = `${progressPercent}%`;
    
    // Aggiorna il contatore dei modelli completati
    document.getElementById('completed-models').textContent = window.trainingIndex;
    
    // Aggiorna il modello corrente in training
    updateCurrentTrainingModel(model.modelType, model.timeframe);
    
    // Avvia il training
    trainModelWithParams(model)
        .then(() => {
            // Incrementa il contatore e vai al prossimo
            window.trainingIndex++;
            window.completedModels++;
            
            // Attendere un po' per evitare sovraccarichi
            setTimeout(() => {
                processNextTraining();
            }, 2000);
        })
        .catch(error => {
            console.error(`Errore durante l'addestramento di ${model.modelType} ${model.timeframe}:`, error);
            
            // Mostra un messaggio di errore
            appendToLog(`❌ Errore durante l'addestramento di ${getModelTypeName(model.modelType)} - ${model.timeframe}: ${error.message}`);
            
            // Incrementa il contatore e vai al prossimo
            window.trainingIndex++;
            
            // Attendere un po' e continuare
            setTimeout(() => {
                processNextTraining();
            }, 2000);
        });
}

// Funzione per finalizzare il training di tutti i modelli
function finishAllTraining() {
    // Aggiorna la barra di progresso al 100%
    document.getElementById('train-all-progress-bar').style.width = '100%';
    
    // Aggiorna il contatore dei modelli completati
    document.getElementById('completed-models').textContent = window.completedModels;
    
    // Aggiorna il titolo
    document.getElementById('train-all-title').textContent = 'Training completato!';
    
    // Aggiorna il modello corrente in training
    resetCurrentTrainingModel('completato');
    
    // Aggiorna lo stato di tutti i modelli
    checkAllModelsStatus();
    
    // Dopo un po', nascondi la barra di progresso
    setTimeout(() => {
        document.getElementById('train-all-progress').classList.add('d-none');
    }, 5000);
}

// Funzione per ottenere il nome leggibile del tipo di modello
function getModelTypeName(modelType) {
    switch (modelType) {
        case 'lstm': return 'LSTM';
        case 'rf': return 'Random Forest';
        case 'xgb': return 'XGBoost';
        default: return modelType.toUpperCase();
    }
}

// Funzione per addestrare un modello con determinati parametri
async function trainModelWithParams(params) {
    const { modelType, timeframe } = params;
    
    // Aggiorna lo stato del training
    appendToLog(`🔄 Avvio training di ${getModelTypeName(modelType)} - ${timeframe}...`);
    
    // Prepara i dati per la richiesta
    const data = {
        model_type: modelType,
        timeframe: timeframe,
        data_limit_days: 30,
        top_train_crypto: 5
    };
    
    try {
        // Effettua la richiesta di training
        const response = await fetch(`${API_BASE_URL}/api/train-model`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'api-key': apiKey,
                'api-secret': apiSecret
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            throw new Error(`Errore HTTP ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.error) {
            throw new Error(result.error);
        }
        
        // Monitora lo stato del training
        await monitorTrainingStatus(modelType, timeframe);
        
        // Training completato con successo
        appendToLog(`✅ Training di ${getModelTypeName(modelType)} - ${timeframe} completato con successo!`);
        return result;
    } catch (error) {
        appendToLog(`❌ Errore durante il training di ${getModelTypeName(modelType)} - ${timeframe}: ${error.message}`);
        throw error;
    }
}

// Funzione per monitorare lo stato del training
async function monitorTrainingStatus(modelType, timeframe) {
    return new Promise((resolve, reject) => {
        const checkStatus = async () => {
            try {
                const response = await fetch(`${API_BASE_URL}/api/training-status/${modelType}/${timeframe}`, {
                    headers: {
                        'api-key': apiKey,
                        'api-secret': apiSecret
                    }
                });
                
                if (!response.ok) {
                    if (response.status === 404) {
                        // Training non trovato, probabilmente completato o non avviato
                        return resolve();
                    }
                    throw new Error(`Errore HTTP ${response.status}`);
                }
                
                const status = await response.json();
                
                // Aggiorna lo stato del training nell'interfaccia
                updateTrainingProgress(status.progress || 0);
                updateStepDetails(status.current_step || '');
                
                if (status.estimated_time) {
                    updateTrainingTime(status.estimated_time);
                    document.querySelector('.training-time-container').classList.remove('d-none');
                }
                
                // Controlla se il training è completato
                if (status.status === 'completed') {
                    return resolve(status);
                } else if (status.status === 'error') {
                    return reject(new Error(status.error || 'Errore durante il training'));
                }
                
                // Controlla di nuovo dopo un po'
                setTimeout(checkStatus, 2000);
            } catch (error) {
                reject(error);
            }
        };
        
        // Inizia a controllare lo stato
        checkStatus();
    });
}

// Funzione per aggiornare il modello corrente in training
function updateCurrentTrainingModel(modelType, timeframe) {
    const currentTrainingModel = document.getElementById('current-training-model');
    if (currentTrainingModel) {
        currentTrainingModel.textContent = `${getModelTypeName(modelType)} - ${timeframe}`;
    }
    
    const currentTrainingInfo = document.getElementById('current-training-info');
    if (currentTrainingInfo) {
        currentTrainingInfo.classList.remove('alert-light');
        currentTrainingInfo.classList.add('alert-primary');
    }
    
    // Rimuovi i dettagli del training precedente
    document.getElementById('training-metrics').classList.add('d-none');
    document.getElementById('metrics-content').innerHTML = '';
}

// Funzione per resettare il modello corrente in training
function resetCurrentTrainingModel(status) {
    const currentTrainingModel = document.getElementById('current-training-model');
    if (currentTrainingModel) {
        if (status === 'completato') {
            currentTrainingModel.textContent = 'Training completato';
        } else {
            currentTrainingModel.textContent = 'Nessun training in corso';
        }
    }
    
    const currentTrainingInfo = document.getElementById('current-training-info');
    if (currentTrainingInfo) {
        currentTrainingInfo.classList.remove('alert-primary');
        currentTrainingInfo.classList.add('alert-light');
    }
}

// Funzione per aggiornare i dettagli di una fase
function updateStepDetails(currentStep) {
    // Resetta tutti i dettagli
    document.getElementById('step-init-details').textContent = '';
    document.getElementById('step-data-details').textContent = '';
    document.getElementById('step-prep-details').textContent = '';
    document.getElementById('step-train-details').textContent = '';
    document.getElementById('step-complete-details').textContent = '';
    
    // Identifica quale fase è attiva in base alla descrizione
    let activeStep = '';
    let details = '';
    
    // Estrai potenziali dettagli
    if (typeof currentStep === 'string') {
        const parts = currentStep.split(':');
        if (parts.length > 0) {
            activeStep = parts[0].trim().toLowerCase();
            if (parts.length > 1) {
                details = parts[1].trim();
            }
        }
    }
    
    // Aggiorna gli elementi UI in base alla fase attiva
    if (activeStep.includes('inizializz')) {
        document.getElementById('step-init-details').textContent = details;
    } else if (activeStep.includes('recupero') || activeStep.includes('dati')) {
        document.getElementById('step-data-details').textContent = details;
    } else if (activeStep.includes('prep') || activeStep.includes('validazione')) {
        document.getElementById('step-prep-details').textContent = details;
    } else if (activeStep.includes('train')) {
        document.getElementById('step-train-details').textContent = details;
    } else if (activeStep.includes('complet')) {
        document.getElementById('step-complete-details').textContent = details;
    }
}

function displayMetricsComparison(timeframe) {
    const container = document.getElementById('metrics-comparison-data');
    if (!container) return;
    
    const updateButton = document.querySelector('.btn-update-metrics');
    if (updateButton) {
        updateButton.disabled = true;
        updateButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Loading...';
    }

    // Recupera i dati delle metriche per tutti i modelli
    container.innerHTML = '<div class="text-center"><div class="spinner-border text-primary" role="status"></div><p>Caricamento dati...</p></div>';
    
    document.getElementById('metrics-comparison-title').textContent = `Model Comparison - ${timeframe}`;
    
    // Crea la tabella per confrontare le metriche
    const table = document.createElement('table');
    table.className = 'table table-sm table-hover table-bordered';
    
    // Intestazione della tabella
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    
    headerRow.innerHTML = `
        <th>Model</th>
        <th>Accuracy</th>
        <th>Precision</th>
        <th>Recall</th>
        <th>F1-Score</th>
    `;
    
    thead.appendChild(headerRow);
    table.appendChild(thead);
    
    // Corpo della tabella
    const tbody = document.createElement('tbody');
    const models = ['lstm', 'rf', 'xgb'];
    
    // Carica i dati per ciascun modello
    let promises = models.map(model => {
        return fetchModelMetrics(model, timeframe);
    });
    
    Promise.all(promises)
        .then(results => {
            if (results.some(r => r === null)) {
                container.innerHTML = '<div class="alert alert-warning">Alcuni modelli non hanno metriche disponibili per questo timeframe.</div>';
                if (updateButton) {
                    updateButton.disabled = false;
                    updateButton.textContent = 'Aggiorna';
                }
                return;
            }
            
            // Popola la tabella con i dati di ciascun modello
            for (let i = 0; i < models.length; i++) {
                const model = models[i];
                const metrics = results[i];
                if (!metrics) continue;
                
                const row = document.createElement('tr');
                
                // Evidenzia la riga per il modello attuale
                if (currentMetric === 'accuracy' && metrics.accuracy > results.filter(Boolean).map(m => m.accuracy).reduce((a, b) => Math.max(a, b), 0) * 0.95) {
                    row.className = 'table-success';
                } else if (currentMetric === 'precision' && metrics.precision > results.filter(Boolean).map(m => m.precision).reduce((a, b) => Math.max(a, b), 0) * 0.95) {
                    row.className = 'table-success';
                } else if (currentMetric === 'recall' && metrics.recall > results.filter(Boolean).map(m => m.recall).reduce((a, b) => Math.max(a, b), 0) * 0.95) {
                    row.className = 'table-success';
                } else if (currentMetric === 'f1' && metrics.f1 > results.filter(Boolean).map(m => m.f1).reduce((a, b) => Math.max(a, b), 0) * 0.95) {
                    row.className = 'table-success';
                }
                
                // Evidenzia celle individuali
                const accuracyClass = currentMetric === 'accuracy' ? 'table-primary' : '';
                const precisionClass = currentMetric === 'precision' ? 'table-primary' : '';
                const recallClass = currentMetric === 'recall' ? 'table-primary' : '';
                const f1Class = currentMetric === 'f1' ? 'table-primary' : '';
                
                // Calcola valori percentuali
                const accuracy = (metrics.accuracy || 0) * 100;
                const precision = (metrics.precision || 0) * 100;
                const recall = (metrics.recall || 0) * 100;
                const f1 = (metrics.f1 || 0) * 100;
                
                row.innerHTML = `
                    <td><strong>${model.toUpperCase()}</strong></td>
                    <td class="${accuracyClass}">${accuracy.toFixed(2)}%</td>
                    <td class="${precisionClass}">${precision.toFixed(2)}%</td>
                    <td class="${recallClass}">${recall.toFixed(2)}%</td>
                    <td class="${f1Class}">${f1.toFixed(2)}%</td>
                `;
                
                tbody.appendChild(row);
            }
            
            table.appendChild(tbody);
            container.innerHTML = '';
            container.appendChild(table);
            
            if (updateButton) {
                updateButton.disabled = false;
                updateButton.textContent = 'Aggiorna';
            }
            
            // Aggiorna il grafico
            updateMetricsChart(results);
        })
        .catch(error => {
            console.error('Error loading metrics comparison:', error);
            container.innerHTML = `<div class="alert alert-danger">Errore durante il caricamento dei dati: ${error.message}</div>`;
            
            if (updateButton) {
                updateButton.disabled = false;
                updateButton.textContent = 'Aggiorna';
            }
        });
}

// Funzione per caricare le predizioni attuali
async function loadPredictions() {
    const loadingEl = document.getElementById('predictions-loading');
    const errorEl = document.getElementById('predictions-error');
    const errorMsgEl = document.getElementById('predictions-error-message');
    
    try {
        // Mostra il loader solo se è il primo caricamento
        if (!document.querySelector('#predictions-table tbody').children.length) {
            loadingEl.classList.remove('d-none');
        }
        errorEl.classList.add('d-none');
        
        // Recupera le predizioni dal server
        const result = await makeApiRequest('/predictions');
        
        if (!result || !result.predictions) {
            throw new Error('Formato dati predizioni non valido');
        }
        
        // Raggruppa le predizioni per simbolo
        const groupedPredictions = groupPredictionsBySymbol(result.predictions);
        
        // Calcola il consenso ensemble per ogni simbolo
        const consensusPredictions = calculateEnsembleConsensus(groupedPredictions);
        
        // Visualizza le predizioni elaborate
        await displayPredictions(consensusPredictions, result.timeframes, result.default_timeframe);
        
        // Aggiorna il timestamp
        updateLastUpdateTimestamp();
        
    } catch (error) {
        console.error('Errore nel caricamento delle predizioni:', error);
        errorEl.classList.remove('d-none');
        errorMsgEl.textContent = error.message || 'Errore durante il caricamento delle predizioni';
        
        // Non fermare il loop in caso di errore
        if (isPredictionsRunning) {
            const tableBody = document.querySelector('#predictions-table tbody');
            if (tableBody) {
                tableBody.innerHTML = '<tr><td colspan="8" class="text-center text-warning">Riprovo tra 1 minuto...</td></tr>';
            }
        }
    } finally {
        loadingEl.classList.add('d-none');
    }
}

// Funzione per raggruppare le predizioni per simbolo
function groupPredictionsBySymbol(predictions) {
    return predictions.reduce((acc, pred) => {
        if (!acc[pred.symbol]) {
            acc[pred.symbol] = [];
        }
        acc[pred.symbol].push(pred);
        return acc;
    }, {});
}

// Funzione per calcolare il consenso ensemble delle predizioni
function calculateEnsembleConsensus(groupedPredictions) {
    const consensusPredictions = [];
    
    for (const [symbol, predictions] of Object.entries(groupedPredictions)) {
        // Calcola la media pesata delle predizioni per ogni modello
        const modelPredictions = {
            lstm: calculateWeightedModelPrediction(predictions, 'lstm'),
            rf: calculateWeightedModelPrediction(predictions, 'rf'),
            xgb: calculateWeightedModelPrediction(predictions, 'xgb')
        };
        
        // Calcola il valore ensemble complessivo
        const ensembleValue = calculateEnsembleValue(modelPredictions);
        
        // Determina la direzione e il colore in base al consenso
        const { direction, color } = determineConsensusDirection(ensembleValue);
        
        // Calcola il valore RSI medio
        const avgRsi = predictions.reduce((sum, p) => sum + p.rsi_value, 0) / predictions.length;
        
        consensusPredictions.push({
            symbol,
            ensemble_value: ensembleValue,
            direction,
            color,
            rsi_value: avgRsi,
            models: modelPredictions,
            timeframes: predictions.map(p => p.timeframe)
        });
    }
    
    // Ordina le predizioni per forza del segnale
    return consensusPredictions.sort((a, b) => Math.abs(b.ensemble_value - 0.5) - Math.abs(a.ensemble_value - 0.5));
}

// Funzione per calcolare la predizione pesata per un singolo modello
function calculateWeightedModelPrediction(predictions, modelType) {
    const weights = {
        '5m': 0.1,
        '15m': 0.2,
        '30m': 0.3,
        '1h': 0.25,
        '4h': 0.15
    };
    
    const modelPredictions = {};
    let totalWeight = 0;
    let weightedSum = 0;
    
    predictions.forEach(pred => {
        if (pred.models[modelType] && pred.timeframe) {
            const weight = weights[pred.timeframe] || 0.2;
            const value = pred.models[modelType];
            
            modelPredictions[pred.timeframe] = value;
            weightedSum += value * weight;
            totalWeight += weight;
        }
    });
    
    return {
        weighted_average: totalWeight > 0 ? weightedSum / totalWeight : 0.5,
        predictions: modelPredictions
    };
}

// Funzione per calcolare il valore ensemble finale
function calculateEnsembleValue(modelPredictions) {
    const weights = {
        lstm: 0.4,
        rf: 0.3,
        xgb: 0.3
    };
    
    let ensembleValue = 0;
    let totalWeight = 0;
    
    for (const [model, weight] of Object.entries(weights)) {
        if (modelPredictions[model]) {
            ensembleValue += modelPredictions[model].weighted_average * weight;
            totalWeight += weight;
        }
    }
    
    return totalWeight > 0 ? ensembleValue / totalWeight : 0.5;
}

// Funzione per determinare la direzione e il colore del consenso
function determineConsensusDirection(ensembleValue) {
    const strongThreshold = 0.65;
    const weakThreshold = 0.55;
    
    if (ensembleValue >= strongThreshold) {
        return { direction: 'Buy', color: 'green' };
    } else if (ensembleValue <= (1 - strongThreshold)) {
        return { direction: 'Sell', color: 'red' };
    } else if (ensembleValue >= weakThreshold) {
        return { direction: 'Buy', color: 'yellow' };
    } else if (ensembleValue <= (1 - weakThreshold)) {
        return { direction: 'Sell', color: 'yellow' };
    } else {
        return { direction: 'Neutral', color: 'yellow' };
    }
}

// Funzione per visualizzare le predizioni nella tabella
async function displayPredictions(predictions, timeframes, defaultTimeframe) {
    const tableBody = document.querySelector('#predictions-table tbody');
    tableBody.innerHTML = '';
    
    if (predictions.length === 0) {
        const row = document.createElement('tr');
        row.innerHTML = '<td colspan="8" class="text-center">Nessuna predizione disponibile</td>';
        tableBody.appendChild(row);
        return;
    }
    
    // Crea una riga per ogni predizione
    predictions.forEach(prediction => {
        const row = document.createElement('tr');
        
        // Applica lo stile in base alla forza del segnale
        const signalStrength = Math.abs(prediction.ensemble_value - 0.5) * 2; // Normalizza tra 0 e 1
        row.style.opacity = 0.5 + (signalStrength * 0.5); // Opacity tra 0.5 e 1
        
        // Imposta il colore di sfondo
        if (prediction.color === 'green') {
            row.classList.add('table-success');
        } else if (prediction.color === 'red') {
            row.classList.add('table-danger');
        } else if (prediction.color === 'yellow') {
            row.classList.add('table-warning');
        }
        
        // Formatta il valore di confidenza
        const confidencePercent = (Math.abs(prediction.ensemble_value - 0.5) * 200).toFixed(1);
        
        // Crea il badge della direzione
        const directionBadge = createDirectionBadge(prediction.direction, prediction.ensemble_value);
        
        // Formatta il valore RSI con indicatori
        const rsiDisplay = formatRSIDisplay(prediction.rsi_value);
        
        // Crea le barre di progresso per ogni modello
        const modelBars = createModelProgressBars(prediction.models);
        
        // Crea il pulsante per il grafico con tooltip
        const chartButton = createChartButton(prediction.symbol, prediction.timeframes);
        
        // Popola la riga
        row.innerHTML = `
            <td>
                <strong>${prediction.symbol}</strong>
                ${createSignalStrengthIndicator(signalStrength)}
            </td>
            <td>
                <div class="d-flex align-items-center">
                    <div class="progress flex-grow-1 me-2" style="height: 6px;">
                        <div class="progress-bar bg-${prediction.color}" 
                             role="progressbar" 
                             style="width: ${confidencePercent}%" 
                             aria-valuenow="${confidencePercent}" 
                             aria-valuemin="0" 
                             aria-valuemax="100">
                        </div>
                    </div>
                    <span class="small">${confidencePercent}%</span>
                </div>
            </td>
            <td>${directionBadge}</td>
            <td>${rsiDisplay}</td>
            ${modelBars}
            <td>${chartButton}</td>
        `;
        
        tableBody.appendChild(row);
    });
    
    // Inizializza i tooltip
    initializeTooltips();
    
    // Aggiungi event listeners ai pulsanti del grafico
    addChartButtonListeners(defaultTimeframe);
}

// Funzione per creare l'indicatore di forza del segnale
function createSignalStrengthIndicator(strength) {
    const bars = Math.round(strength * 3); // 0-3 barre
    let html = '<div class="signal-strength ms-2">';
    
    for (let i = 0; i < 3; i++) {
        const active = i < bars ? 'active' : '';
        html += `<span class="signal-bar ${active}"></span>`;
    }
    
    return html + '</div>';
}

// Funzione per creare il badge della direzione
function createDirectionBadge(direction, value) {
    const strength = Math.abs(value - 0.5) * 2;
    let badgeClass = 'bg-warning text-dark';
    let icon = 'fa-minus';
    
    if (direction === 'Buy') {
        badgeClass = strength > 0.3 ? 'bg-success' : 'bg-success bg-opacity-50';
        icon = 'fa-arrow-up';
    } else if (direction === 'Sell') {
        badgeClass = strength > 0.3 ? 'bg-danger' : 'bg-danger bg-opacity-50';
        icon = 'fa-arrow-down';
    }
    
    return `
        <span class="badge ${badgeClass}">
            <i class="fas ${icon} me-1"></i>${direction}
        </span>
    `;
}

// Funzione per formattare il display RSI
function formatRSIDisplay(rsiValue) {
    let rsiClass = '';
    let icon = '';
    
    if (rsiValue < 30) {
        rsiClass = 'text-success';
        icon = '<i class="fas fa-arrow-down text-success me-1" title="Sovravenduto"></i>';
    } else if (rsiValue > 70) {
        rsiClass = 'text-danger';
        icon = '<i class="fas fa-arrow-up text-danger me-1" title="Ipercomprato"></i>';
    }
    
    return `
        <div class="d-flex align-items-center">
            ${icon}
            <span class="${rsiClass}">${rsiValue.toFixed(1)}</span>
        </div>
    `;
}

// Funzione per creare le barre di progresso dei modelli
function createModelProgressBars(models) {
    const modelTypes = ['lstm', 'rf', 'xgb'];
    let html = '';
    
    modelTypes.forEach(type => {
        const model = models[type];
        if (!model) return;
        
        const value = model.weighted_average;
        const percent = (value * 100).toFixed(1);
        const predictions = model.predictions;
        
        // Crea il tooltip con i dettagli per timeframe
        const details = Object.entries(predictions)
            .map(([tf, val]) => `${tf}: ${(val * 100).toFixed(1)}%`)
            .join('<br>');
        
        html += `
            <td data-bs-toggle="tooltip" data-bs-html="true" title="${details}">
                <div class="d-flex align-items-center">
                    <div class="progress flex-grow-1 me-2" style="height: 6px;">
                        <div class="progress-bar ${getModelBarColor(value)}" 
                             role="progressbar" 
                             style="width: ${percent}%" 
                             aria-valuenow="${percent}" 
                             aria-valuemin="0" 
                             aria-valuemax="100">
                        </div>
                    </div>
                    <span class="small">${percent}%</span>
                </div>
            </td>
        `;
    });
    
    return html;
}

// Funzione per ottenere il colore della barra del modello
function getModelBarColor(value) {
    if (value > 0.6) return 'bg-success';
    if (value < 0.4) return 'bg-danger';
    return 'bg-warning';
}

// Funzione per creare il pulsante del grafico
function createChartButton(symbol, timeframes) {
    const timeframesList = timeframes.join(', ');
    
    return `
        <button class="btn btn-sm btn-outline-primary view-prediction" 
                data-symbol="${symbol}" 
                data-bs-toggle="tooltip" 
                title="Timeframes disponibili: ${timeframesList}">
            <i class="fas fa-chart-line"></i>
        </button>
    `;
}

// Funzione per aggiungere i listener ai pulsanti del grafico
function addChartButtonListeners(defaultTimeframe) {
    document.querySelectorAll('.view-prediction').forEach(button => {
        button.addEventListener('click', function() {
            const symbol = this.getAttribute('data-symbol');
            
            // Seleziona il simbolo nel selettore del grafico
            const symbolSelect = document.getElementById('chart-symbol-select');
            const timeframeSelect = document.getElementById('chart-timeframe-select');
            
            if (symbolSelect && timeframeSelect) {
                symbolSelect.value = symbol;
                timeframeSelect.value = defaultTimeframe;
                
                // Carica il grafico
                loadChartData(symbol, defaultTimeframe);
                
                // Scorri fino al grafico
                document.querySelector('.card:has(#position-chart)').scrollIntoView({ 
                    behavior: 'smooth',
                    block: 'center'
                });
            }
        });
    });
}

// Funzione per aggiornare il timestamp dell'ultimo aggiornamento
function updateLastUpdateTimestamp() {
    const timestampEl = document.createElement('div');
    timestampEl.className = 'text-muted mt-2 small';
    timestampEl.innerHTML = `
        <i class="fas fa-clock me-1"></i>
        Ultimo aggiornamento: ${new Date().toLocaleTimeString()}
        ${isPredictionsRunning ? '<span class="badge bg-success ms-2">Attivo</span>' : ''}
    `;
    
    const container = document.querySelector('#predictions-table').parentNode;
    const existingTimestamp = container.querySelector('.text-muted');
    
    if (existingTimestamp) {
        existingTimestamp.replaceWith(timestampEl);
    } else {
        container.appendChild(timestampEl);
    }
}

// ... existing code ...

// Funzione per inizializzare il controllo delle predizioni
function initializePredictionsControl() {
    const controlBtn = document.getElementById('predictions-control-btn');
    if (!controlBtn) return;

    controlBtn.addEventListener('click', togglePredictions);
}

// Funzione per gestire lo stato di inizializzazione
function initializationManager() {
    const container = document.getElementById('initialization-status');
    const badge = document.getElementById('init-status-badge');
    const progressBar = document.getElementById('init-progress-bar');
    const steps = ['connect', 'markets', 'data', 'models', 'predictions'];
    let currentStep = -1;
    
    function showContainer() {
        container.classList.remove('d-none');
    }
    
    function hideContainer() {
        container.classList.add('d-none');
    }
    
    function updateProgress() {
        const progress = ((currentStep + 1) / steps.length) * 100;
        progressBar.style.width = `${progress}%`;
        progressBar.setAttribute('aria-valuenow', progress);
    }
    
    function updateBadge(status) {
        badge.className = 'badge';
        switch (status) {
            case 'running':
                badge.classList.add('running');
                badge.textContent = 'In corso...';
                break;
            case 'completed':
                badge.classList.add('completed');
                badge.textContent = 'Completato';
                break;
            case 'error':
                badge.classList.add('error');
                badge.textContent = 'Errore';
                break;
            default:
                badge.textContent = 'In attesa...';
        }
    }
    
    function setStepStatus(stepIndex, status) {
        const stepElement = document.getElementById(`step-${steps[stepIndex]}`);
        if (!stepElement) return;
        
        // Rimuovi tutte le classi di stato
        stepElement.classList.remove('active', 'completed', 'error');
        
        // Aggiorna l'icona di stato
        const statusIcon = stepElement.querySelector('.step-status i');
        statusIcon.className = 'fas';
        
        switch (status) {
            case 'active':
                stepElement.classList.add('active');
                statusIcon.classList.add('fa-circle-notch', 'fa-spin');
                break;
            case 'completed':
                stepElement.classList.add('completed');
                statusIcon.classList.add('fa-check');
                break;
            case 'error':
                stepElement.classList.add('error');
                statusIcon.classList.add('fa-times');
                break;
            default:
                statusIcon.classList.add('fa-circle-notch');
        }
    }
    
    return {
        start() {
            currentStep = -1;
            showContainer();
            updateBadge('running');
            steps.forEach((_, index) => setStepStatus(index, 'waiting'));
            updateProgress();
        },
        
        nextStep() {
            if (currentStep >= 0) {
                setStepStatus(currentStep, 'completed');
            }
            currentStep++;
            if (currentStep < steps.length) {
                setStepStatus(currentStep, 'active');
                updateProgress();
            }
        },
        
        complete() {
            setStepStatus(currentStep, 'completed');
            updateBadge('completed');
            updateProgress();
            // Nascondi il container dopo 2 secondi
            setTimeout(hideContainer, 2000);
        },
        
        error(stepIndex) {
            setStepStatus(stepIndex, 'error');
            updateBadge('error');
        },
        
        reset() {
            hideContainer();
            currentStep = -1;
            updateBadge('waiting');
            steps.forEach((_, index) => setStepStatus(index, 'waiting'));
            updateProgress();
        }
    };
}

// Crea l'istanza del manager
const initManager = initializationManager();

// Modifica la funzione togglePredictions per utilizzare il manager
async function togglePredictions() {
    const controlBtn = document.getElementById('predictions-control-btn');
    
    if (!isPredictionsRunning) {
        try {
            // Valida la selezione
            if (!validateSelection()) {
                return;
            }
            
            // Disabilita il pulsante e mostra il loader
            controlBtn.disabled = true;
            
            // Inizia la sequenza di inizializzazione
            initManager.start();
            
            // Step 1: Connessione
            initManager.nextStep();
            const initResult = await makeApiRequest('/initialize', 'POST', {
                models: getSelectedModels(),
                timeframes: getSelectedTimeframes()
            });
            
            if (!initResult) {
                throw new Error('Errore durante l\'inizializzazione');
            }
            
            // Step 2: Mercati
            initManager.nextStep();
            const startResult = await makeApiRequest('/start', 'POST');
            if (!startResult) {
                throw new Error('Errore durante l\'avvio');
            }
            
            // Step 3: Dati
            initManager.nextStep();
            await loadPredictions();
            
            // Step 4: Modelli
            initManager.nextStep();
            
            // Step 5: Predizioni
            initManager.nextStep();
            
            // Avvia le predizioni
            isPredictionsRunning = true;
            controlBtn.classList.add('running');
            controlBtn.disabled = false;
            controlBtn.innerHTML = '<i class="fas fa-stop me-1"></i> Ferma';
            
            // Disabilita i controlli durante l'esecuzione
            document.querySelectorAll('.btn-check').forEach(checkbox => {
                checkbox.disabled = true;
            });
            
            // Imposta l'intervallo per gli aggiornamenti
            predictionsInterval = setInterval(async () => {
                if (isPredictionsRunning) {
                    await loadPredictions();
                }
            }, 60000); // Aggiorna ogni minuto
            
            // Completa l'inizializzazione
            initManager.complete();
            
        } catch (error) {
            console.error('Errore durante l\'avvio:', error);
            initManager.error(currentStep);
            
            controlBtn.disabled = false;
            controlBtn.innerHTML = '<i class="fas fa-play me-1"></i> Avvia';
            
            showAlert(error.message || 'Errore durante l\'avvio delle predizioni', 'danger');
        }
    } else {
        // Ferma le predizioni
        stopPredictions();
        
        // Riabilita i controlli
        document.querySelectorAll('.btn-check').forEach(checkbox => {
            checkbox.disabled = false;
        });
        
        // Resetta il manager
        initManager.reset();
    }
}

// ... existing code ...

// Funzione per fermare le predizioni
function stopPredictions() {
    isPredictionsRunning = false;
    
    // Pulisci l'intervallo
    if (predictionsInterval) {
        clearInterval(predictionsInterval);
        predictionsInterval = null;
    }
    
    // Resetta il pulsante
    const controlBtn = document.getElementById('predictions-control-btn');
    if (controlBtn) {
        controlBtn.classList.remove('running');
        controlBtn.innerHTML = '<i class="fas fa-play me-1"></i> Avvia';
    }
    
    // Pulisci la tabella delle predizioni
    const tableBody = document.querySelector('#predictions-table tbody');
    if (tableBody) {
        tableBody.innerHTML = '<tr><td colspan="8" class="text-center text-muted">Predizioni fermate</td></tr>';
    }
    
    // Rimuovi il timestamp
    const timestamp = document.querySelector('#predictions-table').parentNode.querySelector('.text-muted');
    if (timestamp) {
        timestamp.remove();
    }
}

// ... existing code ...

// Funzione per ottenere i modelli selezionati
function getSelectedModels() {
    const models = [];
    if (document.getElementById('lstm-model').checked) models.push('lstm');
    if (document.getElementById('rf-model').checked) models.push('rf');
    if (document.getElementById('xgb-model').checked) models.push('xgb');
    return models;
}

// Funzione per ottenere i timeframe selezionati
function getSelectedTimeframes() {
    const timeframes = [];
    if (document.getElementById('tf-5m').checked) timeframes.push('5m');
    if (document.getElementById('tf-15m').checked) timeframes.push('15m');
    if (document.getElementById('tf-30m').checked) timeframes.push('30m');
    if (document.getElementById('tf-1h').checked) timeframes.push('1h');
    if (document.getElementById('tf-4h').checked) timeframes.push('4h');
    return timeframes;
}

// Funzione per validare la selezione
function validateSelection() {
    const models = getSelectedModels();
    const timeframes = getSelectedTimeframes();
    
    if (models.length === 0) {
        showAlert('Seleziona almeno un modello', 'warning');
        return false;
    }
    
    if (timeframes.length === 0) {
        showAlert('Seleziona almeno un timeframe', 'warning');
        return false;
    }
    
    return true;
}

// Funzione per mostrare alert
function showAlert(message, type = 'warning') {
    const alertsContainer = document.createElement('div');
    alertsContainer.className = 'predictions-alerts';
    alertsContainer.style.position = 'absolute';
    alertsContainer.style.top = '10px';
    alertsContainer.style.right = '10px';
    alertsContainer.style.zIndex = '1000';
    
    const alert = document.createElement('div');
    alert.className = `alert alert-${type} alert-dismissible fade show`;
    alert.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    alertsContainer.appendChild(alert);
    
    // Rimuovi eventuali alert esistenti
    const existingAlerts = document.querySelector('.predictions-alerts');
    if (existingAlerts) existingAlerts.remove();
    
    // Aggiungi il nuovo alert
    const cardBody = document.querySelector('#predictions-table').closest('.card-body');
    cardBody.style.position = 'relative';
    cardBody.appendChild(alertsContainer);
    
    // Auto-dismiss dopo 3 secondi
    setTimeout(() => {
        alert.classList.remove('show');
        setTimeout(() => alertsContainer.remove(), 150);
    }, 3000);
}

// Funzione per inizializzare i controlli delle predizioni
function initializePredictionsControl() {
    const controlBtn = document.getElementById('predictions-control-btn');
    if (!controlBtn) return;
    
    // Aggiungi event listener per i checkbox
    document.querySelectorAll('.btn-check').forEach(checkbox => {
        checkbox.addEventListener('change', () => {
            const models = getSelectedModels();
            const timeframes = getSelectedTimeframes();
            
            // Aggiorna lo stato del pulsante
            controlBtn.disabled = models.length === 0 || timeframes.length === 0;
        });
    });

    controlBtn.addEventListener('click', togglePredictions);
}

// Modifica la funzione togglePredictions per includere la validazione
async function togglePredictions() {
    const controlBtn = document.getElementById('predictions-control-btn');
    
    if (!isPredictionsRunning) {
        // Valida la selezione
        if (!validateSelection()) {
            return;
        }
        
        try {
            // Mostra loader sul pulsante
            controlBtn.disabled = true;
            controlBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i> Inizializzazione...';
            
            // Ottieni i modelli e timeframe selezionati
            const selectedModels = getSelectedModels();
            const selectedTimeframes = getSelectedTimeframes();
            
            // Inizializza il bot con le selezioni
            const initResult = await makeApiRequest('/initialize', 'POST', {
                models: selectedModels,
                timeframes: selectedTimeframes
            });
            
            if (!initResult) {
                throw new Error('Errore durante l\'inizializzazione');
            }
            
            // Avvia il bot
            const startResult = await makeApiRequest('/start', 'POST');
            if (!startResult) {
                throw new Error('Errore durante l\'avvio');
            }
            
            // Avvia le predizioni
            isPredictionsRunning = true;
            controlBtn.classList.add('running');
            controlBtn.disabled = false;
            controlBtn.innerHTML = '<i class="fas fa-stop me-1"></i> Ferma';
            
            // Disabilita i controlli durante l'esecuzione
            document.querySelectorAll('.btn-check').forEach(checkbox => {
                checkbox.disabled = true;
            });
            
            // Carica le predizioni immediatamente
            await loadPredictions();
            
            // Imposta l'intervallo per gli aggiornamenti
            predictionsInterval = setInterval(async () => {
                if (isPredictionsRunning) {
                    await loadPredictions();
                }
            }, 60000); // Aggiorna ogni minuto
            
        } catch (error) {
            console.error('Errore durante l\'avvio:', error);
            controlBtn.disabled = false;
            controlBtn.innerHTML = '<i class="fas fa-play me-1"></i> Avvia';
            
            // Mostra errore all'utente
            showAlert(error.message || 'Errore durante l\'avvio delle predizioni', 'danger');
        }
    } else {
        // Ferma le predizioni
        stopPredictions();
        
        // Riabilita i controlli
        document.querySelectorAll('.btn-check').forEach(checkbox => {
            checkbox.disabled = false;
        });
    }
}

// Modifica la funzione loadPredictions per includere i modelli e timeframe selezionati
async function loadPredictions() {
    const loadingEl = document.getElementById('predictions-loading');
    const errorEl = document.getElementById('predictions-error');
    const errorMsgEl = document.getElementById('predictions-error-message');
    
    try {
        // Mostra il loader solo se è il primo caricamento
        if (!document.querySelector('#predictions-table tbody').children.length) {
            loadingEl.classList.remove('d-none');
        }
        errorEl.classList.add('d-none');
        
        // Ottieni i modelli e timeframe selezionati
        const selectedModels = getSelectedModels();
        const selectedTimeframes = getSelectedTimeframes();
        
        // Recupera le predizioni dal server con i parametri selezionati
        const result = await makeApiRequest('/predictions', 'GET', {
            models: selectedModels,
            timeframes: selectedTimeframes
        });
        
        if (!result || !result.predictions) {
            throw new Error('Formato dati predizioni non valido');
        }
        
        // Raggruppa le predizioni per simbolo
        const groupedPredictions = groupPredictionsBySymbol(result.predictions);
        
        // Calcola il consenso ensemble per ogni simbolo
        const consensusPredictions = calculateEnsembleConsensus(groupedPredictions);
        
        // Visualizza le predizioni elaborate
        await displayPredictions(consensusPredictions, selectedTimeframes, selectedTimeframes[0]);
        
        // Aggiorna il timestamp
        updateLastUpdateTimestamp();
        
    } catch (error) {
        console.error('Errore nel caricamento delle predizioni:', error);
        errorEl.classList.remove('d-none');
        errorMsgEl.textContent = error.message || 'Errore durante il caricamento delle predizioni';
        
        // Non fermare il loop in caso di errore
        if (isPredictionsRunning) {
            const tableBody = document.querySelector('#predictions-table tbody');
            if (tableBody) {
                tableBody.innerHTML = '<tr><td colspan="8" class="text-center text-warning">Riprovo tra 1 minuto...</td></tr>';
            }
        }
    } finally {
        loadingEl.classList.add('d-none');
    }
}

// ... existing code ...

// Funzione per gestire la selezione dei modelli e timeframe
function initializeSelectionHandlers() {
    const modelCheckboxes = document.querySelectorAll('.model-select');
    const timeframeCheckboxes = document.querySelectorAll('.timeframe-select');
    const modelsCounter = document.getElementById('models-counter');
    const timeframesCounter = document.getElementById('timeframes-counter');
    const controlBtn = document.getElementById('predictions-control-btn');
    
    function updateCounter(counter, checkboxes) {
        const selected = Array.from(checkboxes).filter(cb => cb.checked).length;
        counter.textContent = `${selected}/3`;
        
        // Aggiorna lo stile del counter in base al numero di selezioni
        counter.className = 'selection-counter';
        if (selected === 0) {
            counter.classList.add('error');
        } else if (selected > 3) {
            counter.classList.add('error');
        } else if (selected === 3) {
            counter.classList.add('warning');
        }
        
        return selected;
    }
    
    function updateControlButton() {
        const selectedModels = Array.from(modelCheckboxes).filter(cb => cb.checked).length;
        const selectedTimeframes = Array.from(timeframeCheckboxes).filter(cb => cb.checked).length;
        
        const isValid = selectedModels > 0 && selectedModels <= 3 && 
                       selectedTimeframes > 0 && selectedTimeframes <= 3;
        
        controlBtn.disabled = !isValid;
    }
    
    function handleSelection(e, checkboxes, counter) {
        const selected = Array.from(checkboxes).filter(cb => cb.checked).length;
        
        // Se si sta cercando di selezionare più di 3 elementi
        if (selected > 3 && e.target.checked) {
            e.preventDefault();
            e.target.checked = false;
            showAlert('Puoi selezionare al massimo 3 elementi', 'warning');
            return;
        }
        
        updateCounter(counter, checkboxes);
        updateControlButton();
    }
    
    // Aggiungi event listeners per i modelli
    modelCheckboxes.forEach(checkbox => {
        checkbox.addEventListener('change', (e) => {
            handleSelection(e, modelCheckboxes, modelsCounter);
        });
    });
    
    // Aggiungi event listeners per i timeframe
    timeframeCheckboxes.forEach(checkbox => {
        checkbox.addEventListener('change', (e) => {
            handleSelection(e, timeframeCheckboxes, timeframesCounter);
        });
    });
    
    // Inizializza i contatori
    updateCounter(modelsCounter, modelCheckboxes);
    updateCounter(timeframesCounter, timeframeCheckboxes);
    updateControlButton();
    
    // Inizializza i tooltip
    const tooltips = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    tooltips.forEach(tooltip => {
        new bootstrap.Tooltip(tooltip, {
            placement: 'top',
            trigger: 'hover'
        });
    });
}

// Modifica la funzione initializePredictionsControl per includere la nuova gestione
function initializePredictionsControl() {
    const controlBtn = document.getElementById('predictions-control-btn');
    if (!controlBtn) return;
    
    // Inizializza i gestori di selezione
    initializeSelectionHandlers();
    
    controlBtn.addEventListener('click', togglePredictions);
}

// ... existing code ...