// predictions.js - Gestisce tutte le funzionalità relative alle predizioni
import { makeApiRequest } from './api.js';
import { appendToLog, showAlert, initializeTooltips } from './ui.js';
import { loadChartData } from './charts.js';

// Variabili globali per lo stato delle predizioni
let isPredictionsRunning = false;
let predictionsInterval = null;
let activityStats = {
    analysis: 0,
    buy: 0,
    sell: 0,
    total: 0
};
let topCryptoCount = 3;

// Variabili globali per i parametri di trading
let leverageValue = 5;
let marginValue = 40;

// Variabili per la visualizzazione dell'attività del bot
let tradingEvents = [];
let performanceChart = null;
let performanceData = {
    labels: [],
    buySignals: [],
    sellSignals: [],
    profit: []
};

// Funzione per inizializzare il controllo delle predizioni
export function initializePredictionsControl() {
    const controlBtn = document.getElementById('predictions-control-btn');
    if (!controlBtn) return;

    // Inizializza i gestori di selezione
    initializeSelectionHandlers();
    
    // Inizializza i controlli dei parametri di trading
    initializeTradeParamsHandlers();
    
    controlBtn.addEventListener('click', togglePredictions);
    
    // Inizializza anche la visualizzazione
    initActivityVisualization();
}

// Funzione per inizializzare i gestori dei parametri di trading
function initializeTradeParamsHandlers() {
    // Gestisce la selezione del numero di cripto
    const topCryptoSelect = document.getElementById('top-crypto-select');
    const topCryptoValue = document.getElementById('top-crypto-value');
    
    if (topCryptoSelect && topCryptoValue) {
        topCryptoSelect.addEventListener('change', (e) => {
            const newValue = parseInt(e.target.value);
            topCryptoCount = newValue;
            topCryptoValue.textContent = newValue;
            appendToLog(`Numero cripto da analizzare impostato a: ${newValue}`);
        });
    }
    
    // Gestisce lo slider della leva finanziaria
    const leverageRange = document.getElementById('leverage-range');
    const leverageValueEl = document.getElementById('leverage-value');
    
    if (leverageRange && leverageValueEl) {
        leverageRange.addEventListener('input', (e) => {
            const newValue = parseInt(e.target.value);
            leverageValue = newValue;
            leverageValueEl.textContent = `${newValue}x`;
            
            // Aggiorna il colore del badge in base al valore di leva
            if (newValue >= 8) {
                leverageValueEl.className = 'badge bg-danger';
            } else if (newValue >= 5) {
                leverageValueEl.className = 'badge bg-warning';
            } else {
                leverageValueEl.className = 'badge bg-success';
            }
            
            updateTotalPositionValue();
        });
    }
    
    // Gestisce lo slider del margine
    const marginRange = document.getElementById('margin-range');
    const marginValueEl = document.getElementById('margin-value');
    
    if (marginRange && marginValueEl) {
        marginRange.addEventListener('input', (e) => {
            const newValue = parseInt(e.target.value);
            marginValue = newValue;
            marginValueEl.textContent = `${newValue} USDT`;
            
            // Aggiorna il colore del badge in base al valore del margine
            if (newValue >= 70) {
                marginValueEl.className = 'badge bg-danger';
            } else if (newValue >= 40) {
                marginValueEl.className = 'badge bg-info';
            } else {
                marginValueEl.className = 'badge bg-success';
            }
            
            updateTotalPositionValue();
        });
    }
    
    // Inizializza il valore totale
    updateTotalPositionValue();
}

// Funzione per aggiornare il valore totale dell'operazione
function updateTotalPositionValue() {
    const leverageEl = document.getElementById('leverage-range');
    const marginEl = document.getElementById('margin-range');
    const totalValueEl = document.getElementById('total-position-value');
    const riskIndicator = document.getElementById('risk-indicator');
    
    if (leverageEl && marginEl && totalValueEl) {
        const leverage = parseInt(leverageEl.value);
        const margin = parseInt(marginEl.value);
        const totalValue = leverage * margin;
        
        // Aggiorna il valore totale
        totalValueEl.textContent = `${totalValue} USDT`;
        
        // Aggiorna il colore del badge in base al rischio
        if (totalValue > 500) {
            totalValueEl.className = 'badge bg-danger px-3 py-2 fs-6';
        } else if (totalValue > 300) {
            totalValueEl.className = 'badge bg-warning px-3 py-2 fs-6';
        } else {
            totalValueEl.className = 'badge bg-primary px-3 py-2 fs-6';
        }
        
        // Aggiorna l'indicatore di rischio
        if (riskIndicator) {
            // Resetta tutti i puntini
            const dots = riskIndicator.querySelectorAll('.risk-dot');
            dots.forEach(dot => {
                dot.className = 'risk-dot';
            });
            
            // Calcolo del livello di rischio da 1 a 5
            let riskLevel;
            if (totalValue <= 100) riskLevel = 1;
            else if (totalValue <= 200) riskLevel = 2;
            else if (totalValue <= 350) riskLevel = 3;
            else if (totalValue <= 500) riskLevel = 4;
            else riskLevel = 5;
            
            // Imposta i colori appropriati per i puntini
            for (let i = 0; i < riskLevel; i++) {
                if (i < 2) {
                    dots[i].classList.add('active'); // Verde per rischio basso
                } else if (i < 4) {
                    dots[i].classList.add('medium'); // Giallo per rischio medio
                } else {
                    dots[i].classList.add('high'); // Rosso per rischio alto
                }
            }
        }
    }
}

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
    initializeTooltips();
}

// Funzione per controllare lo stato delle predizioni
export async function togglePredictions() {
    const controlBtn = document.getElementById('predictions-control-btn');
    
    // Verifica che il pulsante esista
    if (!controlBtn) {
        console.error("Elemento 'predictions-control-btn' non trovato");
        return false;
    }
    
    controlBtn.disabled = true;
    
    // Se non stiamo già eseguendo predizioni, avviale
    if (!isPredictionsRunning) {
        try {
            // Verifica che ci sia almeno un modello e un timeframe selezionato
            if (!validateSelection()) {
                controlBtn.disabled = false;
                return false;
            }
            
            // Cambia immediatamente lo stato in "in esecuzione" per mostrare subito l'animazione
            isPredictionsRunning = true;
            updateRunningUI(true);
            
            // Mostra il loader durante il caricamento iniziale delle predizioni
            const loadingEl = document.getElementById('predictions-loading');
            if (loadingEl) loadingEl.classList.remove('d-none');
            
            // Imposta i valori predefiniti per i parametri di trading
            const topCryptoSelect = document.getElementById('top-crypto-select');
            const topCrypto = topCryptoSelect ? parseInt(topCryptoSelect.value) : 3;
            
            // Ottieni i valori di leva e margine dagli slider
            const leverageRange = document.getElementById('leverage-range');
            const leverage = leverageRange ? parseInt(leverageRange.value) : 5;
            
            const marginRange = document.getElementById('margin-range');
            const margin = marginRange ? parseInt(marginRange.value) : 40;
            
            // Ottieni i modelli e i timeframe selezionati
            const selectedModels = getSelectedModels();
            const selectedTimeframes = getSelectedTimeframes();
            
            // Disabilita i controlli durante l'esecuzione
            document.querySelectorAll('.btn-check').forEach(checkbox => {
                checkbox.disabled = true;
            });
            
            // Disabilita anche i controlli dei parametri di trading se esistono
            if (leverageRange) leverageRange.disabled = true;
            if (marginRange) marginRange.disabled = true;
            
            // Aggiorna i parametri nei log
            appendToLog(`Analisi con: Top ${topCrypto} cripto, Leva ${leverage}x, Margine ${margin} USDT`);
            
            // Inizializza il bot con le selezioni e i parametri di trading
            const initResult = await makeApiRequest('/initialize', 'POST', {
                models: selectedModels,
                timeframes: selectedTimeframes,
                trading_params: {
                    top_analysis_crypto: topCrypto,
                    leverage: leverage,
                    margin_usdt: margin
                }
            });
            
            if (!initResult) {
                throw new Error('Errore durante l\'inizializzazione');
            }
            
            // Avvia il bot
            const startResult = await makeApiRequest('/start', 'POST');
            if (!startResult) {
                throw new Error('Errore durante l\'avvio');
            }
            
            // Carica le predizioni e avvia un intervallo per aggiornarle
            try {
                await loadPredictions();
                
                // Nascondi il loader
                if (loadingEl) loadingEl.classList.add('d-none');
                
                // Imposta un intervallo per ricaricare periodicamente le predizioni
                if (isPredictionsRunning) {
                    // Imposta l'intervallo di aggiornamento delle predizioni ogni 30 secondi
                    predictionsInterval = setInterval(() => {
                        if (isPredictionsRunning) {
                            loadPredictions().catch(err => {
                                console.error("Errore nell'aggiornamento delle predizioni:", err);
                            });
                        } else {
                            clearInterval(predictionsInterval);
                        }
                    }, 30000); // Aggiorna ogni 30 secondi
                    
                    // Aggiorna la UI con lo stato "in esecuzione"
                    updateActivityStatus('Attivo', 'success');
                }
            } catch (predError) {
                console.error("Errore nel caricamento delle predizioni:", predError);
                // Nascondi il loader anche in caso di errore
                if (loadingEl) loadingEl.classList.add('d-none');
                
                // Mostra l'errore ma mantieni il bot in esecuzione
                showAlert("Errore nel caricamento delle predizioni: " + predError.message, 'warning');
            }
            
        } catch (error) {
            console.error('Errore durante l\'avvio:', error);
            
            // In caso di errore, ripristina lo stato del pulsante
            isPredictionsRunning = false;
            
            // Aggiorna l'UI solo se il pulsante esiste
            if (document.getElementById('predictions-control-btn')) {
                updateRunningUI(false);
            }
            
            // Nascondi il loader in caso di errore
            const loadingEl = document.getElementById('predictions-loading');
            if (loadingEl) loadingEl.classList.add('d-none');
            
            // Riabilita i controlli
            document.querySelectorAll('.btn-check').forEach(checkbox => {
                checkbox.disabled = false;
            });
            
            // Riabilita anche i controlli dei parametri di trading
            if (leverageRange) leverageRange.disabled = false;
            if (marginRange) marginRange.disabled = false;
            
            // Mostra errore all'utente
            showAlert(error.message || 'Errore durante l\'avvio delle predizioni', 'danger');
        }
    } else {
        // Ferma le predizioni
        await stopPredictions();
        
        // Riabilita i controlli
        document.querySelectorAll('.btn-check').forEach(checkbox => {
            checkbox.disabled = false;
        });
        
        // Riabilita anche i controlli dei parametri di trading se esistono
        const leverageRange = document.getElementById('leverage-range');
        const marginRange = document.getElementById('margin-range');
        
        if (leverageRange) leverageRange.disabled = false;
        if (marginRange) marginRange.disabled = false;
        
        // Aggiorna lo stato quando le predizioni vengono fermate
        updateActivityStatus('Fermato', 'danger');
    }
    
    return isPredictionsRunning;
}

// Aggiorna la funzione updateRunningUI per un'animazione più evidente
function updateRunningUI(isRunning) {
    const controlBtn = document.getElementById('predictions-control-btn');
    
    // Verifica che il pulsante esista prima di manipolarlo
    if (!controlBtn) {
        console.error("Elemento 'predictions-control-btn' non trovato");
        return;
    }
    
    if (isRunning) {
        // Cambia il testo e lo stile del pulsante con effetto visivo
        controlBtn.classList.add('running');
        controlBtn.innerHTML = '<i class="fas fa-stop me-1"></i> Ferma';
        
        // Animazione del pulsante durante la transizione
        controlBtn.animate([
            { transform: 'scale(0.95)' },
            { transform: 'scale(1.05)' },
            { transform: 'scale(1.0)' }
        ], {
            duration: 300,
            easing: 'ease-out'
        });
        
        // Aggiunge l'animazione di caricamento
        controlBtn.classList.add('position-relative');
        
        // Crea un elemento di progress se non esiste
        if (!document.getElementById('progress-animation')) {
            const progressEl = document.createElement('span');
            progressEl.id = 'progress-animation';
            progressEl.className = 'position-absolute top-0 start-0 bottom-0 bg-white bg-opacity-25';
            progressEl.style.width = '10%';
            progressEl.style.animation = 'button-progress 2s infinite';
            controlBtn.appendChild(progressEl);
            
            // Aggiungi lo stile dell'animazione se non esiste
            if (!document.getElementById('progress-animation-style')) {
                const styleEl = document.createElement('style');
                styleEl.id = 'progress-animation-style';
                styleEl.textContent = `
                    @keyframes button-progress {
                        0% { width: 0%; opacity: 0.2; }
                        50% { width: 100%; opacity: 0.5; }
                        100% { width: 0%; opacity: 0.2; }
                    }
                    
                    @keyframes pulse-icon {
                        0% { transform: scale(1); }
                        50% { transform: scale(1.2); }
                        100% { transform: scale(1); }
                    }
                    
                    #predictions-control-btn.running i {
                        animation: pulse-icon 1s infinite;
                    }
                    
                    .loading-pulse {
                        animation: pulse-bg 1.5s infinite;
                    }
                    
                    @keyframes pulse-bg {
                        0% { background-color: rgba(220, 53, 69, 0.1); }
                        50% { background-color: rgba(220, 53, 69, 0.2); }
                        100% { background-color: rgba(220, 53, 69, 0.1); }
                    }
                `;
                document.head.appendChild(styleEl);
            }
        }
    } else {
        // Ripristina lo stile e il testo del pulsante con effetto visivo
        controlBtn.animate([
            { transform: 'scale(1.05)' },
            { transform: 'scale(0.95)' },
            { transform: 'scale(1.0)' }
        ], {
            duration: 300,
            easing: 'ease-out'
        });
        
        controlBtn.classList.remove('running');
        controlBtn.innerHTML = '<i class="fas fa-play me-1"></i> Avvia';
        
        // Rimuove l'animazione di caricamento
        const progressEl = document.getElementById('progress-animation');
        if (progressEl) {
            progressEl.remove();
        }
        
        controlBtn.classList.remove('position-relative');
    }
    
    // Riabilita il pulsante
    controlBtn.disabled = false;
}

// Funzione per fermare le predizioni
async function stopPredictions() {
    // Cambia lo stato
    isPredictionsRunning = false;
    
    // Invia la richiesta al server
    await makeApiRequest('/stop', 'POST');
    
    // Pulisci l'intervallo se esistente
    if (predictionsInterval) {
        clearInterval(predictionsInterval);
        predictionsInterval = null;
    }
    
    // Aggiorna l'interfaccia
    updateRunningUI(false);
}

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

// Funzione per caricare le predizioni
export async function loadPredictions() {
    try {
        // Aggiungi un effetto pulsante al pulsante Avvia durante il caricamento
        const controlBtn = document.getElementById('predictions-control-btn');
        if (controlBtn && controlBtn.classList.contains('running')) {
            controlBtn.classList.add('loading-pulse');
        }

        // Mostra un indicatore di caricamento più evidente
        const predictionContainer = document.getElementById('prediction-cards-container');
        if (predictionContainer) {
            predictionContainer.innerHTML = `
                <div class="d-flex justify-content-center my-5 py-5">
                    <div class="text-center">
                        <div class="spinner-border text-primary mb-4" style="width: 3rem; height: 3rem;" role="status">
                            <span class="visually-hidden">Caricamento predizioni...</span>
                        </div>
                        <h5 class="mt-3 fw-bold">Caricamento predizioni in corso...</h5>
                        <p class="text-muted">Analizzando i modelli per generare i segnali</p>
                    </div>
                </div>
            `;
        }
        
        // Ottieni i modelli e i timeframe selezionati
        const selectedModels = getSelectedModels();
        const selectedTimeframes = getSelectedTimeframes();
        
        // Costruisci l'URL con i parametri
        const modelsParams = selectedModels.map(m => `models=${m}`).join('&');
        const timeframesParams = selectedTimeframes.map(tf => `timeframes=${tf}`).join('&');
        const url = `/predictions?${modelsParams}&${timeframesParams}`;
        
        // Effettua la richiesta API
        const response = await makeApiRequest(url);
        
        // Rimuovi l'effetto pulsante dal pulsante una volta completato il caricamento
        const controlBtn = document.getElementById('predictions-control-btn');
        if (controlBtn) {
            controlBtn.classList.remove('loading-pulse');
        }
        
        if (response && response.predictions && response.predictions.length > 0) {
            // Raggruppa le predizioni per simbolo
            const groupedPredictions = groupPredictionsBySymbol(response.predictions);
            
            // Calcola il consenso dell'ensemble per ogni simbolo
            const ensembleResults = calculateEnsembleConsensus(groupedPredictions);
            
            // Visualizza le predizioni nell'UI
            await displayPredictions(ensembleResults, selectedTimeframes, selectedTimeframes[0]);
            
            // Mostra un'animazione e una notifica per le nuove predizioni
            showNewPredictionsNotification(ensembleResults.length);
            
            // Aggiorna il timestamp dell'ultimo aggiornamento
            updateLastUpdateTimestamp();
            
            // Inizializza la visualizzazione dell'attività
            initActivityVisualization();
            
            // Inizializza il grafico delle performance se necessario
            if (!window.performanceChart) {
                initPerformanceChart();
            }
            
            // Aggiungi un pulsante per esportare le predizioni
            addExportButton(ensembleResults);
            
            // Avvia la simulazione di eventi se non è già in corso
            if (!window.simulationInterval) {
                window.simulationInterval = setInterval(simulateEvents, 10000);
            }
        } else {
            // Mostra un messaggio se non ci sono predizioni
            if (predictionContainer) {
                predictionContainer.innerHTML = `
                    <div class="alert alert-warning text-center">
                        <i class="fas fa-exclamation-triangle me-2"></i>
                        <strong>Nessuna predizione disponibile.</strong><br>
                        Verifica che i modelli selezionati siano addestrati e che ci siano dati sufficienti.
                    </div>
                `;
            }
        }
    } catch (error) {
        console.error('Errore durante il caricamento delle predizioni:', error);
        
        // Mostra un messaggio di errore
        const predictionContainer = document.getElementById('prediction-cards-container');
        if (predictionContainer) {
            predictionContainer.innerHTML = `
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-circle me-2"></i>
                    <strong>Errore:</strong> Impossibile caricare le predizioni. ${error.message || ''}
                </div>
            `;
        }
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
    const tableHeader = document.querySelector('#predictions-table thead tr');
    const cardsContainer = document.getElementById('prediction-cards-container');
    
    tableBody.innerHTML = '';
    cardsContainer.innerHTML = '';
    
    // Aggiorna l'intestazione per includere colonne per ogni modello
    tableHeader.innerHTML = `
        <th>Simbolo</th>
        <th>Confidenza</th>
        <th>Direzione</th>
        <th>RSI</th>
        <th>LSTM</th>
        <th>Random Forest</th>
        <th>XGBoost</th>
        <th>Azioni</th>
    `;
    
    if (predictions.length === 0) {
        const row = document.createElement('tr');
        row.innerHTML = '<td colspan="8" class="text-center">Nessuna predizione disponibile</td>';
        tableBody.appendChild(row);
        
        cardsContainer.innerHTML = `
            <div class="col-12">
                <div class="alert alert-warning text-center">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    <strong>Nessuna predizione disponibile.</strong><br>
                    Verifica che i modelli selezionati siano addestrati e che ci siano dati sufficienti.
                </div>
            </div>
        `;
        return;
    }
    
    // Salva le predizioni localmente per uso futuro
    window.savedPredictions = predictions;
    
    // Crea una riga per ogni predizione e una carta corrispondente
    predictions.forEach((prediction, index) => {
        // Crea la riga della tabella
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
        
        // Ottiene i valori di ogni modello
        const modelValues = getModelValues(prediction.models);
        
        // Crea il pulsante per il grafico con tooltip
        const chartButton = createChartButton(prediction.symbol, prediction.timeframes);
        
        // Popola la riga della tabella
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
            <td>${formatModelValue(modelValues.lstm)}</td>
            <td>${formatModelValue(modelValues.rf)}</td>
            <td>${formatModelValue(modelValues.xgb)}</td>
            <td>${chartButton}</td>
        `;
        
        tableBody.appendChild(row);
        
        // Crea la carta per la predizione
        const card = document.createElement('div');
        card.className = `col-md-4 col-lg-3 mb-4 prediction-card-${index}`;
        
        // Decidi il colore della carta basato sulla direzione della predizione
        let cardClass = 'border-warning';
        let cardHeaderClass = 'bg-warning text-dark';
        let directionIcon = 'fa-minus';
        
        if (prediction.direction === 'Buy') {
            cardClass = 'border-success';
            cardHeaderClass = signalStrength > 0.3 ? 'bg-success text-white' : 'bg-success bg-opacity-50';
            directionIcon = 'fa-arrow-up';
        } else if (prediction.direction === 'Sell') {
            cardClass = 'border-danger';
            cardHeaderClass = signalStrength > 0.3 ? 'bg-danger text-white' : 'bg-danger bg-opacity-50';
            directionIcon = 'fa-arrow-down';
        }
        
        // Crea gli indicatori di direzione per ogni modello
        const lstmIndicator = createModelDirectionIndicator(modelValues.lstm);
        const rfIndicator = createModelDirectionIndicator(modelValues.rf);
        const xgbIndicator = createModelDirectionIndicator(modelValues.xgb);
        
        // Aggiungi il contenuto della carta
        card.innerHTML = `
            <div class="prediction-card card ${cardClass}">
                <div class="card-header ${cardHeaderClass} d-flex justify-content-between align-items-center">
                    <span><i class="fas ${directionIcon} me-2"></i> ${prediction.symbol}</span>
                    <span class="badge bg-white text-dark">${confidencePercent}%</span>
                </div>
                <div class="card-body">
                    <div class="d-flex justify-content-between align-items-center mb-3">
                        <div class="prediction-direction">
                            <strong>Segnale:</strong> ${prediction.direction}
                        </div>
                        <div class="rsi-value">
                            <strong>RSI:</strong> ${rsiDisplay}
                        </div>
                    </div>
                    
                    <h6 class="mb-2">Modelli:</h6>
                    <div class="model-predictions">
                        <div class="row g-2">
                            <div class="col-4">
                                <div class="model-prediction p-2 border rounded text-center">
                                    <div class="model-name small">LSTM</div>
                                    ${lstmIndicator}
                                    <div class="model-value">${formatModelValue(modelValues.lstm)}</div>
                                </div>
                            </div>
                            <div class="col-4">
                                <div class="model-prediction p-2 border rounded text-center">
                                    <div class="model-name small">RF</div>
                                    ${rfIndicator}
                                    <div class="model-value">${formatModelValue(modelValues.rf)}</div>
                                </div>
                            </div>
                            <div class="col-4">
                                <div class="model-prediction p-2 border rounded text-center">
                                    <div class="model-name small">XGB</div>
                                    ${xgbIndicator}
                                    <div class="model-value">${formatModelValue(modelValues.xgb)}</div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="timeframes mt-3 small text-muted">
                        <strong>Timeframes:</strong> ${prediction.timeframes.join(', ')}
                    </div>
                    
                    <div class="mt-3 d-flex justify-content-between">
                        <button class="btn btn-sm btn-outline-primary view-prediction" 
                                data-symbol="${prediction.symbol}">
                            <i class="fas fa-chart-line me-1"></i> Grafico
                        </button>
                        <div class="btn-group btn-group-sm">
                            <button class="btn btn-sm btn-success buy-action" 
                                    data-symbol="${prediction.symbol}">
                                <i class="fas fa-arrow-up"></i>
                            </button>
                            <button class="btn btn-sm btn-danger sell-action" 
                                    data-symbol="${prediction.symbol}">
                                <i class="fas fa-arrow-down"></i>
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        cardsContainer.appendChild(card);
        
        // Aggiungi l'animazione di entrata con ritardo in base all'indice
        setTimeout(() => {
            const cardElement = document.querySelector(`.prediction-card-${index} .prediction-card`);
            if (cardElement) {
                cardElement.classList.add('highlight-prediction');
                setTimeout(() => {
                    cardElement.classList.remove('highlight-prediction');
                }, 1500);
            }
        }, index * 150);
    });
    
    // Inizializza i tooltip
    initializeTooltips();
    
    // Aggiungi event listeners ai pulsanti del grafico
    addChartButtonListeners(defaultTimeframe);
    
    // Aggiungi pulsante per esportare le predizioni
    addExportButton(predictions);
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

// Funzione per creare il pulsante del grafico
function createChartButton(symbol, timeframes) {
    const timeframesList = timeframes.join(', ');
    
    return `
        <div class="btn-group btn-group-sm" role="group">
            <button class="btn btn-sm btn-outline-primary view-prediction" 
                    data-symbol="${symbol}" 
                    data-bs-toggle="tooltip" 
                    title="Timeframes disponibili: ${timeframesList}">
                <i class="fas fa-chart-line"></i>
            </button>
            <button class="btn btn-sm btn-success buy-action" 
                    data-symbol="${symbol}"
                    data-bs-toggle="tooltip"
                    title="Compra ${symbol}">
                <i class="fas fa-arrow-up"></i>
            </button>
            <button class="btn btn-sm btn-danger sell-action" 
                    data-symbol="${symbol}"
                    data-bs-toggle="tooltip"
                    title="Vendi ${symbol}">
                <i class="fas fa-arrow-down"></i>
            </button>
        </div>
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
    
    // Aggiungi listener per i pulsanti Buy
    document.querySelectorAll('.buy-action').forEach(button => {
        button.addEventListener('click', function() {
            const symbol = this.getAttribute('data-symbol');
            executeTrade(symbol, 'Buy');
        });
    });
    
    // Aggiungi listener per i pulsanti Sell
    document.querySelectorAll('.sell-action').forEach(button => {
        button.addEventListener('click', function() {
            const symbol = this.getAttribute('data-symbol');
            executeTrade(symbol, 'Sell');
        });
    });
}

// Funzione per eseguire un trade
async function executeTrade(symbol, action) {
    try {
        // Aggiungi l'evento prima di eseguire il trade
        addTradingEvent(
            action.toLowerCase(), 
            symbol, 
            `${action} ${symbol} con parametri: Leva=${document.getElementById('leverage-range').value}x, Margine=${document.getElementById('margin-range').value} USDT`
        );
        
        // Mostra una conferma all'utente
        if (!confirm(`Confermi di voler ${action === 'Buy' ? 'acquistare' : 'vendere'} ${symbol}?`)) {
            return;
        }
        
        // Ottieni i parametri di trading
        const leverage = document.getElementById('leverage-range').value;
        const margin = document.getElementById('margin-range').value;
        
        // Effettua la chiamata API per eseguire il trade
        const result = await makeApiRequest('/execute-trade', 'POST', {
            symbol: symbol,
            side: action,
            leverage: parseInt(leverage),
            margin: parseInt(margin)
        });
        
        if (result && result.status === 'success') {
            showAlert(`Trade ${action} eseguito per ${symbol}`, 'success');
            appendToLog(`${action} ${symbol} eseguito: ${result.message}`);
        } else {
            showAlert(`Errore nell'esecuzione del trade: ${result ? result.message : 'Errore sconosciuto'}`, 'danger');
            appendToLog(`Errore nell'esecuzione del trade ${action} per ${symbol}: ${result ? result.message : 'Errore sconosciuto'}`);
        }
    } catch (error) {
        console.error(`Errore nell'esecuzione del trade:`, error);
        showAlert(`Errore nell'esecuzione del trade: ${error.message}`, 'danger');
        appendToLog(`Errore nell'esecuzione del trade ${action} per ${symbol}: ${error.message}`);
    }
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

// Funzione per formattare il valore di un singolo modello
function formatModelValue(value) {
    if (!value) return '<span class="text-muted">N/A</span>';
    
    const percent = (value * 100).toFixed(1);
    let colorClass = 'text-warning';
    
    if (value > 0.6) colorClass = 'text-success';
    else if (value < 0.4) colorClass = 'text-danger';
    
    return `<span class="${colorClass}">${percent}%</span>`;
}

// Funzione per ottenere i valori di ciascun modello
function getModelValues(models) {
    return {
        lstm: models.lstm ? models.lstm.weighted_average : null,
        rf: models.rf ? models.rf.weighted_average : null,
        xgb: models.xgb ? models.xgb.weighted_average : null
    };
}

// Funzione per aggiungere un pulsante di esportazione
function addExportButton(predictions) {
    // Controlla se esiste già un pulsante di esportazione
    if (document.getElementById('export-predictions-btn')) return;
    
    const container = document.querySelector('#predictions-table').parentNode;
    const button = document.createElement('button');
    button.id = 'export-predictions-btn';
    button.className = 'btn btn-sm btn-outline-primary mt-3';
    button.innerHTML = '<i class="fas fa-file-export me-1"></i> Esporta predizioni';
    button.addEventListener('click', () => exportPredictions(predictions));
    
    container.appendChild(button);
}

// Funzione per esportare le predizioni in CSV
function exportPredictions(predictions) {
    // Crea intestazioni CSV
    let csv = 'Simbolo,Confidenza,Direzione,RSI,LSTM,RandomForest,XGBoost\n';
    
    // Aggiungi ogni riga
    predictions.forEach(prediction => {
        const confidencePercent = (Math.abs(prediction.ensemble_value - 0.5) * 200).toFixed(1);
        const models = getModelValues(prediction.models);
        
        csv += `${prediction.symbol},`;
        csv += `${confidencePercent}%,`;
        csv += `${prediction.direction},`;
        csv += `${prediction.rsi_value.toFixed(1)},`;
        csv += `${models.lstm ? (models.lstm * 100).toFixed(1) + '%' : 'N/A'},`;
        csv += `${models.rf ? (models.rf * 100).toFixed(1) + '%' : 'N/A'},`;
        csv += `${models.xgb ? (models.xgb * 100).toFixed(1) + '%' : 'N/A'}\n`;
    });
    
    // Crea un elemento per il download
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `predictions_${new Date().toISOString().slice(0,10)}.csv`);
    link.style.visibility = 'hidden';
    
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// Funzione per inizializzare la visualizzazione dell'attività
function initActivityVisualization() {
    // Inizializza lo stato dell'attività
    updateActivityStatus('In attesa', 'secondary');
    
    // Inizializza il grafico di performance
    initPerformanceChart();
    
    // Aggiungi handler per l'auto-refresh
    document.getElementById('auto-refresh-chart').addEventListener('change', function() {
        if (this.checked && isPredictionsRunning) {
            // Se abilitato e le predizioni sono in corso, aggiorna il grafico
            updatePerformanceChart();
        }
    });
    
    // Aggiungi handler per il cambio di timeframe
    document.getElementById('performance-timeframe').addEventListener('change', function() {
        updatePerformanceChart();
    });
}

// Funzione per inizializzare il grafico delle performance
function initPerformanceChart() {
    const ctx = document.getElementById('performance-chart');
    if (!ctx) return;
    
    // Controlla se esiste già un grafico e distruggilo prima di crearne uno nuovo
    if (window.performanceChart) {
        window.performanceChart.destroy();
    }
    
    try {
        performanceChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Profitto',
                        data: [],
                        borderColor: 'rgba(75, 192, 192, 1)',
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4
                    },
                    {
                        label: 'Segnali Buy',
                        data: [],
                        borderColor: 'rgba(40, 167, 69, 1)',
                        backgroundColor: 'rgba(40, 167, 69, 0.5)',
                        borderWidth: 1,
                        pointRadius: 5,
                        pointStyle: 'circle',
                        showLine: false
                    },
                    {
                        label: 'Segnali Sell',
                        data: [],
                        borderColor: 'rgba(220, 53, 69, 1)',
                        backgroundColor: 'rgba(220, 53, 69, 0.5)',
                        borderWidth: 1,
                        pointRadius: 5,
                        pointStyle: 'triangle',
                        showLine: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false
                },
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Tempo'
                        }
                    },
                    y: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Profitto'
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'top',
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                if (context.parsed.y !== null) {
                                    label += context.parsed.y.toFixed(2) + ' USDT';
                                }
                                return label;
                            }
                        }
                    }
                }
            }
        });
        
        // Salva il riferimento al grafico in window per potervi accedere globalmente
        window.performanceChart = performanceChart;
    } catch (error) {
        console.error('Errore durante l\'inizializzazione del grafico delle performance:', error);
    }
}

// Funzione per aggiungere un evento all'attività
function addTradingEvent(type, symbol, details) {
    // Disabilitato - Non aggiungiamo più eventi simulati
    return;
}

// Funzione per creare un effetto di particelle
function createParticleEffect(event) {
    // Disabilitato - Non creiamo più effetti particellari
    return;
}

// Funzione per creare una singola particella
function createParticle(x, y, color, container) {
    // Disabilitato - Non creiamo più particelle
    return;
}

// Genera un ID univoco per l'evento
function generateEventId() {
    // Disabilitato - Non generiamo più ID per eventi simulati
    return;
}

// Aggiorna le statistiche di attività
function updateActivityStats() {
    // Disabilitato - Non aggiorniamo più le statistiche delle attività
    return;
}

// Aggiorna lo stato dell'attività
function updateActivityStatus(status, type) {
    // Disabilitato - Non aggiorniamo più lo stato delle attività
    return;
}

// Funzione per visualizzare un evento nella timeline con anime.js
function visualizeEvent(event) {
    // Disabilitato - Non visualizziamo più gli eventi
    return;
}

// Funzione per animare il marker di attività corrente
function animateActivityMarker() {
    // Disabilitato - Non animiamo più il marker di attività
    return;
}

// Funzione per pulire eventi vecchi
function cleanupEvents() {
    // Disabilitato - Non facciamo più pulizia degli eventi
    return;
}

// Funzione per aggiungere un evento alla timeline delle attività recenti
function addTimelineEvent(event) {
    // Disabilitato - Non aggiungiamo più eventi alla timeline
    return;
}

// Mostra una notifica toast per eventi di trading
function showTradeNotification(event) {
    // Disabilitato - Non mostriamo più notifiche di trade simulate
    return;
}

// Funzione per aggiornare i dati di performance
function updatePerformanceData(event) {
    // Disabilitato - Non aggiorniamo più i dati di performance simulati
    return;
}

// Funzione per aggiornare il grafico delle performance
function updatePerformanceChart() {
    // Disabilitato - Non aggiorniamo più il grafico delle performance simulate
    return;
}

// Funzione per simulare eventi (solo per demo)
function simulateEvents() {
    // Disabilitato - Non generiamo più eventi simulati
    return;
}

// Funzione per mostrare una notifica quando arrivano nuove predizioni
function showNewPredictionsNotification(count) {
    // Crea un elemento di notifica
    const notificationContainer = document.getElementById('notifications-container') || 
                                  document.createElement('div');
    
    if (!document.getElementById('notifications-container')) {
        notificationContainer.id = 'notifications-container';
        notificationContainer.className = 'position-fixed top-0 end-0 p-3';
        notificationContainer.style.zIndex = '1080';
        document.body.appendChild(notificationContainer);
    }
    
    // Crea la notifica
    const notification = document.createElement('div');
    notification.className = 'toast show prediction-notification';
    notification.style.backgroundColor = '#ffffff';
    notification.style.borderLeft = '4px solid #28a745';
    notification.style.boxShadow = '0 0.5rem 1rem rgba(0, 0, 0, 0.15)';
    
    // Contenuto della notifica
    notification.innerHTML = `
        <div class="toast-header">
            <i class="fas fa-chart-line text-success me-2"></i>
            <strong class="me-auto">Nuove Predizioni</strong>
            <small>Adesso</small>
            <button type="button" class="btn-close" data-bs-dismiss="toast" aria-label="Close"></button>
        </div>
        <div class="toast-body">
            <strong>${count}</strong> nuove predizioni sono state generate con successo.
            <div class="mt-2 pt-2 border-top">
                <button type="button" class="btn btn-sm btn-success close-notification">
                    <i class="fas fa-check me-1"></i> OK
                </button>
            </div>
        </div>
    `;
    
    // Aggiungi l'elemento di notifica al container
    notificationContainer.appendChild(notification);
    
    // Aggiungi l'event listener per chiudere la notifica
    notification.querySelector('.close-notification').addEventListener('click', () => {
        notification.classList.remove('show');
        setTimeout(() => {
            notificationContainer.removeChild(notification);
        }, 500);
    });
    
    // Chiudi automaticamente la notifica dopo 5 secondi
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => {
            if (notification.parentNode) {
                notificationContainer.removeChild(notification);
            }
        }, 500);
    }, 5000);
    
    // Aggiungi un'animazione alle carte delle predizioni
    const predictionCards = document.querySelectorAll('.prediction-card');
    predictionCards.forEach((card, index) => {
        setTimeout(() => {
            // Aggiungi una classe per l'animazione di highlight
            card.classList.add('highlight-prediction');
            
            // Rimuovi la classe dopo l'animazione
            setTimeout(() => {
                card.classList.remove('highlight-prediction');
            }, 2000);
        }, index * 100);
    });
    
    // Riproduci un suono di notifica
    playNotificationSound();
}

// Funzione per riprodurre un suono di notifica
function playNotificationSound() {
    // Disabilitato temporaneamente
    return;
}

// Funzione per creare indicatori di direzione per i modelli
function createModelDirectionIndicator(modelValue) {
    if (!modelValue) return '<div class="direction-indicator">?</div>';
    
    let directionClass, icon;
    
    if (modelValue > 0.6) {
        directionClass = 'direction-up';
        icon = 'fa-arrow-up';
    } else if (modelValue < 0.4) {
        directionClass = 'direction-down';
        icon = 'fa-arrow-down';
    } else {
        directionClass = 'direction-neutral';
        icon = 'fa-minus';
    }
    
    return `<div class="direction-indicator ${directionClass}"><i class="fas ${icon}"></i></div>`;
} 