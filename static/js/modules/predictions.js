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
    if (topCryptoSelect) {
        topCryptoSelect.addEventListener('change', (e) => {
            topCryptoCount = parseInt(e.target.value);
            appendToLog(`Numero cripto da analizzare: ${topCryptoCount}`);
        });
    }
    
    // Gestisce lo slider della leva finanziaria
    const leverageRange = document.getElementById('leverage-range');
    const leverageValueEl = document.getElementById('leverage-value');
    if (leverageRange && leverageValueEl) {
        leverageRange.addEventListener('input', (e) => {
            const newValue = parseInt(e.target.value);
            window.leverageValue = newValue;
            leverageValueEl.textContent = `${newValue}x`;
            
            // Aggiorna il colore del badge in base al valore di leva
            if (newValue >= 8) {
                leverageValueEl.className = 'badge bg-danger ms-auto';
            } else if (newValue >= 5) {
                leverageValueEl.className = 'badge bg-warning ms-auto';
            } else {
                leverageValueEl.className = 'badge bg-success ms-auto';
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
            window.marginValue = newValue;
            marginValueEl.textContent = `${newValue} USDT`;
            
            // Aggiorna il colore del badge in base al valore del margine
            if (newValue >= 70) {
                marginValueEl.className = 'badge bg-danger ms-auto';
            } else if (newValue >= 40) {
                marginValueEl.className = 'badge bg-warning ms-auto';
            } else {
                marginValueEl.className = 'badge bg-info ms-auto';
            }
            
            updateTotalPositionValue();
        });
    }
    
    // Crea l'effetto di sfumatura colorata per gli slider
    const createGradientForSlider = (slider, startColor, endColor) => {
        if (!slider) return;
        
        // Calcola il valore percentuale attuale
        const min = parseInt(slider.min) || 1;
        const max = parseInt(slider.max) || 100;
        const value = parseInt(slider.value) || min;
        const percentage = ((value - min) / (max - min)) * 100;
        
        // Applica il gradiente
        slider.style.background = `linear-gradient(to right, ${startColor} 0%, ${startColor} ${percentage}%, #e9ecef ${percentage}%, #e9ecef 100%)`;
    };
    
    // Applica il gradiente iniziale
    if (leverageRange) {
        createGradientForSlider(leverageRange, '#28a745', '#dc3545');
        leverageRange.addEventListener('input', () => {
            createGradientForSlider(leverageRange, '#28a745', '#dc3545');
        });
    }
    
    if (marginRange) {
        createGradientForSlider(marginRange, '#17a2b8', '#dc3545');
        marginRange.addEventListener('input', () => {
            createGradientForSlider(marginRange, '#17a2b8', '#dc3545');
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
export function togglePredictions() {
    const controlBtn = document.getElementById('predictions-control-btn');
    controlBtn.disabled = true;
    
    // Se non stiamo già eseguendo predizioni, avviale
    if (!isPredictionsRunning) {
        try {
            // Verifica che ci sia almeno un modello e un timeframe selezionato
            if (!validateSelection()) {
                controlBtn.disabled = false;
                return false;
            }
            
            // Ottieni i parametri di trading
            const topCrypto = document.getElementById('top-crypto-select').value;
            const leverage = document.getElementById('leverage-range').value;
            const margin = document.getElementById('margin-range').value;
            
            // Ottieni i modelli e i timeframe selezionati
            const selectedModels = getSelectedModels();
            const selectedTimeframes = getSelectedTimeframes();
            
            // Inizializza il bot con le selezioni e i parametri di trading
            const initResult = makeApiRequest('/initialize', 'POST', {
                models: selectedModels,
                timeframes: selectedTimeframes,
                trading_params: {
                    top_analysis_crypto: 3, // Forziamo sempre a 3
                    leverage: parseInt(leverage),
                    margin_usdt: parseInt(margin)
                }
            });
            
            if (!initResult) {
                throw new Error('Errore durante l\'inizializzazione');
            }
            
            // Aggiorna i parametri nei log
            appendToLog(`Analisi con: Top ${topCrypto} cripto, Leva ${leverage}x, Margine ${margin} USDT`);
            
            // Avvia il bot
            const startResult = makeApiRequest('/start', 'POST');
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
            
            // Disabilita anche i controlli dei parametri di trading
            document.getElementById('top-crypto-select').disabled = true;
            document.getElementById('leverage-range').disabled = true;
            document.getElementById('margin-range').disabled = true;
            
            // Mostra il loader durante il caricamento iniziale delle predizioni
            document.getElementById('predictions-loading').classList.remove('d-none');
            
            // Carica le predizioni una volta sola, senza intervallo di refresh
            loadPredictions().then(() => {
                // Nascondi il loader
                document.getElementById('predictions-loading').classList.add('d-none');
                
                // Una volta completato, reimpostiamo lo stato per permettere un nuovo avvio
                isPredictionsRunning = false;
                
                // Cambia il pulsante in "Avvia" per consentire all'utente di generare nuove predizioni
                controlBtn.classList.remove('running');
                controlBtn.innerHTML = '<i class="fas fa-play me-1"></i> Avvia';
                
                // Riabilita i controlli
                document.querySelectorAll('.btn-check').forEach(checkbox => {
                    checkbox.disabled = false;
                });
                
                // Riabilita anche i controlli dei parametri di trading
                document.getElementById('top-crypto-select').disabled = false;
                document.getElementById('leverage-range').disabled = false;
                document.getElementById('margin-range').disabled = false;
            });
            
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
        
        // Riabilita anche i controlli dei parametri di trading
        document.getElementById('top-crypto-select').disabled = false;
        document.getElementById('leverage-range').disabled = false;
        document.getElementById('margin-range').disabled = false;
    }
    
    // Se le predizioni sono state avviate, inizia a generare eventi simulati
    if (isPredictionsRunning) {
        initActivityVisualization();
        
        // Reset delle statistiche
        activityStats = {
            analysis: 0,
            buy: 0,
            sell: 0,
            total: 0
        };
        updateActivityStats();
        
        // Simulazione di eventi periodici (solo per demo)
        simulateEvents();
    } else {
        // Aggiorna lo stato quando le predizioni vengono fermate
        updateActivityStatus('Fermato', 'danger');
    }
    
    return isPredictionsRunning;
}

// Funzione per fermare le predizioni
function stopPredictions() {
    isPredictionsRunning = false;
    
    // Pulisci l'intervallo se esistente
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
    
    // Rimuovi il timestamp e il pulsante di esportazione
    const timestamp = document.querySelector('#predictions-table').parentNode.querySelector('.text-muted');
    if (timestamp) {
        timestamp.remove();
    }
    
    const exportBtn = document.getElementById('export-predictions-btn');
    if (exportBtn) {
        exportBtn.remove();
    }
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
        // Mostra un indicatore di caricamento
        const predictionContainer = document.getElementById('prediction-cards-container');
        if (predictionContainer) {
            predictionContainer.innerHTML = `
                <div class="d-flex justify-content-center my-4">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">Caricamento predizioni...</span>
                    </div>
                </div>
                <div class="text-center">Caricamento predizioni in corso...</div>
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
    // Aggiorna statistiche
    activityStats[type]++;
    activityStats.total++;
    updateActivityStats();
    
    // Crea l'evento
    const event = {
        id: generateEventId(),
        type: type,
        symbol: symbol,
        details: details,
        timestamp: new Date()
    };
    
    tradingEvents.push(event);
    
    // Aggiorna lo stato dell'attività
    updateActivityStatus('Attivo', 'success');
    
    // Visualizza l'evento nella timeline
    visualizeEvent(event);
    
    // Aggiorna la timeline delle attività recenti
    addTimelineEvent(event);
    
    // Mostra una notifica
    showTradeNotification(event);
    
    // Crea effetto di particelle
    createParticleEffect(event);
    
    // Aggiorna il grafico delle performance se necessario
    if (type === 'buy' || type === 'sell') {
        updatePerformanceData(event);
        if (document.getElementById('auto-refresh-chart').checked) {
            updatePerformanceChart();
        }
    }
}

// Funzione per creare un effetto di particelle
function createParticleEffect(event) {
    const container = document.getElementById('trading-events-container');
    if (!container) return;
    
    // Determina il colore delle particelle in base al tipo di evento
    let particleColor = '#ffc107'; // default per analysis
    if (event.type === 'buy') {
        particleColor = '#28a745';
    } else if (event.type === 'sell') {
        particleColor = '#dc3545';
    }
    
    // Ottieni le coordinate dell'elemento dell'evento
    const eventElement = document.getElementById(event.id);
    if (!eventElement) return;
    
    const rect = eventElement.getBoundingClientRect();
    const containerRect = container.getBoundingClientRect();
    
    // Crea particelle
    const PARTICLE_COUNT = 10;
    for (let i = 0; i < PARTICLE_COUNT; i++) {
        createParticle(
            rect.left - containerRect.left + rect.width / 2,
            rect.top - containerRect.top + rect.height / 2,
            particleColor,
            container
        );
    }
}

// Funzione per creare una singola particella
function createParticle(x, y, color, container) {
    const particle = document.createElement('div');
    particle.style.position = 'absolute';
    particle.style.width = '6px';
    particle.style.height = '6px';
    particle.style.backgroundColor = color;
    particle.style.borderRadius = '50%';
    particle.style.left = x + 'px';
    particle.style.top = y + 'px';
    particle.style.opacity = '0.8';
    particle.style.pointerEvents = 'none';
    
    container.appendChild(particle);
    
    // Animazione della particella
    const angle = Math.random() * Math.PI * 2;
    const speed = Math.random() * 50 + 30;
    const size = Math.random() * 4 + 2;
    
    anime({
        targets: particle,
        translateX: Math.cos(angle) * speed,
        translateY: Math.sin(angle) * speed,
        opacity: 0,
        easing: 'easeOutExpo',
        scale: [1, 0],
        duration: Math.random() * 1000 + 500,
        complete: function() {
            if (container.contains(particle)) {
                container.removeChild(particle);
            }
        }
    });
}

// Genera un ID univoco per l'evento
function generateEventId() {
    return 'event-' + Math.random().toString(36).substr(2, 9);
}

// Aggiorna le statistiche di attività
function updateActivityStats() {
    document.getElementById('analysis-count').textContent = activityStats.analysis;
    document.getElementById('buy-count').textContent = activityStats.buy;
    document.getElementById('sell-count').textContent = activityStats.sell;
    document.getElementById('total-events').textContent = activityStats.total;
}

// Aggiorna lo stato dell'attività
function updateActivityStatus(status, type) {
    const badge = document.getElementById('activity-status-badge');
    const timestamp = document.getElementById('activity-timestamp');
    
    if (badge) {
        badge.textContent = status;
        badge.className = `badge bg-${type}`;
    }
    
    if (timestamp) {
        timestamp.textContent = new Date().toLocaleTimeString();
    }
}

// Funzione per visualizzare un evento nella timeline con anime.js
function visualizeEvent(event) {
    const container = document.getElementById('trading-events-container');
    if (!container) return;
    
    // Crea un elemento per l'evento
    const eventElement = document.createElement('div');
    eventElement.id = event.id;
    eventElement.className = `trading-event ${event.type}`;
    eventElement.style.opacity = '0';
    eventElement.style.transform = 'scale(0)';
    
    // Aggiungi un'icona in base al tipo
    let icon = '';
    switch (event.type) {
        case 'buy':
            icon = '<i class="fas fa-arrow-up"></i>';
            break;
        case 'sell':
            icon = '<i class="fas fa-arrow-down"></i>';
            break;
        case 'analysis':
            icon = '<i class="fas fa-search"></i>';
            break;
    }
    
    eventElement.innerHTML = icon;
    
    // Aggiungi l'evento al container
    container.appendChild(eventElement);
    
    // Posiziona l'evento nella timeline in base al tipo
    const containerHeight = container.offsetHeight;
    
    // Posiziona verticalmente in base al tipo
    const yPosition = event.type === 'analysis' 
        ? containerHeight / 2 + (Math.random() * 20 - 10) 
        : (event.type === 'buy' ? containerHeight / 3 : 2 * containerHeight / 3);
    
    // Calcola una posizione x che si sposti verso destra nel tempo
    const latestEvents = tradingEvents.slice(-10);
    const xOffset = Math.min(70, 100 - latestEvents.length * 2);
    const xPosition = xOffset + Math.random() * 20 + tradingEvents.length * 3;
    
    eventElement.style.top = `${yPosition}px`;
    eventElement.style.left = `${xPosition}%`;
    
    // Aggiungi tooltip con informazioni dettagliate
    const detailsText = event.details || `${event.type.toUpperCase()} - ${event.symbol}`;
    eventElement.setAttribute('data-bs-toggle', 'tooltip');
    eventElement.setAttribute('data-bs-placement', 'top');
    eventElement.setAttribute('title', detailsText);
    
    // Animazione con anime.js
    anime({
        targets: eventElement,
        opacity: 1,
        scale: [0, 1],
        translateY: [20, 0],
        easing: 'spring(1, 80, 10, 0)',
        duration: 800
    });
    
    // Inizializza il tooltip
    initializeTooltips();
    
    // Pulisci eventi vecchi se ci sono troppi
    cleanupEvents();
    
    // Animazione del marker di attività corrente
    animateActivityMarker();
}

// Funzione per animare il marker di attività corrente
function animateActivityMarker() {
    const marker = document.getElementById('current-activity-marker');
    if (!marker) return;
    
    // Calcola una nuova posizione per il marker
    const containerWidth = document.querySelector('.activity-visualization-container').offsetWidth;
    const randomX = 20 + Math.random() * (containerWidth - 40);
    
    // Anima il movimento
    anime({
        targets: marker,
        left: randomX,
        easing: 'spring(1, 80, 10, 0)',
        duration: 1500
    });
}

// Funzione per pulire eventi vecchi
function cleanupEvents() {
    // Limita a massimo 20 eventi visualizzati
    const MAX_EVENTS = 20;
    
    if (tradingEvents.length > MAX_EVENTS) {
        // Rimuovi gli eventi più vecchi
        const eventsToRemove = tradingEvents.length - MAX_EVENTS;
        
        for (let i = 0; i < eventsToRemove; i++) {
            const oldestEvent = tradingEvents.shift();
            const element = document.getElementById(oldestEvent.id);
            
            if (element) {
                // Anima l'uscita
                element.classList.remove('appear');
                setTimeout(() => {
                    element.remove();
                }, 300);
            }
        }
    }
}

// Funzione per aggiungere un evento alla timeline delle attività recenti
function addTimelineEvent(event) {
    const container = document.querySelector('.timeline-container');
    const noActivitiesMsg = document.getElementById('no-activities-msg');
    
    // Rimuovi il messaggio "nessuna attività"
    if (noActivitiesMsg) {
        noActivitiesMsg.remove();
    }
    
    // Crea l'elemento per la timeline
    const timelineItem = document.createElement('div');
    timelineItem.className = 'timeline-item new-event';
    
    // Determina il colore del badge in base al tipo
    let badgeClass = 'bg-secondary';
    let icon = 'fa-info';
    let title = 'Evento';
    
    switch (event.type) {
        case 'buy':
            badgeClass = 'bg-success';
            icon = 'fa-arrow-up';
            title = 'Acquisto';
            break;
        case 'sell':
            badgeClass = 'bg-danger';
            icon = 'fa-arrow-down';
            title = 'Vendita';
            break;
        case 'analysis':
            badgeClass = 'bg-warning';
            icon = 'fa-search';
            title = 'Analisi';
            break;
    }
    
    timelineItem.innerHTML = `
        <div class="timeline-badge ${badgeClass}">
            <i class="fas ${icon}"></i>
        </div>
        <div class="timeline-content">
            <div class="timeline-time">${event.timestamp.toLocaleTimeString()}</div>
            <div class="timeline-title">${title} - ${event.symbol}</div>
            <div class="timeline-details">${event.details || 'Nessun dettaglio disponibile'}</div>
        </div>
    `;
    
    // Inserisci l'evento all'inizio della timeline
    container.insertBefore(timelineItem, container.firstChild);
    
    // Rimuovi eventi vecchi se ce ne sono troppi
    const MAX_TIMELINE_ITEMS = 10;
    const timelineItems = container.querySelectorAll('.timeline-item');
    
    if (timelineItems.length > MAX_TIMELINE_ITEMS) {
        container.removeChild(timelineItems[timelineItems.length - 1]);
    }
}

// Mostra una notifica toast per eventi di trading
function showTradeNotification(event) {
    // Crea l'elemento di notifica
    const notification = document.createElement('div');
    notification.className = `trade-notification ${event.type}`;
    
    // Determina icona e titolo
    let icon = 'fa-info';
    let title = 'Evento';
    
    switch (event.type) {
        case 'buy':
            icon = 'fa-arrow-up';
            title = 'Nuovo Acquisto';
            break;
        case 'sell':
            icon = 'fa-arrow-down';
            title = 'Nuova Vendita';
            break;
        case 'analysis':
            icon = 'fa-search';
            title = 'Nuova Analisi';
            break;
    }
    
    notification.innerHTML = `
        <div class="trade-notification-icon">
            <i class="fas ${icon}"></i>
        </div>
        <div class="trade-notification-content">
            <div class="trade-notification-title">${title} - ${event.symbol}</div>
            <div class="trade-notification-message">${event.details || 'Nessun dettaglio disponibile'}</div>
        </div>
        <button class="trade-notification-close">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    // Aggiungi al corpo del documento
    document.body.appendChild(notification);
    
    // Mostra la notifica
    setTimeout(() => {
        notification.classList.add('show');
    }, 100);
    
    // Configura il pulsante di chiusura
    const closeButton = notification.querySelector('.trade-notification-close');
    closeButton.addEventListener('click', () => {
        notification.classList.remove('show');
        setTimeout(() => {
            notification.remove();
        }, 500);
    });
    
    // Rimuovi automaticamente dopo 5 secondi
    setTimeout(() => {
        if (document.body.contains(notification)) {
            notification.classList.remove('show');
            setTimeout(() => {
                if (document.body.contains(notification)) {
                    notification.remove();
                }
            }, 500);
        }
    }, 5000);
}

// Funzione per aggiornare i dati di performance
function updatePerformanceData(event) {
    // Aggiungi timestamp
    const timestamp = event.timestamp.toLocaleTimeString();
    performanceData.labels.push(timestamp);
    
    // Genera un valore casuale per il profit (da -5 a +15)
    // Questo è solo simulato - in un'implementazione reale dovresti usare i dati effettivi
    const lastProfit = performanceData.profit.length > 0 ? performanceData.profit[performanceData.profit.length - 1] : 0;
    let profitChange = 0;
    
    if (event.type === 'buy') {
        // Per i segnali di acquisto, tendenzialmente positivo
        profitChange = (Math.random() * 8) - 2;
    } else if (event.type === 'sell') {
        // Per i segnali di vendita, mix di positivo e negativo
        profitChange = (Math.random() * 6) - 3;
    }
    
    const newProfit = lastProfit + profitChange;
    performanceData.profit.push(newProfit);
    
    // Aggiungi punti per segnali buy/sell
    const buyPoint = event.type === 'buy' ? newProfit : null;
    const sellPoint = event.type === 'sell' ? newProfit : null;
    
    performanceData.buySignals.push(buyPoint);
    performanceData.sellSignals.push(sellPoint);
    
    // Limita la dimensione dei dati
    const MAX_DATA_POINTS = 50;
    if (performanceData.labels.length > MAX_DATA_POINTS) {
        performanceData.labels = performanceData.labels.slice(-MAX_DATA_POINTS);
        performanceData.profit = performanceData.profit.slice(-MAX_DATA_POINTS);
        performanceData.buySignals = performanceData.buySignals.slice(-MAX_DATA_POINTS);
        performanceData.sellSignals = performanceData.sellSignals.slice(-MAX_DATA_POINTS);
    }
}

// Funzione per aggiornare il grafico delle performance
function updatePerformanceChart() {
    // Controlla il riferimento al grafico da window
    if (!window.performanceChart) {
        // Se il grafico non esiste, inizializzalo
        initPerformanceChart();
        return;
    }
    
    try {
        window.performanceChart.data.labels = performanceData.labels;
        window.performanceChart.data.datasets[0].data = performanceData.profit;
        window.performanceChart.data.datasets[1].data = performanceData.buySignals;
        window.performanceChart.data.datasets[2].data = performanceData.sellSignals;
        
        window.performanceChart.update();
    } catch (error) {
        console.error('Errore nell\'aggiornamento del grafico delle performance:', error);
        // Se riscontriamo un errore, reinizializziamo il grafico
        initPerformanceChart();
    }
}

// Funzione per simulare eventi (solo per demo)
function simulateEvents() {
    if (!isPredictionsRunning) return;
    
    // Ottieni simboli dalle predizioni salvate
    const availableSymbols = window.savedPredictions 
        ? window.savedPredictions.map(p => p.symbol) 
        : ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'DOGE/USDT:USDT'];
    
    // Genera un evento casuale
    const randomType = Math.random() < 0.7 ? 'analysis' : (Math.random() < 0.5 ? 'buy' : 'sell');
    const randomSymbol = availableSymbols[Math.floor(Math.random() * availableSymbols.length)];
    
    let details = '';
    if (randomType === 'analysis') {
        details = `Analisi tecnica per ${randomSymbol} completata`;
    } else if (randomType === 'buy') {
        details = `Segnale di acquisto rilevato per ${randomSymbol} - Confidenza: ${(Math.random() * 30 + 70).toFixed(1)}%`;
    } else {
        details = `Segnale di vendita rilevato per ${randomSymbol} - Confidenza: ${(Math.random() * 30 + 70).toFixed(1)}%`;
    }
    
    // Aggiungi l'evento
    addTradingEvent(randomType, randomSymbol, details);
    
    // Pianifica il prossimo evento
    const delay = Math.random() * 5000 + 2000; // 2-7 secondi
    setTimeout(simulateEvents, delay);
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
    try {
        // Crea un elemento audio e riproducilo
        const audio = new Audio('/static/sounds/notification.mp3');
        audio.volume = 0.5;
        audio.play().catch(e => console.log('Notifica audio non supportata:', e));
    } catch (error) {
        console.log('Impossibile riprodurre il suono di notifica:', error);
    }
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