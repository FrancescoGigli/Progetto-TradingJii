// predictions.js - Gestisce tutte le funzionalità relative alle predizioni
import { makeApiRequest } from './api.js';
import { appendToLog, showAlert, initializeTooltips } from './ui.js';
import { loadChartData } from './charts.js';

// Variabili globali per il controllo delle predizioni
let predictionsInterval = null;
let isPredictionsRunning = false;

// Funzione per inizializzare il controllo delle predizioni
export function initializePredictionsControl() {
    const controlBtn = document.getElementById('predictions-control-btn');
    if (!controlBtn) return;

    // Inizializza i gestori di selezione
    initializeSelectionHandlers();
    
    controlBtn.addEventListener('click', togglePredictions);
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

// Funzione per attivare/disattivare le predizioni
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

// Funzione per caricare le predizioni attuali
export async function loadPredictions() {
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