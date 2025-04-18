// predictionDisplay.js - Gestione della visualizzazione delle predizioni
import { initializeTooltips } from './ui.js';
import { loadChartData } from './charts.js';
import { exportPredictions } from './predictionData.js';
import { getModelValues, formatRSIDisplay } from './predictionModels.js';
import { createDirectionBadge, formatModelValue, createModelDirectionIndicator } from './predictionUI.js';
import { executeTrade } from './predictionTrading.js';

// Funzione per visualizzare le predizioni nella tabella e come carte
export async function displayPredictions(predictions, timeframes, defaultTimeframe) {
    const tableBody = document.querySelector('#predictions-table tbody');
    const tableHeader = document.querySelector('#predictions-table thead tr');
    const cardsContainer = document.getElementById('prediction-cards-container');
    
    if (!tableBody || !cardsContainer) {
        console.error('Elementi della tabella o container delle carte non trovati');
        return;
    }
    
    tableBody.innerHTML = '';
    cardsContainer.innerHTML = '';
    
    // Aggiorna l'intestazione per includere colonne per ogni modello
    if (tableHeader) {
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
    }
    
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

// Funzione per aggiungere un pulsante di esportazione
export function addExportButton(predictions) {
    // Controlla se esiste già un pulsante di esportazione
    if (document.getElementById('export-predictions-btn')) return;
    
    const container = document.querySelector('#predictions-table')?.parentNode;
    if (!container) return;
    
    const button = document.createElement('button');
    button.id = 'export-predictions-btn';
    button.className = 'btn btn-sm btn-outline-primary mt-3';
    button.innerHTML = '<i class="fas fa-file-export me-1"></i> Esporta predizioni';
    button.addEventListener('click', () => exportPredictions(predictions));
    
    container.appendChild(button);
} 