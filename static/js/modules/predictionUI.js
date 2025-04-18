// predictionUI.js - Gestione dell'interfaccia utente per le previsioni

import { showNotification } from './ui.js';
import * as predictionAPI from './predictionAPI.js';

// Elementi DOM principali
let predictionStatusEl;
let predictionListEl;
let predictionControlsEl;
let predictionChartEl;

// Stato delle previsioni
let activePrevisioni = {};
let lastUpdateTime = null;
let isPredictionsRunning = false;

// Inizializza l'interfaccia utente delle previsioni
export function initPredictionUI() {
    // Ottieni riferimenti agli elementi DOM
    predictionStatusEl = document.getElementById('prediction-status');
    predictionListEl = document.getElementById('prediction-list');
    predictionControlsEl = document.getElementById('prediction-controls');
    predictionChartEl = document.getElementById('prediction-chart');
    
    // Inizializza i controlli e gli eventi
    setupPredictionControls();
    setupRefreshButton();
    setupFilterControls();
    
    // Carica le previsioni e aggiorna l'interfaccia
    refreshPredictions();
}

// Configura i pulsanti di controllo delle previsioni
function setupPredictionControls() {
    // Ottieni i riferimenti ai pulsanti
    const startBtn = document.getElementById('start-predictions');
    const stopBtn = document.getElementById('stop-predictions');
    const configBtn = document.getElementById('configure-predictions');
    
    // Aggiungi event listener per iniziare le previsioni
    if (startBtn) {
        startBtn.addEventListener('click', async () => {
            try {
                // Ottieni i valori correnti dai selettori
                const symbols = Array.from(document.querySelectorAll('#symbol-selector option:checked')).map(option => option.value);
                const model = document.getElementById('model-selector').value;
                const timeframe = document.getElementById('timeframe-selector').value;
                
                // Controlla che siano stati selezionati simboli
                if (!symbols || symbols.length === 0) {
                    showNotification('warning', 'Seleziona almeno un simbolo per avviare le previsioni', true);
                    return;
                }
                
                // Abilita indicatore di caricamento
                startBtn.disabled = true;
                startBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Avvio...';
                
                // Avvia le previsioni tramite API
                const result = await predictionAPI.startPredictions(symbols, model, timeframe);
                
                // Aggiorna l'interfaccia in base al risultato
                if (result.success) {
                    showNotification('success', `Previsioni avviate per ${symbols.length} simboli`, true);
                    refreshPredictions();
                }
                
            } catch (error) {
                console.error('Errore nell\'avvio delle previsioni:', error);
                showNotification('error', `Errore nell'avvio delle previsioni: ${error.message}`, true);
            } finally {
                // Ripristina il pulsante
                startBtn.disabled = false;
                startBtn.innerHTML = '<i class="fas fa-play"></i> Avvia previsioni';
            }
        });
    }
    
    // Aggiungi event listener per fermare le previsioni
    if (stopBtn) {
        stopBtn.addEventListener('click', async () => {
            try {
                // Ottieni i valori correnti dai selettori
                const symbols = Array.from(document.querySelectorAll('#symbol-selector option:checked')).map(option => option.value);
                const model = document.getElementById('model-selector').value;
                const timeframe = document.getElementById('timeframe-selector').value;
                
                // Abilita indicatore di caricamento
                stopBtn.disabled = true;
                stopBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Arresto...';
                
                // Ferma le previsioni tramite API
                const result = await predictionAPI.stopPredictions(symbols, model, timeframe);
                
                // Aggiorna l'interfaccia in base al risultato
                if (result.success) {
                    const message = symbols && symbols.length > 0 
                        ? `Previsioni fermate per ${symbols.length} simboli` 
                        : 'Tutte le previsioni sono state fermate';
                    showNotification('success', message, true);
                    refreshPredictions();
                }
                
            } catch (error) {
                console.error('Errore nell\'arresto delle previsioni:', error);
                showNotification('error', `Errore nell'arresto delle previsioni: ${error.message}`, true);
            } finally {
                // Ripristina il pulsante
                stopBtn.disabled = false;
                stopBtn.innerHTML = '<i class="fas fa-stop"></i> Ferma previsioni';
            }
        });
    }
    
    // Aggiungi event listener per aprire le configurazioni
    if (configBtn) {
        configBtn.addEventListener('click', () => {
            // Apri il pannello di configurazione o il modale
            const configPanel = document.getElementById('prediction-config-panel');
            if (configPanel) {
                configPanel.classList.toggle('hidden');
                
                // Aggiorna il testo del pulsante
                const isHidden = configPanel.classList.contains('hidden');
                configBtn.innerHTML = isHidden 
                    ? '<i class="fas fa-cog"></i> Configura' 
                    : '<i class="fas fa-times"></i> Chiudi';
            }
        });
    }
}

// Configura il pulsante di aggiornamento
function setupRefreshButton() {
    const refreshBtn = document.getElementById('refresh-predictions');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', () => {
            refreshBtn.disabled = true;
            refreshBtn.innerHTML = '<i class="fas fa-sync fa-spin"></i>';
            
            refreshPredictions().finally(() => {
                refreshBtn.disabled = false;
                refreshBtn.innerHTML = '<i class="fas fa-sync"></i>';
            });
        });
    }
}

// Configura i controlli dei filtri
function setupFilterControls() {
    // Filtra per simbolo
    const symbolFilter = document.getElementById('symbol-filter');
    if (symbolFilter) {
        symbolFilter.addEventListener('change', () => {
            updatePredictionList();
        });
    }
    
    // Filtra per modello
    const modelFilter = document.getElementById('model-filter');
    if (modelFilter) {
        modelFilter.addEventListener('change', () => {
            updatePredictionList();
        });
    }
    
    // Filtra per timeframe
    const timeframeFilter = document.getElementById('timeframe-filter');
    if (timeframeFilter) {
        timeframeFilter.addEventListener('change', () => {
            updatePredictionList();
        });
    }
}

// Aggiorna le previsioni
export async function refreshPredictions() {
    try {
        // Aggiorna l'indicatore di stato
        if (predictionStatusEl) {
            predictionStatusEl.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Aggiornamento...';
        }
        
        // Ottieni i valori correnti dai filtri
        const symbolFilter = document.getElementById('symbol-filter')?.value || '';
        const modelFilter = document.getElementById('model-filter')?.value || '';
        const timeframeFilter = document.getElementById('timeframe-filter')?.value || '';
        
        // Recupera le previsioni attive
        const result = await predictionAPI.fetchActivePredictions(
            symbolFilter ? [symbolFilter] : null,
            timeframeFilter ? [timeframeFilter] : null,
            modelFilter || null
        );
        
        // Aggiorna lo stato locale
        activePrevisioni = result.predictions || [];
        lastUpdateTime = new Date();
        
        // Aggiorna gli elementi dell'interfaccia
        updatePredictionStatus();
        updatePredictionList();
        
        return result;
        
    } catch (error) {
        console.error('Errore nell\'aggiornamento delle previsioni:', error);
        
        // Aggiorna l'indicatore di stato in caso di errore
        if (predictionStatusEl) {
            predictionStatusEl.innerHTML = '<i class="fas fa-exclamation-triangle text-red-500"></i> Errore di aggiornamento';
        }
        
        return { predictions: [], error: error.message };
    }
}

// Aggiorna lo stato delle previsioni nell'interfaccia
function updatePredictionStatus() {
    if (!predictionStatusEl) return;
    
    const activeCount = activePrevisioni.length;
    
    // Aggiorna lo stato in base alle previsioni attive
    if (activeCount > 0) {
        predictionStatusEl.innerHTML = `
            <span class="flex items-center">
                <i class="fas fa-check-circle text-green-500 mr-2"></i>
                <span>${activeCount} predizion${activeCount === 1 ? 'e attiva' : 'i attive'}</span>
            </span>
            <span class="text-sm text-gray-500 ml-2">
                Ultimo aggiornamento: ${formatDateTime(lastUpdateTime)}
            </span>
        `;
    } else {
        predictionStatusEl.innerHTML = `
            <span class="flex items-center">
                <i class="fas fa-info-circle text-blue-500 mr-2"></i>
                <span>Nessuna predizione attiva</span>
            </span>
            <span class="text-sm text-gray-500 ml-2">
                Ultimo aggiornamento: ${formatDateTime(lastUpdateTime)}
            </span>
        `;
    }
}

// Aggiorna la lista delle previsioni
function updatePredictionList() {
    if (!predictionListEl) return;
    
    // Filtra le previsioni in base ai selettori
    const symbolFilter = document.getElementById('symbol-filter')?.value || '';
    const modelFilter = document.getElementById('model-filter')?.value || '';
    const timeframeFilter = document.getElementById('timeframe-filter')?.value || '';
    
    const filteredPredictions = activePrevisioni.filter(prediction => {
        return (!symbolFilter || prediction.symbol === symbolFilter) &&
               (!modelFilter || prediction.model === modelFilter) &&
               (!timeframeFilter || prediction.timeframe === timeframeFilter);
    });
    
    // Svuota la lista corrente
    predictionListEl.innerHTML = '';
    
    // Se non ci sono previsioni dopo il filtraggio, mostra un messaggio
    if (filteredPredictions.length === 0) {
        predictionListEl.innerHTML = `
            <div class="p-4 text-center text-gray-500">
                <i class="fas fa-search mr-2"></i>
                Nessuna predizione trovata con i filtri selezionati
            </div>
        `;
        return;
    }
    
    // Aggiungi ogni predizione alla lista
    filteredPredictions.forEach(prediction => {
        const predictionCard = createPredictionCard(prediction);
        predictionListEl.appendChild(predictionCard);
    });
}

// Crea una card per la predizione
function createPredictionCard(prediction) {
    const card = document.createElement('div');
    card.className = 'prediction-card bg-white shadow rounded-lg p-4 mb-4';
    card.dataset.symbol = prediction.symbol;
    card.dataset.model = prediction.model;
    card.dataset.timeframe = prediction.timeframe;
    
    // Formatta il valore della predizione in base al modello
    const formattedValue = formatPredictionValue(prediction);
    
    // Determina la classe di stile in base alla direzione
    const directionClass = prediction.direction === 'up' 
        ? 'text-green-500' 
        : prediction.direction === 'down' 
            ? 'text-red-500' 
            : 'text-gray-500';
    
    // Crea l'icona di direzione
    const directionIcon = prediction.direction === 'up' 
        ? '<i class="fas fa-arrow-up"></i>' 
        : prediction.direction === 'down' 
            ? '<i class="fas fa-arrow-down"></i>' 
            : '<i class="fas fa-minus"></i>';
    
    // Calcola il tempo trascorso dall'ultima predizione
    const lastUpdate = new Date(prediction.timestamp);
    const timeAgo = formatTimeAgo(lastUpdate);
    
    // Popola la card con le informazioni della predizione
    card.innerHTML = `
        <div class="flex justify-between items-start">
            <div class="font-bold text-lg">${prediction.symbol}</div>
            <div class="text-sm px-2 py-1 rounded bg-gray-100">${prediction.timeframe}</div>
        </div>
        
        <div class="mt-2 font-medium">${prediction.model}</div>
        
        <div class="flex items-center mt-3">
            <div class="text-2xl font-bold ${directionClass} flex items-center">
                ${directionIcon}
                <span class="ml-2">${formattedValue}</span>
            </div>
        </div>
        
        <div class="mt-3 text-sm text-gray-500">
            <div>Aggiornato ${timeAgo}</div>
            <div>${formatDateTime(lastUpdate)}</div>
        </div>
        
        <div class="mt-3 flex justify-end gap-2">
            <button class="view-details-btn px-3 py-1 text-sm bg-blue-100 text-blue-700 rounded hover:bg-blue-200">
                <i class="fas fa-chart-line mr-1"></i> Dettagli
            </button>
            <button class="stop-prediction-btn px-3 py-1 text-sm bg-red-100 text-red-700 rounded hover:bg-red-200">
                <i class="fas fa-stop mr-1"></i> Ferma
            </button>
        </div>
    `;
    
    // Aggiungi gli event listener per i pulsanti
    card.querySelector('.view-details-btn').addEventListener('click', () => {
        showPredictionDetails(prediction);
    });
    
    card.querySelector('.stop-prediction-btn').addEventListener('click', async () => {
        try {
            // Ferma la predizione specifica
            const result = await predictionAPI.stopPredictions(
                [prediction.symbol], 
                prediction.model, 
                prediction.timeframe
            );
            
            // Se ha successo, rimuovi la card e aggiorna lo stato
            if (result.success) {
                card.classList.add('fade-out');
                setTimeout(() => {
                    card.remove();
                    
                    // Aggiorna anche lo stato complessivo
                    const index = activePrevisioni.findIndex(p => 
                        p.symbol === prediction.symbol && 
                        p.model === prediction.model && 
                        p.timeframe === prediction.timeframe
                    );
                    
                    if (index !== -1) {
                        activePrevisioni.splice(index, 1);
                        updatePredictionStatus();
                    }
                    
                }, 300);
                
                showNotification('success', `Predizione fermata per ${prediction.symbol}`, true);
            }
        } catch (error) {
            console.error('Errore nell\'arresto della predizione:', error);
            showNotification('error', `Errore nell'arresto della predizione: ${error.message}`, true);
        }
    });
    
    return card;
}

// Mostra i dettagli della predizione
function showPredictionDetails(prediction) {
    // Implementazione del modale o visualizzazione dei dettagli
    const modal = document.getElementById('prediction-details-modal');
    const modalTitle = document.getElementById('prediction-details-title');
    const modalBody = document.getElementById('prediction-details-body');
    
    if (!modal || !modalTitle || !modalBody) {
        console.error('Elementi del modale non trovati');
        return;
    }
    
    // Imposta titolo e contenuto
    modalTitle.textContent = `${prediction.symbol} - ${prediction.timeframe} - ${prediction.model}`;
    
    // Carica i dettagli della predizione
    modalBody.innerHTML = '<div class="text-center py-4"><i class="fas fa-spinner fa-spin"></i> Caricamento dettagli...</div>';
    
    // Ottieni i dettagli della cronologia
    predictionAPI.fetchPredictionHistory(prediction.symbol, prediction.model, prediction.timeframe)
        .then(data => {
            if (data.history && data.history.length > 0) {
                renderPredictionDetails(modalBody, prediction, data.history);
            } else {
                modalBody.innerHTML = '<div class="text-center py-4 text-gray-500">Nessun dato storico disponibile</div>';
            }
        })
        .catch(error => {
            console.error('Errore nel caricamento dei dettagli:', error);
            modalBody.innerHTML = `<div class="text-center py-4 text-red-500">Errore nel caricamento dei dettagli: ${error.message}</div>`;
        });
    
    // Mostra il modale
    modal.classList.remove('hidden');
    
    // Aggiungi event listener per chiudere
    const closeButtons = modal.querySelectorAll('.close-modal');
    closeButtons.forEach(button => {
        button.addEventListener('click', () => {
            modal.classList.add('hidden');
        });
    });
}

// Renderizza i dettagli della predizione
function renderPredictionDetails(container, prediction, history) {
    // Implementazione del rendering dei dettagli
    // Questo è un esempio, da adattare in base ai dati reali
    
    const formattedValue = formatPredictionValue(prediction);
    const lastUpdate = new Date(prediction.timestamp);
    
    // Ottieni le statistiche
    predictionAPI.fetchPredictionStats(prediction.symbol, prediction.model, prediction.timeframe)
        .then(data => {
            // Crea il contenuto
            container.innerHTML = `
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div class="bg-gray-50 p-4 rounded">
                        <h3 class="font-bold mb-3">Predizione attuale</h3>
                        <div class="text-3xl font-bold ${prediction.direction === 'up' ? 'text-green-500' : 'text-red-500'}">
                            ${formattedValue}
                        </div>
                        <div class="mt-2 text-sm text-gray-500">
                            <div>Aggiornato il ${formatDateTime(lastUpdate)}</div>
                        </div>
                        
                        <h3 class="font-bold mt-4 mb-2">Statistiche</h3>
                        <div class="grid grid-cols-2 gap-2 text-sm">
                            <div class="bg-white p-2 rounded shadow-sm">
                                <div class="text-gray-500">Precisione</div>
                                <div class="font-bold">${data.stats.accuracy || 'N/A'}</div>
                            </div>
                            <div class="bg-white p-2 rounded shadow-sm">
                                <div class="text-gray-500">Precisione Up</div>
                                <div class="font-bold">${data.stats.accuracy_up || 'N/A'}</div>
                            </div>
                            <div class="bg-white p-2 rounded shadow-sm">
                                <div class="text-gray-500">Precisione Down</div>
                                <div class="font-bold">${data.stats.accuracy_down || 'N/A'}</div>
                            </div>
                            <div class="bg-white p-2 rounded shadow-sm">
                                <div class="text-gray-500">Totale Previsioni</div>
                                <div class="font-bold">${data.stats.total_predictions || '0'}</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="bg-gray-50 p-4 rounded">
                        <h3 class="font-bold mb-3">Storico previsioni</h3>
                        <div class="max-h-60 overflow-y-auto">
                            <table class="w-full text-sm">
                                <thead class="bg-gray-100">
                                    <tr>
                                        <th class="py-2 px-2 text-left">Data</th>
                                        <th class="py-2 px-2 text-left">Valore</th>
                                        <th class="py-2 px-2 text-left">Esito</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    ${history.map(item => `
                                        <tr class="border-b">
                                            <td class="py-2 px-2">${formatDateTime(new Date(item.timestamp))}</td>
                                            <td class="py-2 px-2 ${item.direction === 'up' ? 'text-green-500' : 'text-red-500'}">
                                                ${formatPredictionValue(item)}
                                            </td>
                                            <td class="py-2 px-2">
                                                ${item.result === 'correct' 
                                                    ? '<span class="text-green-500"><i class="fas fa-check"></i></span>' 
                                                    : item.result === 'incorrect'
                                                        ? '<span class="text-red-500"><i class="fas fa-times"></i></span>'
                                                        : '<span class="text-gray-400">In attesa</span>'}
                                            </td>
                                        </tr>
                                    `).join('')}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
                
                <div class="mt-4 bg-gray-50 p-4 rounded">
                    <h3 class="font-bold mb-3">Grafico</h3>
                    <div id="prediction-detail-chart" class="w-full h-64">
                        <!-- Qui verrà inserito il grafico -->
                    </div>
                </div>
            `;
            
            // Inizializza il grafico (questa è solo una funzione di esempio)
            // initDetailChart('prediction-detail-chart', prediction, history);
            
        })
        .catch(error => {
            console.error('Errore nel caricamento delle statistiche:', error);
            container.innerHTML = `<div class="text-center py-4 text-red-500">Errore nel caricamento delle statistiche: ${error.message}</div>`;
        });
}

// Funzione per formattare il valore della predizione in base al modello
function formatPredictionValue(prediction) {
    switch(prediction.model.toLowerCase()) {
        case 'rsi':
            // Formatta valore RSI
            return `RSI: ${prediction.value.toFixed(2)}`;
            
        case 'macd':
            // Formatta valore MACD
            return `MACD: ${prediction.value.toFixed(4)}`;
            
        case 'bollinger':
            // Formatta valore Bollinger Bands
            return prediction.position === 'upper' 
                ? 'Banda superiore' 
                : prediction.position === 'lower' 
                    ? 'Banda inferiore' 
                    : 'Banda media';
                    
        case 'ensemble':
            // Formatta valore Ensemble
            return `${(prediction.confidence * 100).toFixed(1)}% ${prediction.direction === 'up' ? 'rialzo' : 'ribasso'}`;
            
        case 'ml':
        case 'machine learning':
            // Formatta valore Machine Learning
            return `${(prediction.probability * 100).toFixed(1)}% probabilità`;
            
        default:
            // Valore generico
            return prediction.direction === 'up' 
                ? 'Rialzo previsto' 
                : prediction.direction === 'down' 
                    ? 'Ribasso previsto' 
                    : 'Neutrale';
    }
}

// Formatta data e ora
function formatDateTime(date) {
    if (!date) return 'N/A';
    
    const options = { 
        day: '2-digit', 
        month: '2-digit', 
        year: 'numeric', 
        hour: '2-digit', 
        minute: '2-digit',
        second: '2-digit'
    };
    
    return new Date(date).toLocaleString('it-IT', options);
}

// Formatta il tempo trascorso
function formatTimeAgo(date) {
    if (!date) return 'N/A';
    
    const now = new Date();
    const diffMs = now - new Date(date);
    const diffSec = Math.floor(diffMs / 1000);
    
    if (diffSec < 60) return 'Pochi secondi fa';
    if (diffSec < 3600) return `${Math.floor(diffSec / 60)} minuti fa`;
    if (diffSec < 86400) return `${Math.floor(diffSec / 3600)} ore fa`;
    
    return `${Math.floor(diffSec / 86400)} giorni fa`;
}

// Funzione per creare un badge di direzione
function createDirectionBadge(direction, value) {
    let badgeClass = '';
    let icon = '';
    let label = '';

    if (direction === 'Buy' || direction === 'up') {
        badgeClass = 'badge bg-success';
        icon = '<i class="fas fa-arrow-up"></i>';
        label = 'Rialzo';
    } else if (direction === 'Sell' || direction === 'down') {
        badgeClass = 'badge bg-danger';
        icon = '<i class="fas fa-arrow-down"></i>';
        label = 'Ribasso';
    } else {
        badgeClass = 'badge bg-secondary';
        icon = '<i class="fas fa-minus"></i>';
        label = 'Neutrale';
    }
    
    return `<span class="${badgeClass}">${icon} ${label}</span>`;
}

function createModelDirectionIndicator(modelValue) {
    if (modelValue == null) {
        return '<span class="text-muted"><i class="fas fa-minus"></i></span>';
    }
    if (modelValue > 0) {
        return '<span class="text-success"><i class="fas fa-arrow-up"></i></span>';
    }
    if (modelValue < 0) {
        return '<span class="text-danger"><i class="fas fa-arrow-down"></i></span>';
    }
    return '<span class="text-secondary"><i class="fas fa-minus"></i></span>';
}

function formatModelValue(value, model) {
    if (typeof value !== 'number') return 'N/A';
    if (model && model.toLowerCase() === 'rsi') {
        return `${value.toFixed(2)} RSI`;
    }
    return value.toFixed(2);
}

// Funzione per aggiornare il timestamp dell'ultimo aggiornamento
export function updateLastUpdateTimestamp(success = true) {
    const lastUpdateEl = document.getElementById('last-update-timestamp');
    if (!lastUpdateEl) return;
    
    const now = new Date();
    const formattedTime = formatDateTime(now);
    
    lastUpdateEl.textContent = formattedTime;
    lastUpdateEl.className = success ? 'text-success' : 'text-warning';
    
    // Aggiorna anche il timestamp globale
    lastUpdateTime = now;
    
    // Effetto visivo per mostrare che è stato aggiornato
    lastUpdateEl.classList.add('update-flash');
    setTimeout(() => {
        lastUpdateEl.classList.remove('update-flash');
    }, 1000);
}

// Funzione per mostrare una notifica per le nuove previsioni
export function showNewPredictionsNotification(count) {
    if (!count || count <= 0) return;
    
    // Crea una notifica toast per le nuove previsioni
    showNotification(
        'success', 
        `${count} nuov${count === 1 ? 'a predizione caricata' : 'e predizioni caricate'}`, 
        true,
        5000 // Durata più lunga per questa notifica
    );
    
    // Animazione per indicare nuove predizioni
    const predictionsContainer = document.getElementById('prediction-cards-container');
    if (predictionsContainer) {
        predictionsContainer.classList.add('new-predictions-animation');
        setTimeout(() => {
            predictionsContainer.classList.remove('new-predictions-animation');
        }, 2000);
    }
}

// Funzione per inizializzare la visualizzazione dell'attività
export function initActivityVisualization() {
    // Inizializza gli indicatori di attività
    const activityIndicator = document.getElementById('activity-indicator');
    if (activityIndicator) {
        activityIndicator.innerHTML = `
            <div class="activity-status">
                <span class="status-dot inactive"></span>
                <span class="status-text">Inattivo</span>
            </div>
        `;
    }
    
    // Inizializza il contatore del tempo di esecuzione se esiste
    const runtimeCounter = document.getElementById('prediction-runtime');
    if (runtimeCounter) {
        runtimeCounter.textContent = '00:00:00';
        
        // Avvia il contatore se le previsioni sono attive
        if (isPredictionsRunning) {
            startRuntimeCounter();
        }
    }
}

// Funzione per aggiornare l'interfaccia quando lo stato di esecuzione cambia
export function updateRunningUI(isRunning) {
    // Aggiorna lo stato di esecuzione
    isPredictionsRunning = isRunning;
    
    // Aggiorna il pulsante di controllo
    const controlBtn = document.getElementById('predictions-control-btn');
    if (controlBtn) {
        if (isRunning) {
            controlBtn.innerHTML = '<i class="fas fa-stop me-2"></i>Ferma';
            controlBtn.classList.remove('btn-primary');
            controlBtn.classList.add('btn-danger', 'running');
            controlBtn.setAttribute('title', 'Ferma le previsioni');
            
            // Aggiungi l'animazione di pulsazione
            controlBtn.style.animation = 'pulse 2s infinite';
        } else {
            controlBtn.innerHTML = '<i class="fas fa-play me-2"></i>Avvia';
            controlBtn.classList.remove('btn-danger', 'running');
            controlBtn.classList.add('btn-primary');
            controlBtn.setAttribute('title', 'Avvia le previsioni');
            
            // Rimuovi l'animazione
            controlBtn.style.animation = '';
        }
    }
    
    // Aggiorna lo stato dell'indicatore di attività
    updateActivityStatus(isRunning ? 'Attivo' : 'Inattivo', isRunning ? 'success' : 'danger');
    
    // Gestisci il contatore del runtime
    const runtimeCounter = document.getElementById('prediction-runtime');
    if (runtimeCounter) {
        if (isRunning) {
            startRuntimeCounter();
        } else {
            // Ripristina il contatore
            runtimeCounter.textContent = '00:00:00';
        }
    }
}

// Funzione per aggiornare lo stato di attività
export function updateActivityStatus(status, type = 'info') {
    const activityIndicator = document.getElementById('activity-indicator');
    if (!activityIndicator) return;
    
    // Mappatura dei tipi ai colori e classi
    const typeMap = {
        'success': { class: 'active', color: 'text-success' },
        'danger': { class: 'inactive', color: 'text-danger' },
        'warning': { class: 'warning', color: 'text-warning' },
        'info': { class: '', color: 'text-info' }
    };
    
    const typeInfo = typeMap[type] || typeMap.info;
    
    activityIndicator.innerHTML = `
        <div class="activity-status">
            <span class="status-dot ${typeInfo.class}"></span>
            <span class="status-text ${typeInfo.color}">${status}</span>
        </div>
    `;
}

// Contatore globale per il runtime
let runtimeInterval = null;
let runtimeStart = null;

// Funzione per avviare il contatore del runtime
function startRuntimeCounter() {
    // Pulisci qualsiasi intervallo esistente
    if (runtimeInterval) {
        clearInterval(runtimeInterval);
    }
    
    // Imposta l'ora di inizio
    runtimeStart = new Date();
    
    // Aggiorna il contatore ogni secondo
    const runtimeCounter = document.getElementById('prediction-runtime');
    if (runtimeCounter) {
        runtimeInterval = setInterval(() => {
            const now = new Date();
            const diff = now - runtimeStart;
            
            // Converti in ore, minuti, secondi
            const hours = Math.floor(diff / 3600000);
            const minutes = Math.floor((diff % 3600000) / 60000);
            const seconds = Math.floor((diff % 60000) / 1000);
            
            // Formatta il tempo
            const formattedTime = 
                `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
            
            runtimeCounter.textContent = formattedTime;
        }, 1000);
    }
}

export {
    createDirectionBadge,
    createModelDirectionIndicator,
    formatModelValue
};