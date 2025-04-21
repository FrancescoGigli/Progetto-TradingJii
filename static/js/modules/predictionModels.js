// predictionModels.js - Gestione dei modelli di previsione e timeframes
import { showNotification } from './ui.js';

// Timeframes supportati dal backend (min: 1, max: 3)
const availableTimeframes = ['5m', '15m', '30m', '1h', '4h'];

// Numero massimo di selezioni consentite
const MAX_SELECTION = 3;

// Array dei modelli disponibili
const availableModels = [
    { id: 'rsi', name: 'RSI Basic', description: 'Modello basato su Relative Strength Index' },
    { id: 'macd', name: 'MACD', description: 'Moving Average Convergence Divergence' },
    { id: 'bb', name: 'Bollinger Bands', description: 'Indicatore di volatilità' },
    { id: 'ensemble', name: 'Ensemble', description: 'Combinazione di più indicatori' },
    { id: 'ml', name: 'Machine Learning', description: 'Modello basato su machine learning' }
];

// Funzione per inizializzare i selettori dei modelli e timeframes
export function initializeModelSelectors() {
    // Inizializza il selettore dei modelli
    initializeModelSelector();
    
    // Inizializza il selettore dei timeframes
    initializeTimeframeSelector();
    
    // Aggiungi i listener per i cambiamenti
    addModelChangeListeners();
}

// Funzione per inizializzare il selettore dei modelli
function initializeModelSelector() {
    const modelSelector = document.getElementById('model-selector');
    if (!modelSelector) return;
    
    // Pulisci il selettore
    modelSelector.innerHTML = '';
    
    // Aggiungi ogni modello disponibile
    availableModels.forEach(model => {
        const option = document.createElement('option');
        option.value = model.id;
        option.textContent = model.name;
        option.dataset.description = model.description;
        modelSelector.appendChild(option);
    });
    
    // Imposta il modello predefinito
    if (modelSelector.options.length > 0) {
        modelSelector.selectedIndex = 0;
        
        // Aggiorna la descrizione del modello
        updateModelDescription(modelSelector.options[0].dataset.description);
    }
}

// Funzione per inizializzare il selettore dei timeframes
function initializeTimeframeSelector() {
    const timeframeSelector = document.getElementById('timeframe-selector');
    if (!timeframeSelector) return;
    
    // Pulisci il selettore
    timeframeSelector.innerHTML = '';
    
    // Aggiungi ogni timeframe disponibile
    availableTimeframes.forEach(timeframe => {
        const option = document.createElement('option');
        option.value = timeframe;
        option.textContent = timeframe;
        timeframeSelector.appendChild(option);
    });
    
    // Imposta il timeframe predefinito a '1h'
    const defaultTimeframeIndex = availableTimeframes.indexOf('1h');
    if (defaultTimeframeIndex !== -1) {
        timeframeSelector.selectedIndex = defaultTimeframeIndex;
    } else if (timeframeSelector.options.length > 0) {
        timeframeSelector.selectedIndex = 0;
    }
}

// Funzione per aggiungere i listener per i cambiamenti nei selettori
function addModelChangeListeners() {
    // Aggiungi listener per il cambio di modello
    const modelSelector = document.getElementById('model-selector');
    if (modelSelector) {
        modelSelector.addEventListener('change', function() {
            const selectedOption = this.options[this.selectedIndex];
            updateModelDescription(selectedOption.dataset.description);
            
            // Aggiorna l'interfaccia in base al modello selezionato
            updateUIForSelectedModel(this.value);
            
            // Aggiorna le opzioni di configurazione del modello
            loadModelConfiguration(this.value);
        });
    }
    
    // Aggiungi listener per il cambio di timeframe
    const timeframeSelector = document.getElementById('timeframe-selector');
    if (timeframeSelector) {
        timeframeSelector.addEventListener('change', function() {
            // Se necessario, aggiorna le configurazioni in base al timeframe
            const modelSelector = document.getElementById('model-selector');
            if (modelSelector) {
                loadModelConfiguration(modelSelector.value, this.value);
            }
            
            // Mostra notifica del cambio di timeframe
            showNotification('info', `Timeframe cambiato a ${this.value}`, true);
        });
    }
}

// Funzione per aggiornare la descrizione del modello
function updateModelDescription(description) {
    const descriptionElement = document.getElementById('model-description');
    if (descriptionElement && description) {
        descriptionElement.textContent = description;
    }
}

// Funzione per aggiornare l'interfaccia in base al modello selezionato
function updateUIForSelectedModel(modelId) {
    // Nascondi tutti i pannelli di configurazione specifici dei modelli
    document.querySelectorAll('.model-config-panel').forEach(panel => {
        panel.classList.add('d-none');
    });
    
    // Mostra il pannello di configurazione specifico per il modello selezionato
    const modelConfigPanel = document.getElementById(`${modelId}-config-panel`);
    if (modelConfigPanel) {
        modelConfigPanel.classList.remove('d-none');
    }
    
    // Aggiorna l'etichetta del modello attivo
    const activeModelLabel = document.getElementById('active-model-label');
    if (activeModelLabel) {
        const selectedModel = availableModels.find(model => model.id === modelId);
        if (selectedModel) {
            activeModelLabel.textContent = selectedModel.name;
        }
    }
    
    // Mostra notifica del cambio di modello
    showNotification('info', `Modello cambiato a ${modelId.toUpperCase()}`, true);
}

// Funzione per caricare la configurazione di un modello
export async function loadModelConfiguration(modelId, timeframe) {
    try {
        // Se il timeframe non è specificato, usa quello attualmente selezionato
        if (!timeframe) {
            const timeframeSelector = document.getElementById('timeframe-selector');
            if (timeframeSelector) {
                timeframe = timeframeSelector.value;
            }
        }
        
        // Richiedi la configurazione dal server
        const response = await fetch(`/api/model_config?model=${modelId}&timeframe=${timeframe}`);
        
        if (!response.ok) {
            console.error('Errore nel caricamento della configurazione del modello:', response.statusText);
            showNotification('error', 'Impossibile caricare la configurazione del modello', true);
            return;
        }
        
        const config = await response.json();
        
        // Aggiorna i campi di configurazione
        updateConfigurationFields(modelId, config);
        
    } catch (error) {
        console.error('Errore nella richiesta della configurazione del modello:', error);
        showNotification('error', 'Errore nella richiesta della configurazione', true);
    }
}

// Funzione per aggiornare i campi di configurazione
function updateConfigurationFields(modelId, config) {
    // Seleziona il pannello di configurazione
    const configPanel = document.getElementById(`${modelId}-config-panel`);
    if (!configPanel) return;
    
    // Per ogni parametro nella configurazione
    Object.entries(config).forEach(([key, value]) => {
        const inputElement = configPanel.querySelector(`[name="${key}"]`);
        if (inputElement) {
            // Imposta il valore in base al tipo di input
            if (inputElement.type === 'checkbox') {
                inputElement.checked = value;
            } else {
                inputElement.value = value;
            }
            
            // Aggiorna eventuali etichette di valore
            const valueLabel = configPanel.querySelector(`#${key}-value`);
            if (valueLabel) {
                valueLabel.textContent = value;
            }
        }
    });
}

// Funzione per salvare la configurazione di un modello
export async function saveModelConfiguration(modelId) {
    try {
        // Seleziona il pannello di configurazione
        const configPanel = document.getElementById(`${modelId}-config-panel`);
        if (!configPanel) {
            showNotification('error', 'Pannello di configurazione non trovato', true);
            return false;
        }
        
        // Raccogli tutti i campi di input
        const formData = new FormData();
        formData.append('model', modelId);
        
        // Aggiungi il timeframe
        const timeframeSelector = document.getElementById('timeframe-selector');
        if (timeframeSelector) {
            formData.append('timeframe', timeframeSelector.value);
        }
        
        // Aggiungi tutti i campi di input
        configPanel.querySelectorAll('input, select').forEach(input => {
            const name = input.name;
            if (!name) return;
            
            let value;
            if (input.type === 'checkbox') {
                value = input.checked;
            } else {
                value = input.value;
            }
            
            formData.append(name, value);
        });
        
        // Invia la configurazione al server
        const response = await fetch('/api/save_model_config', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            console.error('Errore nel salvataggio della configurazione:', response.statusText);
            showNotification('error', 'Impossibile salvare la configurazione', true);
            return false;
        }
        
        const result = await response.json();
        
        // Mostra notifica di successo
        showNotification('success', 'Configurazione salvata con successo', true);
        
        return true;
        
    } catch (error) {
        console.error('Errore nella richiesta di salvataggio:', error);
        showNotification('error', 'Errore nel salvataggio della configurazione', true);
        return false;
    }
}

// Funzione per ottenere i valori correnti del modello
export function getModelValues() {
    // Ottieni il modello e il timeframe selezionati
    const modelSelector = document.getElementById('model-selector');
    const timeframeSelector = document.getElementById('timeframe-selector');
    
    if (!modelSelector || !timeframeSelector) {
        return { model: null, timeframe: null };
    }
    
    return {
        model: modelSelector.value,
        modelName: modelSelector.options[modelSelector.selectedIndex].text,
        timeframe: timeframeSelector.value
    };
}

// Funzione per formattare il valore di un modello per la visualizzazione
export function formatModelValue(value, modelId) {
    if (value === null || value === undefined) return '-';
    
    // Formatta in base al tipo di modello
    switch (modelId) {
        case 'rsi':
            return value.toFixed(2);
        case 'macd':
            return `${value.histogram.toFixed(2)} (${value.signal.toFixed(2)})`;
        case 'bb':
            return `${value.pct.toFixed(2)}%`;
        case 'ensemble':
            return `${(value * 100).toFixed(2)}%`;
        case 'ml':
            return `${(value * 100).toFixed(2)}%`;
        default:
            return value.toString();
    }
}

// Funzione per creare un indicatore di direzione del modello
export function createModelDirectionIndicator(direction) {
    if (!direction) return '';
    
    const directionClass = direction === 'Buy' ? 'text-success' : 'text-danger';
    const iconClass = direction === 'Buy' ? 'fa-arrow-up' : 'fa-arrow-down';
    
    return `<i class="fas ${iconClass} ${directionClass}"></i>`;
}

// Funzione per formattare la visualizzazione dell'RSI
export function formatRSIDisplay(rsiValue) {
    if (rsiValue === null || rsiValue === undefined) return '-';
    
    const rsiFixed = rsiValue.toFixed(2);
    
    // Classe in base al valore RSI
    let className = 'text-warning';
    if (rsiValue >= 70) {
        className = 'text-danger'; // Ipercomprato
    } else if (rsiValue <= 30) {
        className = 'text-success'; // Ipervenduto
    }
    
    // Restituisci il testo formattato
    return `<span class="${className}">${rsiFixed}</span>`;
}

// Funzione per calcolare il consenso dell'ensemble
export function calculateEnsembleConsensus(predictions) {
    // Implementazione: voto a maggioranza sulla direzione
    if (!Array.isArray(predictions) || predictions.length === 0) return null;
    
    const up = predictions.filter(p => p.direction === 'up').length;
    const down = predictions.filter(p => p.direction === 'down').length;
    
    if (up > down) return 'up';
    if (down > up) return 'down';
    return 'neutral';
}

// Esporta anche funzioni per la gestione delle selezioni
export function getSelectedModels() {
    const checked = Array.from(document.querySelectorAll('.model-select:checked'));
    return checked.map(cb => {
        if (cb.id.startsWith('lstm')) return 'lstm';
        if (cb.id.startsWith('rf')) return 'rf';
        if (cb.id.startsWith('xgb')) return 'xgb';
        return cb.value || cb.id;
    });
}

export function getSelectedTimeframes() {
    const checked = Array.from(document.querySelectorAll('.timeframe-select:checked'));
    return checked.map(cb => cb.id.replace('tf-', ''));
}

export function validateSelection() {
    const models = getSelectedModels();
    const timeframes = getSelectedTimeframes();
    
    if (models.length === 0) {
        showNotification('warning', 'Seleziona un modello prima di avviare le previsioni', true);
        return false;
    }
    
    if (timeframes.length === 0) {
        showNotification('warning', 'Seleziona almeno un timeframe', true);
        return false;
    }
    
    // Verifiche max
    if (models.length > MAX_SELECTION) {
        showNotification('warning', `Puoi selezionare al massimo ${MAX_SELECTION} modelli`, true);
        return false;
    }
    if (timeframes.length > MAX_SELECTION) {
        showNotification('warning', `Puoi selezionare al massimo ${MAX_SELECTION} timeframe`, true);
        return false;
    }
    return true;
}

// === FUNZIONE DI COMPATIBILITÀ RICHIESTA DA predictionCore.js ===
// Alcune parti dell'app (predictionCore.js) si aspettano di poter importare
// una funzione initializeSelectionHandlers() da questo modulo. In questa
// implementazione facciamo sì che chiami semplicemente initializeModelSelectors
// (che prepara i selettori di modelli e timeframe) e restituisca true.

export function initializeSelectionHandlers() {
    try {
        // Gestione dei checkbox (modelli e timeframe)
        const modelCheckboxes = document.querySelectorAll('.model-select');
        const timeframeCheckboxes = document.querySelectorAll('.timeframe-select');
        if (modelCheckboxes.length === 0 && timeframeCheckboxes.length === 0) {
            // Se l'UI con i checkbox non esiste, fallback ai selettori a tendina
            initializeModelSelectors();
            return true;
        }

        // Funzione per aggiornare i contatori e lo stato del pulsante Avvia
        const updateUISelections = () => {
            const selectedModels = getSelectedModels();
            const selectedTimeframes = getSelectedTimeframes();

            // Aggiorna contatori
            const modelsCounter = document.getElementById('models-counter');
            const tfsCounter = document.getElementById('timeframes-counter');
            if (modelsCounter) modelsCounter.textContent = `${selectedModels.length}/${MAX_SELECTION}`;
            if (tfsCounter) tfsCounter.textContent = `${selectedTimeframes.length}/${MAX_SELECTION}`;

            // Aggiorna stato pulsante Avvia
            const startBtn = document.getElementById('predictions-control-btn');
            if (startBtn) {
                const enable = selectedModels.length >= 1 && selectedModels.length <= MAX_SELECTION &&
                               selectedTimeframes.length >= 1 && selectedTimeframes.length <= MAX_SELECTION;
                startBtn.disabled = !enable;
            }
        };

        // Aggiungi event listener ai checkbox
        [...modelCheckboxes, ...timeframeCheckboxes].forEach(cb => {
            cb.addEventListener('change', () => {
                // Impedisci selezioni oltre il limite
                if (cb.classList.contains('model-select') && getSelectedModels().length > MAX_SELECTION) {
                    cb.checked = false;
                    showNotification('warning', `Puoi selezionare al massimo ${MAX_SELECTION} modelli`, true);
                }
                if (cb.classList.contains('timeframe-select') && getSelectedTimeframes().length > MAX_SELECTION) {
                    cb.checked = false;
                    showNotification('warning', `Puoi selezionare al massimo ${MAX_SELECTION} timeframe`, true);
                }
                updateUISelections();
            });
        });

        // Esegui una prima inizializzazione
        updateUISelections();
        return true;
    } catch (error) {
        console.error('Errore in initializeSelectionHandlers:', error);
        return false;
    }
}