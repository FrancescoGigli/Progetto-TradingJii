// predictionCore.js - Funzionalità di base delle predizioni
import { makeApiRequest } from './api.js';
import { appendToLog, showAlert } from './ui.js';
import { loadPredictions } from './predictionData.js';

// Funzioni da moduli esterni che potrebbero non essere ancora definite
let updateRunningUI, updateActivityStatus, initActivityVisualization;

// Variabili globali per lo stato delle predizioni
export let isPredictionsRunning = false;
export let predictionsInterval = null;
export let statusCheckInterval = null;
export let autoStartDisabled = false;

// Imposta il flag autoStartDisabled in base al valore esterno
export function setAutoStartDisabled(value) {
    autoStartDisabled = value;
}

// Funzione per inizializzare il controllo delle predizioni
export function initializePredictionsControl() {
    const controlBtn = document.getElementById('predictions-control-btn');
    if (!controlBtn) return;

    // Inizializza i gestori di selezione
    initializeSelectionHandlers();
    
    // Inizializza i controlli dei parametri di trading
    initializeTradeParamsHandlers();
    
    controlBtn.addEventListener('click', togglePredictions);
    
    // Carica le funzioni UI e inizializza la visualizzazione
    loadUIFunctions();
    
    // Controlla lo stato subito all'avvio
    checkBotStatus();
    
    // Avvia automaticamente le predizioni se non disabilitato
    if (!autoStartDisabled) {
        // Avvia le predizioni con un leggero ritardo per dare tempo al caricamento completo
        setTimeout(() => {
            if (!isPredictionsRunning) {
                appendToLog('Avvio automatico delle predizioni...');
                togglePredictions();
            }
        }, 2000);
    }
}

// Funzione per controllare lo stato delle predizioni
export function togglePredictions() {
    const controlBtn = document.getElementById('predictions-control-btn');
    
    // Verifica che il pulsante esista
    if (!controlBtn) {
        console.error("Elemento 'predictions-control-btn' non trovato");
        return false;
    }
    
    // Disabilita temporaneamente il pulsante durante l'operazione
    controlBtn.disabled = true;
    
    // Se non stiamo già eseguendo predizioni, avviale
    if (!isPredictionsRunning) {
        startPredictions(controlBtn);
    } else {
        stopPredictions(controlBtn);
    }
    
    return isPredictionsRunning;
}

// Funzione per avviare le predizioni
async function startPredictions(controlBtn) {
    try {
        // Verifica che ci sia almeno un modello e un timeframe selezionato
        if (!validateSelection()) {
            controlBtn.disabled = false;
            return false;
        }
        
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
        
        // Aggiorna l'UI prima della chiamata API
        if (typeof updateRunningUI === 'function') {
            updateRunningUI(true);
        } else {
            console.warn('updateRunningUI non è disponibile');
        }
        
        // Inizializza il bot con le selezioni e i parametri di trading
        appendToLog(`Inizializzazione del bot in corso...`);
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
        
        // Aggiorna i parametri nei log
        appendToLog(`Analisi con: Top ${topCrypto} cripto, Leva ${leverage}x, Margine ${margin} USDT`);
        
        // Avvia il bot
        appendToLog(`Avvio del bot in corso...`);
        const startResult = await makeApiRequest('/start', 'POST');
        if (!startResult) {
            throw new Error('Errore durante l\'avvio');
        }
        
        if (startResult.status && startResult.status.includes("Bot avviato")) {
            appendToLog(`Bot avviato con successo`);
        } else {
            appendToLog(`Risposta dell'avvio: ${JSON.stringify(startResult)}`);
        }
        
        // Cambia lo stato in "in esecuzione"
        isPredictionsRunning = true;
        
        // Disabilita i controlli durante l'esecuzione
        document.querySelectorAll('.btn-check').forEach(checkbox => {
            checkbox.disabled = true;
        });
        
        // Disabilita anche i controlli dei parametri di trading se esistono
        if (leverageRange) leverageRange.disabled = true;
        if (marginRange) marginRange.disabled = true;
        
        // Mostra il loader durante il caricamento iniziale delle predizioni
        document.getElementById('predictions-loading').classList.remove('d-none');
        
        // Carica le predizioni e avvia un intervallo per aggiornarle
        await loadPredictions();
        
        // Nascondi il loader una volta caricate le predizioni
        document.getElementById('predictions-loading').classList.add('d-none');
        
        // Imposta un intervallo per ricaricare periodicamente le predizioni
        if (isPredictionsRunning) {
            // Verifica se esiste già un intervallo attivo e cancellalo prima di crearne uno nuovo
            if (predictionsInterval) {
                clearInterval(predictionsInterval);
            }
            
            predictionsInterval = setInterval(() => {
                if (isPredictionsRunning) {
                    loadPredictions();
                } else {
                    clearInterval(predictionsInterval);
                }
            }, 30000); // Aggiorna ogni 30 secondi
            
            // Verifica se esiste già un intervallo per il controllo dello stato e cancellalo prima di crearne uno nuovo
            if (statusCheckInterval) {
                clearInterval(statusCheckInterval);
            }
            
            // Imposta un controllo periodico dello stato del bot
            statusCheckInterval = setInterval(() => {
                checkBotStatus();
            }, 10000); // Controlla ogni 10 secondi
            
            // Controllo immediato dello stato
            checkBotStatus();
            
            // Aggiorna la UI con lo stato "in esecuzione"
            if (typeof updateActivityStatus === 'function') {
                updateActivityStatus('Attivo', 'success');
            } else {
                console.warn('updateActivityStatus non è disponibile');
            }
        }
        
        // Riabilita il pulsante
        controlBtn.disabled = false;
        
    } catch (error) {
        console.error('Errore durante l\'avvio:', error);
        
        // In caso di errore, ripristina lo stato del pulsante
        isPredictionsRunning = false;
        
        if (typeof updateRunningUI === 'function') {
            updateRunningUI(false);
        } else {
            console.warn('updateRunningUI non è disponibile');
        }
        
        // Mostra errore all'utente
        showAlert(error.message || 'Errore durante l\'avvio delle predizioni', 'danger');
        appendToLog(`Errore: ${error.message || 'Errore durante l\'avvio delle predizioni'}`);
        
        // Riabilita il pulsante
        controlBtn.disabled = false;
    }
}

// Funzione per fermare le predizioni
export async function stopPredictions(controlBtn) {
    try {
        appendToLog(`Arresto del bot in corso...`);
        const response = await makeApiRequest('/stop', 'POST');
        if (!response) {
            throw new Error('Errore durante l\'arresto');
        }
        
        if (response.status) {
            appendToLog(`Risposta server: ${response.status}`);
        }
        
        // Pulisci gli intervalli
        if (predictionsInterval) {
            clearInterval(predictionsInterval);
            predictionsInterval = null;
        }
        
        if (statusCheckInterval) {
            clearInterval(statusCheckInterval);
            statusCheckInterval = null;
        }
        
        // Aggiorna lo stato
        isPredictionsRunning = false;
        
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
        if (typeof updateActivityStatus === 'function') {
            updateActivityStatus('Fermato', 'danger');
        } else {
            console.warn('updateActivityStatus non è disponibile');
        }
        
        // Aggiorna l'UI
        if (typeof updateRunningUI === 'function') {
            updateRunningUI(false);
        } else {
            console.warn('updateRunningUI non è disponibile');
        }
        
    } catch (error) {
        console.error('Errore durante l\'arresto:', error);
        showAlert('Errore durante l\'arresto delle predizioni', 'danger');
        appendToLog(`Errore: ${error.message || 'Errore durante l\'arresto delle predizioni'}`);
    } finally {
        // Riabilita sempre il pulsante
        if (controlBtn) controlBtn.disabled = false;
    }
}

// Funzione per controllare lo stato del bot sul server
export async function checkBotStatus() {
    try {
        console.log("Controllo stato del bot...");
        const statusResponse = await makeApiRequest('/status');
        
        // Se abbiamo una risposta valida
        if (statusResponse && typeof statusResponse.running !== 'undefined') {
            console.log(`Stato bot dal server: ${statusResponse.running ? 'In esecuzione' : 'Fermo'}`);
            console.log(`Stato locale: ${isPredictionsRunning ? 'In esecuzione' : 'Fermo'}`);
            
            // Se lo stato sul server è diverso da quello locale, aggiorna lo stato locale
            if (statusResponse.running !== isPredictionsRunning) {
                console.log(`Stato bot server (${statusResponse.running}) diverso da stato locale (${isPredictionsRunning})`);
                
                if (statusResponse.running) {
                    // Il bot è in esecuzione sul server ma non localmente
                    appendToLog(`Bot rilevato in esecuzione sul server`);
                    isPredictionsRunning = true;
                    
                    if (typeof updateRunningUI === 'function') {
                        updateRunningUI(true);
                    } else {
                        console.warn('updateRunningUI non è disponibile');
                    }
                } else {
                    // Il bot è stato arrestato sul server
                    if (isPredictionsRunning) {
                        appendToLog(`Bot non più in esecuzione sul server. Aggiornamento stato...`);
                        isPredictionsRunning = false;
                        
                        // Pulisci gli intervalli
                        if (predictionsInterval) {
                            clearInterval(predictionsInterval);
                            predictionsInterval = null;
                        }
                        
                        if (statusCheckInterval) {
                            clearInterval(statusCheckInterval);
                            statusCheckInterval = null;
                        }
                        
                        // Aggiorna l'UI
                        if (typeof updateRunningUI === 'function') {
                            updateRunningUI(false);
                        } else {
                            console.warn('updateRunningUI non è disponibile');
                        }
                        
                        // Riabilita i controlli
                        document.querySelectorAll('.btn-check').forEach(checkbox => {
                            checkbox.disabled = false;
                        });
                        
                        // Riabilita anche i controlli dei parametri di trading
                        const leverageRange = document.getElementById('leverage-range');
                        const marginRange = document.getElementById('margin-range');
                        
                        if (leverageRange) leverageRange.disabled = false;
                        if (marginRange) marginRange.disabled = false;
                        
                        // Notifica l'utente
                        showAlert('Il bot è stato arrestato automaticamente', 'warning');
                    }
                }
            } else {
                // Anche se lo stato non è cambiato, assicurati che l'UI rifletta lo stato corretto
                if (typeof updateRunningUI === 'function') {
                    updateRunningUI(isPredictionsRunning);
                }
            }
            
            // Aggiorna il pulsante di controllo per essere sicuri
            const controlBtn = document.getElementById('predictions-control-btn');
            if (controlBtn) {
                if (statusResponse.running) {
                    controlBtn.innerHTML = '<i class="fas fa-stop me-2"></i>Ferma';
                    controlBtn.classList.remove('btn-primary');
                    controlBtn.classList.add('btn-danger', 'running');
                    // Aggiungi l'animazione di pulsazione
                    controlBtn.style.animation = 'pulse 2s infinite';
                } else {
                    controlBtn.innerHTML = '<i class="fas fa-play me-2"></i>Avvia';
                    controlBtn.classList.remove('btn-danger', 'running');
                    controlBtn.classList.add('btn-primary');
                    controlBtn.style.animation = '';
                }
            }
            
            // Se il task ha uno stato, lo registriamo nei log (solo per debug)
            if (statusResponse.task_status && statusResponse.task_status !== 'unknown') {
                console.log(`Stato task bot: ${statusResponse.task_status}`);
            }
        }
    } catch (error) {
        console.error('Errore durante il controllo dello stato del bot:', error);
        // Non aggiorniamo lo stato locale in caso di errore per evitare falsi negativi
    }
}

// Funzione per inizializzare i gestori dei parametri di trading
function initializeTradeParamsHandlers() {
    // Importo funzioni di gestione dagli altri moduli
    import('./predictionParams.js').then(module => {
        module.initializeTradeParamsHandlers();
    }).catch(error => {
        console.error('Errore nel caricamento del modulo predictionParams.js:', error);
    });
}

// Funzione per inizializzare i gestori delle selezioni
function initializeSelectionHandlers() {
    // Importa funzioni di gestione dagli altri moduli
    import('./predictionModels.js').then(module => {
        module.initializeSelectionHandlers();
    }).catch(error => {
        console.error('Errore nel caricamento del modulo predictionModels.js:', error);
    });
}

// Carica le funzioni UI dal modulo predictionUI.js
function loadUIFunctions() {
    // Importa funzioni di inizializzazione dagli altri moduli
    import('./predictionUI.js').then(module => {
        // Salva i riferimenti alle funzioni
        if (module.initActivityVisualization) {
            initActivityVisualization = module.initActivityVisualization;
            // Esegui la funzione di inizializzazione dopo averla acquisita
            initActivityVisualization();
        } else {
            console.warn('initActivityVisualization non trovata nel modulo predictionUI.js');
        }
        
        if (module.updateRunningUI) {
            updateRunningUI = module.updateRunningUI;
        } else {
            console.warn('updateRunningUI non trovata nel modulo predictionUI.js');
        }
        
        if (module.updateActivityStatus) {
            updateActivityStatus = module.updateActivityStatus;
        } else {
            console.warn('updateActivityStatus non trovata nel modulo predictionUI.js');
        }
    }).catch(error => {
        console.error('Errore nel caricamento del modulo predictionUI.js:', error);
    });
}

// Funzioni per ottenere i modelli e i timeframe selezionati
// Queste funzioni potrebbero essere reimplementate qui se non sono disponibili dal modulo predictionModels
function getSelectedModels() {
    try {
        const modelsModule = require('./predictionModels.js');
        if (modelsModule.getSelectedModels) {
            return modelsModule.getSelectedModels();
        }
    } catch (error) {
        console.warn('Impossibile importare getSelectedModels da predictionModels.js');
    }
    
    // Implementazione di fallback
    return Array.from(document.querySelectorAll('.model-checkbox:checked')).map(checkbox => checkbox.value);
}

function getSelectedTimeframes() {
    try {
        const modelsModule = require('./predictionModels.js');
        if (modelsModule.getSelectedTimeframes) {
            return modelsModule.getSelectedTimeframes();
        }
    } catch (error) {
        console.warn('Impossibile importare getSelectedTimeframes da predictionModels.js');
    }
    
    // Implementazione di fallback
    return Array.from(document.querySelectorAll('.timeframe-checkbox:checked')).map(checkbox => checkbox.value);
}

// Funzione per validare la selezione
function validateSelection() {
    try {
        const modelsModule = require('./predictionModels.js');
        if (modelsModule.validateSelection) {
            return modelsModule.validateSelection();
        }
    } catch (error) {
        console.warn('Impossibile importare validateSelection da predictionModels.js');
    }
    
    // Implementazione di fallback
    const selectedModels = getSelectedModels();
    const selectedTimeframes = getSelectedTimeframes();
    
    if (selectedModels.length === 0) {
        showAlert('Seleziona almeno un modello prima di avviare le previsioni', 'warning');
        return false;
    }
    
    if (selectedTimeframes.length === 0) {
        showAlert('Seleziona almeno un timeframe prima di avviare le previsioni', 'warning');
        return false;
    }
    
    return true;
}