// models.js - Gestisce tutte le funzionalità relative ai modelli di ML
import { makeApiRequest, setupApiService } from './api.js';
import { showAlert } from './ui.js';
import { createComparisonChart } from './charts.js';

// Variabili globali per lo stato del training
let modelMetricsData = {};
let currentMetric = 'accuracy';
let trainingInProgress = false;

// Funzione per controllare lo stato di tutti i modelli
export async function checkAllModelsStatus() {
    const modelTypes = ['lstm', 'rf', 'xgb'];
    const timeframes = ['5m', '15m', '30m', '1h', '4h'];
    
    // Utilizziamo fetch per recuperare la lista dei modelli disponibili
    try {
        const response = await makeApiRequest('/list-models');
        
        if (response) {
            const modelFiles = response.models || [];
            
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
export function loadModelMetrics() {
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

// Funzione per recuperare le metriche dei modelli
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
            const modelResponse = await makeApiRequest(`/check-model-exists/${modelFile}`);
            
            if (modelResponse) {
                // Ora ottieni il contenuto effettivo delle metriche
                const metricDetailsResponse = await makeApiRequest(`/trained_models/${metricsFile}`);
                
                if (metricDetailsResponse) {
                    // Estrai le metriche rilevanti e standardizzale
                    metrics[type] = {
                        name: modelNames[type],
                        type: type,
                        accuracy: extractMetric(metricDetailsResponse, 'accuracy'),
                        precision: extractMetric(metricDetailsResponse, 'precision'),
                        recall: extractMetric(metricDetailsResponse, 'recall'),
                        f1: extractMetric(metricDetailsResponse, 'f1'),
                        auc: extractMetric(metricDetailsResponse, 'auc'),
                        // Salva anche i dati grezzi per i dettagli
                        rawMetrics: metricDetailsResponse
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

// Funzione per visualizzare il grafico delle metriche
function displayMetricsChart() {
    const ctx = document.getElementById('metrics-comparison-chart').getContext('2d');
    
    // Distruggi il grafico esistente, se presente
    if (window.metricsChart) {
        window.metricsChart.destroy();
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
    window.metricsChart = new Chart(ctx, {
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
}

// Funzione per visualizzare la tabella delle metriche
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

// Funzione per ottenere l'etichetta di una metrica
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

// Funzione per gestire l'invio del form di training
export async function handleTrainingFormSubmit(e) {
    e.preventDefault();
    console.log('Form di training inviato');
    
    if (trainingInProgress) {
        showAlert('Un training è già in corso, attendere il completamento', 'warning');
        return;
    }
    
    try {
        // Configura le credenziali API se non sono già state configurate
        const apiKey = localStorage.getItem('apiKey');
        const apiSecret = localStorage.getItem('apiSecret');
        if (apiKey && apiSecret) {
            setupApiService(apiKey, apiSecret);
        } else {
            showAlert('Errore: Credenziali API non configurate', 'danger');
            return;
        }
        
        const modelType = document.getElementById('model-type')?.value;
        const timeframe = document.getElementById('timeframe')?.value;
        // Utilizziamo valori fissi invece di quelli dinamici
        const dataLimitDays = 50;  // Valore fisso: 50 giorni
        const topTrainCrypto = 100; // Valore fisso: 100 criptovalute
        
        console.log('Parametri di training:', { modelType, timeframe, dataLimitDays, topTrainCrypto });
        
        if (!modelType || !timeframe) {
            console.error('Parametri mancanti', { modelType, timeframe });
            showAlert('Errore: Parametri mancanti per il training.', 'danger');
            return;
        }
        
        // Imposta lo stato del training
        trainingInProgress = true;
        
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
        
        console.log(`Invio richiesta a /api/train-model`);
        
        const response = await makeApiRequest('/api/train-model', 'POST', {
            model_type: modelType,
            timeframe: timeframe,
            data_limit_days: dataLimitDays,
            top_train_crypto: topTrainCrypto
        });
        
        if (!response) {
            throw new Error('Errore durante il training');
        }

        // Polling per aggiornamenti di stato
        let progress = 0;
        const pollInterval = setInterval(async () => {
            try {
                // Codifica i parametri per l'URL
                const encodedModelType = encodeURIComponent(modelType);
                const encodedTimeframe = encodeURIComponent(timeframe);
                
                const statusResponse = await makeApiRequest(`/api/training-status/${encodedModelType}/${encodedTimeframe}`);
                
                if (!statusResponse) {
                    throw new Error('Errore nel recupero dello stato del training');
                }
                
                if (statusResponse.status === 'completed') {
                    clearInterval(pollInterval);
                    updateTrainingProgress(100);
                    updateTrainingStatus('Completato', 'success');
                    updateTerminalProgress(100, "Completato");
                    
                    // Aggiorna l'informazione sul modello completato
                    resetCurrentTrainingModel('completed');
                    
                    // Aggiorna le metriche se disponibili
                    if (statusResponse.metrics) {
                        displayMetrics(statusResponse.metrics, modelType, timeframe);
                    }
                    
                    // Aggiorna lo stato del modello appena addestrato
                    const modelCell = document.getElementById(`${modelType}-${timeframe}`);
                    if (modelCell) {
                        modelCell.innerHTML = '<i class="fas fa-check-circle text-success me-2"></i> Disponibile';
                        modelCell.className = "model-status model-available";
                    }
                    
                    // Aggiorna il grafico delle metriche dopo l'addestramento
                    loadModelMetrics();
                    
                    // Reset dello stato del training
                    trainingInProgress = false;
                } else if (statusResponse.status === 'error') {
                    clearInterval(pollInterval);
                    updateTrainingStatus('Errore', 'danger');
                    updateTerminalProgress(0, "Errore");
                    
                    // Aggiorna l'informazione sul modello in errore
                    resetCurrentTrainingModel('error');
                    
                    // Reset dello stato del training
                    trainingInProgress = false;
                } else {
                    // Aggiorna progresso
                    progress = statusResponse.progress || progress;
                    updateTrainingProgress(progress);
                    
                    // Aggiorna la barra di progresso testuale
                    let currentStep = "Elaborazione";
                    let totalItems = 0;
                    let currentItem = 0;
                    
                    // Estrai informazioni dal current_step se disponibile
                    if (statusResponse.current_step) {
                        currentStep = statusResponse.current_step.split(':')[0].trim();
                        
                        // Prova a estrarre informazioni sul conteggio
                        const countMatch = statusResponse.current_step.match(/(\d+)\/(\d+)/);
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
                
                // Reset dello stato del training
                trainingInProgress = false;
            }
        }, 2000); // Polling ogni 2 secondi

    } catch (error) {
        console.error('Errore durante il training:', error);
        updateTrainingStatus('Errore', 'danger');
        updateTerminalProgress(0, "Errore");
        
        // Reset dello stato del training
        trainingInProgress = false;
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
        // Ignora oggetti complessi
        if (typeof value === 'object') return;
        
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

// Funzione per aggiornare la visualizzazione delle fasi del training
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

// Funzione per aggiornare il terminale con informazioni più semplici
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

// Funzione per aggiornare i dettagli di una fase specifica
function updateStepDetails(step, details) {
    const detailsElement = document.getElementById(`step-${step}-details`);
    if (detailsElement) {
        detailsElement.textContent = details;
    }
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
        if (status === 'completed') {
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

// Funzione per ottenere il nome leggibile del tipo di modello
function getModelTypeName(modelType) {
    switch (modelType) {
        case 'lstm': return 'LSTM';
        case 'rf': return 'Random Forest';
        case 'xgb': return 'XGBoost';
        default: return modelType.toUpperCase();
    }
}

// Inizializza gli event listener della sezione modelli
export function setupModelsEventListeners() {
    // Aggiungi event listener per il form di training
    const trainingForm = document.getElementById('model-training-form');
    if (trainingForm) {
        trainingForm.addEventListener('submit', handleTrainingFormSubmit);
    }

    // Aggiungi event listener per i pulsanti delle metriche
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
    
    // Event listener per i selettori delle metriche
    const metricsTimeframeSelect = document.getElementById('metrics-timeframe-select');
    if (metricsTimeframeSelect) {
        metricsTimeframeSelect.addEventListener('change', function() {
            loadModelMetrics();
        });
    }
    
    // Carica eventi quando la sezione modelli viene selezionata
    document.addEventListener('models-selected', () => {
        checkAllModelsStatus();
    });
} 