// predictionData.js - Gestione dei dati delle predizioni
import { makeApiRequest } from './api.js';
import { showAlert, showNotification } from './ui.js';
import { getSelectedModels, getSelectedTimeframes, calculateEnsembleConsensus } from './predictionModels.js';
import { displayPredictions } from './predictionDisplay.js';

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
            showNotification('success', `${ensembleResults.length} nuove predizioni caricate`, true);
            
            // Aggiorna il timestamp dell'ultimo aggiornamento
            updateLastUpdateTimestamp();
            
            return ensembleResults;
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
            return [];
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
        
        showAlert(`Errore durante il caricamento delle predizioni: ${error.message || 'Errore sconosciuto'}`, 'danger');
        return [];
    }
}

// Funzione per raggruppare le predizioni per simbolo
export function groupPredictionsBySymbol(predictions) {
    return predictions.reduce((acc, pred) => {
        if (!acc[pred.symbol]) {
            acc[pred.symbol] = [];
        }
        acc[pred.symbol].push(pred);
        return acc;
    }, {});
}

// Funzione per esportare le predizioni in CSV
export function exportPredictions(predictions) {
    // Crea intestazioni CSV
    let csv = 'Simbolo,Confidenza,Direzione,RSI,LSTM,RandomForest,XGBoost\n';
    
    // Aggiungi ogni riga
    predictions.forEach(prediction => {
        const confidencePercent = (Math.abs(prediction.ensemble_value - 0.5) * 200).toFixed(1);
        const models = prediction.models;
        
        csv += `${prediction.symbol},`;
        csv += `${confidencePercent}%,`;
        csv += `${prediction.direction},`;
        csv += `${prediction.rsi_value.toFixed(1)},`;
        csv += `${models.lstm ? (models.lstm.weighted_average * 100).toFixed(1) + '%' : 'N/A'},`;
        csv += `${models.rf ? (models.rf.weighted_average * 100).toFixed(1) + '%' : 'N/A'},`;
        csv += `${models.xgb ? (models.xgb.weighted_average * 100).toFixed(1) + '%' : 'N/A'}\n`;
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

// Funzione semplice per aggiornare l'ultimo timestamp
function updateLastUpdateTimestamp() {
    const lastUpdateEl = document.getElementById('last-update-timestamp');
    if (lastUpdateEl) {
        lastUpdateEl.textContent = new Date().toLocaleString('it-IT');
    }
} 