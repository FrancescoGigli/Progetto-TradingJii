// Versione semplificata che non richiede API
function check_model_status(model, timeframe) {
    // Simulazione: qui puoi integrare la logica effettiva di controllo dei modelli
    // Per ora ritorna "disponibile" o "non disponibile" in modo simulato
    const hash = (model + timeframe).split('').reduce((a, b) => {
        return ((a << 5) - a) + b.charCodeAt(0);
    }, 0);
    
    // Simulazione deterministica - in produzione sostituire con verifica reale
    return (hash % 3 === 0) ? 'non disponibile' : 'disponibile';
}

function updateModelStatus() {
    console.log("Aggiornamento stato modelli...");
    const models = ['lstm', 'rf', 'xgb'];
    const modelNames = {
        'lstm': 'LSTM',
        'rf': 'Random Forest',
        'xgb': 'XGBoost'
    };
    const timeframes = ['5m', '15m', '30m', '1h', '4h'];
    
    for (const model of models) {
        for (const timeframe of timeframes) {
            const cellId = `${model}-${timeframe}`;
            const cell = document.getElementById(cellId);
            if (!cell) {
                console.warn(`Cella con ID ${cellId} non trovata`);
                continue;
            }
            
            // Imposta lo stato iniziale
            cell.textContent = 'Verifica in corso...';
            cell.className = 'model-status model-checking';
            
            // Utilizza setTimeout per dare tempo all'interfaccia di aggiornarsi
            setTimeout(() => {
                try {
                    // Verifica lo stato del modello
                    const status = check_model_status(modelNames[model], timeframe);
                    cell.textContent = status;
                    cell.className = `model-status model-${status === 'disponibile' ? 'available' : 'unavailable'}`;
                } catch (error) {
                    console.error(`Errore verifica modello ${model}-${timeframe}:`, error);
                    cell.textContent = 'non disponibile';
                    cell.className = 'model-status model-unavailable';
                }
            }, 500 + Math.random() * 1000); // Aggiunta casualità per effetto visivo
        }
    }
}

// Aggiungi stili CSS direttamente nel file JS
const style = document.createElement('style');
style.textContent = `
    .model-status {
        text-align: center;
        padding: 8px;
        border-radius: 4px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .model-checking {
        background-color: rgba(255, 193, 7, 0.1);
        color: #ffc107;
    }
    
    .model-available {
        background-color: rgba(40, 167, 69, 0.1);
        color: #28a745;
    }
    
    .model-unavailable {
        background-color: rgba(220, 53, 69, 0.1);
        color: #dc3545;
    }
`;
document.head.appendChild(style);

// Esegui immediatamente all'inizializzazione del file
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        console.log("DOM caricato, aggiornamento stato modelli...");
        setTimeout(updateModelStatus, 500);
    });
} else {
    console.log("DOM già caricato, aggiornamento immediato stato modelli...");
    setTimeout(updateModelStatus, 500);
}

// Aggiungi anche un evento per il cambio di tab, in caso l'utente cambi sezione
document.addEventListener('click', function(e) {
    if (e.target && e.target.getAttribute('href') === '#models') {
        console.log("Cambio a tab modelli, aggiornamento stato...");
        setTimeout(updateModelStatus, 500);
    }
}); 