// Funzioni per la gestione del training dei modelli
async function check_model_status(model, timeframe) {
    try {
        // Chiamata all'API per verificare lo stato del modello
        const response = await fetch(`http://localhost:5000/api/status?model=${model}&timeframe=${timeframe}`);
        if (!response.ok) {
            throw new Error('Errore nella verifica del modello');
        }
        
        const data = await response.json();
        return data.available ? 'disponibile' : 'non disponibile';
    } catch (error) {
        console.error('Errore nella verifica del modello:', error);
        return 'non disponibile';
    }
}

async function updateModelStatus() {
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
            
            try {
                // Effettua la verifica del modello
                const status = await check_model_status(model, timeframe);
                
                // Aggiorna l'interfaccia utente
                setTimeout(() => {
                    cell.textContent = status;
                    cell.className = `model-status model-${status === 'disponibile' ? 'available' : 'unavailable'}`;
                }, 100);
            } catch (error) {
                console.error(`Errore verifica modello ${model}-${timeframe}:`, error);
                cell.textContent = 'non disponibile';
                cell.className = 'model-status model-unavailable';
            }
        }
    }
}

// Funzione per avviare il training
async function startTraining(options) {
    try {
        const response = await fetch('http://localhost:5000/api/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(options)
        });
        
        if (!response.ok) {
            throw new Error('Errore nell\'avvio del training');
        }
        
        const data = await response.json();
        console.log('Training avviato:', data);
        return true;
    } catch (error) {
        console.error('Errore nell\'avvio del training:', error);
        return false;
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