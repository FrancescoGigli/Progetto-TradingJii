// Funzione che controlla lo stato dei modelli e aggiorna la tabella
async function checkModelStatus() {
    const models = ['lstm', 'rf', 'xgb'];
    const modelNames = {
        'lstm': 'LSTM',
        'rf': 'Random Forest',
        'xgb': 'XGBoost'
    };
    const fileExtensions = {
        'lstm': '.h5',
        'rf': '.pkl',
        'xgb': '.pkl'
    };
    const timeframes = ['5m', '15m', '30m', '1h', '4h'];

    console.log("Auto-status.js: Controllo effettivo dei file dei modelli...");
    
    // Funzione che verifica l'esistenza del file del modello
    async function checkModelFile(model, timeframe) {
        try {
            const ext = fileExtensions[model] || '.pkl';
            const filename = `${model}_model_${timeframe}${ext}`;
            const response = await fetch(`/api/check-model-exists/${filename}`);
            
            if (!response.ok) {
                return 'non disponibile';
            }
            
            const data = await response.json();
            return data.exists ? 'disponibile' : 'non disponibile';
        } catch (error) {
            console.error(`Errore nel controllo del modello ${model} ${timeframe}:`, error);
            return 'non disponibile';
        }
    }
    
    // Aggiorna tutte le celle con "Verifica in corso..." prima di iniziare
    for (const model of models) {
        for (const timeframe of timeframes) {
            const cellId = `${model}-${timeframe}`;
            const cell = document.getElementById(cellId);
            
            if (cell) {
                cell.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span> Verifica...';
                cell.className = 'model-status model-checking';
            }
        }
    }
    
    // Controlla ogni modello in sequenza con un piccolo ritardo tra i check
    for (const model of models) {
        for (const timeframe of timeframes) {
            const cellId = `${model}-${timeframe}`;
            const cell = document.getElementById(cellId);
            
            if (!cell) {
                console.warn(`Cella ${cellId} non trovata`);
                continue;
            }
            
            // Piccolo ritardo per non sovraccaricare il server
            await new Promise(resolve => setTimeout(resolve, 100));
            
            try {
                const status = await checkModelFile(model, timeframe);
                
                // Usa icone invece di solo testo
                if (status === 'disponibile') {
                    cell.innerHTML = '<i class="fas fa-check-circle me-2"></i> Disponibile';
                    cell.className = 'model-status model-available';
                } else {
                    cell.innerHTML = '<i class="fas fa-times-circle me-2"></i> Non disponibile';
                    cell.className = 'model-status model-unavailable';
                }
            } catch (error) {
                console.error(`Errore per ${model}-${timeframe}:`, error);
                cell.innerHTML = '<i class="fas fa-exclamation-triangle me-2"></i> Errore';
                cell.className = 'model-status model-error';
            }
        }
    }
}

// Esegui la funzione quando il DOM è caricato
document.addEventListener('DOMContentLoaded', function() {
    console.log("Auto-status.js: DOM caricato, inizializzazione...");
    
    // Aggiungi stili CSS migliorati
    const style = document.createElement('style');
    style.textContent = `
        .model-status {
            text-align: center;
            padding: 10px;
            border-radius: 6px;
            font-weight: 500;
            transition: all 0.3s ease;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        
        .model-checking {
            color: #6c757d;
            background: linear-gradient(135deg, rgba(255,193,7,0.1) 0%, rgba(255,193,7,0.15) 100%);
            border: 1px solid rgba(255,193,7,0.3);
            animation: pulse 1.5s infinite;
        }
        
        .model-available {
            color: #28a745;
            background: linear-gradient(135deg, rgba(40,167,69,0.05) 0%, rgba(40,167,69,0.15) 100%);
            border: 1px solid rgba(40,167,69,0.3);
        }
        
        .model-available:hover {
            background: linear-gradient(135deg, rgba(40,167,69,0.1) 0%, rgba(40,167,69,0.2) 100%);
            box-shadow: 0 4px 10px rgba(40,167,69,0.2);
            transform: translateY(-2px);
        }
        
        .model-unavailable {
            color: #dc3545;
            background: linear-gradient(135deg, rgba(220,53,69,0.05) 0%, rgba(220,53,69,0.15) 100%);
            border: 1px solid rgba(220,53,69,0.3);
        }
        
        .model-unavailable:hover {
            background: linear-gradient(135deg, rgba(220,53,69,0.1) 0%, rgba(220,53,69,0.2) 100%);
            box-shadow: 0 4px 10px rgba(220,53,69,0.2);
            transform: translateY(-2px);
        }
        
        .model-error {
            color: #6c757d;
            background: linear-gradient(135deg, rgba(108,117,125,0.05) 0%, rgba(108,117,125,0.15) 100%);
            border: 1px solid rgba(108,117,125,0.3);
        }
        
        @keyframes pulse {
            0% { opacity: 0.7; }
            50% { opacity: 1; }
            100% { opacity: 0.7; }
        }
    `;
    document.head.appendChild(style);
    
    // Esegui la verifica iniziale
    setTimeout(checkModelStatus, 1000);
    
    // Aggiungi un listener per l'evento di cambio tab
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', function(e) {
            if (e.target.getAttribute('href') === '#models') {
                console.log("Auto-status.js: Cambio a tab modelli, aggiornamento stato...");
                setTimeout(checkModelStatus, 300);
            }
        });
    });
    
    // Aggiungi un pulsante per aggiornare manualmente lo stato
    const tableHeader = document.querySelector('.card-header h5');
    if (tableHeader && tableHeader.textContent.trim() === 'Stato Modelli') {
        const refreshButton = document.createElement('button');
        refreshButton.className = 'btn btn-sm btn-outline-secondary ms-2';
        refreshButton.innerHTML = '<i class="fas fa-sync-alt"></i> Aggiorna';
        refreshButton.style.cssText = 'float: right; margin-top: -5px;';
        refreshButton.addEventListener('click', function() {
            checkModelStatus();
        });
        tableHeader.appendChild(refreshButton);
    }
});

// Esegui la funzione immediatamente se il DOM è già caricato
if (document.readyState === 'interactive' || document.readyState === 'complete') {
    console.log("Auto-status.js: DOM già caricato, esecuzione immediata...");
    setTimeout(checkModelStatus, 1000);
} 