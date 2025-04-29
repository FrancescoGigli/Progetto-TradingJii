// Variabile globale per tenere traccia dei modelli mancanti
let missingModels = {};

// Variabile globale per tenere traccia del training in corso
let activeTraining = {
    isRunning: false,
    model: null,
    type: null, // 'single', 'missing', 'all', 'selected'
    progress: 0,
    phase: 'idle', // 'idle', 'preparing', 'fetching', 'training', 'finalizing', 'completed', 'error'
    startTime: null
};

// Funzione che simula un aggiornamento del progresso del training
function startProgressSimulation() {
    // Reset del progresso
    activeTraining.progress = 0;
    activeTraining.phase = 'preparing';
    activeTraining.startTime = new Date();
    
    // Aggiorna il terminale di progresso
    updateTerminalProgress();
    
    // Simula le fasi del training
    const phases = ['preparing', 'fetching', 'training', 'finalizing', 'completed'];
    const phaseMessages = {
        'preparing': 'Preparazione ambiente di training...',
        'fetching': 'Recupero dati storici...',
        'training': 'Addestramento modello in corso...',
        'finalizing': 'Finalizzazione e salvataggio modello...',
        'completed': 'Training completato con successo!'
    };
    
    let currentPhaseIndex = 0;
    
    // Aggiorna il progresso ogni 500ms
    const progressInterval = setInterval(() => {
        // Se il training è stato interrotto, ferma la simulazione
        if (!activeTraining.isRunning) {
            clearInterval(progressInterval);
            return;
        }
        
        // Incrementa il progresso
        activeTraining.progress += Math.random() * 2;
        
        // Cambia fase quando raggiungiamo determinate soglie
        if (activeTraining.progress >= 25 && currentPhaseIndex < 1) {
            currentPhaseIndex = 1;
            activeTraining.phase = phases[currentPhaseIndex];
        } else if (activeTraining.progress >= 40 && currentPhaseIndex < 2) {
            currentPhaseIndex = 2;
            activeTraining.phase = phases[currentPhaseIndex];
        } else if (activeTraining.progress >= 85 && currentPhaseIndex < 3) {
            currentPhaseIndex = 3;
            activeTraining.phase = phases[currentPhaseIndex];
        } else if (activeTraining.progress >= 100) {
            activeTraining.progress = 100;
            activeTraining.phase = 'completed';
            clearInterval(progressInterval);
            
            // Aggiorna lo stato dei modelli dopo un breve ritardo
            setTimeout(() => {
                if (activeTraining.isRunning) {
                    // Simula la fine del training
                    const oldModel = activeTraining.model;
                    const oldType = activeTraining.type;
                    
                    activeTraining.isRunning = false;
                    activeTraining.model = null;
                    activeTraining.type = null;
                    
                    // Ricarica lo stato dei modelli
                    checkModelStatus();
                }
            }, 1500);
        }
        
        // Aggiorna il terminale di progresso
        updateTerminalProgress();
    }, 500);
}

// Funzione per aggiornare il terminale di progresso
function updateTerminalProgress() {
    const terminal = document.querySelector('.terminal-progress-container');
    if (!terminal) return;
    
    // Mostra il terminale
    terminal.style.display = 'block';
    
    // Aggiorna il testo del terminale
    const terminalText = terminal.querySelector('#terminal-progress-text');
    if (!terminalText) return;
    
    // Ottieni il messaggio della fase corrente
    const phaseMessages = {
        'idle': 'In attesa...',
        'preparing': 'Preparazione ambiente di training...',
        'fetching': 'Recupero dati storici e calcolo indicatori...',
        'training': 'Addestramento modello in corso...',
        'finalizing': 'Ottimizzazione e salvataggio modello...',
        'completed': 'Training completato con successo!',
        'error': 'Errore durante il training.'
    };
    
    // Calcola il tempo trascorso
    let elapsedTime = '';
    if (activeTraining.startTime) {
        const now = new Date();
        const elapsed = Math.floor((now - activeTraining.startTime) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;
        elapsedTime = ` (${minutes}m ${seconds}s)`;
    }
    
    // Crea un indicatore di progresso visivo
    const progressBar = '█'.repeat(Math.floor(activeTraining.progress / 5)) + 
                        '░'.repeat(20 - Math.floor(activeTraining.progress / 5));
    
    terminalText.innerHTML = `<span class="terminal-phase">${phaseMessages[activeTraining.phase] || 'In corso...'}</span> <span class="terminal-time">${elapsedTime}</span><br>
                             <span class="terminal-progress">[${progressBar}] ${Math.round(activeTraining.progress)}%</span>`;
    
    // Aggiorna anche la barra di progresso se presente
    const progressBarFill = document.getElementById('progress-bar-fill');
    if (progressBarFill) {
        progressBarFill.style.width = `${activeTraining.progress}%`;
    }
    
    // Aggiungi stili CSS per il terminale
    const styleId = 'terminal-styles';
    if (!document.getElementById(styleId)) {
        const style = document.createElement('style');
        style.id = styleId;
        style.textContent = `
            .terminal-progress-container {
                background-color: #1a1a1a;
                color: #33ff33;
                font-family: 'Courier New', monospace;
                padding: 15px;
                border-radius: 8px;
                border: 1px solid #444;
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                margin-top: 20px;
                position: relative;
                overflow: hidden;
            }
            
            .terminal-progress-container::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 4px;
                background: linear-gradient(90deg, #33ff33, #33ff33, transparent);
                opacity: 0.7;
                animation: terminal-scan 3s linear infinite;
            }
            
            @keyframes terminal-scan {
                0% { transform: translateX(-100%); }
                100% { transform: translateX(100%); }
            }
            
            #terminal-progress-text {
                line-height: 1.5;
                white-space: pre-wrap;
            }
            
            .terminal-phase {
                color: #33ff33;
                font-weight: bold;
            }
            
            .terminal-time {
                color: #aaa;
                font-style: italic;
            }
            
            .terminal-progress {
                color: #33ff33;
                letter-spacing: 1px;
            }
            
            /* Rimuovo gli stili per le notifiche */
            .training-notification {
                display: none;
            }
            
            /* Nuovi stili per la barra di progresso a fasi migliorata */
            .progress-tracker {
                margin: 30px 0;
                padding: 20px 0;
            }
        `;
        document.head.appendChild(style);
    }
}

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
    
    // Reset dei modelli mancanti
    missingModels = {};
    models.forEach(model => {
        missingModels[model] = [];
    });
    
    // Funzione che verifica l'esistenza del file del modello
    async function checkModelFile(model, timeframe) {
        try {
            const response = await fetch(`http://localhost:5000/api/status?model=${model}&timeframe=${timeframe}`);
            
            if (!response.ok) {
                return 'non disponibile';
            }
            
            const data = await response.json();
            return data.available ? 'disponibile' : 'non disponibile';
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
                    
                    // Aggiungi alla lista dei modelli mancanti
                    missingModels[model].push(timeframe);
                }
            } catch (error) {
                console.error(`Errore per ${model}-${timeframe}:`, error);
                cell.innerHTML = '<i class="fas fa-exclamation-triangle me-2"></i> Errore';
                cell.className = 'model-status model-error';
                
                // Per sicurezza, aggiungi anche i modelli con errore alla lista dei mancanti
                missingModels[model].push(timeframe);
            }
        }
    }
    
    // Aggiorna i pulsanti di training in base ai modelli mancanti
    updateTrainingButtons();
}

// Funzione per aggiornare i pulsanti di training
function updateTrainingButtons() {
    const trainButtonContainer = document.querySelector('.d-flex.justify-content-center.my-3');
    if (!trainButtonContainer) return;
    
    // Rimuovi i pulsanti esistenti
    trainButtonContainer.innerHTML = '';
    
    // Conta quanti modelli mancano
    let totalMissing = 0;
    let hasMissing = false;
    
    for (const model in missingModels) {
        if (missingModels[model].length > 0) {
            hasMissing = true;
            totalMissing += missingModels[model].length;
        }
    }
    
    // Crea un container grid per i pulsanti
    const gridContainer = document.createElement('div');
    gridContainer.className = 'container p-0';
    
    // Crea la prima riga per i pulsanti dei singoli modelli
    const firstRow = document.createElement('div');
    firstRow.className = 'row g-2 mb-2';
    
    // Crea la seconda riga per il pulsante del modello selezionato e tutti i modelli
    const secondRow = document.createElement('div');
    secondRow.className = 'row g-2';
    
    // Definisci colori e icone per i modelli
    const modelColors = {
        'lstm': 'info',
        'rf': 'success',
        'xgb': 'purple'
    };
    
    const modelIcons = {
        'lstm': 'brain',
        'rf': 'tree',
        'xgb': 'bolt'
    };
    
    const modelNames = {
        'lstm': 'LSTM',
        'rf': 'Random Forest',
        'xgb': 'XGBoost'
    };
    
    // Crea pulsanti per ogni singolo modello nella prima riga
    for (const model of Object.keys(missingModels)) {
        const missingCount = missingModels[model].length;
        const btnColor = modelColors[model] || 'secondary';
        const icon = modelIcons[model] || 'cog';
        const modelName = modelNames[model] || model.toUpperCase();
        
        // Crea una colonna per il pulsante
        const col = document.createElement('div');
        col.className = 'col-md-4';
        
        // Crea il pulsante
        const singleModelBtn = document.createElement('button');
        singleModelBtn.type = 'button';
        singleModelBtn.id = `train-${model}-btn`;
        singleModelBtn.className = `btn btn-${btnColor} btn-md btn-train w-100`;
        
        // Controlla se questo modello ha un training attivo
        if (activeTraining.isRunning && activeTraining.model === model && activeTraining.type === 'single') {
            singleModelBtn.classList.add('training-btn-active');
            singleModelBtn.classList.add('btn-stop');
        }
        
        // Aggiungi badge se ci sono modelli mancanti
        let badgeHtml = '';
        if (missingCount > 0) {
            badgeHtml = `<span class="position-absolute top-0 start-100 translate-middle badge rounded-pill bg-danger">${missingCount}</span>`;
            singleModelBtn.classList.add('position-relative');
        }
        
        singleModelBtn.innerHTML = `<i class="fas fa-${icon} me-2"></i>Train ${modelName} ${badgeHtml}`;
        
        singleModelBtn.addEventListener('click', function() {
            // Se c'è già un training attivo per questo modello, fermalo
            if (activeTraining.isRunning && activeTraining.model === model && activeTraining.type === 'single') {
                stopTraining();
            } else {
                // Altrimenti avvia il training
                trainSingleModel(model);
            }
        });
        
        col.appendChild(singleModelBtn);
        firstRow.appendChild(col);
    }
    
    // Pulsante per addestrare solo il modello selezionato (nella seconda riga, prima colonna)
    const selectedModelCol = document.createElement('div');
    selectedModelCol.className = 'col-md-4';
    
    const trainSelectedModelBtn = document.createElement('button');
    trainSelectedModelBtn.type = 'button';
    trainSelectedModelBtn.id = 'train-selected-model-btn';
    trainSelectedModelBtn.className = 'btn btn-md btn-primary btn-train w-100';
    
    // Controlla se c'è un training di tipo "selected" attivo
    if (activeTraining.isRunning && activeTraining.type === 'selected') {
        trainSelectedModelBtn.classList.add('training-btn-active');
        trainSelectedModelBtn.classList.add('btn-stop');
    }
    
    trainSelectedModelBtn.innerHTML = '<i class="fas fa-check-double me-2"></i>Train modello selezionato';
    
    trainSelectedModelBtn.addEventListener('click', function() {
        // Se c'è già un training attivo di tipo "selected", fermalo
        if (activeTraining.isRunning && activeTraining.type === 'selected') {
            stopTraining();
        } else {
            // Altrimenti avvia il training per il modello selezionato
            trainSelectedModel();
        }
    });
    
    selectedModelCol.appendChild(trainSelectedModelBtn);
    secondRow.appendChild(selectedModelCol);
    
    // Pulsante per addestrare solo i modelli mancanti (nella seconda riga, seconda colonna)
    const missingModelCol = document.createElement('div');
    missingModelCol.className = 'col-md-4';
    
    const trainMissingBtn = document.createElement('button');
    trainMissingBtn.type = 'button';
    trainMissingBtn.id = 'train-missing-btn';
    trainMissingBtn.className = 'btn btn-md btn-train w-100';
    trainMissingBtn.disabled = !hasMissing;
    
    // Controlla se c'è un training di tipo "missing" attivo
    if (activeTraining.isRunning && activeTraining.type === 'missing') {
        trainMissingBtn.classList.add('training-btn-active');
        trainMissingBtn.classList.add('btn-stop');
    } else {
        // Imposta il colore appropriato in base allo stato
        if (hasMissing) {
            trainMissingBtn.className += ' btn-warning';
        } else {
            trainMissingBtn.className += ' btn-outline-success';
        }
    }
    
    if (hasMissing) {
        trainMissingBtn.innerHTML = `<i class="fas fa-exclamation-triangle me-2"></i>Modelli mancanti (${totalMissing})`;
    } else {
        trainMissingBtn.innerHTML = `<i class="fas fa-check-circle me-2"></i>Nessun modello mancante`;
    }
    
    trainMissingBtn.addEventListener('click', function() {
        // Se c'è già un training attivo di tipo "missing", fermalo
        if (activeTraining.isRunning && activeTraining.type === 'missing') {
            stopTraining();
        } else {
            // Altrimenti avvia il training
            trainSelectedModels(true);
        }
    });
    
    missingModelCol.appendChild(trainMissingBtn);
    secondRow.appendChild(missingModelCol);
    
    // Pulsante per addestrare tutti i modelli (nella seconda riga, terza colonna)
    const allModelsCol = document.createElement('div');
    allModelsCol.className = 'col-md-4';
    
    const trainAllBtn = document.createElement('button');
    trainAllBtn.type = 'button';
    trainAllBtn.id = 'train-all-btn';
    trainAllBtn.className = 'btn btn-md btn-primary btn-train w-100';
    
    // Controlla se c'è un training di tipo "all" attivo
    if (activeTraining.isRunning && activeTraining.type === 'all') {
        trainAllBtn.classList.add('training-btn-active');
        trainAllBtn.classList.add('btn-stop');
        trainAllBtn.innerHTML = '<i class="fas fa-stop-circle me-2"></i>Stop training';
    } else {
        trainAllBtn.innerHTML = '<i class="fas fa-sync-alt me-2"></i>Train tutti i modelli';
    }
    
    trainAllBtn.addEventListener('click', function() {
        // Se c'è già un training attivo di tipo "all", fermalo
        if (activeTraining.isRunning && activeTraining.type === 'all') {
            stopTraining();
        } else {
            // Altrimenti avvia il training
            trainSelectedModels(false);
        }
    });
    
    allModelsCol.appendChild(trainAllBtn);
    secondRow.appendChild(allModelsCol);
    
    // Aggiungi le righe al container grid
    gridContainer.appendChild(firstRow);
    gridContainer.appendChild(secondRow);
    
    // Aggiungi il container grid al container principale
    trainButtonContainer.appendChild(gridContainer);
}

// Funzione per fermare il training in corso
function stopTraining() {
    // Qui chiameresti l'API per fermare il training
    console.log('Stopping training:', activeTraining);
    
    // Chiamata API per fermare il training (implementazione futura)
    // Per ora, resettiamo solo lo stato di training locale
    const oldType = activeTraining.type;
    const oldModel = activeTraining.model;
    
    activeTraining.isRunning = false;
    activeTraining.model = null;
    activeTraining.type = null;
    activeTraining.phase = 'idle';
    
    // Aggiorna l'interfaccia
    updateTrainingButtons();
    
    // Mostra un messaggio di interruzione
    const statusCard = document.querySelector('.training-status-card');
    if (statusCard) {
        const infoAlert = document.getElementById('current-training-info');
        if (infoAlert) {
            infoAlert.className = 'alert alert-warning mb-2';
            infoAlert.innerHTML = `
                <strong>Training interrotto:</strong>
                <div class="mt-2">
                    <div class="mb-1">Il training è stato fermato manualmente.</div>
                </div>
            `;
        }
        
        // Aggiorna il terminale di progresso
        const terminal = statusCard.querySelector('.terminal-progress-container');
        if (terminal && terminal.style.display === 'block') {
            const terminalText = terminal.querySelector('#terminal-progress-text');
            if (terminalText) {
                terminalText.innerHTML = `<span class="terminal-phase" style="color: #ffcc00;">Training interrotto dall'utente</span>`;
            }
        }
    }
}

// Funzione per avviare il training di un modello e monitorare il progresso
async function startRealTraining(taskData) {
    try {
        // Converto i dati per il formato atteso dal backend
        const apiData = {
            models: taskData.model_type ? 
                   (Array.isArray(taskData.model_type) ? taskData.model_type : [taskData.model_type]) : [],
            timeframes: taskData.timeframe ? 
                   (Array.isArray(taskData.timeframe) ? taskData.timeframe : [taskData.timeframe]) : [],
            symbols: taskData.top_train_crypto || 30
        };
        
        console.log('Invio richiesta di training con dati:', apiData);
        
        // Chiamata API per avviare il training
        const response = await fetch('http://localhost:5000/api/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(apiData)
        });
        
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Errore nell'avvio del training: ${errorText}`);
        }
        
        const data = await response.json();
        
        console.log(`Training avviato con successo`);
        
        // Impostazione manuale dello stato di avanzamento
        activeTraining.progress = 5;
        activeTraining.phase = 'fetching_data';
        updateTerminalProgress();
        
        // Simula avanzamento del training
        startProgressSimulation();
        
        return true;
    } catch (error) {
        console.error("Errore nell'avvio del training:", error);
        
        // Reset dello stato di training in caso di errore
        activeTraining.isRunning = false;
        activeTraining.model = null;
        activeTraining.type = null;
        activeTraining.phase = 'error';
        
        // Aggiorna i pulsanti
        updateTrainingButtons();
        
        return false;
    }
}

// Funzione per monitorare il progresso del training
async function monitorTrainingProgress(taskId) {
    let checkCount = 0;
    const maxChecks = 1000; // Limitato per evitare loop infiniti
    
    while (activeTraining.isRunning && checkCount < maxChecks) {
        try {
            const response = await fetch(`http://localhost:8000/api/training-status/${taskId}`);
            
            if (!response.ok) {
                throw new Error(`Errore nel controllo dello stato: ${await response.text()}`);
            }
            
            const status = await response.json();
            
            // Aggiorna il progresso in base alla risposta
            if (status.status === 'running') {
                activeTraining.progress = status.progress || 0;
                activeTraining.phase = status.current_step || 'training';
                
                // Aggiorna il terminale di progresso
                updateTerminalProgress();
                
                // Attendi prima del prossimo controllo
                await new Promise(resolve => setTimeout(resolve, 2000));
                checkCount++;
            } else if (status.status === 'success' || status.status === 'completed') {
                // Training completato con successo
                activeTraining.progress = 100;
                activeTraining.phase = 'completed';
                updateTerminalProgress();
                
                // Simula il delay di completamento prima di aggiornare l'interfaccia
                setTimeout(() => {
                    if (activeTraining.isRunning) {
                        // Riporta lo stato a non in esecuzione
                        activeTraining.isRunning = false;
                        activeTraining.model = null;
                        activeTraining.type = null;
                        
                        // Aggiorna i pulsanti e ricarica lo stato dei modelli
                        updateTrainingButtons();
                        checkModelStatus();
                    }
                }, 3000);
                
                break;
            } else if (status.status === 'failure' || status.status === 'error') {
                // Training fallito
                activeTraining.progress = 0;
                activeTraining.phase = 'error';
                updateTerminalProgress();
                
                // Riporta lo stato a non in esecuzione
                setTimeout(() => {
                    activeTraining.isRunning = false;
                    activeTraining.model = null;
                    activeTraining.type = null;
                    updateTrainingButtons();
                }, 3000);
                
                break;
            }
        } catch (error) {
            console.error('Errore nel monitoraggio del training:', error);
            
            if (checkCount > 5) {
                // Aggiorna lo stato
                activeTraining.phase = 'error';
                updateTerminalProgress();
                
                // Attendi un po' prima di riprovare
                await new Promise(resolve => setTimeout(resolve, 5000));
            }
            
            checkCount++;
        }
    }
    
    // Se usciamo dal loop a causa del limite massimo di controlli
    if (checkCount >= maxChecks && activeTraining.isRunning) {
        console.warn('Raggiunto il numero massimo di controlli per il task');
        
        // Manteniamo lo stato di training attivo ma aggiorniamo la fase
        activeTraining.phase = 'unknown';
        updateTerminalProgress();
    }
}

// Funzione per addestrare un singolo modello
function trainSingleModel(modelType) {
    // Recupera i timeframe selezionati dal form
    const selectedTimeframes = [...document.querySelectorAll('.timeframe-select:checked')].map(cb => cb.value);
    if (selectedTimeframes.length === 0) {
        return;
    }
    
    // Se c'è già un training attivo, blocca
    if (activeTraining.isRunning) {
        return;
    }
    
    // Recupera gli altri parametri dal form
    const dataLimitDays = document.querySelector('input[name="data-limit-days"]:checked')?.value || '30';
    const trainCryptoCount = document.querySelector('input[name="train-crypto-count"]:checked')?.value || '30';
    
    const modelNames = {
        'lstm': 'LSTM',
        'rf': 'Random Forest',
        'xgb': 'XGBoost'
    };
    
    const modelName = modelNames[modelType] || modelType.toUpperCase();
    
    console.log(`Training per modello ${modelName}:`, {
        model: modelType,
        timeframes: selectedTimeframes,
        dataLimitDays,
        trainCryptoCount
    });
    
    // Imposta lo stato del training
    activeTraining.isRunning = true;
    activeTraining.model = modelType;
    activeTraining.type = 'single';
    activeTraining.progress = 0;
    activeTraining.phase = 'preparing';
    activeTraining.startTime = new Date();
    
    // Aggiorna i pulsanti per mostrare quale è attivo
    updateTrainingButtons();
    
    // Mostra un messaggio di avvio del training
    const statusCard = document.querySelector('.training-status-card');
    if (statusCard) {
        const infoAlert = document.getElementById('current-training-info');
        if (infoAlert) {
            infoAlert.className = 'alert alert-info mb-2';
            infoAlert.innerHTML = `
                <strong>Training in corso - ${modelName}:</strong>
                <div class="mt-2">
                    <div class="mb-1"><strong>Timeframes:</strong> ${selectedTimeframes.join(', ')}</div>
                    <div class="mb-1"><strong>Giorni di dati:</strong> ${dataLimitDays}</div>
                    <div><strong>Criptovalute:</strong> ${trainCryptoCount}</div>
                </div>
                <div class="mt-2 text-danger">
                    <small><i class="fas fa-info-circle me-1"></i> Puoi interrompere il training premendo nuovamente il pulsante</small>
                </div>
            `;
        }
    }
    
    // Aggiorna il terminale di progresso iniziale
    updateTerminalProgress();
    
    // Prepara i dati per la richiesta API
    const taskData = {
        model_type: modelType,
        timeframe: selectedTimeframes[0], // Per ora, accetta un solo timeframe
        data_limit_days: parseInt(dataLimitDays),
        top_train_crypto: parseInt(trainCryptoCount)
    };
    
    // Avvia il training reale
    startRealTraining(taskData);
}

// Funzione per addestrare il modello selezionato nei checkbox
function trainSelectedModel() {
    // Recupera il modello selezionato dai checkbox
    const selectedModelCheckbox = document.querySelector('.model-select:checked');
    if (!selectedModelCheckbox) {
        return;
    }
    
    const modelType = selectedModelCheckbox.value;
    const modelName = selectedModelCheckbox.closest('label').textContent.trim();
    
    // Recupera i timeframe selezionati dal form
    const selectedTimeframes = [...document.querySelectorAll('.timeframe-select:checked')].map(cb => cb.value);
    if (selectedTimeframes.length === 0) {
        return;
    }
    
    // Se c'è già un training attivo, blocca
    if (activeTraining.isRunning) {
        return;
    }
    
    // Recupera gli altri parametri dal form
    const dataLimitDays = document.querySelector('input[name="data-limit-days"]:checked')?.value || '30';
    const trainCryptoCount = document.querySelector('input[name="train-crypto-count"]:checked')?.value || '30';
    
    console.log(`Training per modello selezionato (${modelName}):`, {
        model: modelType,
        timeframes: selectedTimeframes,
        dataLimitDays,
        trainCryptoCount
    });
    
    // Imposta lo stato del training
    activeTraining.isRunning = true;
    activeTraining.model = modelType;
    activeTraining.type = 'selected';
    activeTraining.progress = 0;
    activeTraining.phase = 'preparing';
    activeTraining.startTime = new Date();
    
    // Aggiorna i pulsanti per mostrare quale è attivo
    updateTrainingButtons();
    
    // Mostra un messaggio di avvio del training
    const statusCard = document.querySelector('.training-status-card');
    if (statusCard) {
        const infoAlert = document.getElementById('current-training-info');
        if (infoAlert) {
            infoAlert.className = 'alert alert-info mb-2';
            infoAlert.innerHTML = `
                <strong>Training in corso - Modello selezionato (${modelName}):</strong>
                <div class="mt-2">
                    <div class="mb-1"><strong>Timeframes:</strong> ${selectedTimeframes.join(', ')}</div>
                    <div class="mb-1"><strong>Giorni di dati:</strong> ${dataLimitDays}</div>
                    <div><strong>Criptovalute:</strong> ${trainCryptoCount}</div>
                </div>
                <div class="mt-2 text-danger">
                    <small><i class="fas fa-info-circle me-1"></i> Puoi interrompere il training premendo nuovamente il pulsante</small>
                </div>
            `;
        }
    }
    
    // Aggiorna il terminale di progresso iniziale
    updateTerminalProgress();
    
    // Prepara i dati per la richiesta API
    const taskData = {
        model_type: modelType,
        timeframe: selectedTimeframes[0], // Per ora, accetta un solo timeframe
        data_limit_days: parseInt(dataLimitDays),
        top_train_crypto: parseInt(trainCryptoCount)
    };
    
    // Avvia il training reale
    startRealTraining(taskData);
}

// Funzione per addestrare i modelli selezionati
function trainSelectedModels(onlyMissing) {
    // Prima raccogliamo tutti i dati dal form
    const modelSelectForm = document.getElementById('model-training-form');
    if (!modelSelectForm) return;
    
    const formData = new FormData(modelSelectForm);
    
    // Recupera i modelli selezionati dal form (o usa tutti i modelli)
    let selectedModels = [...document.querySelectorAll('.model-select:checked')].map(cb => cb.value);
    if (selectedModels.length === 0) {
        return;
    }
    
    // Se c'è già un training attivo, blocca
    if (activeTraining.isRunning) {
        return;
    }
    
    // Recupera i timeframe selezionati o mancanti
    let selectedTimeframes = [];
    
    if (onlyMissing) {
        // Per ogni modello selezionato, aggiungi solo i timeframe mancanti
        selectedModels.forEach(model => {
            if (missingModels[model] && missingModels[model].length > 0) {
                // Se il training è solo per i mancanti, filtra per i timeframe mancanti
                selectedTimeframes = [...new Set([...selectedTimeframes, ...missingModels[model]])];
            }
        });
        
        if (selectedTimeframes.length === 0) {
            return;
        }
    } else {
        // Se addestriamo tutti, prendi i timeframe selezionati dal form
        selectedTimeframes = [...document.querySelectorAll('.timeframe-select:checked')].map(cb => cb.value);
        if (selectedTimeframes.length === 0) {
            return;
        }
    }
    
    // Recupera gli altri parametri dal form
    const dataLimitDays = document.querySelector('input[name="data-limit-days"]:checked')?.value || '30';
    const trainCryptoCount = document.querySelector('input[name="train-crypto-count"]:checked')?.value || '30';
    
    // Se ci sono più modelli o timeframe, avvisiamo l'utente che verranno creati più task
    const totalTasks = selectedModels.length * selectedTimeframes.length;
    
    // Imposta lo stato del training
    activeTraining.isRunning = true;
    activeTraining.model = selectedModels.join(',');
    activeTraining.type = onlyMissing ? 'missing' : 'all';
    activeTraining.progress = 0;
    activeTraining.phase = 'preparing';
    activeTraining.startTime = new Date();
    
    console.log('Training per:', {
        models: selectedModels,
        timeframes: selectedTimeframes,
        dataLimitDays,
        trainCryptoCount,
        onlyMissing
    });
    
    // Aggiorna i pulsanti per mostrare quale è attivo
    updateTrainingButtons();
    
    // Mostra un messaggio di avvio del training
    const statusCard = document.querySelector('.training-status-card');
    if (statusCard) {
        const infoAlert = document.getElementById('current-training-info');
        if (infoAlert) {
            infoAlert.className = 'alert alert-info mb-2';
            infoAlert.innerHTML = `
                <strong>Training in corso - ${onlyMissing ? 'Modelli mancanti' : 'Tutti i modelli'}:</strong>
                <div class="mt-2">
                    <div class="mb-1"><strong>Modelli:</strong> ${selectedModels.join(', ')}</div>
                    <div class="mb-1"><strong>Timeframes:</strong> ${selectedTimeframes.join(', ')}</div>
                    <div class="mb-1"><strong>Giorni di dati:</strong> ${dataLimitDays}</div>
                    <div><strong>Criptovalute:</strong> ${trainCryptoCount}</div>
                </div>
                <div class="mt-2 text-danger">
                    <small><i class="fas fa-info-circle me-1"></i> Puoi interrompere il training premendo nuovamente il pulsante</small>
                </div>
            `;
        }
    }
    
    // Aggiorna il terminale di progresso iniziale
    updateTerminalProgress();
    
    // Prepara i dati per la richiesta API - Adesso inviamo TUTTI i modelli e timeframe
    const taskData = {
        model_type: selectedModels,
        timeframe: selectedTimeframes,
        data_limit_days: parseInt(dataLimitDays),
        top_train_crypto: parseInt(trainCryptoCount)
    };
    
    // Avvia il training reale
    startRealTraining(taskData);
    
    // Mostra una notifica invece di un alert
    const tasksMessage = totalTasks > 1 ? ` (${totalTasks} modelli)` : '';
    showTrainingNotification('info', `Training ${onlyMissing ? 'dei modelli mancanti' : 'di tutti i modelli'} avviato${tasksMessage}`);
}

// Esegui la funzione quando il DOM è caricato
document.addEventListener('DOMContentLoaded', function() {
    console.log("Auto-status.js: DOM caricato, inizializzazione...");
    
    // Aggiungi event listener per i radio button dei giorni
    document.querySelectorAll('input[name="data-limit-days"]').forEach(radio => {
        radio.addEventListener('change', function() {
            document.getElementById('days-value').textContent = this.value;
        });
    });

    // Aggiungi event listener per i radio button delle criptovalute
    document.querySelectorAll('input[name="train-crypto-count"]').forEach(radio => {
        radio.addEventListener('change', function() {
            document.getElementById('train-crypto-counter').textContent = this.value;
        });
    });
    
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

        /* Stili per i pulsanti radio */
        .form-check-input:checked {
            background-color: #0d6efd;
            border-color: #0d6efd;
        }

        .form-check-input:checked + .form-check-label {
            color: #0d6efd;
            font-weight: bold;
        }

        /* Stili per i gruppi di pulsanti radio */
        .radio-group {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 15px;
        }

        .radio-option {
            position: relative;
            flex: 1;
            min-width: 120px;
            text-align: center;
        }

        .radio-option input[type="radio"] {
            position: absolute;
            opacity: 0;
            width: 0;
            height: 0;
        }

        .radio-option label {
            display: block;
            padding: 8px 15px;
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .radio-option input[type="radio"]:checked + label {
            background-color: #e7f1ff;
            border-color: #0d6efd;
            color: #0d6efd;
            font-weight: bold;
        }

        .radio-option label:hover {
            background-color: #e9ecef;
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
    
    // Rimuoviamo il pulsante di training esistente e lasciamo che venga ricreato dalla nostra logica
    const oldTrainButton = document.getElementById('train-model-btn');
    if (oldTrainButton) {
        const trainButtonContainer = oldTrainButton.parentElement;
        if (trainButtonContainer) {
            // Svuota il contenitore per essere riempito dalla nostra logica
            trainButtonContainer.innerHTML = '';
            
            // Aggiungiamo un contenitore di placeholder temporaneo
            const placeholder = document.createElement('div');
            placeholder.className = 'text-center text-muted';
            placeholder.innerHTML = '<div class="spinner-border spinner-border-sm me-2" role="status"></div> Verifica modelli in corso...';
            trainButtonContainer.appendChild(placeholder);
        }
    }
});

// Esegui la funzione immediatamente se il DOM è già caricato
if (document.readyState === 'interactive' || document.readyState === 'complete') {
    console.log("Auto-status.js: DOM già caricato, esecuzione immediata...");
    setTimeout(checkModelStatus, 1000);
} 