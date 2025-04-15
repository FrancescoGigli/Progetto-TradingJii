// app.js - File principale che inizializza l'applicazione
// Costanti
const API_BASE_URL = 'http://localhost:8000';
const LOCAL_STORAGE_API_KEY = 'trae_api_key';
const LOCAL_STORAGE_API_SECRET = 'trae_api_secret';
const DEFAULT_API_KEY = 'hRI4q8EB3ryaURdyBm';
const DEFAULT_API_SECRET = 'xQpYxVtEinsD6yqa84PGbYVsgYrT9O3k0MRf';

// Variabili globali per lo stato dell'applicazione
let botRunning = false;
let apiKey = localStorage.getItem(LOCAL_STORAGE_API_KEY) || DEFAULT_API_KEY;
let apiSecret = localStorage.getItem(LOCAL_STORAGE_API_SECRET) || DEFAULT_API_SECRET;

// Elementi DOM principali
const botStatusBtn = document.getElementById('botStatusBtn');
const logContent = document.getElementById('log-content');
const navLinks = document.querySelectorAll('.nav-link');
const sections = document.querySelectorAll('.section');

// Esporta variabili globali per essere usate in altri moduli
export {
    API_BASE_URL,
    LOCAL_STORAGE_API_KEY,
    LOCAL_STORAGE_API_SECRET,
    DEFAULT_API_KEY,
    DEFAULT_API_SECRET,
    botStatusBtn,
    logContent,
    navLinks,
    sections,
    botRunning,
    apiKey,
    apiSecret
};

// Importazione dei moduli
import { initializeUI, appendToLog, showAlert } from './modules/ui.js';
import { testConnection, setupApiService } from './modules/api.js';
import { loadBalance, loadPositions, loadOpenOrders, setupDashboardEventListeners } from './modules/dashboard.js';
import { initializePredictionsControl } from './modules/predictions.js';
import { loadChartSymbols } from './modules/charts.js';
import { handleTrainingFormSubmit, setupModelsEventListeners } from './modules/models.js';

document.addEventListener('DOMContentLoaded', async () => {
    try {
        console.log('Inizializzazione applicazione...');
        
        // Mostra il container dei log
        showLogContainer();
        
        // Configura il servizio API con le chiavi di default
        setupApiService(apiKey, apiSecret, API_BASE_URL);
        
        // Inizializza l'interfaccia utente
        initializeUI();
        
        // Inizializza gli event listener della dashboard
        setupDashboardEventListeners();
        
        // Inizializza i moduli solo se gli elementi necessari esistono
        safeInitialize();
        
        // Log di avvio applicazione
        appendToLog('Applicazione avviata con successo');
        
        // Test connessione API
        setTimeout(async () => {
            try {
                const connected = await testConnection();
                if (connected) {
                    appendToLog('Connessione alle API stabilita');
                    updateStatusElement('health-status', 'Online', 'text-success');
                } else {
                    appendToLog('Impossibile connettersi alle API. Verifica le credenziali.');
                    updateStatusElement('health-status', 'Offline', 'text-danger');
                }
            } catch (error) {
                appendToLog(`Errore connessione API: ${error.message}`);
                showAlert('danger', `Errore connessione API: ${error.message}`);
                updateStatusElement('health-status', 'Errore', 'text-danger');
            }
        }, 1000);
        
    } catch (error) {
        console.error('Errore durante l\'inizializzazione:', error);
        appendToLog(`Errore durante l'inizializzazione: ${error.message}`);
        showAlert('danger', `Errore durante l'inizializzazione: ${error.message}`);
    }
});

// Funzione per aggiornare lo stato del bot
export function updateBotStatus(status) {
    botRunning = status;
    updateBotStatusDisplay();
}

// Funzione per aggiornare la visualizzazione dello stato del bot
function updateBotStatusDisplay() {
    updateStatusElement('bot-status', botRunning ? 'In esecuzione' : 'Fermo', botRunning ? 'text-success' : 'text-danger');
}

// Funzione per aggiornare un elemento di stato in modo sicuro
function updateStatusElement(id, text, className) {
    const element = document.getElementById(id);
    if (element) {
        element.textContent = text;
        if (className) {
            // Rimuovi tutte le classi text-*
            element.classList.remove('text-success', 'text-danger', 'text-warning', 'text-info');
            element.classList.add(className);
        }
    }
}

// Funzione per mostrare il log container
function showLogContainer() {
    const logContainer = document.getElementById('log-container');
    if (logContainer) {
        logContainer.classList.remove('d-none');
    }
}

// Funzione per inizializzare moduli in modo sicuro
function safeInitialize() {
    try {
        // Inizializza il controllo delle predizioni se gli elementi esistono
        const predictionControlBtn = document.getElementById('predictions-control-btn');
        if (predictionControlBtn) {
            initializePredictionsControl();
        }
        
        // Inizializza i simboli del grafico se gli elementi esistono
        const chartContainer = document.getElementById('position-chart');
        if (chartContainer) {
            loadChartSymbols();
        }
        
        // Inizializza gli event listener per i modelli se gli elementi esistono
        const modelsSection = document.getElementById('models-section');
        if (modelsSection) {
            setupModelsEventListeners();
        }
        
        // Carica i dati iniziali della dashboard se gli elementi esistono
        const totalWallet = document.getElementById('total-wallet');
        if (totalWallet) {
            loadBalance();
        }
        
        const positions = document.getElementById('positions');
        if (positions) {
            loadPositions();
        }
        
        const ordersTable = document.getElementById('open-orders-table');
        if (ordersTable) {
            loadOpenOrders();
        }
    } catch (error) {
        console.error('Errore inizializzazione moduli:', error);
        appendToLog(`Errore inizializzazione moduli: ${error.message}`);
    }
}