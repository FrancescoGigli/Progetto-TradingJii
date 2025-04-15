// ui.js - Gestisce tutte le funzioni dell'interfaccia utente

import { toggleBotStatus, saveApiKeys } from './api.js';

// Elemento del log
let logContainer;

// Esporto funzioni utilizzate dagli altri moduli
export {
    initializeUI,
    showSection,
    appendToLog,
    updateBotStatusUI,
    showAlert,
    initTooltips as initializeTooltips
};

// Inizializza elementi UI
function initializeUI() {
    logContainer = document.getElementById('log-container');
    setupEventListeners();
    setupNavigation();
}

// Funzione per configurare i listener degli eventi
function setupEventListeners() {
    // Listener per il pulsante di status del bot
    const botStatusBtn = document.getElementById('toggle-bot');
    if (botStatusBtn) {
        botStatusBtn.addEventListener('click', async function() {
            const currentStatus = this.getAttribute('data-status') === 'true';
            const newStatus = await toggleBotStatus(currentStatus);
            updateBotStatusUI(newStatus);
        });
    }
    
    // Listener per salvataggio delle chiavi API
    const apiKeysForm = document.getElementById('api-keys-form');
    if (apiKeysForm) {
        apiKeysForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            const apiKey = document.getElementById('api-key').value;
            const secretKey = document.getElementById('secret-key').value;
            
            if (await saveApiKeys(apiKey, secretKey)) {
                showAlert('success', 'Chiavi API salvate con successo!');
            } else {
                showAlert('danger', 'Errore nel salvataggio delle chiavi API.');
            }
        });
    }
    
    // Altri event listener specifici dell'UI
    setupModalButtons();
}

// Configura i pulsanti modali
function setupModalButtons() {
    // Chiusura modali
    document.querySelectorAll('[data-dismiss="modal"]').forEach(button => {
        button.addEventListener('click', function() {
            const modalId = this.closest('.modal').id;
            const modalElement = document.getElementById(modalId);
            const modalInstance = bootstrap.Modal.getInstance(modalElement);
            if (modalInstance) {
                modalInstance.hide();
            }
        });
    });
}

// Configura la navigazione
function setupNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Rimuovi la classe attiva da tutti i link
            navLinks.forEach(l => l.classList.remove('active'));
            
            // Aggiungi la classe attiva al link cliccato
            this.classList.add('active');
            
            // Mostra la sezione corrispondente
            // Estrai l'ID della sezione dall'attributo href (es. #dashboard -> dashboard)
            const href = this.getAttribute('href');
            if (href && href.startsWith('#')) {
                const sectionId = href.substring(1) + '-section';
                showSection(sectionId);
            }
        });
    });
    
    // Imposta la sezione iniziale (dashboard)
    navLinks[0].classList.add('active');
    showSection('dashboard-section');
}

// Mostra una sezione specifica e nasconde le altre
function showSection(sectionId) {
    // Nascondi tutte le sezioni
    document.querySelectorAll('.section').forEach(section => {
        section.classList.add('d-none');
    });
    
    // Mostra la sezione richiesta
    const targetSection = document.getElementById(sectionId);
    if (targetSection) {
        targetSection.classList.remove('d-none');
    } else {
        console.error(`Sezione ${sectionId} non trovata`);
    }
}

// Aggiunge un messaggio al log
function appendToLog(message) {
    if (!logContainer) {
        logContainer = document.getElementById('log-container');
    }
    
    if (logContainer) {
        const timestamp = new Date().toLocaleTimeString();
        const logEntry = document.createElement('div');
        logEntry.className = 'log-entry';
        logEntry.innerHTML = `<span class="log-time">[${timestamp}]</span> ${message}`;
        
        logContainer.appendChild(logEntry);
        
        // Auto-scroll al fondo
        logContainer.scrollTop = logContainer.scrollHeight;
    } else {
        console.log(`[${new Date().toLocaleTimeString()}] ${message}`);
    }
}

// Aggiorna l'UI in base allo stato del bot
function updateBotStatusUI(isRunning) {
    const botStatusElement = document.getElementById('bot-status');
    const botStatusBtn = document.getElementById('toggle-bot');
    
    if (botStatusElement) {
        botStatusElement.textContent = isRunning ? 'In esecuzione' : 'Fermo';
        botStatusElement.className = isRunning ? 
            'card-text status-online' : 
            'card-text status-offline';
    }
    
    if (botStatusBtn) {
        botStatusBtn.textContent = isRunning ? 'Ferma Bot' : 'Avvia Bot';
        botStatusBtn.className = isRunning ? 
            'btn btn-danger' : 
            'btn btn-success';
        botStatusBtn.setAttribute('data-status', isRunning);
    }
}

// Mostra un alert
function showAlert(type, message, timeout = 5000) {
    // Crea un elemento contenitore di alert se non esiste
    let alertsContainer = document.getElementById('alerts-container');
    if (!alertsContainer) {
        alertsContainer = document.createElement('div');
        alertsContainer.id = 'alerts-container';
        alertsContainer.className = 'position-fixed top-0 end-0 p-3';
        alertsContainer.style.zIndex = '1050';
        document.body.appendChild(alertsContainer);
    }
    
    const alert = document.createElement('div');
    alert.className = `alert alert-${type} alert-dismissible fade show`;
    alert.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    alertsContainer.appendChild(alert);
    
    // Auto-dismissibile
    if (timeout > 0) {
        setTimeout(() => {
            alert.classList.remove('show');
            setTimeout(() => alert.remove(), 150);
        }, timeout);
    }
}

// Inizializza i tooltip di Bootstrap
function initTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    if (tooltipTriggerList.length > 0) {
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
} 