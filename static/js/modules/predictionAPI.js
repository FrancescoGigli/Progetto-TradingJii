// predictionAPI.js - Gestione delle chiamate API per le previsioni

import { showNotification } from './ui.js';

// Costanti e configurazioni
const API_ENDPOINT = '/api/predictions';
const DEFAULT_TIMEOUT = 30000; // 30 secondi di timeout per le chiamate API

/**
 * Recupera le previsioni attive in base ai filtri specificati
 * @param {Array<string>} symbols - Elenco di simboli da filtrare (opzionale)
 * @param {Array<string>} timeframes - Elenco di timeframe da filtrare (opzionale)
 * @param {string} model - Modello specifico da filtrare (opzionale)
 * @returns {Promise<Object>} - Le previsioni attive
 */
export async function fetchActivePredictions(symbols, timeframes, model) {
    try {
        // Prepara i parametri della query
        const params = new URLSearchParams();
        
        if (symbols && symbols.length > 0) {
            params.append('symbols', symbols.join(','));
        }
        
        if (timeframes && timeframes.length > 0) {
            params.append('timeframes', timeframes.join(','));
        }
        
        if (model) {
            params.append('model', model);
        }
        
        // Prepara la richiesta con timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), DEFAULT_TIMEOUT);
        
        // Effettua la chiamata API
        const response = await fetch(`${API_ENDPOINT}/active?${params.toString()}`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            signal: controller.signal
        });
        
        // Pulisci il timeout
        clearTimeout(timeoutId);
        
        // Gestisci la risposta
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.message || `Errore ${response.status}: ${response.statusText}`);
        }
        
        return await response.json();
        
    } catch (error) {
        // Gestisci gli errori specifici di timeout
        if (error.name === 'AbortError') {
            showNotification('error', 'La richiesta delle previsioni è scaduta. Riprova più tardi.', true);
            return { predictions: [], error: 'Timeout' };
        }
        
        console.error('Errore nel recupero delle previsioni attive:', error);
        showNotification('error', `Errore nel recupero delle previsioni: ${error.message}`, true);
        return { predictions: [], error: error.message };
    }
}

/**
 * Avvia le previsioni per i simboli, modello e timeframe specificati
 * @param {Array<string>} symbols - I simboli per cui avviare le previsioni
 * @param {string} model - Il modello da utilizzare
 * @param {string} timeframe - Il timeframe da utilizzare
 * @returns {Promise<Object>} - Il risultato dell'operazione
 */
export async function startPredictions(symbols, model, timeframe) {
    try {
        // Verifica i parametri obbligatori
        if (!symbols || !Array.isArray(symbols) || symbols.length === 0) {
            throw new Error('Specificare almeno un simbolo per avviare le previsioni');
        }
        
        if (!model) {
            throw new Error('Specificare un modello per avviare le previsioni');
        }
        
        if (!timeframe) {
            throw new Error('Specificare un timeframe per avviare le previsioni');
        }
        
        // Prepara la richiesta con timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), DEFAULT_TIMEOUT);
        
        // Effettua la chiamata API
        const response = await fetch(`${API_ENDPOINT}/start`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify({
                symbols,
                model,
                timeframe
            }),
            signal: controller.signal
        });
        
        // Pulisci il timeout
        clearTimeout(timeoutId);
        
        // Gestisci la risposta
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.message || `Errore ${response.status}: ${response.statusText}`);
        }
        
        const result = await response.json();
        return { ...result, success: true };
        
    } catch (error) {
        // Gestisci gli errori specifici di timeout
        if (error.name === 'AbortError') {
            showNotification('error', 'La richiesta di avvio delle previsioni è scaduta. Riprova più tardi.', true);
            return { success: false, error: 'Timeout' };
        }
        
        console.error('Errore nell\'avvio delle previsioni:', error);
        showNotification('error', `Errore nell'avvio delle previsioni: ${error.message}`, true);
        return { success: false, error: error.message };
    }
}

/**
 * Ferma le previsioni per i simboli, modello e timeframe specificati
 * @param {Array<string>} symbols - I simboli per cui fermare le previsioni (opzionale, se non specificato ferma tutte)
 * @param {string} model - Il modello specifico da fermare (opzionale)
 * @param {string} timeframe - Il timeframe specifico da fermare (opzionale)
 * @returns {Promise<Object>} - Il risultato dell'operazione
 */
export async function stopPredictions(symbols, model, timeframe) {
    try {
        // Prepara il corpo della richiesta con i parametri disponibili
        const requestBody = {};
        
        if (symbols && Array.isArray(symbols) && symbols.length > 0) {
            requestBody.symbols = symbols;
        }
        
        if (model) {
            requestBody.model = model;
        }
        
        if (timeframe) {
            requestBody.timeframe = timeframe;
        }
        
        // Prepara la richiesta con timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), DEFAULT_TIMEOUT);
        
        // Effettua la chiamata API
        const response = await fetch(`${API_ENDPOINT}/stop`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify(requestBody),
            signal: controller.signal
        });
        
        // Pulisci il timeout
        clearTimeout(timeoutId);
        
        // Gestisci la risposta
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.message || `Errore ${response.status}: ${response.statusText}`);
        }
        
        const result = await response.json();
        return { ...result, success: true };
        
    } catch (error) {
        // Gestisci gli errori specifici di timeout
        if (error.name === 'AbortError') {
            showNotification('error', 'La richiesta di arresto delle previsioni è scaduta. Riprova più tardi.', true);
            return { success: false, error: 'Timeout' };
        }
        
        console.error('Errore nell\'arresto delle previsioni:', error);
        showNotification('error', `Errore nell'arresto delle previsioni: ${error.message}`, true);
        return { success: false, error: error.message };
    }
}

/**
 * Recupera lo stato attuale delle previsioni
 * @returns {Promise<Object>} - Lo stato delle previsioni
 */
export async function getPredictionStatus() {
    try {
        // Prepara la richiesta con timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), DEFAULT_TIMEOUT);
        
        // Effettua la chiamata API
        const response = await fetch(`${API_ENDPOINT}/status`, {
            method: 'GET',
            headers: {
                'Accept': 'application/json'
            },
            signal: controller.signal
        });
        
        // Pulisci il timeout
        clearTimeout(timeoutId);
        
        // Gestisci la risposta
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.message || `Errore ${response.status}: ${response.statusText}`);
        }
        
        return await response.json();
        
    } catch (error) {
        // Gestisci gli errori specifici di timeout
        if (error.name === 'AbortError') {
            console.error('Timeout durante il recupero dello stato delle previsioni');
            return { active: 0, status: 'error', error: 'Timeout' };
        }
        
        console.error('Errore nel recupero dello stato delle previsioni:', error);
        return { active: 0, status: 'error', error: error.message };
    }
}

/**
 * Recupera la cronologia delle previsioni per un simbolo, modello e timeframe specifici
 * @param {string} symbol - Il simbolo di cui recuperare la cronologia
 * @param {string} model - Il modello di cui recuperare la cronologia
 * @param {string} timeframe - Il timeframe di cui recuperare la cronologia
 * @param {number} limit - Numero massimo di risultati da recuperare (opzionale)
 * @returns {Promise<Object>} - La cronologia delle previsioni
 */
export async function fetchPredictionHistory(symbol, model, timeframe, limit = 10) {
    try {
        // Verifica i parametri obbligatori
        if (!symbol) {
            throw new Error('Specificare un simbolo per recuperare la cronologia');
        }
        
        if (!model) {
            throw new Error('Specificare un modello per recuperare la cronologia');
        }
        
        if (!timeframe) {
            throw new Error('Specificare un timeframe per recuperare la cronologia');
        }
        
        // Prepara i parametri della query
        const params = new URLSearchParams({
            symbol,
            model,
            timeframe,
            limit: limit.toString()
        });
        
        // Prepara la richiesta con timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), DEFAULT_TIMEOUT);
        
        // Effettua la chiamata API
        const response = await fetch(`${API_ENDPOINT}/history?${params.toString()}`, {
            method: 'GET',
            headers: {
                'Accept': 'application/json'
            },
            signal: controller.signal
        });
        
        // Pulisci il timeout
        clearTimeout(timeoutId);
        
        // Gestisci la risposta
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.message || `Errore ${response.status}: ${response.statusText}`);
        }
        
        return await response.json();
        
    } catch (error) {
        // Gestisci gli errori specifici di timeout
        if (error.name === 'AbortError') {
            console.error('Timeout durante il recupero della cronologia delle previsioni');
            return { history: [], error: 'Timeout' };
        }
        
        console.error('Errore nel recupero della cronologia delle previsioni:', error);
        return { history: [], error: error.message };
    }
}

/**
 * Recupera le statistiche per le previsioni di un simbolo, modello e timeframe specifici
 * @param {string} symbol - Il simbolo di cui recuperare le statistiche
 * @param {string} model - Il modello di cui recuperare le statistiche
 * @param {string} timeframe - Il timeframe di cui recuperare le statistiche
 * @returns {Promise<Object>} - Le statistiche delle previsioni
 */
export async function fetchPredictionStats(symbol, model, timeframe) {
    try {
        // Verifica i parametri obbligatori
        if (!symbol) {
            throw new Error('Specificare un simbolo per recuperare le statistiche');
        }
        
        if (!model) {
            throw new Error('Specificare un modello per recuperare le statistiche');
        }
        
        if (!timeframe) {
            throw new Error('Specificare un timeframe per recuperare le statistiche');
        }
        
        // Prepara i parametri della query
        const params = new URLSearchParams({
            symbol,
            model,
            timeframe
        });
        
        // Prepara la richiesta con timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), DEFAULT_TIMEOUT);
        
        // Effettua la chiamata API
        const response = await fetch(`${API_ENDPOINT}/stats?${params.toString()}`, {
            method: 'GET',
            headers: {
                'Accept': 'application/json'
            },
            signal: controller.signal
        });
        
        // Pulisci il timeout
        clearTimeout(timeoutId);
        
        // Gestisci la risposta
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.message || `Errore ${response.status}: ${response.statusText}`);
        }
        
        return await response.json();
        
    } catch (error) {
        // Gestisci gli errori specifici di timeout
        if (error.name === 'AbortError') {
            console.error('Timeout durante il recupero delle statistiche delle previsioni');
            return { stats: {}, error: 'Timeout' };
        }
        
        console.error('Errore nel recupero delle statistiche delle previsioni:', error);
        return { stats: {}, error: error.message };
    }
}

/**
 * Salva le impostazioni delle previsioni
 * @param {Object} settings - Le impostazioni da salvare
 * @returns {Promise<Object>} - Il risultato dell'operazione
 */
export async function savePredictionSettings(settings) {
    try {
        // Verifica i parametri obbligatori
        if (!settings || typeof settings !== 'object') {
            throw new Error('Specificare le impostazioni da salvare');
        }
        
        // Prepara la richiesta con timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), DEFAULT_TIMEOUT);
        
        // Effettua la chiamata API
        const response = await fetch(`${API_ENDPOINT}/settings`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify(settings),
            signal: controller.signal
        });
        
        // Pulisci il timeout
        clearTimeout(timeoutId);
        
        // Gestisci la risposta
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.message || `Errore ${response.status}: ${response.statusText}`);
        }
        
        const result = await response.json();
        showNotification('success', 'Impostazioni delle previsioni salvate con successo', true);
        return { ...result, success: true };
        
    } catch (error) {
        // Gestisci gli errori specifici di timeout
        if (error.name === 'AbortError') {
            showNotification('error', 'La richiesta di salvataggio delle impostazioni è scaduta. Riprova più tardi.', true);
            return { success: false, error: 'Timeout' };
        }
        
        console.error('Errore nel salvataggio delle impostazioni delle previsioni:', error);
        showNotification('error', `Errore nel salvataggio delle impostazioni: ${error.message}`, true);
        return { success: false, error: error.message };
    }
}

/**
 * Carica le impostazioni delle previsioni
 * @returns {Promise<Object>} - Le impostazioni caricate
 */
export async function loadPredictionSettings() {
    try {
        // Prepara la richiesta con timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), DEFAULT_TIMEOUT);
        
        // Effettua la chiamata API
        const response = await fetch(`${API_ENDPOINT}/settings`, {
            method: 'GET',
            headers: {
                'Accept': 'application/json'
            },
            signal: controller.signal
        });
        
        // Pulisci il timeout
        clearTimeout(timeoutId);
        
        // Gestisci la risposta
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.message || `Errore ${response.status}: ${response.statusText}`);
        }
        
        return await response.json();
        
    } catch (error) {
        // Gestisci gli errori specifici di timeout
        if (error.name === 'AbortError') {
            console.error('Timeout durante il caricamento delle impostazioni delle previsioni');
            showNotification('error', 'Timeout durante il caricamento delle impostazioni. Riprova più tardi.', true);
            return { settings: {}, error: 'Timeout' };
        }
        
        console.error('Errore nel caricamento delle impostazioni delle previsioni:', error);
        showNotification('error', `Errore nel caricamento delle impostazioni: ${error.message}`, true);
        return { settings: {}, error: error.message };
    }
} 