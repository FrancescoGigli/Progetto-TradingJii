// api.js - Gestisce tutte le chiamate API

// Configurazione delle API
let apiKey;
let apiSecret;
let apiBaseUrl = 'http://localhost:8000';

// Funzione per configurare il servizio API
export function setupApiService(key, secret, baseUrl = 'http://localhost:8000') {
    apiKey = key;
    apiSecret = secret;
    apiBaseUrl = baseUrl;
}

// Funzione per aggiornare le credenziali API
export function updateApiCredentials(key, secret) {
    apiKey = key;
    apiSecret = secret;
}

// Funzione per effettuare richieste API
export async function makeApiRequest(endpoint, method = 'GET', data = null) {
    try {
        const headers = {
            'Content-Type': 'application/json',
            'api-key': apiKey,
            'api-secret': apiSecret
        };
        
        const options = {
            method,
            headers
        };
        
        if (data && (method === 'POST' || method === 'PUT')) {
            options.body = JSON.stringify(data);
        }

        // Log solo per operazioni importanti o POST/PUT
        if (method !== 'GET' || 
            endpoint === '/status' || 
            endpoint === '/balance' || 
            endpoint === '/health') {
            console.log(`[${new Date().toLocaleTimeString()}] Richiesta ${method} a ${endpoint}...`);
        }
        
        const response = await fetch(`${apiBaseUrl}${endpoint}`, options);
        
        if (!response.ok) {
            let errorMessage = `Errore ${response.status}: `;
            try {
                const errorData = await response.json();
                errorMessage += errorData.detail || 'Errore sconosciuto';
            } catch (e) {
                errorMessage += 'Impossibile leggere dettaglio errore';
            }
            
            throw new Error(errorMessage);
        }
        
        const result = await response.json();
        
        // Log solo per operazioni importanti o POST/PUT
        if (method !== 'GET' || endpoint === '/status') {
            console.log(`[${new Date().toLocaleTimeString()}] Risposta ricevuta da ${endpoint}`, result);
        }
        
        return result;
    } catch (error) {
        console.error(`[${new Date().toLocaleTimeString()}] Errore API: ${error.message}`);
        throw error; // Rilancia l'errore invece di restituire null
    }
}

// Funzione per testare la connessione
export async function testConnection() {
    console.log('Test di connessione in corso...');
    
    try {
        // Test dell'endpoint di health senza autenticazione
        const healthResponse = await fetch(`${apiBaseUrl}/health`);
        
        if (!healthResponse.ok) {
            console.log(`Connessione al server API: FALLITA - Errore ${healthResponse.status}`);
            return false;
        }
        
        const healthResult = await healthResponse.json();
        
        if (healthResult && healthResult.status === 'ok') {
            console.log('Connessione al server API: OK');
            
            // Test endpoint autenticato
            const statusResponse = await fetch(`${apiBaseUrl}/status`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    'api-key': apiKey,
                    'api-secret': apiSecret
                }
            });
            
            // Se riceviamo 401 Unauthorized, le chiavi API sono errate
            if (statusResponse.status === 401) {
                console.log('Autenticazione API: FALLITA - Chiavi API non valide');
                return false;
            }
            
            if (!statusResponse.ok) {
                console.log(`Autenticazione API: FALLITA - Errore ${statusResponse.status}`);
                return false;
            }
            
            const statusResult = await statusResponse.json();
            console.log('Autenticazione API: OK');
            return true;
        } else {
            console.log('Connessione al server API: FALLITA - Risposta non valida');
            return false;
        }
    } catch (error) {
        console.log(`Connessione al server API: FALLITA - ${error.message}`);
        console.error('Error in test connection:', error);
        return false;
    }
}

// Funzione per verificare lo stato del bot
export async function checkStatus() {
    const result = await makeApiRequest('/status');
    if (result) {
        return result.running;
    }
    return false;
}

// Funzione per verificare la salute del sistema
export async function checkHealth() {
    const healthStatusElement = document.getElementById('health-status');
    if (healthStatusElement) {
        healthStatusElement.textContent = 'In attesa...';
        healthStatusElement.className = 'card-text text-warning';
    }
    
    try {
        // Tentativo diretto senza passare attraverso makeApiRequest per evitare gestione degli errori
        const response = await fetch(`${apiBaseUrl}/health`);
        
        if (response.ok) {
            const result = await response.json();
            if (healthStatusElement) {
                healthStatusElement.textContent = 'Online';
                healthStatusElement.className = 'card-text status-online';
            }
            return true;
        }
    } catch (error) {
        console.error("Errore controllo salute:", error);
    }
    
    // Se arriviamo qui, c'è stato un errore o la risposta non è ok
    if (healthStatusElement) {
        healthStatusElement.textContent = 'Offline';
        healthStatusElement.className = 'card-text status-offline';
    }
    return false;
}

// Funzione per avviare/fermare il bot
export async function toggleBotStatus(currentStatus) {
    if (currentStatus) {
        const result = await makeApiRequest('/stop', 'POST');
        if (result) {
            console.log('Richiesta di arresto del bot inviata.');
            return false; // Il bot è stato fermato
        }
    } else {
        const result = await makeApiRequest('/start', 'POST');
        if (result) {
            console.log('Bot avviato con successo.');
            return true; // Il bot è stato avviato
        }
    }
    return currentStatus; // Mantieni lo stato attuale in caso di errore
}

// Funzione per salvare le API keys
export async function saveApiKeys(apiKeyValue, secretKeyValue) {    
    if (!apiKeyValue || !secretKeyValue) {
        console.log('Errore: Entrambe le chiavi sono richieste.');
        return false;
    }
    
    const result = await makeApiRequest('/set-keys', 'POST', { 
        api_key: apiKeyValue,
        secret_key: secretKeyValue
    });
    
    if (result) {
        updateApiCredentials(apiKeyValue, secretKeyValue);
        console.log('Chiavi API salvate con successo!');
        return true;
    }
    
    return false;
} 