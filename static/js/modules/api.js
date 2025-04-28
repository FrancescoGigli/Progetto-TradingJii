// api.js - Gestisce tutte le chiamate API

// Configurazione delle API
let apiKey;
let apiSecret;
let apiBaseUrl = 'http://localhost:8000';
let authToken = null;

// Funzione per configurare il servizio API
export function setupApiService(key, secret, baseUrl = 'http://localhost:8000') {
    apiKey = key;
    apiSecret = secret;
    apiBaseUrl = baseUrl;
    // Ottieni il token JWT
    getAuthToken();
}

// Funzione per aggiornare le credenziali API
export function updateApiCredentials(key, secret) {
    apiKey = key;
    apiSecret = secret;
    // Ottieni un nuovo token JWT
    getAuthToken();
}

// Funzione per ottenere un token JWT
async function getAuthToken() {
    try {
        console.log('Richiesta token JWT in corso...');
        
        if (!apiKey || !apiSecret) {
            console.error('Credenziali API mancanti');
            return null;
        }
        
        const response = await fetch(`${apiBaseUrl}/auth/token`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                api_key: apiKey,
                secret_key: apiSecret
            })
        });
        
        if (!response.ok) {
            let errorMessage = `Errore ${response.status}: `;
            try {
                const errorData = await response.json();
                errorMessage += errorData.detail || 'Errore sconosciuto';
            } catch (e) {
                errorMessage += 'Impossibile leggere dettaglio errore';
            }
            console.error(`Errore nell'ottenimento del token JWT: ${errorMessage}`);
            return null;
        }
        
        const data = await response.json();
        authToken = data.token;
        console.log('Token JWT ottenuto con successo');
        return authToken;
    } catch (error) {
        console.error('Errore nella richiesta del token JWT:', error);
        return null;
    }
}

// Funzione per effettuare richieste API
export async function makeApiRequest(endpoint, method = 'GET', data = null) {
    try {
        // Se non abbiamo un token e abbiamo le credenziali, proviamo a ottenerlo
        if (!authToken && apiKey && apiSecret) {
            await getAuthToken();
            if (!authToken) {
                throw new Error('Impossibile ottenere il token di autenticazione');
            }
        }
        
        const headers = {
            'Content-Type': 'application/json'
        };
        
        // Aggiungi l'header di autorizzazione se abbiamo un token
        if (authToken) {
            headers['Authorization'] = `Bearer ${authToken}`;
        }
        
        const options = {
            method,
            headers,
            credentials: 'same-origin'
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
        
        let response = await fetch(`${apiBaseUrl}${endpoint}`, options);
        
        // Se riceviamo un 401 o 403, proviamo a rinnovare il token
        if ((response.status === 401 || response.status === 403) && apiKey && apiSecret) {
            console.log('Token non valido o scaduto, tentativo di rinnovo...');
            
            // Forza il rinnovo del token impostandolo a null
            authToken = null;
            await getAuthToken();
            
            if (!authToken) {
                throw new Error('Impossibile rinnovare il token di autenticazione');
            }
            
            // Aggiorna l'header con il nuovo token
            headers['Authorization'] = `Bearer ${authToken}`;
            options.headers = headers;
            
            // Riprova la richiesta con il nuovo token
            console.log('Riprovando la richiesta con il nuovo token...');
            response = await fetch(`${apiBaseUrl}${endpoint}`, options);
        }
        
        if (!response.ok) {
            let errorMessage = `Errore ${response.status}: `;
            try {
                const errorData = await response.json();
                errorMessage += errorData.detail || errorData.error || 'Errore sconosciuto';
            } catch (e) {
                errorMessage += await response.text() || 'Impossibile leggere dettaglio errore';
            }
            
            throw new Error(errorMessage);
        }
        
        const result = await response.json();
        
        // Log solo per operazioni importanti o POST/PUT
        if (method !== 'GET' || endpoint === '/status') {
            console.log(`[${new Date().toLocaleTimeString()}] Risposta ricevuta da ${endpoint}`);
        }
        
        return result;
    } catch (error) {
        console.error(`[${new Date().toLocaleTimeString()}] Errore API:`, error.message);
        
        // Se l'errore è relativo all'autenticazione, aggiorna l'UI
        if (error.message.includes('401') || error.message.includes('403') || 
            error.message.includes('token') || error.message.includes('autenticazione')) {
            console.error('Errore di autenticazione. Verifica le credenziali API.');
            // Aggiorna lo stato dell'UI se necessario
            const statusElement = document.getElementById('connection-status');
            if (statusElement) {
                statusElement.textContent = 'Non autenticato';
                statusElement.className = 'status-error';
            }
        }
        
        return null;
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
            
            // Se abbiamo credenziali, proviamo a ottenere un token
            if (apiKey && apiSecret) {
                await getAuthToken();
                
                // Test endpoint autenticato
                const statusResponse = await fetch(`${apiBaseUrl}/status`, {
                    method: 'GET',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${authToken}`
                    }
                });
                
                if (statusResponse.ok) {
                    console.log('Autenticazione API: OK');
                    return true;
                } else {
                    console.log('Autenticazione API: FALLITA');
                    return false;
                }
            } else {
                console.log('Nessuna credenziale API configurata');
                return false;
            }
        } else {
            console.log('Connessione al server API: FALLITA - Risposta non valida');
            return false;
        }
    } catch (error) {
        console.error('Errore durante il test di connessione:', error);
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
    try {
        // Aggiorna le credenziali locali
        updateApiCredentials(apiKeyValue, secretKeyValue);
        
        // Ottieni un token JWT
        await getAuthToken();
        
        // Salva le chiavi sul server
        const response = await makeApiRequest('/set-keys', 'POST', {
            api_key: apiKeyValue,
            secret_key: secretKeyValue
        });
        
        if (response && response.status === 'Chiavi salvate.') {
            console.log('Chiavi API salvate con successo');
            return true;
        } else {
            console.error('Errore nel salvataggio delle chiavi API');
            return false;
        }
    } catch (error) {
        console.error('Errore nel salvataggio delle chiavi API:', error);
        return false;
    }
} 