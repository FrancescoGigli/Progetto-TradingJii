// predictions.js - Punto di ingresso centrale per il sistema di predizioni
// Questo file è stato ridotto drasticamente e ora utilizza moduli più piccoli

// Importa le funzionalità dai moduli più specifici (senza autoStartDisabled)
import { initializePredictionsControl, setAutoStartDisabled, togglePredictions, checkBotStatus } from './predictionCore.js';
import { loadPredictions } from './predictionData.js';

// Esporta solo le funzioni principali necessarie ad altri moduli
export { 
    initializePredictionsControl,
    togglePredictions,
    loadPredictions,
    checkBotStatus
};

// Inizializza il controllo delle predizioni quando questo modulo viene caricato
document.addEventListener('DOMContentLoaded', async () => {
    console.log('Modulo predictions.js caricato - Inizializzazione del controller predizioni...');
    
    try {
        // Importa app.js in modo dinamico per evitare dipendenze circolari e problemi di inizializzazione
        const appModule = await import('../app.js');
        
        // Imposta lo stato di avvio automatico dopo aver caricato il modulo
        setAutoStartDisabled(appModule.autoStartDisabled);
        console.log(`Stato autoStartDisabled impostato a: ${appModule.autoStartDisabled}`);
    } catch (error) {
        console.error('Errore nel caricamento del modulo app.js:', error);
        // In caso di errore, usa il valore predefinito (false)
        setAutoStartDisabled(false);
        console.log('Usando valore predefinito autoStartDisabled = false');
    }
    
    // Inizializza il controllo delle predizioni solo dopo aver impostato autoStartDisabled
    initializePredictionsControl();
}); 