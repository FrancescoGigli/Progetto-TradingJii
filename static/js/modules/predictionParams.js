/* predictionParams.js - Gestione parametri di trading per le predizioni.
   Questo file fornisce un'implementazione minima di compatibilità affinché
   predictionCore.js possa importare initializeTradeParamsHandlers() senza errori.
*/

// Funzione principale esportata
export function initializeTradeParamsHandlers() {
    try {
        // Legenda: leverage-range e margin-range sono slider/range input presenti nel DOM
        const leverageRange = document.getElementById('leverage-range');
        const leverageValueLabel = document.getElementById('leverage-value');
        const marginRange = document.getElementById('margin-range');
        const marginValueLabel = document.getElementById('margin-value');
        const topCryptoValue = document.getElementById('top-crypto-value');

        // Se gli elementi non esistono, esci silenziosamente (compatibilità)
        if (!leverageRange && !marginRange) return;

        // Aggiorna etichetta al cambio leva
        if (leverageRange) {
            const updateLeverage = () => {
                if (leverageValueLabel) leverageValueLabel.textContent = `${leverageRange.value}x`;
            };
            leverageRange.addEventListener('input', updateLeverage);
            updateLeverage();
        }

        // Aggiorna etichetta al cambio margine
        if (marginRange) {
            const updateMargin = () => {
                if (marginValueLabel) marginValueLabel.textContent = `${marginRange.value} USDT`;
            };
            marginRange.addEventListener('input', updateMargin);
            updateMargin();
        }
        
        // Inizializza i pulsanti radio per il numero di criptovalute
        const cryptoButtons = document.querySelectorAll('input[name="crypto-count"]');
        if (cryptoButtons.length && topCryptoValue) {
            // Imposta il valore iniziale
            const checkedButton = document.querySelector('input[name="crypto-count"]:checked');
            if (checkedButton) {
                topCryptoValue.textContent = checkedButton.value;
            }
            
            // Aggiorna il contatore quando cambia la selezione
            cryptoButtons.forEach(button => {
                button.addEventListener('change', function() {
                    if (this.checked) {
                        topCryptoValue.textContent = this.value;
                    }
                });
            });
        }

        console.log('initializeTradeParamsHandlers: handler inizializzati');
    } catch (error) {
        console.error('Errore in initializeTradeParamsHandlers:', error);
    }
} 