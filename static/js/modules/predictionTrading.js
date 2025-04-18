// predictionTrading.js - Gestione delle funzionalità di trading
import { showNotification } from './ui.js';

// Funzione per eseguire un'operazione di trading
export async function executeTrade(symbol, direction) {
    try {
        // Mostra notifica di operazione in corso
        showNotification('info', `Esecuzione ${direction.toLowerCase()} su ${symbol} in corso...`, false);
        
        // Prepara i dati del trade
        const tradeData = {
            symbol: symbol,
            direction: direction,
            timestamp: new Date().toISOString()
        };
        
        // Invia la richiesta di trade al backend
        const response = await fetch('/api/execute_trade', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(tradeData)
        });
        
        // Gestisci la risposta
        const result = await response.json();
        
        if (response.ok) {
            // Trade eseguito con successo
            showNotification('success', `${direction} su ${symbol} eseguito con successo!`, true);
            
            // Aggiorna la lista dei trade
            updateTradesList(result.trade);
            
            // Aggiorna il bilancio se disponibile
            if (result.balance) {
                updateBalanceDisplay(result.balance);
            }
            
            return true;
        } else {
            // Errore nell'esecuzione del trade
            showNotification('error', `Errore: ${result.error || 'Impossibile eseguire il trade'}`, true);
            console.error('Errore nell\'esecuzione del trade:', result.error);
            return false;
        }
    } catch (error) {
        // Errore nella chiamata API
        showNotification('error', `Errore di connessione: ${error.message}`, true);
        console.error('Errore nella richiesta di trade:', error);
        return false;
    }
}

// Funzione per aggiornare la lista dei trade
function updateTradesList(trade) {
    const tradesContainer = document.getElementById('trades-list');
    if (!tradesContainer) return;
    
    // Crea un nuovo elemento per il trade
    const tradeItem = document.createElement('div');
    tradeItem.className = `alert ${trade.direction === 'Buy' ? 'alert-success' : 'alert-danger'} d-flex justify-content-between align-items-center mb-2`;
    
    // Formatta la data
    const tradeDate = new Date(trade.timestamp);
    const formattedDate = tradeDate.toLocaleString();
    
    // Imposta il contenuto
    tradeItem.innerHTML = `
        <div>
            <strong>${trade.symbol}</strong>: 
            <span class="badge ${trade.direction === 'Buy' ? 'bg-success' : 'bg-danger'}">
                ${trade.direction === 'Buy' ? 'LONG' : 'SHORT'}
            </span>
            <small class="ms-2 text-muted">${formattedDate}</small>
        </div>
        <div>
            ${trade.price ? `<strong>Prezzo:</strong> ${trade.price}` : ''}
            ${trade.quantity ? `<strong>Quantità:</strong> ${trade.quantity}` : ''}
        </div>
    `;
    
    // Aggiungi alla lista
    tradesContainer.prepend(tradeItem);
    
    // Limita il numero di trade visualizzati a 10
    const tradeElements = tradesContainer.querySelectorAll('.alert');
    if (tradeElements.length > 10) {
        tradesContainer.removeChild(tradeElements[tradeElements.length - 1]);
    }
}

// Funzione per aggiornare il bilancio
function updateBalanceDisplay(balance) {
    const balanceElement = document.getElementById('account-balance');
    if (!balanceElement) return;
    
    // Formatta il valore del bilancio
    const formattedBalance = typeof balance === 'number' 
        ? balance.toFixed(2) 
        : balance;
    
    // Aggiorna il display
    balanceElement.textContent = formattedBalance;
    
    // Aggiungi una piccola animazione
    balanceElement.classList.add('highlight-value');
    setTimeout(() => {
        balanceElement.classList.remove('highlight-value');
    }, 1000);
}

// Funzione per caricare la cronologia dei trade
export async function loadTradeHistory() {
    try {
        const response = await fetch('/api/trade_history');
        
        if (!response.ok) {
            console.error('Errore nel caricamento della cronologia dei trade:', response.statusText);
            return;
        }
        
        const trades = await response.json();
        
        // Aggiorna l'interfaccia con i trade storici
        displayTradeHistory(trades);
        
    } catch (error) {
        console.error('Errore nella richiesta della cronologia dei trade:', error);
    }
}

// Funzione per visualizzare la cronologia dei trade
function displayTradeHistory(trades) {
    const historyContainer = document.getElementById('trade-history-container');
    if (!historyContainer) return;
    
    // Pulisci il contenitore
    historyContainer.innerHTML = '';
    
    if (trades.length === 0) {
        historyContainer.innerHTML = `
            <div class="alert alert-info">
                <i class="fas fa-info-circle me-2"></i>
                Nessun trade eseguito finora.
            </div>
        `;
        return;
    }
    
    // Crea una tabella per i trade
    const table = document.createElement('table');
    table.className = 'table table-sm table-hover';
    
    // Aggiungi l'intestazione
    table.innerHTML = `
        <thead>
            <tr>
                <th>Data</th>
                <th>Simbolo</th>
                <th>Operazione</th>
                <th>Prezzo</th>
                <th>Quantità</th>
                <th>P/L</th>
            </tr>
        </thead>
        <tbody id="trade-history-body"></tbody>
    `;
    
    historyContainer.appendChild(table);
    
    const tableBody = document.getElementById('trade-history-body');
    
    // Aggiungi le righe per ogni trade
    trades.forEach(trade => {
        const row = document.createElement('tr');
        
        // Assegna una classe in base alla direzione
        if (trade.direction === 'Buy') {
            row.classList.add('table-success', 'text-success');
        } else {
            row.classList.add('table-danger', 'text-danger');
        }
        
        // Formatta la data
        const tradeDate = new Date(trade.timestamp);
        const formattedDate = tradeDate.toLocaleString();
        
        // Formatta il P/L (profit/loss)
        let profitLoss = '';
        if (trade.profit_loss) {
            const numValue = parseFloat(trade.profit_loss);
            profitLoss = `
                <span class="${numValue >= 0 ? 'text-success' : 'text-danger'}">
                    ${numValue >= 0 ? '+' : ''}${numValue.toFixed(2)}
                </span>`;
        }
        
        // Popola la riga
        row.innerHTML = `
            <td><small>${formattedDate}</small></td>
            <td><strong>${trade.symbol}</strong></td>
            <td>
                <span class="badge ${trade.direction === 'Buy' ? 'bg-success' : 'bg-danger'}">
                    ${trade.direction === 'Buy' ? 'LONG' : 'SHORT'}
                </span>
            </td>
            <td>${trade.price || '-'}</td>
            <td>${trade.quantity || '-'}</td>
            <td>${profitLoss || '-'}</td>
        `;
        
        tableBody.appendChild(row);
    });
}

// Funzione per aggiornare il riepilogo del trading
export async function updateTradingSummary() {
    try {
        const response = await fetch('/api/trading_summary');
        
        if (!response.ok) {
            console.error('Errore nel caricamento del riepilogo del trading:', response.statusText);
            return;
        }
        
        const summary = await response.json();
        
        // Aggiorna l'interfaccia con il riepilogo
        displayTradingSummary(summary);
        
    } catch (error) {
        console.error('Errore nella richiesta del riepilogo del trading:', error);
    }
}

// Funzione per visualizzare il riepilogo del trading
function displayTradingSummary(summary) {
    // Aggiorna i contatori di successo
    updateSummaryCounter('total-trades', summary.total_trades || 0);
    updateSummaryCounter('winning-trades', summary.winning_trades || 0);
    updateSummaryCounter('losing-trades', summary.losing_trades || 0);
    
    // Aggiorna i guadagni totali
    if (summary.total_profit !== undefined) {
        const totalProfitElement = document.getElementById('total-profit');
        if (totalProfitElement) {
            const numValue = parseFloat(summary.total_profit);
            totalProfitElement.textContent = numValue.toFixed(2);
            totalProfitElement.className = numValue >= 0 ? 'text-success' : 'text-danger';
        }
    }
    
    // Aggiorna il win rate
    if (summary.win_rate !== undefined) {
        const winRateElement = document.getElementById('win-rate');
        if (winRateElement) {
            winRateElement.textContent = `${summary.win_rate.toFixed(1)}%`;
        }
        
        // Aggiorna anche la progress bar del win rate
        const winRateBar = document.getElementById('win-rate-bar');
        if (winRateBar) {
            winRateBar.style.width = `${summary.win_rate}%`;
            
            // Cambia il colore in base al win rate
            if (summary.win_rate >= 60) {
                winRateBar.className = 'progress-bar bg-success';
            } else if (summary.win_rate >= 40) {
                winRateBar.className = 'progress-bar bg-warning';
            } else {
                winRateBar.className = 'progress-bar bg-danger';
            }
        }
    }
    
    // Aggiorna i simboli di maggior successo
    if (summary.best_symbols && Array.isArray(summary.best_symbols)) {
        const bestSymbolsElement = document.getElementById('best-symbols');
        if (bestSymbolsElement) {
            if (summary.best_symbols.length > 0) {
                bestSymbolsElement.innerHTML = summary.best_symbols
                    .map(symb => `<span class="badge bg-success me-1">${symb}</span>`)
                    .join(' ');
            } else {
                bestSymbolsElement.innerHTML = '<small class="text-muted">Nessun dato disponibile</small>';
            }
        }
    }
}

// Funzione per aggiornare un contatore nel riepilogo
function updateSummaryCounter(elementId, value) {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    // Mostra il valore precedente
    const currentValue = parseInt(element.textContent) || 0;
    
    // Se il valore è cambiato, applica un'animazione
    if (currentValue !== value) {
        element.textContent = value;
        element.classList.add('highlight-value');
        setTimeout(() => {
            element.classList.remove('highlight-value');
        }, 1000);
    }
} 