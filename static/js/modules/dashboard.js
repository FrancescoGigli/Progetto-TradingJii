// dashboard.js - Gestisce tutte le funzionalità della dashboard
import { makeApiRequest } from './api.js';
import { appendToLog } from './ui.js';

// Funzione per caricare i dati del bilancio
export async function loadBalance() {
    try {
        const result = await makeApiRequest('/balance');
        if (result) {
            // I campi rinominati
            const totalWallet = document.getElementById('total-wallet');
            const availableBalance = document.getElementById('available-balance');
            const usedBalance = document.getElementById('used-balance');
            const unrealizedPnl = document.getElementById('unrealized-pnl');
            const equity = document.getElementById('equity');
            
            // Aggiorniamo i valori
            if (totalWallet) totalWallet.textContent = `${result.total_wallet.toFixed(2)} USDT`;
            if (availableBalance) availableBalance.textContent = `${result.available.toFixed(2)} USDT`;
            if (usedBalance) usedBalance.textContent = `${result.used.toFixed(2)} USDT`;
            
            // Formatta il PnL con colore a seconda se è positivo o negativo
            if (unrealizedPnl) {
                const pnl = result.pnl;
                const pnlClass = pnl > 0 ? 'text-success' : (pnl < 0 ? 'text-danger' : '');
                unrealizedPnl.textContent = `${pnl.toFixed(2)} USDT`;
                unrealizedPnl.className = `card-text ${pnlClass}`;
            }
            
            // Calcola e formatta il Totale non realizzato (ex Equity)
            if (equity) {
                const total = result.total_wallet || 0;
                const pnl = result.pnl || 0;
                const equityValue = total + pnl;
                const equityClass = pnl > 0 ? 'text-success' : (pnl < 0 ? 'text-danger' : '');
                equity.textContent = `${equityValue.toFixed(2)} USDT`;
                equity.className = `card-text ${equityClass}`;
            }
            
            // Se c'è un errore, mostralo
            if (result.error) {
                appendToLog(`Errore nel caricamento del bilancio: ${result.error}`, 'error');
            }
        }
    } catch (error) {
        console.error('Errore nel caricamento del bilancio:', error);
        appendToLog(`Errore nel caricamento del bilancio: ${error.message}`, 'error');
    }
}

// Funzione per caricare le posizioni
export async function loadPositions() {
    const result = await makeApiRequest('/positions');
    if (result) {
        document.getElementById('positions').textContent = result.open_positions;
    }
}

// Funzione per caricare gli ordini aperti
export async function loadOpenOrders() {
    const result = await makeApiRequest('/orders/open');
    if (result) {
        const tableBody = document.getElementById('open-orders-table');
        tableBody.innerHTML = '';
        
        // Aggiorniamo le intestazioni della tabella per includere le nuove colonne
        const tableHeader = document.querySelector('#open-orders-table-header tr');
        if (tableHeader) {
            // Verifica se abbiamo già le nuove colonne
            if (!document.getElementById('sl-header')) {
                const newHeader = `
                    <th id="symbol-header">Simbolo</th>
                    <th id="side-header">Direzione</th>
                    <th id="amount-header">Quantità</th>
                    <th id="price-header">Prezzo</th>
                    <th id="pnl-header">P/L</th>
                    <th id="usdt-header">USDT Usati</th>
                    <th id="sl-header">Stop Loss</th>
                    <th id="sl-profit-header">P/L SL</th>
                    <th id="status-header">Stato</th>
                    <th id="action-header">Azione</th>
                `;
                tableHeader.innerHTML = newHeader;
            }
        }
        
        if (result.length === 0) {
            const row = document.createElement('tr');
            row.innerHTML = '<td colspan="10" class="text-center">Nessun ordine aperto</td>';
            tableBody.appendChild(row);
        } else {
            result.forEach(item => {
                const row = document.createElement('tr');
                
                // Verifica se è una posizione o un ordine
                const isPosition = item.type === 'position';
                
                // Formatta il PnL se presente (solo per posizioni)
                let pnlDisplay = '-';
                if (isPosition && item.pnl !== undefined) {
                    const pnl = parseFloat(item.pnl);
                    const pnlClass = pnl > 0 ? 'text-success' : (pnl < 0 ? 'text-danger' : '');
                    pnlDisplay = `<span class="${pnlClass}">${pnl.toFixed(2)} USDT</span>`;
                }
                
                // Formatta il lato (Buy/Sell)
                const sideClass = item.side === 'Buy' ? 'text-success' : 'text-danger';
                
                // Formatta lo stato
                let statusDisplay = item.status || '';
                if (isPosition) {
                    statusDisplay = 'Aperta';
                }
                
                // Mostra gli USDT utilizzati per la posizione (margine)
                let usedUSDT = '-';
                if (isPosition && item.margin) {
                    usedUSDT = `${parseFloat(item.margin).toFixed(2)} USDT`;
                }
                
                // Formatta il valore di stop loss e profitto potenziale
                let stopLossDisplay = 'N/A';
                let slProfitDisplay = 'N/A';
                
                if (isPosition) {
                    // Visualizza stop loss
                    if (item.stop_loss && item.stop_loss !== 'N/A') {
                        stopLossDisplay = parseFloat(item.stop_loss).toFixed(4);
                    }
                    
                    // Visualizza profitto potenziale
                    if (item.sl_profit && item.sl_profit !== 'N/A') {
                        const slProfit = parseFloat(item.sl_profit);
                        const slProfitClass = slProfit > 0 ? 'text-success' : 'text-danger';
                        slProfitDisplay = `<span class="${slProfitClass}">${slProfit.toFixed(2)} USDT</span>`;
                    }
                }
                
                // Crea pulsante di azione per chiudere posizioni/annullare ordini
                let actionButton = '';
                if (isPosition) {
                    actionButton = `<button class="btn btn-sm btn-danger" onclick="closePosition('${item.symbol}', '${item.side}')">Chiudi</button>`;
                } else {
                    actionButton = `<button class="btn btn-sm btn-secondary" onclick="cancelOrder('${item.id}')">Annulla</button>`;
                }
                
                row.innerHTML = `
                    <td>${item.symbol}</td>
                    <td class="${sideClass}">${item.side}</td>
                    <td>${item.amount}</td>
                    <td>${typeof item.price === 'number' ? item.price.toFixed(4) : item.price}</td>
                    <td>${pnlDisplay}</td>
                    <td>${usedUSDT}</td>
                    <td>${stopLossDisplay}</td>
                    <td>${slProfitDisplay}</td>
                    <td>${statusDisplay}</td>
                    <td>${actionButton}</td>
                `;
                tableBody.appendChild(row);
            });
        }
    }
}

// Funzione per caricare i trades
export async function loadTrades() {
    const result = await makeApiRequest('/trades');
    if (result) {
        const tableBody = document.getElementById('trades-table');
        tableBody.innerHTML = '';
        
        if (result.length === 0) {
            const row = document.createElement('tr');
            row.innerHTML = '<td colspan="7" class="text-center">Nessun trade recente</td>';
            tableBody.appendChild(row);
        } else {
            result.forEach(trade => {
                const timestamp = new Date(trade.timestamp).toLocaleString();
                const pnl = parseFloat(trade.realized_pnl || 0);
                const pnlClass = pnl > 0 ? 'text-success' : (pnl < 0 ? 'text-danger' : '');
                
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${trade.symbol}</td>
                    <td>${trade.type}</td>
                    <td>${trade.side}</td>
                    <td>${trade.amount}</td>
                    <td>${trade.price}</td>
                    <td class="${pnlClass}">${pnl.toFixed(2)} USDT</td>
                    <td>${timestamp}</td>
                `;
                tableBody.appendChild(row);
            });
        }
    }
}

// Funzione per chiudere una posizione
export async function closePosition(symbol, side) {
    if (!confirm(`Confermi la chiusura della posizione ${symbol}?`)) return;
    
    const closeSide = side === 'Buy' ? 'Sell' : 'Buy';
    appendToLog(`Chiusura posizione ${symbol} (${closeSide})...`);
    
    try {
        // Chiamata all'API per chiudere la posizione
        const response = await makeApiRequest('/close-position', 'POST', {
            symbol: symbol,
            side: side
        });
        
        if (response && response.status === 'success') {
            appendToLog(`✅ Posizione ${symbol} chiusa con successo`);
            // Aggiorna le posizioni e gli ordini
            loadPositions();
            loadOpenOrders();
            loadTrades();
        } else {
            appendToLog(`❌ Errore nella chiusura della posizione ${symbol}: ${response.message || 'Errore sconosciuto'}`);
        }
    } catch (error) {
        appendToLog(`❌ Errore nella chiusura della posizione ${symbol}: ${error.message || error}`);
    }
}

// Funzione per annullare un ordine
export async function cancelOrder(orderId) {
    if (!confirm(`Confermi l'annullamento dell'ordine?`)) return;
    
    appendToLog(`Annullamento ordine ${orderId}...`);
    
    try {
        // Chiamata all'API per annullare l'ordine
        const response = await makeApiRequest('/cancel-order', 'POST', {
            order_id: orderId
        });
        
        if (response && response.status === 'success') {
            appendToLog(`✅ Ordine ${orderId} annullato con successo`);
            // Aggiorna gli ordini aperti
            loadOpenOrders();
        } else {
            appendToLog(`❌ Errore nell'annullamento dell'ordine ${orderId}: ${response.message || 'Errore sconosciuto'}`);
        }
    } catch (error) {
        appendToLog(`❌ Errore nell'annullamento dell'ordine ${orderId}: ${error.message || error}`);
    }
}

// Inizializza gli event listener della dashboard
export function setupDashboardEventListeners() {
    // Carica eventi quando la dashboard viene selezionata
    document.addEventListener('dashboard-selected', () => {
        loadBalance();
        loadPositions();
        loadOpenOrders();
        loadTrades();
    });
    
    // Aggiunge gestori per i bottoni di chiusura posizione e cancellazione ordini
    window.closePosition = closePosition;
    window.cancelOrder = cancelOrder;
} 