// main.js - File principale per l'inizializzazione dell'applicazione

document.addEventListener('DOMContentLoaded', function() {
    // Inizializza la gestione delle tab
    initTabs();
    
    // Inizializza altri componenti dell'applicazione
    initAppComponents();
    
    // Carica i dati iniziali
    loadInitialData();
});

/**
 * Inizializza il sistema di tab
 */
function initTabs() {
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Ottieni il target della pagina dall'attributo data-page
            const pageTarget = this.getAttribute('data-page');
            if (!pageTarget) return;
            
            // Rimuovi la classe active da tutti i link e tutte le pagine
            navLinks.forEach(l => l.classList.remove('active'));
            document.querySelectorAll('.page').forEach(page => {
                page.classList.remove('active');
            });
            
            // Aggiungi la classe active al link cliccato
            this.classList.add('active');
            
            // Mostra la pagina corrispondente
            const targetPage = document.getElementById(`${pageTarget}-page`);
            if (targetPage) {
                targetPage.classList.add('active');
                // Trigger di un evento custom per notificare il cambio di pagina
                const event = new CustomEvent(`${pageTarget}-selected`);
                document.dispatchEvent(event);
            } else {
                console.error(`Pagina ${pageTarget}-page non trovata`);
            }
        });
    });
    
    // Assicurati che la dashboard sia attiva all'avvio
    console.log('Inizializzazione delle tab completata');
}

/**
 * Inizializza i vari componenti dell'applicazione
 */
function initAppComponents() {
    // Inizializza il modulo dei modelli se presente
    if (typeof initTrainingInterface === 'function') {
        initTrainingInterface();
    }
    
    // Inizializza i grafici se necessario
    initializeCharts();
    
    console.log('Componenti dell\'applicazione inizializzati');
}

/**
 * Carica i dati iniziali necessari all'avvio dell'applicazione
 */
function loadInitialData() {
    // Carica i dati delle criptovalute per la dashboard
    loadCryptoData();
    
    console.log('Dati iniziali caricati');
}

/**
 * Carica i dati delle criptovalute per la tabella della dashboard
 */
function loadCryptoData() {
    fetch('/api/cryptos')
        .then(response => {
            if (!response.ok) {
                throw new Error('Errore nel caricamento dei dati delle criptovalute');
            }
            return response.json();
        })
        .then(data => {
            updateCryptoTable(data);
        })
        .catch(error => {
            console.error('Errore:', error);
            // Mostra dati di esempio in caso di errore
            showSampleCryptoData();
        });
}

/**
 * Aggiorna la tabella delle criptovalute con i dati ricevuti
 * @param {Array} data - Array di dati delle criptovalute
 */
function updateCryptoTable(data) {
    const container = document.getElementById('crypto-table-container');
    if (!container) return;
    
    let tableHtml = `
        <table class="table table-crypto">
            <thead>
                <tr>
                    <th>Simbolo</th>
                    <th>Prezzo</th>
                    <th>Cambio 24h</th>
                    <th>Volume</th>
                </tr>
            </thead>
            <tbody>
    `;
    
    if (data && data.length > 0) {
        data.forEach(crypto => {
            const priceChange = parseFloat(crypto.priceChangePercent || 0);
            const changeClass = priceChange >= 0 ? 'text-success' : 'text-danger';
            const changeIcon = priceChange >= 0 ? 'bi-arrow-up' : 'bi-arrow-down';
            
            tableHtml += `
                <tr class="crypto-row">
                    <td>
                        <strong>${crypto.symbol}</strong>
                    </td>
                    <td>${parseFloat(crypto.lastPrice).toFixed(2)} USDT</td>
                    <td class="${changeClass}">
                        <i class="bi ${changeIcon}"></i> ${Math.abs(priceChange).toFixed(2)}%
                    </td>
                    <td>${Math.round(parseFloat(crypto.volume)).toLocaleString()} USDT</td>
                </tr>
            `;
        });
    } else {
        tableHtml += `
            <tr>
                <td colspan="4" class="text-center">Nessun dato disponibile</td>
            </tr>
        `;
    }
    
    tableHtml += `
            </tbody>
        </table>
    `;
    
    container.innerHTML = tableHtml;
}

/**
 * Mostra dati di esempio nel caso l'API non risponda
 */
function showSampleCryptoData() {
    const sampleData = [
        { symbol: 'BTC/USDT', lastPrice: '40000', priceChangePercent: '2.5', volume: '1000000000' },
        { symbol: 'ETH/USDT', lastPrice: '2800', priceChangePercent: '1.8', volume: '500000000' },
        { symbol: 'BNB/USDT', lastPrice: '350', priceChangePercent: '-0.5', volume: '150000000' },
        { symbol: 'SOL/USDT', lastPrice: '120', priceChangePercent: '3.2', volume: '200000000' },
        { symbol: 'ADA/USDT', lastPrice: '1.2', priceChangePercent: '-1.5', volume: '100000000' }
    ];
    
    updateCryptoTable(sampleData);
}

/**
 * Inizializza i grafici dell'applicazione
 */
function initializeCharts() {
    const btcChartElement = document.getElementById('btcChart');
    if (!btcChartElement) return;
    
    // Configurazione del grafico
    const btcChart = new Chart(btcChartElement, {
        type: 'line',
        data: {
            labels: [], // Verrà popolato con le date
            datasets: [{
                label: 'BTC/USDT',
                data: [], // Verrà popolato con i prezzi
                borderColor: 'rgba(29, 161, 242, 1)',
                backgroundColor: 'rgba(29, 161, 242, 0.1)',
                borderWidth: 2,
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            },
            scales: {
                x: {
                    grid: {
                        display: false
                    }
                },
                y: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                }
            }
        }
    });
    
    // Aggiungi listener per i bottoni del timeframe
    document.querySelectorAll('.timeframe-selector .btn').forEach(button => {
        button.addEventListener('click', function() {
            // Rimuovi la classe active da tutti i bottoni
            document.querySelectorAll('.timeframe-selector .btn').forEach(btn => {
                btn.classList.remove('active');
            });
            
            // Aggiungi la classe active al bottone cliccato
            this.classList.add('active');
            
            // Ottieni il timeframe selezionato
            const timeframe = this.getAttribute('data-timeframe');
            
            // Carica i dati per il timeframe selezionato
            loadChartData(btcChart, timeframe);
        });
    });
    
    // Carica i dati iniziali (1h)
    loadChartData(btcChart, '1h');
}

/**
 * Carica i dati per il grafico in base al timeframe selezionato
 * @param {Chart} chart - Istanza del grafico
 * @param {string} timeframe - Timeframe selezionato
 */
function loadChartData(chart, timeframe) {
    // In una versione reale, questi dati sarebbero caricati da un'API
    fetch(`/api/chart/btcusdt/${timeframe}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Errore nel caricamento dei dati del grafico');
            }
            return response.json();
        })
        .then(data => {
            updateChartData(chart, data);
        })
        .catch(error => {
            console.error('Errore:', error);
            // Mostra dati di esempio in caso di errore
            showSampleChartData(chart, timeframe);
        });
}

/**
 * Aggiorna i dati del grafico
 * @param {Chart} chart - Istanza del grafico
 * @param {Array} data - Dati del grafico
 */
function updateChartData(chart, data) {
    if (!data || !data.length) return;
    
    const labels = data.map(item => new Date(item.time).toLocaleTimeString());
    const prices = data.map(item => item.price);
    
    chart.data.labels = labels;
    chart.data.datasets[0].data = prices;
    chart.update();
}

/**
 * Mostra dati di esempio per il grafico in caso di errore
 * @param {Chart} chart - Istanza del grafico
 * @param {string} timeframe - Timeframe selezionato
 */
function showSampleChartData(chart, timeframe) {
    const now = new Date();
    const labels = [];
    const data = [];
    
    // Genera dati fittizi in base al timeframe
    let points = 24;
    let interval = 60 * 60 * 1000; // 1 ora in ms
    
    if (timeframe === '4h') {
        points = 30;
        interval = 4 * 60 * 60 * 1000; // 4 ore in ms
    } else if (timeframe === '1d') {
        points = 30;
        interval = 24 * 60 * 60 * 1000; // 1 giorno in ms
    }
    
    let basePrice = 40000;
    
    for (let i = points - 1; i >= 0; i--) {
        const time = new Date(now.getTime() - (i * interval));
        labels.push(timeframe === '1d' ? time.toLocaleDateString() : time.toLocaleTimeString());
        
        // Genera un movimento casuale del prezzo
        basePrice = basePrice + (Math.random() * 1000 - 500);
        data.push(basePrice);
    }
    
    chart.data.labels = labels;
    chart.data.datasets[0].data = data;
    chart.update();
} 