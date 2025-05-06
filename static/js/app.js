/**
 * Trading Dashboard App
 * 
 * Main JavaScript file for the trading dashboard application.
 * Handles API calls, data visualization, and UI interactions.
 */

// Global state
const state = {
    timeframes: [],
    selectedTimeframe: null,
    symbols: [],
    selectedSymbol: null,
    autoRefreshInterval: null,
    isLoading: false,
    isDetailView: false,
    detailData: null
};

// Refresh intervals (in milliseconds)
const REFRESH_INTERVALS = {
    OFF: 0,
    '30s': 30000,
    '1m': 60000,
    '5m': 300000,
    '15m': 900000
};

// Initialize the dashboard
async function initDashboard() {
    try {
        // Get available timeframes
        const timeframesResponse = await fetchData('/api/timeframes');
        if (timeframesResponse.timeframes && timeframesResponse.timeframes.length > 0) {
            state.timeframes = timeframesResponse.timeframes;
            state.selectedTimeframe = timeframesResponse.timeframes[0];
            
            // Populate timeframe selector
            populateTimeframeSelector();
            
            // Get symbols for selected timeframe
            await loadSymbols(state.selectedTimeframe);
            
            // Load market data
            await loadMarketData();
            
            // Setup UI event listeners
            setupEventListeners();
        } else {
            showError('Nessun timeframe disponibile. Eseguire prima il pipeline di volatilità.');
        }
    } catch (error) {
        console.error('Error initializing dashboard:', error);
        showError('Errore durante l\'inizializzazione della dashboard. Controllare la console per dettagli.');
    }
}

// Fetch data from API
async function fetchData(url) {
    try {
        const response = await fetch(url);
        const data = await response.json();
        return data;
    } catch (error) {
        console.error(`Error fetching data from ${url}:`, error);
        throw error;
    }
}

// Populate timeframe selector
function populateTimeframeSelector() {
    const selector = document.getElementById('timeframeSelector');
    if (selector) {
        selector.innerHTML = '';
        
        state.timeframes.forEach(timeframe => {
            const option = document.createElement('option');
            option.value = timeframe;
            option.textContent = timeframe;
            option.selected = timeframe === state.selectedTimeframe;
            selector.appendChild(option);
        });
    }
}

// Load symbols for a timeframe
async function loadSymbols(timeframe) {
    try {
        state.isLoading = true;
        updateLoadingState();
        
        const response = await fetchData(`/api/symbols/${timeframe}`);
        if (response.symbols) {
            state.symbols = response.symbols;
            if (state.symbols.length > 0 && !state.symbols.includes(state.selectedSymbol)) {
                state.selectedSymbol = state.symbols[0];
            }
        } else {
            state.symbols = [];
            state.selectedSymbol = null;
            showError('Nessun simbolo disponibile per il timeframe selezionato.');
        }
        
        state.isLoading = false;
        updateLoadingState();
        return state.symbols;
    } catch (error) {
        state.isLoading = false;
        updateLoadingState();
        console.error('Error loading symbols:', error);
        showError('Errore durante il caricamento dei simboli.');
        return [];
    }
}

// Load market data
async function loadMarketData() {
    try {
        if (!state.selectedTimeframe) return;
        
        state.isLoading = true;
        updateLoadingState();
        
        const response = await fetchData(`/api/market/${state.selectedTimeframe}`);
        
        if (response.status === 'success' && response.market) {
            renderMarketGrid(response.market);
        } else {
            showError('Errore nel caricamento dei dati di mercato.');
        }
        
        state.isLoading = false;
        updateLoadingState();
    } catch (error) {
        state.isLoading = false;
        updateLoadingState();
        console.error('Error loading market data:', error);
        showError('Errore durante il caricamento dei dati di mercato.');
    }
}

// Load detail view data
async function loadDetailView(symbol) {
    try {
        if (!state.selectedTimeframe || !symbol) return;
        
        state.isLoading = true;
        state.isDetailView = true;
        state.selectedSymbol = symbol;
        updateLoadingState();
        
        // Get summary data
        const summaryResponse = await fetchData(`/api/summary/${symbol}/${state.selectedTimeframe}`);
        
        // Get pattern prediction
        const patternResponse = await fetchData(`/api/pattern/${symbol}/${state.selectedTimeframe}`);
        
        // Get price history data
        const historyResponse = await fetchData(`/api/data/${symbol}/${state.selectedTimeframe}?limit=100`);
        
        if (summaryResponse.status === 'success' && patternResponse.status === 'success') {
            state.detailData = {
                summary: summaryResponse,
                pattern: patternResponse,
                history: historyResponse.status === 'success' ? historyResponse.data : []
            };
            
            renderDetailView();
        } else {
            showError(`Errore nel caricamento dei dati dettaglio per ${symbol}.`);
            goBackToMarketView();
        }
        
        state.isLoading = false;
        updateLoadingState();
    } catch (error) {
        state.isLoading = false;
        updateLoadingState();
        console.error('Error loading detail view:', error);
        showError(`Errore durante il caricamento dei dati dettaglio per ${symbol}.`);
        goBackToMarketView();
    }
}

// Render market grid
function renderMarketGrid(marketData) {
    const container = document.getElementById('mainContent');
    if (!container) return;
    
    // Clear previous content
    container.innerHTML = '';
    
    // Add header
    const header = document.createElement('div');
    header.className = 'header d-flex justify-content-between align-items-center';
    header.innerHTML = `
        <h1 class="header-title">Dashboard <span>Trading</span></h1>
        <div class="controls">
            <select id="timeframeSelector" class="form-select"></select>
            <div class="dropdown">
                <button class="btn btn-refresh" id="refreshBtn">
                    <span class="refresh-icon">⟳</span> Aggiorna
                </button>
                <div class="dropdown-content">
                    <div class="dropdown-item" data-interval="OFF">Auto-refresh: Off</div>
                    <div class="dropdown-item" data-interval="30s">Auto-refresh: 30s</div>
                    <div class="dropdown-item" data-interval="1m">Auto-refresh: 1m</div>
                    <div class="dropdown-item" data-interval="5m">Auto-refresh: 5m</div>
                    <div class="dropdown-item" data-interval="15m">Auto-refresh: 15m</div>
                </div>
            </div>
        </div>
    `;
    container.appendChild(header);
    
    // Update timeframe selector
    populateTimeframeSelector();
    
    // Add market grid
    const gridContainer = document.createElement('div');
    gridContainer.className = 'market-grid';
    
    // Check if we have data
    if (marketData.length === 0) {
        const noData = document.createElement('div');
        noData.className = 'no-data';
        noData.textContent = 'Nessun dato di mercato disponibile';
        container.appendChild(noData);
        return;
    }
    
    // Add cards for each symbol
    marketData.forEach(crypto => {
        if (crypto.status !== 'success') return;
        
        const card = document.createElement('div');
        card.className = 'card crypto-card';
        card.dataset.symbol = crypto.symbol;
        
        // Determine price change class
        const priceChangeClass = crypto.price_change_percent >= 0 ? 'positive' : 'negative';
        const priceChangeIcon = crypto.price_change_percent >= 0 ? '▲' : '▼';
        
        // Determine trading signal class
        let signalClass = 'signal-hold';
        if (crypto.trading_signal === 'BUY') {
            signalClass = 'signal-buy';
        } else if (crypto.trading_signal === 'SELL') {
            signalClass = 'signal-sell';
        }
        
        card.innerHTML = `
            <div class="crypto-symbol">${crypto.symbol}</div>
            <div class="crypto-price">${formatPrice(crypto.price)}</div>
            <div class="price-change ${priceChangeClass}">
                ${priceChangeIcon} ${crypto.price_change_percent.toFixed(2)}%
            </div>
            
            <div class="crypto-info">
                <div>
                    <span class="info-label">Volume</span>
                    <span class="info-value">${formatVolume(crypto.volume)}</span>
                </div>
                <div>
                    <span class="info-label">Volatilità media</span>
                    <span class="info-value">${crypto.avg_volatility?.toFixed(2)}%</span>
                </div>
                <div>
                    <span class="info-label">Ultimo aggiornamento</span>
                    <span class="info-value">${formatTime(crypto.last_updated)}</span>
                </div>
            </div>
            
            <div class="pattern-container">
                <div class="pattern-title">Pattern corrente</div>
                <div class="pattern">${crypto.pattern || 'N/A'}</div>
                
                <div class="prediction">
                    <div class="prediction-pattern">
                        <div class="pattern-title">Prossimo pattern</div>
                        <div class="prediction-value">${crypto.next_pattern || 'N/A'}</div>
                    </div>
                    <div class="prediction-probability">${crypto.probability?.toFixed(1)}%</div>
                </div>
            </div>
            
            <div class="trading-signal ${signalClass}">
                ${crypto.trading_signal} (${crypto.confidence?.toFixed(1)}%)
            </div>
        `;
        
        gridContainer.appendChild(card);
    });
    
    container.appendChild(gridContainer);
    
    // Add auto-refresh indicator if active
    if (state.autoRefreshInterval) {
        const refreshIndicator = document.createElement('div');
        refreshIndicator.className = 'last-updated';
        const interval = Object.entries(REFRESH_INTERVALS).find(([key, value]) => 
            value === state.autoRefreshInterval)[0];
        refreshIndicator.innerHTML = `
            <div class="auto-refresh-badge">
                <span class="refresh-icon">⟳</span> Auto-refresh attivo: ${interval}
            </div>
        `;
        container.appendChild(refreshIndicator);
    }
    
    // Add last updated
    const lastUpdated = document.createElement('div');
    lastUpdated.className = 'last-updated';
    lastUpdated.textContent = `Ultimo aggiornamento: ${new Date().toLocaleString()}`;
    container.appendChild(lastUpdated);
}

// Render detail view
function renderDetailView() {
    const container = document.getElementById('mainContent');
    if (!container || !state.detailData) return;
    
    // Clear previous content
    container.innerHTML = '';
    
    const { summary, pattern, history } = state.detailData;
    
    // Add header with back button
    const header = document.createElement('div');
    header.className = 'header d-flex justify-content-between align-items-center';
    header.innerHTML = `
        <div class="d-flex align-items-center gap-3">
            <button id="backButton" class="btn btn-sm">← Torna al mercato</button>
            <h1 class="header-title">${summary.symbol}</h1>
        </div>
        <div class="controls">
            <select id="timeframeSelector" class="form-select"></select>
            <button class="btn btn-refresh" id="refreshDetailBtn">
                <span class="refresh-icon">⟳</span> Aggiorna
            </button>
        </div>
    `;
    container.appendChild(header);
    
    // Update timeframe selector
    populateTimeframeSelector();
    
    // Create detail container
    const detailContainer = document.createElement('div');
    detailContainer.className = 'detail-container';
    
    // Determine price change class
    const priceChangeClass = summary.price_change_percent >= 0 ? 'positive' : 'negative';
    const priceChangeIcon = summary.price_change_percent >= 0 ? '▲' : '▼';
    
    // Determine trading signal class
    let signalClass = 'signal-hold';
    let signalFillClass = 'hold-fill';
    if (summary.trading_signal === 'BUY') {
        signalClass = 'signal-buy';
        signalFillClass = 'buy-fill';
    } else if (summary.trading_signal === 'SELL') {
        signalClass = 'signal-sell';
        signalFillClass = 'sell-fill';
    }
    
    // Create price and chart section
    const priceSection = document.createElement('div');
    priceSection.className = 'card';
    priceSection.innerHTML = `
        <div class="card-header">
            <h2 class="card-title">Prezzo & Volume</h2>
            <div class="price-change ${priceChangeClass}">
                ${priceChangeIcon} ${summary.price_change_percent?.toFixed(2)}%
            </div>
        </div>
        
        <div class="d-flex justify-content-between mb-4">
            <div>
                <div class="info-label">Prezzo corrente</div>
                <div class="crypto-price">${formatPrice(summary.price)}</div>
            </div>
            <div>
                <div class="info-label">Volume</div>
                <div class="info-value">${formatVolume(summary.volume)}</div>
            </div>
            <div>
                <div class="info-label">Volatilità media</div>
                <div class="info-value">${summary.avg_volatility?.toFixed(2)}%</div>
            </div>
        </div>
        
        <div class="price-chart" id="priceChart">
            <canvas id="chartCanvas"></canvas>
        </div>
    `;
    
    // Create pattern section
    const patternSection = document.createElement('div');
    patternSection.className = 'card';
    
    let multiStepHTML = '';
    if (pattern.multi_step && pattern.multi_step.length > 0) {
        multiStepHTML = `
            <div class="pattern-forecast">
                <h3 class="mb-3">Previsione Multi-Step</h3>
                ${pattern.multi_step.map((step, index) => `
                    <div class="forecast-item">
                        <div class="forecast-pattern">Step ${index+1}: ${step.pattern}</div>
                        <div class="forecast-probability">
                            <div class="probability-bar">
                                <div class="probability-fill ${signalFillClass}" 
                                     style="width: ${step.probability}%;"></div>
                            </div>
                            <span>${step.probability}%</span>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
    }
    
    patternSection.innerHTML = `
        <div class="card-header">
            <h2 class="card-title">Analisi Pattern</h2>
        </div>
        
        <div class="pattern-container mb-4">
            <div class="pattern-title">Pattern corrente</div>
            <div class="pattern">${pattern.current_pattern || 'N/A'}</div>
            
            <div class="prediction">
                <div class="prediction-pattern">
                    <div class="pattern-title">Prossimo pattern</div>
                    <div class="prediction-value">${pattern.next_pattern || 'N/A'}</div>
                </div>
                <div class="prediction-probability">${pattern.probability}%</div>
            </div>
        </div>
        
        <div class="mb-4">
            <h3 class="mb-2">Segnale di Trading</h3>
            <div class="trading-signal ${signalClass}">
                ${pattern.trading_signal} (${pattern.confidence?.toFixed(1)}%)
            </div>
            <div class="confidence-meter">
                <div class="confidence-fill ${signalFillClass}" style="width: ${pattern.confidence}%;"></div>
            </div>
        </div>
        
        ${multiStepHTML}
        
        <div class="last-updated mt-4">
            Analisi generata il: ${formatDateTime(pattern.timestamp)}
        </div>
    `;
    
    detailContainer.appendChild(priceSection);
    detailContainer.appendChild(patternSection);
    container.appendChild(detailContainer);
    
    // Initialize chart if we have data
    if (history && history.length > 0) {
        setTimeout(() => initChart(history), 100);
    }
}

// Initialize price chart
function initChart(data) {
    if (!data || data.length === 0) return;
    
    const ctx = document.getElementById('chartCanvas');
    if (!ctx) return;
    
    // Reverse data to show oldest first
    const chartData = [...data].reverse();
    
    // Extract data for chart
    const labels = chartData.map(item => formatTime(item.formatted_time));
    const prices = chartData.map(item => item.close);
    const volumes = chartData.map(item => item.volume);
    
    // Get volatility if available
    const volatility = chartData.map(item => item.close_volatility);
    const hasVolatility = volatility.some(v => v !== null && v !== undefined);
    
    // Create datasets
    const datasets = [
        {
            label: 'Prezzo',
            data: prices,
            borderColor: '#2196F3',
            backgroundColor: 'rgba(33, 150, 243, 0.1)',
            borderWidth: 2,
            tension: 0.4,
            pointRadius: 0,
            pointHoverRadius: 4,
            yAxisID: 'y',
        },
        {
            label: 'Volume',
            data: volumes,
            backgroundColor: 'rgba(255, 171, 0, 0.5)',
            borderColor: 'rgba(255, 171, 0, 0.8)',
            borderWidth: 1,
            type: 'bar',
            yAxisID: 'y1',
        }
    ];
    
    // Add volatility dataset if available
    if (hasVolatility) {
        datasets.push({
            label: 'Volatilità',
            data: volatility,
            borderColor: '#FF3B69',
            backgroundColor: 'rgba(255, 59, 105, 0.1)',
            borderWidth: 1,
            borderDash: [5, 5],
            tension: 0.4,
            pointRadius: 0,
            pointHoverRadius: 4,
            yAxisID: 'y2',
        });
    }
    
    // Create chart
    const chart = new Chart(ctx, {
        type: 'line',
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                tooltip: {
                    enabled: true,
                    mode: 'index',
                    intersect: false,
                    backgroundColor: 'rgba(26, 34, 51, 0.9)',
                    titleFont: {
                        size: 12,
                        weight: 'bold',
                    },
                    bodyFont: {
                        size: 11,
                    },
                    padding: 10,
                    borderColor: 'rgba(255, 255, 255, 0.1)',
                    borderWidth: 1,
                },
                legend: {
                    position: 'top',
                    labels: {
                        color: '#e0e0e0',
                        boxWidth: 12,
                        padding: 15
                    }
                },
            },
            scales: {
                x: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)',
                    },
                    ticks: {
                        color: '#a0a0a0',
                        maxRotation: 0,
                        autoSkip: true,
                        maxTicksLimit: 10
                    }
                },
                y: {
                    position: 'left',
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)',
                    },
                    ticks: {
                        color: '#a0a0a0',
                    }
                },
                y1: {
                    position: 'right',
                    grid: {
                        display: false,
                        drawOnChartArea: false,
                    },
                    ticks: {
                        color: '#FFAB00',
                    },
                    display: true
                },
                y2: hasVolatility ? {
                    position: 'right',
                    grid: {
                        display: false,
                        drawOnChartArea: false,
                    },
                    ticks: {
                        color: '#FF3B69',
                    },
                    display: true
                } : undefined
            }
        }
    });
}

// Setup event listeners
function setupEventListeners() {
    // Timeframe selector
    document.addEventListener('change', function(event) {
        if (event.target && event.target.id === 'timeframeSelector') {
            state.selectedTimeframe = event.target.value;
            if (state.isDetailView && state.selectedSymbol) {
                loadDetailView(state.selectedSymbol);
            } else {
                loadMarketData();
            }
        }
    });
    
    // Refresh button
    document.addEventListener('click', function(event) {
        if (event.target && (event.target.id === 'refreshBtn' || event.target.closest('#refreshBtn'))) {
            refreshData();
        }
    });
    
    // Refresh detail button
    document.addEventListener('click', function(event) {
        if (event.target && (event.target.id === 'refreshDetailBtn' || event.target.closest('#refreshDetailBtn'))) {
            loadDetailView(state.selectedSymbol);
        }
    });
    
    // Back button
    document.addEventListener('click', function(event) {
        if (event.target && (event.target.id === 'backButton' || event.target.closest('#backButton'))) {
            goBackToMarketView();
        }
    });
    
    // Crypto card click (for detail view)
    document.addEventListener('click', function(event) {
        const card = event.target.closest('.crypto-card');
        if (card && card.dataset.symbol) {
            loadDetailView(card.dataset.symbol);
        }
    });
    
    // Auto-refresh interval selection
    document.addEventListener('click', function(event) {
        const item = event.target.closest('.dropdown-item');
        if (item && item.dataset.interval) {
            setAutoRefreshInterval(item.dataset.interval);
        }
    });
}

// Go back to market view
function goBackToMarketView() {
    state.isDetailView = false;
    state.detailData = null;
    loadMarketData();
}

// Set auto-refresh interval
function setAutoRefreshInterval(interval) {
    // Clear existing interval
    if (state.autoRefreshInterval !== null) {
        clearInterval(state.autoRefreshInterval);
        state.autoRefreshInterval = null;
    }
    
    // Set new interval
    const intervalMs = REFRESH_INTERVALS[interval];
    if (intervalMs > 0) {
        state.autoRefreshInterval = setInterval(refreshData, intervalMs);
    }
    
    // Refresh immediately and update UI
    refreshData();
}

// Refresh data based on current view
function refreshData() {
    if (state.isDetailView && state.selectedSymbol) {
        loadDetailView(state.selectedSymbol);
    } else {
        loadMarketData();
    }
}

// Update loading state
function updateLoadingState() {
    const container = document.getElementById('mainContent');
    if (!container) return;
    
    if (state.isLoading) {
        // Add loading overlay if it doesn't exist
        if (!document.getElementById('loadingOverlay')) {
            const overlay = document.createElement('div');
            overlay.id = 'loadingOverlay';
            overlay.className = 'loading-overlay';
            overlay.innerHTML = '<div class="loading"></div>';
            document.body.appendChild(overlay);
        }
    } else {
        // Remove loading overlay if it exists
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) {
            overlay.remove();
        }
    }
    
    // Update refresh button state
    const refreshBtn = document.getElementById('refreshBtn');
    const refreshDetailBtn = document.getElementById('refreshDetailBtn');
    
    if (refreshBtn) {
        refreshBtn.disabled = state.isLoading;
        refreshBtn.classList.toggle('refreshing', state.isLoading);
    }
    
    if (refreshDetailBtn) {
        refreshDetailBtn.disabled = state.isLoading;
        refreshDetailBtn.classList.toggle('refreshing', state.isLoading);
    }
}

// Show error message
function showError(message) {
    const container = document.getElementById('mainContent');
    if (!container) return;
    
    const errorElement = document.createElement('div');
    errorElement.className = 'alert alert-danger mt-3';
    errorElement.textContent = message;
    
    // Add to container
    container.appendChild(errorElement);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (errorElement.parentNode) {
            errorElement.remove();
        }
    }, 5000);
}

// Format functions
function formatPrice(price) {
    if (price === undefined || price === null) return 'N/A';
    
    // Format based on price magnitude
    if (price < 0.01) {
        return '$' + price.toFixed(6);
    } else if (price < 1) {
        return '$' + price.toFixed(4);
    } else if (price < 1000) {
        return '$' + price.toFixed(2);
    } else {
        return '$' + price.toFixed(2);
    }
}

function formatVolume(volume) {
    if (volume === undefined || volume === null) return 'N/A';
    
    // Format with K, M, B suffixes
    if (volume >= 1000000000) {
        return (volume / 1000000000).toFixed(2) + 'B';
    } else if (volume >= 1000000) {
        return (volume / 1000000).toFixed(2) + 'M';
    } else if (volume >= 1000) {
        return (volume / 1000).toFixed(2) + 'K';
    } else {
        return volume.toFixed(2);
    }
}

function formatTime(timestamp) {
    if (!timestamp) return 'N/A';
    
    // Extract just the time portion or short date+time based on context
    try {
        const date = new Date(timestamp);
        const now = new Date();
        const isToday = date.toDateString() === now.toDateString();
        
        if (isToday) {
            return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        } else {
            return date.toLocaleDateString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
        }
    } catch (e) {
        return timestamp;
    }
}

function formatDateTime(timestamp) {
    if (!timestamp) return 'N/A';
    
    try {
        const date = new Date(timestamp);
        return date.toLocaleString();
    } catch (e) {
        return timestamp;
    }
}

// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', initDashboard);
