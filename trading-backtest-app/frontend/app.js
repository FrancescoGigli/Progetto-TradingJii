/**
 * TradingJii Backtest Suite - Frontend Application
 */

// API Configuration
const API_URL = 'http://localhost:8000';

// Global State
let currentSymbol = 'BTC/USDT:USDT';
let currentLeverage = 1;
let currentTakeProfit = 0.02;
let currentStopLoss = 0.02; // Added stop loss variable
let currentStrategy = null;
let chart = null;
let candleSeries = null;
let volumeSeries = null;
let markers = [];

// Indicator Management
let indicatorSeries = {};  // Store indicator series
let oscillatorCharts = {}; // Store separate charts for oscillators
let indicatorPanels = {};  // Store panel references

// Strategy Information
const STRATEGY_INFO = {
    'rsi_mean_reversion': {
        name: 'RSI Mean Reversion',
        description: 'Long when RSI < 30 and rising, Short when RSI > 70 and falling'
    },
    'ema_crossover': {
        name: 'EMA Crossover', 
        description: 'Long when EMA20 crosses above EMA50, Short when crosses below'
    },
    'breakout_range': {
        name: 'Breakout Range',
        description: 'Long on break above 20-period high, Short on break below 20-period low'
    },
    'bollinger_rebound': {
        name: 'Bollinger Rebound',
        description: 'Long when price bounces off lower band, Short when bounces off upper band'
    },
    'macd_histogram': {
        name: 'MACD Histogram',
        description: 'Long when histogram turns positive, Short when turns negative'
    },
    'donchian_breakout': {
        name: 'Donchian Breakout',
        description: 'Long on upper channel break, Short on lower channel break'
    },
    'adx_filter_crossover': {
        name: 'ADX Filtered Crossover',
        description: 'EMA crossover signals only when ADX > 20 (trending market)'
    }
};

// Initialize Application
document.addEventListener('DOMContentLoaded', async () => {
    try {
        // Initialize event listeners
        initializeEventListeners();
        
        // Load strategies from API
        await loadStrategies();
        
        // Initialize chart
        if (!initializeChart()) {
            console.error("Failed to initialize chart during startup");
        }
        
        // Initialize update data button
        initializeUpdateDataButton();
        
        // Aggiorna il date range in base ai dati presenti nel database
        await updateDateRangeFromDatabase();
        
        console.log("Application initialized successfully");
    } catch (error) {
        console.error("Error initializing application:", error);
        showError("Failed to initialize application: " + error.message);
    }
});

// Event Listeners
function initializeEventListeners() {
    // Symbol selection
    document.getElementById('symbol-select').addEventListener('change', (e) => {
        currentSymbol = e.target.value;
        document.getElementById('current-symbol').textContent = currentSymbol.replace(':USDT', '');
        if (currentStrategy) {
            runBacktest(currentStrategy);
        }
    });

    // Leverage buttons
    document.querySelectorAll('.leverage-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.leverage-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentLeverage = parseInt(btn.dataset.leverage);
            if (currentStrategy) {
                runBacktest(currentStrategy);
            }
        });
    });

    // Take profit buttons
    document.querySelectorAll('.tp-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.tp-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentTakeProfit = parseFloat(btn.dataset.tp);
            if (currentStrategy) {
                runBacktest(currentStrategy);
            }
        });
    });
    
    // Stop loss buttons
    document.querySelectorAll('.sl-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.sl-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentStopLoss = parseFloat(btn.dataset.sl);
            if (currentStrategy) {
                runBacktest(currentStrategy);
            }
        });
    });

    // Compare all button
    document.getElementById('compare-all-btn').addEventListener('click', compareAllStrategies);
    
    // Le date verranno impostate automaticamente dalla funzione updateDateRangeFromDatabase()
}

// Load Strategies
async function loadStrategies() {
    try {
        const response = await fetch(`${API_URL}/api/strategies`);
        const data = await response.json();
        
        const container = document.getElementById('strategy-cards');
        container.innerHTML = '';
        
        data.strategies.forEach(strategy => {
            const card = createStrategyCard(strategy);
            container.appendChild(card);
        });
    } catch (error) {
        console.error('Error loading strategies:', error);
        showError('Failed to load strategies');
    }
}

// Create Strategy Card
function createStrategyCard(strategyKey) {
    const info = STRATEGY_INFO[strategyKey];
    const card = document.createElement('div');
    card.className = 'strategy-card';
    card.dataset.strategy = strategyKey;
    
    card.innerHTML = `
        <h3>${info.name}</h3>
        <p>${info.description}</p>
        <div class="strategy-preview" id="preview-${strategyKey}">
            <span>Click to run backtest</span>
        </div>
    `;
    
    card.addEventListener('click', () => {
        console.log('Strategy clicked:', strategyKey);
        document.querySelectorAll('.strategy-card').forEach(c => c.classList.remove('active'));
        card.classList.add('active');
        currentStrategy = strategyKey;
        runBacktest(strategyKey);
    });
    
    return card;
}

// Initialize Chart
function initializeChart() {
    try {
        const container = document.getElementById('chart-container');
        
        // Check if LightweightCharts is available and has the expected API
        if (typeof LightweightCharts === 'undefined') {
            console.error("LightweightCharts library not loaded");
            showError("Chart library not loaded. Please check your internet connection and refresh the page.");
            
            // Try to dynamically load the library as a fallback
            const script = document.createElement('script');
            script.src = 'https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js';
            script.async = true;
            script.onload = () => {
                console.log("LightweightCharts library loaded dynamically");
                setTimeout(initializeChart, 100); // Try initialization again after loading
            };
            document.head.appendChild(script);
            
            return false;
        }
        
        // Verify API is available
        if (typeof LightweightCharts.createChart !== 'function') {
            console.error("LightweightCharts API is not as expected");
            showError("Chart library API is incompatible. Please check console for details.");
            return false;
        }
        
        // Check if container exists
        if (!container) {
            console.error("Chart container not found");
            return false;
        }
        
        // Clear container first in case there's any existing content
        container.innerHTML = '';
        
        // Create chart
        try {
            chart = LightweightCharts.createChart(container, {
                layout: {
                    background: { color: '#141414' },
                    textColor: '#ffffff',
                },
                grid: {
                    vertLines: { color: '#2a2a2a' },
                    horzLines: { color: '#2a2a2a' },
                },
                crosshair: {
                    mode: LightweightCharts.CrosshairMode.Normal,
                },
                rightPriceScale: {
                    borderColor: '#2a2a2a',
                },
                timeScale: {
                    borderColor: '#2a2a2a',
                    timeVisible: true,
                    secondsVisible: false,
                },
            });
        } catch (err) {
            console.error("Failed to create chart:", err);
            showError("Failed to create chart: " + err.message);
            return false;
        }
        
        // Check if chart was created successfully
        if (!chart) {
            console.error("Failed to create chart - chart object is null");
            return false;
        }
        
        // Check if chart API is as expected
        if (typeof chart.addCandlestickSeries !== 'function') {
            console.error("Chart API is not as expected - addCandlestickSeries is not a function");
            console.log("Chart object:", chart);
            showError("Chart library API is incompatible. Please check console for details.");
            return false;
        }
        
        // Add series
        try {
            candleSeries = chart.addCandlestickSeries({
                upColor: '#00ff88',
                downColor: '#ff3b3b',
                borderVisible: false,
                wickUpColor: '#00ff88',
                wickDownColor: '#ff3b3b',
            });
            
            volumeSeries = chart.addHistogramSeries({
                color: '#26a69a',
                priceFormat: {
                    type: 'volume',
                },
                priceScaleId: '',
                scaleMargins: {
                    top: 0.8,
                    bottom: 0,
                },
            });
        } catch (err) {
            console.error("Failed to add series to chart:", err);
            showError("Failed to add series to chart: " + err.message);
            return false;
        }
        
        console.log("Chart initialized successfully");
        return true;
    } catch (error) {
        console.error("Error initializing chart:", error);
        showError("Failed to initialize chart: " + error.message);
        return false;
    }
}

// Run Backtest
async function runBacktest(strategy) {
    showLoading(true);
    
    try {
        // Get date range
        const startDate = document.getElementById('start-date').value;
        const endDate = document.getElementById('end-date').value;
        
        // Load market data - encode properly for URL
        const encodedSymbol = currentSymbol.replace('/', '%2F').replace(':', '%3A');
        const marketResponse = await fetch(`${API_URL}/api/data/${encodedSymbol}?start_date=${startDate}&end_date=${endDate}`);
        const marketData = await marketResponse.json();
        
        // Run backtest
        const backtestResponse = await fetch(`${API_URL}/api/backtest`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                symbol: currentSymbol,
                strategy: strategy,
                leverage: currentLeverage,
                take_profit_pct: currentTakeProfit,
                stop_loss_pct: currentStopLoss, // Add stop loss parameter
                start_date: startDate,
                end_date: endDate
            })
        });
        
        if (!backtestResponse.ok) {
            const error = await backtestResponse.text();
            console.error('Backtest error:', error);
            throw new Error(`Backtest failed: ${error}`);
        }
        
        const backtestData = await backtestResponse.json();
        console.log('Backtest data:', backtestData);
        console.log('Indicators received:', backtestData.indicators);
        
        // Check if data is valid
        if (!backtestData.trades || !backtestData.metrics) {
            throw new Error('Invalid backtest data received');
        }
        
        // Update UI
        updateChart(marketData.data, backtestData.trades, strategy);
        updateStats(backtestData.metrics);
        updateTradesList(backtestData.trades);
        updateStrategyPreview(strategy, backtestData.metrics);
        
        // Add indicators to chart
        if (backtestData.indicators) {
            console.log('Updating indicators...');
            updateIndicators(backtestData.indicators);
        } else {
            console.log('No indicators data received');
        }
        
        // Update chart title
        document.getElementById('chart-title').textContent = 
            `${STRATEGY_INFO[strategy].name} - ${currentSymbol.replace(':USDT', '')}`;
        
    } catch (error) {
        console.error('Error running backtest:', error);
        showError(`Failed to run backtest: ${error.message}`);
    } finally {
        showLoading(false);
    }
}

// Create a description of the indicators involved in a trade
function createTradeDescription(strategy, signal) {
    // Definizioni per ogni strategia in base agli indicatori usati
    const descriptions = {
        'rsi_mean_reversion': signal === 1 
            ? "RSI < 30 in risalita" 
            : "RSI > 70 in discesa",
            
        'ema_crossover': signal === 1 
            ? "EMA20 sopra EMA50" 
            : "EMA20 sotto EMA50",
            
        'breakout_range': signal === 1 
            ? "Breakout sopra max 20 periodi" 
            : "Breakdown sotto min 20 periodi",
            
        'bollinger_rebound': signal === 1 
            ? "Rimbalzo banda inferiore" 
            : "Rimbalzo banda superiore",
            
        'macd_histogram': signal === 1 
            ? "MACD istogramma positivo" 
            : "MACD istogramma negativo",
            
        'donchian_breakout': signal === 1 
            ? "Breakout canale Donchian sup" 
            : "Breakdown canale Donchian inf",
            
        'adx_filter_crossover': signal === 1 
            ? "ADX>20 + EMA20 sopra EMA50" 
            : "ADX>20 + EMA20 sotto EMA50"
    };
    
    return descriptions[strategy] || `${strategy.replace('_', ' ')}`;
}

// Update Chart
function updateChart(data, trades, strategy) {
    // Check if chart is initialized properly
    if (!chart || !candleSeries || !volumeSeries) {
        console.error("Chart not properly initialized");
        // Reinitialize chart
        initializeChart();
        
        // If still not initialized, show error and return
        if (!chart || !candleSeries || !volumeSeries) {
            showError("Failed to initialize chart. Please refresh the page.");
            return;
        }
    }
    
    // Validate data before setting
    if (!Array.isArray(data) || data.length === 0) {
        showError("Invalid or empty data received");
        return;
    }
    
    console.log('Chart data range:', new Date(data[0].time * 1000), 'to', new Date(data[data.length-1].time * 1000));
    console.log('Number of trades:', trades.length);
    
    try {
        // Set candlestick data
        candleSeries.setData(data);
        
        // Set volume data
        const volumeData = data.map(d => ({
            time: d.time,
            value: d.volume,
            color: d.close >= d.open ? '#00ff8833' : '#ff3b3b33'
        }));
        volumeSeries.setData(volumeData);
        
        // Clear previous markers
        candleSeries.setMarkers([]);
    } catch (error) {
        console.error("Error updating chart:", error);
        showError("Failed to update chart: " + error.message);
        return;
    }
    
    // Add trade markers
    try {
        // Validate trades data
        if (!Array.isArray(trades)) {
            console.error("Invalid trades data:", trades);
            trades = [];
        }
        
        const newMarkers = [];
        trades.forEach((trade, index) => {
            if (!trade.entry_time || !trade.exit_time) {
                console.warn("Invalid trade data:", trade);
                return; // Skip this trade
            }
            
            console.log(`Trade ${index}:`, new Date(trade.entry_time * 1000), trade.signal === 1 ? 'LONG' : 'SHORT');
            
            // Create descriptive text based on strategy
            const strategyInfo = createTradeDescription(strategy, trade.signal);
            
            // Entry marker with indicator info
            newMarkers.push({
                time: trade.entry_time,
                position: trade.signal === 1 ? 'belowBar' : 'aboveBar',
                color: trade.signal === 1 ? '#00ff88' : '#ff3b3b',
                shape: trade.signal === 1 ? 'arrowUp' : 'arrowDown',
                text: (trade.signal === 1 ? 'BUY' : 'SELL') + ` (${strategyInfo})`
            });
            
            // Exit marker
            const exitColor = trade.pnl > 0 ? '#00ff88' : '#ff3b3b';
            newMarkers.push({
                time: trade.exit_time,
                position: trade.signal === 1 ? 'aboveBar' : 'belowBar',
                color: exitColor,
                shape: 'circle',
                text: 'EXIT'
            });
        });
        
        console.log('Setting markers:', newMarkers.length);
        candleSeries.setMarkers(newMarkers);
        
        // Ensure we show the full data range
        setTimeout(() => {
            chart.timeScale().fitContent();
        }, 100);
        
    } catch (error) {
        console.error("Error setting markers:", error);
        // Don't show error to user here as the chart data is already displayed
    }
}

// Update Statistics
function updateStats(metrics) {
    const statsHtml = `
        <div class="stat">
            <span class="stat-label">Total Return</span>
            <span class="stat-value ${metrics.total_return_pct >= 0 ? 'positive' : 'negative'}">
                ${metrics.total_return_pct >= 0 ? '+' : ''}${((metrics.total_return_pct || 0) * currentLeverage).toFixed(2)}%
            </span>
        </div>
        <div class="stat">
            <span class="stat-label">Trades</span>
            <span class="stat-value">${metrics.total_trades || 0}</span>
        </div>
        <div class="stat">
            <span class="stat-label">Win Rate</span>
            <span class="stat-value">${(metrics.win_rate || 0).toFixed(1)}%</span>
        </div>
        <div class="stat">
            <span class="stat-label">Sharpe</span>
            <span class="stat-value">${(metrics.sharpe_ratio || 0).toFixed(2)}</span>
        </div>
        <div class="stat">
            <span class="stat-label">Max DD</span>
            <span class="stat-value negative">-${(metrics.max_drawdown_pct || 0).toFixed(1)}%</span>
        </div>
    `;
    
    document.getElementById('chart-stats').innerHTML = statsHtml;
}

// Update Trades List
function updateTradesList(trades) {
    const tradesHtml = `
        <h3>Recent Trades (${trades.length} total)</h3>
        <div class="trades-list">
            ${trades.slice(-10).reverse().map(trade => `
                <div class="trade-item ${trade.signal === 1 ? 'long' : 'short'}">
                    <span class="trade-type">${trade.signal === 1 ? 'LONG' : 'SHORT'}</span>
                    <span>$${(trade.entry_price || 0).toFixed(2)}</span>
                    <span>→</span>
                    <span>$${(trade.exit_price || 0).toFixed(2)}</span>
                    <span class="trade-profit ${trade.pnl >= 0 ? 'positive' : 'negative'}">
                        ${trade.pnl >= 0 ? '+' : ''}$${(trade.leveraged_pnl || 0).toFixed(2)}
                        (${trade.pnl >= 0 ? '+' : ''}${((trade.pnl_pct || 0) * currentLeverage).toFixed(2)}%)
                    </span>
                </div>
            `).join('')}
        </div>
    `;
    
    document.getElementById('trades-timeline').innerHTML = tradesHtml;
}

// Update Strategy Preview
function updateStrategyPreview(strategy, metrics) {
    const preview = document.getElementById(`preview-${strategy}`);
    preview.innerHTML = `
        <span>Return: <span class="${metrics.total_return_pct >= 0 ? 'positive' : 'negative'}">
            ${metrics.total_return_pct >= 0 ? '+' : ''}${((metrics.total_return_pct || 0) * currentLeverage).toFixed(1)}%
        </span></span>
        <span>Win: ${(metrics.win_rate || 0).toFixed(0)}%</span>
    `;
}

// Compare All Strategies
async function compareAllStrategies() {
    showLoading(true);
    
    try {
        const startDate = document.getElementById('start-date').value;
        const endDate = document.getElementById('end-date').value;
        
        const response = await fetch(`${API_URL}/api/compare`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                symbol: currentSymbol,
                leverage: currentLeverage,
                take_profit_pct: currentTakeProfit,
                stop_loss_pct: currentStopLoss, // Add stop loss parameter
                start_date: startDate,
                end_date: endDate
            })
        });
        
        const data = await response.json();
        showComparisonModal(data);
        
    } catch (error) {
        console.error('Error comparing strategies:', error);
        showError('Failed to compare strategies');
    } finally {
        showLoading(false);
    }
}

// Show Comparison Modal
function showComparisonModal(data) {
    const tableHtml = `
        <table class="comparison-table">
            <thead>
                <tr>
                    <th>Strategy</th>
                    <th>Return</th>
                    <th>Trades</th>
                    <th>Win Rate</th>
                    <th>Sharpe</th>
                    <th>Max DD</th>
                    <th>Profit Factor</th>
                </tr>
            </thead>
            <tbody>
                ${data.comparison.map(row => `
                    <tr>
                        <td class="strategy-name">${STRATEGY_INFO[row.strategy].name}</td>
                        <td class="${row.leveraged_return_pct >= 0 ? 'positive' : 'negative'}">
                            ${row.leveraged_return_pct >= 0 ? '+' : ''}${(row.leveraged_return_pct || 0).toFixed(2)}%
                        </td>
                        <td>${row.total_trades || 0}</td>
                        <td>${(row.win_rate || 0).toFixed(1)}%</td>
                        <td>${(row.sharpe_ratio || 0).toFixed(2)}</td>
                        <td class="negative">-${(row.max_drawdown_pct || 0).toFixed(1)}%</td>
                        <td>${(row.profit_factor || 0).toFixed(2)}</td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;
    
    document.getElementById('comparison-table').innerHTML = tableHtml;
    document.getElementById('comparison-modal').classList.add('active');
}

// Close Comparison Modal
function closeComparisonModal() {
    document.getElementById('comparison-modal').classList.remove('active');
}

// Show Loading
function showLoading(show) {
    document.getElementById('loading').classList.toggle('active', show);
}

// Show Error
function showError(message) {
    alert(message); // TODO: Implement better error handling
}

// Update Indicators
function updateIndicators(indicators) {
    console.log('updateIndicators called with:', indicators);
    
    // Clear previous indicators
    clearIndicators();
    
    // Process each indicator
    Object.entries(indicators).forEach(([name, indicatorData]) => {
        console.log(`Processing indicator ${name}:`, indicatorData);
        const config = indicatorData.config;
        const data = indicatorData.data;
        
        if (config.panel === 'main') {
            console.log(`Adding overlay indicator ${name} to main chart`);
            // Add overlay indicators to main chart
            addOverlayIndicator(name, data, config);
        } else if (config.panel === 'separate') {
            console.log(`Adding oscillator indicator ${name} to separate panel`);
            // Add oscillator indicators to separate panel
            addOscillatorIndicator(name, data, config);
        }
    });
}

// Clear all indicators
function clearIndicators() {
    // Remove overlay indicators from main chart
    Object.values(indicatorSeries).forEach(series => {
        if (chart && series) {
            chart.removeSeries(series);
        }
    });
    indicatorSeries = {};
    
    // Remove oscillator charts
    Object.entries(oscillatorCharts).forEach(([name, oscChart]) => {
        if (oscChart) {
            oscChart.remove();
            const panel = document.getElementById(`oscillator-${name}`);
            if (panel) {
                panel.remove();
            }
        }
    });
    oscillatorCharts = {};
    indicatorPanels = {};
}

// Add overlay indicator to main chart
function addOverlayIndicator(name, data, config) {
    if (!chart) return;
    
    try {
        // Create line series with proper configuration
        const seriesOptions = {
            color: config.color || '#2196f3',
            lineWidth: config.lineWidth || 2,
            priceLineVisible: false,
            lastValueVisible: false
        };
        
        // Add line style if specified
        if (config.dashStyle === 'dash') {
            seriesOptions.lineStyle = 2; // Dashed line
        }
        
        const series = chart.addLineSeries(seriesOptions);
        
        // Set opacity if specified
        if (config.opacity) {
            series.applyOptions({
                color: config.color + Math.round(config.opacity * 255).toString(16).padStart(2, '0')
            });
        }
        
        // Set the data
        series.setData(data);
        
        // Store the series reference
        indicatorSeries[name] = series;
    } catch (error) {
        console.error(`Error adding overlay indicator ${name}:`, error);
    }
}

// Add oscillator indicator to separate panel
function addOscillatorIndicator(name, data, config) {
    try {
        // Create container for oscillator
        const container = document.createElement('div');
        container.id = `oscillator-${name}`;
        container.className = 'oscillator-panel';
        container.style.height = '150px';
        container.style.marginTop = '10px';
        
        // Add to chart area
        const chartArea = document.querySelector('.chart-area');
        chartArea.appendChild(container);
        
        // Create oscillator chart
        const oscChart = LightweightCharts.createChart(container, {
            layout: {
                background: { color: '#141414' },
                textColor: '#ffffff',
            },
            grid: {
                vertLines: { color: '#2a2a2a' },
                horzLines: { color: '#2a2a2a' },
            },
            timeScale: {
                visible: false,
            },
            rightPriceScale: {
                borderColor: '#2a2a2a',
            }
        });
        
        // Sync time scale with main chart
        chart.timeScale().subscribeVisibleTimeRangeChange(timeRange => {
            oscChart.timeScale().setVisibleRange(timeRange);
        });
        
        oscChart.timeScale().subscribeVisibleTimeRangeChange(timeRange => {
            chart.timeScale().setVisibleRange(timeRange);
        });
        
        // Add appropriate series based on style
        let series;
        if (config.style === 'histogram') {
            series = oscChart.addHistogramSeries({
                color: config.color || '#4caf50',
                priceLineVisible: false,
            });
        } else {
            series = oscChart.addLineSeries({
                color: config.color || '#2196f3',
                lineWidth: config.lineWidth || 2,
                priceLineVisible: false,
            });
        }
        
        // Set the data
        series.setData(data);
        
        // Add levels if specified (e.g., RSI 30/70 levels)
        if (config.levels && config.levels.length > 0) {
            config.levels.forEach((level, index) => {
                series.createPriceLine({
                    price: level,
                    color: config.level_colors ? config.level_colors[index] : '#666',
                    lineWidth: 1,
                    lineStyle: 2, // Dashed
                    axisLabelVisible: true,
                });
            });
        }
        
        // Add title to panel
        const title = document.createElement('div');
        title.className = 'oscillator-title';
        title.style.position = 'absolute';
        title.style.top = '5px';
        title.style.left = '10px';
        title.style.color = '#999';
        title.style.fontSize = '12px';
        title.style.zIndex = '10';
        title.textContent = name.toUpperCase();
        container.appendChild(title);
        
        // Store references
        oscillatorCharts[name] = oscChart;
        indicatorPanels[name] = container;
        
        // Fit content
        oscChart.timeScale().fitContent();
        
    } catch (error) {
        console.error(`Error adding oscillator indicator ${name}:`, error);
    }
}

// Make closeComparisonModal available globally
window.closeComparisonModal = closeComparisonModal;

// Initialize Update Data Button
function initializeUpdateDataButton() {
    const updateBtn = document.getElementById('updateDataBtn');
    if (updateBtn) {
        updateBtn.addEventListener('click', showUpdateModal);
    }
}

// Show Update Modal
function showUpdateModal() {
    document.getElementById('update-modal').classList.add('active');
    
    // Ensure download start date is set to 2024-01-01 by default
    const downloadStartDate = document.getElementById('download-start-date');
    if (downloadStartDate && (!downloadStartDate.value || downloadStartDate.value === "")) {
        downloadStartDate.value = "2024-01-01";
    }
    
    // Resize modal to give more space to logs
    resizeUpdateModal();
    
    // Assicuriamoci che i pulsanti "Start Update" e "Cancel" siano visibili subito
    const updateActionsDiv = document.querySelector('.update-actions');
    if (updateActionsDiv) {
        updateActionsDiv.style.display = 'flex';
        updateActionsDiv.style.justifyContent = 'center';
        updateActionsDiv.style.marginTop = '20px';
        updateActionsDiv.style.marginBottom = '20px';
    }
    
    // Assicurarsi che il contenuto della modale sia visualizzato correttamente
    setTimeout(() => {
        // Scorri automaticamente verso il basso per mostrare i pulsanti
        const modalBody = document.querySelector('.modal-body');
        if (modalBody) {
            modalBody.scrollTop = 0; // Prima resettiamo lo scroll
            
            // Dopo un breve delay, scorri fino a visualizzare i pulsanti
            setTimeout(() => {
                const updateOptions = document.querySelector('.update-options');
                if (updateOptions) {
                    const optionsHeight = updateOptions.offsetHeight;
                    modalBody.scrollTop = optionsHeight - 50; // Scroll sufficiente per vedere i pulsanti
                }
            }, 100);
        }
    }, 50);
}

// Close Update Modal
function closeUpdateModal() {
    document.getElementById('update-modal').classList.remove('active');
    
    // Se l'aggiornamento è stato completato con successo, aggiorna il date range
    const progressStatus = document.getElementById('progress-status');
    if (progressStatus && progressStatus.textContent.includes("completed successfully")) {
        updateDateRangeFromDatabase();
    }
}

// Aggiorna il date range in base ai dati più recenti nel database
async function updateDateRangeFromDatabase() {
    try {
        const response = await fetch(`${API_URL}/api/data-info`);
        if (!response.ok) {
            throw new Error('Failed to fetch data info');
        }
        
        const data = await response.json();
        if (!data || !data.data_info || !data.data_info['1h']) {
            console.log('No 1h data found in database');
            return;
        }
        
        // Trova la data più recente disponibile
        let latestDate = "2024-01-01"; // Default
        let earliestDate = "2024-01-01"; // Default
        
        // Utilizza i dati del timeframe 1h (il più comune)
        const symbolsData = data.data_info['1h'].symbols;
        
        for (const symbolData of symbolsData) {
            if (symbolData.last_date && symbolData.last_date > latestDate) {
                latestDate = symbolData.last_date.substring(0, 10); // Solo la parte della data
            }
            
            if (symbolData.first_date && (!earliestDate || symbolData.first_date < earliestDate)) {
                earliestDate = symbolData.first_date.substring(0, 10); // Solo la parte della data
            }
        }
        
        console.log(`Updating date range from database: ${earliestDate} to ${latestDate}`);
        
        // Aggiorna i campi data nell'interfaccia
        const startDateInput = document.getElementById('start-date');
        const endDateInput = document.getElementById('end-date');
        
        if (startDateInput && earliestDate) {
            startDateInput.value = earliestDate;
            startDateInput.min = earliestDate;
        }
        
        if (endDateInput && latestDate) {
            endDateInput.value = latestDate;
            endDateInput.max = latestDate;
        }
        
    } catch (error) {
        console.error('Error updating date range:', error);
    }
}

// Start Data Update
async function startDataUpdate() {
    // Get selected timeframes only (only checkboxes with name="timeframe")
    const timeframes = [];
    document.querySelectorAll('#update-modal .checkbox-group input[name="timeframe"]:checked').forEach(cb => {
        if (cb.value && ['1m', '5m', '15m', '30m', '1h', '4h', '1d'].includes(cb.value)) {
            timeframes.push(cb.value);
        }
    });
    
    // Logging for debugging
    console.log("Selected timeframes:", timeframes);
    
    // Calculate days from the selected download start date until today
    const startDate = new Date(document.getElementById('download-start-date').value);
    const today = new Date();
    const diffTime = Math.abs(today - startDate);
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
    
    const numSymbols = parseInt(document.getElementById('num-symbols').value);
    // Use calculated days from the download start date
    const days = diffDays;
    const sequential = document.getElementById('sequential').checked;
    const noTa = document.getElementById('no-ta').checked;
    
    console.log(`Downloading data from ${startDate.toLocaleDateString()} to today (${diffDays} days)`);
    
    // Hide options and show progress
    document.querySelector('.update-options').style.display = 'none';
    document.querySelector('.update-actions').style.display = 'none';
    document.getElementById('update-progress').style.display = 'block';
    
    try {
        // Start update
        const response = await fetch(`${API_URL}/api/update-data`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                timeframes: timeframes,
                num_symbols: numSymbols,
                days: days,
                sequential: sequential,
                no_ta: noTa
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            console.error('Update failed:', errorData);
            
            // If update is already in progress, try to reset and retry
            if (response.status === 400 && errorData.detail === "Update already in progress") {
                console.log('Update already in progress, trying to reset...');
                
                // Try to reset the state
                const resetResponse = await fetch(`${API_URL}/api/reset-update-state`, {
                    method: 'POST'
                });
                
                if (resetResponse.ok) {
                    console.log('State reset successful, retrying update...');
                    
                    // Retry the update
                    const retryResponse = await fetch(`${API_URL}/api/update-data`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            timeframes: timeframes,
                            num_symbols: numSymbols,
                            days: days,
                            sequential: sequential,
                            no_ta: noTa
                        })
                    });
                    
                    if (!retryResponse.ok) {
                        throw new Error('Failed to start update after reset');
                    }
                } else {
                    throw new Error('Failed to reset update state');
                }
            } else {
                throw new Error(errorData.detail || 'Failed to start update');
            }
        }
        
        // Start polling for status
        pollUpdateStatus();
        
    } catch (error) {
        console.error('Error starting update:', error);
        
        // Show error in progress area instead of closing modal
        const progressStatus = document.getElementById('progress-status');
        progressStatus.innerHTML = `<span style="color: #ff3b3b;">✗ ${error.message}</span>`;
        
        // Add retry button
        setTimeout(() => {
            const retryBtn = document.createElement('button');
            retryBtn.className = 'btn-primary';
            retryBtn.textContent = 'Retry';
            retryBtn.onclick = () => {
                // Reset UI
                document.querySelector('.update-options').style.display = 'block';
                document.querySelector('.update-actions').style.display = 'flex';
                document.getElementById('update-progress').style.display = 'none';
                document.getElementById('progress-fill').style.width = '0%';
                document.getElementById('progress-logs').textContent = '';
            };
            document.getElementById('update-progress').appendChild(retryBtn);
        }, 500);
    }
}

// Poll Update Status
let updatePollInterval = null;

async function pollUpdateStatus() {
    // Clear any existing interval
    if (updatePollInterval) {
        clearInterval(updatePollInterval);
    }
    
    const progressFill = document.getElementById('progress-fill');
    const progressStatus = document.getElementById('progress-status');
    const progressLogs = document.getElementById('progress-logs');
    
    updatePollInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_URL}/api/update-status`);
            const status = await response.json();
            
            // Update progress bar
            progressFill.style.width = `${status.progress}%`;
            
            // Update status text
            progressStatus.textContent = status.status;
            
            // Update logs (show all lines with ANSI color support)
            if (status.logs && status.logs.length > 0) {
                // Convert logs to HTML with ANSI color support
                const formattedLogs = formatLogsWithAnsiColors(status.logs);
                progressLogs.innerHTML = formattedLogs;
                progressLogs.scrollTop = progressLogs.scrollHeight;
            }
            
            // Check if completed
            if (!status.is_running && status.progress === 100) {
                clearInterval(updatePollInterval);
                updatePollInterval = null;
                
                // Show success message
                progressStatus.innerHTML = '<span style="color: #00ff88;">✓ Update completed successfully!</span>';
                
                // Add close button
                setTimeout(() => {
                    const closeBtn = document.createElement('button');
                    closeBtn.className = 'btn-primary';
                    closeBtn.textContent = 'Close';
                    closeBtn.onclick = () => {
                        closeUpdateModal();
                        // Reset modal state
                        document.querySelector('.update-options').style.display = 'block';
                        document.querySelector('.update-actions').style.display = 'flex';
                        document.getElementById('update-progress').style.display = 'none';
                        progressFill.style.width = '0%';
                        progressLogs.textContent = '';
                    };
                    document.getElementById('update-progress').appendChild(closeBtn);
                }, 1000);
            } else if (!status.is_running && status.status.includes('Error')) {
                // Handle error
                clearInterval(updatePollInterval);
                updatePollInterval = null;
                progressStatus.innerHTML = '<span style="color: #ff3b3b;">✗ ' + status.status + '</span>';
            }
            
        } catch (error) {
            console.error('Error polling status:', error);
        }
    }, 1000); // Poll every second
}

/**
 * Format logs with ANSI color support and detect tables/sections
 * @param {string[]} logs - Array of log lines
 * @return {string} HTML formatted logs
 */
function formatLogsWithAnsiColors(logs) {
    // Join all logs into a single string
    let logText = logs.join('\n');
    
    // Replace ANSI color codes with span elements with appropriate CSS classes
    const ansiColorMap = {
        // Regular colors
        '\u001b[30m': '<span class="ansi-black">',    // Black
        '\u001b[31m': '<span class="ansi-red">',      // Red
        '\u001b[32m': '<span class="ansi-green">',    // Green
        '\u001b[33m': '<span class="ansi-yellow">',   // Yellow
        '\u001b[34m': '<span class="ansi-blue">',     // Blue
        '\u001b[35m': '<span class="ansi-magenta">',  // Magenta
        '\u001b[36m': '<span class="ansi-cyan">',     // Cyan
        '\u001b[37m': '<span class="ansi-white">',    // White
        
        // Bright colors
        '\u001b[90m': '<span class="ansi-bright-black">',    // Bright Black
        '\u001b[91m': '<span class="ansi-bright-red">',      // Bright Red
        '\u001b[92m': '<span class="ansi-bright-green">',    // Bright Green
        '\u001b[93m': '<span class="ansi-bright-yellow">',   // Bright Yellow
        '\u001b[94m': '<span class="ansi-bright-blue">',     // Bright Blue
        '\u001b[95m': '<span class="ansi-bright-magenta">',  // Bright Magenta
        '\u001b[96m': '<span class="ansi-bright-cyan">',     // Bright Cyan
        '\u001b[97m': '<span class="ansi-bright-white">',    // Bright White
        
        // Background colors
        '\u001b[40m': '<span class="ansi-bg-black">',   // Background Black
        '\u001b[41m': '<span class="ansi-bg-red">',     // Background Red
        '\u001b[42m': '<span class="ansi-bg-green">',   // Background Green
        '\u001b[43m': '<span class="ansi-bg-yellow">',  // Background Yellow
        '\u001b[44m': '<span class="ansi-bg-blue">',    // Background Blue
        '\u001b[45m': '<span class="ansi-bg-magenta">', // Background Magenta
        '\u001b[46m': '<span class="ansi-bg-cyan">',    // Background Cyan
        '\u001b[47m': '<span class="ansi-bg-white">',   // Background White
        
        // Reset
        '\u001b[0m': '</span>'
    };
    
    // Replace all ANSI codes with their HTML equivalents
    for (const [ansiCode, htmlTag] of Object.entries(ansiColorMap)) {
        // Escape special characters in the ANSI code for use in regex
        const escapedCode = ansiCode.replace(/[\[\]]/g, '\\$&');
        const regex = new RegExp(escapedCode, 'g');
        logText = logText.replace(regex, htmlTag);
    }
    
    // Detect and format section headers (lines with ======)
    logText = logText.replace(/^(=+)([^=]+)(=+)$/gm, '<div class="log-section-title">$2</div>');
    
    // Detect and format tables
    // First, look for lines with multiple | characters which indicates a table structure
    let lines = logText.split('\n');
    let inTable = false;
    let tableContent = '';
    let formattedLines = [];
    
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        
        // Check if this line could be part of a table (contains multiple | or has many - characters)
        if (line.includes('|') && line.split('|').length > 2) {
            // If we weren't in a table before, start a new table
            if (!inTable) {
                inTable = true;
                tableContent = '<div class="log-table">';
            }
            
            // Check if this is a header separator line (---+---+---)
            if (line.match(/^[\s\-\+]+$/)) {
                // Skip separator lines
                continue;
            }
            
            // Add this line as a table row
            tableContent += '<div class="log-table-row">';
            const cells = line.split('|').filter(Boolean);
            
            for (const cell of cells) {
                tableContent += `<span class="log-table-cell">${cell.trim()}</span>`;
            }
            
            tableContent += '</div>';
        } else if (inTable) {
            // We were in a table but this line is not a table row, so end the table
            inTable = false;
            tableContent += '</div>';
            formattedLines.push(tableContent);
            formattedLines.push(line);
        } else {
            // Not a table line and not in a table
            formattedLines.push(line);
        }
    }
    
    // If we were still in a table at the end, close it
    if (inTable) {
        tableContent += '</div>';
        formattedLines.push(tableContent);
    }
    
    // Detect and format special sections like "RESOCONTO AGGIORNAMENTO DATI COMPLETATO"
    let finalHtml = formattedLines.join('\n');
    finalHtml = finalHtml.replace(/(RESOCONTO[\s\S]*?DATI[\s\S]*?COMPLETATO)/g, '<div class="log-section">$1</div>');
    finalHtml = finalHtml.replace(/(STATISTICHE PER TIMEFRAME)/g, '<div class="log-section-title">$1</div>');
    
    // Ensure all ANSI spans are properly closed
    let openSpans = (finalHtml.match(/<span class="ansi-[^"]+?">/g) || []).length;
    let closeSpans = (finalHtml.match(/<\/span>/g) || []).length;
    let spanDiff = openSpans - closeSpans;
    
    if (spanDiff > 0) {
        finalHtml += '</span>'.repeat(spanDiff);
    }
    
    return finalHtml;
}

// Resize the update modal to give more space to logs
function resizeUpdateModal() {
    const updateModal = document.querySelector('.update-modal');
    if (updateModal) {
        updateModal.style.maxWidth = '90%';
        updateModal.style.width = '90%';
        updateModal.style.maxHeight = '90vh';
    }
}

// Make functions available globally
window.closeUpdateModal = closeUpdateModal;
window.startDataUpdate = startDataUpdate;
window.resizeUpdateModal = resizeUpdateModal;
