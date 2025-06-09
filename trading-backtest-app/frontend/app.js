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
    
    // Update end date to match database
    document.getElementById('end-date').value = '2025-06-09';
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
        
        // Check if data is valid
        if (!backtestData.trades || !backtestData.metrics) {
            throw new Error('Invalid backtest data received');
        }
        
        // Update UI
        updateChart(marketData.data, backtestData.trades);
        updateStats(backtestData.metrics);
        updateTradesList(backtestData.trades);
        updateStrategyPreview(strategy, backtestData.metrics);
        
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

// Update Chart
function updateChart(data, trades) {
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
        trades.forEach(trade => {
            if (!trade.entry_time || !trade.exit_time) {
                console.warn("Invalid trade data:", trade);
                return; // Skip this trade
            }
            
            // Entry marker
            newMarkers.push({
                time: trade.entry_time,
                position: trade.signal === 1 ? 'belowBar' : 'aboveBar',
                color: trade.signal === 1 ? '#00ff88' : '#ff3b3b',
                shape: trade.signal === 1 ? 'arrowUp' : 'arrowDown',
                text: trade.signal === 1 ? 'L' : 'S'
            });
            
            // Exit marker
            const exitColor = trade.pnl > 0 ? '#00ff88' : '#ff3b3b';
            newMarkers.push({
                time: trade.exit_time,
                position: 'aboveBar',
                color: exitColor,
                shape: 'circle',
                text: trade.pnl > 0 ? '+' : '-'
            });
        });
        
        candleSeries.setMarkers(newMarkers);
        
        // Fit content
        chart.timeScale().fitContent();
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

// Make closeComparisonModal available globally
window.closeComparisonModal = closeComparisonModal;
