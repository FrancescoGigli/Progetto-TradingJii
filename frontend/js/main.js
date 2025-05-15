/**
 * Main Application Orchestrator for TradingJii Dashboard
 */
import { state } from './modules/state.js';
import { elements, initDynamicElements } from './modules/dom-elements.js';
import * as api from './modules/api.js';
import * as ui from './modules/ui-updater.js';
import * as chartManager from './modules/chart-manager.js';
import { setupAllEventListeners, initializeEventHandlers } from './modules/event-listeners.js';

// --- Main Application Logic ---

/**
 * Load all data for a specific cryptocurrency and timeframe, then update UI.
 * @param {string} symbol - Cryptocurrency symbol
 * @param {string} timeframe - Timeframe ('5m', '15m')
 */
async function loadCryptoData(symbol, timeframe) {
    ui.showLoadingOverlay(true); // Show loading state
    state.currentSymbol = symbol; // Update state early for UI consistency
    state.currentTimeframe = timeframe;
    ui.updateCryptoDetailsHeader(); // Update header (symbol name, timeframe badge)

    try {
        // Fetch all data in parallel
        // Note: API functions already update state.currentData
        const [ohlcvData, volatilityData, patternData, indicatorData] = await Promise.all([
            api.fetchOHLCVData(symbol, timeframe),
            api.fetchVolatilityData(symbol, timeframe),
            api.fetchPatternData(symbol, timeframe),
            api.fetchIndicatorData(symbol, timeframe)
        ]);

        console.log('All data fetched for', symbol, timeframe, { ohlcvData, volatilityData, patternData, indicatorData });

        // Determine which main chart is active to render it
        const isPriceChartActive = !elements.priceChartWrapper.classList.contains('hidden');
        const isVolatilityChartActive = !elements.volatilityChartWrapper.classList.contains('hidden');

        if (isPriceChartActive && ohlcvData && ohlcvData.length > 0) {
            chartManager.createPriceChart(symbol, timeframe, ohlcvData, state.currentPriceChartStyle);
        } else if (isVolatilityChartActive && volatilityData && volatilityData.length > 0) {
            chartManager.createVolatilityChart(symbol, timeframe, volatilityData);
        } else if (ohlcvData && ohlcvData.length > 0) { 
            // Default to price chart if neither explicitly active but data exists
            ui.togglePriceVolatilityCharts('price'); // Ensure price chart is visible
            chartManager.createPriceChart(symbol, timeframe, ohlcvData, state.currentPriceChartStyle);
        }


        if (patternData) {
            chartManager.renderPatternVisualization(symbol, timeframe, patternData);
        }

        // If an indicator is selected, render its chart
        if (state.currentIndicator !== 'none' && indicatorData && indicatorData.length > 0) {
            ui.toggleIndicatorChartVisibility(true);
            chartManager.createIndicatorChart(symbol, timeframe, indicatorData, state.currentIndicator);
        } else {
            ui.toggleIndicatorChartVisibility(false);
        }
        
        ui.updateActiveSymbolUI(symbol);

    } catch (error) {
        console.error(`Error loading all crypto data for ${symbol}:`, error);
        ui.showErrorModal(`Failed to load data for ${symbol}. ${error.message}`);
    } finally {
        ui.showLoadingOverlay(false); // Hide loading state
        // Ensure charts are resized after all operations
        setTimeout(() => {
            chartManager.resizePriceChart();
            chartManager.resizeVolatilityChart();
            chartManager.resizeIndicatorChart();
        }, 150);
    }
}

/**
 * Refresh all data for the current symbol and timeframe.
 */
async function refreshAllData() {
    if (state.currentSymbol) {
        console.log(`Refreshing data for ${state.currentSymbol} (${state.currentTimeframe})`);
        await loadCryptoData(state.currentSymbol, state.currentTimeframe);
    } else {
        console.log('No current symbol to refresh. Loading symbol list instead.');
        await loadSymbols();
    }
}

/**
 * Load the list of symbols and render them.
 */
async function loadSymbols() {
    ui.showLoadingOverlay(true);
    const symbols = await api.fetchSymbols(); // fetchSymbols updates state.symbols
    ui.renderCryptoList(symbols);
    ui.showLoadingOverlay(false);
}

/**
 * Initialize the application
 */
async function initApp() {
    console.log('Initializing TradingJii App...');
    
    // Initialize dynamic DOM elements if any are created by other scripts before this.
    // For now, zoom controls are handled within event-listeners.js after DOM content loaded.
    // initDynamicElements(); 
    
    // Pass main control functions to event listeners module
    initializeEventHandlers({
        loadCryptoData: loadCryptoData,
        refreshAllData: refreshAllData,
        loadSymbols: loadSymbols
    });

    // Pass main loadCryptoData to uiUpdater for crypto list item clicks
    ui.setLoadCryptoDataHandler(loadCryptoData);
    
    // Pass the UI's showErrorModal to the API module
    api.setShowErrorFunction(ui.showErrorModal);

    // Setup all event listeners
    setupAllEventListeners(); // This also loads theme preference

    // Initial load of symbols
    await loadSymbols();

    // Hide loading overlay (should be already hidden by default CSS or early script)
    ui.showLoadingOverlay(false);
    document.body.style.overflow = ''; // Ensure scrollbars are enabled

    console.log('TradingJii App Initialized.');
}

// --- Application Entry Point ---
document.addEventListener('DOMContentLoaded', initApp);
