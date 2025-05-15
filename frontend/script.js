/**
 * Main JavaScript for TradingJii Dashboard
 * 
 * This script handles:
 * - API requests to the backend
 * - User interactions
 * - UI state management
 * - Cryptocurrency list population
 * - Chart display toggling
 */

// Global state
const state = {
    symbols: [],
    currentSymbol: null,
    currentTimeframe: '5m',
    currentData: {
        ohlcv: null,
        volatility: null,
        patterns: null,
        indicators: null
    },
    currentIndicator: 'none',
    currentPriceChartStyle: 'heikin-ashi', // Added for price chart style
    theme: 'dark' // Store the current theme
};

// DOM references
const elements = {
    // Lists and containers
    cryptoList: document.getElementById('crypto-list'),
    cryptoCount: document.getElementById('crypto-count'),
    welcomeMessage: document.getElementById('welcome-message'),
    cryptoDetails: document.getElementById('crypto-details'),
    
    // Charts and visualization
    priceChartWrapper: document.getElementById('price-chart-wrapper'),
    volatilityChartWrapper: document.getElementById('volatility-chart-wrapper'),
    indicatorChartWrapper: document.getElementById('indicator-chart-wrapper'),
    patternInfo: document.getElementById('pattern-info'),
    patternVisualization: document.getElementById('pattern-visualization'),
    
    // Controls
    timeframeSelect: document.getElementById('timeframe-select'),
    indicatorSelect: document.getElementById('indicator-select'),
    themeToggle: document.getElementById('theme-toggle'),
    refreshButton: document.getElementById('refresh-btn'),
    searchInput: document.getElementById('search-crypto'),
    chartToggles: document.querySelectorAll('.chart-toggle'),
    priceStyleToggles: document.querySelectorAll('.price-style-toggle'), // Added for price style toggles
    
    // Dynamic elements
    cryptoName: document.getElementById('crypto-name'),
    currentTimeframeBadge: document.getElementById('current-timeframe'),
    
    // Overlays and modals
    loadingOverlay: document.getElementById('loading-overlay'),
    errorModal: document.getElementById('error-modal'),
    errorMessage: document.getElementById('error-message'),
    closeModalBtn: document.getElementById('close-modal')
};

// API endpoints
const API_BASE_URL = '/api';
const API = {
    symbols: `${API_BASE_URL}/symbols`,
    ohlcv: (symbol, timeframe) => `${API_BASE_URL}/ohlcv/${symbol}/${timeframe}`,
    volatility: (symbol, timeframe) => `${API_BASE_URL}/volatility/${symbol}/${timeframe}`,
    patterns: (symbol, timeframe) => `${API_BASE_URL}/patterns/${symbol}/${timeframe}`,
    indicators: (symbol, timeframe) => `${API_BASE_URL}/indicators/${symbol}/${timeframe}`
};

/**
 * Initialize the application
 */
function initApp() {
    // Force hide loading overlay
    elements.loadingOverlay.classList.add('hidden');
    document.body.style.overflow = '';
    
    // Load the list of available symbols
    fetchSymbols();
    
    // Setup event listeners
    setupEventListeners();
}

/**
 * Setup all event listeners for the application
 */
function setupEventListeners() {
    // Timeframe selection
    elements.timeframeSelect.addEventListener('change', handleTimeframeChange);
    
    // Indicator selection
    elements.indicatorSelect.addEventListener('change', handleIndicatorChange);
    
    // Theme toggle
    if (elements.themeToggle) {
        elements.themeToggle.addEventListener('click', handleThemeToggle);
        
        // Load saved theme preference
        loadThemePreference();
    }
    
    // Refresh button
    elements.refreshButton.addEventListener('click', handleRefresh);
    
    // Chart toggle buttons
    elements.chartToggles.forEach(toggle => {
        toggle.addEventListener('click', handleChartToggle);
    });

    // Price style toggle buttons
    elements.priceStyleToggles.forEach(toggle => {
        toggle.addEventListener('click', handlePriceStyleToggle);
    });
    
    // Create chart controls container with zoom controls
        const chartControls = document.querySelector('.chart-controls');
        const chartControlsRight = document.createElement('div');
        chartControlsRight.className = 'chart-controls-right';
        
        // Create zoom info button
        const zoomInfoBtn = document.createElement('button');
        zoomInfoBtn.className = 'zoom-info-btn';
        zoomInfoBtn.innerHTML = '<i class="fas fa-info-circle"></i>';
        zoomInfoBtn.title = 'Zoom instructions';
        
        // Create reset zoom button
        const resetZoomBtn = document.createElement('button');
        resetZoomBtn.id = 'reset-zoom-btn';
        resetZoomBtn.innerHTML = '<i class="fas fa-search-minus"></i> Reset Zoom';
        resetZoomBtn.className = 'reset-zoom-btn';
        resetZoomBtn.title = 'Reset chart zoom';
        resetZoomBtn.style.display = 'none'; // Hidden by default
        
        // Add buttons to container
        chartControlsRight.appendChild(zoomInfoBtn);
        chartControlsRight.appendChild(resetZoomBtn);
        chartControls.appendChild(chartControlsRight);
        
        // Create zoom instructions tooltip
        const zoomInstructions = document.createElement('div');
        zoomInstructions.className = 'zoom-instructions';
        zoomInstructions.innerHTML = `
            <h4>Chart Zoom Controls</h4>
            <ul>
                <li><strong>Zoom In/Out:</strong> Hold Ctrl + Mouse Wheel</li>
                <li><strong>Pan Chart:</strong> Hold Shift + Mouse Drag</li>
                <li><strong>Reset Zoom:</strong> Click the Reset Zoom button</li>
            </ul>
        `;
        zoomInstructions.style.display = 'none';
        document.querySelector('.chart-container').appendChild(zoomInstructions);
        
        // Toggle zoom instructions when info button is clicked
        zoomInfoBtn.addEventListener('click', function() {
            const instructions = document.querySelector('.zoom-instructions');
            instructions.style.display = instructions.style.display === 'none' ? 'block' : 'none';
        });
        
        // Add event listener to reset zoom button
        resetZoomBtn.addEventListener('click', function() {
            if (window.ChartHandler) {
                // Reset zoom for either active chart
                if (!document.getElementById('price-chart-wrapper').classList.contains('hidden')) {
                    if (window.priceChart && window.priceChart.resetZoom) {
                        window.priceChart.resetZoom();
                    }
                } else if (!document.getElementById('volatility-chart-wrapper').classList.contains('hidden')) {
                    if (window.volatilityChart && window.volatilityChart.resetZoom) {
                        window.volatilityChart.resetZoom();
                    }
                }
                this.style.display = 'none'; // Hide button after reset
            }
        });
        
        // Add window event listener for zoom events
        window.addEventListener('wheel', function(e) {
            if (e.ctrlKey) {
                // Show reset zoom button when user zooms with Ctrl+wheel
                document.getElementById('reset-zoom-btn').style.display = 'block';
            }
        }, { passive: true });
    
    // Search input
    elements.searchInput.addEventListener('input', handleSearch);
    
    // Close error modal
    elements.closeModalBtn.addEventListener('click', () => {
        elements.errorModal.classList.remove('active');
    });
}

/**
 * Handle theme toggle button clicks
 */
function handleThemeToggle() {
    // Toggle between light and dark theme
    if (document.body.classList.contains('light-theme')) {
        document.body.classList.remove('light-theme');
        document.body.classList.add('transition-theme');
        state.theme = 'dark';
    } else {
        document.body.classList.add('light-theme');
        document.body.classList.add('transition-theme');
        state.theme = 'light';
    }
    
    // Save theme preference
    saveThemePreference();
    
    // Update charts if they exist
    if (state.currentSymbol) {
        setTimeout(() => {
            loadCryptoData(state.currentSymbol, state.currentTimeframe);
        }, 300);
    }
}

/**
 * Save theme preference to local storage
 */
function saveThemePreference() {
    try {
        localStorage.setItem('tradingJii_theme', state.theme);
    } catch (error) {
        console.error('Could not save theme preference:', error);
    }
}

/**
 * Load theme preference from local storage
 */
function loadThemePreference() {
    try {
        const savedTheme = localStorage.getItem('tradingJii_theme');
        if (savedTheme) {
            state.theme = savedTheme;
            if (savedTheme === 'light') {
                document.body.classList.add('light-theme');
            }
        }
    } catch (error) {
        console.error('Could not load theme preference:', error);
    }
}


/**
 * Fetch OHLCV data for a specific cryptocurrency and timeframe
 * @param {string} symbol - Cryptocurrency symbol
 * @param {string} timeframe - Timeframe ('5m', '15m')
 */
async function fetchOHLCVData(symbol, timeframe) {
    try {
        const response = await fetch(API.ohlcv(symbol, timeframe));
        const result = await response.json();
        
        if (result.status === 'success') {
            state.currentData.ohlcv = result.data;
            return result.data;
        } else {
            throw new Error(result.message || 'Failed to fetch OHLCV data');
        }
    } catch (error) {
        showError(`Error fetching OHLCV data: ${error.message}`);
        return null;
    }
}

/**
 * Fetch volatility data for a specific cryptocurrency and timeframe
 * @param {string} symbol - Cryptocurrency symbol
 * @param {string} timeframe - Timeframe ('5m', '15m')
 */
async function fetchVolatilityData(symbol, timeframe) {
    try {
        const response = await fetch(API.volatility(symbol, timeframe));
        const result = await response.json();
        
        if (result.status === 'success') {
            state.currentData.volatility = result.data;
            return result.data;
        } else {
            throw new Error(result.message || 'Failed to fetch volatility data');
        }
    } catch (error) {
        showError(`Error fetching volatility data: ${error.message}`);
        return null;
    }
}

/**
 * Fetch pattern data for a specific cryptocurrency and timeframe
 * @param {string} symbol - Cryptocurrency symbol
 * @param {string} timeframe - Timeframe ('5m', '15m')
 */
async function fetchPatternData(symbol, timeframe) {
    try {
        const response = await fetch(API.patterns(symbol, timeframe));
        const result = await response.json();
        
        if (result.status === 'success') {
            state.currentData.patterns = result.data;
            return result.data;
        } else {
            throw new Error(result.message || 'Failed to fetch pattern data');
        }
    } catch (error) {
        showError(`Error fetching pattern data: ${error.message}`);
        return null;
    }
}

/**
 * Fetch technical indicator data for a specific cryptocurrency and timeframe
 * @param {string} symbol - Cryptocurrency symbol
 * @param {string} timeframe - Timeframe ('5m', '15m')
 */
async function fetchIndicatorData(symbol, timeframe) {
    try {
        const response = await fetch(API.indicators(symbol, timeframe));
        const result = await response.json();
        
        if (result.status === 'success') {
            state.currentData.indicators = result.data;
            return result.data;
        } else {
            throw new Error(result.message || 'Failed to fetch indicator data');
        }
    } catch (error) {
        showError(`Error fetching indicator data: ${error.message}`);
        return null;
    }
}

/**
 * Load all data for a specific cryptocurrency and timeframe
 * @param {string} symbol - Cryptocurrency symbol
 * @param {string} timeframe - Timeframe ('5m', '15m')
 */
async function loadCryptoData(symbol, timeframe) {
    // Don't show loading overlay to avoid issues
    // Just handle things in the background
    
    // Make sure we immediately update the UI before fetching data
    console.log(`Loading data for ${symbol} (${timeframe})`);
    
    // Update UI for the selected symbol
    elements.cryptoName.textContent = symbol;
    elements.currentTimeframeBadge.textContent = timeframe;
    
    // Force show crypto details and hide welcome message with !important styling
    elements.welcomeMessage.setAttribute('style', 'display: none !important');
    elements.cryptoDetails.setAttribute('style', 'display: block !important');
    elements.cryptoDetails.classList.remove('hidden');
    
    // Make chart containers visible
    document.querySelector('.chart-container').setAttribute('style', 'display: block !important; height: 400px !important;');
    elements.priceChartWrapper.setAttribute('style', 'display: block !important; height: 100% !important;');
    document.querySelector('.content').setAttribute('style', 'display: block !important; width: 100% !important;');
    document.querySelector('.app-container').setAttribute('style', 'display: flex !important; width: 100% !important;');

    // Hide indicator chart initially
    elements.indicatorChartWrapper.classList.add('hidden');
    
    try {
        // Fetch all data in parallel
        const [ohlcvData, volatilityData, patternData, indicatorData] = await Promise.all([
            fetchOHLCVData(symbol, timeframe),
            fetchVolatilityData(symbol, timeframe),
            fetchPatternData(symbol, timeframe),
            fetchIndicatorData(symbol, timeframe)
        ]);
        
        console.log('Data fetched successfully:', {
            ohlcvLength: ohlcvData?.length || 0,
            volatilityLength: volatilityData?.length || 0,
            patternCount: patternData ? Object.keys(patternData).length : 0,
            indicatorDataLength: indicatorData?.length || 0
        });
        
        // Create charts and visualizations
        if (ohlcvData && ohlcvData.length > 0) {
            console.log('Creating price chart with style:', state.currentPriceChartStyle);
            ChartHandler.createPriceChart(symbol, timeframe, ohlcvData, state.currentPriceChartStyle);
        }
        
        if (volatilityData && volatilityData.length > 0) {
            console.log('Creating volatility chart');
            ChartHandler.createVolatilityChart(symbol, timeframe, volatilityData);
        }
        
        if (patternData) {
            console.log('Rendering pattern visualization');
            ChartHandler.renderPatternVisualization(symbol, timeframe, patternData);
        }

    // If we have a selected indicator that's not "none", display it
    // Ensure indicatorData is passed correctly
    if (state.currentIndicator !== 'none' && state.currentData.indicators && state.currentData.indicators.length > 0) {
        showIndicatorChart(symbol, timeframe, state.currentData.indicators, state.currentIndicator);
    }
        
        // Update state
        state.currentSymbol = symbol;
        state.currentTimeframe = timeframe;
        
        // Update active class on list item
        updateActiveSymbol(symbol);
        
    } catch (error) {
        console.error('Error loading data:', error);
        showError(`Error loading cryptocurrency data: ${error.message}`);
    }
}

/**
 * Handle indicator selection change
 * @param {Event} event - The change event
 */
function handleIndicatorChange(event) {
    const indicatorType = event.target.value;
    state.currentIndicator = indicatorType;
    
    // If we don't have a selected symbol, nothing to do
    if (!state.currentSymbol) return;
    
    if (indicatorType === 'none') {
        // Hide indicator chart
        elements.indicatorChartWrapper.classList.add('hidden');
        
        // Remove the 'with-indicator' class from price and volatility charts
        elements.priceChartWrapper.classList.remove('with-indicator');
        elements.volatilityChartWrapper.classList.remove('with-indicator');
        
        // Force redraw the active chart
        if (!elements.priceChartWrapper.classList.contains('hidden')) {
            ChartHandler.resizePriceChart();
        } else if (!elements.volatilityChartWrapper.classList.contains('hidden')) {
            ChartHandler.resizeVolatilityChart();
        }
        
        return;
    }
    
    showIndicatorChart(state.currentSymbol, state.currentTimeframe, state.currentData.indicators, indicatorType);
}

/**
 * Show the indicator chart based on selected indicator type
 * @param {string} symbol - Cryptocurrency symbol
 * @param {string} timeframe - Timeframe
 * @param {Array} data - Indicator data
 * @param {string} indicatorType - Type of indicator to display
 */
function showIndicatorChart(symbol, timeframe, data, indicatorType) {
    if (!data || data.length === 0) {
        console.warn('No indicator data available');
        return;
    }
    
    console.log(`Showing ${indicatorType} chart for ${symbol}`);
    
    // Show indicator chart
    elements.indicatorChartWrapper.classList.remove('hidden');
    
    // Add the 'with-indicator' class to price and volatility charts
    elements.priceChartWrapper.classList.add('with-indicator');
    elements.volatilityChartWrapper.classList.add('with-indicator');
    
    // Create the indicator chart
    ChartHandler.createIndicatorChart(symbol, timeframe, data, indicatorType);
    
    // Force redraw the active chart
    if (!elements.priceChartWrapper.classList.contains('hidden')) {
        ChartHandler.resizePriceChart();
    } else if (!elements.volatilityChartWrapper.classList.contains('hidden')) {
        ChartHandler.resizeVolatilityChart();
    }
}

/**
 * Render the cryptocurrency list in the sidebar
 * @param {Array} symbols - Array of cryptocurrency symbols
 */
function renderCryptoList(symbols) {
    elements.cryptoList.innerHTML = '';
    
    if (!symbols || symbols.length === 0) {
        elements.cryptoList.innerHTML = '<li class="loading">No cryptocurrencies found</li>';
        elements.cryptoCount.textContent = '0';
        return;
    }
    
    // Update the count
    elements.cryptoCount.textContent = symbols.length;
    
    // Create list items
    symbols.forEach(symbol => {
        const listItem = document.createElement('li');
        listItem.textContent = symbol;
        listItem.dataset.symbol = symbol;
        
        // Add click handler
        listItem.addEventListener('click', () => {
            loadCryptoData(symbol, state.currentTimeframe);
        });
        
        elements.cryptoList.appendChild(listItem);
    });
}

/**
 * Update the active state of the cryptocurrency list
 * @param {string} symbol - The currently selected symbol
 */
function updateActiveSymbol(symbol) {
    // Clear all active classes
    document.querySelectorAll('#crypto-list li').forEach(item => {
        item.classList.remove('active');
    });
    
    // Add active class to the current symbol
    const activeItem = document.querySelector(`#crypto-list li[data-symbol="${symbol}"]`);
    if (activeItem) {
        activeItem.classList.add('active');
        
        // Scroll into view if needed
        activeItem.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
}

/**
 * Handle timeframe selection change
 * @param {Event} event - The change event
 */
function handleTimeframeChange(event) {
    const timeframe = event.target.value;
    state.currentTimeframe = timeframe;
    
    // If a cryptocurrency is already selected, reload with new timeframe
    if (state.currentSymbol) {
        loadCryptoData(state.currentSymbol, timeframe);
    }
}

/**
 * Handle refresh button click
 */
function handleRefresh() {
    // If a cryptocurrency is selected, reload its data
    if (state.currentSymbol) {
        loadCryptoData(state.currentSymbol, state.currentTimeframe);
    } else {
        // Otherwise, refresh the symbols list
        fetchSymbols();
    }
}

/**
 * Handle chart toggle button clicks
 * @param {Event} event - The click event
 */
function handleChartToggle(event) {
    const target = event.target;
    const chartType = target.dataset.chart;
    
    // Update active state of toggle buttons
    elements.chartToggles.forEach(toggle => {
        toggle.classList.remove('active');
    });
    target.classList.add('active');
    
    // Handle chart transition with proper timing
    if (chartType === 'price') {
        // First start hiding volatility chart
        elements.volatilityChartWrapper.classList.add('hidden');
        
        // After a small delay to allow for transition, show price chart
        setTimeout(() => {
            // Remove any inline styles that might interfere
            elements.priceChartWrapper.removeAttribute('style');
            elements.priceChartWrapper.classList.remove('hidden');
            
            // Force redraw of the price chart
            if (window.ChartHandler && window.ChartHandler.resizePriceChart) {
                window.ChartHandler.resizePriceChart();
            }
        }, 300);
    } else if (chartType === 'volatility') {
        // First start hiding price chart
        elements.priceChartWrapper.classList.add('hidden');
        
        // After a small delay to allow for transition, show volatility chart
        setTimeout(() => {
            // Remove any inline styles that might interfere
            elements.volatilityChartWrapper.removeAttribute('style');
            elements.volatilityChartWrapper.classList.remove('hidden');
            
            // Force redraw of the volatility chart
            if (window.ChartHandler && window.ChartHandler.resizeVolatilityChart) {
                window.ChartHandler.resizeVolatilityChart();
            }
        }, 300);
    }
}

/**
 * Fetch list of available cryptocurrency symbols from the API
 */
async function fetchSymbols() {
    // Don't show loading overlay to avoid potential issues
    
    try {
        const response = await fetch(API.symbols);
        const result = await response.json();
        
        if (result.status === 'success') {
            state.symbols = result.data;
            renderCryptoList(state.symbols);
        } else {
            throw new Error(result.message || 'Failed to fetch symbols');
        }
    } catch (error) {
        showError(`Error fetching cryptocurrency list: ${error.message}`);
    }
}

/**
 * Handle search input
 * @param {Event} event - The input event
 */
function handleSearch(event) {
    const searchTerm = event.target.value.toLowerCase();
    
    // If search term is empty, show all symbols
    if (!searchTerm) {
        renderCryptoList(state.symbols);
        return;
    }
    
    // Filter symbols by search term
    const filteredSymbols = state.symbols.filter(symbol => 
        symbol.toLowerCase().includes(searchTerm)
    );
    
    // Render the filtered list
    renderCryptoList(filteredSymbols);
}

/**
 * Show or hide the loading overlay
 * @param {boolean} show - Whether to show or hide the overlay
 */
function showLoading(show) {
    // Disable loading overlay completely to fix the issue
    elements.loadingOverlay.classList.add('hidden');
    document.body.style.overflow = '';
}

/**
 * Show an error message in the modal
 * @param {string} message - The error message to display
 */
function showError(message) {
    elements.errorMessage.textContent = message;
    elements.errorModal.classList.add('active');
}

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Immediately hide loading overlay
    elements.loadingOverlay.classList.add('hidden');
    document.body.style.overflow = '';
    
    // Initialize the app
    initApp();
});

/**
 * Handle price style toggle button clicks
 * @param {Event} event - The click event
 */
function handlePriceStyleToggle(event) {
    const target = event.target;
    const newStyle = target.dataset.style;

    if (newStyle === state.currentPriceChartStyle) {
        return; // No change
    }

    state.currentPriceChartStyle = newStyle;

    // Update active state of toggle buttons
    elements.priceStyleToggles.forEach(toggle => {
        toggle.classList.remove('active');
    });
    target.classList.add('active');

    // If a cryptocurrency is selected, reload its price chart with the new style
    if (state.currentSymbol && state.currentData.ohlcv) {
        // Ensure the main price chart is visible before attempting to recreate
        if (!elements.priceChartWrapper.classList.contains('hidden')) {
            console.log(`Recreating price chart with style: ${newStyle}`);
            ChartHandler.createPriceChart(
                state.currentSymbol,
                state.currentTimeframe,
                state.currentData.ohlcv, // Use existing ohlcv data
                newStyle
            );
        } else {
            // If price chart is not active, just store the style. It will be applied when price chart is shown.
            console.log(`Price chart style set to ${newStyle}, will apply when price chart is active.`);
        }
    }
}
