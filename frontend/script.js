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
        patterns: null
    },
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
    patternInfo: document.getElementById('pattern-info'),
    patternVisualization: document.getElementById('pattern-visualization'),
    
    // Controls
    timeframeSelect: document.getElementById('timeframe-select'),
    themeToggle: document.getElementById('theme-toggle'),
    refreshButton: document.getElementById('refresh-btn'),
    searchInput: document.getElementById('search-crypto'),
    chartToggles: document.querySelectorAll('.chart-toggle'),
    
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
    patterns: (symbol, timeframe) => `${API_BASE_URL}/patterns/${symbol}/${timeframe}`
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
    
    try {
        // Fetch all data in parallel
        const [ohlcvData, volatilityData, patternData] = await Promise.all([
            fetchOHLCVData(symbol, timeframe),
            fetchVolatilityData(symbol, timeframe),
            fetchPatternData(symbol, timeframe)
        ]);
        
        console.log('Data fetched successfully:', {
            ohlcvLength: ohlcvData?.length || 0,
            volatilityLength: volatilityData?.length || 0,
            patternCount: patternData ? Object.keys(patternData).length : 0
        });
        
        // Create charts and visualizations
        if (ohlcvData && ohlcvData.length > 0) {
            console.log('Creating price chart');
            ChartHandler.createPriceChart(symbol, timeframe, ohlcvData);
        }
        
        if (volatilityData && volatilityData.length > 0) {
            console.log('Creating volatility chart');
            ChartHandler.createVolatilityChart(symbol, timeframe, volatilityData);
        }
        
        if (patternData) {
            console.log('Rendering pattern visualization');
            ChartHandler.renderPatternVisualization(symbol, timeframe, patternData);
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
