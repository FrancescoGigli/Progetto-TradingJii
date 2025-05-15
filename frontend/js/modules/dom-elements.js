/**
 * DOM Elements Module
 *
 * Centralizes references to DOM elements used throughout the application.
 */

export const elements = {
    // Lists and containers
    cryptoList: document.getElementById('crypto-list'),
    cryptoCount: document.getElementById('crypto-count'),
    welcomeMessage: document.getElementById('welcome-message'),
    cryptoDetails: document.getElementById('crypto-details'),
    
    // Charts and visualization
    priceChartWrapper: document.getElementById('price-chart-wrapper'),
    priceChartCanvas: document.getElementById('price-chart'), // Added canvas ref
    volumeChartCanvas: document.getElementById('volume-chart'), // May need to be created dynamically still
    volatilityChartWrapper: document.getElementById('volatility-chart-wrapper'),
    volatilityChartCanvas: document.getElementById('volatility-chart'), // Added canvas ref
    indicatorChartWrapper: document.getElementById('indicator-chart-wrapper'),
    indicatorChartCanvas: document.getElementById('indicator-chart'), // Added canvas ref
    patternInfo: document.getElementById('pattern-info'),
    patternVisualization: document.getElementById('pattern-visualization'),
    
    // Controls
    timeframeSelect: document.getElementById('timeframe-select'),
    indicatorSelect: document.getElementById('indicator-select'),
    themeToggle: document.getElementById('theme-toggle'),
    refreshButton: document.getElementById('refresh-btn'),
    searchInput: document.getElementById('search-crypto'),
    chartToggles: document.querySelectorAll('.chart-toggle'),
    priceStyleToggles: document.querySelectorAll('.price-style-toggle'),
    
    // Dynamic elements (elements whose content changes)
    cryptoNameText: document.getElementById('crypto-name'), // Renamed for clarity
    currentTimeframeBadge: document.getElementById('current-timeframe'),
    
    // Overlays and modals
    loadingOverlay: document.getElementById('loading-overlay'),
    errorModal: document.getElementById('error-modal'),
    errorMessageText: document.getElementById('error-message'), // Renamed for clarity
    closeModalBtn: document.getElementById('close-modal'),

    // Zoom related (some are dynamically added by script.js, might need adjustment)
    resetZoomBtn: document.getElementById('reset-zoom-btn'), // This is dynamically added
    zoomInfoBtnContainer: null, // Placeholder, will be dynamically created
    zoomInstructionsPanel: null // Placeholder, will be dynamically created
};

// Function to initialize dynamically created elements if needed,
// or ensure they are selected after creation.
export function initDynamicElements() {
    // Example: if resetZoomBtn is not found initially, it might be because it's added later.
    // This function could be called after such elements are added to the DOM.
    if (!elements.resetZoomBtn) {
        elements.resetZoomBtn = document.getElementById('reset-zoom-btn');
    }
    // Similar checks for other dynamically added elements like zoomInfoBtnContainer and zoomInstructionsPanel
    // For now, these are handled in event-listeners.js or main.js where they are created.
}
