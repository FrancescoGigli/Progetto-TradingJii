/**
 * UI Updater Module
 *
 * Handles general UI updates, excluding chart rendering.
 */
import { elements } from './dom-elements.js';
import { state } from './state.js';
// import { loadCryptoData } from './main.js'; // This will be the main app logic, careful with circular deps

// This function will be set by main.js or an event listener module that has access to loadCryptoData
let loadCryptoDataHandler = async (symbol, timeframe) => {
    console.warn('loadCryptoDataHandler not yet implemented in ui-updater');
};
export function setLoadCryptoDataHandler(handler) {
    loadCryptoDataHandler = handler;
}

/**
 * Render the cryptocurrency list in the sidebar
 * @param {Array} symbols - Array of cryptocurrency symbols to render
 */
export function renderCryptoList(symbolsToRender) {
    elements.cryptoList.innerHTML = ''; // Clear existing list
    
    if (!symbolsToRender || symbolsToRender.length === 0) {
        elements.cryptoList.innerHTML = '<li class="loading">No cryptocurrencies found</li>';
        elements.cryptoCount.textContent = '0';
        return;
    }
    
    elements.cryptoCount.textContent = symbolsToRender.length;
    
    symbolsToRender.forEach(symbol => {
        const listItem = document.createElement('li');
        listItem.textContent = symbol;
        listItem.dataset.symbol = symbol;
        
        listItem.addEventListener('click', () => {
            // Call the main data loading function when a symbol is clicked
            loadCryptoDataHandler(symbol, state.currentTimeframe);
        });
        
        elements.cryptoList.appendChild(listItem);
    });

    // If a current symbol is selected, ensure it's marked active
    if (state.currentSymbol) {
        updateActiveSymbolUI(state.currentSymbol);
    }
}

/**
 * Update the active state of the cryptocurrency list in the UI
 * @param {string} symbol - The currently selected symbol
 */
export function updateActiveSymbolUI(symbol) {
    document.querySelectorAll('#crypto-list li').forEach(item => {
        item.classList.remove('active');
    });
    
    const activeItem = document.querySelector(`#crypto-list li[data-symbol="${symbol}"]`);
    if (activeItem) {
        activeItem.classList.add('active');
        activeItem.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
}

/**
 * Update the displayed crypto name and timeframe badge.
 */
export function updateCryptoDetailsHeader() {
    if (state.currentSymbol) {
        elements.cryptoNameText.textContent = state.currentSymbol;
        elements.currentTimeframeBadge.textContent = state.currentTimeframe;
        elements.welcomeMessage.style.display = 'none';
        elements.cryptoDetails.style.display = 'block';
        elements.cryptoDetails.classList.remove('hidden');
    } else {
        elements.welcomeMessage.style.display = 'flex'; // Or initial display type
        elements.cryptoDetails.style.display = 'none';
        elements.cryptoDetails.classList.add('hidden');
    }
}


/**
 * Show or hide the loading overlay
 * @param {boolean} show - Whether to show or hide the overlay
 */
export function showLoadingOverlay(show) {
    // elements.loadingOverlay.classList.toggle('hidden', !show);
    // document.body.style.overflow = show ? 'hidden' : '';
    // Per script.js, loading overlay is now permanently hidden to fix issues
    elements.loadingOverlay.classList.add('hidden');
    document.body.style.overflow = '';
}

/**
 * Show an error message in the modal
 * @param {string} message - The error message to display
 */
export function showErrorModal(message) {
    elements.errorMessageText.textContent = message;
    elements.errorModal.style.display = 'flex'; // Show modal
    elements.errorModal.classList.add('active');
}

/**
 * Hide the error modal
 */
export function hideErrorModal() {
    elements.errorModal.style.display = 'none';
    elements.errorModal.classList.remove('active');
}

/**
 * Apply theme to the body and save preference.
 * @param {string} themeName - 'light' or 'dark'
 */
export function applyTheme(themeName) {
    if (themeName === 'light') {
        document.body.classList.add('light-theme');
    } else {
        document.body.classList.remove('light-theme');
    }
    document.body.classList.add('transition-theme'); // For smooth transition
    state.theme = themeName;
    try {
        localStorage.setItem('tradingJii_theme', themeName);
    } catch (error) {
        console.error('Could not save theme preference:', error);
    }
    // Remove transition class after animation to prevent conflicts
    setTimeout(() => document.body.classList.remove('transition-theme'), 300);
}

/**
 * Load theme preference from local storage and apply it.
 */
export function loadThemePreference() {
    try {
        const savedTheme = localStorage.getItem('tradingJii_theme');
        if (savedTheme) {
            applyTheme(savedTheme); // Apply it without re-saving
        } else {
            applyTheme(state.theme); // Apply default theme
        }
    } catch (error) {
        console.error('Could not load theme preference:', error);
        applyTheme(state.theme); // Apply default in case of error
    }
}

/**
 * Toggle chart visibility for Price vs Volatility
 * @param {'price' | 'volatility'} chartToShow 
 */
export function togglePriceVolatilityCharts(chartToShow) {
    const priceWrapper = elements.priceChartWrapper;
    const volatilityWrapper = elements.volatilityChartWrapper;
    const volumeWrapper = elements.volumeChartWrapper; // Added volume wrapper

    // Update active state of toggle buttons (this is handled by an event listener module)

    if (chartToShow === 'price') {
        volatilityWrapper.classList.add('hidden');
        volatilityWrapper.style.opacity = '0'; 
        
        // Show price and volume charts
        priceWrapper.classList.remove('hidden');
        priceWrapper.style.opacity = '1';
        if (volumeWrapper) { // Check if volumeWrapper exists
            volumeWrapper.classList.remove('hidden');
            volumeWrapper.style.opacity = '1';
        }
        // Resizing will be handled by main.js or event-listeners.js after data load / toggle
        
    } else if (chartToShow === 'volatility') {
        priceWrapper.classList.add('hidden');
        priceWrapper.style.opacity = '0';
        if (volumeWrapper) { // Check if volumeWrapper exists
            volumeWrapper.classList.add('hidden');
            volumeWrapper.style.opacity = '0';
        }
        
        volatilityWrapper.classList.remove('hidden');
        volatilityWrapper.style.opacity = '1';
        // Resizing will be handled by main.js or event-listeners.js
    }
    // Note: The actual chart resizing (e.g., chartManager.resizePriceChart()) 
    // should be called after these UI changes, typically in the event handler in event-listeners.js
    // or after data loading in main.js, to ensure charts resize to their new container dimensions.
}

/**
 * Show/Hide the indicator chart section and adjust main chart area.
 * @param {boolean} show - True to show indicator chart, false to hide.
 */
export function toggleIndicatorChartVisibility(show) {
    const indicatorWrapper = elements.indicatorChartWrapper;
    const priceWrapper = elements.priceChartWrapper;
    const volatilityWrapper = elements.volatilityChartWrapper;
    const volumeWrapper = elements.volumeChartWrapper; // Added volume wrapper

    if (show) {
        indicatorWrapper.classList.remove('hidden');
        indicatorWrapper.style.opacity = '1';
        priceWrapper.classList.add('with-indicator');
        volatilityWrapper.classList.add('with-indicator');
        if (volumeWrapper) volumeWrapper.classList.add('with-indicator'); // Add to volume too
    } else {
        indicatorWrapper.classList.add('hidden');
        indicatorWrapper.style.opacity = '0';
        priceWrapper.classList.remove('with-indicator');
        volatilityWrapper.classList.remove('with-indicator');
        if (volumeWrapper) volumeWrapper.classList.remove('with-indicator'); // Remove from volume too
    }
    // Resizing of charts will be handled by chart-manager after this UI update
}
