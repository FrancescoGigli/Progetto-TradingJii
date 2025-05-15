/**
 * Event Listeners Module
 *
 * Sets up and manages all primary event listeners for the application.
 */
import { elements } from './dom-elements.js';
import { state } from './state.js';
import * as ui from './ui-updater.js';
import * as chartManager from './chart-manager.js';

// Handler functions to be set by main.js or another orchestrator module
let mainLoadCryptoData;
let mainRefreshData;
let mainLoadSymbols;

export function initializeEventHandlers(appController) {
    mainLoadCryptoData = appController.loadCryptoData;
    mainRefreshData = appController.refreshAllData; // Assumes a general refresh function
    mainLoadSymbols = appController.loadSymbols;   // Assumes a function to load symbols
}


function handleTimeframeChange(event) {
    const newTimeframe = event.target.value;
    if (newTimeframe !== state.currentTimeframe) {
        state.currentTimeframe = newTimeframe;
        if (state.currentSymbol) {
            mainLoadCryptoData(state.currentSymbol, state.currentTimeframe);
        }
    }
}

function handleIndicatorChange(event) {
    const indicatorType = event.target.value;
    state.currentIndicator = indicatorType;

    if (!state.currentSymbol) return;

    ui.toggleIndicatorChartVisibility(indicatorType !== 'none');

    if (indicatorType !== 'none' && state.currentData.indicators) {
        chartManager.createIndicatorChart(state.currentSymbol, state.currentTimeframe, state.currentData.indicators, indicatorType);
    }
    
    // Resize main chart (price or volatility)
    if (!elements.priceChartWrapper.classList.contains('hidden')) {
        chartManager.resizePriceChart();
    } else if (!elements.volatilityChartWrapper.classList.contains('hidden')) {
        chartManager.resizeVolatilityChart();
    }
    if (indicatorType !== 'none') {
        chartManager.resizeIndicatorChart();
    }
}

function handleThemeToggle() {
    const newTheme = state.theme === 'dark' ? 'light' : 'dark';
    ui.applyTheme(newTheme);
    // Re-render charts if a symbol is selected, as colors might change
    if (state.currentSymbol) {
        // Small delay to allow theme transition CSS to apply before chart redraw
        setTimeout(() => {
            mainLoadCryptoData(state.currentSymbol, state.currentTimeframe);
        }, 50);
    }
}

function handleRefreshClick() {
    if (state.currentSymbol) {
        mainRefreshData(); // Or mainLoadCryptoData if refresh implies full reload
    } else {
        mainLoadSymbols();
    }
}

function handleChartTypeToggle(event) {
    const chartType = event.target.dataset.chart; // 'price' or 'volatility'
    
    elements.chartToggles.forEach(toggle => toggle.classList.remove('active'));
    event.target.classList.add('active');

    ui.togglePriceVolatilityCharts(chartType);

    // Resize the newly shown chart
    setTimeout(() => {
        if (chartType === 'price') {
            chartManager.resizePriceChart();
        } else if (chartType === 'volatility') {
            chartManager.resizeVolatilityChart();
        }
    }, 100); // After CSS transition
}

function handlePriceStyleToggle(event) {
    const newStyle = event.target.dataset.style;
    if (newStyle === state.currentPriceChartStyle) return;

    state.currentPriceChartStyle = newStyle;
    elements.priceStyleToggles.forEach(toggle => toggle.classList.remove('active'));
    event.target.classList.add('active');

    if (state.currentSymbol && state.currentData.ohlcv) {
        if (!elements.priceChartWrapper.classList.contains('hidden')) {
            chartManager.createPriceChart(
                state.currentSymbol,
                state.currentTimeframe,
                state.currentData.ohlcv,
                newStyle
            );
        }
    }
}

function handleSearchInput(event) {
    const searchTerm = event.target.value.toLowerCase();
    const filteredSymbols = state.symbols.filter(s => s.toLowerCase().includes(searchTerm));
    ui.renderCryptoList(filteredSymbols.length > 0 || searchTerm === '' ? filteredSymbols : state.symbols);
}

function setupZoomControls() {
    const chartControls = document.querySelector('.chart-controls');
    if (!chartControls.querySelector('.chart-controls-right')) { // Avoid duplicating
        const chartControlsRight = document.createElement('div');
        chartControlsRight.className = 'chart-controls-right';

        const zoomInfoBtn = document.createElement('button');
        zoomInfoBtn.className = 'zoom-info-btn';
        zoomInfoBtn.innerHTML = '<i class="fas fa-info-circle"></i>';
        zoomInfoBtn.title = 'Zoom instructions';

        const resetZoomBtn = document.createElement('button');
        resetZoomBtn.id = 'reset-zoom-btn'; // Used by dom-elements.js
        resetZoomBtn.innerHTML = '<i class="fas fa-search-minus"></i> Reset Zoom';
        resetZoomBtn.className = 'reset-zoom-btn';
        resetZoomBtn.title = 'Reset chart zoom';
        resetZoomBtn.style.display = 'none';

        chartControlsRight.appendChild(zoomInfoBtn);
        chartControlsRight.appendChild(resetZoomBtn);
        chartControls.appendChild(chartControlsRight);
        
        // Update elements reference if it was initially null
        if (!elements.resetZoomBtn) elements.resetZoomBtn = resetZoomBtn;


        const zoomInstructions = document.createElement('div');
        zoomInstructions.className = 'zoom-instructions';
        zoomInstructions.innerHTML = `<h4>Chart Zoom Controls</h4><ul><li><strong>Zoom In/Out:</strong> Hold Ctrl + Mouse Wheel</li><li><strong>Pan Chart:</strong> Hold Shift + Mouse Drag</li><li><strong>Reset Zoom:</strong> Click Reset Zoom</li></ul>`;
        zoomInstructions.style.display = 'none';
        document.querySelector('.chart-container').appendChild(zoomInstructions);

        zoomInfoBtn.addEventListener('click', () => {
            zoomInstructions.style.display = zoomInstructions.style.display === 'none' ? 'block' : 'none';
        });

        resetZoomBtn.addEventListener('click', () => {
            const priceChart = chartManager.getPriceChartInstance();
            const volChart = chartManager.getVolatilityChartInstance();
            const indChart = chartManager.getIndicatorChartInstance();

            if (priceChart && !elements.priceChartWrapper.classList.contains('hidden') && priceChart.resetZoom) priceChart.resetZoom();
            if (volChart && !elements.volatilityChartWrapper.classList.contains('hidden') && volChart.resetZoom) volChart.resetZoom();
            if (indChart && !elements.indicatorChartWrapper.classList.contains('hidden') && indChart.resetZoom) indChart.resetZoom();
            
            resetZoomBtn.style.display = 'none';
        });

        window.addEventListener('wheel', (e) => {
            if (e.ctrlKey && elements.resetZoomBtn) {
                elements.resetZoomBtn.style.display = 'block';
            }
        }, { passive: true });
    }
}


export function setupAllEventListeners() {
    if (elements.timeframeSelect) elements.timeframeSelect.addEventListener('change', handleTimeframeChange);
    if (elements.indicatorSelect) elements.indicatorSelect.addEventListener('change', handleIndicatorChange);
    if (elements.themeToggle) elements.themeToggle.addEventListener('click', handleThemeToggle);
    if (elements.refreshButton) elements.refreshButton.addEventListener('click', handleRefreshClick);
    if (elements.searchInput) elements.searchInput.addEventListener('input', handleSearchInput);
    
    elements.chartToggles.forEach(toggle => toggle.addEventListener('click', handleChartTypeToggle));
    elements.priceStyleToggles.forEach(toggle => toggle.addEventListener('click', handlePriceStyleToggle));
    
    if (elements.closeModalBtn) elements.closeModalBtn.addEventListener('click', ui.hideErrorModal);

    setupZoomControls(); // Setup for dynamically added zoom controls
    
    // Load initial theme preference
    ui.loadThemePreference();
}
