/**
 * Global Application State Module
 */

export const state = {
    symbols: [],
    currentSymbol: null,
    currentTimeframe: '5m', // Default timeframe
    currentData: {
        ohlcv: null,
        volatility: null,
        patterns: null,
        indicators: null
    },
    currentIndicator: 'none',
    currentPriceChartStyle: 'heikin-ashi', // Default price chart style
    theme: 'dark' // Default theme
};

// Optional: Add functions to update state if more complex logic is needed
// export function setCurrentSymbol(symbol) {
//     state.currentSymbol = symbol;
// }
// export function setCurrentTimeframe(timeframe) {
//     state.currentTimeframe = timeframe;
// }
// ... etc.
