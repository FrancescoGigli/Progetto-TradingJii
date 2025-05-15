/**
 * Configuration Module
 *
 * Stores API endpoint constants and other configuration values.
 */

export const API_BASE_URL = '/api';

export const API_ENDPOINTS = {
    symbols: `${API_BASE_URL}/symbols`,
    ohlcv: (symbol, timeframe) => `${API_BASE_URL}/ohlcv/${symbol}/${timeframe}`,
    volatility: (symbol, timeframe) => `${API_BASE_URL}/volatility/${symbol}/${timeframe}`,
    patterns: (symbol, timeframe) => `${API_BASE_URL}/patterns/${symbol}/${timeframe}`,
    indicators: (symbol, timeframe) => `${API_BASE_URL}/indicators/${symbol}/${timeframe}`
};

// Add any other shared configuration constants here if needed
// export const DEFAULT_TIMEFRAME = '5m';
// export const DEFAULT_PRICE_CHART_STYLE = 'heikin-ashi';
