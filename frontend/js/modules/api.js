/**
 * API Module
 *
 * Handles all communication with the backend API.
 */
import { API_ENDPOINTS } from './config.js';
import { state } from './state.js'; // To store fetched data

// Placeholder for showError - this will eventually be imported from ui-updater.js
// For now, to avoid circular dependencies during refactoring, we can define a simple one
// or rely on it being globally available from the old script.js temporarily.
// A better approach would be to pass it as a dependency or use a global event bus for errors.
let showError = (message) => console.error(message); // Temporary simple showError

export function setShowErrorFunction(fn) {
    showError = fn;
}

/**
 * Fetch list of available cryptocurrency symbols from the API
 */
export async function fetchSymbols() {
    try {
        const response = await fetch(API_ENDPOINTS.symbols);
        const result = await response.json();
        
        if (result.status === 'success') {
            state.symbols = result.data; // Store in global state
            return result.data;
        } else {
            throw new Error(result.message || 'Failed to fetch symbols');
        }
    } catch (error) {
        showError(`Error fetching cryptocurrency list: ${error.message}`);
        return []; // Return empty array on error
    }
}

/**
 * Fetch OHLCV data for a specific cryptocurrency and timeframe
 * @param {string} symbol - Cryptocurrency symbol
 * @param {string} timeframe - Timeframe ('5m', '15m')
 */
export async function fetchOHLCVData(symbol, timeframe) {
    try {
        const response = await fetch(API_ENDPOINTS.ohlcv(symbol, timeframe));
        const result = await response.json();
        
        if (result.status === 'success') {
            state.currentData.ohlcv = result.data; // Store in global state
            return result.data;
        } else {
            throw new Error(result.message || 'Failed to fetch OHLCV data');
        }
    } catch (error) {
        showError(`Error fetching OHLCV data for ${symbol}: ${error.message}`);
        return null;
    }
}

/**
 * Fetch volatility data for a specific cryptocurrency and timeframe
 * @param {string} symbol - Cryptocurrency symbol
 * @param {string} timeframe - Timeframe ('5m', '15m')
 */
export async function fetchVolatilityData(symbol, timeframe) {
    try {
        const response = await fetch(API_ENDPOINTS.volatility(symbol, timeframe));
        const result = await response.json();
        
        if (result.status === 'success') {
            state.currentData.volatility = result.data; // Store in global state
            return result.data;
        } else {
            throw new Error(result.message || 'Failed to fetch volatility data');
        }
    } catch (error) {
        showError(`Error fetching volatility data for ${symbol}: ${error.message}`);
        return null;
    }
}

/**
 * Fetch pattern data for a specific cryptocurrency and timeframe
 * @param {string} symbol - Cryptocurrency symbol
 * @param {string} timeframe - Timeframe ('5m', '15m')
 */
export async function fetchPatternData(symbol, timeframe) {
    try {
        const response = await fetch(API_ENDPOINTS.patterns(symbol, timeframe));
        const result = await response.json();
        
        if (result.status === 'success') {
            state.currentData.patterns = result.data; // Store in global state
            return result.data;
        } else {
            throw new Error(result.message || 'Failed to fetch pattern data');
        }
    } catch (error) {
        showError(`Error fetching pattern data for ${symbol}: ${error.message}`);
        return null;
    }
}

/**
 * Fetch technical indicator data for a specific cryptocurrency and timeframe
 * @param {string} symbol - Cryptocurrency symbol
 * @param {string} timeframe - Timeframe ('5m', '15m')
 */
export async function fetchIndicatorData(symbol, timeframe) {
    try {
        const response = await fetch(API_ENDPOINTS.indicators(symbol, timeframe));
        const result = await response.json();
        
        if (result.status === 'success') {
            state.currentData.indicators = result.data; // Store in global state
            return result.data;
        } else {
            throw new Error(result.message || 'Failed to fetch indicator data');
        }
    } catch (error) {
        showError(`Error fetching indicator data for ${symbol}: ${error.message}`);
        return null;
    }
}
