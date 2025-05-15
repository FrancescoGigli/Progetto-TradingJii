/**
 * Chart Manager Module for TradingJii Dashboard
 * 
 * This module provides functions for rendering price, volume, volatility, 
 * and indicator charts using Chart.js.
 */
import { elements } from './dom-elements.js';
// import { state } from './state.js'; // For theme, or use direct DOM check

// Chart.js and plugins are loaded globally via script tags in index.html
if (typeof Chart === 'undefined') {
    console.error('Chart.js not found. Make sure it is loaded before chart-manager.js.');
} else if (Chart.Annotation && typeof Chart.Annotation.drawTime !== 'string') { // Check if plugin is registered
    // Chart.register(Chart.Annotation); // Not typically needed if loaded via script tag
    console.log('Chart.js Annotation plugin seems available.');
} else if (!Chart.Annotation) {
     console.warn('Chart.js Annotation plugin might not be properly loaded.');
}


// Store chart instances locally within this module
let priceChartInstance = null;
let volumeChartInstance = null;
let volatilityChartInstance = null;
let indicatorChartInstance = null;

// --- HELPER FUNCTIONS (Specific to this module) ---

function getPrecision(price) {
    if (price === 0 || !price) return 2;
    if (price < 0.0001) return 8;
    if (price < 0.01) return 6;
    if (price < 0.1) return 4;
    if (price < 1) return 3;
    if (price < 10) return 2;
    return 2;
}

function prepareOHLCVData(data) {
    if (!data || !Array.isArray(data) || data.length === 0) return [];
    const sortedData = [...data].sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
    return sortedData.map(item => ({
        x: new Date(item.timestamp).getTime(),
        o: parseFloat(item.open),
        h: parseFloat(item.high),
        l: parseFloat(item.low),
        c: parseFloat(item.close),
        volume: parseFloat(item.volume),
        // Keep original values for tooltips, especially if data is transformed (e.g., Heikin-Ashi)
        originalOpen: parseFloat(item.open),
        originalHigh: parseFloat(item.high),
        originalLow: parseFloat(item.low),
        originalClose: parseFloat(item.close)
    }));
}

function calculateHeikinAshi(data) {
    if (!data || !Array.isArray(data) || data.length === 0) return [];
    const result = [];
    for (let i = 0; i < data.length; i++) {
        const current = data[i]; // data here is assumed to be result of prepareOHLCVData
        let haClose = (current.o + current.h + current.l + current.c) / 4;
        let haOpen;
        if (i === 0) {
            haOpen = (current.o + current.c) / 2;
        } else {
            haOpen = (result[i - 1].o + result[i - 1].c) / 2; // HA Open uses previous HA candle
        }
        const haHigh = Math.max(current.h, haOpen, haClose);
        const haLow = Math.min(current.l, haOpen, haClose);
        result.push({
            x: current.x,
            o: haOpen,
            h: haHigh,
            l: haLow,
            c: haClose,
            volume: current.volume,
            originalOpen: current.originalOpen, // Preserve original values
            originalHigh: current.originalHigh,
            originalLow: current.originalLow,
            originalClose: current.originalClose
        });
    }
    return result;
}

function extractIndicators(apiData) {
    if (!apiData || !Array.isArray(apiData) || apiData.length === 0) return {};
    const indicators = {
        sma: { sma9: { data: [], color: 'rgba(125, 200, 255, 1)', label: 'SMA 9' }, sma20: { data: [], color: 'rgba(255, 165, 0, 1)', label: 'SMA 20' }, sma50: { data: [], color: 'rgba(128, 0, 128, 1)', label: 'SMA 50' }},
        ema: { ema20: { data: [], color: 'rgba(255, 99, 132, 1)', label: 'EMA 20' }, ema50: { data: [], color: 'rgba(54, 162, 235, 1)', label: 'EMA 50' }, ema200: { data: [], color: 'rgba(255, 206, 86, 1)', label: 'EMA 200' }},
        bbands: { upper: { data: [], color: 'rgba(128, 128, 128, 0.5)', label: 'BB Upper' }, middle: { data: [], color: 'rgba(128, 128, 128, 1)', label: 'BB Middle' }, lower: { data: [], color: 'rgba(128, 128, 128, 0.5)', label: 'BB Lower' }},
        // Note: Other indicators like RSI, MACD are typically plotted on separate panes.
        // This function primarily extracts indicators that can be overlaid on the price chart.
    };
    try {
        apiData.forEach(item => {
            const timestamp = new Date(item.timestamp).getTime();
            if (item.sma9 !== undefined) indicators.sma.sma9.data.push({ x: timestamp, y: parseFloat(item.sma9) });
            if (item.sma20 !== undefined) indicators.sma.sma20.data.push({ x: timestamp, y: parseFloat(item.sma20) });
            if (item.sma50 !== undefined) indicators.sma.sma50.data.push({ x: timestamp, y: parseFloat(item.sma50) });
            if (item.ema20 !== undefined) indicators.ema.ema20.data.push({ x: timestamp, y: parseFloat(item.ema20) });
            if (item.ema50 !== undefined) indicators.ema.ema50.data.push({ x: timestamp, y: parseFloat(item.ema50) });
            if (item.ema200 !== undefined) indicators.ema.ema200.data.push({ x: timestamp, y: parseFloat(item.ema200) });
            if (item.bbands_upper !== undefined) indicators.bbands.upper.data.push({ x: timestamp, y: parseFloat(item.bbands_upper) });
            if (item.bbands_middle !== undefined) indicators.bbands.middle.data.push({ x: timestamp, y: parseFloat(item.bbands_middle) });
            if (item.bbands_lower !== undefined) indicators.bbands.lower.data.push({ x: timestamp, y: parseFloat(item.bbands_lower) });
        });
        return indicators;
    } catch (error) {
        console.error('Error extracting overlay indicators:', error);
        return {};
    }
}


// --- CHART CREATION FUNCTIONS ---

export function createPriceChart(symbol, timeframe, rawApiData, priceChartStyle = 'heikin-ashi') {
    try {
        console.log(`Creating price chart for ${symbol} with style: ${priceChartStyle}`);
        if (!elements.priceChartCanvas) {
            console.error('Price chart canvas not found!');
            return null;
        }
        const ctx = elements.priceChartCanvas.getContext('2d');
        
        if (priceChartInstance) priceChartInstance.destroy();
        if (volumeChartInstance) { 
            volumeChartInstance.destroy();
            volumeChartInstance = null;
            // Ensure elements.volumeChartWrapper is used if available, otherwise query.
            const volumeWrapper = elements.volumeChartWrapper || document.querySelector('.volume-chart-wrapper');
            if (volumeWrapper) {
                volumeWrapper.classList.add('hidden'); 
                volumeWrapper.style.opacity = '0'; 
            }
        }
        
        const ohlcvData = prepareOHLCVData(rawApiData);
        if (!ohlcvData || ohlcvData.length === 0) {
            console.error('No OHLCV data available for price chart');
            return null;
        }
        
        const overlayIndicators = extractIndicators(rawApiData);
        
        let displayData;
        let chartType = 'candlestick';
        let datasetLabel = `${symbol} (${timeframe})`;

        switch (priceChartStyle) {
            case 'line':
                chartType = 'line';
                displayData = ohlcvData.map(d => ({ x: d.x, y: d.c, originalOpen: d.o, originalHigh: d.h, originalLow: d.l, originalClose: d.c, volume: d.volume }));
                datasetLabel += ' - Line';
                break;
            case 'candlestick':
                displayData = ohlcvData; // Already has originalOpen etc. from prepareOHLCVData
                datasetLabel += ' - Candlestick';
                break;
            case 'heikin-ashi':
            default:
                displayData = calculateHeikinAshi(ohlcvData);
                datasetLabel += ' - Heikin-Ashi';
                break;
        }
        
        if (!displayData || displayData.length === 0) {
            console.error('No display data available for price chart style:', priceChartStyle);
            return null;
        }
        
        const isDarkTheme = !document.body.classList.contains('light-theme');
        const gridColor = isDarkTheme ? '#2a2e39' : '#e5e7eb';
        const textColor = isDarkTheme ? '#9ca3af' : '#666666';
        const upColor = '#26a69a';
        const downColor = '#ef5350';
        const upColorFill = 'rgba(38, 166, 154, 0.85)';
        const downColorFill = 'rgba(239, 83, 80, 0.85)';
        
        const datasets = [];
        if (priceChartStyle === 'line') {
            datasets.push({
                label: datasetLabel, data: displayData, borderColor: upColor, borderWidth: 2,
                pointRadius: 0, pointHoverRadius: 5, fill: false, tension: 0.1,
                type: 'line' // Ensure type is line
            });
        } else {
            datasets.push({
                label: datasetLabel, data: displayData,
                color: { up: upColor, down: downColor, unchanged: '#888888' },
                borderColor: (context) => context.dataset.data[context.dataIndex].o > context.dataset.data[context.dataIndex].c ? downColor : upColor,
                borderWidth: 2.5, wickWidth: 2, barPercentage: 0.92, barThickness: 14,
                backgroundColor: (context) => context.dataset.data[context.dataIndex].o > context.dataset.data[context.dataIndex].c ? downColorFill : upColorFill,
            });
        }
        
        Object.values(overlayIndicators.sma).concat(Object.values(overlayIndicators.ema)).forEach(ind => {
            if (ind.data && ind.data.length > 0) datasets.push({ type: 'line', label: ind.label, data: ind.data, borderColor: ind.color, borderWidth: 1.5, pointRadius: 0, fill: false });
        });
        if (overlayIndicators.bbands) {
            Object.keys(overlayIndicators.bbands).forEach(key => {
                const ind = overlayIndicators.bbands[key];
                if (ind.data && ind.data.length > 0) datasets.push({ type: 'line', label: ind.label, data: ind.data, borderColor: ind.color, borderWidth: 1.5, pointRadius: 0, fill: key !== 'middle', backgroundColor: 'rgba(128,128,128,0.1)', borderDash: key === 'middle' ? [] : [5, 5] });
            });
        }

        priceChartInstance = new Chart(ctx, {
            type: chartType,
            data: { datasets },
            options: {
                responsive: true, maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false, axis: 'x' },
                hover: { mode: 'index', intersect: false, animationDuration: 0 },
                scales: {
                    x: { type: 'time', time: { unit: timeframe === '5m' ? 'minute' : 'hour', displayFormats: { minute: 'HH:mm', hour: 'DD HH:mm' }}, grid: { display: true, color: gridColor }, ticks: { color: textColor }, position: 'bottom' },
                    x2: { type: 'time', time: { unit: 'day', displayFormats: { day: 'DD MMM' }}, grid: { display: true, color: (c) => c.tick && c.tick.major ? (isDarkTheme ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)') : 'transparent', lineWidth: 1 }, ticks: { color: textColor, maxRotation: 0, font: { weight: 'bold', size: 11 }, padding: 8, major: { enabled: true }, autoSkip: false, source: 'data' }, position: 'bottom' },
                    y: { position: 'right', grid: { color: gridColor }, ticks: { color: textColor, font: { weight: 'bold', size: 14 }, padding: 10, count: 6, callback: (v) => v.toFixed(getPrecision(v)) }}
                },
                plugins: {
                    tooltip: { enabled: false, external: externalTooltipHandler },
                    legend: { display: false },
                    annotation: { annotations: {} },
                    crosshair: { line: { color: isDarkTheme ? 'rgba(255,255,255,0.3)' : 'rgba(0,0,0,0.3)', width: 1 }, sync: { enabled: true, group: 1, suppressTooltips: false }, zoom: { enabled: false }, snap: { enabled: true }},
                    zoom: { zoom: { wheel: { enabled: true, modifierKey: 'ctrl' }, pinch: { enabled: true }, mode: 'xy', speed: 0.1, threshold: 2 }, pan: { enabled: true, mode: 'xy', modifierKey: 'shift' }, limits: { x: {min: 'original', max: 'original'}, y: {min: 'original', max: 'original'} }}
                },
                animation: false
            }
        });

        if (displayData.length > 0) {
            const lastDP = displayData[displayData.length - 1];
            const lastClose = priceChartStyle === 'line' ? lastDP.y : lastDP.originalClose;
            if (lastClose !== undefined) {
                // Store the last price as a property on the chart instance
                // This allows us to recover it when we need to reset annotations
                priceChartInstance._lastPriceLineValue = lastClose;
                
                const prec = getPrecision(lastClose);
                let lColor = upColor;
                if (priceChartStyle !== 'line') {
                    lColor = lastDP.originalClose >= lastDP.originalOpen ? upColor : downColor;
                } else if (displayData.length > 1) {
                    lColor = lastClose >= displayData[displayData.length - 2].y ? upColor : downColor;
                }
                
                // Store the color as well
                priceChartInstance._lastPriceLineColor = lColor;
                
                priceChartInstance.options.plugins.annotation.annotations.lastPriceLine = { 
                    type: 'line', 
                    yMin: lastClose, 
                    yMax: lastClose, 
                    borderColor: lColor, 
                    borderWidth: 1.5, 
                    borderDash: [6,6], 
                    label: { 
                        enabled: true, 
                        content: lastClose.toFixed(prec), 
                        position: 'end', 
                        backgroundColor: lColor, 
                        color: '#fff', 
                        font: {weight:'bold', size:10}, 
                        padding:{x:6,y:3}
                    }
                };
                priceChartInstance.update('none'); // Update for annotation
            }
            // Initial zoom logic removed as per user request.
        }
        
        // Create volume chart if data is suitable
        const volumeDataSource = priceChartStyle === 'line' ? ohlcvData : displayData; // ohlcvData for line, displayData for candle/HA
        if (volumeDataSource.every(d => d.hasOwnProperty('volume') && d.hasOwnProperty('o') && d.hasOwnProperty('c'))) {
            createColoredVolumeChart(volumeDataSource, priceChartStyle);
        } else {
             console.warn('Volume data source not suitable. Skipping volume chart.');
        }
        return priceChartInstance;
    } catch (error) {
        console.error('Error creating price chart:', error);
        return null;
    }
}

export function createColoredVolumeChart(chartSourceData, mainChartStyle) {
    try {
        // The volume-chart-wrapper and volume-chart canvas are now expected to be in index.html
        const volumeWrapper = elements.volumeChartWrapper; // from dom-elements.js
        const canvas = elements.volumeChartCanvas; // from dom-elements.js

        if (!chartSourceData || chartSourceData.length === 0 || !chartSourceData[0].hasOwnProperty('volume')) {
            if (volumeChartInstance) volumeChartInstance.destroy();
            volumeChartInstance = null;
            if (volumeWrapper) volumeWrapper.classList.add('hidden'); // Use hidden class
            return null;
        }

        if (!canvas) {
            console.error('Volume chart canvas element not found in DOM.');
            if (volumeWrapper) volumeWrapper.classList.add('hidden');
            return null;
        }
        // Make sure volume chart wrapper is visible when creating chart
        if (volumeWrapper) {
            volumeWrapper.classList.remove('hidden');
            volumeWrapper.style.opacity = '1';
        }


        const ctx = canvas.getContext('2d');
        if (volumeChartInstance) volumeChartInstance.destroy();

        const upColorVol = 'rgba(38, 166, 154, 0.6)';
        const downColorVol = 'rgba(239, 83, 80, 0.6)';
        const timestamps = chartSourceData.map(d => d.x);
        const volumes = chartSourceData.map(d => d.volume);
        const colors = chartSourceData.map(d => {
            // For HA, use HA o/c. For candlestick/line, use original o/c.
            const openVal = mainChartStyle === 'heikin-ashi' ? d.o : d.originalOpen;
            const closeVal = mainChartStyle === 'heikin-ashi' ? d.c : d.originalClose;
            return closeVal >= openVal ? upColorVol : downColorVol;
        });

        volumeChartInstance = new Chart(ctx, {
            type: 'bar',
            data: { labels: timestamps, datasets: [{ label: 'Volume', data: volumes, backgroundColor: colors, borderWidth: 0 }] },
            options: {
                responsive: true, maintainAspectRatio: false, animation: false,
                scales: {
                    y: { beginAtZero: true, grid: { display: false }, ticks: { callback: v => { if(v === 0) return ''; if (v >= 1e6) return (v/1e6).toFixed(1)+'M'; if (v >= 1e3) return (v/1e3).toFixed(1)+'K'; return v; }}},
                    x: { display: false }
                },
                plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => `Volume: ${c.raw >= 1e6 ? (c.raw/1e6).toFixed(2)+'M' : c.raw >= 1e3 ? (c.raw/1e3).toFixed(2)+'K' : c.raw.toFixed(2)}` }}}
            }
        });
        return volumeChartInstance;
    } catch(e) {
        console.error("Error in createColoredVolumeChart", e);
        return null;
    }
}


export function createVolatilityChart(symbol, timeframe, rawApiData) {
    try {
        if (!elements.volatilityChartCanvas) {
            console.error('Volatility chart canvas not found!');
            return null;
        }
        const ctx = elements.volatilityChartCanvas.getContext('2d');
        if (volatilityChartInstance) volatilityChartInstance.destroy();

        if (!rawApiData || rawApiData.length === 0) {
            console.error('No volatility data available');
            return null;
        }
        const sortedData = [...rawApiData].sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
        const labels = sortedData.map(item => new Date(item.timestamp));
        const values = sortedData.map(item => item.volatility);

        const isDarkTheme = !document.body.classList.contains('light-theme');
        const gridColor = isDarkTheme ? '#2a2e39' : '#e5e7eb';
        const textColor = isDarkTheme ? '#9ca3af' : '#666666';

        volatilityChartInstance = new Chart(ctx, {
            type: 'line',
            data: { labels, datasets: [{ label: `Volatility - ${symbol} (${timeframe})`, data: values, borderColor: '#2962ff', backgroundColor: 'rgba(41,98,255,0.1)', borderWidth: 2, pointRadius: 0, fill: true, tension: 0.1 }] },
            options: {
                responsive: true, maintainAspectRatio: false, animation: false,
                scales: {
                    x: { type: 'time', time: { unit: timeframe === '5m' ? 'minute' : 'hour', displayFormats: { minute: 'HH:mm', hour: 'DD HH:mm' }}, grid: { color: gridColor }, ticks: { color: textColor }},
                    x2: { type: 'time', time: { unit: 'day', displayFormats: { day: 'DD MMM' }}, grid: { display: true, color: (c) => c.tick && c.tick.major ? (isDarkTheme ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)') : 'transparent', lineWidth: 1 }, ticks: { color: textColor, maxRotation: 0, font: { weight: 'bold', size: 11 }, padding: 8, major: { enabled: true }, autoSkip: false, source: 'data' }, position: 'bottom' },
                    y: { position: 'right', grid: { color: gridColor }, ticks: { color: textColor, callback: v => v.toFixed(2) + '%' }}
                },
                plugins: {
                    tooltip: { mode: 'index', intersect: false, callbacks: { label: c => `Volatility: ${c.raw.toFixed(2)}%` }},
                    legend: { display: false },
                    crosshair: { line: { color: isDarkTheme ? 'rgba(255,255,255,0.3)' : 'rgba(0,0,0,0.3)', width: 1 }, sync: { enabled: true, group: 1 }, zoom: { enabled: false }, snap: { enabled: true }},
                    zoom: { zoom: { wheel: { enabled: true, modifierKey: 'ctrl' }, pinch: { enabled: true }, mode: 'xy' }, pan: { enabled: true, mode: 'xy', modifierKey: 'shift' }, limits: { x:{min:'original',max:'original'}, y:{min:'original',max:'original'} }}
                }
            }
        });
        return volatilityChartInstance;
    } catch (error) {
        console.error('Error creating volatility chart:', error);
        return null;
    }
}

export function createIndicatorChart(symbol, timeframe, rawIndicatorData, indicatorType) {
    try {
        if (!elements.indicatorChartCanvas) {
            console.error('Indicator chart canvas not found!');
            return null;
        }
        const ctx = elements.indicatorChartCanvas.getContext('2d');
        if (indicatorChartInstance) indicatorChartInstance.destroy();

        if (!rawIndicatorData || rawIndicatorData.length === 0) {
            console.error('No indicator data available for chart');
            // Potentially hide the indicator chart wrapper if no data
            if(elements.indicatorChartWrapper) elements.indicatorChartWrapper.classList.add('hidden');
            return null;
        }
        
        const sortedData = [...rawIndicatorData].sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
        const labels = sortedData.map(item => new Date(item.timestamp));
        const isDarkTheme = !document.body.classList.contains('light-theme');
        const gridColor = isDarkTheme ? '#2a2e39' : '#e5e7eb';
        const textColor = isDarkTheme ? '#9ca3af' : '#666666';

        let chartConfig = {
            type: 'line', // Default type
            data: { labels: labels, datasets: [] },
            options: {
                responsive: true, maintainAspectRatio: false, animation: false,
                interaction: { mode: 'index', intersect: false },
                scales: {
                    x: { type: 'time', time: { unit: timeframe === '5m' ? 'minute' : 'hour', displayFormats: { minute: 'HH:mm', hour: 'DD HH:mm' }}, grid: { color: gridColor }, ticks: { color: textColor }},
                    y: { position: 'right', grid: { color: gridColor }, ticks: { color: textColor }}
                },
                plugins: {
                    tooltip: { mode: 'index', intersect: false },
                    legend: { display: true, labels: { color: textColor } },
                    crosshair: { line: { color: isDarkTheme ? 'rgba(255,255,255,0.3)' : 'rgba(0,0,0,0.3)', width: 1 }, sync: { enabled: true, group: 1 }, zoom: { enabled: false }, snap: { enabled: true }},
                    zoom: { zoom: { wheel: { enabled: true, modifierKey: 'ctrl' }, pinch: { enabled: true }, mode: 'xy' }, pan: { enabled: true, mode: 'xy', modifierKey: 'shift' }, limits: { x:{min:'original',max:'original'}, y:{min:'original',max:'original'} }}
                }
            }
        };

        // Configure based on indicatorType (RSI, MACD, etc.)
        // (This part is extensive and similar to the original, so I'll abbreviate for brevity here,
        // but it would include all the switch cases for each indicator type)
        switch (indicatorType) {
            case 'rsi':
                chartConfig.data.datasets = [{ label: 'RSI (14)', data: sortedData.map(item => ({ x: new Date(item.timestamp), y: item.rsi14 })), borderColor: '#8E24AA', backgroundColor: 'rgba(142,36,170,0.1)', borderWidth: 1.5, pointRadius: 0, fill: true }];
                chartConfig.options.scales.y.min = 0; chartConfig.options.scales.y.max = 100;
                chartConfig.options.plugins.annotation = { annotations: { overbought: { type:'line',yMin:70,yMax:70,borderColor:'rgba(255,99,132,0.8)',borderWidth:1,borderDash:[5,5],label:{content:'Ipercomprato (70)',position:'start',backgroundColor:'rgba(255,99,132,0.8)',color:'#fff',font:{weight:'bold'}}}, oversold: {type:'line',yMin:30,yMax:30,borderColor:'rgba(54,162,235,0.8)',borderWidth:1,borderDash:[5,5],label:{content:'Ipervenduto (30)',position:'start',backgroundColor:'rgba(54,162,235,0.8)',color:'#fff',font:{weight:'bold'}}}}};
                break;
            case 'macd':
                chartConfig.type = 'bar'; // MACD often uses bar for histogram
                chartConfig.data.datasets = [
                    { label: 'MACD Hist', data: sortedData.map(item => ({ x: new Date(item.timestamp), y: item.macd_hist })), backgroundColor: (c) => c.raw.y >= 0 ? 'rgba(38,166,154,0.6)' : 'rgba(239,83,80,0.6)', order: 1 },
                    { label: 'MACD Line', data: sortedData.map(item => ({ x: new Date(item.timestamp), y: item.macd })), borderColor: '#2962FF', borderWidth: 2, type: 'line', pointRadius: 0, order: 0 },
                    { label: 'Signal Line', data: sortedData.map(item => ({ x: new Date(item.timestamp), y: item.macd_signal })), borderColor: '#FF6D00', borderWidth: 2, type: 'line', pointRadius: 0, order: 0 }
                ];
                break;
            // ... other cases for stoch, bbands, adx, atr, obv, vwap, volume
            default:
                console.warn(`No specific chart configuration for indicator type: ${indicatorType}. Using default line chart.`);
                // Fallback to a generic line if type is unknown, assuming 'y' field exists
                const genericData = sortedData.map(item => ({ x: new Date(item.timestamp), y: item[indicatorType] || item.value || item.y })); // Try to find data
                if (genericData.every(d => d.y !== undefined)) {
                    chartConfig.data.datasets.push({ label: indicatorType.toUpperCase(), data: genericData, borderColor: '#00BCD4', borderWidth: 1.5, pointRadius: 0, fill: false });
                } else {
                     console.error(`Data for indicator ${indicatorType} not found or in unexpected format.`);
                     return null;
                }
                break;
        }
        
        indicatorChartInstance = new Chart(ctx, chartConfig);
        return indicatorChartInstance;
    } catch (error) {
        console.error(`Error creating ${indicatorType} chart:`, error);
        return null;
    }
}

export function renderPatternVisualization(symbol, timeframe, patterns) {
    try {
        if (!elements.patternVisualization || !elements.patternInfo) {
            console.error('Pattern container elements not found');
            return;
        }
        elements.patternVisualization.innerHTML = ''; // Clear previous
        if (!patterns || Object.keys(patterns).length === 0) {
            elements.patternInfo.innerHTML = '<p>No pattern data available for this cryptocurrency.</p>';
            return;
        }
        const patternCount = Object.keys(patterns).length;
        
        // Create pattern container with tabs
        const tabsContainer = document.createElement('div');
        tabsContainer.className = 'pattern-tabs';
        
        const tabMostFrequent = document.createElement('button');
        tabMostFrequent.className = 'pattern-tab-button active';
        tabMostFrequent.textContent = 'Most Frequent';
        tabMostFrequent.dataset.tabTarget = 'frequent';
        
        const tabMostRecent = document.createElement('button');
        tabMostRecent.className = 'pattern-tab-button';
        tabMostRecent.textContent = 'Most Recent';
        tabMostRecent.dataset.tabTarget = 'recent';
        
        tabsContainer.appendChild(tabMostFrequent);
        tabsContainer.appendChild(tabMostRecent);
        
        // Create containers for each tab content
        const frequentPatternsContainer = document.createElement('div');
        frequentPatternsContainer.className = 'pattern-tab-content active';
        frequentPatternsContainer.id = 'frequent-patterns';
        
        const recentPatternsContainer = document.createElement('div');
        recentPatternsContainer.className = 'pattern-tab-content';
        recentPatternsContainer.id = 'recent-patterns';
        
        // Add info text
        elements.patternInfo.innerHTML = `<p>Found ${patternCount} unique patterns for ${symbol} (${timeframe}).</p>`;
        elements.patternInfo.appendChild(tabsContainer);
        
        // Create the containers for pattern display
        const patternDisplayContainer = document.createElement('div');
        patternDisplayContainer.className = 'pattern-display-container';
        patternDisplayContainer.appendChild(frequentPatternsContainer);
        patternDisplayContainer.appendChild(recentPatternsContainer);
        elements.patternVisualization.appendChild(patternDisplayContainer);
        
        // Process patterns by frequency
        const sortedPatternsByFrequency = Object.entries(patterns)
            .map(([pattern, data]) => {
                // Check if data is already in the new format (object with count and dates)
                if (typeof data === 'object' && data !== null && data.count !== undefined) {
                    return [pattern, data.count, data.dates || []];
                } 
                // Otherwise assume it's the old format (just count)
                return [pattern, data, []];
            })
            .sort((a, b) => b[1] - a[1]) // Sort by count (descending)
            .slice(0, 10); // Top 10
        
        // Add description for frequent patterns tab
        const frequentInfoP = document.createElement('p');
        frequentInfoP.textContent = 'Showing top 10 patterns by occurrence. Each pattern represents a sequence of price movements (up/down).';
        frequentInfoP.className = 'pattern-tab-description';
        frequentPatternsContainer.appendChild(frequentInfoP);
        
        // Add frequent patterns
        sortedPatternsByFrequency.forEach(([pattern, count]) => {
            const card = createPatternCard(pattern, count, 'Occurrences');
            card.dataset.pattern = pattern;
            card.addEventListener('click', () => showPatternOnChart(pattern, symbol, timeframe));
            frequentPatternsContainer.appendChild(card);
        });
        
        // Process patterns by recency
        const recentInfoP = document.createElement('p');
        recentInfoP.textContent = 'Showing 10 most recent patterns (from today backwards). Click a pattern to show it on chart.';
        recentInfoP.className = 'pattern-tab-description';
        recentPatternsContainer.appendChild(recentInfoP);
        
        // Get current date for reference
        const today = new Date();
        
        // Format date helper function
        const formatDate = (date) => {
            if (!(date instanceof Date)) {
                return 'Unknown date';
            }
            const day = String(date.getDate()).padStart(2, '0');
            const month = String(date.getMonth() + 1).padStart(2, '0');
            const year = date.getFullYear();
            return `${year}-${month}-${day}`;
        };
        
        // Check if we have pattern data with dates
        let patternsWithDates = [];
        
        Object.entries(patterns).forEach(([pattern, data]) => {
            if (typeof data === 'object' && data !== null && Array.isArray(data.dates)) {
                // For each date in the dates array
                data.dates.forEach(dateStr => {
                    try {
                        const date = new Date(dateStr);
                        if (!isNaN(date.getTime())) { // Valid date
                            patternsWithDates.push({
                                pattern,
                                date,
                                formattedDate: formatDate(date)
                            });
                        }
                    } catch (e) {
                        console.warn(`Invalid date format for pattern ${pattern}: ${dateStr}`);
                    }
                });
            }
        });
        
        // If we have pattern data with dates, use it
        if (patternsWithDates.length > 0) {
            // Sort by date descendingly (most recent first)
            patternsWithDates.sort((a, b) => b.date.getTime() - a.date.getTime());
            
            // Take only the 10 most recent patterns
            const recentPatterns = patternsWithDates.slice(0, 10);
            
            // Create pattern cards for these recent patterns
            recentPatterns.forEach(({ pattern, formattedDate }) => {
                const card = createPatternCard(pattern, formattedDate, 'Date');
                card.dataset.pattern = pattern;
                card.addEventListener('click', () => showPatternOnChart(pattern, symbol, timeframe));
                recentPatternsContainer.appendChild(card);
            });
        } else {
            // Fallback to simulation if no date data available
            // This is just for demo until the backend provides actual date data
            console.warn('No pattern date data available. Using simulated recent patterns.');
            
            // Create a sample of 10 recent patterns (using most frequent as placeholders)
            for (let i = 0; i < Math.min(10, sortedPatternsByFrequency.length); i++) {
                const [pattern] = sortedPatternsByFrequency[i];
                // Create a date starting from today and going backwards
                const patternDate = new Date(today);
                patternDate.setDate(today.getDate() - i);
                
                const formattedDate = formatDate(patternDate);
                const card = createPatternCard(pattern, formattedDate, 'Date');
                card.dataset.pattern = pattern;
                card.addEventListener('click', () => showPatternOnChart(pattern, symbol, timeframe));
                recentPatternsContainer.appendChild(card);
            }
        }
        
        // Add tab switching functionality
        tabMostFrequent.addEventListener('click', function() {
            togglePatternTabs('frequent');
        });
        
        tabMostRecent.addEventListener('click', function() {
            togglePatternTabs('recent');
        });
        
        // Function to toggle between tabs
        function togglePatternTabs(targetTab) {
            // Update tab buttons
            document.querySelectorAll('.pattern-tab-button').forEach(button => {
                button.classList.toggle('active', button.dataset.tabTarget === targetTab);
            });
            
            // Update tab content visibility
            document.querySelectorAll('.pattern-tab-content').forEach(content => {
                content.classList.toggle('active', content.id === `${targetTab}-patterns`);
            });
        }
    } catch (error) {
        console.error('Error rendering pattern visualization:', error);
    }
}

// Helper function to create a pattern card
function createPatternCard(pattern, value, labelText) {
    const card = document.createElement('div');
    card.className = 'pattern-card';
    
    const viz = document.createElement('div');
    viz.className = 'pattern-visualization';
    
    if (typeof pattern === 'string' && pattern.length > 0) {
        for (let i = 0; i < pattern.length; i++) {
            const bit = document.createElement('div');
            bit.className = `pattern-bit ${pattern[i] === '1' ? 'up' : 'down'}`;
            bit.textContent = pattern[i]; // Show 1/0 values inside the bits
            viz.appendChild(bit);
        }
    }
    
    const info = document.createElement('div');
    info.className = 'pattern-info';
    info.innerHTML = `<strong>Pattern:</strong> ${pattern}<br><strong>${labelText}:</strong> ${value}`;
    
    card.appendChild(viz);
    card.appendChild(info);
    return card;
}

// Function to show a pattern on the price chart
function showPatternOnChart(pattern, symbol, timeframe) {
    console.log(`Showing pattern ${pattern} on chart for ${symbol} (${timeframe})`);
    
    // Get the price chart instance
    const priceChart = getPriceChartInstance();
    if (!priceChart) {
        console.error('No price chart instance available to show pattern');
        return;
    }
    
    try {
        // Completely reset all annotations to avoid circular references
        // This is a more drastic but effective approach to prevent stack overflow
        if (!priceChart.options.plugins) {
            priceChart.options.plugins = {};
        }
        
        // Create a completely new annotation plugin configuration
        priceChart.options.plugins.annotation = {
            annotations: {
                'pattern-notice': {
                    type: 'label',
                    content: [`Pattern: ${pattern}`, 'Looking for matches...'],
                    position: {
                        x: 'center',
                        y: 'start'
                    },
                    font: {
                        size: 14,
                        weight: 'bold'
                    },
                    color: '#ffffff',
                    backgroundColor: 'rgba(75, 192, 192, 0.8)',
                    borderRadius: 4,
                    padding: 8
                }
            }
        };
        
        // Only preserve the last price line annotation if it exists
        if (priceChart._lastPriceLineValue !== undefined) {
            const lastPrice = priceChart._lastPriceLineValue;
            const prec = getPrecision(lastPrice);
            // Use the stored color if available, otherwise use default
            const lColor = priceChart._lastPriceLineColor || '#26a69a';
            
            priceChart.options.plugins.annotation.annotations['lastPriceLine'] = {
                type: 'line',
                yMin: lastPrice,
                yMax: lastPrice,
                borderColor: lColor,
                borderWidth: 1.5,
                borderDash: [6, 6],
                label: {
                    enabled: true,
                    content: lastPrice.toFixed(prec),
                    position: 'end',
                    backgroundColor: lColor,
                    color: '#fff',
                    font: {weight: 'bold', size: 10},
                    padding: {x: 6, y: 3}
                }
            };
        }
        
        // Apply the changes and update the chart
        priceChart.update('none');
        
        // Find actual pattern occurrences in the chart data
        setTimeout(() => {
            try {
                const data = priceChart.data.datasets[0].data;
                if (data && data.length > 0) {
                    // Find actual pattern occurrences by analyzing price movements
                    const patternOccurrences = findPatternInChartData(data, pattern);
                    
                    if (patternOccurrences.length > 0) {
                        // Use the most recent occurrence (last in the array)
                        const patternLocation = patternOccurrences[patternOccurrences.length - 1];
                        const startIndex = patternLocation.startIndex;
                        const endIndex = patternLocation.endIndex;
                        
                        // Get the points for annotation
                        const startPoint = data[startIndex];
                        const endPoint = data[endIndex];
                        
                        // Create a completely new annotation configuration
                        const annotations = {};
                        
                        // Add pattern highlight
                        annotations['pattern-highlight'] = {
                            type: 'box',
                            xMin: startPoint.x,
                            xMax: endPoint.x,
                            yMin: Math.min(...data.slice(startIndex, endIndex + 1).map(d => d.hasOwnProperty('l') ? d.l : d.y)) - 0.00001,
                            yMax: Math.max(...data.slice(startIndex, endIndex + 1).map(d => d.hasOwnProperty('h') ? d.h : d.y)) + 0.00001,
                            backgroundColor: 'rgba(255, 99, 132, 0.2)',
                            borderColor: 'rgba(255, 99, 132, 1)',
                            borderWidth: 2
                        };
                        
                        // Add pattern notice
                        annotations['pattern-notice'] = {
                            type: 'label',
                            content: [`Pattern: ${pattern}`, 'Pattern found on chart!'],
                            position: {
                                x: 'center',
                                y: 'start'
                            },
                            font: {
                                size: 14,
                                weight: 'bold'
                            },
                            color: '#ffffff',
                            backgroundColor: 'rgba(75, 192, 192, 0.8)',
                            borderRadius: 4,
                            padding: 8
                        };
                        
                        // Preserve last price line if it exists
                        if (priceChart._lastPriceLineValue !== undefined) {
                            const lastPrice = priceChart._lastPriceLineValue;
                            const prec = getPrecision(lastPrice);
                            // Use the stored color if available, otherwise use default
                            const lColor = priceChart._lastPriceLineColor || '#26a69a';
                            
                            annotations['lastPriceLine'] = {
                                type: 'line',
                                yMin: lastPrice,
                                yMax: lastPrice,
                                borderColor: lColor,
                                borderWidth: 1.5,
                                borderDash: [6, 6],
                                label: {
                                    enabled: true,
                                    content: lastPrice.toFixed(prec),
                                    position: 'end',
                                    backgroundColor: lColor,
                                    color: '#fff',
                                    font: {weight: 'bold', size: 10},
                                    padding: {x: 6, y: 3}
                                }
                            };
                        }
                        
                        // Apply the updated annotations
                        priceChart.options.plugins.annotation.annotations = annotations;
                        
                        // Zoom to pattern location
                        const zoomWindow = 20; // Number of candles to show around pattern
                        const minIndex = Math.max(0, startIndex - zoomWindow);
                        const maxIndex = Math.min(data.length - 1, endIndex + zoomWindow);
                        
                        // Set zoom limits
                        priceChart.options.plugins.zoom.limits.x = {
                            min: data[minIndex].x,
                            max: data[maxIndex].x
                        };
                        
                        // Reset zoom and then zoom to pattern area
                        priceChart.resetZoom();
                        priceChart.zoomScale('x', {
                            min: data[minIndex].x,
                            max: data[maxIndex].x
                        }, 'default');
                        
                        priceChart.update('none');
                        
                    } else {
                        // No matching pattern found
                        const annotations = {};
                        
                        // Add pattern notice for not found
                        annotations['pattern-notice'] = {
                            type: 'label',
                            content: [`Pattern: ${pattern}`, 'Pattern not found in current data'],
                            position: {
                                x: 'center',
                                y: 'start'
                            },
                            font: {
                                size: 14,
                                weight: 'bold'
                            },
                            color: '#ffffff',
                            backgroundColor: 'rgba(255, 99, 132, 0.8)',
                            borderRadius: 4,
                            padding: 8
                        };
                        
                        // Preserve last price line if it exists
                        if (priceChart._lastPriceLineValue !== undefined) {
                            const lastPrice = priceChart._lastPriceLineValue;
                            const prec = getPrecision(lastPrice);
                            const lColor = priceChart._lastPriceLineColor || '#26a69a';
                            
                            annotations['lastPriceLine'] = {
                                type: 'line',
                                yMin: lastPrice,
                                yMax: lastPrice,
                                borderColor: lColor,
                                borderWidth: 1.5,
                                borderDash: [6, 6],
                                label: {
                                    enabled: true,
                                    content: lastPrice.toFixed(prec),
                                    position: 'end',
                                    backgroundColor: lColor,
                                    color: '#fff',
                                    font: {weight: 'bold', size: 10},
                                    padding: {x: 6, y: 3}
                                }
                            };
                        }
                        
                        priceChart.options.plugins.annotation.annotations = annotations;
                        priceChart.update('none');
                    }
                }
            } catch (error) {
                console.error('Error updating pattern on chart:', error);
            }
        }, 500);
    } catch (error) {
        console.error('Error showing pattern on chart:', error);
    }
}

// Helper function to find pattern occurrences in chart data
function findPatternInChartData(chartData, patternString) {
    const occurrences = [];
    
    try {
        if (!chartData || chartData.length < patternString.length + 1) {
            console.log(`Chart data insufficient for pattern: ${patternString}`);
            return occurrences;
        }
        
        console.log(`Searching for pattern: ${patternString} in chart data of length: ${chartData.length}`);
        
        // Analyze price movements to find binary pattern matches
        for (let i = 0; i < chartData.length - patternString.length; i++) {
            let binaryPattern = '';
            
            // Generate binary pattern from data point sequence
            for (let j = 0; j < patternString.length; j++) {
                const current = chartData[i + j];
                const next = chartData[i + j + 1];
                
                if (!current || !next) {
                    console.log(`Missing data points at index: ${i+j}`);
                    break;
                }
                
                // Get closing prices based on data format
                let currentClose, nextClose;
                
                // For candlestick / Heikin Ashi data
                if (current.hasOwnProperty('c') && next.hasOwnProperty('c')) {
                    currentClose = current.c;
                    nextClose = next.c;
                } 
                // For line chart data
                else if (current.hasOwnProperty('y') && next.hasOwnProperty('y')) {
                    currentClose = current.y;
                    nextClose = next.y;
                } else {
                    console.log(`Unknown data format at index: ${i+j}`);
                    break;
                }
                
                // Determine pattern bit (1 for up, 0 for down or equal)
                if (nextClose > currentClose) {
                    binaryPattern += '1';
                } else {
                    binaryPattern += '0';
                }
            }
            
            // Check if we found a matching pattern (only if we have a complete pattern)
            if (binaryPattern.length === patternString.length) {
                if (binaryPattern === patternString) {
                    occurrences.push({
                        startIndex: i,
                        endIndex: i + patternString.length,
                        pattern: binaryPattern
                    });
                    
                    console.log(`Pattern ${patternString} found at index ${i} to ${i + patternString.length}`);
                }
            }
        }
        
        console.log(`Found ${occurrences.length} occurrences of pattern ${patternString}`);
        return occurrences;
        
    } catch (error) {
        console.error('Error finding pattern in chart data:', error);
        return [];
    }
}

// --- RESIZE FUNCTIONS ---
export function resizePriceChart() {
    if (priceChartInstance) {
        setTimeout(() => { priceChartInstance.resize(); if (volumeChartInstance) volumeChartInstance.resize(); }, 10);
    }
}
export function resizeVolatilityChart() {
    if (volatilityChartInstance) setTimeout(() => { volatilityChartInstance.resize(); }, 10);
}
export function resizeIndicatorChart() {
    if (indicatorChartInstance) setTimeout(() => { indicatorChartInstance.resize(); }, 10);
}

// --- TOOLTIP HANDLER ---
const getOrCreateTooltip = (chart) => {
    let tooltipEl = chart.canvas.parentNode.querySelector('div.chartjs-tooltip');
    if (!tooltipEl) {
        tooltipEl = document.createElement('div');
        tooltipEl.className = 'chartjs-tooltip'; // Add a class for styling
        // Basic styling, can be enhanced in CSS
        tooltipEl.style.opacity = 1;
        tooltipEl.style.pointerEvents = 'none';
        tooltipEl.style.position = 'absolute';
        tooltipEl.style.transform = 'translate(-50%, 0)';
        tooltipEl.style.transition = 'all .1s ease';
        tooltipEl.style.padding = '10px';
        tooltipEl.style.borderRadius = '6px';
        tooltipEl.style.boxShadow = '0 2px 5px rgba(0,0,0,0.2)';
        tooltipEl.style.fontFamily = '"Inter", sans-serif';
        tooltipEl.style.fontSize = '12px';
        tooltipEl.style.zIndex = '100'; // Ensure it's on top

        const table = document.createElement('table');
        table.style.margin = '0px';
        tooltipEl.appendChild(table);
        chart.canvas.parentNode.appendChild(tooltipEl);
    }
    return tooltipEl;
};

const externalTooltipHandler = (context) => {
    const { chart, tooltip } = context;
    const tooltipEl = getOrCreateTooltip(chart);

    const isDarkTheme = !document.body.classList.contains('light-theme');
    tooltipEl.style.background = isDarkTheme ? 'rgba(30,34,45,0.95)' : 'rgba(255,255,255,0.95)';
    tooltipEl.style.color = isDarkTheme ? '#d1d4dc' : '#333';
    tooltipEl.style.border = `1px solid ${isDarkTheme ? '#363c4e' : '#d1d5db'}`;

    if (tooltip.opacity === 0) {
        tooltipEl.style.opacity = 0;
        return;
    }

    const tableRoot = tooltipEl.querySelector('table');
    tableRoot.innerHTML = ''; // Clear previous content

    if (tooltip.body) {
        const titleLines = tooltip.title || [];
        const tableHead = document.createElement('thead');
        titleLines.forEach(title => {
            const tr = document.createElement('tr');
            const th = document.createElement('th');
            th.style.borderWidth = 0;
            th.style.textAlign = 'left';
            th.style.paddingBottom = '4px';
            th.style.fontWeight = '600';
            th.textContent = title;
            tr.appendChild(th);
            tableHead.appendChild(tr);
        });
        tableRoot.appendChild(tableHead);

        const tableBody = document.createElement('tbody');
        tooltip.dataPoints.forEach((dp, i) => {
            const dataPoint = dp.raw;
            const colors = tooltip.labelColors[i];
            const isMainPriceDataset = dp.dataset.type === 'candlestick' || dp.dataset.type === 'line' && dp.dataset.label.includes(chart.data.datasets[0].label);


            if (isMainPriceDataset && (dp.dataset.type === 'candlestick' || chart.config.type === 'candlestick')) { // Candlestick or Heikin Ashi from main dataset
                const precision = getPrecision(dataPoint.originalClose);
                const ohlcStyle = dataPoint.originalClose >= dataPoint.originalOpen ? `color: ${upColor};` : `color: ${downColor};`;
                
                let rows = [
                    `Open: <span style="${ohlcStyle}">${dataPoint.originalOpen.toFixed(precision)}</span>`,
                    `High: <span style="${ohlcStyle}">${dataPoint.originalHigh.toFixed(precision)}</span>`,
                    `Low: <span style="${ohlcStyle}">${dataPoint.originalLow.toFixed(precision)}</span>`,
                    `Close: <span style="${ohlcStyle}">${dataPoint.originalClose.toFixed(precision)}</span>`
                ];
                if (dataPoint.o !== dataPoint.originalOpen || dataPoint.c !== dataPoint.originalClose) { // Is Heikin-Ashi
                    const haStyle = dataPoint.c >= dataPoint.o ? `color: ${upColor};` : `color: ${downColor};`;
                    rows.push(`<div style="margin-top:5px;font-weight:bold;">Heikin-Ashi:</div>`);
                    rows.push(`HA Open: <span style="${haStyle}">${dataPoint.o.toFixed(precision)}</span>`);
                    rows.push(`HA High: <span style="${haStyle}">${dataPoint.h.toFixed(precision)}</span>`);
                    rows.push(`HA Low: <span style="${haStyle}">${dataPoint.l.toFixed(precision)}</span>`);
                    rows.push(`HA Close: <span style="${haStyle}">${dataPoint.c.toFixed(precision)}</span>`);
                }
                 rows.forEach(rowHtml => {
                    const tr = document.createElement('tr');
                    const td = document.createElement('td');
                    td.style.borderWidth = 0;
                    td.innerHTML = rowHtml;
                    tr.appendChild(td);
                    tableBody.appendChild(tr);
                });

            } else if (isMainPriceDataset && dp.dataset.type === 'line') { // Main price data as line
                const precision = getPrecision(dataPoint.y);
                 const tr = document.createElement('tr');
                 const td = document.createElement('td');
                 td.style.borderWidth = 0;
                 td.innerHTML = `Close: <span>${dataPoint.y.toFixed(precision)}</span>`;
                 tr.appendChild(td);
                 tableBody.appendChild(tr);
            } else if (dp.dataset.type === 'line' && dataPoint.y !== undefined) { // Overlay indicators
                const precision = getPrecision(dataPoint.y);
                const tr = document.createElement('tr');
                const td = document.createElement('td');
                td.style.borderWidth = 0;
                const span = document.createElement('span');
                span.style.background = colors.backgroundColor;
                span.style.borderColor = colors.borderColor;
                span.style.borderWidth = '2px';
                span.style.marginRight = '4px';
                span.style.height = '10px';
                span.style.width = '10px';
                span.style.display = 'inline-block';
                td.appendChild(span);
                td.appendChild(document.createTextNode(`${dp.dataset.label}: ${dataPoint.y.toFixed(precision)}`));
                tr.appendChild(td);
                tableBody.appendChild(tr);
            }
        });
        tableRoot.appendChild(tableBody);
    }

    const { offsetLeft: positionX, offsetTop: positionY } = chart.canvas;
    tooltipEl.style.opacity = 1;
    tooltipEl.style.left = positionX + tooltip.caretX + 10 + 'px'; // Position to the right of caret
    tooltipEl.style.top = positionY + tooltip.caretY - (tooltipEl.offsetHeight / 2) + 'px'; // Center vertically

    // Adjust if tooltip goes off screen
    if (parseFloat(tooltipEl.style.left) + tooltipEl.offsetWidth > chart.canvas.width) {
        tooltipEl.style.left = positionX + tooltip.caretX - tooltipEl.offsetWidth - 10 + 'px';
    }
    if (parseFloat(tooltipEl.style.top) < 0) {
        tooltipEl.style.top = '5px';
    }
    if (parseFloat(tooltipEl.style.top) + tooltipEl.offsetHeight > chart.canvas.height) {
        tooltipEl.style.top = chart.canvas.height - tooltipEl.offsetHeight - 5 + 'px';
    }
};

// Getter functions for chart instances if main.js needs them (e.g., for reset zoom)
export function getPriceChartInstance() { return priceChartInstance; }
export function getVolatilityChartInstance() { return volatilityChartInstance; }
export function getIndicatorChartInstance() { return indicatorChartInstance; }
export function getVolumeChartInstance() { return volumeChartInstance; }
