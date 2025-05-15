/**
 * Chart Handler Module for TradingJii Dashboard
 * 
 * This module provides functions for rendering price and volatility charts using Chart.js.
 */

// The annotation plugin is automatically registered when loaded via script tag
// No need to manually register it, but we'll check if it's available
if (typeof Chart === 'undefined') {
    console.error('Chart.js not found. Make sure it is loaded before chart-handler.js.');
} else if (!Chart.Annotation) {
    console.warn('Chart.js Annotation plugin might not be properly loaded. Some features may not work correctly.');
}

// Store chart instances to update or destroy them later
let priceChart = null;
let volumeChart = null;
let volatilityChart = null;
let indicatorChart = null;

/**
 * Create a price candlestick chart using Chart.js with TradingView style
 * @param {string} symbol - Cryptocurrency symbol
 * @param {string} timeframe - Timeframe like '5m', '15m'
 * @param {Array} data - OHLCV data from the API
 * @param {string} priceChartStyle - Style of the price chart ('heikin-ashi', 'candlestick', 'line')
 */
function createPriceChart(symbol, timeframe, data, priceChartStyle = 'heikin-ashi') {
    try {
        console.log(`Creating price chart for ${symbol} with style: ${priceChartStyle}`);
        
        // Get the canvas element
        const ctx = document.getElementById('price-chart').getContext('2d');
        
        // If chart already exists, destroy it
        if (priceChart) {
            priceChart.destroy();
        }
        
        if (volumeChart) {
            volumeChart.destroy();
        }
        
        // Prepare data for Chart.js
        const chartData = prepareOHLCVData(data);
        
        if (!chartData || chartData.length === 0) {
            console.error('No chart data available');
            return null;
        }
        
        // Extract indicators from the data if they exist
        const indicators = extractIndicators(data); // Assuming 'data' is the raw API data for indicators
        
        let displayData;
        let chartType = 'candlestick'; // Default for heikin-ashi and candlestick
        let datasetLabel = `${symbol} (${timeframe})`;

        switch (priceChartStyle) {
            case 'line':
                chartType = 'line';
                displayData = chartData.map(d => ({ x: d.x, y: d.c }));
                datasetLabel += ' - Line';
                break;
            case 'candlestick':
                displayData = chartData.map(d => ({
                    x: d.x, o: d.o, h: d.h, l: d.l, c: d.c,
                    originalOpen: d.o, originalHigh: d.h, originalLow: d.l, originalClose: d.c, volume: d.volume
                }));
                datasetLabel += ' - Candlestick';
                break;
            case 'heikin-ashi':
            default: // Default to Heikin-Ashi
                const haData = calculateHeikinAshi(chartData);
                displayData = haData.map(d => ({
                    x: d.x, o: d.o, h: d.h, l: d.l, c: d.c,
                    originalOpen: d.originalOpen, originalHigh: d.originalHigh,
                    originalLow: d.originalLow, originalClose: d.originalClose, volume: d.volume // Ensure volume is passed for HA too
                }));
                datasetLabel += ' - Heikin-Ashi';
                break;
        }
        
        if (!displayData || displayData.length === 0) {
            console.error('No display data available for price chart');
            return null;
        }
        console.log(`Chart data prepared for ${priceChartStyle}, ${displayData.length} points available`);
        
        // Define colors based on theme
        const isDarkTheme = !document.body.classList.contains('light-theme');
        const gridColor = isDarkTheme ? '#2a2e39' : '#e5e7eb';
        const textColor = isDarkTheme ? '#9ca3af' : '#666666';
        
        // Define colors for candlesticks with improved visibility
        const upColor = '#26a69a'; // Green for up candles
        const downColor = '#ef5350'; // Red for down candles
        const upColorFill = 'rgba(38, 166, 154, 0.85)'; // More opaque fill for up candles
        const downColorFill = 'rgba(239, 83, 80, 0.85)'; // More opaque fill for down candles
        
        console.log(`Creating ${priceChartStyle} chart for ${symbol}`);
        
        // Prepare the datasets array
        const datasets = [];
        
        if (priceChartStyle === 'line') {
            datasets.push({
                label: datasetLabel,
                data: displayData,
                borderColor: upColor, // Or a neutral color like accent-color
                borderWidth: 2,
                pointRadius: 0,
                pointHoverRadius: 5,
                fill: false,
                tension: 0.1
            });
        } else { // Candlestick or Heikin-Ashi
            datasets.push({
                label: datasetLabel,
                data: displayData, // This now contains original O,H,L,C for tooltip if not HA
                color: { // For chartjs-chart-financial
                    up: upColor,
                    down: downColor,
                    unchanged: '#888888',
                },
                borderColor: function(context) {
                    // For candlestick/HA, color border based on open vs close
                    const currentData = context.dataset.data[context.dataIndex];
                    return currentData.o > currentData.c ? downColor : upColor;
                },
                borderWidth: 2.5,
                wickWidth: 2,
                barPercentage: 0.92,
                barThickness: 14,
                backgroundColor: function(context) {
                    // For candlestick/HA, color fill based on open vs close
                    const currentData = context.dataset.data[context.dataIndex];
                    return currentData.o > currentData.c ? downColorFill : upColorFill;
                },
                pointHoverRadius: 5,
                pointHoverBorderWidth: 2
            });
        }
        
        // Add indicator datasets if they exist
        // Add SMA indicators
        if (indicators.sma) {
            Object.keys(indicators.sma).forEach(key => {
                const indicator = indicators.sma[key];
                if (indicator.data && indicator.data.length > 0) {
                    datasets.push({
                        type: 'line',
                        label: indicator.label,
                        data: indicator.data,
                        borderColor: indicator.color,
                        borderWidth: 1.5,
                        pointRadius: 0,
                        pointHoverRadius: 5,
                        fill: false
                    });
                }
            });
        }
        
        // Add EMA indicators
        if (indicators.ema) {
            Object.keys(indicators.ema).forEach(key => {
                const indicator = indicators.ema[key];
                if (indicator.data && indicator.data.length > 0) {
                    datasets.push({
                        type: 'line',
                        label: indicator.label,
                        data: indicator.data,
                        borderColor: indicator.color,
                        borderWidth: 1.5,
                        pointRadius: 0,
                        pointHoverRadius: 5,
                        fill: false
                    });
                }
            });
        }
        
        // Add Bollinger Bands
        if (indicators.bbands) {
            Object.keys(indicators.bbands).forEach(key => {
                const indicator = indicators.bbands[key];
                if (indicator.data && indicator.data.length > 0) {
                    datasets.push({
                        type: 'line',
                        label: indicator.label,
                        data: indicator.data,
                        borderColor: indicator.color,
                        borderWidth: 1.5,
                        pointRadius: 0,
                        pointHoverRadius: 0,
                        fill: false,
                        borderDash: key === 'middle' ? [] : [5, 5]
                    });
                }
            });
        }
        
        // Create the price chart
        priceChart = new Chart(ctx, {
            type: chartType, // This will be 'line' or 'candlestick'
            data: { datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false,
                    axis: 'x'
                },
                hover: {
                    mode: 'index',
                    intersect: false,
                    animationDuration: 0
                },
                events: ['mousemove', 'mouseout', 'click', 'touchstart', 'touchmove'],
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: timeframe === '5m' ? 'minute' : 'hour',
                            displayFormats: {
                                minute: 'HH:mm',
                                hour: 'DD HH:mm'
                            }
                        },
                        grid: {
                            display: true,
                            color: gridColor
                        },
                        ticks: {
                            color: textColor
                        },
                        position: 'bottom',
                        display: true
                    },
                    x2: {
                        type: 'time',
                        time: {
                            unit: 'day',
                            displayFormats: {
                                day: 'DD MMM' // Show date in format "01 Jan" at bottom
                            }
                        },
                        grid: {
                            display: true, // Show grid lines for daily separation
                            color: function(context) {
                                return context.tick && context.tick.major ? 
                                    isDarkTheme ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)' : 
                                    'transparent';
                            },
                            lineWidth: 1
                        },
                        ticks: {
                            color: textColor,
                            maxRotation: 0, // Don't rotate labels
                            font: {
                                weight: 'bold',
                                size: 11
                            },
                            padding: 8,
                            major: {
                                enabled: true
                            },
                            autoSkip: false,
                            source: 'data'
                        },
                        position: 'bottom',
                        display: true
                    },
                    y: {
                        position: 'right',
                        grid: {
                            color: gridColor
                        },
                        ticks: {
                            color: textColor,
                            font: {
                                weight: 'bold',
                                size: 14
                            },
                            padding: 10,
                            count: 6,
                            callback: function(value) {
                                const precision = getPrecision(value);
                                return value.toFixed(precision);
                            },
                            z: 1
                        }
                    }
                },
                plugins: {
                    tooltip: {
                        enabled: false, // Disable default tooltip
                        external: externalTooltipHandler // Use custom HTML tooltip
                    },
                    legend: {
                        display: false
                    },
                    annotation: { 
                        annotations: {} 
                    },
                    crosshair: {
                        line: {
                            color: isDarkTheme ? 'rgba(255, 255, 255, 0.3)' : 'rgba(0, 0, 0, 0.3)', // Crosshair line color
                            width: 1 // Crosshair line width
                        },
                        sync: {
                            enabled: true, // Enable synchronization for all charts
                            group: 1, // Group name for synchronization
                            suppressTooltips: false // Show tooltips when synchronized
                        },
                        zoom: {
                            enabled: false // Disable the plugin's own zoom, use chartjs-plugin-zoom
                        },
                        snap: {
                            enabled: true // Snap to the nearest data point
                        }
                    },
                    zoom: {
                        zoom: {
                            wheel: {
                                enabled: true, // Enable zoom with mouse wheel
                                modifierKey: 'ctrl' // Require ctrl key to be pressed
                            },
                            pinch: {
                                enabled: true // Enable zoom with pinch gesture on touch devices
                            },
                            mode: 'xy', // Allow zooming both axes
                            speed: 0.1, // Adjust zoom speed
                            threshold: 2, // Minimum zoom level
                        },
                        pan: {
                            enabled: true,
                            mode: 'xy',
                            modifierKey: 'shift' // Require shift key to be pressed for panning
                        },
                        limits: {
                            x: {min: 'original', max: 'original'},
                            y: {min: 'original', max: 'original'}
                        }
                    }
                },
                animation: false 
            }
        });

        // Add annotation for the last closing price if data is available
        if (displayData.length > 0) {
            const lastDataPoint = displayData[displayData.length - 1];
            // Use originalClose if available (from candlestick/HA), otherwise use 'y' (from line)
            const lastClosePrice = lastDataPoint.originalClose !== undefined ? lastDataPoint.originalClose : lastDataPoint.y;
            
            if (lastClosePrice !== undefined) {
                const precision = getPrecision(lastClosePrice);
                let lastCandleColor = upColor; // Default for line or if open is not available
                if (lastDataPoint.originalOpen !== undefined && lastDataPoint.originalClose !== undefined) {
                    lastCandleColor = lastDataPoint.originalClose >= lastDataPoint.originalOpen ? upColor : downColor;
                } else if (priceChartStyle === 'line' && displayData.length > 1) {
                    const prevDataPoint = displayData[displayData.length - 2];
                    lastCandleColor = lastClosePrice >= prevDataPoint.y ? upColor : downColor;
                }


                priceChart.options.plugins.annotation.annotations.lastPriceLine = {
                    type: 'line',
                    yMin: lastClosePrice,
                    yMax: lastClosePrice,
                    borderColor: lastCandleColor,
                    borderWidth: 1.5,
                    borderDash: [6, 6],
                    label: {
                        enabled: true,
                        content: lastClosePrice.toFixed(precision),
                        position: 'end',
                        backgroundColor: lastCandleColor,
                        color: '#ffffff',
                        font: {
                            weight: 'bold',
                            size: 10
                        },
                        padding: {
                            x: 6,
                            y: 3
                        },
                        yAdjust: 0,
                        xAdjust: 0
                    }
                };
                priceChart.update();
            }
        }
        
        console.log(`${priceChartStyle} chart created successfully`);
        
        try {
            // Volume chart should use data that has o, c, and volume properties.
            // If current style is 'line', chartData (original OHLCV) is suitable.
            // If 'candlestick' or 'heikin-ashi', displayData has these.
            const volumeChartDataSource = (priceChartStyle === 'line') ? chartData : displayData;
            if (volumeChartDataSource && volumeChartDataSource.every(d => d.hasOwnProperty('volume') && d.hasOwnProperty('o') && d.hasOwnProperty('c'))) {
                 createColoredVolumeChart(symbol, timeframe, volumeChartDataSource, priceChartStyle);
            } else {
                console.warn('Volume data source is not suitable for colored volume chart. Skipping volume chart.');
                 if (volumeChart) { // Destroy if exists and cannot be updated
                    volumeChart.destroy();
                    volumeChart = null;
                    const volumeWrapper = document.querySelector('.volume-chart-wrapper');
                    if (volumeWrapper) volumeWrapper.remove();
                }
            }
        } catch (volumeError) {
            console.error('Error creating volume chart:', volumeError);
        }
        
        return priceChart;
    } catch (error) {
        console.error('Error creating price chart:', error);
        return null;
    }
}

/**
 * Create a volume bar chart with colors matching candlesticks
 * @param {string} symbol - Cryptocurrency symbol
 * @param {string} timeframe - Timeframe
 * @param {Array} ohlcOrHaData - Data array (either raw OHLC or Heikin-Ashi), must contain x, o, c, volume
 * @param {string} priceChartStyle - The style of the main price chart
 */
function createColoredVolumeChart(symbol, timeframe, ohlcOrHaData, priceChartStyle) {
    const volumeCanvas = document.getElementById('volume-chart');
    let volumeWrapper = document.querySelector('.volume-chart-wrapper');

    if (!ohlcOrHaData || ohlcOrHaData.length === 0 || !ohlcOrHaData[0].hasOwnProperty('volume')) {
        console.warn('Volume data not available or not in correct format. Hiding volume chart.');
        if (volumeChart) {
            volumeChart.destroy();
            volumeChart = null;
        }
        if (volumeWrapper) {
            volumeWrapper.style.display = 'none';
        }
        return;
    }

    if (!volumeCanvas && !volumeWrapper) {
        const priceChartWrapper = document.getElementById('price-chart-wrapper');
        volumeWrapper = document.createElement('div');
        const volumeWrapper = document.createElement('div');
        volumeWrapper.className = 'volume-chart-wrapper';
        volumeWrapper.style.height = '20%'; // Adjust as needed
        volumeWrapper.style.marginTop = '10px';
        const newCanvas = document.createElement('canvas');
        newCanvas.id = 'volume-chart';
        volumeWrapper.appendChild(newCanvas);
        if (priceChartWrapper && priceChartWrapper.parentNode) {
            priceChartWrapper.parentNode.insertBefore(volumeWrapper, priceChartWrapper.nextSibling);
        }
    } else if (volumeWrapper) {
        volumeWrapper.style.display = 'block'; // Ensure it's visible if previously hidden
    }
    
    const currentVolumeCanvas = document.getElementById('volume-chart');
    if (!currentVolumeCanvas) {
        console.error("Volume canvas element not found after attempting to create it.");
        return;
    }
    const volumeCtx = currentVolumeCanvas.getContext('2d');

    if (volumeChart) {
        volumeChart.destroy();
    }

    const upColorVol = 'rgba(38, 166, 154, 0.6)';
    const downColorVol = 'rgba(239, 83, 80, 0.6)';

    const timestamps = ohlcOrHaData.map(d => d.x);
    const volumes = ohlcOrHaData.map(d => d.volume);
    
    // Determine colors based on close vs open. For Heikin-Ashi, d.c and d.o are HA values.
    // For regular candlestick, d.c and d.o are original close/open.
    // For line chart, ohlcOrHaData is original OHLC, so d.c and d.o are original.
    const colors = ohlcOrHaData.map(d => {
        // If priceChartStyle is 'heikin-ashi', d.o and d.c are Heikin-Ashi o/c.
        // Otherwise, they are original o/c.
        return d.c >= d.o ? upColorVol : downColorVol;
    });
    
    volumeChart = new Chart(volumeCtx, {
        type: 'bar',
        data: {
            labels: timestamps,
            datasets: [{
                label: 'Volume',
                data: volumes,
                backgroundColor: colors,
                borderColor: colors,
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { display: false },
                    ticks: {
                        callback: function(value) {
                            if (value === 0) return '';
                            if (value >= 1000000) return (value / 1000000).toFixed(1) + 'M';
                            if (value >= 1000) return (value / 1000).toFixed(1) + 'K';
                            return value;
                        }
                    }
                },
                x: { display: false }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const volume = context.raw;
                            if (volume >= 1000000) return `Volume: ${(volume / 1000000).toFixed(2)}M`;
                            if (volume >= 1000) return `Volume: ${(volume / 1000).toFixed(2)}K`;
                            return `Volume: ${volume.toFixed(2)}`;
                        }
                    }
                }
            },
            animation: false
        }
    });
    return volumeChart;
}

/**
 * Extract technical indicators from OHLCV data if they exist
 * @param {Array} data - OHLCV data from the API
 * @returns {Object} - Object containing extracted indicator data
 */
function extractIndicators(data) {
    if (!data || !Array.isArray(data) || data.length === 0) return {};

    // Initialize indicators object
    const indicators = {
        sma: {
            sma9: { data: [], color: 'rgba(125, 200, 255, 1)', label: 'SMA 9' },
            sma20: { data: [], color: 'rgba(255, 165, 0, 1)', label: 'SMA 20' },
            sma50: { data: [], color: 'rgba(128, 0, 128, 1)', label: 'SMA 50' }
        },
        ema: {
            ema20: { data: [], color: 'rgba(255, 99, 132, 1)', label: 'EMA 20' },
            ema50: { data: [], color: 'rgba(54, 162, 235, 1)', label: 'EMA 50' },
            ema200: { data: [], color: 'rgba(255, 206, 86, 1)', label: 'EMA 200' }
        },
        bbands: {
            upper: { data: [], color: 'rgba(128, 128, 128, 0.5)', label: 'BB Upper' },
            middle: { data: [], color: 'rgba(128, 128, 128, 1)', label: 'BB Middle' },
            lower: { data: [], color: 'rgba(128, 128, 128, 0.5)', label: 'BB Lower' }
        },
        rsi: { data: [], color: '#8E24AA', label: 'RSI (14)' },
        macd: {
            macd: { data: [], color: '#2962FF', label: 'MACD Line' },
            signal: { data: [], color: '#FF6D00', label: 'Signal Line' },
            hist: { data: [], colorPositive: 'rgba(38, 166, 154, 0.6)', colorNegative: 'rgba(239, 83, 80, 0.6)', label: 'MACD Histogram' }
        },
        stoch: {
            k: { data: [], color: '#2962FF', label: '%K' },
            d: { data: [], color: '#FF6D00', label: '%D' }
        },
        adx: { data: [], color: '#673AB7', label: 'ADX (14)' },
        atr: { data: [], color: '#9C27B0', label: 'ATR (14)' },
        obv: { data: [], color: '#3F51B5', label: 'OBV' },
        vwap: { data: [], color: '#009688', label: 'VWAP' },
        volume: {
            volume: { data: [], colorPositive: 'rgba(38, 166, 154, 0.6)', colorNegative: 'rgba(239, 83, 80, 0.6)', label: 'Volume' },
            sma20: { data: [], color: '#FF9800', label: 'Volume SMA (20)' }
        }
    };

    try {
        // Extract timestamps as Date objects for Chart.js
        const timestamps = data.map(item => new Date(item.timestamp));

        // Extract indicators if they exist in the data
        data.forEach((item, index) => {
            const timestamp = timestamps[index];

            // Extract Simple Moving Averages
            if (item.sma9 !== undefined) indicators.sma.sma9.data.push({ x: timestamp, y: parseFloat(item.sma9) });
            if (item.sma20 !== undefined) indicators.sma.sma20.data.push({ x: timestamp, y: parseFloat(item.sma20) });
            if (item.sma50 !== undefined) indicators.sma.sma50.data.push({ x: timestamp, y: parseFloat(item.sma50) });

            // Extract Exponential Moving Averages
            if (item.ema20 !== undefined) indicators.ema.ema20.data.push({ x: timestamp, y: parseFloat(item.ema20) });
            if (item.ema50 !== undefined) indicators.ema.ema50.data.push({ x: timestamp, y: parseFloat(item.ema50) });
            if (item.ema200 !== undefined) indicators.ema.ema200.data.push({ x: timestamp, y: parseFloat(item.ema200) });

            // Extract Bollinger Bands
            if (item.bbands_upper !== undefined) indicators.bbands.upper.data.push({ x: timestamp, y: parseFloat(item.bbands_upper) });
            if (item.bbands_middle !== undefined) indicators.bbands.middle.data.push({ x: timestamp, y: parseFloat(item.bbands_middle) });
            if (item.bbands_lower !== undefined) indicators.bbands.lower.data.push({ x: timestamp, y: parseFloat(item.bbands_lower) });

            // Extract RSI
            if (item.rsi14 !== undefined) indicators.rsi.data.push({ x: timestamp, y: parseFloat(item.rsi14) });
            
            // Extract MACD
            if (item.macd !== undefined) indicators.macd.macd.data.push({ x: timestamp, y: parseFloat(item.macd) });
            if (item.macd_signal !== undefined) indicators.macd.signal.data.push({ x: timestamp, y: parseFloat(item.macd_signal) });
            if (item.macd_hist !== undefined) indicators.macd.hist.data.push({ x: timestamp, y: parseFloat(item.macd_hist) });

            // Extract Stochastic
            if (item.stoch_k !== undefined) indicators.stoch.k.data.push({ x: timestamp, y: parseFloat(item.stoch_k) });
            if (item.stoch_d !== undefined) indicators.stoch.d.data.push({ x: timestamp, y: parseFloat(item.stoch_d) });

            // Extract ADX
            if (item.adx14 !== undefined) indicators.adx.data.push({ x: timestamp, y: parseFloat(item.adx14) });
            
            // Extract ATR
            if (item.atr14 !== undefined) indicators.atr.data.push({ x: timestamp, y: parseFloat(item.atr14) });

            // Extract OBV
            if (item.obv !== undefined) indicators.obv.data.push({ x: timestamp, y: parseFloat(item.obv) });
            
            // Extract VWAP
            if (item.vwap !== undefined) indicators.vwap.data.push({ x: timestamp, y: parseFloat(item.vwap) });

            // Extract Volume and Volume SMA
            if (item.volume !== undefined) indicators.volume.volume.data.push({ x: timestamp, y: parseFloat(item.volume), close: parseFloat(item.close) }); // Add close for coloring
            if (item.volume_sma20 !== undefined) indicators.volume.sma20.data.push({ x: timestamp, y: parseFloat(item.volume_sma20) });
        });

        console.log('Extracted indicator data:', indicators);
        return indicators;
    } catch (error) {
        console.error('Error extracting indicators:', error);
        return {};
    }
}

/**
 * Helper function to determine appropriate decimal precision for price display
 */
function getPrecision(price) {
    if (price === 0 || !price) return 2; // Added !price check
    if (price < 0.0001) return 8;
    if (price < 0.01) return 6;
    if (price < 0.1) return 4;
    if (price < 1) return 3;
    if (price < 10) return 2;
    return 2;
}

/**
 * Create a volatility chart using Chart.js
 */
function createVolatilityChart(symbol, timeframe, data) {
    try {
        console.log('Creating volatility chart for', symbol);
        const ctx = document.getElementById('volatility-chart').getContext('2d');
        if (volatilityChart) {
            volatilityChart.destroy();
        }
        if (volumeChart) {
            volumeChart.destroy();
            volumeChart = null;
            const volumeWrapper = document.querySelector('.volume-chart-wrapper');
            if (volumeWrapper) {
                volumeWrapper.remove();
            }
        }
        if (!data || !Array.isArray(data) || data.length === 0) {
            console.error('No volatility data available');
            return null;
        }
        const sortedData = [...data].sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
        console.log(`Volatility data prepared, ${sortedData.length} points available`);
        const labels = sortedData.map(item => new Date(item.timestamp));
        const values = sortedData.map(item => item.volatility);
        const isDarkTheme = !document.body.classList.contains('light-theme');
        const gridColor = isDarkTheme ? '#2a2e39' : '#e5e7eb';
        const textColor = isDarkTheme ? '#9ca3af' : '#666666';
        volatilityChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: `Volatility - ${symbol} (${timeframe})`,
                    data: values,
                    borderColor: '#2962ff',
                    backgroundColor: 'rgba(41, 98, 255, 0.1)',
                    borderWidth: 2,
                    pointRadius: 0,
                    pointHoverRadius: 5,
                    fill: true,
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: timeframe === '5m' ? 'minute' : 'hour',
                            displayFormats: { minute: 'HH:mm', hour: 'DD HH:mm' }
                        },
                        grid: { color: gridColor },
                        ticks: { color: textColor },
                        position: 'bottom',
                        display: true
                    },
                    x2: {
                        type: 'time',
                        time: {
                            unit: 'day',
                            displayFormats: {
                                day: 'DD MMM' // Show date in format "01 Jan" at bottom
                            }
                        },
                        grid: {
                            display: true, // Show grid lines for daily separation
                            color: function(context) {
                                return context.tick && context.tick.major ? 
                                    isDarkTheme ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)' : 
                                    'transparent';
                            },
                            lineWidth: 1
                        },
                        ticks: {
                            color: textColor,
                            maxRotation: 0, // Don't rotate labels
                            font: {
                                weight: 'bold',
                                size: 11
                            },
                            padding: 8,
                            major: {
                                enabled: true
                            },
                            autoSkip: false,
                            source: 'data'
                        },
                        position: 'bottom',
                        display: true
                    },
                    y: {
                        position: 'right',
                        grid: { color: gridColor },
                        ticks: {
                            color: textColor,
                            callback: function(value) { return value.toFixed(2) + '%'; }
                        }
                    }
                },
                plugins: {
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        callbacks: {
                            label: function(context) { return `Volatility: ${context.raw.toFixed(2)}%`; }
                        }
                    },
                    legend: { display: false },
                    crosshair: {
                        line: {
                            color: isDarkTheme ? 'rgba(255, 255, 255, 0.3)' : 'rgba(0, 0, 0, 0.3)',
                            width: 1
                        },
                        sync: {
                            enabled: true,
                            group: 1,
                            suppressTooltips: false
                        },
                        zoom: {
                            enabled: false
                        },
                        snap: {
                            enabled: true
                        }
                    },
                    zoom: {
                        zoom: {
                            wheel: {
                                enabled: true, // Enable zoom with mouse wheel
                                modifierKey: 'ctrl' // Require ctrl key to be pressed
                            },
                            pinch: {
                                enabled: true // Enable zoom with pinch gesture on touch devices
                            },
                            mode: 'xy', // Allow zooming both axes
                            speed: 0.1, // Adjust zoom speed
                            threshold: 2, // Minimum zoom level
                        },
                        pan: {
                            enabled: true,
                            mode: 'xy',
                            modifierKey: 'shift' // Require shift key to be pressed for panning
                        },
                        limits: {
                            x: {min: 'original', max: 'original'},
                            y: {min: 'original', max: 'original'}
                        }
                    }
                },
                animation: false
            }
        });
        console.log('Volatility chart created successfully');
        return volatilityChart;
    } catch (error) {
        console.error('Error creating volatility chart:', error);
        return null;
    }
}

/**
 * Transform API OHLCV data into the format required by Chart.js candlestick
 */
function prepareOHLCVData(data) {
    if (!data || !Array.isArray(data) || data.length === 0) return [];
    const sortedData = [...data].sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
    return sortedData.map(item => {
        let timestamp;
        try {
            const date = new Date(item.timestamp);
            timestamp = date.getTime();
        } catch (e) {
            console.error('Invalid timestamp:', item.timestamp);
            timestamp = Date.now();
        }
        return {
            x: timestamp,
            o: parseFloat(item.open),
            h: parseFloat(item.high),
            l: parseFloat(item.low),
            c: parseFloat(item.close),
            volume: parseFloat(item.volume)
        };
    });
}

/**
 * Calculate Heikin-Ashi candle data from regular OHLCV data
 */
function calculateHeikinAshi(data) {
    if (!data || !Array.isArray(data) || data.length === 0) return [];
    const result = [];
    for (let i = 0; i < data.length; i++) {
        const current = data[i];
        let haOpen, haClose, haHigh, haLow;
        haClose = (current.o + current.h + current.l + current.c) / 4;
        if (i === 0) {
            haOpen = current.o;
            haHigh = current.h;
            haLow = current.l;
        } else {
            const prev = result[i - 1];
            haOpen = (prev.o + prev.c) / 2;
            haHigh = Math.max(current.h, haOpen, haClose);
            haLow = Math.min(current.l, haOpen, haClose);
        }
        result.push({
            x: current.x, o: haOpen, h: haHigh, l: haLow, c: haClose,
            volume: current.volume,
            originalOpen: current.o, originalHigh: current.h,
            originalLow: current.l, originalClose: current.c
        });
    }
    return result;
}

/**
 * Render the pattern visualization
 */
function renderPatternVisualization(symbol, timeframe, patterns) {
    try {
        console.log('Rendering pattern visualization for', symbol);
        const container = document.getElementById('pattern-visualization');
        const infoContainer = document.getElementById('pattern-info');
        if (!container || !infoContainer) {
            console.error('Pattern container elements not found');
            return;
        }
        container.innerHTML = '';
        if (!patterns || Object.keys(patterns).length === 0) {
            console.log('No pattern data available');
            infoContainer.innerHTML = '<p>No pattern data available for this cryptocurrency.</p>';
            return;
        }
        const patternCount = Object.keys(patterns).length;
        infoContainer.innerHTML = `<p>Found ${patternCount} unique patterns for ${symbol} in ${timeframe} timeframe.</p><p>Each pattern represents a sequence of price movements (up/down).</p>`;
        console.log(`Found ${patternCount} patterns for ${symbol}`);
        const sortedPatterns = Object.entries(patterns).sort((a, b) => b[1] - a[1]).slice(0, 10);
        sortedPatterns.forEach(([pattern, count]) => {
            const card = document.createElement('div');
            card.className = 'pattern-card';
            const visualization = document.createElement('div');
            visualization.className = 'pattern-visualization';
            if (typeof pattern === 'string' && pattern.length > 0) {
                for (let i = 0; i < pattern.length; i++) {
                    const bit = document.createElement('div');
                    bit.className = `pattern-bit ${pattern[i] === '1' ? 'up' : 'down'}`;
                    bit.textContent = pattern[i];
                    visualization.appendChild(bit);
                }
            } else {
                const errorBit = document.createElement('div');
                errorBit.textContent = 'Invalid pattern';
                errorBit.style.color = 'red';
                visualization.appendChild(errorBit);
            }
            const info = document.createElement('div');
            info.className = 'pattern-info';
            info.textContent = `Occurrences: ${count}`;
            card.appendChild(visualization);
            card.appendChild(info);
            container.appendChild(card);
        });
        console.log('Pattern visualization rendered successfully');
    } catch (error) {
        console.error('Error rendering pattern visualization:', error);
    }
}

/**
 * Resize and redraw the price chart to fix display issues
 */
function resizePriceChart() {
    if (priceChart) {
        setTimeout(() => {
            priceChart.resize();
            if (volumeChart) {
                volumeChart.resize();
            }
        }, 10);
    }
}

/**
 * Resize and redraw the volatility chart to fix display issues
 */
function resizeVolatilityChart() {
    if (volatilityChart) {
        setTimeout(() => {
            volatilityChart.resize();
        }, 10);
    }
}

/**
 * Create a technical indicator chart
 * @param {string} symbol - Cryptocurrency symbol
 * @param {string} timeframe - Timeframe
 * @param {Array} data - Technical indicator data
 * @param {string} indicatorType - Type of indicator to display
 */
function createIndicatorChart(symbol, timeframe, data, indicatorType) {
    try {
        console.log(`Creating ${indicatorType} chart for ${symbol}`);
        
        // Get the canvas element
        const ctx = document.getElementById('indicator-chart').getContext('2d');
        
        // If chart already exists, destroy it
        if (indicatorChart) {
            indicatorChart.destroy();
        }
        
        if (!data || data.length === 0) {
            console.error('No indicator data available');
            return null;
        }
        
        const sortedData = [...data].sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
        const labels = sortedData.map(item => new Date(item.timestamp));
        
        // Define colors based on theme
        const isDarkTheme = !document.body.classList.contains('light-theme');
        const gridColor = isDarkTheme ? '#2a2e39' : '#e5e7eb';
        const textColor = isDarkTheme ? '#9ca3af' : '#666666';
        
        // Configure datasets and options based on the indicator type
        let chartConfig = {
            type: 'line',
            data: {
                labels: labels,
                datasets: []
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false
                },
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: timeframe === '5m' ? 'minute' : 'hour',
                            displayFormats: { minute: 'HH:mm', hour: 'DD HH:mm' }
                        },
                        grid: { color: gridColor },
                        ticks: { color: textColor },
                        position: 'bottom',
                        display: true
                    },
                    y: {
                        position: 'right',
                        grid: { color: gridColor },
                        ticks: { color: textColor }
                    }
                },
                plugins: {
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    },
                    legend: { display: true },
                    crosshair: {
                        line: {
                            color: isDarkTheme ? 'rgba(255, 255, 255, 0.3)' : 'rgba(0, 0, 0, 0.3)',
                            width: 1
                        },
                        sync: {
                            enabled: true,
                            group: 1, // Sync with price and volatility charts
                            suppressTooltips: false
                        },
                        zoom: {
                            enabled: false // Use chartjs-plugin-zoom
                        },
                        snap: {
                            enabled: true
                        }
                    },
                    zoom: {
                        zoom: {
                            wheel: {
                                enabled: true,
                                modifierKey: 'ctrl'
                            },
                            pinch: { enabled: true },
                            mode: 'xy'
                        },
                        pan: {
                            enabled: true,
                            mode: 'xy',
                            modifierKey: 'shift'
                        },
                        limits: {
                            x: {min: 'original', max: 'original'},
                            y: {min: 'original', max: 'original'}
                        }
                    }
                },
                animation: false
            }
        };
        
        // Configure chart based on indicator type
        switch (indicatorType) {
            case 'rsi':
                // RSI (Relative Strength Index)
                chartConfig.data.datasets = [{
                    label: 'RSI (14)',
                    data: sortedData.map(item => ({ x: new Date(item.timestamp), y: item.rsi14 })),
                    borderColor: '#8E24AA',
                    backgroundColor: 'rgba(142, 36, 170, 0.1)',
                    borderWidth: 1.5,
                    pointRadius: 0,
                    pointHoverRadius: 5,
                    fill: true
                }];
                
                // Add horizontal lines for overbought/oversold levels
                chartConfig.options.scales.y.min = 0;
                chartConfig.options.scales.y.max = 100;
                
                // Add annotations for overbought/oversold levels
                chartConfig.options.plugins.annotation = {
                    annotations: {
                        overbought: {
                            type: 'line',
                            yMin: 70,
                            yMax: 70,
                            borderColor: 'rgba(255, 99, 132, 0.8)',
                            borderWidth: 1,
                            borderDash: [5, 5],
                            label: {
                                content: 'Ipercomprato (70)',
                                position: 'start',
                                backgroundColor: 'rgba(255, 99, 132, 0.8)',
                                color: '#ffffff',
                                font: { weight: 'bold', size: 11 }
                            }
                        },
                        oversold: {
                            type: 'line',
                            yMin: 30,
                            yMax: 30,
                            borderColor: 'rgba(54, 162, 235, 0.8)',
                            borderWidth: 1,
                            borderDash: [5, 5],
                            label: {
                                content: 'Ipervenduto (30)',
                                position: 'start',
                                backgroundColor: 'rgba(54, 162, 235, 0.8)',
                                color: '#ffffff',
                                font: { weight: 'bold', size: 11 }
                            }
                        }
                    }
                };
                
                // Custom tooltip format for RSI
                chartConfig.options.plugins.tooltip.callbacks = {
                    label: function(context) {
                        let value = context.raw.y;
                        if (value === null || value === undefined) return '';
                        return `RSI: ${value.toFixed(2)}`;
                    }
                };
                break;
                
            case 'macd':
                // MACD (Moving Average Convergence Divergence)
                chartConfig = {
                    type: 'bar',
                    data: {
                        labels: labels,
                        datasets: [
                            {
                                label: 'MACD Histogram',
                                data: sortedData.map(item => ({ 
                                    x: new Date(item.timestamp), 
                                    y: item.macd_hist
                                })),
                                backgroundColor: function(context) {
                                    const value = context.dataset.data[context.dataIndex].y;
                                    return value >= 0 ? 'rgba(38, 166, 154, 0.6)' : 'rgba(239, 83, 80, 0.6)';
                                },
                                order: 1
                            },
                            {
                                label: 'MACD Line',
                                data: sortedData.map(item => ({ 
                                    x: new Date(item.timestamp), 
                                    y: item.macd 
                                })),
                                borderColor: '#2962FF',
                                borderWidth: 2,
                                pointRadius: 0,
                                pointHoverRadius: 5,
                                type: 'line',
                                order: 0
                            },
                            {
                                label: 'Signal Line',
                                data: sortedData.map(item => ({ 
                                    x: new Date(item.timestamp), 
                                    y: item.macd_signal
                                })),
                                borderColor: '#FF6D00',
                                borderWidth: 2,
                                pointRadius: 0,
                                pointHoverRadius: 5,
                                type: 'line',
                                order: 0
                            }
                        ]
                    },
                    options: chartConfig.options
                };
                break;
                
            case 'stoch':
                // Stochastic Oscillator
                chartConfig.data.datasets = [
                    {
                        label: '%K',
                        data: sortedData.map(item => ({ 
                            x: new Date(item.timestamp), 
                            y: item.stoch_k 
                        })),
                        borderColor: '#2962FF',
                        backgroundColor: 'rgba(41, 98, 255, 0.1)',
                        borderWidth: 1.5,
                        pointRadius: 0,
                        pointHoverRadius: 5,
                        fill: false
                    },
                    {
                        label: '%D',
                        data: sortedData.map(item => ({ 
                            x: new Date(item.timestamp), 
                            y: item.stoch_d 
                        })),
                        borderColor: '#FF6D00',
                        backgroundColor: 'rgba(255, 109, 0, 0.1)',
                        borderWidth: 1.5,
                        pointRadius: 0,
                        pointHoverRadius: 5,
                        fill: false
                    }
                ];
                
                // Add horizontal threshold lines
                chartConfig.options.scales.y.min = 0;
                chartConfig.options.scales.y.max = 100;
                
                // Add annotations
                chartConfig.options.plugins.annotation = {
                    annotations: {
                        overbought: {
                            type: 'line',
                            yMin: 80,
                            yMax: 80,
                            borderColor: 'rgba(255, 99, 132, 0.8)',
                            borderWidth: 1,
                            borderDash: [5, 5],
                            label: {
                                content: 'Ipercomprato (80)',
                                position: 'start',
                                backgroundColor: 'rgba(255, 99, 132, 0.8)',
                                color: '#ffffff',
                                font: { weight: 'bold', size: 11 }
                            }
                        },
                        oversold: {
                            type: 'line',
                            yMin: 20,
                            yMax: 20,
                            borderColor: 'rgba(54, 162, 235, 0.8)',
                            borderWidth: 1,
                            borderDash: [5, 5],
                            label: {
                                content: 'Ipervenduto (20)',
                                position: 'start',
                                backgroundColor: 'rgba(54, 162, 235, 0.8)',
                                color: '#ffffff',
                                font: { weight: 'bold', size: 11 }
                            }
                        }
                    }
                };
                break;
                
            case 'bbands':
                // Bollinger Bands
                chartConfig.data.datasets = [
                    {
                        label: 'Upper Band',
                        data: sortedData.map(item => ({ 
                            x: new Date(item.timestamp), 
                            y: item.bbands_upper
                        })),
                        borderColor: 'rgba(255, 99, 132, 0.8)',
                        backgroundColor: 'rgba(255, 99, 132, 0.1)',
                        borderWidth: 1.5,
                        pointRadius: 0,
                        pointHoverRadius: 5,
                        fill: false
                    },
                    {
                        label: 'Middle Band',
                        data: sortedData.map(item => ({ 
                            x: new Date(item.timestamp), 
                            y: item.bbands_middle
                        })),
                        borderColor: 'rgba(54, 162, 235, 0.8)',
                        backgroundColor: 'rgba(54, 162, 235, 0.1)',
                        borderWidth: 1.5,
                        pointRadius: 0,
                        pointHoverRadius: 5,
                        fill: false
                    },
                    {
                        label: 'Lower Band',
                        data: sortedData.map(item => ({ 
                            x: new Date(item.timestamp), 
                            y: item.bbands_lower
                        })),
                        borderColor: 'rgba(75, 192, 192, 0.8)',
                        backgroundColor: 'rgba(75, 192, 192, 0.1)',
                        borderWidth: 1.5,
                        pointRadius: 0,
                        pointHoverRadius: 5,
                        fill: false
                    }
                ];
                break;
                
            case 'adx':
                // ADX (Average Directional Index)
                chartConfig.data.datasets = [{
                    label: 'ADX (14)',
                    data: sortedData.map(item => ({ 
                        x: new Date(item.timestamp), 
                        y: item.adx14 
                    })),
                    borderColor: '#673AB7',
                    backgroundColor: 'rgba(103, 58, 183, 0.1)',
                    borderWidth: 1.5,
                    pointRadius: 0,
                    pointHoverRadius: 5,
                    fill: true
                }];
                
                // ADX threshold lines
                chartConfig.options.plugins.annotation = {
                    annotations: {
                        strongTrend: {
                            type: 'line',
                            yMin: 25,
                            yMax: 25,
                            borderColor: 'rgba(54, 162, 235, 0.8)',
                            borderWidth: 1,
                            borderDash: [5, 5],
                            label: {
                                content: 'Trend forte (25)',
                                position: 'start',
                                backgroundColor: 'rgba(54, 162, 235, 0.8)',
                                color: '#ffffff',
                                font: { weight: 'bold', size: 11 }
                            }
                        }
                    }
                };
                break;
                
            case 'atr':
                // ATR (Average True Range)
                chartConfig.data.datasets = [{
                    label: 'ATR (14)',
                    data: sortedData.map(item => ({ 
                        x: new Date(item.timestamp), 
                        y: item.atr14
                    })),
                    borderColor: '#9C27B0',
                    backgroundColor: 'rgba(156, 39, 176, 0.1)',
                    borderWidth: 1.5,
                    pointRadius: 0,
                    pointHoverRadius: 5,
                    fill: true
                }];
                break;
                
            case 'obv':
                // OBV (On-Balance Volume)
                chartConfig.data.datasets = [{
                    label: 'OBV',
                    data: sortedData.map(item => ({ 
                        x: new Date(item.timestamp), 
                        y: item.obv
                    })),
                    borderColor: '#3F51B5',
                    backgroundColor: 'rgba(63, 81, 181, 0.1)',
                    borderWidth: 1.5,
                    pointRadius: 0,
                    pointHoverRadius: 5,
                    fill: true
                }];
                break;
                
            case 'vwap':
                // VWAP (Volume-Weighted Average Price)
                chartConfig.data.datasets = [{
                    label: 'VWAP',
                    data: sortedData.map(item => ({ 
                        x: new Date(item.timestamp), 
                        y: item.vwap
                    })).filter(item => item.y !== null && item.y !== undefined),
                    borderColor: '#009688',
                    backgroundColor: 'rgba(0, 150, 136, 0.1)',
                    borderWidth: 1.5,
                    pointRadius: 0,
                    pointHoverRadius: 5,
                    fill: true
                }];
                break;
                
            case 'volume':
                // Volume chart
                chartConfig = {
                    type: 'bar',
                    data: {
                        labels: labels,
                        datasets: [{
                            label: 'Volume',
                            data: sortedData.map(item => ({ 
                                x: new Date(item.timestamp), 
                                y: item.volume
                            })),
                            backgroundColor: sortedData.map((item, i) => {
                                if (i > 0 && sortedData[i].close > sortedData[i-1].close) {
                                    return 'rgba(38, 166, 154, 0.6)'; // Up volume
                                } else {
                                    return 'rgba(239, 83, 80, 0.6)'; // Down volume
                                }
                            }),
                            borderColor: 'transparent',
                            borderWidth: 0
                        }, {
                            label: 'Volume SMA (20)',
                            data: sortedData.map(item => ({ 
                                x: new Date(item.timestamp), 
                                y: item.volume_sma20
                            })),
                            borderColor: '#FF9800',
                            borderWidth: 1.5,
                            pointRadius: 0,
                            pointHoverRadius: 0,
                            type: 'line',
                            fill: false
                        }]
                    },
                    options: {
                        ...chartConfig.options,
                        scales: {
                            ...chartConfig.options.scales,
                            y: {
                                ...chartConfig.options.scales.y,
                                ticks: {
                                    callback: function(value) {
                                        if (value === 0) return '';
                                        if (value >= 1000000) return (value / 1000000).toFixed(1) + 'M';
                                        if (value >= 1000) return (value / 1000).toFixed(1) + 'K';
                                        return value;
                                    }
                                }
                            }
                        },
                        plugins: {
                            ...chartConfig.options.plugins,
                            tooltip: {
                                ...chartConfig.options.plugins.tooltip,
                                callbacks: {
                                    label: function(context) {
                                        const volume = context.raw.y;
                                        if (context.dataset.label === "Volume") {
                                            if (volume >= 1000000) return `Volume: ${(volume / 1000000).toFixed(2)}M`;
                                            if (volume >= 1000) return `Volume: ${(volume / 1000).toFixed(2)}K`;
                                            return `Volume: ${volume.toFixed(2)}`;
                                        } else {
                                            return `${context.dataset.label}: ${volume}`;
                                        }
                                    }
                                }
                            }
                        }
                    }
                };
                break;
                
            default:
                console.warn(`No configuration for indicator type: ${indicatorType}`);
                return null;
        }
        
        // Create the chart
        indicatorChart = new Chart(ctx, chartConfig);
        window.indicatorChart = indicatorChart;
        
        console.log(`${indicatorType} chart created successfully`);
        return indicatorChart;
    } catch (error) {
        console.error(`Error creating ${indicatorType} chart:`, error);
        return null;
    }
}

/**
 * Resize and redraw the indicator chart to fix display issues
 */
function resizeIndicatorChart() {
    if (indicatorChart) {
        setTimeout(() => {
            indicatorChart.resize();
        }, 10);
    }
}

// Make functions available globally
window.ChartHandler = {
    createPriceChart,
    createVolatilityChart,
    createIndicatorChart,
    renderPatternVisualization,
    resizePriceChart,
    resizeVolatilityChart,
    resizeIndicatorChart
};

// Expose chart instances globally for reset zoom functionality
window.priceChart = priceChart;
window.volumeChart = volumeChart;
window.volatilityChart = volatilityChart;
window.indicatorChart = indicatorChart;

// Update window chart references when creating new charts
const originalCreatePriceChart = createPriceChart;
const originalCreateVolatilityChart = createVolatilityChart;
const originalCreateIndicatorChart = createIndicatorChart;

window.ChartHandler.createPriceChart = function(...args) {
    const chart = originalCreatePriceChart.apply(this, args);
    window.priceChart = priceChart;
    return chart;
};

window.ChartHandler.createVolatilityChart = function(...args) {
    const chart = originalCreateVolatilityChart.apply(this, args);
    window.volatilityChart = volatilityChart;
    return chart;
};

window.ChartHandler.createIndicatorChart = function(...args) {
    const chart = originalCreateIndicatorChart.apply(this, args);
    window.indicatorChart = indicatorChart;
    return chart;
};

// Helper function to create the custom HTML tooltip
const getOrCreateTooltip = (chart) => {
    let tooltipEl = chart.canvas.parentNode.querySelector('div.chartjs-tooltip');
    if (!tooltipEl) {
        tooltipEl = document.createElement('div');
        tooltipEl.classList.add('chartjs-tooltip');
        tooltipEl.style.opacity = 1;
        tooltipEl.style.pointerEvents = 'none';
        tooltipEl.style.position = 'absolute';
        tooltipEl.style.transform = 'translate(-50%, 0)';
        tooltipEl.style.transition = 'all .1s ease';
        tooltipEl.style.padding = '12px';
        tooltipEl.style.borderRadius = '8px';
        tooltipEl.style.boxShadow = '0 4px 6px rgba(0,0,0,0.3)';
        tooltipEl.style.fontFamily = '"Inter", sans-serif';
        tooltipEl.style.zIndex = '1000';
        const table = document.createElement('table');
        table.style.margin = '0px';
        tooltipEl.appendChild(table);
        chart.canvas.parentNode.appendChild(tooltipEl);
    }
    return tooltipEl;
};

const externalTooltipHandler = (context) => {
    const {chart, tooltip} = context;
    const tooltipEl = getOrCreateTooltip(chart);

    const isDarkTheme = !document.body.classList.contains('light-theme');
    tooltipEl.style.background = isDarkTheme ? 'rgba(20, 22, 30, 0.95)' : 'rgba(255, 255, 255, 0.95)';
    tooltipEl.style.color = isDarkTheme ? '#d1d4dc' : '#333333';
    tooltipEl.style.border = `1px solid ${isDarkTheme ? '#363c4e' : '#d1d5db'}`;

    if (tooltip.opacity === 0) {
        tooltipEl.style.opacity = 0;
        return;
    }

    if (tooltip.body) {
        const titleLines = tooltip.title || [];
        const tableHead = document.createElement('thead');
        tableHead.style.fontWeight = 'bold';
        tableHead.style.fontSize = '14px';
        titleLines.forEach(title => {
            const tr = document.createElement('tr');
            tr.style.borderWidth = 0;
            const th = document.createElement('th');
            th.style.borderWidth = 0;
            th.style.textAlign = 'left';
            th.style.paddingBottom = '5px';
            const text = document.createTextNode(title);
            th.appendChild(text);
            tr.appendChild(th);
            tableHead.appendChild(tr);
        });

        const tableBody = document.createElement('tbody');
        tableBody.style.fontSize = '13px';
        tableBody.style.fontWeight = 'bold';
        const dataPoint = tooltip.dataPoints && tooltip.dataPoints.length > 0 ? tooltip.dataPoints[0].raw : null;

        if (dataPoint) {
            const isLineChart = chart.config.type === 'line';
            const precision = getPrecision(isLineChart ? dataPoint.y : dataPoint.c);
            const upColor = '#26a69a'; 
            const downColor = '#ef5350';
            
            let tooltipRows = [];

            if (isLineChart) {
                tooltipRows.push(`<div style="font-weight:bold;font-size:16px;margin-bottom:5px;">Value:</div>`);
                tooltipRows.push(`<div>Close: ${dataPoint.y ? dataPoint.y.toFixed(precision) : 'N/A'}</div>`);
            } else {
                 // Candlestick or Heikin-Ashi
                const ohlcStyle = dataPoint.originalClose >= dataPoint.originalOpen ? `color: ${upColor}; font-weight: bold;` : `color: ${downColor}; font-weight: bold;`;
                tooltipRows = [
                    `<div style="font-weight:bold;font-size:16px;margin-bottom:5px;">📊 Values (Original):</div>`,
                    `<div style="${ohlcStyle}">Open: ${dataPoint.originalOpen ? dataPoint.originalOpen.toFixed(precision) : 'N/A'}</div>`,
                    `<div style="${ohlcStyle}">High: ${dataPoint.originalHigh ? dataPoint.originalHigh.toFixed(precision) : 'N/A'}</div>`,
                    `<div style="${ohlcStyle}">Low: ${dataPoint.originalLow ? dataPoint.originalLow.toFixed(precision) : 'N/A'}</div>`,
                    `<div style="${ohlcStyle}">Close: ${dataPoint.originalClose ? dataPoint.originalClose.toFixed(precision) : 'N/A'}</div>`
                ];
                // If Heikin-Ashi, show HA values too
                if (dataPoint.o !== dataPoint.originalOpen || dataPoint.c !== dataPoint.originalClose) {
                    const haStyle = dataPoint.c >= dataPoint.o ? `color: ${upColor}; font-weight: bold;` : `color: ${downColor}; font-weight: bold;`;
                    tooltipRows.push(`<div style="font-weight:bold;font-size:14px;margin-top:8px;margin-bottom:3px;">Heikin-Ashi:</div>`);
                    tooltipRows.push(`<div style="${haStyle}">HA Open: ${dataPoint.o ? dataPoint.o.toFixed(precision) : 'N/A'}</div>`);
                    tooltipRows.push(`<div style="${haStyle}">HA High: ${dataPoint.h ? dataPoint.h.toFixed(precision) : 'N/A'}</div>`);
                    tooltipRows.push(`<div style="${haStyle}">HA Low: ${dataPoint.l ? dataPoint.l.toFixed(precision) : 'N/A'}</div>`);
                    tooltipRows.push(`<div style="${haStyle}">HA Close: ${dataPoint.c ? dataPoint.c.toFixed(precision) : 'N/A'}</div>`);
                }
            }
            
            // Add indicator values if available (common for all chart types)
            if (tooltip.dataPoints && tooltip.dataPoints.length > 1) {
                // Start indicators section
                tooltipRows.push(`<div style="font-weight:bold;font-size:16px;margin-top:10px;margin-bottom:5px;">📈 Indicators:</div>`);
                
                // Add indicator values
                tooltip.dataPoints.forEach(point => {
                    // Skip the candlestick dataset point which we already handled
                    if (point.dataset.type === 'candlestick') return;
                    
                    // Skip points with undefined values
                    if (point.raw && point.raw.y !== undefined) {
                        const indicatorName = point.dataset.label || 'Indicator';
                        const indicatorValue = parseFloat(point.raw.y).toFixed(precision);
                        const indicatorColor = point.dataset.borderColor || '#888';
                        
                        tooltipRows.push(`<div style="color: ${indicatorColor}; font-weight: bold;">${indicatorName}: ${indicatorValue}</div>`);
                    }
                });
            }
            
            tooltipRows.forEach(htmlContent => {
                const tr = document.createElement('tr');
                tr.style.borderWidth = 0;
                const td = document.createElement('td');
                td.style.borderWidth = 0;
                td.innerHTML = htmlContent;
                tr.appendChild(td);
                tableBody.appendChild(tr);
            });
        }

        const tableRoot = tooltipEl.querySelector('table');
        while (tableRoot.firstChild) {
            tableRoot.firstChild.remove();
        }
        tableRoot.appendChild(tableHead);
        tableRoot.appendChild(tableBody);
    }

    const {offsetLeft: positionX, offsetTop: positionY} = chart.canvas;
    tooltipEl.style.opacity = 1;
    let tooltipX = positionX + tooltip.caretX - tooltipEl.offsetWidth - 10;
    let tooltipY = positionY + tooltip.caretY - (tooltipEl.offsetHeight / 2);

    if (tooltipX < 0) tooltipX = positionX + tooltip.caretX + 10;
    if (tooltipX + tooltipEl.offsetWidth > chart.canvas.width) tooltipX = chart.canvas.width - tooltipEl.offsetWidth - 5;
    if (tooltipY < 0) tooltipY = 5;
    if (tooltipY + tooltipEl.offsetHeight > chart.canvas.height) tooltipY = chart.canvas.height - tooltipEl.offsetHeight - 5;

    tooltipEl.style.left = tooltipX + 'px';
    tooltipEl.style.top = tooltipY + 'px';
};
