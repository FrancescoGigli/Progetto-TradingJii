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

/**
 * Create a price candlestick chart using Chart.js with TradingView style
 * @param {string} symbol - Cryptocurrency symbol
 * @param {string} timeframe - Timeframe like '5m', '15m'
 * @param {Array} data - OHLCV data from the API
 */
function createPriceChart(symbol, timeframe, data) {
    try {
        console.log('Creating price chart for', symbol);
        
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
        
        // Calculate Heikin-Ashi data
        const haData = calculateHeikinAshi(chartData);
        
        console.log(`Chart data prepared, ${haData.length} points available`);
        
        // Define colors based on theme
        const isDarkTheme = !document.body.classList.contains('light-theme');
        const gridColor = isDarkTheme ? '#2a2e39' : '#e5e7eb';
        const textColor = isDarkTheme ? '#9ca3af' : '#666666';
        
        // Define colors for candlesticks with improved visibility
        const upColor = '#26a69a'; // Green for up candles
        const downColor = '#ef5350'; // Red for down candles
        const upColorFill = 'rgba(38, 166, 154, 0.85)'; // More opaque fill for up candles
        const downColorFill = 'rgba(239, 83, 80, 0.85)'; // More opaque fill for down candles
        
        console.log('Creating Heikin-Ashi chart for', symbol);
        
        // Create Heikin-Ashi candlestick chart
        priceChart = new Chart(ctx, {
            type: 'candlestick',
            data: {
                datasets: [{
                    label: `${symbol} (${timeframe}) - Heikin-Ashi`,
                    data: haData.map(d => ({
                        x: d.x,
                        o: d.o,
                        h: d.h,
                        l: d.l,
                        c: d.c,
                        // Store original values for tooltip
                        originalOpen: d.originalOpen,
                        originalHigh: d.originalHigh,
                        originalLow: d.originalLow,
                        originalClose: d.originalClose
                    })),
                    color: {
                        up: upColor,
                        down: downColor,
                        unchanged: '#888888',
                    },
                    borderColor: function(context) {
                        return context.dataset.data[context.dataIndex].o > context.dataset.data[context.dataIndex].c ? 
                            downColor : upColor;
                    },
                    borderWidth: 2.5, // Thicker border for better visibility
                    wickWidth: 2, // Thicker wicks
                    barPercentage: 0.92,
                    barThickness: 14, // Wider candles for better visibility
                    backgroundColor: function(context) {
                        return context.dataset.data[context.dataIndex].o > context.dataset.data[context.dataIndex].c ? 
                            downColorFill : upColorFill;
                    },
                    pointHoverRadius: 5,  // Makes hover area larger
                    pointHoverBorderWidth: 2
                }]
            },
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
                        }
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
                    }
                },
                animation: false 
            }
        });

        // Add annotation for the last closing price if data is available
        if (haData.length > 0) {
            const lastCandle = haData[haData.length - 1];
            const lastOriginalClosePrice = lastCandle.originalClose;
            const precision = getPrecision(lastOriginalClosePrice);
            const lastCandleColor = lastCandle.originalClose >= lastCandle.originalOpen ? upColor : downColor;

            priceChart.options.plugins.annotation.annotations.lastPriceLine = {
                type: 'line',
                yMin: lastOriginalClosePrice,
                yMax: lastOriginalClosePrice,
                borderColor: lastCandleColor,
                borderWidth: 1.5,
                borderDash: [6, 6],
                label: {
                    enabled: true,
                    content: lastOriginalClosePrice.toFixed(precision),
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
        
        console.log('Candlestick chart created successfully');
        
        try {
            createColoredVolumeChart(symbol, timeframe, haData);
        } catch (volumeError) {
            console.error('Error creating volume chart:', volumeError);
        }
        
        console.log('Heikin-Ashi chart created successfully');
        
        return priceChart;
    } catch (error) {
        console.error('Error creating price chart:', error);
        return null;
    }
}

/**
 * Create a volume bar chart with colors matching candlesticks
 */
function createColoredVolumeChart(symbol, timeframe, chartData) {
    const volumeCanvas = document.getElementById('volume-chart');
    if (!volumeCanvas) {
        const priceChartWrapper = document.getElementById('price-chart-wrapper');
        const volumeWrapper = document.createElement('div');
        volumeWrapper.className = 'volume-chart-wrapper';
        volumeWrapper.style.height = '20%';
        volumeWrapper.style.marginTop = '10px';
        const canvas = document.createElement('canvas');
        canvas.id = 'volume-chart';
        volumeWrapper.appendChild(canvas);
        if (priceChartWrapper && priceChartWrapper.parentNode) {
            priceChartWrapper.parentNode.insertBefore(volumeWrapper, priceChartWrapper.nextSibling);
        }
    }
    const volumeCtx = document.getElementById('volume-chart').getContext('2d');
    if (volumeChart) {
        volumeChart.destroy();
    }
    const upColorVol = 'rgba(38, 166, 154, 0.6)';
    const downColorVol = 'rgba(239, 83, 80, 0.6)';
    const timestamps = chartData.map(d => d.x);
    const volumes = chartData.map(d => d.volume);
    const colors = chartData.map(d => d.c >= d.o ? upColorVol : downColorVol);
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
                        ticks: { color: textColor }
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
                    legend: { display: false }
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

// Make functions available globally
window.ChartHandler = {
    createPriceChart,
    createVolatilityChart,
    renderPatternVisualization,
    resizePriceChart,
    resizeVolatilityChart
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
            const precision = getPrecision(dataPoint.c);
            const upColor = '#26a69a'; 
            const downColor = '#ef5350';
            const haColorStyle = dataPoint.c >= dataPoint.o ? `color: ${upColor}; font-weight: bold;` : `color: ${downColor}; font-weight: bold;`;
            const originalColorStyle = dataPoint.originalClose >= dataPoint.originalOpen ? `color: ${upColor}; font-weight: bold;` : `color: ${downColor}; font-weight: bold;`;

            const tooltipRows = [
                `<div style="font-weight:bold;font-size:16px;margin-bottom:5px;">📊 Values:</div>`,
                `<div style="${originalColorStyle}">Open: ${dataPoint.originalOpen ? dataPoint.originalOpen.toFixed(precision) : 'N/A'}</div>`,
                `<div style="${originalColorStyle}">High: ${dataPoint.originalHigh ? dataPoint.originalHigh.toFixed(precision) : 'N/A'}</div>`,
                `<div style="${originalColorStyle}">Low: ${dataPoint.originalLow ? dataPoint.originalLow.toFixed(precision) : 'N/A'}</div>`,
                `<div style="${originalColorStyle}">Close: ${dataPoint.originalClose ? dataPoint.originalClose.toFixed(precision) : 'N/A'}</div>`
            ];
            
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
