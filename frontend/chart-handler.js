/**
 * Chart Handler Module for TradingJii Dashboard
 * 
 * This module provides functions for rendering price and volatility charts using Chart.js.
 */

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
                    borderWidth: 2, // Thinner border for cleaner look
                    wickWidth: 1.5, // Slightly thinner wicks
                    barPercentage: 0.9,
                    barThickness: 12, // Slightly wider candles for better visibility
                    backgroundColor: function(context) {
                        return context.dataset.data[context.dataIndex].o > context.dataset.data[context.dataIndex].c ? 
                            downColorFill : upColorFill;
                    }
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
                            color: textColor
                        }
                    }
                },
                plugins: {
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        callbacks: {
                            label: function(tooltipItem) {
                                const dataPoint = tooltipItem.raw;
                                const precision = getPrecision(dataPoint.c);
                                
                                return [
                                    `Heikin-Ashi Values:`,
                                    `HA-Open: ${dataPoint.o.toFixed(precision)}`,
                                    `HA-High: ${dataPoint.h.toFixed(precision)}`,
                                    `HA-Low: ${dataPoint.l.toFixed(precision)}`,
                                    `HA-Close: ${dataPoint.c.toFixed(precision)}`,
                                    ``,
                                    `Original Values:`,
                                    `Open: ${dataPoint.originalOpen.toFixed(precision)}`,
                                    `High: ${dataPoint.originalHigh.toFixed(precision)}`,
                                    `Low: ${dataPoint.originalLow.toFixed(precision)}`,
                                    `Close: ${dataPoint.originalClose.toFixed(precision)}`
                                ];
                            }
                        }
                    },
                    legend: {
                        display: false
                    }
                },
                animation: false // Disable animation for better performance
            }
        });
        
        console.log('Candlestick chart created successfully');
        
        // Create a volume bar chart with original data (not HA)
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
    // Try to get the canvas element
    const volumeCanvas = document.getElementById('volume-chart');
    
    // If volume canvas doesn't exist, create it
    if (!volumeCanvas) {
        const priceChartWrapper = document.getElementById('price-chart-wrapper');
        
        // Create a container for the volume chart
        const volumeWrapper = document.createElement('div');
        volumeWrapper.className = 'volume-chart-wrapper';
        volumeWrapper.style.height = '20%';
        volumeWrapper.style.marginTop = '10px';
        
        // Create the canvas
        const canvas = document.createElement('canvas');
        canvas.id = 'volume-chart';
        volumeWrapper.appendChild(canvas);
        
        // Add to DOM after price chart
        if (priceChartWrapper && priceChartWrapper.parentNode) {
            priceChartWrapper.parentNode.insertBefore(volumeWrapper, priceChartWrapper.nextSibling);
        }
    }
    
    // Get the context
    const volumeCtx = document.getElementById('volume-chart').getContext('2d');
    
    // If volume chart already exists, destroy it
    if (volumeChart) {
        volumeChart.destroy();
    }
    
    // Define colors for up and down volume bars
    const upColor = 'rgba(38, 166, 154, 0.6)';  // Green for up volume
    const downColor = 'rgba(239, 83, 80, 0.6)'; // Red for down volume
    
    const timestamps = chartData.map(d => d.x);
    const volumes = chartData.map(d => d.volume);
    const colors = chartData.map(d => d.c >= d.o ? upColor : downColor);
    
    // Create a colored volume chart
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
                    grid: {
                        display: false
                    },
                    ticks: {
                        callback: function(value) {
                            if (value === 0) return '';
                            if (value >= 1000000) return (value / 1000000).toFixed(1) + 'M';
                            if (value >= 1000) return (value / 1000).toFixed(1) + 'K';
                            return value;
                        }
                    }
                },
                x: {
                    display: false
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const volume = context.raw;
                            if (volume >= 1000000) {
                                return `Volume: ${(volume / 1000000).toFixed(2)}M`;
                            } else if (volume >= 1000) {
                                return `Volume: ${(volume / 1000).toFixed(2)}K`;
                            }
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
    if (price === 0) return 2;
    
    // For very small values (like many crypto prices), use more decimal places
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
        
        // Get the canvas element for the volatility chart
        const ctx = document.getElementById('volatility-chart').getContext('2d');
        
        // Destroy existing volatility chart if it exists
        if (volatilityChart) {
            volatilityChart.destroy();
        }
        
        // We need to hide any volume chart when showing volatility
        if (volumeChart) {
            volumeChart.destroy();
            volumeChart = null;
            
            // Remove any volume chart container that might exist
            const volumeWrapper = document.querySelector('.volume-chart-wrapper');
            if (volumeWrapper) {
                volumeWrapper.remove();
            }
        }
        
        if (!data || !Array.isArray(data) || data.length === 0) {
            console.error('No volatility data available');
            return null;
        }
        
        // Sort data by timestamp in ascending order
        const sortedData = [...data].sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
        
        console.log(`Volatility data prepared, ${sortedData.length} points available`);
        
        // Prepare labels and values
        const labels = sortedData.map(item => new Date(item.timestamp));
        const values = sortedData.map(item => item.volatility);
        
        // Define colors based on theme
        const isDarkTheme = !document.body.classList.contains('light-theme');
        
        // Set theme-specific colors
        const gridColor = isDarkTheme ? '#2a2e39' : '#e5e7eb';
        const textColor = isDarkTheme ? '#9ca3af' : '#666666';
        
        // Create a simple volatility chart
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
                            displayFormats: {
                                minute: 'HH:mm',
                                hour: 'DD HH:mm'
                            }
                        },
                        grid: {
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
                            callback: function(value) {
                                return value.toFixed(2) + '%';
                            }
                        }
                    }
                },
                plugins: {
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        callbacks: {
                            label: function(context) {
                                return `Volatility: ${context.raw.toFixed(2)}%`;
                            }
                        }
                    },
                    legend: {
                        display: false
                    }
                },
                animation: false // Disable animation for better performance
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
    if (!data || !Array.isArray(data) || data.length === 0) {
        return [];
    }
    
    // Sort data by timestamp in ascending order
    const sortedData = [...data].sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
    
    // Transform for Chart.js
    return sortedData.map(item => {
        const timestamp = new Date(item.timestamp);
        const open = parseFloat(item.open);
        const high = parseFloat(item.high);
        const low = parseFloat(item.low);
        const close = parseFloat(item.close);
        const volume = parseFloat(item.volume);
        
        // Return format compatible with candlestick drawing
        return {
            x: timestamp,
            o: open,
            h: high,
            l: low,
            c: close,
            volume: volume
        };
    });
}

/**
 * Calculate Heikin-Ashi candle data from regular OHLCV data
 * Heikin-Ashi formula:
 * - HA-Close = (Open + High + Low + Close) / 4
 * - HA-Open = (Previous HA-Open + Previous HA-Close) / 2
 * - HA-High = Max(High, HA-Open, HA-Close)
 * - HA-Low = Min(Low, HA-Open, HA-Close)
 */
function calculateHeikinAshi(data) {
    if (!data || !Array.isArray(data) || data.length === 0) {
        return [];
    }
    
    const result = [];
    
    // Process each candle
    for (let i = 0; i < data.length; i++) {
        const current = data[i];
        
        // Calculate Heikin-Ashi values
        let haOpen, haClose, haHigh, haLow;
        
        // Calculate HA Close (always the same formula)
        haClose = (current.o + current.h + current.l + current.c) / 4;
        
        if (i === 0) {
            // For the first candle, use regular values
            haOpen = current.o;
            haHigh = current.h;
            haLow = current.l;
        } else {
            // For subsequent candles, use the HA formula
            const prev = result[i - 1];
            haOpen = (prev.o + prev.c) / 2;
            haHigh = Math.max(current.h, haOpen, haClose);
            haLow = Math.min(current.l, haOpen, haClose);
        }
        
        // Add to result array
        result.push({
            x: current.x,
            o: haOpen,
            h: haHigh,
            l: haLow,
            c: haClose,
            volume: current.volume,
            // Keep original values for reference if needed
            originalOpen: current.o,
            originalHigh: current.h,
            originalLow: current.l,
            originalClose: current.c
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
        
        // Safety checks
        if (!container || !infoContainer) {
            console.error('Pattern container elements not found');
            return;
        }
        
        // Clear previous content
        container.innerHTML = '';
        
        // If no patterns available
        if (!patterns || Object.keys(patterns).length === 0) {
            console.log('No pattern data available');
            infoContainer.innerHTML = '<p>No pattern data available for this cryptocurrency.</p>';
            return;
        }
        
        // Update info text
        const patternCount = Object.keys(patterns).length;
        infoContainer.innerHTML = `
            <p>Found ${patternCount} unique patterns for ${symbol} in ${timeframe} timeframe.</p>
            <p>Each pattern represents a sequence of price movements (up/down).</p>
        `;
        
        console.log(`Found ${patternCount} patterns for ${symbol}`);
        
        // Sort patterns by number of occurrences (most frequent first)
        const sortedPatterns = Object.entries(patterns)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 10); // Show top 10 patterns
        
        // Create pattern cards
        sortedPatterns.forEach(([pattern, count]) => {
            const card = document.createElement('div');
            card.className = 'pattern-card';
            
            // Create visualization
            const visualization = document.createElement('div');
            visualization.className = 'pattern-visualization';
            
            // Safety check for pattern string
            if (typeof pattern === 'string' && pattern.length > 0) {
                // Add bits to visualization
                for (let i = 0; i < pattern.length; i++) {
                    const bit = document.createElement('div');
                    bit.className = `pattern-bit ${pattern[i] === '1' ? 'up' : 'down'}`;
                    bit.textContent = pattern[i];
                    visualization.appendChild(bit);
                }
            } else {
                // Handle invalid pattern
                const errorBit = document.createElement('div');
                errorBit.textContent = 'Invalid pattern';
                errorBit.style.color = 'red';
                visualization.appendChild(errorBit);
            }
            
            // Add info about occurrences
            const info = document.createElement('div');
            info.className = 'pattern-info';
            info.textContent = `Occurrences: ${count}`;
            
            // Add components to card
            card.appendChild(visualization);
            card.appendChild(info);
            
            // Add card to container
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
        // Force chart resize to fix display issues
        setTimeout(() => {
            priceChart.resize();
            
            // If volume chart exists, resize it too
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
        // Force chart resize to fix display issues
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
