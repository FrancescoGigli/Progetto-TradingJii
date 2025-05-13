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
        console.log('Creating simplified chart for', symbol);
        
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
        
        console.log(`Chart data prepared, ${chartData.length} points available`);
        
        // Create a simpler line chart to avoid potential issues with candlestick rendering
        const prices = chartData.map(d => d.c); // Close prices
        const timestamps = chartData.map(d => d.x);
        
        // Define colors based on theme
        const isDarkTheme = !document.body.classList.contains('light-theme');
        const gridColor = isDarkTheme ? '#2a2e39' : '#e5e7eb';
        const textColor = isDarkTheme ? '#9ca3af' : '#666666';
        
        // Create a simple line chart first to get things working
        priceChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: timestamps,
                datasets: [{
                    label: `${symbol} Price (${timeframe})`,
                    data: prices,
                    fill: false,
                    borderColor: '#2962ff',
                    tension: 0.1,
                    borderWidth: 2,
                    pointRadius: 0,
                    pointHoverRadius: 5
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
                            },
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
                        intersect: false
                    },
                    legend: {
                        display: false
                    }
                },
                animation: false // Disable animation for better performance
            }
        });
        
        console.log('Chart created successfully');
        
        // Create a basic volume bar chart
        try {
            createSimpleVolumeChart(symbol, timeframe, chartData);
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
 * Create a simple volume bar chart
 */
function createSimpleVolumeChart(symbol, timeframe, chartData) {
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
    
    // Get the context (again, in case we just created it)
    const volumeCtx = document.getElementById('volume-chart').getContext('2d');
    
    // If volume chart already exists, destroy it
    if (volumeChart) {
        volumeChart.destroy();
    }
    
    // Very simple volume chart
    const volumes = chartData.map(d => d.volume);
    const timestamps = chartData.map(d => d.x);
    
    volumeChart = new Chart(volumeCtx, {
        type: 'bar',
        data: {
            labels: timestamps,
            datasets: [{
                label: `Volume`,
                data: volumes,
                backgroundColor: 'rgba(41, 98, 255, 0.5)'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true
                },
                x: {
                    display: false
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            },
            animation: false
        }
    });
    
    return volumeChart;
}

/**
 * Create a volume bar chart in TradingView style
 * @param {string} symbol - Cryptocurrency symbol
 * @param {string} timeframe - Timeframe like '5m', '15m'
 * @param {Array} chartData - OHLCV data from the price chart
 */
function createVolumeChart(symbol, timeframe, chartData) {
    // Get the canvas element for volume
    const volumeCtx = document.getElementById('volume-chart').getContext('2d');
    
    // Define colors
    const upColor = 'rgba(38, 166, 154, 0.6)';  // Semi-transparent green for up volumes
    const downColor = 'rgba(239, 83, 80, 0.6)'; // Semi-transparent red for down volumes
    
    // Define colors based on theme
    const isDarkTheme = !document.body.classList.contains('light-theme');
    
    // Set theme-specific colors
    const gridColor = isDarkTheme ? '#2a2e39' : '#e5e7eb';
    const textColor = isDarkTheme ? '#9ca3af' : '#666666';
    
    // Prepare volume data
    const volumeData = chartData.map(d => ({
        x: d.x,
        y: d.volume,
        isBullish: d.c >= d.o
    }));
    
    // Calculate max volume for y-axis scaling
    const maxVolume = Math.max(...volumeData.map(d => d.y));
    
    // Create the volume chart
    volumeChart = new Chart(volumeCtx, {
        type: 'bar',
        data: {
            datasets: [{
                label: 'Volume',
                data: volumeData.map(d => ({
                    x: d.x,
                    y: d.y
                })),
                backgroundColor: volumeData.map(d => d.isBullish ? upColor : downColor),
                borderColor: volumeData.map(d => d.isBullish ? upColor : downColor),
                borderWidth: 0,
                barPercentage: 0.9,
                categoryPercentage: 0.8,
                order: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index',
            },
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: timeframe === '5m' ? 'minute' : 'hour',
                        displayFormats: {
                            minute: 'HH:mm',
                            hour: 'DD HH:mm'
                        },
                    },
                    grid: {
                        display: false,
                        drawBorder: false
                    },
                    ticks: {
                        display: false
                    },
                    border: {
                        display: false
                    }
                },
                y: {
                    position: 'right',
                    max: maxVolume * 1.5,
                    grid: {
                        display: true,
                        color: gridColor,
                        drawBorder: false,
                        lineWidth: 0.5
                    },
                    ticks: {
                        color: textColor,
                        callback: function(value) {
                            if (value === 0) return '';
                            if (value >= 1000000) return (value / 1000000).toFixed(1) + 'M';
                            if (value >= 1000) return (value / 1000).toFixed(1) + 'K';
                            return value;
                        },
                        font: {
                            size: 9,
                            family: "'Inter', sans-serif"
                        },
                        maxTicksLimit: 2
                    },
                    border: {
                        color: gridColor
                    }
                }
            },
            plugins: {
                tooltip: {
                    enabled: true,
                    mode: 'index',
                    intersect: false,
                    backgroundColor: 'rgba(21, 25, 32, 0.9)',
                    titleColor: '#d1d4dc',
                    bodyColor: '#d1d4dc',
                    bodyFont: {
                        size: 11
                    },
                    titleFont: {
                        weight: 'normal',
                        size: 11
                    },
                    displayColors: false,
                    callbacks: {
                        label: function(context) {
                            return `Volume: ${context.raw.y.toFixed(2)}`;
                        }
                    }
                },
                legend: {
                    display: false
                }
            },
            animation: {
                duration: 400
            }
        }
    });
    
    return volumeChart;
}

/**
 * Helper function to determine appropriate decimal precision for price display
 * @param {number} price - The price value
 * @returns {number} - The number of decimal places to display
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
 * @param {string} symbol - Cryptocurrency symbol
 * @param {string} timeframe - Timeframe like '5m', '15m'
 * @param {Array} data - Volatility data from the API
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
                            },
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
                        intersect: false
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
 * @param {Array} data - OHLCV data from API
 * @returns {Array} - Formatted data for Chart.js
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
 * Render the pattern visualization
 * @param {string} symbol - Cryptocurrency symbol
 * @param {string} timeframe - Timeframe like '5m', '15m'
 * @param {Object} patterns - Pattern data from API
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
