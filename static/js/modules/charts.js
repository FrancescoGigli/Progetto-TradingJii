// charts.js - Gestisce tutti i grafici dell'applicazione
import { makeApiRequest } from './api.js';
import { appendToLog } from './ui.js';

// Variabili globali per i grafici
let positionChart = null;
let metricsChart = null;
let comparisonChart = null;

// Funzione per caricare i simboli per il grafico
export async function loadChartSymbols() {
    const symbolSelect = document.getElementById('chart-symbol-select');
    const timeframeSelect = document.getElementById('chart-timeframe-select');
    
    if (!symbolSelect || !timeframeSelect) return;
    
    try {
        // Ottieni le posizioni aperte per popolare il selettore
        const positions = await makeApiRequest('/orders/open');
        
        // Pulisci il selettore dei simboli e aggiungi l'opzione predefinita
        symbolSelect.innerHTML = '<option value="">Seleziona simbolo</option>';
        
        // Set per evitare duplicati
        const uniqueSymbols = new Set();
        
        // Aggiungi simboli delle posizioni aperte
        if (positions && positions.length > 0) {
            positions.forEach(position => {
                if (position.symbol && !uniqueSymbols.has(position.symbol)) {
                    uniqueSymbols.add(position.symbol);
                    const option = document.createElement('option');
                    option.value = position.symbol;
                    option.textContent = position.symbol;
                    symbolSelect.appendChild(option);
                }
            });
        }
        
        // Usa timeframe predefiniti invece di chiamare l'API
        const defaultTimeframes = [
            { value: '15m', text: '15 minuti' },
            { value: '30m', text: '30 minuti' },
            { value: '1h', text: '1 ora' }
        ];
        
        // Pulisci il selettore dei timeframe
        timeframeSelect.innerHTML = '';
        
        // Aggiungi i timeframe predefiniti
        defaultTimeframes.forEach(tf => {
            const option = document.createElement('option');
            option.value = tf.value;
            option.textContent = tf.text;
            timeframeSelect.appendChild(option);
        });
        
        // Event listener per il cambio di simbolo
        symbolSelect.addEventListener('change', (e) => {
            const selectedSymbol = e.target.value;
            if (selectedSymbol) {
                const selectedTimeframe = timeframeSelect.value;
                loadChartData(selectedSymbol, selectedTimeframe);
            } else {
                // Se nessun simbolo è selezionato, puliamo il grafico
                clearChart();
            }
        });
        
        // Event listener per il cambio di timeframe
        timeframeSelect.addEventListener('change', (e) => {
            const selectedTimeframe = e.target.value;
            const selectedSymbol = symbolSelect.value;
            if (selectedSymbol) {
                loadChartData(selectedSymbol, selectedTimeframe);
            }
        });
        
        // Se ci sono simboli, carichiamo automaticamente il primo
        if (uniqueSymbols.size > 0) {
            const firstSymbol = [...uniqueSymbols][0];
            symbolSelect.value = firstSymbol;
            loadChartData(firstSymbol, timeframeSelect.value);
        }
    } catch (error) {
        console.error('Errore nel caricamento dei simboli:', error);
        appendToLog(`Errore nel caricamento dei simboli: ${error.message}`);
    }
}

// Funzione per caricare i dati del grafico
export async function loadChartData(symbol, timeframe = '15m', limit = 100) {
    try {
        appendToLog(`Caricamento grafico per ${symbol} (${timeframe})...`);
        
        // Codifica il simbolo per l'URL
        const encodedSymbol = encodeURIComponent(symbol);
        
        const chartData = await makeApiRequest(`/chart-data/${encodedSymbol}?timeframe=${timeframe}&limit=${limit}`);
        
        if (chartData && chartData.labels && chartData.open) {
            // Crea o aggiorna il grafico
            createOrUpdateChart(chartData, symbol, timeframe);
        }
    } catch (error) {
        console.error('Errore nel caricamento dei dati del grafico:', error);
        appendToLog(`Errore nel caricamento del grafico: ${error.message}`);
    }
}

// Funzione per creare o aggiornare il grafico
function createOrUpdateChart(data, symbol, timeframe) {
    const ctx = document.getElementById('position-chart').getContext('2d');
    
    // Se esiste già un grafico, distruggilo
    if (positionChart) {
        positionChart.destroy();
    }
    
    // Prepara i dati per il grafico a candele
    const candleData = [];
    const volumeData = [];
    
    for (let i = 0; i < data.timestamps.length; i++) {
        candleData.push({
            x: data.timestamps[i],
            o: data.open[i],
            h: data.high[i],
            l: data.low[i],
            c: data.close[i]
        });
        
        // Prepara i dati dei volumi
        const isGreen = data.close[i] >= data.open[i];
        volumeData.push({
            x: data.timestamps[i],
            y: data.volumes[i],
            color: isGreen ? 'rgba(75, 192, 192, 0.5)' : 'rgba(255, 99, 132, 0.5)'
        });
    }
    
    // Calcola min e max per la scala del grafico
    const minPrice = Math.min(...data.low);
    const maxPrice = Math.max(...data.high);
    const padding = (maxPrice - minPrice) * 0.1;
    
    // Calcola max volume per la scala
    const maxVolume = Math.max(...data.volumes);
    
    // Crea un nuovo grafico a candele con volumi
    positionChart = new Chart(ctx, {
        data: {
            datasets: [
                {
                    type: 'candlestick',
                    label: `${symbol} (${timeframe})`,
                    data: candleData,
                    color: {
                        up: 'rgba(75, 192, 192, 1)',
                        down: 'rgba(255, 99, 132, 1)',
                        unchanged: 'rgba(110, 110, 110, 1)',
                    },
                    yAxisID: 'y-price'
                },
                {
                    type: 'bar',
                    label: 'Volume',
                    data: volumeData,
                    backgroundColor: volumeData.map(v => v.color),
                    yAxisID: 'y-volume'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function(context) {
                            const dataset = context.dataset;
                            if (dataset.type === 'bar') {
                                return `Volume: ${parseFloat(context.raw.y).toLocaleString()}`;
                            } else {
                                const point = context.raw;
                                return [
                                    `Apertura: ${point.o.toFixed(4)}`,
                                    `Massimo: ${point.h.toFixed(4)}`,
                                    `Minimo: ${point.l.toFixed(4)}`,
                                    `Chiusura: ${point.c.toFixed(4)}`
                                ];
                            }
                        }
                    }
                },
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: `${symbol} - Timeframe: ${timeframe}`
                }
            },
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'minute',
                        displayFormats: {
                            minute: 'HH:mm',
                            hour: 'dd HH:mm',
                            day: 'MMM dd'
                        },
                        tooltipFormat: 'dd MMM yyyy HH:mm'
                    },
                    ticks: {
                        maxRotation: 0,
                        autoSkip: true,
                        maxTicksLimit: 10
                    }
                },
                'y-price': {
                    position: 'right',
                    beginAtZero: false,
                    min: minPrice - padding,
                    max: maxPrice + padding,
                    ticks: {
                        callback: function(value) {
                            return value.toFixed(4);
                        }
                    }
                },
                'y-volume': {
                    position: 'left',
                    display: true,
                    beginAtZero: true,
                    max: maxVolume * 3,
                    grid: {
                        drawOnChartArea: false
                    },
                    ticks: {
                        callback: function(value) {
                            if (value >= 1000000) return (value / 1000000).toFixed(1) + 'M';
                            if (value >= 1000) return (value / 1000).toFixed(1) + 'K';
                            return value;
                        }
                    }
                }
            },
            interaction: {
                mode: 'index',
                intersect: false
            }
        }
    });
    
    appendToLog(`Grafico a candele con volumi per ${symbol} (${timeframe}) aggiornato.`);
}

// Funzione per pulire il grafico
export function clearChart() {
    if (positionChart) {
        positionChart.destroy();
        positionChart = null;
    }
}

// Funzione per creare il grafico comparativo dei modelli
export function createComparisonChart(metricsArray, timeframe) {
    // Cancella il grafico esistente se presente
    if (comparisonChart) {
        comparisonChart.destroy();
        comparisonChart = null;
    }
    
    const ctx = document.getElementById('comparison-chart');
    if (!ctx) return;
    
    const ctxContext = ctx.getContext('2d');
    
    // Prepara i dati per il grafico
    const labels = ['Accuratezza', 'Precisione', 'Richiamo', 'F1-Score'];
    const datasets = metricsArray.map((metrics, index) => {
        // Colori per ogni modello
        const colors = [
            'rgba(54, 162, 235, 0.7)',  // LSTM - blu
            'rgba(75, 192, 192, 0.7)',  // RF - verde
            'rgba(255, 159, 64, 0.7)'   // XGB - arancione
        ];
        
        // Estrai i valori delle metriche
        return {
            label: metrics.model,
            data: [
                metrics.accuracy !== undefined ? metrics.accuracy * 100 : 0,
                metrics.precision !== undefined ? metrics.precision * 100 : 0,
                metrics.recall !== undefined ? metrics.recall * 100 : 0,
                metrics.f1_score !== undefined ? metrics.f1_score * 100 : 0
            ],
            backgroundColor: colors[index % colors.length],
            borderColor: colors[index % colors.length].replace('0.7', '1'),
            borderWidth: 1
        };
    });
    
    // Crea il grafico a barre
    comparisonChart = new Chart(ctxContext, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: `Confronto Modelli - Timeframe ${timeframe}`
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: ${context.raw.toFixed(2)}%`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Percentuale'
                    },
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                }
            }
        }
    });
    
    return comparisonChart;
}

// Esporta le variabili e le funzioni
export {
    positionChart,
    metricsChart,
    comparisonChart
}; 