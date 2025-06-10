"""
Strategy Indicators Mapping

This module defines which technical indicators are used by each strategy
and how they should be displayed on the chart.
"""

STRATEGY_INDICATORS = {
    'rsi_mean_reversion': {
        'name': 'RSI Mean Reversion',
        'indicators': {
            'rsi': {
                'type': 'oscillator',
                'field': 'rsi14',
                'panel': 'separate',  # Display in separate panel
                'color': '#ff9800',
                'style': 'line',
                'levels': [30, 70],  # Oversold and overbought levels
                'level_colors': ['#00ff88', '#ff3b3b']
            }
        }
    },
    
    'ema_crossover': {
        'name': 'EMA Crossover',
        'indicators': {
            'ema20': {
                'type': 'overlay',
                'field': 'ema20',
                'panel': 'main',  # Display on main chart
                'color': '#2196f3',
                'style': 'line',
                'lineWidth': 2
            },
            'ema50': {
                'type': 'overlay',
                'field': 'ema50',
                'panel': 'main',
                'color': '#ff5722',
                'style': 'line',
                'lineWidth': 2
            }
        }
    },
    
    'breakout_range': {
        'name': 'Breakout Range',
        'indicators': {
            'high_20': {
                'type': 'overlay',
                'field': 'highest_20',
                'panel': 'main',
                'color': '#00ff88',
                'style': 'line',
                'lineWidth': 1,
                'opacity': 0.7
            },
            'low_20': {
                'type': 'overlay', 
                'field': 'lowest_20',
                'panel': 'main',
                'color': '#ff3b3b',
                'style': 'line',
                'lineWidth': 1,
                'opacity': 0.7
            }
        }
    },
    
    'bollinger_rebound': {
        'name': 'Bollinger Rebound',
        'indicators': {
            'bb_upper': {
                'type': 'overlay',
                'field': 'bbands_upper',
                'panel': 'main',
                'color': '#9c27b0',
                'style': 'line',
                'lineWidth': 1,
                'opacity': 0.8
            },
            'bb_middle': {
                'type': 'overlay',
                'field': 'bbands_middle',
                'panel': 'main',
                'color': '#9c27b0',
                'style': 'line',
                'lineWidth': 1,
                'opacity': 0.5,
                'dashStyle': 'dash'
            },
            'bb_lower': {
                'type': 'overlay',
                'field': 'bbands_lower',
                'panel': 'main',
                'color': '#9c27b0',
                'style': 'line',
                'lineWidth': 1,
                'opacity': 0.8
            }
        }
    },
    
    'macd_histogram': {
        'name': 'MACD Histogram',
        'indicators': {
            'macd': {
                'type': 'oscillator',
                'field': 'macd',
                'panel': 'separate',
                'color': '#2196f3',
                'style': 'line',
                'lineWidth': 2
            },
            'macd_signal': {
                'type': 'oscillator',
                'field': 'macd_signal',
                'panel': 'separate',
                'color': '#ff5722',
                'style': 'line',
                'lineWidth': 2
            },
            'macd_histogram': {
                'type': 'oscillator',
                'field': 'macd_hist',
                'panel': 'separate',
                'color': '#4caf50',
                'style': 'histogram'
            }
        }
    },
    
    'donchian_breakout': {
        'name': 'Donchian Breakout',
        'indicators': {
            'donchian_upper': {
                'type': 'overlay',
                'field': 'donchian_upper',
                'panel': 'main',
                'color': '#00bcd4',
                'style': 'line',
                'lineWidth': 2
            },
            'donchian_middle': {
                'type': 'overlay',
                'field': 'donchian_middle',
                'panel': 'main',
                'color': '#00bcd4',
                'style': 'line',
                'lineWidth': 1,
                'opacity': 0.5,
                'dashStyle': 'dash'
            },
            'donchian_lower': {
                'type': 'overlay',
                'field': 'donchian_lower',
                'panel': 'main',
                'color': '#00bcd4',
                'style': 'line',
                'lineWidth': 2
            }
        }
    },
    
    'adx_filter_crossover': {
        'name': 'ADX Filtered Crossover',
        'indicators': {
            'ema20': {
                'type': 'overlay',
                'field': 'ema20',
                'panel': 'main',
                'color': '#2196f3',
                'style': 'line',
                'lineWidth': 2
            },
            'ema50': {
                'type': 'overlay',
                'field': 'ema50',
                'panel': 'main',
                'color': '#ff5722',
                'style': 'line',
                'lineWidth': 2
            },
            'adx': {
                'type': 'oscillator',
                'field': 'adx14',
                'panel': 'separate',
                'color': '#795548',
                'style': 'line',
                'lineWidth': 2,
                'levels': [20],  # ADX threshold for trending market
                'level_colors': ['#ffffff']
            }
        }
    }
}
