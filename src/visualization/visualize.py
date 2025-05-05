# visualize.py

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os
import logging
import sqlite3
from src.utils.config import DB_FILE, ENABLED_TIMEFRAMES

def get_symbol_data(symbol, timeframe):
    """Get data for a specific symbol and timeframe from the database."""
    conn = sqlite3.connect(DB_FILE)
    
    try:
        query = f"SELECT * FROM data_{timeframe} WHERE symbol = ? ORDER BY timestamp ASC"
        df = pd.read_sql_query(query, conn, params=(symbol,))
        
        if not df.empty:
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        conn.close()
        return df
    except Exception as e:
        logging.error(f"Error retrieving data for {symbol} ({timeframe}): {e}")
        conn.close()
        return pd.DataFrame()

def plot_ohlcv(symbol, timeframe, output_dir="plots"):
    """
    Create an interactive OHLCV candlestick chart with technical indicators.
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe for the chart
        output_dir: Directory to save the HTML file
    
    Returns:
        str: Path to the saved HTML file, or None if error
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get data
    df = get_symbol_data(symbol, timeframe)
    
    if df.empty:
        logging.error(f"No data found for {symbol} ({timeframe})")
        return None
    
    # Create subplots with 3 rows
    fig = make_subplots(
        rows=3, 
        cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f"{symbol} ({timeframe})", "Volume", "Indicators")
    )
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="OHLC"
        ),
        row=1, col=1
    )
    
    # Add EMAs
    for period, color in zip([5, 10, 20], ['blue', 'orange', 'purple']):
        ema_col = f'ema{period}'
        if ema_col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[ema_col],
                    name=f"EMA {period}",
                    line=dict(color=color)
                ),
                row=1, col=1
            )
    
    # Add Bollinger Bands
    if 'bollinger_hband' in df.columns and 'bollinger_lband' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['bollinger_hband'],
                name="BB Upper",
                line=dict(color='rgba(250, 0, 0, 0.7)', dash='dash')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['bollinger_lband'],
                name="BB Lower",
                line=dict(color='rgba(250, 0, 0, 0.7)', dash='dash'),
                fill='tonexty',
                fillcolor='rgba(250, 0, 0, 0.1)'
            ),
            row=1, col=1
        )
    
    # Add volume bar chart
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['volume'],
            name="Volume",
            marker=dict(color='rgba(0, 150, 0, 0.7)')
        ),
        row=2, col=1
    )
    
    # Add OBV
    if 'obv' in df.columns:
        # Normalize OBV for better visualization
        max_obv = max(abs(df['obv']))
        if max_obv > 0:
            normalized_obv = df['obv'] / max_obv * df['volume'].max() * 0.8
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=normalized_obv,
                    name="OBV (scaled)",
                    line=dict(color='rgba(255, 165, 0, 0.7)')
                ),
                row=2, col=1
            )
    
    # Add RSI
    if 'rsi_fast' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['rsi_fast'],
                name="RSI (14)",
                line=dict(color='blue')
            ),
            row=3, col=1
        )
        
        # Add RSI reference lines at 70 and 30
        fig.add_shape(
            type="line",
            x0=df.index[0],
            y0=70,
            x1=df.index[-1],
            y1=70,
            line=dict(color="red", width=1, dash="dash"),
            row=3, col=1
        )
        
        fig.add_shape(
            type="line",
            x0=df.index[0],
            y0=30,
            x1=df.index[-1],
            y1=30,
            line=dict(color="green", width=1, dash="dash"),
            row=3, col=1
        )
    
    # Add MACD
    if all(x in df.columns for x in ['macd', 'macd_signal']):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['macd'],
                name="MACD",
                line=dict(color='blue')
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['macd_signal'],
                name="MACD Signal",
                line=dict(color='red')
            ),
            row=3, col=1
        )
        
        # Calculate MACD histogram
        if 'macd_histogram' in df.columns:
            hist_colors = ['green' if val >= 0 else 'red' for val in df['macd_histogram']]
            
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['macd_histogram'],
                    name="MACD Histogram",
                    marker=dict(color=hist_colors),
                    opacity=0.5
                ),
                row=3, col=1
            )
    
    # Update layout
    fig.update_layout(
        title=f"{symbol} Chart - {timeframe} Timeframe",
        xaxis_title="Date",
        yaxis_title="Price",
        height=800,
        width=1200,
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    # Set y-axis for RSI
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
    
    # Save to HTML file
    output_file = os.path.join(output_dir, f"{symbol.replace('/', '_')}_{timeframe}.html")
    fig.write_html(output_file)
    
    logging.info(f"Chart saved to {output_file}")
    
    return output_file

def plot_multi_indicator_dashboard(symbol, timeframe, output_dir="plots"):
    """
    Create a comprehensive technical analysis dashboard with multiple indicators.
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe for the chart
        output_dir: Directory to save the HTML file
    
    Returns:
        str: Path to the saved HTML file, or None if error
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get data
    df = get_symbol_data(symbol, timeframe)
    
    if df.empty:
        logging.error(f"No data found for {symbol} ({timeframe})")
        return None
    
    # Create subplots with 4 rows
    fig = make_subplots(
        rows=4, 
        cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.2, 0.15, 0.15],
        subplot_titles=(
            f"{symbol} ({timeframe}) - Price Action",
            "Volume Analysis",
            "Momentum Indicators",
            "Trend Indicators"
        )
    )
    
    # Row 1: Price action with various overlays
    # OHLC Candlesticks
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="OHLC"
        ),
        row=1, col=1
    )
    
    # EMAs
    for period, color in zip([5, 10, 20], ['blue', 'orange', 'purple']):
        ema_col = f'ema{period}'
        if ema_col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[ema_col],
                    name=f"EMA {period}",
                    line=dict(color=color)
                ),
                row=1, col=1
            )
    
    # Bollinger Bands
    if 'bollinger_hband' in df.columns and 'bollinger_lband' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['bollinger_hband'],
                name="BB Upper",
                line=dict(color='rgba(250, 0, 0, 0.7)', dash='dash')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['bollinger_lband'],
                name="BB Lower",
                line=dict(color='rgba(250, 0, 0, 0.7)', dash='dash'),
                fill='tonexty',
                fillcolor='rgba(250, 0, 0, 0.1)'
            ),
            row=1, col=1
        )
    
    # Ichimoku Cloud (if available)
    if all(x in df.columns for x in ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b']):
        # Tenkan-sen (Conversion Line)
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['tenkan_sen'],
                name="Tenkan-sen",
                line=dict(color='red', width=1)
            ),
            row=1, col=1
        )
        
        # Kijun-sen (Base Line)
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['kijun_sen'],
                name="Kijun-sen",
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
        
        # Senkou Span A (Leading Span A) and Senkou Span B (Leading Span B) - forms the cloud
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['senkou_span_a'],
                name="Senkou Span A",
                line=dict(color='green', width=0.5)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['senkou_span_b'],
                name="Senkou Span B",
                line=dict(color='red', width=0.5),
                fill='tonexty',
                fillcolor='rgba(0, 250, 0, 0.1)'
            ),
            row=1, col=1
        )
    
    # Row 2: Volume analysis
    # Volume bars
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['volume'],
            name="Volume",
            marker=dict(color='rgba(0, 150, 0, 0.7)')
        ),
        row=2, col=1
    )
    
    # OBV
    if 'obv' in df.columns:
        # Calculate normalized OBV for better visualization
        max_obv = df['obv'].max() if df['obv'].max() > abs(df['obv'].min()) else abs(df['obv'].min())
        if max_obv > 0:
            normalized_obv = df['obv'] / max_obv * df['volume'].max() * 0.8
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=normalized_obv,
                    name="OBV (scaled)",
                    line=dict(color='rgba(255, 165, 0, 0.7)')
                ),
                row=2, col=1
            )
    
    # VWAP
    if 'vwap' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['vwap'],
                name="VWAP",
                line=dict(color='purple', dash='dash')
            ),
            row=1, col=1
        )
    
    # MFI (Money Flow Index)
    if 'mfi' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['mfi'],
                name="MFI",
                line=dict(color='purple')
            ),
            row=3, col=1
        )
        
        # Add MFI reference lines at 80 and 20
        fig.add_shape(
            type="line",
            x0=df.index[0],
            y0=80,
            x1=df.index[-1],
            y1=80,
            line=dict(color="red", width=1, dash="dash"),
            row=3, col=1
        )
        
        fig.add_shape(
            type="line",
            x0=df.index[0],
            y0=20,
            x1=df.index[-1],
            y1=20,
            line=dict(color="green", width=1, dash="dash"),
            row=3, col=1
        )
    
    # Row 3: Momentum indicators
    # RSI
    if 'rsi_fast' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['rsi_fast'],
                name="RSI (14)",
                line=dict(color='blue')
            ),
            row=3, col=1
        )
        
        # Add RSI reference lines at 70 and 30
        fig.add_shape(
            type="line",
            x0=df.index[0],
            y0=70,
            x1=df.index[-1],
            y1=70,
            line=dict(color="red", width=1, dash="dash"),
            row=3, col=1
        )
        
        fig.add_shape(
            type="line",
            x0=df.index[0],
            y0=30,
            x1=df.index[-1],
            y1=30,
            line=dict(color="green", width=1, dash="dash"),
            row=3, col=1
        )
    
    # Stochastic RSI
    if 'stoch_rsi' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['stoch_rsi'],
                name="Stoch RSI",
                line=dict(color='orange')
            ),
            row=3, col=1
        )
    
    # Williams %R
    if 'williams_r' in df.columns:
        # Williams %R is typically negative, so we add 100 to match RSI scale
        williams_r_adjusted = df['williams_r'] + 100
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=williams_r_adjusted,
                name="Williams %R",
                line=dict(color='green')
            ),
            row=3, col=1
        )
    
    # Row 4: Trend indicators
    # MACD
    if all(x in df.columns for x in ['macd', 'macd_signal']):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['macd'],
                name="MACD",
                line=dict(color='blue')
            ),
            row=4, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['macd_signal'],
                name="MACD Signal",
                line=dict(color='red')
            ),
            row=4, col=1
        )
        
        # Calculate MACD histogram colors
        if 'macd_histogram' in df.columns:
            hist_colors = ['green' if val >= 0 else 'red' for val in df['macd_histogram']]
            
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['macd_histogram'],
                    name="MACD Histogram",
                    marker=dict(color=hist_colors),
                    opacity=0.5
                ),
                row=4, col=1
            )
    
    # ADX
    if 'adx' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['adx'],
                name="ADX",
                line=dict(color='purple')
            ),
            row=4, col=1
        )
        
        # Add ADX reference line at 25 (strong trend)
        fig.add_shape(
            type="line",
            x0=df.index[0],
            y0=25,
            x1=df.index[-1],
            y1=25,
            line=dict(color="orange", width=1, dash="dash"),
            row=4, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=f"{symbol} Technical Analysis Dashboard - {timeframe} Timeframe",
        xaxis_title="Date",
        height=1000,
        width=1200,
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    # Set y-axis titles
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="Momentum", range=[0, 100], row=3, col=1)
    fig.update_yaxes(title_text="Trend", row=4, col=1)
    
    # Save to HTML file
    output_file = os.path.join(output_dir, f"{symbol.replace('/', '_')}_{timeframe}_dashboard.html")
    fig.write_html(output_file)
    
    logging.info(f"Dashboard saved to {output_file}")
    
    return output_file

def generate_all_charts(symbols, timeframes=None, output_dir="plots"):
    """
    Generate charts for multiple symbols and timeframes.
    
    Args:
        symbols: List of symbols
        timeframes: List of timeframes (if None, use all enabled timeframes)
        output_dir: Directory to save HTML files
    
    Returns:
        list: List of paths to saved HTML files
    """
    if timeframes is None:
        timeframes = ENABLED_TIMEFRAMES
    
    output_files = []
    
    for symbol in symbols:
        for timeframe in timeframes:
            # Generate basic OHLCV chart
            ohlcv_file = plot_ohlcv(symbol, timeframe, output_dir)
            if ohlcv_file:
                output_files.append(ohlcv_file)
            
            # Generate comprehensive dashboard
            dashboard_file = plot_multi_indicator_dashboard(symbol, timeframe, output_dir)
            if dashboard_file:
                output_files.append(dashboard_file)
    
    return output_files

def main():
    """Main function for visualizing cryptocurrency data."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize cryptocurrency data")
    parser.add_argument("--symbol", help="Specific symbol to visualize (e.g., BTC/USDT:USDT)")
    parser.add_argument("--timeframe", help="Specific timeframe to visualize")
    parser.add_argument("--output", default="plots", help="Output directory for charts")
    parser.add_argument("--dashboard", action="store_true", help="Generate comprehensive dashboard (default: basic OHLCV chart)")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    
    if not args.symbol:
        logging.error("Please provide a symbol with --symbol")
        return
    
    timeframe = args.timeframe if args.timeframe else ENABLED_TIMEFRAMES[0]
    
    if args.dashboard:
        output_file = plot_multi_indicator_dashboard(args.symbol, timeframe, args.output)
    else:
        output_file = plot_ohlcv(args.symbol, timeframe, args.output)
    
    if output_file:
        print(f"\nChart saved to: {output_file}")
        # Try to open the chart in the default browser
        try:
            import webbrowser
            webbrowser.open(f"file://{os.path.abspath(output_file)}")
        except Exception as e:
            logging.warning(f"Could not open chart in browser: {e}")

if __name__ == "__main__":
    main()
