"""
Interactive Backtest Viewer

A graphical interface to visualize backtest results with price charts,
indicators, and entry/exit points.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Assicura che matplotlib usi il backend corretto per GUI
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, List
import sqlite3
from datetime import datetime, timedelta


class BacktestViewer:
    """Interactive viewer for backtest results"""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("TradingJii - Backtest Viewer")
        self.root.geometry("1400x800")
        
        # Data storage
        self.df = None
        self.trades = None
        self.current_strategy = None
        self.current_symbol = None
        
        # Create GUI
        self._create_widgets()
        
    def _create_widgets(self):
        """Create the GUI widgets"""
        # Top frame for controls
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Symbol selection
        ttk.Label(control_frame, text="Symbol:").grid(row=0, column=0, padx=5)
        self.symbol_var = tk.StringVar(value="BTC/USDT:USDT")
        self.symbol_combo = ttk.Combobox(control_frame, textvariable=self.symbol_var,
                                        values=["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"],
                                        width=20)
        self.symbol_combo.grid(row=0, column=1, padx=5)
        
        # Strategy selection
        ttk.Label(control_frame, text="Strategy:").grid(row=0, column=2, padx=5)
        self.strategy_var = tk.StringVar(value="ema_crossover")
        self.strategy_combo = ttk.Combobox(control_frame, textvariable=self.strategy_var,
                                          values=["rsi_mean_reversion", "ema_crossover", 
                                                 "breakout_range", "bollinger_rebound",
                                                 "macd_histogram", "donchian_breakout"],
                                          width=25)
        self.strategy_combo.grid(row=0, column=3, padx=5)
        
        # Date range
        ttk.Label(control_frame, text="Start Date:").grid(row=0, column=4, padx=5)
        self.start_date_var = tk.StringVar(value="2024-01-01")
        ttk.Entry(control_frame, textvariable=self.start_date_var, width=12).grid(row=0, column=5, padx=5)
        
        ttk.Label(control_frame, text="End Date:").grid(row=0, column=6, padx=5)
        self.end_date_var = tk.StringVar(value="2024-12-31")
        ttk.Entry(control_frame, textvariable=self.end_date_var, width=12).grid(row=0, column=7, padx=5)
        
        # Load button
        self.load_btn = ttk.Button(control_frame, text="Load & Run Backtest", 
                                  command=self.load_and_run_backtest)
        self.load_btn.grid(row=0, column=8, padx=20)
        
        # Chart frame
        self.chart_frame = ttk.Frame(self.root)
        self.chart_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Info frame at bottom
        self.info_frame = ttk.Frame(self.root, padding="10")
        self.info_frame.grid(row=2, column=0, sticky=(tk.W, tk.E))
        
        # Info labels
        self.info_label = ttk.Label(self.info_frame, text="Load data to see backtest results", 
                                   font=('Arial', 10))
        self.info_label.pack(side=tk.LEFT, padx=10)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        
    def load_and_run_backtest(self):
        """Load data and run backtest"""
        try:
            # Update UI
            self.load_btn.config(state='disabled')
            self.info_label.config(text="Loading data...")
            self.root.update()
            
            # Load data
            symbol = self.symbol_var.get()
            strategy = self.strategy_var.get()
            start_date = self.start_date_var.get()
            end_date = self.end_date_var.get()
            
            # Load market data
            self.df = self._load_market_data(symbol, start_date, end_date)
            
            # Run strategy to get signals
            self.info_label.config(text="Running backtest...")
            self.root.update()
            
            # Import and run strategy
            strategy_module = __import__(f'strategies.{strategy}', fromlist=['generate_signals'])
            self.df = strategy_module.generate_signals(self.df)
            
            # Run backtest
            from backtesting.backtest_engine import BacktestEngine
            
            config = {
                'initial_capital': 10000,
                'commission': 0.001,
                'slippage': 0.0005,
                'risk_config': {
                    'position_size_pct': 0.02,
                    'stop_loss_pct': 0.02,
                    'take_profit_pct': 0.04,
                    'max_positions': 1
                }
            }
            
            engine = BacktestEngine(config)
            results = engine.run_backtest(self.df.copy(), strategy_module.generate_signals)
            
            self.trades = results['trades']
            self.current_strategy = strategy
            self.current_symbol = symbol
            
            # Update chart
            self._update_chart()
            
            # Update info
            metrics = results['metrics']
            info_text = (f"Total Return: ${metrics['total_return']:.2f} ({metrics['total_return_pct']:.2f}%) | "
                        f"Trades: {metrics['total_trades']} | "
                        f"Win Rate: {metrics['win_rate']:.2f}% | "
                        f"Sharpe: {metrics['sharpe_ratio']:.2f} | "
                        f"Max DD: {metrics['max_drawdown_pct']:.2f}%")
            self.info_label.config(text=info_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data or run backtest:\n{str(e)}")
        finally:
            self.load_btn.config(state='normal')
    
    def _load_market_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Load market data from database"""
        conn = sqlite3.connect("crypto_data.db")
        
        query = """
        SELECT * FROM market_data_1h
        WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
        ORDER BY timestamp
        """
        
        df = pd.read_sql_query(query, conn, params=(symbol, start_date, end_date))
        conn.close()
        
        if df.empty:
            raise ValueError(f"No data found for {symbol} in the specified date range")
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def _update_chart(self):
        """Update the chart with backtest results"""
        # Clear previous chart
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
        
        # Create figure with subplots
        fig = plt.figure(figsize=(14, 8))
        
        # Main price chart (70% height)
        ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
        
        # Indicator chart (30% height)
        ax2 = plt.subplot2grid((4, 1), (3, 0), rowspan=1, sharex=ax1)
        
        # Plot based on strategy
        self._plot_strategy_chart(ax1, ax2)
        
        # Add trade markers
        self._add_trade_markers(ax1)
        
        # Styling
        ax1.set_ylabel('Price ($)', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f'{self.current_symbol} - {self.current_strategy.replace("_", " ").title()}', 
                     fontsize=12, fontweight='bold')
        
        ax2.set_xlabel('Date', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        toolbar = NavigationToolbar2Tk(canvas, self.chart_frame)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _plot_strategy_chart(self, ax1, ax2):
        """Plot chart based on selected strategy"""
        df = self.df.set_index('timestamp')
        
        # Plot candlestick chart
        self._plot_candlesticks(ax1, df)
        
        # Plot strategy-specific indicators
        if self.current_strategy == 'ema_crossover':
            # Plot EMAs on price chart
            ax1.plot(df.index, df['ema20'], 'b-', label='EMA 20', linewidth=1.5)
            ax1.plot(df.index, df['ema50'], 'r-', label='EMA 50', linewidth=1.5)
            ax1.legend(loc='upper left')
        
        elif self.current_strategy == 'rsi_mean_reversion':
            # Plot RSI
            ax2.plot(df.index, df['rsi14'], 'purple', label='RSI', linewidth=1.5)
            ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
            ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
            ax2.set_ylabel('RSI')
            ax2.set_ylim(0, 100)
            ax2.legend(loc='upper left')
        
        elif self.current_strategy == 'bollinger_rebound':
            # Plot Bollinger Bands
            ax1.plot(df.index, df['bb_upper'], 'gray', linestyle='--', label='BB Upper', alpha=0.7)
            ax1.plot(df.index, df['bb_lower'], 'gray', linestyle='--', label='BB Lower', alpha=0.7)
            ax1.fill_between(df.index, df['bb_lower'], df['bb_upper'], alpha=0.1, color='gray')
            ax1.legend(loc='upper left')
        
        elif self.current_strategy == 'macd_histogram':
            # Plot MACD
            ax2.plot(df.index, df['macd'], 'b-', label='MACD', linewidth=1.5)
            ax2.plot(df.index, df['macdsignal'], 'r-', label='Signal', linewidth=1.5)
            ax2.bar(df.index, df['macdhist'], label='Histogram', alpha=0.3)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax2.set_ylabel('MACD')
            ax2.legend(loc='upper left')
        
        elif self.current_strategy in ['breakout_range', 'donchian_breakout']:
            # Calculate and plot 20-period high/low
            high_20 = df['high'].rolling(20).max()
            low_20 = df['low'].rolling(20).min()
            ax1.plot(df.index, high_20, 'r--', label='20-High', alpha=0.7)
            ax1.plot(df.index, low_20, 'g--', label='20-Low', alpha=0.7)
            ax1.legend(loc='upper left')
    
    def _plot_candlesticks(self, ax, df):
        """Plot candlestick chart"""
        # Create candlestick data
        for idx, (timestamp, row) in enumerate(df.iterrows()):
            color = 'g' if row['close'] >= row['open'] else 'r'
            
            # High-Low line
            ax.plot([idx, idx], [row['low'], row['high']], color=color, linewidth=0.5)
            
            # Open-Close box
            height = abs(row['close'] - row['open'])
            bottom = min(row['open'], row['close'])
            rect = Rectangle((idx - 0.3, bottom), 0.6, height, 
                           facecolor=color, edgecolor=color, alpha=0.8)
            ax.add_patch(rect)
        
        # Set x-axis labels
        step = max(1, len(df) // 10)
        ax.set_xticks(range(0, len(df), step))
        ax.set_xticklabels([df.index[i].strftime('%Y-%m-%d') for i in range(0, len(df), step)], 
                          rotation=45, ha='right')
    
    def _add_trade_markers(self, ax):
        """Add entry and exit markers for trades"""
        if not self.trades:
            return
        
        df = self.df.set_index('timestamp')
        
        for trade in self.trades:
            # Find indices for entry and exit times
            # Use searchsorted for finding nearest index
            entry_idx = df.index.searchsorted(trade['entry_time'])
            if entry_idx >= len(df.index):
                entry_idx = len(df.index) - 1
            elif entry_idx > 0:
                # Check which is closer: entry_idx or entry_idx-1
                if abs(df.index[entry_idx] - trade['entry_time']) > abs(df.index[entry_idx-1] - trade['entry_time']):
                    entry_idx = entry_idx - 1
            
            exit_idx = df.index.searchsorted(trade['exit_time'])
            if exit_idx >= len(df.index):
                exit_idx = len(df.index) - 1
            elif exit_idx > 0:
                # Check which is closer: exit_idx or exit_idx-1
                if abs(df.index[exit_idx] - trade['exit_time']) > abs(df.index[exit_idx-1] - trade['exit_time']):
                    exit_idx = exit_idx - 1
            
            # Entry marker
            if trade['signal'] == 1:  # Long
                ax.scatter(entry_idx, trade['entry_price'], marker='^', 
                          color='green', s=100, zorder=5, label='Long Entry' if trade == self.trades[0] else "")
            else:  # Short
                ax.scatter(entry_idx, trade['entry_price'], marker='v', 
                          color='red', s=100, zorder=5, label='Short Entry' if trade == self.trades[0] else "")
            
            # Exit marker
            exit_color = 'darkgreen' if trade['pnl'] > 0 else 'darkred'
            ax.scatter(exit_idx, trade['exit_price'], marker='x', 
                      color=exit_color, s=100, zorder=5, 
                      label='Profitable Exit' if trade == self.trades[0] and trade['pnl'] > 0 else
                            'Loss Exit' if trade == self.trades[0] and trade['pnl'] <= 0 else "")
            
            # Connect entry and exit with a line
            ax.plot([entry_idx, exit_idx], [trade['entry_price'], trade['exit_price']], 
                   color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        # Add legend for trade markers
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left')


def main():
    """Main function to run the viewer"""
    root = tk.Tk()
    
    # Assicura che la finestra appaia in primo piano
    root.lift()
    root.attributes('-topmost', True)
    root.after_idle(root.attributes, '-topmost', False)
    
    app = BacktestViewer(root)
    
    # Centra la finestra sullo schermo
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    # Forza il focus sulla finestra
    root.focus_force()
    
    root.mainloop()


if __name__ == "__main__":
    main()
