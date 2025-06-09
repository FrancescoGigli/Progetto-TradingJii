"""
Main entry point for the TradingJii Interactive Backtest Viewer

A graphical application to visualize backtesting results with:
- Candlestick charts
- Technical indicators
- Entry/exit points
- Performance metrics

Usage:
    python backtest_viewer.py
"""

from backtesting.interactive_viewer import main

if __name__ == "__main__":
    print("Starting TradingJii Backtest Viewer...")
    main()
