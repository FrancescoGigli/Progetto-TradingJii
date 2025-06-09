"""
Visualization Module for Backtesting Results

Creates interactive charts and HTML reports for backtest results.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List
import os
from datetime import datetime


class BacktestVisualizer:
    """
    Create visualizations for backtest results
    """
    
    def __init__(self, results: Dict, strategy_name: str, symbol: str):
        """
        Initialize visualizer with backtest results
        
        Args:
            results: Dictionary containing backtest results
            strategy_name: Name of the strategy
            symbol: Trading symbol
        """
        self.results = results
        self.strategy_name = strategy_name
        self.symbol = symbol
        self.metrics = results['metrics']
        self.trades = results['trades']
        self.equity_curve = results['equity_curve']
        
    def create_full_report(self, output_dir: str = 'backtest_results') -> str:
        """
        Create a complete HTML report with all visualizations
        
        Args:
            output_dir: Directory to save the report
            
        Returns:
            Path to the generated HTML file
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create all charts
        equity_chart = self._create_equity_curve_chart()
        drawdown_chart = self._create_drawdown_chart()
        trade_dist_chart = self._create_trade_distribution()
        monthly_returns = self._create_monthly_returns_heatmap()
        
        # Create metrics table
        metrics_html = self._create_metrics_table()
        
        # Combine into HTML report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Clean symbol name for filename (replace invalid characters)
        clean_symbol = self.symbol.replace('/', '_').replace(':', '_')
        filename = f"{output_dir}/backtest_{self.strategy_name}_{clean_symbol}_{timestamp}.html"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtest Report - {self.strategy_name} on {self.symbol}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .header {{
                    text-align: center;
                    background-color: #2c3e50;
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 30px;
                }}
                .metrics-container {{
                    background-color: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin-bottom: 30px;
                }}
                .chart-container {{
                    background-color: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin-bottom: 30px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                }}
                th, td {{
                    padding: 10px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #34495e;
                    color: white;
                }}
                .positive {{
                    color: #27ae60;
                    font-weight: bold;
                }}
                .negative {{
                    color: #e74c3c;
                    font-weight: bold;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Backtest Report</h1>
                <h2>{self.strategy_name} on {self.symbol}</h2>
                <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
            
            <div class="metrics-container">
                <h2>Performance Metrics</h2>
                {metrics_html}
            </div>
            
            <div class="chart-container">
                <h2>Equity Curve</h2>
                <div id="equity-chart"></div>
            </div>
            
            <div class="chart-container">
                <h2>Drawdown Analysis</h2>
                <div id="drawdown-chart"></div>
            </div>
            
            <div class="chart-container">
                <h2>Trade Distribution</h2>
                <div id="trade-dist-chart"></div>
            </div>
            
            <div class="chart-container">
                <h2>Monthly Returns Heatmap</h2>
                <div id="monthly-returns-chart"></div>
            </div>
            
            <script>
                Plotly.newPlot('equity-chart', {equity_chart['data']}, {equity_chart['layout']});
                Plotly.newPlot('drawdown-chart', {drawdown_chart['data']}, {drawdown_chart['layout']});
                Plotly.newPlot('trade-dist-chart', {trade_dist_chart['data']}, {trade_dist_chart['layout']});
                Plotly.newPlot('monthly-returns-chart', {monthly_returns['data']}, {monthly_returns['layout']});
            </script>
        </body>
        </html>
        """
        
        with open(filename, 'w') as f:
            f.write(html_content)
            
        print(f"\nReport saved to: {filename}")
        return filename
    
    def _create_equity_curve_chart(self) -> Dict:
        """Create equity curve chart"""
        fig = go.Figure()
        
        # Add equity curve
        fig.add_trace(go.Scatter(
            x=self.equity_curve.index,
            y=self.equity_curve.values,
            mode='lines',
            name='Equity',
            line=dict(color='blue', width=2)
        ))
        
        # Add trade markers
        if self.trades:
            df_trades = pd.DataFrame(self.trades)
            
            # Long entries
            long_trades = df_trades[df_trades['signal'] == 1]
            if not long_trades.empty:
                fig.add_trace(go.Scatter(
                    x=long_trades['entry_time'],
                    y=[self.equity_curve.loc[self.equity_curve.index <= t].iloc[-1] 
                       for t in long_trades['entry_time']],
                    mode='markers',
                    name='Long Entry',
                    marker=dict(symbol='triangle-up', size=10, color='green')
                ))
            
            # Short entries
            short_trades = df_trades[df_trades['signal'] == -1]
            if not short_trades.empty:
                fig.add_trace(go.Scatter(
                    x=short_trades['entry_time'],
                    y=[self.equity_curve.loc[self.equity_curve.index <= t].iloc[-1] 
                       for t in short_trades['entry_time']],
                    mode='markers',
                    name='Short Entry',
                    marker=dict(symbol='triangle-down', size=10, color='red')
                ))
        
        fig.update_layout(
            title='Equity Curve',
            xaxis_title='Date',
            yaxis_title='Equity ($)',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return {'data': fig.data, 'layout': fig.layout}
    
    def _create_drawdown_chart(self) -> Dict:
        """Create drawdown chart"""
        # Calculate drawdown
        running_max = self.equity_curve.expanding().max()
        drawdown = (self.equity_curve - running_max) / running_max * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode='lines',
            fill='tozeroy',
            name='Drawdown %',
            line=dict(color='red', width=1),
            fillcolor='rgba(255, 0, 0, 0.2)'
        ))
        
        fig.update_layout(
            title='Drawdown Analysis',
            xaxis_title='Date',
            yaxis_title='Drawdown %',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return {'data': fig.data, 'layout': fig.layout}
    
    def _create_trade_distribution(self) -> Dict:
        """Create trade P&L distribution histogram"""
        if not self.trades:
            return {'data': [], 'layout': {}}
            
        df_trades = pd.DataFrame(self.trades)
        
        fig = go.Figure()
        
        # Create histogram
        fig.add_trace(go.Histogram(
            x=df_trades['pnl'],
            nbinsx=30,
            name='Trade P&L',
            marker_color=['red' if x < 0 else 'green' for x in df_trades['pnl']]
        ))
        
        # Add average line
        avg_pnl = df_trades['pnl'].mean()
        fig.add_vline(x=avg_pnl, line_dash="dash", line_color="blue",
                      annotation_text=f"Avg: ${avg_pnl:.2f}")
        
        fig.update_layout(
            title='Trade P&L Distribution',
            xaxis_title='P&L ($)',
            yaxis_title='Frequency',
            template='plotly_white'
        )
        
        return {'data': fig.data, 'layout': fig.layout}
    
    def _create_monthly_returns_heatmap(self) -> Dict:
        """Create monthly returns heatmap"""
        # Calculate daily returns
        returns = self.equity_curve.pct_change().fillna(0)
        
        # Convert to monthly returns
        monthly_returns = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1) * 100
        
        # Handle empty or single-value series
        if len(monthly_returns) == 0:
            return {'data': [], 'layout': {}}
        
        # Create pivot table for heatmap
        monthly_returns_df = pd.DataFrame({
            'Year': monthly_returns.index.year.values,
            'Month': monthly_returns.index.month.values,
            'Return': monthly_returns.values.flatten()
        })
        
        pivot_table = monthly_returns_df.pivot(index='Month', columns='Year', values='Return')
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_table.values,
            x=pivot_table.columns,
            y=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(pivot_table.index)],
            colorscale='RdYlGn',
            zmid=0,
            text=np.round(pivot_table.values, 2),
            texttemplate='%{text}%',
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title='Monthly Returns Heatmap (%)',
            xaxis_title='Year',
            yaxis_title='Month',
            template='plotly_white'
        )
        
        return {'data': fig.data, 'layout': fig.layout}
    
    def _create_metrics_table(self) -> str:
        """Create HTML table with performance metrics"""
        def format_value(key, value):
            if 'pct' in key or 'return' in key or 'rate' in key:
                return f"{value:.2f}%"
            elif 'ratio' in key or 'factor' in key:
                return f"{value:.2f}"
            elif isinstance(value, float):
                return f"${value:.2f}"
            else:
                return str(value)
        
        def get_class(key, value):
            positive_metrics = ['total_return', 'total_return_pct', 'win_rate', 
                              'sharpe_ratio', 'profit_factor', 'expectancy']
            if key in positive_metrics and value > 0:
                return 'positive'
            elif 'drawdown' in key or (key == 'avg_loss' and value < 0):
                return 'negative'
            return ''
        
        html = "<table>"
        html += "<tr><th>Metric</th><th>Value</th></tr>"
        
        # Organize metrics by category
        categories = {
            'Returns': ['total_return', 'total_return_pct', 'annualized_return'],
            'Risk': ['max_drawdown', 'max_drawdown_pct', 'sharpe_ratio', 'sortino_ratio'],
            'Trading': ['total_trades', 'win_rate', 'avg_win', 'avg_loss', 'profit_factor']
        }
        
        for category, keys in categories.items():
            html += f"<tr><td colspan='2' style='background-color: #ecf0f1; font-weight: bold;'>{category}</td></tr>"
            for key in keys:
                if key in self.metrics:
                    value = self.metrics[key]
                    formatted = format_value(key, value)
                    css_class = get_class(key, value)
                    html += f"<tr><td>{key.replace('_', ' ').title()}</td>"
                    html += f"<td class='{css_class}'>{formatted}</td></tr>"
        
        html += "</table>"
        return html
