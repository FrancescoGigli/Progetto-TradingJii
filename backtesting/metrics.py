"""
Performance Metrics Module for Backtesting

Calculates various performance metrics for trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime


class PerformanceMetrics:
    """
    Calculate and store performance metrics for backtesting results
    """
    
    def __init__(self, trades: List[Dict], equity_curve: pd.Series, initial_capital: float):
        """
        Initialize with trade history and equity curve
        
        Args:
            trades: List of completed trades
            equity_curve: Pandas Series with equity values over time
            initial_capital: Starting capital
        """
        self.trades = trades
        self.equity_curve = equity_curve
        self.initial_capital = initial_capital
        self.metrics = {}
        
    def calculate_all_metrics(self) -> Dict[str, float]:
        """
        Calculate all performance metrics
        
        Returns:
            Dictionary with all calculated metrics
        """
        self.metrics = {
            # Basic metrics
            'total_trades': self._total_trades(),
            'winning_trades': self._winning_trades(),
            'losing_trades': self._losing_trades(),
            'win_rate': self._win_rate(),
            
            # Return metrics
            'total_return': self._total_return(),
            'total_return_pct': self._total_return_pct(),
            'annualized_return': self._annualized_return(),
            
            # Risk metrics
            'max_drawdown': self._max_drawdown(),
            'max_drawdown_pct': self._max_drawdown_pct(),
            'sharpe_ratio': self._sharpe_ratio(),
            'sortino_ratio': self._sortino_ratio(),
            
            # Trade statistics
            'avg_win': self._average_win(),
            'avg_loss': self._average_loss(),
            'avg_trade': self._average_trade(),
            'profit_factor': self._profit_factor(),
            'expectancy': self._expectancy(),
            
            # Time metrics
            'avg_trade_duration': self._avg_trade_duration(),
            'max_consecutive_wins': self._max_consecutive_wins(),
            'max_consecutive_losses': self._max_consecutive_losses(),
            
            # Risk-adjusted metrics
            'calmar_ratio': self._calmar_ratio(),
            'recovery_factor': self._recovery_factor(),
        }
        
        return self.metrics
    
    def _total_trades(self) -> int:
        """Total number of trades"""
        return len(self.trades)
    
    def _winning_trades(self) -> int:
        """Number of winning trades"""
        return len([t for t in self.trades if t['pnl'] > 0])
    
    def _losing_trades(self) -> int:
        """Number of losing trades"""
        return len([t for t in self.trades if t['pnl'] < 0])
    
    def _win_rate(self) -> float:
        """Percentage of winning trades"""
        if self._total_trades() == 0:
            return 0.0
        return (self._winning_trades() / self._total_trades()) * 100
    
    def _total_return(self) -> float:
        """Total return in currency"""
        if len(self.equity_curve) == 0:
            return 0.0
        return self.equity_curve.iloc[-1] - self.initial_capital
    
    def _total_return_pct(self) -> float:
        """Total return as percentage"""
        return (self._total_return() / self.initial_capital) * 100
    
    def _annualized_return(self) -> float:
        """Annualized return percentage"""
        if len(self.equity_curve) < 2:
            return 0.0
        
        # Calculate time period in years
        days = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days
        years = days / 365.25
        
        if years == 0:
            return 0.0
        
        # Calculate annualized return
        final_value = self.equity_curve.iloc[-1]
        annual_return = ((final_value / self.initial_capital) ** (1 / years) - 1) * 100
        
        return annual_return
    
    def _max_drawdown(self) -> float:
        """Maximum drawdown in currency"""
        if len(self.equity_curve) == 0:
            return 0.0
        
        # Calculate running maximum
        running_max = self.equity_curve.expanding().max()
        drawdown = self.equity_curve - running_max
        
        return abs(drawdown.min())
    
    def _max_drawdown_pct(self) -> float:
        """Maximum drawdown as percentage"""
        if len(self.equity_curve) == 0:
            return 0.0
        
        # Calculate running maximum
        running_max = self.equity_curve.expanding().max()
        drawdown_pct = ((self.equity_curve - running_max) / running_max) * 100
        
        return abs(drawdown_pct.min())
    
    def _sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio (assuming annual risk-free rate)
        
        Args:
            risk_free_rate: Annual risk-free rate (default 2%)
        """
        if len(self.equity_curve) < 2:
            return 0.0
        
        # Calculate daily returns
        returns = self.equity_curve.pct_change().dropna()
        
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        # Annualize returns and volatility
        annual_return = returns.mean() * 252
        annual_vol = returns.std() * np.sqrt(252)
        
        # Calculate Sharpe ratio
        sharpe = (annual_return - risk_free_rate) / annual_vol
        
        return sharpe
    
    def _sortino_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sortino ratio (using downside deviation)
        
        Args:
            risk_free_rate: Annual risk-free rate (default 2%)
        """
        if len(self.equity_curve) < 2:
            return 0.0
        
        # Calculate daily returns
        returns = self.equity_curve.pct_change().dropna()
        
        # Calculate downside returns only
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')  # No downside risk
        
        # Annualize returns and downside deviation
        annual_return = returns.mean() * 252
        downside_dev = downside_returns.std() * np.sqrt(252)
        
        if downside_dev == 0:
            return float('inf')
        
        # Calculate Sortino ratio
        sortino = (annual_return - risk_free_rate) / downside_dev
        
        return sortino
    
    def _average_win(self) -> float:
        """Average profit of winning trades"""
        wins = [t['pnl'] for t in self.trades if t['pnl'] > 0]
        return np.mean(wins) if wins else 0.0
    
    def _average_loss(self) -> float:
        """Average loss of losing trades"""
        losses = [t['pnl'] for t in self.trades if t['pnl'] < 0]
        return np.mean(losses) if losses else 0.0
    
    def _average_trade(self) -> float:
        """Average P&L per trade"""
        if not self.trades:
            return 0.0
        return np.mean([t['pnl'] for t in self.trades])
    
    def _profit_factor(self) -> float:
        """Ratio of gross profits to gross losses"""
        gross_profits = sum([t['pnl'] for t in self.trades if t['pnl'] > 0])
        gross_losses = abs(sum([t['pnl'] for t in self.trades if t['pnl'] < 0]))
        
        if gross_losses == 0:
            return float('inf') if gross_profits > 0 else 0.0
        
        return gross_profits / gross_losses
    
    def _expectancy(self) -> float:
        """Mathematical expectancy per trade"""
        if not self.trades:
            return 0.0
        
        win_rate = self._win_rate() / 100
        avg_win = self._average_win()
        avg_loss = abs(self._average_loss())
        
        return (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    
    def _avg_trade_duration(self) -> float:
        """Average duration of trades in hours"""
        if not self.trades:
            return 0.0
        
        durations = []
        for trade in self.trades:
            if 'entry_time' in trade and 'exit_time' in trade:
                duration = (trade['exit_time'] - trade['entry_time']).total_seconds() / 3600
                durations.append(duration)
        
        return np.mean(durations) if durations else 0.0
    
    def _max_consecutive_wins(self) -> int:
        """Maximum number of consecutive winning trades"""
        if not self.trades:
            return 0
        
        max_streak = 0
        current_streak = 0
        
        for trade in self.trades:
            if trade['pnl'] > 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    def _max_consecutive_losses(self) -> int:
        """Maximum number of consecutive losing trades"""
        if not self.trades:
            return 0
        
        max_streak = 0
        current_streak = 0
        
        for trade in self.trades:
            if trade['pnl'] < 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    def _calmar_ratio(self) -> float:
        """Calmar ratio (annual return / max drawdown)"""
        max_dd = self._max_drawdown_pct()
        if max_dd == 0:
            return float('inf')
        
        return self._annualized_return() / abs(max_dd)
    
    def _recovery_factor(self) -> float:
        """Recovery factor (total return / max drawdown)"""
        max_dd = self._max_drawdown()
        if max_dd == 0:
            return float('inf')
        
        return self._total_return() / max_dd
    
    def get_summary_string(self) -> str:
        """
        Get a formatted string summary of all metrics
        
        Returns:
            Formatted string with all metrics
        """
        if not self.metrics:
            self.calculate_all_metrics()
        
        summary = "=== PERFORMANCE METRICS ===\n\n"
        
        # General Statistics
        summary += "GENERAL STATISTICS:\n"
        summary += f"Total Trades: {self.metrics['total_trades']}\n"
        summary += f"Winning Trades: {self.metrics['winning_trades']}\n"
        summary += f"Losing Trades: {self.metrics['losing_trades']}\n"
        summary += f"Win Rate: {self.metrics['win_rate']:.2f}%\n"
        summary += f"Avg Trade Duration: {self.metrics['avg_trade_duration']:.1f} hours\n\n"
        
        # Returns
        summary += "RETURNS:\n"
        summary += f"Total Return: ${self.metrics['total_return']:.2f}\n"
        summary += f"Total Return %: {self.metrics['total_return_pct']:.2f}%\n"
        summary += f"Annualized Return: {self.metrics['annualized_return']:.2f}%\n\n"
        
        # Risk Metrics
        summary += "RISK METRICS:\n"
        summary += f"Max Drawdown: ${self.metrics['max_drawdown']:.2f}\n"
        summary += f"Max Drawdown %: {self.metrics['max_drawdown_pct']:.2f}%\n"
        summary += f"Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}\n"
        summary += f"Sortino Ratio: {self.metrics['sortino_ratio']:.2f}\n"
        summary += f"Calmar Ratio: {self.metrics['calmar_ratio']:.2f}\n\n"
        
        # Trade Analysis
        summary += "TRADE ANALYSIS:\n"
        summary += f"Average Win: ${self.metrics['avg_win']:.2f}\n"
        summary += f"Average Loss: ${self.metrics['avg_loss']:.2f}\n"
        summary += f"Average Trade: ${self.metrics['avg_trade']:.2f}\n"
        summary += f"Profit Factor: {self.metrics['profit_factor']:.2f}\n"
        summary += f"Expectancy: ${self.metrics['expectancy']:.2f}\n"
        summary += f"Max Consecutive Wins: {self.metrics['max_consecutive_wins']}\n"
        summary += f"Max Consecutive Losses: {self.metrics['max_consecutive_losses']}\n"
        
        return summary
