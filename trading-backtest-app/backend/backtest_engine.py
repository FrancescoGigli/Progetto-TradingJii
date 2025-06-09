"""
Optimized Backtest Engine for Web App
Supports leverage and configurable take profit
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Callable
from datetime import datetime


class BacktestEngine:
    """
    Simplified backtest engine optimized for web app
    - Fixed $1000 capital
    - Configurable leverage
    - Configurable take profit
    - All-in positions (100% of capital)
    """
    
    def __init__(self, initial_capital: float = 1000, leverage: int = 1, 
                 take_profit_pct: float = 0.02, stop_loss_pct: float = 0.02):
        """
        Initialize backtest engine
        
        Args:
            initial_capital: Starting capital (default $1000)
            leverage: Leverage multiplier (1, 2, 5, 10, 20)
            take_profit_pct: Take profit percentage (0.01 = 1%)
            stop_loss_pct: Stop loss percentage (0.01 = 1%)
        """
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.commission = 0.001  # 0.1% commission per trade
        
    def run_backtest(self, df: pd.DataFrame, strategy_name: str) -> Dict:
        """
        Run backtest for a strategy
        
        Args:
            df: DataFrame with market data
            strategy_name: Name of strategy to run
            
        Returns:
            Dictionary with results
        """
        # Import strategy function
        from strategies import STRATEGIES
        
        if strategy_name not in STRATEGIES:
            raise ValueError(f"Strategy {strategy_name} not found")
            
        strategy_func = STRATEGIES[strategy_name]
        
        # Prepare data
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        
        # Generate signals
        df = strategy_func(df)
        
        # Run simulation
        trades = []
        equity_curve = {}
        capital = self.initial_capital
        position = None
        
        for timestamp, row in df.iterrows():
            # Record equity
            equity_curve[timestamp] = capital
            
            # Check for exit conditions if in position
            if position is not None:
                # Calculate current P&L
                if position['signal'] == 1:  # Long
                    current_pnl_pct = (row['close'] - position['entry_price']) / position['entry_price']
                else:  # Short
                    current_pnl_pct = (position['entry_price'] - row['close']) / position['entry_price']
                
                # Apply leverage to P&L
                leveraged_pnl_pct = current_pnl_pct * self.leverage
                
                # Check take profit
                exit_trade = False
                exit_reason = None
                
                # Take profit is applied directly to the leveraged P&L
                if leveraged_pnl_pct >= self.take_profit_pct:
                    exit_trade = True
                    exit_reason = 'take_profit'
                # Stop loss is applied directly to the leveraged P&L
                # This ensures that higher leverages reach stop loss faster
                # For example, with 10x leverage and 10% stop loss:
                # The position will exit when the asset moves -1% (which is -10% leveraged)
                elif leveraged_pnl_pct <= -self.stop_loss_pct:
                    exit_trade = True
                    exit_reason = 'stop_loss'
                # Check for opposite signal
                elif pd.notna(row['signal']) and row['signal'] != 0 and row['signal'] != position['signal']:
                    exit_trade = True
                    exit_reason = 'signal_reversal'
                
                if exit_trade:
                    # Close position
                    exit_price = row['close']
                    
                    # Calculate final P&L
                    if position['signal'] == 1:  # Long
                        pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
                    else:  # Short
                        pnl_pct = (position['entry_price'] - exit_price) / position['entry_price']
                    
                    # Apply leverage and commission
                    leveraged_pnl_pct = pnl_pct * self.leverage
                    commission_cost = 2 * self.commission  # Entry + exit
                    final_pnl_pct = leveraged_pnl_pct - commission_cost
                    
                    # Update capital (all-in position)
                    pnl = capital * final_pnl_pct
                    capital += pnl
                    
                    # Record trade
                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': timestamp,
                        'signal': position['signal'],
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'pnl_pct': final_pnl_pct * 100,
                        'exit_reason': exit_reason
                    })
                    
                    position = None
            
            # Check for new entry signal
            if position is None and pd.notna(row['signal']) and row['signal'] != 0:
                # Enter new position
                position = {
                    'entry_time': timestamp,
                    'entry_price': row['close'],
                    'signal': row['signal']
                }
        
        # Close any remaining position at the end
        if position is not None:
            exit_price = df.iloc[-1]['close']
            
            if position['signal'] == 1:  # Long
                pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
            else:  # Short
                pnl_pct = (position['entry_price'] - exit_price) / position['entry_price']
            
            leveraged_pnl_pct = pnl_pct * self.leverage
            commission_cost = 2 * self.commission
            final_pnl_pct = leveraged_pnl_pct - commission_cost
            
            pnl = capital * final_pnl_pct
            capital += pnl
            
            trades.append({
                'entry_time': position['entry_time'],
                'exit_time': df.index[-1],
                'signal': position['signal'],
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'pnl': pnl,
                'pnl_pct': final_pnl_pct * 100,
                'exit_reason': 'end_of_period'
            })
        
        # Update final equity
        equity_curve[df.index[-1]] = capital
        
        # Calculate metrics
        metrics = self._calculate_metrics(trades, equity_curve)
        
        return {
            'trades': trades,
            'equity_curve': equity_curve,
            'metrics': metrics
        }
    
    def _calculate_metrics(self, trades: List[Dict], equity_curve: Dict) -> Dict:
        """Calculate performance metrics"""
        
        if not trades:
            return {
                'total_return': 0,
                'total_return_pct': 0,
                'total_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'max_drawdown_pct': 0
            }
        
        # Basic metrics
        df_trades = pd.DataFrame(trades)
        winning_trades = df_trades[df_trades['pnl'] > 0]
        losing_trades = df_trades[df_trades['pnl'] <= 0]
        
        total_return = list(equity_curve.values())[-1] - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100
        
        win_rate = (len(winning_trades) / len(trades)) * 100 if trades else 0
        
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        
        # Profit factor
        total_wins = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        total_losses = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Sharpe ratio
        equity_series = pd.Series(list(equity_curve.values()))
        returns = equity_series.pct_change().dropna()
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252 * 24) if returns.std() > 0 else 0
        
        # Drawdown
        running_max = equity_series.expanding().max()
        drawdown = equity_series - running_max
        max_drawdown = drawdown.min()
        max_drawdown_pct = (max_drawdown / running_max[drawdown.idxmin()]) * 100 if len(equity_series) > 0 else 0
        
        return {
            'total_return': round(total_return, 2),
            'total_return_pct': round(total_return_pct, 2),
            'total_trades': len(trades),
            'win_rate': round(win_rate, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'profit_factor': round(profit_factor, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'max_drawdown': round(max_drawdown, 2),
            'max_drawdown_pct': round(abs(max_drawdown_pct), 2)
        }
