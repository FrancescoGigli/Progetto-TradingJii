"""
Backtesting Engine Module

Core engine for simulating trades based on strategy signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from datetime import datetime
from colorama import Fore, Style
import logging

from .risk_manager import RiskManager
from .metrics import PerformanceMetrics


class Position:
    """Represents a trading position"""
    
    def __init__(self, symbol: str, signal: int, entry_price: float, 
                 entry_time: datetime, size: float, stop_loss: float, 
                 take_profit: float):
        self.symbol = symbol
        self.signal = signal  # 1 for long, -1 for short
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.size = size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.exit_price = None
        self.exit_time = None
        self.pnl = 0.0
        self.exit_reason = None
        
    def close(self, exit_price: float, exit_time: datetime, reason: str):
        """Close the position"""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_reason = reason
        
        # Calculate P&L
        if self.signal == 1:  # Long position
            self.pnl = (exit_price - self.entry_price) * self.size
        else:  # Short position
            self.pnl = (self.entry_price - exit_price) * self.size
            
    def to_dict(self) -> Dict:
        """Convert position to dictionary"""
        return {
            'symbol': self.symbol,
            'signal': self.signal,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time,
            'exit_price': self.exit_price,
            'exit_time': self.exit_time,
            'size': self.size,
            'pnl': self.pnl,
            'exit_reason': self.exit_reason,
            'return_pct': ((self.exit_price / self.entry_price - 1) * self.signal * 100) if self.exit_price else 0
        }


class BacktestEngine:
    """
    Main backtesting engine that simulates trading based on signals
    """
    
    def __init__(self, config: Dict):
        """
        Initialize backtesting engine
        
        Args:
            config: Configuration dictionary containing:
                - initial_capital: Starting capital
                - commission: Commission per trade (as decimal, e.g., 0.001 for 0.1%)
                - slippage: Slippage factor (as decimal)
                - risk_config: Configuration for RiskManager
        """
        self.initial_capital = config.get('initial_capital', 10000)
        self.commission = config.get('commission', 0.001)
        self.slippage = config.get('slippage', 0.0005)
        
        # Initialize risk manager
        risk_config = config.get('risk_config', {})
        risk_config['initial_capital'] = self.initial_capital
        self.risk_manager = RiskManager(risk_config)
        
        # Trading state
        self.current_capital = self.initial_capital
        self.positions = []  # Open positions
        self.closed_positions = []  # Closed positions
        self.equity_curve = []  # Track equity over time
        self.trade_history = []  # All trades
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    def run_backtest(self, data: pd.DataFrame, strategy_func: Callable) -> Dict:
        """
        Run backtest on historical data
        
        Args:
            data: DataFrame with OHLCV and indicator data
            strategy_func: Strategy function that generates signals
            
        Returns:
            Dictionary with backtest results
        """
        print(f"\n{Fore.CYAN}Starting backtest...{Style.RESET_ALL}")
        
        # Apply strategy to get signals
        df = strategy_func(data.copy())
        
        # Ensure data is sorted by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Convert timestamp to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Initialize tracking variables
        self.current_capital = self.initial_capital
        self.positions = []
        self.closed_positions = []
        self.equity_curve = []
        
        # Process each candle
        for idx, row in df.iterrows():
            current_time = row['timestamp']
            current_price = row['close']
            
            # Check existing positions for exit conditions
            self._check_exits(row)
            
            # Calculate current equity
            equity = self._calculate_equity(current_price)
            self.equity_curve.append({
                'timestamp': current_time,
                'equity': equity
            })
            
            # Check for new signals
            if 'signal' in row and row['signal'] != 0 and not pd.isna(row['signal']):
                self._process_signal(row)
                
            # Progress indicator
            if idx % 100 == 0:
                progress = (idx / len(df)) * 100
                print(f"\rProgress: {progress:.1f}%", end='')
        
        print(f"\r{Fore.GREEN}Backtest completed!{Style.RESET_ALL}")
        
        # Close any remaining positions at the last price
        last_row = df.iloc[-1]
        for position in self.positions[:]:
            position.close(last_row['close'], last_row['timestamp'], 'End of data')
            self.closed_positions.append(position)
            self.positions.remove(position)
        
        # Prepare results
        results = self._prepare_results()
        
        return results
    
    def _check_exits(self, row: pd.Series):
        """Check if any positions should be closed"""
        current_price = row['close']
        current_time = row['timestamp']
        high = row['high']
        low = row['low']
        
        for position in self.positions[:]:
            should_exit = False
            exit_reason = None
            exit_price = current_price
            
            if position.signal == 1:  # Long position
                # Check stop loss
                if low <= position.stop_loss:
                    should_exit = True
                    exit_reason = 'Stop Loss'
                    exit_price = position.stop_loss
                # Check take profit
                elif high >= position.take_profit:
                    should_exit = True
                    exit_reason = 'Take Profit'
                    exit_price = position.take_profit
                    
            else:  # Short position
                # Check stop loss
                if high >= position.stop_loss:
                    should_exit = True
                    exit_reason = 'Stop Loss'
                    exit_price = position.stop_loss
                # Check take profit
                elif low <= position.take_profit:
                    should_exit = True
                    exit_reason = 'Take Profit'
                    exit_price = position.take_profit
            
            if should_exit:
                # Apply slippage
                if position.signal == 1:
                    exit_price *= (1 - self.slippage)
                else:
                    exit_price *= (1 + self.slippage)
                
                # Close position
                position.close(exit_price, current_time, exit_reason)
                
                # Update capital (add P&L minus commission)
                commission_cost = position.size * exit_price * self.commission
                self.current_capital += position.pnl - commission_cost
                
                # Move to closed positions
                self.closed_positions.append(position)
                self.positions.remove(position)
                
                # Log trade
                self.logger.info(f"Closed {position.signal} position: {position.pnl:.2f} ({exit_reason})")
    
    def _process_signal(self, row: pd.Series):
        """Process a new trading signal"""
        signal = int(row['signal'])
        current_price = row['close']
        current_time = row['timestamp']
        
        # Check risk limits
        current_drawdown = self._calculate_drawdown()
        risk_check = self.risk_manager.check_risk_limits(
            len(self.positions), 
            current_drawdown
        )
        
        if not risk_check['overall_risk_ok']:
            self.logger.warning(f"Risk limits exceeded, skipping signal")
            return
        
        # Calculate position size
        position_size = self.risk_manager.calculate_position_size(
            self.current_capital, 
            current_price
        )
        
        if position_size == 0:
            self.logger.warning(f"Position size is 0, skipping signal")
            return
        
        # Apply slippage to entry price
        if signal == 1:
            entry_price = current_price * (1 + self.slippage)
        else:
            entry_price = current_price * (1 - self.slippage)
        
        # Calculate stop loss and take profit
        atr = row.get('atr14', None)
        stop_loss = self.risk_manager.calculate_stop_loss(entry_price, signal, atr)
        take_profit = self.risk_manager.calculate_take_profit(entry_price, signal)
        
        # Calculate required capital
        required_capital = position_size * entry_price
        commission_cost = required_capital * self.commission
        total_required = required_capital + commission_cost
        
        # Check if we have enough capital
        if total_required > self.current_capital:
            self.logger.warning(f"Insufficient capital for position")
            return
        
        # Create position
        position = Position(
            symbol=row.get('symbol', 'UNKNOWN'),
            signal=signal,
            entry_price=entry_price,
            entry_time=current_time,
            size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        # Update capital
        self.current_capital -= total_required
        
        # Add to open positions
        self.positions.append(position)
        
        self.logger.info(f"Opened {signal} position at {entry_price:.2f}")
    
    def _calculate_equity(self, current_price: float) -> float:
        """Calculate current total equity"""
        equity = self.current_capital
        
        # Add unrealized P&L from open positions
        for position in self.positions:
            if position.signal == 1:
                unrealized_pnl = (current_price - position.entry_price) * position.size
            else:
                unrealized_pnl = (position.entry_price - current_price) * position.size
            equity += unrealized_pnl
            
        return equity
    
    def _calculate_drawdown(self) -> float:
        """Calculate current drawdown percentage"""
        if not self.equity_curve:
            return 0.0
            
        current_equity = self.equity_curve[-1]['equity']
        max_equity = max([e['equity'] for e in self.equity_curve])
        
        if max_equity == 0:
            return 0.0
            
        drawdown = (current_equity - max_equity) / max_equity
        return abs(drawdown)
    
    def _prepare_results(self) -> Dict:
        """Prepare backtest results"""
        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('timestamp', inplace=True)
        
        # Convert trades to list of dictionaries
        trades = [pos.to_dict() for pos in self.closed_positions]
        
        # Calculate performance metrics
        metrics_calculator = PerformanceMetrics(
            trades=trades,
            equity_curve=equity_df['equity'],
            initial_capital=self.initial_capital
        )
        
        metrics = metrics_calculator.calculate_all_metrics()
        
        # Prepare results dictionary
        results = {
            'metrics': metrics,
            'trades': trades,
            'equity_curve': equity_df,
            'initial_capital': self.initial_capital,
            'final_capital': equity_df['equity'].iloc[-1] if len(equity_df) > 0 else self.initial_capital,
            'total_trades': len(trades),
            'open_positions': len(self.positions),
            'config': {
                'commission': self.commission,
                'slippage': self.slippage,
                'risk_config': self.risk_manager.__dict__
            }
        }
        
        return results
