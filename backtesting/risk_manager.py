"""
Risk Management Module for Backtesting

Handles position sizing, stop loss, take profit, and risk calculations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


class RiskManager:
    """
    Risk management system for backtesting
    
    Handles:
    - Position sizing based on risk percentage
    - Stop loss and take profit calculations
    - Maximum exposure limits
    """
    
    def __init__(self, config: Dict):
        """
        Initialize risk manager with configuration
        
        Args:
            config: Dictionary with risk parameters
                - initial_capital: Starting capital
                - position_size_pct: Percentage of capital per trade (e.g., 0.02 for 2%)
                - stop_loss_pct: Stop loss percentage (e.g., 0.02 for 2%)
                - take_profit_pct: Take profit percentage (e.g., 0.04 for 4%)
                - max_positions: Maximum number of concurrent positions
                - use_atr_stops: Use ATR-based stops instead of fixed percentage
                - atr_multiplier: Multiplier for ATR-based stops
        """
        self.initial_capital = config.get('initial_capital', 10000)
        self.position_size_pct = config.get('position_size_pct', 0.02)  # 2% default
        self.stop_loss_pct = config.get('stop_loss_pct', 0.02)  # 2% default
        self.take_profit_pct = config.get('take_profit_pct', 0.04)  # 4% default
        self.max_positions = config.get('max_positions', 1)
        self.use_atr_stops = config.get('use_atr_stops', False)
        self.atr_multiplier = config.get('atr_multiplier', 2.0)
        
    def calculate_position_size(self, capital: float, price: float) -> float:
        """
        Calculate position size based on available capital and risk parameters
        
        Args:
            capital: Current available capital
            price: Entry price of the asset
            
        Returns:
            Number of units to trade (can be fractional for crypto)
        """
        # Calculate position value based on risk percentage
        position_value = capital * self.position_size_pct
        
        # Calculate number of units (allow fractional units for crypto)
        units = position_value / price
        
        # Round to 8 decimal places (standard for crypto)
        units = round(units, 8)
        
        return units
    
    def calculate_stop_loss(self, entry_price: float, signal: int, 
                          atr: Optional[float] = None) -> float:
        """
        Calculate stop loss price
        
        Args:
            entry_price: Entry price of the position
            signal: 1 for long, -1 for short
            atr: Average True Range value (optional, for ATR-based stops)
            
        Returns:
            Stop loss price
        """
        if self.use_atr_stops and atr is not None:
            # ATR-based stop loss
            stop_distance = atr * self.atr_multiplier
            if signal == 1:  # Long position
                return entry_price - stop_distance
            else:  # Short position
                return entry_price + stop_distance
        else:
            # Fixed percentage stop loss
            if signal == 1:  # Long position
                return entry_price * (1 - self.stop_loss_pct)
            else:  # Short position
                return entry_price * (1 + self.stop_loss_pct)
    
    def calculate_take_profit(self, entry_price: float, signal: int,
                            risk_reward_ratio: Optional[float] = None) -> float:
        """
        Calculate take profit price
        
        Args:
            entry_price: Entry price of the position
            signal: 1 for long, -1 for short
            risk_reward_ratio: Optional risk/reward ratio (default uses take_profit_pct)
            
        Returns:
            Take profit price
        """
        if risk_reward_ratio is not None:
            # Calculate TP based on risk/reward ratio
            sl_distance = abs(entry_price - self.calculate_stop_loss(entry_price, signal))
            tp_distance = sl_distance * risk_reward_ratio
            
            if signal == 1:  # Long position
                return entry_price + tp_distance
            else:  # Short position
                return entry_price - tp_distance
        else:
            # Fixed percentage take profit
            if signal == 1:  # Long position
                return entry_price * (1 + self.take_profit_pct)
            else:  # Short position
                return entry_price * (1 - self.take_profit_pct)
    
    def check_risk_limits(self, current_positions: int, current_drawdown: float,
                         max_drawdown_limit: float = 0.20) -> Dict[str, bool]:
        """
        Check if current risk limits are exceeded
        
        Args:
            current_positions: Number of open positions
            current_drawdown: Current drawdown percentage
            max_drawdown_limit: Maximum allowed drawdown (default 20%)
            
        Returns:
            Dictionary with risk check results
        """
        checks = {
            'can_open_position': current_positions < self.max_positions,
            'within_drawdown_limit': current_drawdown < max_drawdown_limit,
            'overall_risk_ok': True
        }
        
        # Overall risk is OK only if all individual checks pass
        checks['overall_risk_ok'] = all([
            checks['can_open_position'],
            checks['within_drawdown_limit']
        ])
        
        return checks
    
    def calculate_risk_metrics(self, entry_price: float, stop_loss: float,
                             position_size: float) -> Dict[str, float]:
        """
        Calculate risk metrics for a position
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            position_size: Number of units
            
        Returns:
            Dictionary with risk metrics
        """
        # Calculate risk per unit
        risk_per_unit = abs(entry_price - stop_loss)
        
        # Total risk for the position
        total_risk = risk_per_unit * position_size
        
        # Risk as percentage of position value
        position_value = entry_price * position_size
        risk_percentage = (total_risk / position_value) * 100
        
        return {
            'risk_per_unit': risk_per_unit,
            'total_risk': total_risk,
            'position_value': position_value,
            'risk_percentage': risk_percentage
        }
