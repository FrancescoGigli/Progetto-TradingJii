"""
Main script to run backtesting on trading strategies

Usage:
    python backtest_strategies.py --strategy ema_crossover --symbol BTC/USDT:USDT --start 2024-01-01 --end 2024-12-31
"""

import sqlite3
import pandas as pd
import argparse
from datetime import datetime
from colorama import init, Fore, Style
import sys
import os

# Initialize colorama
init()

# Import backtesting modules
from backtesting.backtest_engine import BacktestEngine
from backtesting.visualizer import BacktestVisualizer
from backtesting.metrics import PerformanceMetrics

# Import strategies
from strategies.rsi_mean_reversion import generate_signals as rsi_signals
from strategies.ema_crossover import generate_signals as ema_signals
from strategies.breakout_range import generate_signals as breakout_signals
from strategies.bollinger_rebound import generate_signals as bollinger_signals
from strategies.macd_histogram import generate_signals as macd_signals
from strategies.donchian_breakout import generate_signals as donchian_signals


# Strategy mapping
STRATEGIES = {
    'rsi_mean_reversion': rsi_signals,
    'ema_crossover': ema_signals,
    'breakout_range': breakout_signals,
    'bollinger_rebound': bollinger_signals,
    'macd_histogram': macd_signals,
    'donchian_breakout': donchian_signals
}


def load_data(symbol: str, start_date: str, end_date: str, timeframe: str = '1h') -> pd.DataFrame:
    """Load data from database for backtesting"""
    print(f"{Fore.CYAN}Loading data for {symbol} from {start_date} to {end_date}...{Style.RESET_ALL}")
    
    conn = sqlite3.connect("crypto_data.db")
    
    # Query to load data
    table_name = f"market_data_{timeframe}"
    query = f"""
    SELECT * FROM {table_name}
    WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
    ORDER BY timestamp
    """
    
    df = pd.read_sql_query(query, conn, params=(symbol, start_date, end_date))
    conn.close()
    
    if df.empty:
        raise ValueError(f"No data found for {symbol} in the specified date range")
    
    print(f"{Fore.GREEN}Loaded {len(df)} records{Style.RESET_ALL}")
    return df


def run_single_backtest(strategy_name: str, symbol: str, start_date: str, 
                       end_date: str, config: dict = None) -> dict:
    """Run backtest for a single strategy"""
    
    # Default configuration
    if config is None:
        config = {
            'initial_capital': 10000,
            'commission': 0.001,  # 0.1%
            'slippage': 0.0005,   # 0.05%
            'risk_config': {
                'position_size_pct': 0.02,  # 2% per trade
                'stop_loss_pct': 0.02,      # 2% stop loss
                'take_profit_pct': 0.04,    # 4% take profit
                'max_positions': 1          # Max concurrent positions
            }
        }
    
    # Load data
    df = load_data(symbol, start_date, end_date)
    
    # Get strategy function
    if strategy_name not in STRATEGIES:
        raise ValueError(f"Strategy {strategy_name} not found. Available: {list(STRATEGIES.keys())}")
    
    strategy_func = STRATEGIES[strategy_name]
    
    # Initialize backtest engine
    engine = BacktestEngine(config)
    
    # Run backtest
    print(f"\n{Fore.YELLOW}Running backtest for {strategy_name}...{Style.RESET_ALL}")
    results = engine.run_backtest(df, strategy_func)
    
    # Print summary
    metrics = results['metrics']
    print(f"\n{Fore.CYAN}=== BACKTEST RESULTS ==={Style.RESET_ALL}")
    print(f"Strategy: {strategy_name}")
    print(f"Symbol: {symbol}")
    print(f"Period: {start_date} to {end_date}")
    print(f"\n{Fore.GREEN}Performance Summary:{Style.RESET_ALL}")
    print(f"Total Return: ${metrics['total_return']:.2f} ({metrics['total_return_pct']:.2f}%)")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    
    # Create visualization
    visualizer = BacktestVisualizer(results, strategy_name, symbol)
    report_path = visualizer.create_full_report()
    
    print(f"\n{Fore.GREEN}✅ Backtest completed successfully!{Style.RESET_ALL}")
    print(f"📊 Interactive report saved to: {report_path}")
    
    return results


def compare_strategies(symbol: str, start_date: str, end_date: str):
    """Compare all strategies on the same data"""
    print(f"\n{Fore.CYAN}=== COMPARING ALL STRATEGIES ==={Style.RESET_ALL}")
    
    results_summary = []
    
    for strategy_name in STRATEGIES.keys():
        try:
            print(f"\n{Fore.YELLOW}Testing {strategy_name}...{Style.RESET_ALL}")
            results = run_single_backtest(strategy_name, symbol, start_date, end_date)
            
            metrics = results['metrics']
            results_summary.append({
                'strategy': strategy_name,
                'total_return_pct': metrics['total_return_pct'],
                'total_trades': metrics['total_trades'],
                'win_rate': metrics['win_rate'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'max_drawdown_pct': metrics['max_drawdown_pct'],
                'profit_factor': metrics['profit_factor']
            })
        except Exception as e:
            print(f"{Fore.RED}Error with {strategy_name}: {e}{Style.RESET_ALL}")
    
    # Create comparison DataFrame
    df_comparison = pd.DataFrame(results_summary)
    df_comparison = df_comparison.sort_values('total_return_pct', ascending=False)
    
    print(f"\n{Fore.CYAN}=== STRATEGY COMPARISON ==={Style.RESET_ALL}")
    print(df_comparison.to_string(index=False))
    
    # Save comparison to CSV
    # Clean symbol name for filename (replace invalid characters)
    clean_symbol = symbol.replace('/', '_').replace(':', '_')
    comparison_file = f"backtest_results/comparison_{clean_symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    os.makedirs("backtest_results", exist_ok=True)
    df_comparison.to_csv(comparison_file, index=False)
    print(f"\n{Fore.GREEN}Comparison saved to: {comparison_file}{Style.RESET_ALL}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Backtest trading strategies')
    parser.add_argument('--strategy', type=str, help='Strategy name (or "all" to compare all)')
    parser.add_argument('--symbol', type=str, default='BTC/USDT:USDT', help='Trading symbol')
    parser.add_argument('--start', type=str, default='2024-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2024-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=10000, help='Initial capital')
    parser.add_argument('--position-size', type=float, default=0.02, help='Position size (% of capital)')
    parser.add_argument('--stop-loss', type=float, default=0.02, help='Stop loss %')
    parser.add_argument('--take-profit', type=float, default=0.04, help='Take profit %')
    
    args = parser.parse_args()
    
    print(f"{Fore.GREEN}=== TradingJii Backtesting System ==={Style.RESET_ALL}")
    
    # Validate dates
    try:
        datetime.strptime(args.start, '%Y-%m-%d')
        datetime.strptime(args.end, '%Y-%m-%d')
    except ValueError:
        print(f"{Fore.RED}Error: Invalid date format. Use YYYY-MM-DD{Style.RESET_ALL}")
        return
    
    # Create config from arguments
    config = {
        'initial_capital': args.capital,
        'commission': 0.001,
        'slippage': 0.0005,
        'risk_config': {
            'position_size_pct': args.position_size,
            'stop_loss_pct': args.stop_loss,
            'take_profit_pct': args.take_profit,
            'max_positions': 1
        }
    }
    
    try:
        if args.strategy == 'all':
            # Compare all strategies
            compare_strategies(args.symbol, args.start, args.end)
        else:
            # Run single strategy
            if not args.strategy:
                print(f"{Fore.YELLOW}Available strategies:{Style.RESET_ALL}")
                for strategy in STRATEGIES.keys():
                    print(f"  - {strategy}")
                print(f"\n{Fore.CYAN}Example usage:{Style.RESET_ALL}")
                print("python backtest_strategies.py --strategy ema_crossover --symbol BTC/USDT:USDT")
                return
                
            run_single_backtest(args.strategy, args.symbol, args.start, args.end, config)
            
    except Exception as e:
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
