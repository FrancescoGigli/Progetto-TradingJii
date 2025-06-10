"""
FastAPI Backend for Trading Backtest Web App
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import sqlite3
import sys
import os
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from backtest_engine import BacktestEngine
from strategies import STRATEGIES
from strategy_indicators import STRATEGY_INDICATORS

app = FastAPI(title="TradingJii Backtest API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class BacktestRequest(BaseModel):
    symbol: str
    strategy: str
    leverage: int = 1
    take_profit_pct: float = 0.02
    stop_loss_pct: float = 0.02  # Added stop loss parameter
    start_date: Optional[str] = "2024-01-01"
    end_date: Optional[str] = "2025-06-09"  # Updated to match database end date


class CompareRequest(BaseModel):
    symbol: str
    leverage: int = 1
    take_profit_pct: float = 0.02
    stop_loss_pct: float = 0.02  # Added stop loss parameter
    start_date: Optional[str] = "2024-01-01"
    end_date: Optional[str] = "2025-06-09"  # Updated to match database end date


@app.get("/")
async def root():
    return {"message": "TradingJii Backtest API is running"}


@app.get("/api/symbols")
async def get_symbols():
    """Get available symbols from database"""
    try:
        conn = sqlite3.connect("../../crypto_data.db")
        query = "SELECT DISTINCT symbol FROM market_data_1h ORDER BY symbol"
        symbols = pd.read_sql_query(query, conn)['symbol'].tolist()
        conn.close()
        return {"symbols": symbols}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/strategies")
async def get_strategies():
    """Get available strategies"""
    return {"strategies": list(STRATEGIES.keys())}


@app.get("/api/data/{symbol:path}")
async def get_market_data(symbol: str, start_date: str = "2024-01-01", end_date: str = "2025-06-09"):
    """Get market data for a symbol"""
    try:
        conn = sqlite3.connect("../../crypto_data.db")
        query = """
        SELECT timestamp, symbol, open, high, low, close, volume 
        FROM market_data_1h
        WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
        ORDER BY timestamp
        """
        df = pd.read_sql_query(query, conn, params=(symbol, start_date, end_date))
        conn.close()
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        # Convert to format suitable for TradingView
        data = []
        for _, row in df.iterrows():
            data.append({
                "time": int(pd.to_datetime(row['timestamp']).timestamp()),
                "open": row['open'],
                "high": row['high'],
                "low": row['low'],
                "close": row['close'],
                "volume": row['volume']
            })
        
        return {"data": data}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/backtest")
async def run_backtest(request: BacktestRequest):
    """Run backtest for a single strategy"""
    try:
        # Create backtest engine with $1000 capital
        engine = BacktestEngine(
            initial_capital=1000,
            leverage=request.leverage,
            take_profit_pct=request.take_profit_pct,
            stop_loss_pct=request.stop_loss_pct
        )
        
        # Load data with indicators (all in market_data_1h table)
        conn = sqlite3.connect("../../crypto_data.db")
        
        query = """
        SELECT * FROM market_data_1h
        WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
        ORDER BY timestamp
        """
        
        df = pd.read_sql_query(query, conn, params=(
            request.symbol, request.start_date, request.end_date
        ))
        conn.close()
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {request.symbol}")
        
        # Rename columns to match what strategies expect
        column_mapping = {
            'bbands_upper': 'bb_upper',
            'bbands_middle': 'bb_middle', 
            'bbands_lower': 'bb_lower',
            'macd_signal': 'macdsignal',
            'macd_hist': 'macdhist',
            'adx14': 'adx'
        }
        df = df.rename(columns=column_mapping)
        
        # Run backtest
        results = engine.run_backtest(df, request.strategy)
        
        # Format trades for frontend
        trades = []
        for trade in results['trades']:
            trades.append({
                "entry_time": int(pd.to_datetime(trade['entry_time']).timestamp()),
                "exit_time": int(pd.to_datetime(trade['exit_time']).timestamp()),
                "entry_price": trade['entry_price'],
                "exit_price": trade['exit_price'],
                "signal": trade['signal'],
                "pnl": trade['pnl'],
                "pnl_pct": trade['pnl_pct'],
                "leveraged_pnl": trade['pnl'] * request.leverage
            })
        
        # Format equity curve
        equity_data = []
        for timestamp, value in results['equity_curve'].items():
            equity_data.append({
                "time": int(pd.to_datetime(timestamp).timestamp()),
                "value": value
            })
        
        # Get indicator info for this strategy
        indicator_info = STRATEGY_INDICATORS.get(request.strategy, {})
        print(f"Strategy: {request.strategy}")
        print(f"Indicator info: {indicator_info}")
        print(f"DataFrame columns: {df.columns.tolist()}")
        
        # Prepare indicator data
        indicator_data = {}
        if indicator_info and 'indicators' in indicator_info:
            for indicator_name, indicator_config in indicator_info['indicators'].items():
                field = indicator_config.get('field')
                print(f"Looking for field '{field}' for indicator '{indicator_name}'")
                if field in df.columns:
                    print(f"Found field '{field}' in columns")
                    # Convert indicator data to time series format
                    indicator_values = []
                    for idx, row in df.iterrows():
                        if pd.notna(row[field]):
                            indicator_values.append({
                                "time": int(pd.to_datetime(row['timestamp']).timestamp()),
                                "value": float(row[field])
                            })
                    indicator_data[indicator_name] = {
                        "config": indicator_config,
                        "data": indicator_values
                    }
                    print(f"Added {len(indicator_values)} values for {indicator_name}")
                else:
                    print(f"Field '{field}' not found in columns")
        
        print(f"Final indicator_data: {list(indicator_data.keys())}")
        
        return {
            "strategy": request.strategy,
            "symbol": request.symbol,
            "metrics": results['metrics'],
            "trades": trades,
            "equity_curve": equity_data,
            "indicators": indicator_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/compare")
async def compare_strategies(request: CompareRequest):
    """Compare all strategies"""
    try:
        # Create backtest engine
        engine = BacktestEngine(
            initial_capital=1000,
            leverage=request.leverage,
            take_profit_pct=request.take_profit_pct,
            stop_loss_pct=request.stop_loss_pct
        )
        
        # Load data with indicators (all in market_data_1h table)
        conn = sqlite3.connect("../../crypto_data.db")
        
        query = """
        SELECT * FROM market_data_1h
        WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
        ORDER BY timestamp
        """
        
        df = pd.read_sql_query(query, conn, params=(
            request.symbol, request.start_date, request.end_date
        ))
        conn.close()
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {request.symbol}")
        
        # Rename columns to match what strategies expect
        column_mapping = {
            'bbands_upper': 'bb_upper',
            'bbands_middle': 'bb_middle',
            'bbands_lower': 'bb_lower',
            'macd_signal': 'macdsignal',
            'macd_hist': 'macdhist',
            'adx14': 'adx'
        }
        df = df.rename(columns=column_mapping)
        
        # Run backtest for each strategy
        comparison = []
        for strategy_name in STRATEGIES.keys():
            try:
                results = engine.run_backtest(df.copy(), strategy_name)
                comparison.append({
                    "strategy": strategy_name,
                    "total_return": results['metrics']['total_return'],
                    "total_return_pct": results['metrics']['total_return_pct'],
                    "leveraged_return": results['metrics']['total_return'] * request.leverage,
                    "leveraged_return_pct": results['metrics']['total_return_pct'] * request.leverage,
                    "total_trades": results['metrics']['total_trades'],
                    "win_rate": results['metrics']['win_rate'],
                    "sharpe_ratio": results['metrics']['sharpe_ratio'],
                    "max_drawdown_pct": results['metrics']['max_drawdown_pct'],
                    "profit_factor": results['metrics']['profit_factor']
                })
            except Exception as e:
                print(f"Error with strategy {strategy_name}: {e}")
                continue
        
        # Sort by leveraged return
        comparison.sort(key=lambda x: x['leveraged_return'], reverse=True)
        
        return {
            "symbol": request.symbol,
            "leverage": request.leverage,
            "take_profit_pct": request.take_profit_pct,
            "stop_loss_pct": request.stop_loss_pct,
            "comparison": comparison
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    print("Starting TradingJii Backtest API on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
