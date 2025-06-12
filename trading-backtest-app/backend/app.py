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
import subprocess
import threading
import json

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

# Global state for update process
update_state = {
    "is_running": False,
    "status": "idle",
    "progress": 0,
    "logs": [],
    "start_time": None,
    "end_time": None
}


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


class UpdateDataRequest(BaseModel):
    timeframes: List[str] = ["1h", "4h"]
    num_symbols: int = 5
    days: int = 365
    sequential: bool = False
    no_ta: bool = False


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


def run_data_collector(request: UpdateDataRequest):
    """Run data collector in a subprocess"""
    global update_state
    
    try:
        update_state["logs"] = []
        update_state["progress"] = 0
        update_state["status"] = "Starting data collector..."
        
        # Get the correct path to data_collector.py
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        data_collector_path = os.path.join(project_root, "data_collector.py")
        
        # Create logs directory if it doesn't exist
        logs_dir = os.path.join(project_root, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        # Create log file with timestamp
        log_filename = f"data_update_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_filepath = os.path.join(logs_dir, log_filename)
        
        # Log paths for debugging
        update_state["logs"].append(f"Current dir: {current_dir}")
        update_state["logs"].append(f"Project root: {project_root}")
        update_state["logs"].append(f"Data collector path: {data_collector_path}")
        update_state["logs"].append(f"Log file: {log_filepath}")
        
        if not os.path.exists(data_collector_path):
            raise FileNotFoundError(f"data_collector.py not found at {data_collector_path}")
        
        # Determine Python executable
        python_exe = sys.executable  # Use the same Python executable as the current process
        
        # Build command with -u flag for unbuffered output
        cmd = [python_exe, "-u", data_collector_path]
        
        # Add timeframes
        for tf in request.timeframes:
            cmd.extend(["-t", tf])
        
        # Add other parameters
        cmd.extend(["-n", str(request.num_symbols)])
        cmd.extend(["-d", str(request.days)])
        
        if request.sequential:
            cmd.append("--sequential")
        
        if request.no_ta:
            cmd.append("--no-ta")
        
        # Log command for debugging
        update_state["logs"].append(f"Command: {' '.join(cmd)}")
        
        # Open log file for writing
        with open(log_filepath, 'w', encoding='utf-8') as log_file:
            # Write header information
            log_file.write(f"=== TradingJii Data Update Log ===\n")
            log_file.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"Command: {' '.join(cmd)}\n")
            log_file.write(f"Working Directory: {project_root}\n")
            log_file.write(f"Timeframes: {', '.join(request.timeframes)}\n")
            log_file.write(f"Number of Symbols: {request.num_symbols}\n")
            log_file.write(f"Days of History: {request.days}\n")
            log_file.write(f"Sequential Mode: {request.sequential}\n")
            log_file.write(f"Skip Technical Indicators: {request.no_ta}\n")
            log_file.write("=" * 50 + "\n\n")
            
            # Run the process from the project root directory
            # Assicurarsi che i codici ANSI siano preservati nell'output
            # Su Windows, è necessario impostare PYTHONIOENCODING e usare COLORAMA_FORCE=1
            process_env = os.environ.copy()
            process_env["PYTHONIOENCODING"] = "utf-8"
            process_env["COLORAMA_FORCE"] = "1"
            process_env["PYTHONUNBUFFERED"] = "1"  # Forza output non bufferizzato
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Redirect stderr to stdout
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd=project_root,  # Change working directory to project root
                env=process_env,  # Passiamo le variabili d'ambiente modificate
                encoding='utf-8'  # Ensure UTF-8 encoding
            )
            
            # Read output line by line
            while True:
                output = process.stdout.readline()
                
                if output:
                    line = output.strip()
                    if line:  # Only process non-empty lines
                        update_state["logs"].append(line)
                        # Write to log file with timestamp
                        log_file.write(f"[{datetime.now().strftime('%H:%M:%S')}] {line}\n")
                        log_file.flush()  # Ensure it's written immediately
                        
                        # Print output for debugging
                        print(f"DATA COLLECTOR OUTPUT: {line}")
                        
                        # Update progress based on log content
                        if "Completati:" in line:
                            update_state["status"] = line
                            # Try to extract progress percentage
                            try:
                                if "%" in line:
                                    percent = int(line.split("%")[0].split()[-1])
                                    update_state["progress"] = percent
                            except:
                                pass
                        elif "RESOCONTO AGGIORNAMENTO DATI COMPLETATO" in line:
                            update_state["progress"] = 100
                            update_state["status"] = "Update completed successfully!"
                        elif "Ricalcolo indicatori completato!" in line:
                            update_state["progress"] = 75  # Change to 75% to show we're still finalizing
                            update_state["status"] = "Indicators recalculation completed, finalizing..."
                        elif "Database inizializzato" in line:
                            update_state["progress"] = 10
                            update_state["status"] = "Database initialized, starting download..."
                        elif "Timeframes monitorati:" in line:
                            update_state["progress"] = 5
                            update_state["status"] = "Data collector started..."
                        elif "symbols identificati da scaricare" in line:
                            update_state["progress"] = 8
                            update_state["status"] = line
                        elif "Downloading" in line or "Scaricando" in line:
                            update_state["status"] = line
                            # Try to extract progress from download messages
                            if "[" in line and "]" in line:
                                try:
                                    # Extract current/total from [1/5] format
                                    bracket_content = line[line.find("[")+1:line.find("]")]
                                    current, total = map(int, bracket_content.split("/"))
                                    percent = int((current / total) * 70) + 10  # 10-80% for downloads
                                    update_state["progress"] = percent
                                except:
                                    pass
                
                # Check if process has finished
                if process.poll() is not None:
                    # Get any remaining output
                    remaining_output = process.stdout.read()
                    
                    if remaining_output:
                        for line in remaining_output.strip().split('\n'):
                            if line.strip():
                                update_state["logs"].append(line.strip())
                                log_file.write(f"[{datetime.now().strftime('%H:%M:%S')}] {line.strip()}\n")
                    break
        
        process.wait()
        
        # Write completion information to log file
        with open(log_filepath, 'a', encoding='utf-8') as log_file:
            log_file.write(f"\n{'=' * 50}\n")
            log_file.write(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            if process.returncode == 0:
                update_state["status"] = "Update completed successfully!"
                update_state["progress"] = 100
                log_file.write(f"Status: SUCCESS\n")
            else:
                update_state["status"] = f"Update failed with exit code {process.returncode}"
                log_file.write(f"Status: FAILED (exit code {process.returncode})\n")
            
            log_file.write(f"Log file saved at: {log_filepath}\n")
        
        # Add log file location to the state
        update_state["logs"].append(f"\n=== Log file saved at: {log_filepath} ===")
            
    except Exception as e:
        update_state["status"] = f"Error: {str(e)}"
        update_state["logs"].append(f"Error: {str(e)}")
        
        # Try to write error to log file if possible
        try:
            with open(log_filepath, 'a', encoding='utf-8') as log_file:
                log_file.write(f"\n{'=' * 50}\n")
                log_file.write(f"ERROR: {str(e)}\n")
                log_file.write(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        except:
            pass
            
    finally:
        update_state["is_running"] = False
        update_state["end_time"] = datetime.now().isoformat()


@app.post("/api/update-data")
async def update_data(request: UpdateDataRequest):
    """Start data update process"""
    global update_state
    
    # Log the current state for debugging
    print(f"Current update_state: {update_state}")
    print(f"Request received: {request}")
    
    if update_state["is_running"]:
        # Check if it's a stale state (running for more than 30 minutes)
        if update_state["start_time"]:
            start = datetime.fromisoformat(update_state["start_time"])
            if (datetime.now() - start).seconds > 1800:  # 30 minutes
                print("Stale update state detected, resetting...")
                update_state["is_running"] = False
            else:
                print("Update already in progress, rejecting request")
                raise HTTPException(status_code=400, detail="Update already in progress")
        else:
            print("Update marked as running but no start time, resetting...")
            update_state["is_running"] = False
    
    # Reset state
    update_state["is_running"] = True
    update_state["status"] = "Starting..."
    update_state["progress"] = 0
    update_state["logs"] = []
    update_state["start_time"] = datetime.now().isoformat()
    update_state["end_time"] = None
    
    print("Starting data collector thread...")
    
    # Start update in background thread
    thread = threading.Thread(target=run_data_collector, args=(request,))
    thread.start()
    
    return {"message": "Update started", "status": update_state}


@app.get("/api/update-status")
async def get_update_status():
    """Get current update status"""
    return update_state


@app.post("/api/reset-update-state")
async def reset_update_state():
    """Reset update state - useful for clearing stuck states"""
    global update_state
    
    print("Resetting update state...")
    update_state["is_running"] = False
    update_state["status"] = "idle"
    update_state["progress"] = 0
    update_state["logs"] = []
    update_state["start_time"] = None
    update_state["end_time"] = None
    
    return {"message": "Update state reset", "status": update_state}


@app.get("/api/data-info")
async def get_data_info():
    """Get information about available data in the database"""
    try:
        conn = sqlite3.connect("../../crypto_data.db")
        cursor = conn.cursor()
        
        # Get available timeframes
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name LIKE 'market_data_%'
        """)
        tables = cursor.fetchall()
        timeframes = [table[0].replace('market_data_', '') for table in tables]
        
        # Get data ranges for each timeframe
        data_info = {}
        for tf in timeframes:
            table_name = f"market_data_{tf}"
            
            # Get symbols and date ranges
            cursor.execute(f"""
                SELECT 
                    symbol,
                    MIN(timestamp) as first_date,
                    MAX(timestamp) as last_date,
                    COUNT(*) as candle_count
                FROM {table_name}
                GROUP BY symbol
                ORDER BY symbol
            """)
            
            symbols_data = []
            for row in cursor.fetchall():
                symbols_data.append({
                    "symbol": row[0],
                    "first_date": row[1],
                    "last_date": row[2],
                    "candle_count": row[3]
                })
            
            data_info[tf] = {
                "symbols": symbols_data,
                "total_symbols": len(symbols_data)
            }
        
        conn.close()
        
        return {
            "timeframes": timeframes,
            "data_info": data_info
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    print("Starting TradingJii Backtest API on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
