"""
Script di avvio per TradingJii Backtest Suite
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def main():
    print("🚀 Starting TradingJii Backtest Suite...")
    
    # Get paths
    current_dir = Path(__file__).parent
    backend_dir = current_dir / "backend"
    frontend_file = current_dir / "frontend" / "index.html"
    
    # Change to backend directory
    os.chdir(backend_dir)
    
    # Start backend
    print("\n📡 Starting backend API server...")
    backend_process = subprocess.Popen(
        [sys.executable, "app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    # Wait for backend to start
    print("⏳ Waiting for backend to initialize...")
    time.sleep(3)
    
    # Check if backend is running
    try:
        import requests
        response = requests.get("http://localhost:8000")
        if response.status_code == 200:
            print("✅ Backend is running!")
        else:
            print("❌ Backend failed to start properly")
            return
    except:
        print("⚠️  Backend is starting... (ignore connection warnings)")
    
    # Open frontend in browser
    print(f"\n🌐 Opening frontend in browser...")
    frontend_url = f"file:///{frontend_file.absolute()}"
    webbrowser.open(frontend_url)
    
    print("\n✨ TradingJii Backtest Suite is running!")
    print("\n📌 Backend API: http://localhost:8000")
    print(f"📌 Frontend: {frontend_url}")
    print("\n⚠️  Keep this window open while using the app")
    print("📛 Press Ctrl+C to stop the application\n")
    
    # Keep running
    try:
        while True:
            output = backend_process.stdout.readline()
            if output:
                print(f"[Backend] {output.strip()}")
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down TradingJii Backtest Suite...")
        backend_process.terminate()
        print("👋 Goodbye!")

if __name__ == "__main__":
    main()
