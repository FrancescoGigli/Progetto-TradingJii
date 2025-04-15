# start_frontend.py
import subprocess
import sys
import threading
import webbrowser
import time

def run_api_server():
    """Avvia il server API usando il file app.py."""
    print("Avvio del server API...")
    return subprocess.Popen([sys.executable, "app.py"])

def run_frontend_server():
    """Avvia il server frontend."""
    print("Avvio del server frontend...")
    return subprocess.Popen([sys.executable, "frontend_server.py"])

def open_browser():
    """Apre il browser dopo un breve ritardo."""
    time.sleep(3)
    url = "http://localhost:5000"
    print(f"Apertura del browser su {url}")
    webbrowser.open(url)

def main():
    try:
        api_process = run_api_server()
        frontend_process = run_frontend_server()
        # Avvia l'apertura del browser in un thread separato
        threading.Thread(target=open_browser).start()
        print("Premi Ctrl+C per terminare i server")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nArresto dei server...")
    finally:
        try:
            api_process.terminate()
            frontend_process.terminate()
        except Exception:
            pass
        print("Server arrestati.")

if __name__ == "__main__":
    main()
