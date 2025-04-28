# server.py

import subprocess
import sys
import threading
import webbrowser
import time
import os
import socket
import argparse
import atexit
from flask import Flask, send_from_directory
from flask_cors import CORS

# === Configurazione colori ===
try:
    import colorama
    colorama.init(autoreset=True)
    COLORED_OUTPUT = True
except ImportError:
    COLORED_OUTPUT = False
    print("Per avere output colorato, installa colorama: pip install colorama")

active_processes = []

def colored_text(text, color):
    if not COLORED_OUTPUT:
        return text
    codes = {
        'red': '\033[91m','green': '\033[92m','yellow': '\033[93m',
        'blue': '\033[94m','magenta': '\033[95m','cyan': '\033[96m',
        'white': '\033[97m','reset': '\033[0m'
    }
    return f"{codes.get(color, '')}{text}{codes['reset']}"

def check_port_available(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(('127.0.0.1', port))
        return True
    except:
        return False
    finally:
        sock.close()

def wait_for_port(port, timeout=20):
    start = time.time()
    while time.time() - start < timeout:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect(('localhost', port))
            return True
        except:
            time.sleep(0.5)
        finally:
            sock.close()
    return False

def process_log_line(line, prefix, success, warn, err, info):
    content = line.strip()
    if any(m in content.upper() for m in success):
        print(colored_text(f"[{prefix} SUCCESS] {content}", "green"))
    elif any(m in content.upper() for m in warn):
        print(colored_text(f"[{prefix} WARN] {content}", "yellow"))
    elif any(m in content.upper() for m in err):
        print(colored_text(f"[{prefix} ERROR] {content}", "red"))
    elif any(m in content.upper() for m in info):
        print(colored_text(f"[{prefix} INFO] {content}", "blue"))
    else:
        print(colored_text(f"[{prefix}] {content}", "cyan"))

def stream_subprocess_output(proc, prefix, success, warn, err, info):
    def _stdout():
        for l in proc.stdout:
            process_log_line(l, prefix, success, warn, err, info)
    def _stderr():
        for l in proc.stderr:
            process_log_line(l, prefix, success, warn, err, info)
    t1 = threading.Thread(target=_stdout, daemon=True)
    t2 = threading.Thread(target=_stderr, daemon=True)
    t1.start(); t2.start()
    return t1, t2

def run_api_server():
    print(colored_text("Avvio del server API...", "cyan"))
    if not os.path.exists("app.py"):
        print(colored_text("ERRORE: File app.py non trovato!", "red"))
        return None
    if not check_port_available(8000):
        print(colored_text("ATTENZIONE: Porta 8000 già in uso.", "yellow"))
    proc = subprocess.Popen(
        [sys.executable, "app.py"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True, bufsize=1
    )
    active_processes.append(proc)
    markers = {
        "success": ["[OK]", "SUCCESS", "LOADED"],
        "warn":    ["WARNING", "[AVVISO]", "WARN"],
        "error":   ["ERROR", "EXCEPTION", "FAILED"],
        "info":    ["INFO", "Initialize"]
    }
    stream_subprocess_output(
        proc, "API",
        markers["success"], markers["warn"],
        markers["error"], markers["info"]
    )
    if wait_for_port(8000):
        print(colored_text("Server API avviato!", "green"))
    else:
        print(colored_text("ERRORE: API non è partita su 8000!", "red"))
        proc.terminate(); active_processes.remove(proc)
        return None
    return proc

def run_frontend_server():
    print(colored_text("Avvio del server frontend...", "magenta"))
    app = Flask(__name__, static_folder='static', static_url_path='')
    CORS(app)

    @app.route('/')
    def index():
        return send_from_directory('static', 'index.html')

    thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000), daemon=True)
    thread.start()
    if wait_for_port(5000):
        print(colored_text("Server frontend avviato!", "green"))
        return thread
    else:
        print(colored_text("ERRORE: Frontend non è partito su 5000!", "red"))
        return None

def open_browser(auto_open=True):
    if not auto_open:
        print(colored_text("Apertura browser disabilitata.", "yellow"))
        return
    time.sleep(8)
    url = "http://localhost:5000"
    print(colored_text(f"Apertura browser su {url}", "yellow"))
    webbrowser.open(url)

def terminate_processes():
    print(colored_text("\nArresto server...", "yellow"))
    for p in active_processes:
        try:
            if p.poll() is None:
                p.terminate(); p.wait(timeout=5)
                print(colored_text(f"PID {p.pid} arrestato.", "green"))
        except Exception as e:
            print(colored_text(f"Errore arresto PID {p.pid}: {e}", "red"))
            try: p.kill()
            except: pass
    print(colored_text("Tutti i server sono stoppati.", "green"))

def main():
    atexit.register(terminate_processes)
    parser = argparse.ArgumentParser(description="Avvia l'app Trading Bot")
    parser.add_argument('--no-auto-open',  action='store_true', help='Non aprire browser')
    parser.add_argument('--no-auto-start', action='store_true', help='Non avviare predizioni')
    args = parser.parse_args()

    print("\n" + "="*40)
    print(colored_text("=== AVVIO TRADING BOT ===", "cyan"))
    print("="*40 + "\n")

    api_proc = run_api_server()
    if not api_proc:
        print(colored_text("Fallito avvio API. Esco.", "red")); return

    front_thread = run_frontend_server()
    if not front_thread:
        print(colored_text("Fallito avvio frontend. Termino API.", "red"))
        api_proc.terminate(); return

    if args.no_auto_start:
        os.makedirs("static", exist_ok=True)
        with open("static/no_auto_start", "w") as f:
            f.write("1")
        print(colored_text("Predizioni automatiche disabilitate.", "yellow"))
    else:
        if os.path.exists("static/no_auto_start"):
            os.remove("static/no_auto_start")

    threading.Thread(target=open_browser, args=(not args.no_auto_open,), daemon=True).start()
    print(colored_text("Premi Ctrl+C per terminare", "yellow"))

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(colored_text("\nInterruzione da tastiera.", "yellow"))

if __name__ == "__main__":
    main()
