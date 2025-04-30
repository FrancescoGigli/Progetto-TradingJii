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
import json
from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS
import logging
from datetime import datetime

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

def run_training(params):
    print(colored_text(f"Avvio training con parametri: {params}", "cyan"))
    
    # Estraiamo i parametri principali
    timeframes = params.get('timeframes', [])
    models = params.get('models', [])
    symbols = params.get('symbols', 30)
    
    if not timeframes or not models:
        print(colored_text("Errore: nessun timeframe o modello specificato", "red"))
        return None
    
    # Converti in liste se sono valori singoli
    if not isinstance(timeframes, list):
        timeframes = [timeframes]
    if not isinstance(models, list):
        models = [models]
    
    # Conta quanti modelli verranno addestrati
    total_models = len(models) * len(timeframes)
    print(colored_text(f"Il training includerà {total_models} modelli ({', '.join(models)}) per {len(timeframes)} timeframe ({', '.join(timeframes)})", "cyan"))
    
    # Crea il comando base
    cmd = [sys.executable, "main.py", "--train-only"]
    
    # Aggiungi i parametri
    cmd.extend(['--timeframes', ','.join(timeframes)])
    cmd.extend(['--models', ','.join(models)])
    cmd.extend(['--symbols', str(symbols)])
    
    print(colored_text(f"Esecuzione comando: {' '.join(cmd)}", "cyan"))
    
    # Esegui il processo
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True, bufsize=1
    )
    active_processes.append(proc)
    
    # Definisci i marker per il monitoraggio dei log
    markers = {
        "success": ["[OK]", "SUCCESS", "LOADED", "COMPLETATO"],
        "warn":    ["WARNING", "[AVVISO]", "WARN"],
        "error":   ["ERROR", "EXCEPTION", "FAILED", "ERRORE"],
        "info":    ["INFO", "Initialize", "TRAINING", "Fetching", "Validazione"]
    }
    
    # Avvia lo streaming dei log
    stream_subprocess_output(
        proc, "TRAINING",
        markers["success"], markers["warn"],
        markers["error"], markers["info"]
    )
    
    return proc

def run_frontend_server():
    print(colored_text("Avvio del server frontend...", "magenta"))
    app = Flask(__name__, static_folder='static', static_url_path='')
    CORS(app)
    
    # Disattiva i log delle richieste HTTP
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)  # Mostra solo errori gravi, non richieste 404

    @app.route('/')
    def index():
        return send_from_directory('static', 'index.html')
        
    @app.route('/api/train', methods=['POST'])
    def train_models():
        try:
            data = request.json
            print(colored_text(f"Richiesta di training ricevuta: {data}", "cyan"))
            
            # Avvia il training in un thread separato
            proc = run_training(data)
            
            return jsonify({
                "status": "success", 
                "message": "Training avviato con successo"
            })
        except Exception as e:
            print(colored_text(f"Errore nell'avvio del training: {str(e)}", "red"))
            return jsonify({
                "status": "error", 
                "message": f"Errore nell'avvio del training: {str(e)}"
            }), 500
    
    @app.route('/api/model-metrics/<model_name>/<timeframe>', methods=['GET'])
    def get_model_metrics(model_name, timeframe):
        """Endpoint per ottenere le metriche dettagliate di un modello specifico."""
        try:
            # Percorso del file delle metriche
            metrics_file = os.path.join("logs", "plots", f"training_metrics_{timeframe}.json")
            
            # Controllo se il file esiste
            if not os.path.exists(metrics_file):
                return jsonify({
                    "status": "error", 
                    "message": f"Metriche non disponibili per il modello {model_name} con timeframe {timeframe}"
                }), 404
            
            # Legge il file delle metriche
            with open(metrics_file, 'r') as f:
                metrics_data = json.load(f)
            
            # Aggiungi informazioni sul modello specifico
            metrics_data["model_info"] = {
                "model_name": model_name,
                "timeframe": timeframe,
                "model_file": f"trained_models/{model_name}_model_{timeframe}.h5" 
                              if model_name == "lstm" else f"trained_models/{model_name}_model_{timeframe}.pkl",
                "is_available": True  # Potresti verificare l'esistenza del file del modello
            }
            
            return jsonify({
                "status": "success",
                "data": metrics_data
            })
        except Exception as e:
            print(colored_text(f"Errore nel recupero delle metriche: {str(e)}", "red"))
            return jsonify({
                "status": "error", 
                "message": f"Errore nel recupero delle metriche: {str(e)}"
            }), 500
    
    @app.route('/api/models', methods=['GET'])
    def list_available_models():
        """Endpoint per ottenere la lista di tutti i modelli disponibili con le loro metriche base."""
        try:
            # Directory dei modelli addestrati
            model_dir = "trained_models"
            metrics_dir = os.path.join("logs", "plots")
            
            if not os.path.exists(model_dir):
                return jsonify({
                    "status": "success",
                    "data": {
                        "models": []
                    }
                })
            
            # Lista tutti i file nella directory dei modelli
            models = []
            for file in os.listdir(model_dir):
                if file.endswith('.h5') or file.endswith('.pkl'):
                    # Estrai informazioni dal nome del file
                    parts = file.split('_')
                    if len(parts) >= 3:
                        model_type = parts[0]
                        timeframe = parts[2].split('.')[0]
                        
                        # Controlla se esistono metriche per questo modello
                        metrics_file = os.path.join(metrics_dir, f"training_metrics_{timeframe}.json")
                        metrics_available = os.path.exists(metrics_file)
                        
                        if metrics_available:
                            # Carica le metriche di base se disponibili
                            with open(metrics_file, 'r') as f:
                                metrics = json.load(f)
                                basic_metrics = {
                                    "accuracy": metrics["validation"]["final_accuracy"],
                                    "loss": metrics["validation"]["final_loss"],
                                    "epochs": metrics["summary"]["epochs_trained"],
                                    "condition": metrics["summary"]["model_condition"]
                                }
                        else:
                            basic_metrics = None
                        
                        models.append({
                            "model_type": model_type,
                            "timeframe": timeframe,
                            "file_path": os.path.join(model_dir, file),
                            "metrics_available": metrics_available,
                            "basic_metrics": basic_metrics
                        })
            
            return jsonify({
                "status": "success",
                "data": {
                    "models": models
                }
            })
        except Exception as e:
            print(colored_text(f"Errore nell'elenco dei modelli: {str(e)}", "red"))
            return jsonify({
                "status": "error", 
                "message": f"Errore nell'elenco dei modelli: {str(e)}"
            }), 500

    @app.route('/api/status', methods=['GET'])
    def get_status():
        """Endpoint per lo stato generale del sistema."""
        try:
            # Informazioni generali di stato
            return jsonify({
                "status": "online",
                "api_running": True,
                "version": "1.0.0",
                "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": str(e)
            }), 500

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
