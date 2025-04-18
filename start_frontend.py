# start_frontend.py
import subprocess
import sys
import threading
import webbrowser
import time
import os
import socket
import argparse
import atexit

# Aggiungi supporto per i colori su Windows con colorama
try:
    import colorama
    colorama.init(autoreset=True)
    COLORED_OUTPUT = True
except ImportError:
    COLORED_OUTPUT = False
    print("Per avere output colorato, installa colorama: pip install colorama")

# Lista dei processi attivi per la gestione dell'arresto
active_processes = []

# Funzioni di utilità per i colori
def colored_text(text, color):
    if not COLORED_OUTPUT:
        return text
    
    # Codici colore ANSI
    color_codes = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'reset': '\033[0m'
    }
    
    return f"{color_codes.get(color, '')}{text}{color_codes['reset']}"

def check_port_available(port):
    """Verifica se una porta è disponibile."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = False
    try:
        # Se la connessione ha successo, la porta è in uso
        sock.bind(('127.0.0.1', port))
        result = True
    except:
        pass
    finally:
        sock.close()
    return result

def wait_for_port(port, timeout=20):
    """Attende che una porta sia in uso (server avviato)."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect(('localhost', port))
            sock.close()
            print(colored_text(f"Porta {port} è attiva!", "green"))
            return True
        except:
            time.sleep(0.5)
        finally:
            sock.close()
    return False

def process_log_line(line, prefix, success_markers, warning_markers, error_markers, info_markers):
    """Processa una riga di log e la formatta in base al tipo di messaggio."""
    content = line.strip()
    
    # Verifica se il messaggio contiene indicatori di successo
    if any(marker in content.upper() for marker in success_markers):
        print(colored_text(f"[{prefix} SUCCESS] {content}", "green"))
    # Verifica se contiene warning
    elif any(marker in content.upper() for marker in warning_markers):
        print(colored_text(f"[{prefix} WARN] {content}", "yellow"))
    # Verifica se è un messaggio informativo
    elif any(marker in content.upper() for marker in info_markers):
        print(colored_text(f"[{prefix} INFO] {content}", "blue"))
    # Verifica se è un messaggio di errore
    elif any(marker in content.upper() for marker in error_markers):
        print(colored_text(f"[{prefix} ERROR] {content}", "red"))
    # Altrimenti è un messaggio informativo generico
    else:
        print(colored_text(f"[{prefix}] {content}", "cyan"))

def stream_subprocess_output(process, prefix, success_markers, warning_markers, error_markers, info_markers):
    """Gestisce l'output di un processo in tempo reale."""
    # Thread per stdout
    def handle_stdout():
        for line in process.stdout:
            process_log_line(line, prefix, success_markers, warning_markers, error_markers, info_markers)
    
    # Thread per stderr
    def handle_stderr():
        for line in process.stderr:
            process_log_line(line, prefix, success_markers, warning_markers, error_markers, info_markers)
    
    # Avvia i thread
    stdout_thread = threading.Thread(target=handle_stdout, daemon=True)
    stderr_thread = threading.Thread(target=handle_stderr, daemon=True)
    
    stdout_thread.start()
    stderr_thread.start()
    
    return stdout_thread, stderr_thread

def run_api_server():
    """Avvia il server API usando il file app.py."""
    print(colored_text("Avvio del server API...", "cyan"))
    
    # Verifica che app.py esista
    if not os.path.exists("app.py"):
        print(colored_text("ERRORE: File app.py non trovato!", "red"))
        return None
    
    # Verifica che la porta 8000 sia disponibile
    if not check_port_available(8000):
        print(colored_text("ATTENZIONE: La porta 8000 è già in uso. Il backend potrebbe essere già in esecuzione.", "yellow"))
        
    # Avvia il server con stdout e stderr rediretti
    process = subprocess.Popen(
        [sys.executable, "app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        bufsize=1  # Line buffered
    )
    
    # Aggiungi il processo alla lista dei processi attivi
    active_processes.append(process)
    
    # Definisci i marker per i tipi di messaggi
    success_markers = ["[OK]", "SUCCESS", "LOADED"]
    warning_markers = ["WARNING", "[AVVISO]", "WARN"]
    error_markers = ["ERROR", "EXCEPTION", "FAILED"]
    info_markers = ["INFO", "Modelli caricati", "Initialize"]
    
    # Gestione dell'output
    stdout_thread, stderr_thread = stream_subprocess_output(
        process, "API", success_markers, warning_markers, error_markers, info_markers
    )
    
    # Attendi che il server si avvii
    if wait_for_port(8000):
        print(colored_text("Server API avviato correttamente!", "green"))
    else:
        print(colored_text("ERRORE: Server API non è riuscito ad avviarsi sulla porta 8000!", "red"))
        process.terminate()
        active_processes.remove(process)
        return None
    
    return process

def run_frontend_server():
    """Avvia il server frontend."""
    print(colored_text("Avvio del server frontend...", "magenta"))
    
    # Verifica che frontend_server.py esista
    if not os.path.exists("frontend_server.py"):
        print(colored_text("ERRORE: File frontend_server.py non trovato!", "red"))
        return None
    
    # Avvia il server con stdout e stderr rediretti
    process = subprocess.Popen(
        [sys.executable, "frontend_server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        bufsize=1  # Line buffered
    )
    
    # Aggiungi il processo alla lista dei processi attivi
    active_processes.append(process)
    
    # Definisci i marker per i tipi di messaggi
    success_markers = ["[OK]", "SUCCESS", "LOADED"]
    warning_markers = ["WARNING", "[AVVISO]", "WARN"]
    error_markers = ["ERROR", "EXCEPTION", "FAILED"]
    info_markers = ["INFO", "SERVING", "Initialize"]
    
    # Gestione dell'output
    stdout_thread, stderr_thread = stream_subprocess_output(
        process, "FRONTEND", success_markers, warning_markers, error_markers, info_markers
    )
    
    # Attendi che il server si avvii
    if wait_for_port(5000):
        print(colored_text("Server frontend avviato correttamente!", "green"))
    else:
        print(colored_text("ERRORE: Server frontend non è riuscito ad avviarsi sulla porta 5000!", "red"))
        process.terminate()
        active_processes.remove(process)
        return None
    
    return process

def open_browser(auto_open=True):
    """Apre il browser dopo un breve ritardo."""
    if not auto_open:
        print(colored_text("Apertura automatica del browser disabilitata.", "yellow"))
        return
        
    time.sleep(8)
    url = "http://localhost:5000"
    print(colored_text(f"Apertura del browser su {url}", "yellow"))
    webbrowser.open(url)

def terminate_processes():
    """Funzione per terminare tutti i processi attivi all'uscita."""
    print(colored_text("\nArresto dei server in corso...", "yellow"))
    
    for process in active_processes:
        try:
            if process.poll() is None:  # Processo ancora in esecuzione
                process.terminate()
                process.wait(timeout=5)  # Attendi che il processo termini
                print(colored_text(f"Processo con PID {process.pid} arrestato.", "green"))
        except Exception as e:
            print(colored_text(f"Errore durante l'arresto del processo con PID {process.pid}: {e}", "red"))
            try:
                process.kill()  # Prova a forzare la chiusura
            except:
                pass
    
    print(colored_text("Server arrestati.", "green"))

def main():
    # Registra la funzione per terminare i processi all'uscita
    atexit.register(terminate_processes)
    
    # Configura l'analisi degli argomenti della riga di comando
    parser = argparse.ArgumentParser(description='Avvia l\'applicazione Trading Bot')
    parser.add_argument('--no-auto-open', action='store_true', help='Non aprire automaticamente il browser')
    parser.add_argument('--no-auto-start', action='store_true', help='Non avviare automaticamente le predizioni')
    args = parser.parse_args()
    
    try:
        print("\n" + "="*40)
        print(colored_text("=== AVVIO APPLICAZIONE TRADING BOT ===", "cyan"))
        print("="*40 + "\n")
        
        # Avvia il server API
        api_process = run_api_server()
        if not api_process:
            print(colored_text("Impossibile avviare il server API. Uscita in corso...", "red"))
            return
            
        # Avvia il server frontend
        frontend_process = run_frontend_server()
        if not frontend_process:
            print(colored_text("Impossibile avviare il server frontend. Arresto del server API in corso...", "red"))
            api_process.terminate()
            return
        
        # Se --no-auto-start è specificato, crea un file segnale per il frontend
        if args.no_auto_start:
            try:
                os.makedirs("static", exist_ok=True)  # Assicura che la directory esista
                with open("static/no_auto_start", "w") as f:
                    f.write("1")
                print(colored_text("Predizioni automatiche disabilitate all'avvio.", "yellow"))
            except Exception as e:
                print(colored_text(f"Errore nella creazione del file segnale: {e}", "red"))
        else:
            # Rimuovi il file segnale se esiste
            if os.path.exists("static/no_auto_start"):
                try:
                    os.remove("static/no_auto_start")
                except Exception as e:
                    print(colored_text(f"Errore nella rimozione del file segnale: {e}", "red"))
        
        # Avvia l'apertura del browser in un thread separato
        threading.Thread(target=open_browser, args=(not args.no_auto_open,), daemon=True).start()
        print(colored_text("Premi Ctrl+C per terminare i server", "yellow"))
        
        # Monitora i processi finché entrambi sono attivi
        while api_process.poll() is None and frontend_process.poll() is None:
            time.sleep(1)
            
        # Se un processo è terminato, il ciclo è stato interrotto
        if api_process.poll() is not None:
            print(colored_text("Il server API si è arrestato in modo inatteso con codice di uscita: " + str(api_process.returncode), "red"))
        
        if frontend_process.poll() is not None:
            print(colored_text("Il server frontend si è arrestato in modo inatteso con codice di uscita: " + str(frontend_process.returncode), "red"))
        
    except KeyboardInterrupt:
        print(colored_text("\nInterruzione da tastiera rilevata.", "yellow"))
    finally:
        # Il cleanup finale viene gestito dalla funzione terminate_processes registrata con atexit
        pass

if __name__ == "__main__":
    main()
