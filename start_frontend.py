# start_frontend.py
import subprocess
import sys
import threading
import webbrowser
import time
import os
import socket

# Aggiungi supporto per i colori su Windows con colorama
try:
    import colorama
    colorama.init(autoreset=True)
    COLORED_OUTPUT = True
except ImportError:
    COLORED_OUTPUT = False
    print("Per avere output colorato, installa colorama: pip install colorama")

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
        universal_newlines=True
    )
    
    # Thread per stampare l'output del server
    def log_output():
        for line in process.stdout:
            # Analizza il contenuto per determinare il tipo di messaggio
            content = line.strip()
            
            # Verifica se il messaggio contiene indicatori di successo
            if "[OK]" in content or "SUCCESS" in content.upper() or "LOADED" in content.upper():
                print(colored_text(f"[API SUCCESS] {content}", "green"))
            # Verifica se contiene warning
            elif "WARNING" in content.upper() or "[AVVISO]" in content or "WARN" in content.upper():
                print(colored_text(f"[API WARN] {content}", "yellow"))
            # Verifica se è un messaggio di errore
            elif "ERROR" in content.upper() or "EXCEPTION" in content.upper() or "FAILED" in content.upper():
                print(colored_text(f"[API ERROR] {content}", "red"))
            # Altrimenti è un messaggio informativo
            else:
                print(colored_text(f"[API INFO] {content}", "cyan"))
    
    def log_error():
        for line in process.stderr:
            # Analizza il contenuto per determinare il tipo di messaggio
            content = line.strip()
            
            # Verifica se il messaggio contiene indicatori di successo
            if "[OK]" in content or "SUCCESS" in content.upper() or "LOADED" in content.upper():
                print(colored_text(f"[API INFO] {content}", "green"))
            # Verifica se contiene warning
            elif "WARNING" in content.upper() or "[AVVISO]" in content or "WARN" in content.upper():
                print(colored_text(f"[API WARN] {content}", "yellow"))
            # Verifica se è un messaggio informativo
            elif "INFO" in content.upper() or "Modelli caricati" in content or "Initialize" in content:
                print(colored_text(f"[API INFO] {content}", "blue"))
            # Altrimenti è un errore
            else:
                print(colored_text(f"[API ERROR] {content}", "red"))
    
    threading.Thread(target=log_output, daemon=True).start()
    threading.Thread(target=log_error, daemon=True).start()
    
    # Attendi che il server si avvii
    if wait_for_port(8000):
        print(colored_text("Server API avviato correttamente!", "green"))
    else:
        print(colored_text("ERRORE: Server API non è riuscito ad avviarsi sulla porta 8000!", "red"))
        process.terminate()
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
        universal_newlines=True
    )
    
    # Thread per stampare l'output del server
    def log_output():
        for line in process.stdout:
            # Analizza il contenuto per determinare il tipo di messaggio
            content = line.strip()
            
            # Verifica se il messaggio contiene indicatori di successo
            if "[OK]" in content or "SUCCESS" in content.upper() or "LOADED" in content.upper():
                print(colored_text(f"[FRONTEND SUCCESS] {content}", "green"))
            # Verifica se contiene warning
            elif "WARNING" in content.upper() or "[AVVISO]" in content or "WARN" in content.upper():
                print(colored_text(f"[FRONTEND WARN] {content}", "yellow"))
            # Verifica se è un messaggio di errore
            elif "ERROR" in content.upper() or "EXCEPTION" in content.upper() or "FAILED" in content.upper():
                print(colored_text(f"[FRONTEND ERROR] {content}", "red"))
            # Altrimenti è un messaggio informativo
            else:
                print(colored_text(f"[FRONTEND INFO] {content}", "blue"))
    
    def log_error():
        for line in process.stderr:
            # Analizza il contenuto per determinare il tipo di messaggio
            content = line.strip()
            
            # Verifica se il messaggio contiene indicatori di successo
            if "[OK]" in content or "SUCCESS" in content.upper() or "LOADED" in content.upper():
                print(colored_text(f"[FRONTEND INFO] {content}", "green"))
            # Verifica se contiene warning
            elif "WARNING" in content.upper() or "[AVVISO]" in content or "WARN" in content.upper():
                print(colored_text(f"[FRONTEND WARN] {content}", "yellow"))
            # Verifica se è un messaggio informativo
            elif "INFO" in content.upper() or "SERVING" in content.upper() or "Initialize" in content:
                print(colored_text(f"[FRONTEND INFO] {content}", "blue"))
            # Altrimenti è un errore
            else:
                print(colored_text(f"[FRONTEND ERROR] {content}", "red"))
    
    threading.Thread(target=log_output, daemon=True).start()
    threading.Thread(target=log_error, daemon=True).start()
    
    # Attendi che il server si avvii
    if wait_for_port(5000):
        print(colored_text("Server frontend avviato correttamente!", "green"))
    else:
        print(colored_text("ERRORE: Server frontend non è riuscito ad avviarsi sulla porta 5000!", "red"))
        process.terminate()
        return None
    
    return process

def open_browser():
    """Apre il browser dopo un breve ritardo."""
    time.sleep(8)
    url = "http://localhost:5000"
    print(colored_text(f"Apertura del browser su {url}", "yellow"))
    webbrowser.open(url)

def main():
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
        
        # Avvia l'apertura del browser in un thread separato
        threading.Thread(target=open_browser).start()
        print(colored_text("Premi Ctrl+C per terminare i server", "yellow"))
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(colored_text("\nArresto dei server...", "yellow"))
    finally:
        try:
            if 'api_process' in locals() and api_process:
                api_process.terminate()
                print(colored_text("Server API arrestato.", "green"))
                
            if 'frontend_process' in locals() and frontend_process:
                frontend_process.terminate()
                print(colored_text("Server frontend arrestato.", "green"))
        except Exception as e:
            print(colored_text(f"Errore durante l'arresto dei server: {e}", "red"))
        
        print(colored_text("Server arrestati.", "green"))

if __name__ == "__main__":
    main()
