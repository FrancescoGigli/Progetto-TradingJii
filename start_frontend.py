# start_frontend.py
import subprocess
import sys
import threading
import webbrowser
import time
import os
import socket

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
            print(f"Porta {port} è attiva!")
            return True
        except:
            time.sleep(0.5)
        finally:
            sock.close()
    return False

def run_api_server():
    """Avvia il server API usando il file app.py."""
    print("Avvio del server API...")
    
    # Verifica che app.py esista
    if not os.path.exists("app.py"):
        print("ERRORE: File app.py non trovato!")
        return None
    
    # Verifica che la porta 8000 sia disponibile
    if not check_port_available(8000):
        print("ATTENZIONE: La porta 8000 è già in uso. Il backend potrebbe essere già in esecuzione.")
        
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
            print(f"[API] {line.strip()}")
    
    def log_error():
        for line in process.stderr:
            print(f"[API ERROR] {line.strip()}")
    
    threading.Thread(target=log_output, daemon=True).start()
    threading.Thread(target=log_error, daemon=True).start()
    
    # Attendi che il server si avvii
    if wait_for_port(8000):
        print("Server API avviato correttamente!")
    else:
        print("ERRORE: Server API non è riuscito ad avviarsi sulla porta 8000!")
        process.terminate()
        return None
    
    return process

def run_frontend_server():
    """Avvia il server frontend."""
    print("Avvio del server frontend...")
    
    # Verifica che frontend_server.py esista
    if not os.path.exists("frontend_server.py"):
        print("ERRORE: File frontend_server.py non trovato!")
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
            print(f"[FRONTEND] {line.strip()}")
    
    def log_error():
        for line in process.stderr:
            print(f"[FRONTEND ERROR] {line.strip()}")
    
    threading.Thread(target=log_output, daemon=True).start()
    threading.Thread(target=log_error, daemon=True).start()
    
    # Attendi che il server si avvii
    if wait_for_port(5000):
        print("Server frontend avviato correttamente!")
    else:
        print("ERRORE: Server frontend non è riuscito ad avviarsi sulla porta 5000!")
        process.terminate()
        return None
    
    return process

def open_browser():
    """Apre il browser dopo un breve ritardo."""
    time.sleep(8)
    url = "http://localhost:5000"
    print(f"Apertura del browser su {url}")
    webbrowser.open(url)

def main():
    try:
        print("\n=== AVVIO APPLICAZIONE TRADING BOT ===\n")
        
        # Avvia il server API
        api_process = run_api_server()
        if not api_process:
            print("Impossibile avviare il server API. Uscita in corso...")
            return
            
        # Avvia il server frontend
        frontend_process = run_frontend_server()
        if not frontend_process:
            print("Impossibile avviare il server frontend. Arresto del server API in corso...")
            api_process.terminate()
            return
        
        # Avvia l'apertura del browser in un thread separato
        threading.Thread(target=open_browser).start()
        print("Premi Ctrl+C per terminare i server")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nArresto dei server...")
    finally:
        try:
            if 'api_process' in locals() and api_process:
                api_process.terminate()
                print("Server API arrestato.")
                
            if 'frontend_process' in locals() and frontend_process:
                frontend_process.terminate()
                print("Server frontend arrestato.")
        except Exception as e:
            print(f"Errore durante l'arresto dei server: {e}")
        
        print("Server arrestati.")

if __name__ == "__main__":
    main()
