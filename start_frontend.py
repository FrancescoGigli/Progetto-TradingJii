import subprocess
import sys
import os
import time
import threading
import webbrowser
import psutil  # Può richiedere l'installazione con pip install psutil

def kill_existing_python_processes():
    """Termina tutti i processi Python che potrebbero interferire"""
    print("Verifica di processi Python esistenti...")
    current_pid = os.getpid()
    
    try:
        for proc in psutil.process_iter(['pid', 'name']):
            # Se è un processo Python e non è questo processo
            if proc.info['name'] == 'python.exe' and proc.pid != current_pid:
                try:
                    # Verifica se il processo sta eseguendo uno dei nostri server
                    cmdline = proc.cmdline()
                    if any(x in ' '.join(cmdline) for x in ['api.py', 'frontend_server.py']):
                        print(f"Terminazione processo Python: {proc.pid}")
                        proc.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
    except ImportError:
        print("psutil non installato. Impossibile terminare processi esistenti.")
    
    time.sleep(1)  # Breve attesa per consentire ai processi di terminare

def run_api_server():
    """Avvia il server API"""
    print("Avvio del server API...")
    api_process = subprocess.Popen([sys.executable, "api.py"])
    return api_process

def run_frontend_server():
    """Avvia il server frontend"""
    print("Avvio del server frontend...")
    frontend_process = subprocess.Popen([sys.executable, "frontend_server.py"])
    return frontend_process

def open_browser():
    """Apre il browser dopo un breve ritardo"""
    time.sleep(3)  # Attendi che entrambi i server siano avviati
    url = "http://localhost:5000"
    print(f"Apertura del browser su {url}")
    webbrowser.open(url)

def main():
    try:
        # Controlla se le dipendenze sono installate
        try:
            import flask
            import flask_cors
            try:
                import psutil
            except ImportError:
                print("Installazione di psutil...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
        except ImportError:
            print("Installazione delle dipendenze richieste...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "flask", "flask-cors", "psutil"])
            print("Dipendenze installate con successo!")
        
        # Termina processi Python esistenti
        kill_existing_python_processes()
        
        # Avvia i server
        api_process = run_api_server()
        frontend_process = run_frontend_server()
        
        # Apri il browser (solo una volta)
        threading.Thread(target=open_browser).start()
        
        print("Premi Ctrl+C per terminare i server")
        
        # Attendi che l'utente prema Ctrl+C
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nArresto dei server...")
    finally:
        # Assicurati di terminare i processi
        try:
            api_process.terminate()
            frontend_process.terminate()
        except:
            pass
        print("Server arrestati.")

if __name__ == "__main__":
    main() 