#!/usr/bin/env python3
"""
Pipeline di Analisi della Volatilità
===================================

Questo script avvia automaticamente il sistema completo di analisi della volatilità
delle criptovalute con configurazione di default:
- Top 200 criptovalute per volume
- Ultimi 150 giorni di dati
- Timeframes: 5m e 15m

Per personalizzare i parametri, utilizzare direttamente:
python src/scripts/volatility_pipeline.py --top X --days Y --timeframes "t1,t2,..."
"""

import os
import sys
import asyncio
import subprocess
from colorama import init, Fore, Style

# Inizializza colorama per i colori cross-platform
init(autoreset=True)

def print_header():
    """Stampa l'intestazione del programma"""
    print("\n" + "="*80)
    print(f"{Fore.CYAN}  PIPELINE DI ANALISI DELLA VOLATILITÀ - CONFIGURAZIONE DEFAULT{Style.RESET_ALL}")
    print(f"  • Top {Fore.YELLOW}200{Style.RESET_ALL} criptovalute")
    print(f"  • {Fore.YELLOW}150{Style.RESET_ALL} giorni di dati storici")
    print(f"  • Timeframes: {Fore.GREEN}5m, 15m{Style.RESET_ALL}")
    print("="*80 + "\n")

def run_volatility_pipeline():
    """Esegue il pipeline di analisi della volatilità con i parametri di default"""
    print_header()
    
    # Crea il percorso al file pipeline
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline_script = os.path.join(script_dir, "src", "scripts", "volatility_pipeline.py")
    
    # Verifica che il file esista
    if not os.path.exists(pipeline_script):
        print(f"{Fore.RED}Errore: File non trovato: {pipeline_script}{Style.RESET_ALL}")
        return False
    
    # Parametri di default
    top_n = 200
    days = 150
    timeframes = "5m,15m"
    
    print(f"{Fore.GREEN}Avvio del pipeline con i seguenti parametri:{Style.RESET_ALL}")
    print(f"  • Top {Fore.YELLOW}{top_n}{Style.RESET_ALL} criptovalute")
    print(f"  • {Fore.YELLOW}{days}{Style.RESET_ALL} giorni di dati")
    print(f"  • Timeframes: {Fore.GREEN}{timeframes}{Style.RESET_ALL}")
    print()
    
    # Comando da eseguire
    cmd = [
        sys.executable,
        pipeline_script,
        "--top", str(top_n),
        "--days", str(days),
        "--timeframes", timeframes
    ]
    
    # Esegui il comando
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"{Fore.RED}Errore nell'esecuzione del pipeline: {e}{Style.RESET_ALL}")
        return False
    except FileNotFoundError:
        print(f"{Fore.RED}Errore: Python non trovato. Assicurati che Python sia installato e nel PATH.{Style.RESET_ALL}")
        return False

if __name__ == "__main__":
    # Fix event loop policy per Windows
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Esegui il pipeline
    success = run_volatility_pipeline()
    
    if success:
        print(f"\n{Fore.GREEN}Pipeline di analisi della volatilità completato con successo!{Style.RESET_ALL}")
    else:
        print(f"\n{Fore.RED}Pipeline di analisi della volatilità terminato con errori.{Style.RESET_ALL}")
