#!/usr/bin/env python3
"""
Simplified Crypto Data Collector
===============================

Sistema di raccolta dati crypto semplificato che integra:
1. Download continuo dati OHLCV
2. Calcolo indicatori tecnici
3. Salvataggio persistente su database SQLite

Caratteristiche:
- Monitoraggio continuo multi-timeframe
- Calcolo automatico indicatori tecnici
- Logging strutturato
"""

import sys
import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from colorama import Fore, Style, Back, init

# Moduli personalizzati
from modules.utils.logging_setup import setup_logging
from modules.utils.command_args import parse_arguments
from modules.utils.config import DB_FILE, REALTIME_CONFIG
from modules.core.exchange import create_exchange
from modules.utils.symbol_manager import get_top_symbols
from modules.core.download_orchestrator import process_timeframe
from modules.data.db_manager import init_market_data_tables, get_timestamp_range, cleanup_old_data, delete_warmup_data
from modules.data.indicator_processor import init_indicator_tables, process_and_save_indicators
import sqlite3

# Inizializza colorama
init(autoreset=True)

# Imposta la policy di event loop per Windows se necessario
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Intervallo di aggiornamento in secondi (default: 5 minuti)
UPDATE_INTERVAL = REALTIME_CONFIG['update_interval_seconds']

async def real_time_update(args):
    """
    Aggiornamento dati real-time semplificato.
    
    Args:
        args: Command line arguments
        
    Returns:
        Dictionary with update results
    """
    start_time = datetime.now()
    
    # Variabili per raccogliere statistiche complessive
    all_timeframe_results = {}
    total_records_saved = 0
    grand_total_symbols = {"completati": 0, "saltati": 0, "falliti": 0}
    
    try:
        # Ottieni i simboli da monitorare (specifici o top per volume)
        async_exchange = await create_exchange()

        # Usa simboli specifici se configurato, altrimenti usa top per volume
        if REALTIME_CONFIG['use_specific_symbols']:
            top_symbols = REALTIME_CONFIG['specific_symbols']
            logging.info(f"Utilizzo simboli specifici configurati: {Fore.GREEN}{', '.join(top_symbols)}{Style.RESET_ALL}")
        else:
            top_symbols = await get_top_symbols(async_exchange, limit=args.num_symbols)
            if not top_symbols:
                logging.error("Impossibile ottenere i simboli con maggior volume.")
                return None
        
        await async_exchange.close()
        
        # Aggiorna i dati per ogni timeframe
        if args.sequential:
            logging.info(f"{Fore.YELLOW}Modalità sequenziale attivata. Elaborazione timeframe uno alla volta.{Style.RESET_ALL}")
            for timeframe in args.timeframes:
                timeframe, results = await process_timeframe(timeframe, top_symbols, args.days, 
                                                         args.concurrency, args.batch_size, True)
                all_timeframe_results[timeframe] = results
                # Aggiorna statistiche totali
                grand_total_symbols["completati"] += results["completati"]
                grand_total_symbols["saltati"] += results["saltati"]
                grand_total_symbols["falliti"] += results["falliti"]
                total_records_saved += results["record_totali"]
                
                # Calcola e salva gli indicatori tecnici se non è specificato --no-ta
                if results["completati"] > 0 and not args.no_ta:
                    for sym in top_symbols:
                        process_and_save_indicators(sym, timeframe)
        else:
            logging.info(f"{Fore.YELLOW}Modalità parallela attivata. Concorrenza massima per simbolo: {args.concurrency}{Style.RESET_ALL}")
            timeframe_tasks = []
            for timeframe in args.timeframes:
                timeframe_tasks.append(process_timeframe(timeframe, top_symbols, args.days, 
                                                       args.concurrency, args.batch_size))
            
            # Esegui tutti i timeframe in parallelo e raccoglie risultati
            results = await asyncio.gather(*timeframe_tasks)
            for tf, res in results:
                all_timeframe_results[tf] = res
                # Aggiorna statistiche totali
                grand_total_symbols["completati"] += res["completati"]
                grand_total_symbols["saltati"] += res["saltati"]
                grand_total_symbols["falliti"] += res["falliti"]
                total_records_saved += res["record_totali"]
        
        # Calcola gli indicatori tecnici per tutti i timeframe in parallelo
        if not args.no_ta:
            for tf, res in all_timeframe_results.items():
                if res["completati"] > 0:
                    for sym in top_symbols:
                        process_and_save_indicators(sym, tf)
        
        # Calcola il tempo totale di esecuzione
        end_time = datetime.now()
        execution_time = end_time - start_time
        hours, remainder = divmod(execution_time.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

        return {
            "all_timeframe_results": all_timeframe_results,
            "total_records_saved": total_records_saved,
            "grand_total_symbols": grand_total_symbols,
            "start_time": start_time,
            "end_time": end_time,
            "execution_time": time_str
        }

    except Exception as e:
        logging.error(f"Errore durante l'aggiornamento: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None

def display_days_saved(timeframes=None, symbol=None):
    """
    Visualizza il numero di giorni salvati nel database per ogni simbolo e timeframe.
    
    Args:
        timeframes: Lista dei timeframe da controllare (opzionale)
        symbol: Simbolo specifico da controllare (opzionale)
    """
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            
            # Ottieni tutti i timeframe disponibili se non specificati
            if not timeframes:
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name LIKE 'market_data_%'
                """)
                available_tables = cursor.fetchall()
                timeframes = [table[0].replace('market_data_', '') for table in available_tables]
            
            print("\n" + "="*80)
            print(f"{Back.BLUE}{Fore.WHITE}  GIORNI DI DATI SALVATI NEL DATABASE  {Style.RESET_ALL}")
            print("="*80)
            
            print(f"{'Simbolo':^15} | {'Timeframe':^10} | {'Primo Giorno':^20} | {'Ultimo Giorno':^20} | {'Giorni Totali':^12} | {'Candle':^10}")
            print("-" * 100)
            
            total_results = 0
            
            for tf in sorted(timeframes):
                table_name = f"market_data_{tf}"
                
                # Ottieni simboli per questo timeframe
                if symbol:
                    symbols = [symbol]
                else:
                    cursor.execute(f"SELECT DISTINCT symbol FROM {table_name}")
                    symbols = [row[0] for row in cursor.fetchall()]
                
                for sym in sorted(symbols):
                    # Ottieni prima e ultima data
                    cursor.execute(f"""
                        SELECT MIN(timestamp), MAX(timestamp), COUNT(DISTINCT date(timestamp)) 
                        FROM {table_name}
                        WHERE symbol = ?
                    """, (sym,))
                    
                    row = cursor.fetchone()
                    if row and row[0] and row[1]:
                        first_date = row[0][:10]  # Solo la parte della data
                        last_date = row[1][:10]   # Solo la parte della data
                        days_count = row[2]
                        
                        # Conta il numero di candle
                        cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE symbol = ?", (sym,))
                        candle_count = cursor.fetchone()[0]
                        
                        print(f"{Fore.YELLOW}{sym:^15}{Style.RESET_ALL} | " + 
                              f"{Fore.CYAN}{tf:^10}{Style.RESET_ALL} | " + 
                              f"{Fore.GREEN}{first_date:^20}{Style.RESET_ALL} | " + 
                              f"{Fore.GREEN}{last_date:^20}{Style.RESET_ALL} | " + 
                              f"{Fore.MAGENTA}{days_count:^12}{Style.RESET_ALL} | " +
                              f"{Fore.BLUE}{candle_count:^10,}{Style.RESET_ALL}")
                        
                        total_results += 1
            
            print("-" * 100)
            print(f"\nTotale risultati: {Fore.GREEN}{total_results}{Style.RESET_ALL}")
            print("="*80 + "\n")
            
    except Exception as e:
        logging.error(f"Errore nel visualizzare i giorni salvati: {e}")
        import traceback
        logging.error(traceback.format_exc())

def display_results(results):
    """
    Display results of data collection and processing.
    
    Args:
        results: Update results dictionary
    """
    if not results:
        return
        
    all_timeframe_results = results["all_timeframe_results"]
    total_records_saved = results["total_records_saved"]
    grand_total_symbols = results["grand_total_symbols"]
    start_time = results["start_time"]
    end_time = results["end_time"]
    time_str = results["execution_time"]
    
    # Resoconto finale dell'aggiornamento
    print("\n" + "="*80)
    print(f"{Back.GREEN}{Fore.BLACK}  RESOCONTO AGGIORNAMENTO DATI COMPLETATO  {Style.RESET_ALL}")
    print("="*80)
    print(f"  • Database: {Fore.BLUE}{os.path.abspath(DB_FILE)}{Style.RESET_ALL}")
    print(f"  • Tempo esecuzione: {Fore.CYAN}{time_str}{Style.RESET_ALL}")
    print()
    
    # Tabella dei risultati per timeframe
    print(f"{Back.WHITE}{Fore.BLACK}  STATISTICHE PER TIMEFRAME  {Style.RESET_ALL}")
    print(f"{'Timeframe':^10} | {'Completati':^12} | {'Saltati':^12} | {'Falliti':^12} | {'Record Salvati':^15}")
    print("-" * 70)
    for tf, res in sorted(all_timeframe_results.items()):
        print(f"{Fore.CYAN}{tf:^10}{Style.RESET_ALL} | " + 
              f"{Fore.GREEN}{res['completati']:^12}{Style.RESET_ALL} | " + 
              f"{Fore.BLUE}{res['saltati']:^12}{Style.RESET_ALL} | " + 
              f"{Fore.RED}{res['falliti']:^12}{Style.RESET_ALL} | " + 
              f"{Fore.YELLOW}{res['record_totali']:^15,}{Style.RESET_ALL}")
    
    # Totali
    print("-" * 70)
    print(f"{'TOTALE':^10} | " + 
          f"{Fore.GREEN}{grand_total_symbols['completati']:^12}{Style.RESET_ALL} | " + 
          f"{Fore.BLUE}{grand_total_symbols['saltati']:^12}{Style.RESET_ALL} | " + 
          f"{Fore.RED}{grand_total_symbols['falliti']:^12}{Style.RESET_ALL} | " + 
          f"{Fore.YELLOW}{total_records_saved:^15,}{Style.RESET_ALL}")
    
    print(f"\nInizio: {Fore.CYAN}{start_time.strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}")
    print(f"Fine:   {Fore.CYAN}{end_time.strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}")
    print("="*80 + "\n")

async def main():
    """
    Main entry point for crypto data collector.
    """
    # Configura il logger
    logger = setup_logging(level=logging.INFO)
    
    # Analizza gli argomenti da linea di comando
    args = parse_arguments()
    
    # Mostra i giorni salvati se richiesto
    if hasattr(args, 'show_days') and args.show_days:
        display_days_saved(args.timeframes, getattr(args, 'symbol', None))
        return
        
    mode = "SEQUENZIALE" if args.sequential else "PARALLELA"
    
    # Header generale
    print("\n" + "="*80)
    print(f"{Back.BLUE}{Fore.WHITE}  CRYPTO DATA COLLECTOR SEMPLIFICATO (MODALITÀ {mode})  {Style.RESET_ALL}")
    print("="*80)
    print(f"  • Criptovalute da monitorare: {Fore.YELLOW}{args.num_symbols}{Style.RESET_ALL}")
    print(f"  • Timeframes monitorati: {Fore.GREEN}{', '.join(args.timeframes)}{Style.RESET_ALL}")
    print(f"  • Intervallo aggiornamento: {Fore.CYAN}{UPDATE_INTERVAL} secondi{Style.RESET_ALL}")
    print(f"  • Batch size: {Fore.YELLOW}{args.batch_size}{Style.RESET_ALL}")
    if not args.sequential:
        print(f"  • Concorrenza: {Fore.YELLOW}{args.concurrency}{Style.RESET_ALL} download paralleli per batch")
    
    print(f"  • Database output: {Fore.BLUE}{os.path.abspath(DB_FILE)}{Style.RESET_ALL}")
    
    # Status indicatori tecnici
    ta_status = "Disabilitato (--no-ta)" if args.no_ta else "Abilitato"
    ta_color = Fore.RED if args.no_ta else Fore.GREEN
    print(f"  • Indicatori tecnici: {ta_color}{ta_status}{Style.RESET_ALL}")
    
    print(f"  • Data e ora inizio: {Fore.CYAN}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}")
    print("="*80 + "\n")

    # Inizializza il database
    init_market_data_tables(args.timeframes)
    logging.info(f"Database inizializzato con tabelle per i timeframe: {Fore.GREEN}{', '.join(args.timeframes)}{Style.RESET_ALL}")
    
    # Pulisci i dati precedenti al 1° gennaio 2024
    cleanup_old_data()
    
    # Inizializza le tabelle degli indicatori tecnici se non è specificato --no-ta
    if not args.no_ta:
        init_indicator_tables(args.timeframes)
        logging.info(f"Tabelle degli indicatori tecnici inizializzate per i timeframe: {Fore.GREEN}{', '.join(args.timeframes)}{Style.RESET_ALL}")
        
        # Ricalcola gli indicatori tecnici per tutti i dati esistenti nel database
        logging.info(f"{Fore.CYAN}Ricalcolo di tutti gli indicatori tecnici per i dati esistenti...{Style.RESET_ALL}")
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            for tf in args.timeframes:
                table_name = f"market_data_{tf}"
                cursor.execute(f"SELECT DISTINCT symbol FROM {table_name}")
                symbols = [row[0] for row in cursor.fetchall()]
                
                if symbols:
                    logging.info(f"Ricalcolo indicatori per timeframe {Fore.GREEN}{tf}{Style.RESET_ALL} - {len(symbols)} simboli")
                    for sym in symbols:
                        process_and_save_indicators(sym, tf)
                        logging.info(f"Indicatori aggiornati per {Fore.YELLOW}{sym}{Style.RESET_ALL} ({tf})")
        logging.info(f"{Fore.GREEN}Ricalcolo indicatori completato!{Style.RESET_ALL}")
    else:
        logging.info(f"{Fore.YELLOW}Calcolo degli indicatori tecnici disabilitato (--no-ta){Style.RESET_ALL}")

    # Loop infinito per aggiornamenti continui (se non è single-run)
    iteration = 1
    
    try:
        while True:
            current_time = datetime.now()
            print(f"\n{Back.MAGENTA}{Fore.WHITE} INIZIO CICLO #{iteration} - DATA COLLECTION - {current_time.strftime('%Y-%m-%d %H:%M:%S')} {Style.RESET_ALL}")
            
            # Esegui l'aggiornamento dei dati
            results = await real_time_update(args)
            
            # Visualizza i risultati
            if results:
                display_results(results)
                
            # Elimina i dati di warmup dopo aver completato il ciclo
            delete_warmup_data()
            
            # Se è single-run, esci dopo il primo ciclo
            if args.single_run:
                logging.info(f"{Fore.GREEN}Singolo ciclo completato. Terminazione del data collector.{Style.RESET_ALL}")
                print(f"\n{Fore.GREEN}✓ Aggiornamento dati completato con successo!{Style.RESET_ALL}")
                break
            
            # Calcola il prossimo orario di aggiornamento
            next_update = datetime.now() + timedelta(seconds=UPDATE_INTERVAL)
            logging.info(f"Ciclo #{iteration} completato. Prossimo aggiornamento alle {Fore.CYAN}{next_update.strftime('%H:%M:%S')}{Style.RESET_ALL}")
            
            # Attendi il prossimo ciclo di aggiornamento
            print(f"{Fore.YELLOW}In attesa del prossimo ciclo di aggiornamento tra {UPDATE_INTERVAL} secondi...{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Premi Ctrl+C per interrompere il data collector.{Style.RESET_ALL}")
            
            await asyncio.sleep(UPDATE_INTERVAL)
            iteration += 1
            
    except KeyboardInterrupt:
        print(f"\n{Fore.RED}Data collector interrotto manualmente dall'utente.{Style.RESET_ALL}")
    except Exception as e:
        logging.error(f"Errore nel loop principale: {e}")
        import traceback
        logging.error(traceback.format_exc())
    finally:
        logging.info("Data collector terminato.")

if __name__ == "__main__":
    asyncio.run(main())
