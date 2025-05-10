#!/usr/bin/env python3
"""
Cryptocurrency Data Fetcher
==========================

Questo script scarica i dati OHLCV delle criptovalute con maggior volume da Bybit.
Controlla se i dati sono aggiornati prima di scaricarli e scarica solo i dati necessari.

Timeframes: 5m, 15m, 30m, 1h
Periodo: 100 giorni di dati storici
"""

import sys
import os
import asyncio
import logging
from datetime import datetime, timedelta
from colorama import Fore, Style, Back, init

# Moduli personalizzati
from modules.utils.logging_setup import setup_logging
from modules.utils.command_args import parse_arguments
from modules.utils.config import DB_FILE
from modules.core.exchange import create_exchange, fetch_markets, get_top_symbols
from modules.core.download_orchestrator import process_timeframe
from modules.data.db_manager import init_data_tables

# Inizializza colorama
init(autoreset=True)

# Imposta la policy di event loop per Windows se necessario
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

async def main():
    """Punto di ingresso principale per il recupero dati delle criptovalute."""
    # Registra il tempo di inizio
    start_time = datetime.now()
    
    # Configura il logger
    logger = setup_logging(level=logging.INFO)
    
    # Analizza gli argomenti da linea di comando
    args = parse_arguments()
    mode = "SEQUENZIALE" if args.sequential else "PARALLELA"
    
    # Header generale
    print("\n" + "="*80)
    print(f"{Back.BLUE}{Fore.WHITE}  RECUPERO DATI OHLCV CRIPTOVALUTE (MODALITÀ {mode})  {Style.RESET_ALL}")
    print("="*80)
    print(f"  • Criptovalute da elaborare: {Fore.YELLOW}{args.num_symbols}{Style.RESET_ALL}")
    print(f"  • Periodo storico: {Fore.GREEN}{args.days} giorni{Style.RESET_ALL}")
    print(f"  • Timeframes richiesti: {Fore.GREEN}{', '.join(args.timeframes)}{Style.RESET_ALL}")
    print(f"  • Batch size: {Fore.YELLOW}{args.batch_size}{Style.RESET_ALL}")
    if not args.sequential:
        print(f"  • Concorrenza: {Fore.YELLOW}{args.concurrency}{Style.RESET_ALL} download paralleli per batch")
    print(f"  • Data e ora inizio: {Fore.CYAN}{start_time.strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}")
    print("="*80 + "\n")

    # Inizializza il database
    init_data_tables(args.timeframes)
    logging.info(f"Database inizializzato con tabelle per i timeframe: {Fore.GREEN}{', '.join(args.timeframes)}{Style.RESET_ALL}")

    # Variabili per raccogliere statistiche complessive
    all_timeframe_results = {}
    total_records_saved = 0
    grand_total_symbols = {"completati": 0, "saltati": 0, "falliti": 0}
    
    try:
        # Fase 1: Recupero simboli
        print(f"\n{Back.BLUE}{Fore.WHITE}  FASE 1: RICERCA CRIPTOVALUTE  {Style.RESET_ALL}")
        async_exchange = await create_exchange()
        markets = await fetch_markets(async_exchange)
        
        if not markets:
            logging.error("Nessun mercato trovato. Controlla la tua connessione internet e le credenziali API.")
            return

        all_symbols = list(markets.keys())
        top_symbols = await get_top_symbols(async_exchange, all_symbols, top_n=args.num_symbols)
        await async_exchange.close()

        if not top_symbols:
            logging.error("Impossibile ottenere i simboli con maggior volume. Utilizzo di tutti i simboli disponibili.")
            top_symbols = all_symbols[:args.num_symbols]
        
        # Fase 2: Download per tutti i timeframe
        print(f"\n{Back.BLUE}{Fore.WHITE}  FASE 2: DOWNLOAD DATI OHLCV  {Style.RESET_ALL}")
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

        # Calcola il tempo totale di esecuzione
        end_time = datetime.now()
        execution_time = end_time - start_time
        hours, remainder = divmod(execution_time.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

        # Resoconto finale
        print("\n" + "="*80)
        print(f"{Back.GREEN}{Fore.BLACK}  RESOCONTO FINALE RACCOLTA DATI  {Style.RESET_ALL}")
        print("="*80)
        print(f"  • Criptovalute elaborate: {Fore.YELLOW}{len(top_symbols)}{Style.RESET_ALL}")
        print(f"  • Database: {Fore.BLUE}{os.path.abspath(DB_FILE)}{Style.RESET_ALL}")
        print(f"  • Tempo totale esecuzione: {Fore.CYAN}{time_str}{Style.RESET_ALL}")
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
        average_per_timeframe = round(grand_total_symbols["completati"] / len(args.timeframes)) if args.timeframes else 0
        print(f"{'TOTALE':^10} | " + 
              f"{Fore.GREEN}{grand_total_symbols['completati']:^12}{Style.RESET_ALL} | " + 
              f"{Fore.BLUE}{grand_total_symbols['saltati']:^12}{Style.RESET_ALL} | " + 
              f"{Fore.RED}{grand_total_symbols['falliti']:^12}{Style.RESET_ALL} | " + 
              f"{Fore.YELLOW}{total_records_saved:^15,}{Style.RESET_ALL}")
        print()
        
        # Informazioni aggiuntive
        print(f"Media simboli completati per timeframe: {Fore.GREEN}{average_per_timeframe}{Style.RESET_ALL}")
        print(f"Record medi per simbolo: {Fore.YELLOW}{round(total_records_saved/max(1, grand_total_symbols['completati'])):,}{Style.RESET_ALL}")
        print(f"Inizio: {Fore.CYAN}{start_time.strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}")
        print(f"Fine:   {Fore.CYAN}{end_time.strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}")
        print("="*80 + "\n")

    except Exception as e:
        logging.error(f"Errore: {e}")
        import traceback
        logging.error(traceback.format_exc())
    finally:
        logging.info("Recupero dati completato.")

if __name__ == "__main__":
    asyncio.run(main())
