#!/usr/bin/env python3
"""
Real-Time Crypto Data Collector
===============================

Sistema di raccolta dati crypto real-time che integra:
1. Download continuo dati OHLCV
2. Calcolo indicatori tecnici
3. Calcolo volatilità
4. Validazione e riparazione dati
5. Salvataggio persistente su database SQLite

Caratteristiche:
- Monitoraggio continuo multi-timeframe
- Calcolo automatico indicatori tecnici
- Validazione qualità dati con riparazione automatica
- Logging strutturato e dettagliato
- Supporto modalità sequenziale e parallela
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
from modules.core.exchange import create_exchange, fetch_markets, get_top_symbols
from modules.core.download_orchestrator import process_timeframe
from modules.data.db_manager import init_data_tables
from modules.data.volatility_processor import process_and_save_volatility
from modules.data.indicator_processor import init_indicator_tables, process_and_save_indicators
from modules.data.data_integrity_checker import (
    get_all_symbols_integrity_status, log_integrity_summary
)
from modules.data.data_validator import (
    validate_and_repair_data, log_validation_results, log_validation_summary,
    export_validation_report_csv, generate_validation_charts, DataQualityReport
)


# Inizializza colorama
init(autoreset=True)

# Imposta la policy di event loop per Windows se necessario
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Intervallo di aggiornamento in secondi (default: 5 minuti)
UPDATE_INTERVAL = REALTIME_CONFIG['update_interval_seconds']

async def real_time_update(args):
    """
    Aggiornamento dati real-time con elaborazione completa.
    
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
        # Ottieni i primi N simboli per volume
        async_exchange = await create_exchange()
        markets = await fetch_markets(async_exchange)
        
        if not markets:
            logging.error("Nessun mercato trovato. Controlla la tua connessione internet e le credenziali API.")
            return None

        all_symbols = list(markets.keys())
        top_symbols = await get_top_symbols(async_exchange, all_symbols, top_n=args.num_symbols)
        await async_exchange.close()

        if not top_symbols:
            logging.error("Impossibile ottenere i simboli con maggior volume. Utilizzo di tutti i simboli disponibili.")
            top_symbols = all_symbols[:args.num_symbols]
        
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
                
                # Elabora la volatilità per i simboli completati
                if results["completati"] > 0:
                    for sym in top_symbols:
                        # Calcola e salva la volatilità per ogni simbolo
                        process_and_save_volatility(sym, timeframe)
                        
                        # Calcola e salva gli indicatori tecnici se non è specificato --no-ta
                        if not args.no_ta:
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
        
        # VALIDAZIONE DATI POST-DOWNLOAD (se non saltata)
        # VERIFICA INTEGRITÀ DATI POST-DOWNLOAD
        integrity_results = get_all_symbols_integrity_status(args.timeframes)
        if integrity_results:
            log_integrity_summary(integrity_results)
        
        validation_reports = []
        validation_summary = {
            "symbols_validated": 0,
            "issues_found": 0,
            "auto_repaired": 0,
            "high_quality": 0,
            "medium_quality": 0,
            "low_quality": 0
        }
        
        if not args.skip_validation:
            # Valida dati per tutti i simboli e timeframe dove abbiamo scaricato dati
            for tf, res in all_timeframe_results.items():
                if res["completati"] > 0:
                    for sym in top_symbols:
                        try:
                            validation_report = await validate_and_repair_data(sym, tf)
                            validation_summary["symbols_validated"] += 1
                            
                            if validation_report:  # Issues found
                                validation_summary["issues_found"] += len(validation_report.issues)
                                log_validation_results(sym, tf, validation_report)
                                validation_reports.append(validation_report)
                                
                                # Categorize by quality score
                                if validation_report.score >= 95:
                                    validation_summary["high_quality"] += 1
                                elif validation_report.score >= 85:
                                    validation_summary["medium_quality"] += 1
                                else:
                                    validation_summary["low_quality"] += 1
                                
                                if validation_report.can_repair:
                                    validation_summary["auto_repaired"] += 1
                            else:
                                # No issues found - high quality - create clean report for export
                                clean_report = DataQualityReport(sym, tf)
                                clean_report.score = 100
                                clean_report.total_records = 1  # Placeholder
                                validation_reports.append(clean_report)
                                validation_summary["high_quality"] += 1
                                
                        except Exception as e:
                            logging.error(f"Validation error for {sym} ({tf}): {e}")
            
            # Log overall validation summary
            log_validation_summary(validation_summary)
            
            # EXPORT REPORT CSV (se richiesto)
            if args.export_validation_report and validation_reports:
                try:
                    export_validation_report_csv(validation_reports, validation_summary)
                except Exception as e:
                    logging.error(f"Error exporting validation report: {e}")
            
            # GENERA GRAFICI (se richiesto)
            if args.generate_validation_charts and validation_reports:
                try:
                    generate_validation_charts(validation_reports)
                except Exception as e:
                    logging.error(f"Error generating validation charts: {e}")
        else:
            logging.info(f"{Fore.YELLOW}🔍 Data validation skipped (--skip-validation flag used){Style.RESET_ALL}")
        
        # Elabora la volatilità per i simboli completati in tutti i timeframe
        for tf, res in all_timeframe_results.items():
            if res["completati"] > 0:
                for sym in top_symbols:
                    # Calcola e salva la volatilità per ogni simbolo
                    process_and_save_volatility(sym, tf)
                    
                    # Calcola e salva gli indicatori tecnici se non è specificato --no-ta
                    if not args.no_ta:
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
            "validation_summary": validation_summary,
            "start_time": start_time,
            "end_time": end_time,
            "execution_time": time_str
        }

    except Exception as e:
        logging.error(f"Errore durante l'aggiornamento: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None

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
    validation_summary = results["validation_summary"]
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
    
    # Validazione summary
    if validation_summary["symbols_validated"] > 0:
        print(f"\n{Back.CYAN}{Fore.BLACK}  VALIDAZIONE DATI  {Style.RESET_ALL}")
        print(f"  • Simboli validati: {Fore.GREEN}{validation_summary['symbols_validated']}{Style.RESET_ALL}")
        print(f"  • Problemi trovati: {Fore.YELLOW}{validation_summary['issues_found']}{Style.RESET_ALL}")
        print(f"  • Riparazioni automatiche: {Fore.GREEN}{validation_summary['auto_repaired']}{Style.RESET_ALL}")
        print(f"  • Qualità alta: {Fore.GREEN}{validation_summary['high_quality']}{Style.RESET_ALL}")
        print(f"  • Qualità media: {Fore.YELLOW}{validation_summary['medium_quality']}{Style.RESET_ALL}")
        print(f"  • Qualità bassa: {Fore.RED}{validation_summary['low_quality']}{Style.RESET_ALL}")
    
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
    mode = "SEQUENZIALE" if args.sequential else "PARALLELA"
    
    # Header generale
    print("\n" + "="*80)
    print(f"{Back.BLUE}{Fore.WHITE}  CRYPTO DATA COLLECTOR (MODALITÀ {mode})  {Style.RESET_ALL}")
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
    
    # Status validazione
    validation_status = "Disabilitata (--skip-validation)" if args.skip_validation else "Abilitata"
    validation_color = Fore.RED if args.skip_validation else Fore.GREEN
    print(f"  • Validazione dati: {validation_color}{validation_status}{Style.RESET_ALL}")
    
    print(f"  • Data e ora inizio: {Fore.CYAN}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}")
    print("="*80 + "\n")

    # Inizializza il database
    init_data_tables(args.timeframes)
    logging.info(f"Database inizializzato con tabelle per i timeframe: {Fore.GREEN}{', '.join(args.timeframes)}{Style.RESET_ALL}")
    
    # Inizializza le tabelle degli indicatori tecnici se non è specificato --no-ta
    if not args.no_ta:
        init_indicator_tables(args.timeframes)
        logging.info(f"Tabelle degli indicatori tecnici inizializzate per i timeframe: {Fore.GREEN}{', '.join(args.timeframes)}{Style.RESET_ALL}")
    else:
        logging.info(f"{Fore.YELLOW}Calcolo degli indicatori tecnici disabilitato (--no-ta){Style.RESET_ALL}")

    # Loop infinito per aggiornamenti continui
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
