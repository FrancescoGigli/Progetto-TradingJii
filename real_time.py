#!/usr/bin/env python3
"""
Cryptocurrency Real-Time Data Fetcher
====================================

Questo script esegue continuamente il download di dati OHLCV delle criptovalute da Bybit
in modalità tempo reale, eseguendo un'iterazione ogni 5 minuti.

Mantiene il database costantemente aggiornato con i dati più recenti e genera
dataset per il machine learning supervisionato a partire dalla volatilità calcolata.
"""

import sys
import os
import asyncio
import logging
import time
from datetime import datetime, timedelta
from colorama import Fore, Style, Back, init

# Moduli personalizzati
from modules.utils.logging_setup import setup_logging
from modules.utils.command_args import parse_arguments
from modules.utils.config import DB_FILE, TIMEFRAME_CONFIG
from modules.core.exchange import create_exchange, fetch_markets, get_top_symbols
from modules.core.download_orchestrator import process_timeframe
from modules.data.db_manager import init_data_tables
from modules.data.volatility_processor import process_and_save_volatility
from modules.data.indicator_processor import init_indicator_tables, compute_and_save_indicators
from modules.data.dataset_generator import export_supervised_training_data, generate_ml_dataset

# Inizializza colorama
init(autoreset=True)

# Imposta la policy di event loop per Windows se necessario
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Intervallo di aggiornamento in secondi (default: 5 minuti)
UPDATE_INTERVAL = 5 * 60

async def real_time_update(args):
    """
    Esegue un singolo ciclo di aggiornamento dati.
    
    Args:
        args: Argomenti della linea di comando
        
    Returns:
        Dizionario con i risultati dell'aggiornamento
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
                            await compute_and_save_indicators(sym, timeframe)
                        
                        # Genera dataset per il machine learning supervisionato
                        logging.info(f"Generazione dataset di ML per {Fore.YELLOW}{sym}{Style.RESET_ALL} ({timeframe})")
                        try:
                            output_dir = "datasets"
                            pattern_counts = export_supervised_training_data(
                                symbol=sym,
                                timeframe=timeframe,
                                output_dir=output_dir,
                                window_size=7,  # default: finestra di 7 valori
                                threshold=0.0   # default: soglia a 0.0
                            )
                            
                            # Log dei risultati
                            if pattern_counts:
                                total_patterns = len(pattern_counts)
                                total_records = sum(pattern_counts.values())
                                logging.info(f"Dataset generato: {Fore.GREEN}{total_records}{Style.RESET_ALL} record in "
                                           f"{Fore.CYAN}{total_patterns}{Style.RESET_ALL} categorie per {Fore.YELLOW}{sym}{Style.RESET_ALL}")
                            else:
                                logging.warning(f"Nessun dataset generato per {sym} ({timeframe})")
                        except Exception as e:
                            logging.error(f"Errore nella generazione del dataset per {sym} ({timeframe}): {e}")
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
        
        # Elabora la volatilità per i simboli completati in tutti i timeframe
        for tf, res in all_timeframe_results.items():
            if res["completati"] > 0:
                for sym in top_symbols:
                    # Calcola e salva la volatilità per ogni simbolo
                    process_and_save_volatility(sym, tf)
                    
                    # Calcola e salva gli indicatori tecnici se non è specificato --no-ta
                    if not args.no_ta:
                        await compute_and_save_indicators(sym, tf)
                    
                    # Genera dataset per il machine learning supervisionato
                    logging.info(f"Generazione dataset di ML per {Fore.YELLOW}{sym}{Style.RESET_ALL} ({tf})")
                    try:
                        output_dir = "datasets"
                        pattern_counts = export_supervised_training_data(
                            symbol=sym,
                            timeframe=tf,
                            output_dir=output_dir,
                            window_size=7,  # default: finestra di 7 valori
                            threshold=0.0   # default: soglia a 0.0
                        )
                        
                        # Log dei risultati
                        if pattern_counts:
                            total_patterns = len(pattern_counts)
                            total_records = sum(pattern_counts.values())
                            logging.info(f"Dataset generato: {Fore.GREEN}{total_records}{Style.RESET_ALL} record in "
                                       f"{Fore.CYAN}{total_patterns}{Style.RESET_ALL} categorie per {Fore.YELLOW}{sym}{Style.RESET_ALL}")
                        else:
                            logging.warning(f"Nessun dataset generato per {sym} ({tf})")
                    except Exception as e:
                        logging.error(f"Errore nella generazione del dataset per {sym} ({tf}): {e}")
        
        # Generate ML dataset if request with --ml flag
        # Check if any new volatility data was saved or if force regeneration is requested
        has_new_data = any(res["completati"] > 0 for _, res in all_timeframe_results.items())
        if args.ml and (has_new_data or args.force_ml):
            if has_new_data:
                logging.info(f"{Fore.YELLOW}Generating ML dataset with new volatility data{Style.RESET_ALL}")
            elif args.force_ml:
                logging.info(f"{Fore.YELLOW}Forcing ML dataset regeneration as requested with --force-ml{Style.RESET_ALL}")
            
            # Define output directory
            your_output_dir = "datasets"
            
            # Generate ML dataset
            from modules.data.dataset_generator import generate_ml_dataset

            generate_ml_dataset(
                db_path=DB_FILE,
                output_dir=your_output_dir,
                symbols=top_symbols,
                timeframes=args.timeframes,
                segment_len=7,
                force_regeneration=args.force_ml
            )
        elif args.ml:
            logging.info(f"{Fore.YELLOW}No new volatility data found, skipping ML dataset generation. Use --force-ml to regenerate.{Style.RESET_ALL}")

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

def display_update_results(results):
    """
    Visualizza i risultati di un aggiornamento.
    
    Args:
        results: Dizionario con i risultati dell'aggiornamento
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
    print(f"{Back.GREEN}{Fore.BLACK}  RESOCONTO AGGIORNAMENTO COMPLETATO  {Style.RESET_ALL}")
    print("="*80)
    print(f"  • Database: {Fore.BLUE}{os.path.abspath(DB_FILE)}{Style.RESET_ALL}")
    print(f"  • Dataset ML: {Fore.BLUE}{os.path.abspath('datasets')}{Style.RESET_ALL}")
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
    print(f"Inizio: {Fore.CYAN}{start_time.strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}")
    print(f"Fine:   {Fore.CYAN}{end_time.strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}")
    print("="*80 + "\n")

async def main():
    """
    Punto di ingresso principale per il monitoraggio dati criptovalute in tempo reale.
    Esegue un loop infinito con aggiornamento ogni 5 minuti.
    """
    # Configura il logger
    logger = setup_logging(level=logging.INFO)
    
    # Analizza gli argomenti da linea di comando
    args = parse_arguments()
    mode = "SEQUENZIALE" if args.sequential else "PARALLELA"
    
    # Header generale
    print("\n" + "="*80)
    print(f"{Back.BLUE}{Fore.WHITE}  MONITOR DATI E GENERATORE DATASET ML (MODALITÀ {mode})  {Style.RESET_ALL}")
    print("="*80)
    print(f"  • Criptovalute da monitorare: {Fore.YELLOW}{args.num_symbols}{Style.RESET_ALL}")
    print(f"  • Timeframes monitorati: {Fore.GREEN}{', '.join(args.timeframes)}{Style.RESET_ALL}")
    print(f"  • Intervallo aggiornamento: {Fore.CYAN}{UPDATE_INTERVAL} secondi{Style.RESET_ALL}")
    print(f"  • Batch size: {Fore.YELLOW}{args.batch_size}{Style.RESET_ALL}")
    if not args.sequential:
        print(f"  • Concorrenza: {Fore.YELLOW}{args.concurrency}{Style.RESET_ALL} download paralleli per batch")
    print(f"  • Finestra ML: {Fore.MAGENTA}7{Style.RESET_ALL} valori (pattern a 7 bit)")
    print(f"  • Output dataset: {Fore.BLUE}{os.path.abspath('datasets')}{Style.RESET_ALL}")
    print(f"  • ML dataset: {'Attivato' if args.ml else 'Disattivato'} {Fore.YELLOW}(usa --ml per attivare){Style.RESET_ALL}")
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
            print(f"\n{Back.MAGENTA}{Fore.WHITE} INIZIO CICLO DI AGGIORNAMENTO #{iteration} - {current_time.strftime('%Y-%m-%d %H:%M:%S')} {Style.RESET_ALL}")
            
            # Esegui l'aggiornamento
            results = await real_time_update(args)
            
            # Visualizza i risultati
            if results:
                display_update_results(results)
            
            # Calcola il prossimo orario di aggiornamento
            next_update = datetime.now() + timedelta(seconds=UPDATE_INTERVAL)
            logging.info(f"Aggiornamento #{iteration} completato. Prossimo aggiornamento alle {Fore.CYAN}{next_update.strftime('%H:%M:%S')}{Style.RESET_ALL}")
            
            # Attendi il prossimo ciclo di aggiornamento
            print(f"{Fore.YELLOW}In attesa del prossimo ciclo di aggiornamento tra {UPDATE_INTERVAL} secondi...{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Premi Ctrl+C per interrompere il monitoraggio.{Style.RESET_ALL}")
            
            await asyncio.sleep(UPDATE_INTERVAL)
            iteration += 1
            
    except KeyboardInterrupt:
        print(f"\n{Fore.RED}Monitoraggio interrotto manualmente dall'utente.{Style.RESET_ALL}")
    except Exception as e:
        logging.error(f"Errore nel loop principale: {e}")
        import traceback
        logging.error(traceback.format_exc())
    finally:
        logging.info("Monitoraggio in tempo reale terminato.")

if __name__ == "__main__":
    asyncio.run(main())
