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
from modules.data.dataset_generator import export_supervised_training_data
from modules.data.full_dataset_generator import generate_full_ml_dataset
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
UPDATE_INTERVAL = 5 * 60

async def catch_up_missing_indicators(symbols, timeframes):
    """
    Controlla e calcola gli indicatori tecnici mancanti per tutti i simboli.
    
    Questo risolve il problema dei simboli "vecchi" che hanno dati OHLCV
    ma non hanno indicatori tecnici nelle tabelle ta_<timeframe>.
    
    Args:
        symbols: Lista di simboli da controllare
        timeframes: Lista di timeframe da processare
    """
    import sqlite3
    from modules.utils.config import DB_FILE
    
    logging.info(f"{Fore.CYAN}🔍 Controllo indicatori mancanti per {len(symbols)} simboli su {len(timeframes)} timeframe...{Style.RESET_ALL}")
    
    symbols_processed = 0
    symbols_with_missing_ta = 0
    
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            
            for tf in timeframes:
                data_table = f"data_{tf}"
                ta_table = f"ta_{tf}"
                
                for symbol in symbols:
                    try:
                        # Controlla se ci sono dati OHLCV senza corrispondenti TA
                        query = f"""
                            SELECT COUNT(*) as missing_count
                            FROM {data_table} d
                            LEFT JOIN {ta_table} t ON d.symbol = t.symbol AND d.timestamp = t.timestamp  
                            WHERE d.symbol = ? AND t.timestamp IS NULL
                        """
                        cursor.execute(query, (symbol,))
                        missing_count = cursor.fetchone()[0]
                        
                        if missing_count > 0:
                            logging.info(f"{Fore.YELLOW}📊 {symbol} ({tf}): {missing_count} indicatori mancanti{Style.RESET_ALL}")
                            symbols_with_missing_ta += 1
                            
                            # Calcola gli indicatori mancanti
                            success = await compute_and_save_indicators(symbol, tf)
                            if success:
                                logging.info(f"{Fore.GREEN}✅ {symbol} ({tf}): indicatori calcolati{Style.RESET_ALL}")
                            else:
                                logging.warning(f"{Fore.RED}❌ {symbol} ({tf}): errore nel calcolo indicatori{Style.RESET_ALL}")
                        
                        symbols_processed += 1
                        
                    except Exception as e:
                        logging.error(f"{Fore.RED}Errore controllo {symbol} ({tf}): {e}{Style.RESET_ALL}")
        
        if symbols_with_missing_ta > 0:
            logging.info(f"{Fore.GREEN}🎯 Catch-up completato: {symbols_with_missing_ta}/{symbols_processed} simboli con indicatori mancanti processati{Style.RESET_ALL}")
        else:
            logging.info(f"{Fore.CYAN}✨ Tutti i simboli hanno indicatori tecnici aggiornati!{Style.RESET_ALL}")
            
    except Exception as e:
        logging.error(f"{Fore.RED}Errore durante catch-up indicatori: {e}{Style.RESET_ALL}")
        import traceback
        logging.error(traceback.format_exc())

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
        
        # 🆕 VALIDAZIONE DATI POST-DOWNLOAD (se non saltata)
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
            
            # 🆕 EXPORT REPORT CSV (se richiesto)
            if args.export_validation_report and validation_reports:
                try:
                    export_validation_report_csv(validation_reports, validation_summary)
                except Exception as e:
                    logging.error(f"Error exporting validation report: {e}")
            
            # 🆕 GENERA GRAFICI (se richiesto)
            if args.generate_validation_charts and validation_reports:
                try:
                    generate_validation_charts(validation_reports)
                except Exception as e:
                    logging.error(f"Error generating validation charts: {e}")
        else:
            logging.info(f"{Fore.YELLOW}🔍 Data validation skipped (--skip-validation flag used){Style.RESET_ALL}")
        
        # 🆕 CATCH-UP: Controlla e calcola TA mancanti per tutti i simboli
        if not args.no_ta:
            logging.info(f"{Fore.YELLOW}🔧 Controllo indicatori tecnici mancanti per tutti i simboli...{Style.RESET_ALL}")
            await catch_up_missing_indicators(top_symbols, args.timeframes)
        
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
        
        # Generate ML dataset (now by default unless --no-ml is specified)
        # Check if any new volatility data was saved or if force regeneration is requested
        has_new_data = any(res["completati"] > 0 for _, res in all_timeframe_results.items())
        if not args.no_ml and (has_new_data or args.force_ml):
            if has_new_data:
                logging.info(f"{Fore.YELLOW}Generating ML dataset with new volatility data{Style.RESET_ALL}")
            elif args.force_ml:
                logging.info(f"{Fore.YELLOW}Forcing ML dataset regeneration as requested with --force-ml{Style.RESET_ALL}")
            
            # Process each symbol and timeframe to generate the merged ML dataset
            for sym in top_symbols:
                for tf in args.timeframes:
                    try:
                        # Generate full ML dataset with merged data
                        logging.info(f"Generating merged ML dataset for {Fore.YELLOW}{sym}{Style.RESET_ALL} ({tf})")
                        
                        await generate_full_ml_dataset(
                            symbol=sym,
                            timeframe=tf,
                            window_size=7,  # Use default window size
                            force=args.force_ml,
                            filter_flat_patterns=False  # Keep all patterns by default
                        )
                        

                        
                    except Exception as e:
                        logging.error(f"Error generating merged ML dataset for {sym} ({tf}): {e}")
                        import traceback
                        logging.error(traceback.format_exc())
        elif not args.no_ml:
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
    print(f"  • ML dataset: {'Disattivato' if args.no_ml else 'Attivato'} {Fore.YELLOW}(usa --no-ml per disattivare){Style.RESET_ALL}")
    validation_status = "Saltata (--skip-validation)" if args.skip_validation else "Sempre attiva (controllo qualità e auto-riparazione)"
    validation_color = Fore.YELLOW if args.skip_validation else Fore.GREEN
    print(f"  • Validazione dati: {validation_color}{validation_status}{Style.RESET_ALL}")
    
    # Mostra opzioni di export/grafici se attive
    if not args.skip_validation:
        export_options = []
        if args.export_validation_report:
            export_options.append("CSV export")
        if args.generate_validation_charts:
            export_options.append("Grafici/heatmap")
        if export_options:
            print(f"  • Export validazione: {Fore.CYAN}{', '.join(export_options)}{Style.RESET_ALL}")
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
