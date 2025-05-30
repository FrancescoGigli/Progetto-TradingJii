#!/usr/bin/env python3
"""
Binary Real-Time Trading Signal System
=====================================

Sistema di trading real-time che integra:
1. Download continuo dati OHLCV
2. Generazione dataset ML 
3. Predizione binaria con gestione confidenza
4. Logging strutturato segnali
5. Aggiornamento stato persistente

Caratteristiche:
- Engine di predizione binaria con threshold di confidenza
- Gestione transizioni di stato BUY/SELL/HOLD
- Logging JSON strutturato
- Monitoraggio continuo ogni 5 minuti
- Integration con pipeline ML esistente
"""

import sys
import os
import asyncio
import logging
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from colorama import Fore, Style, Back, init

# Moduli personalizzati
from modules.utils.logging_setup import setup_logging
from modules.utils.command_args import parse_arguments
from modules.utils.config import DB_FILE, TIMEFRAME_CONFIG, REALTIME_CONFIG
from modules.core.exchange import create_exchange, fetch_markets, get_top_symbols
from modules.core.download_orchestrator import process_timeframe
from modules.data.db_manager import init_data_tables
from modules.data.volatility_processor import process_and_save_volatility
from modules.data.indicator_processor import init_indicator_tables, compute_and_save_indicators
from generate_merged_datasets import generate_dataset_for_symbol
from modules.data.data_validator import (
    validate_and_repair_data, log_validation_results, log_validation_summary,
    export_validation_report_csv, generate_validation_charts, DataQualityReport
)
from predict import PredictionEngine

# Inizializza colorama
init(autoreset=True)

# Imposta la policy di event loop per Windows se necessario
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Intervallo di aggiornamento in secondi (default: 5 minuti)
UPDATE_INTERVAL = REALTIME_CONFIG['update_interval_seconds']

# ====================================================================
# BINARY PREDICTION SYSTEM CONFIGURATION
# ====================================================================

BINARY_PREDICTION_CONFIG = {
    'confidence_threshold': 0.7,
    'confidence_improvement_threshold': 0.05,
    'model_path': 'ml_system/models/binary_models/best_binary_model.pkl',
    'enable_predictions': True,
    'prediction_symbols': ['BTC_USDTUSDT', 'ETH_USDTUSDT'],  # Simboli per predizione
    'prediction_timeframes': ['1h'],              # Timeframes per predizione
    'log_all_predictions': True,
    'signal_state_file': 'ml_system/logs/predictions/signal_state.json'
}

class BinaryTradingSystem:
    """
    Sistema di trading binario con predizioni real-time.
    
    Integra download dati, generazione dataset ML e predizioni binarie
    con gestione dello stato dei segnali.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize binary trading system."""
        self.config = {**BINARY_PREDICTION_CONFIG, **(config or {})}
        self.logger = setup_logging(logging.INFO)
        
        # Prediction engines per simbolo/timeframe
        self.prediction_engines = {}
        self.last_predictions = {}
        
        # Inizializza prediction engines se abilitato
        if self.config['enable_predictions']:
            self._initialize_prediction_engines()
    
    def _initialize_prediction_engines(self) -> None:
        """Initialize prediction engines for configured symbols/timeframes."""
        model_path = Path(self.config['model_path'])
        
        if not model_path.exists():
            self.logger.warning(f"Model not found: {model_path}. Predictions disabled.")
            self.config['enable_predictions'] = False
            return
        
        try:
            for symbol in self.config['prediction_symbols']:
                for timeframe in self.config['prediction_timeframes']:
                    key = f"{symbol}_{timeframe}"
                    
                    engine_config = {
                        'model_path': self.config['model_path'],
                        'confidence_threshold': self.config['confidence_threshold'],
                        'confidence_improvement_threshold': self.config['confidence_improvement_threshold'],
                        'log_file': f"signal_log_{symbol}_{timeframe}.json",
                        'state_file': f"ml_system/logs/predictions/signal_state_{symbol}_{timeframe}.json",
                        'enable_detailed_logging': self.config['log_all_predictions']
                    }
                    
                    self.prediction_engines[key] = PredictionEngine(engine_config)
                    self.last_predictions[key] = None
                    
                    self.logger.info(f"Initialized prediction engine for {symbol} ({timeframe})")
            
            self.logger.info(f"Binary prediction system ready with {len(self.prediction_engines)} engines")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize prediction engines: {e}")
            self.config['enable_predictions'] = False
    
    async def run_predictions(self, symbols: List[str], timeframes: List[str]) -> Dict[str, Any]:
        """
        Run predictions on latest data for configured symbols/timeframes.
        
        Args:
            symbols: List of symbols that were updated
            timeframes: List of timeframes that were updated
            
        Returns:
            Dictionary with prediction results
        """
        if not self.config['enable_predictions']:
            return {'predictions_enabled': False}
        
        prediction_results = {}
        total_predictions = 0
        signal_changes = 0
        
        try:
            for symbol in self.config['prediction_symbols']:
                if symbol not in symbols:
                    continue
                    
                for timeframe in self.config['prediction_timeframes']:
                    if timeframe not in timeframes:
                        continue
                    
                    key = f"{symbol}_{timeframe}"
                    
                    if key not in self.prediction_engines:
                        continue
                    
                    try:
                        # Load latest dataset
                        dataset_path = Path(f"ml_datasets/{symbol}/{timeframe}/merged.csv")
                        
                        if not dataset_path.exists():
                            self.logger.warning(f"Dataset not found for prediction: {dataset_path}")
                            continue
                        
                        # Load and prepare data for prediction
                        df = pd.read_csv(dataset_path)
                        
                        if len(df) == 0:
                            self.logger.warning(f"Empty dataset for {symbol} ({timeframe})")
                            continue
                        
                        # Get latest features (excluding label columns)
                        exclude_cols = ['label', 'timestamp', 'datetime']
                        feature_cols = [col for col in df.columns if col not in exclude_cols]
                        
                        # Use latest row for prediction
                        latest_features = df[feature_cols].iloc[-1:].copy()
                        latest_timestamp = datetime.now().isoformat()
                        
                        # Make prediction
                        engine = self.prediction_engines[key]
                        result = engine.predict_single(
                            latest_features,
                            timestamp=latest_timestamp,
                            symbol=symbol,
                            timeframe=timeframe
                        )
                        
                        prediction_results[key] = result
                        total_predictions += 1
                        
                        if result.get('signal_changed', False):
                            signal_changes += 1
                        
                        # Log significant events
                        if result.get('signal_changed', False):
                            self.logger.info(
                                f"{Fore.CYAN}🚨 SIGNAL CHANGE: {symbol} ({timeframe}) → {result['signal']} "
                                f"(confidence: {result['confidence']:.3f}, reason: {result['decision_reason']}){Style.RESET_ALL}"
                            )
                        elif self.config['log_all_predictions']:
                            self.logger.debug(
                                f"Prediction: {symbol} ({timeframe}) → {result['signal']} "
                                f"(confidence: {result['confidence']:.3f})"
                            )
                        
                        # Store last prediction
                        self.last_predictions[key] = result
                        
                    except Exception as e:
                        self.logger.error(f"Prediction failed for {key}: {e}")
                        prediction_results[key] = {
                            'error': str(e),
                            'timestamp': datetime.now().isoformat()
                        }
            
            return {
                'predictions_enabled': True,
                'total_predictions': total_predictions,
                'signal_changes': signal_changes,
                'results': prediction_results,
                'engines_active': len(self.prediction_engines)
            }
            
        except Exception as e:
            self.logger.error(f"Prediction system error: {e}")
            return {
                'predictions_enabled': True,
                'error': str(e),
                'total_predictions': 0,
                'signal_changes': 0
            }
    
    def get_current_signals(self) -> Dict[str, Any]:
        """Get current signals from all engines."""
        signals = {}
        
        for key, engine in self.prediction_engines.items():
            try:
                signal_state = engine.get_current_signal()
                signals[key] = signal_state
            except Exception as e:
                signals[key] = {'error': str(e)}
        
        return signals
    
    def display_signal_summary(self, prediction_results: Dict[str, Any]) -> None:
        """Display summary of current signals."""
        if not prediction_results.get('predictions_enabled', False):
            return
        
        results = prediction_results.get('results', {})
        
        if not results:
            return
        
        print(f"\n{Back.BLUE}{Fore.WHITE}  BINARY TRADING SIGNALS  {Style.RESET_ALL}")
        print(f"{'Symbol':^12} | {'Timeframe':^10} | {'Signal':^6} | {'Confidence':^10} | {'Reason':^20}")
        print("-" * 75)
        
        for key, result in results.items():
            if 'error' in result:
                continue
            
            symbol, timeframe = key.split('_')
            signal = result.get('signal', 'N/A')
            confidence = result.get('confidence', 0.0)
            reason = result.get('decision_reason', 'unknown')[:18]
            
            # Color coding for signals
            if signal == 'BUY':
                signal_colored = f"{Fore.GREEN}{signal}{Style.RESET_ALL}"
            elif signal == 'SELL':
                signal_colored = f"{Fore.RED}{signal}{Style.RESET_ALL}"
            else:  # HOLD
                signal_colored = f"{Fore.YELLOW}{signal}{Style.RESET_ALL}"
            
            print(f"{symbol:^12} | {timeframe:^10} | {signal_colored:^12} | {confidence:^10.3f} | {reason:^20}")
        
        # Summary statistics
        total_predictions = prediction_results.get('total_predictions', 0)
        signal_changes = prediction_results.get('signal_changes', 0)
        engines_active = prediction_results.get('engines_active', 0)
        
        print("-" * 75)
        print(f"Active Engines: {engines_active} | Predictions: {total_predictions} | Signal Changes: {signal_changes}")

async def real_time_update_with_predictions(args, trading_system: BinaryTradingSystem):
    """
    Enhanced real-time update with binary predictions.
    
    Args:
        args: Command line arguments
        trading_system: Binary trading system instance
        
    Returns:
        Dictionary with update and prediction results
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
        
        # Elabora la volatilità per i simboli completati in tutti i timeframe
        for tf, res in all_timeframe_results.items():
            if res["completati"] > 0:
                for sym in top_symbols:
                    # Calcola e salva la volatilità per ogni simbolo
                    process_and_save_volatility(sym, tf)
                    
                    # Calcola e salva gli indicatori tecnici se non è specificato --no-ta
                    if not args.no_ta:
                        await compute_and_save_indicators(sym, tf)
        
        # 🆕 GENERAZIONE DATASET ML (se richiesto)
        symbols_with_ml_datasets = []
        if args.generate_ml_datasets and not args.no_ml:
            logging.info(f"{Fore.YELLOW}🤖 Generazione dataset ML abilitata{Style.RESET_ALL}")
            
            # Process each symbol and timeframe
            for sym in top_symbols:
                for tf in args.timeframes:
                    try:
                        logging.info(f"Generazione merged.csv per {Fore.YELLOW}{sym}{Style.RESET_ALL} ({tf})")
                        
                        success = generate_dataset_for_symbol(
                            symbol=sym,
                            timeframe=tf, 
                            force=args.force_ml_dataset
                        )
                        
                        if success:
                            logging.info(f"{Fore.GREEN}✅ Dataset generato per {sym} ({tf}){Style.RESET_ALL}")
                            if sym not in symbols_with_ml_datasets:
                                symbols_with_ml_datasets.append(sym)
                        else:
                            logging.error(f"{Fore.RED}❌ Errore generazione dataset per {sym} ({tf}){Style.RESET_ALL}")
                            
                    except Exception as e:
                        logging.error(f"{Fore.RED}Errore generazione dataset {sym} ({tf}): {e}{Style.RESET_ALL}")
        
        # 🚀 BINARY PREDICTIONS (nuovo!)
        prediction_results = await trading_system.run_predictions(symbols_with_ml_datasets or top_symbols, args.timeframes)
        
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
            "prediction_results": prediction_results,
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

def display_enhanced_results(results, trading_system: BinaryTradingSystem):
    """
    Display enhanced results including predictions.
    
    Args:
        results: Update results
        trading_system: Binary trading system instance
    """
    if not results:
        return
        
    all_timeframe_results = results["all_timeframe_results"]
    total_records_saved = results["total_records_saved"]
    grand_total_symbols = results["grand_total_symbols"]
    prediction_results = results["prediction_results"]
    start_time = results["start_time"]
    end_time = results["end_time"]
    time_str = results["execution_time"]
    
    # Resoconto finale dell'aggiornamento
    print("\n" + "="*80)
    print(f"{Back.GREEN}{Fore.BLACK}  RESOCONTO AGGIORNAMENTO E PREDIZIONI COMPLETATO  {Style.RESET_ALL}")
    print("="*80)
    print(f"  • Database: {Fore.BLUE}{os.path.abspath(DB_FILE)}{Style.RESET_ALL}")
    print(f"  • Dataset ML: {Fore.BLUE}{os.path.abspath('ml_datasets')}{Style.RESET_ALL}")
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
    
    # 🚀 Display prediction results
    trading_system.display_signal_summary(prediction_results)
    
    print(f"Inizio: {Fore.CYAN}{start_time.strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}")
    print(f"Fine:   {Fore.CYAN}{end_time.strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}")
    print("="*80 + "\n")

async def main():
    """
    Main entry point for binary real-time trading system.
    """
    # Configura il logger
    logger = setup_logging(level=logging.INFO)
    
    # Analizza gli argomenti da linea di comando
    args = parse_arguments()
    mode = "SEQUENZIALE" if args.sequential else "PARALLELA"
    
    # Initialize binary trading system
    trading_system = BinaryTradingSystem()
    
    # Header generale
    print("\n" + "="*80)
    print(f"{Back.BLUE}{Fore.WHITE}  BINARY TRADING SYSTEM (MODALITÀ {mode})  {Style.RESET_ALL}")
    print("="*80)
    print(f"  • Criptovalute da monitorare: {Fore.YELLOW}{args.num_symbols}{Style.RESET_ALL}")
    print(f"  • Timeframes monitorati: {Fore.GREEN}{', '.join(args.timeframes)}{Style.RESET_ALL}")
    print(f"  • Intervallo aggiornamento: {Fore.CYAN}{UPDATE_INTERVAL} secondi{Style.RESET_ALL}")
    print(f"  • Batch size: {Fore.YELLOW}{args.batch_size}{Style.RESET_ALL}")
    if not args.sequential:
        print(f"  • Concorrenza: {Fore.YELLOW}{args.concurrency}{Style.RESET_ALL} download paralleli per batch")
    
    # Prediction system status
    pred_status = "Abilitato" if trading_system.config['enable_predictions'] else "Disabilitato"
    pred_color = Fore.GREEN if trading_system.config['enable_predictions'] else Fore.RED
    print(f"  • Sistema predizioni binarie: {pred_color}{pred_status}{Style.RESET_ALL}")
    
    if trading_system.config['enable_predictions']:
        print(f"  • Simboli predizione: {Fore.CYAN}{', '.join(trading_system.config['prediction_symbols'])}{Style.RESET_ALL}")
        print(f"  • Timeframes predizione: {Fore.CYAN}{', '.join(trading_system.config['prediction_timeframes'])}{Style.RESET_ALL}")
        print(f"  • Soglia confidenza: {Fore.YELLOW}{trading_system.config['confidence_threshold']}{Style.RESET_ALL}")
        print(f"  • Engines attivi: {Fore.GREEN}{len(trading_system.prediction_engines)}{Style.RESET_ALL}")
    
    print(f"  • Output dataset: {Fore.BLUE}{os.path.abspath('ml_datasets')}{Style.RESET_ALL}")
    ml_status = "Disattivato (--no-ml)" if args.no_ml else ("Abilitato (--generate-ml-datasets)" if args.generate_ml_datasets else "Disabilitato")
    ml_color = Fore.RED if args.no_ml else (Fore.GREEN if args.generate_ml_datasets else Fore.YELLOW)
    print(f"  • Generazione ML datasets: {ml_color}{ml_status}{Style.RESET_ALL}")
    
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
            print(f"\n{Back.MAGENTA}{Fore.WHITE} INIZIO CICLO #{iteration} - TRADING BINARIO - {current_time.strftime('%Y-%m-%d %H:%M:%S')} {Style.RESET_ALL}")
            
            # Esegui l'aggiornamento con predizioni
            results = await real_time_update_with_predictions(args, trading_system)
            
            # Visualizza i risultati enhanced
            if results:
                display_enhanced_results(results, trading_system)
            
            # Calcola il prossimo orario di aggiornamento
            next_update = datetime.now() + timedelta(seconds=UPDATE_INTERVAL)
            logging.info(f"Ciclo #{iteration} completato. Prossimo aggiornamento alle {Fore.CYAN}{next_update.strftime('%H:%M:%S')}{Style.RESET_ALL}")
            
            # Attendi il prossimo ciclo di aggiornamento
            print(f"{Fore.YELLOW}In attesa del prossimo ciclo di aggiornamento tra {UPDATE_INTERVAL} secondi...{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Premi Ctrl+C per interrompere il sistema di trading.{Style.RESET_ALL}")
            
            await asyncio.sleep(UPDATE_INTERVAL)
            iteration += 1
            
    except KeyboardInterrupt:
        print(f"\n{Fore.RED}Sistema di trading binario interrotto manualmente dall'utente.{Style.RESET_ALL}")
    except Exception as e:
        logging.error(f"Errore nel loop principale: {e}")
        import traceback
        logging.error(traceback.format_exc())
    finally:
        logging.info("Sistema di trading binario terminato.")

if __name__ == "__main__":
    asyncio.run(main())
