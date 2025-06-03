#!/usr/bin/env python3
"""
TradingJii Full Pipeline Orchestrator
=====================================

Esegue l'intero workflow del sistema TradingJii in sequenza:
1. 📊 Data Collection (single run)
2. 🔧 Feature Engineering (enhanced features)
3. 🤖 Model Training (ensemble models)
4. 🎯 Prediction Testing

Uso:
    python run_full_pipeline.py [options]

Esempi:
    python run_full_pipeline.py                                    # Pipeline completa default
    python run_full_pipeline.py --symbols 10 --days 30            # 10 simboli, 30 giorni di dati
    python run_full_pipeline.py --skip-data-collection             # Salta raccolta dati
    python run_full_pipeline.py --timeframes 1h 4h                # Solo timeframes 1h e 4h
"""

import os
import sys
import asyncio
import logging
import argparse
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
from colorama import Fore, Style, Back, init

# Initialize colorama
init(autoreset=True)

# Add current directory to Python path to fix import issues
current_dir = str(Path.cwd())
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import custom modules
try:
    from modules.utils.logging_setup import setup_logging
    from modules.utils.config import DB_FILE
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("❌ Assicurati di essere nella directory del progetto TradingJii")
    sys.exit(1)

# ====================================================================
# PIPELINE CONFIGURATION
# ====================================================================

DEFAULT_CONFIG = {
    'num_symbols': 10,
    'timeframes': ['1h'],
    'days_of_data': 30,
    'batch_size': 5,
    'concurrency': 3,
    'enable_validation': True,
    'generate_ml_datasets': True,
    'force_rebuild': False
}

class TradingJiiPipeline:
    """
    Orchestratore completo del pipeline TradingJii.
    
    Gestisce l'esecuzione sequenziale di tutti i componenti
    con logging dettagliato e gestione errori.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize pipeline orchestrator.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.logger = setup_logging(logging.INFO)
        self.start_time = datetime.now()
        self.phase_times = {}
        self.results = {}
        
        # Create results directory
        self.results_dir = Path("pipeline_results")
        self.results_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"🚀 TradingJii Pipeline Orchestrator initialized")
        self.logger.info(f"📊 Configuration: {self.config}")
    
    def log_phase_start(self, phase_name: str, description: str) -> None:
        """Log the start of a pipeline phase."""
        print(f"\n{Back.BLUE}{Fore.WHITE}  PHASE: {phase_name.upper()}  {Style.RESET_ALL}")
        print(f"{Fore.CYAN}📍 {description}{Style.RESET_ALL}")
        print("=" * 80)
        
        self.phase_times[phase_name] = {'start': datetime.now()}
        self.logger.info(f"Starting phase: {phase_name}")
    
    def log_phase_end(self, phase_name: str, success: bool = True) -> None:
        """Log the end of a pipeline phase."""
        end_time = datetime.now()
        start_time = self.phase_times[phase_name]['start']
        duration = end_time - start_time
        
        self.phase_times[phase_name]['end'] = end_time
        self.phase_times[phase_name]['duration'] = duration
        self.phase_times[phase_name]['success'] = success
        
        status = "✅ COMPLETED" if success else "❌ FAILED"
        color = Fore.GREEN if success else Fore.RED
        
        print(f"\n{color}{status}: {phase_name} in {duration}{Style.RESET_ALL}")
        print("=" * 80)
        
        self.logger.info(f"Phase {phase_name} {'completed' if success else 'failed'} in {duration}")
    
    async def run_data_collection(self) -> bool:
        """
        Execute data collection phase.
        
        Returns:
            True if successful, False otherwise
        """
        self.log_phase_start("data_collection", "Downloading OHLCV data and generating indicators")
        
        try:
            # Ensure Python path is correct for imports
            import sys
            current_dir_local = str(Path.cwd())
            if current_dir_local not in sys.path:
                sys.path.insert(0, current_dir_local)
            
            # Try to import with error handling
            try:
                from data_collector import real_time_update_with_predictions, BinaryTradingSystem
                self.logger.info("✅ Successfully imported data_collector modules")
            except ImportError as import_error:
                self.logger.error(f"❌ Failed to import data_collector: {import_error}")
                
                # Try alternative approach with subprocess
                self.logger.info("🔄 Trying alternative data collection approach...")
                return await self._run_data_collection_subprocess()
            
            # Create mock args for single run
            class MockArgs:
                def __init__(self, config):
                    self.num_symbols = config['num_symbols']
                    self.timeframes = config['timeframes']
                    self.days = config['days_of_data']
                    self.batch_size = config['batch_size']
                    self.concurrency = config['concurrency']
                    self.sequential = False
                    self.no_ta = False
                    self.no_ml = False
                    self.skip_validation = not config['enable_validation']
                    self.generate_ml_datasets = config['generate_ml_datasets']
                    self.force_ml_dataset = config['force_rebuild']
                    self.export_validation_report = True
                    self.generate_validation_charts = False
            
            args = MockArgs(self.config)
            
            # Initialize trading system (but don't run predictions in this phase)
            trading_system_config = {'enable_predictions': False}
            trading_system = BinaryTradingSystem(trading_system_config)
            
            # Run single data collection cycle
            self.logger.info("🔄 Starting data collection...")
            results = await real_time_update_with_predictions(args, trading_system)
            
            if results:
                self.results['data_collection'] = {
                    'total_records': results['total_records_saved'],
                    'symbols_completed': results['grand_total_symbols']['completati'],
                    'symbols_skipped': results['grand_total_symbols']['saltati'],
                    'symbols_failed': results['grand_total_symbols']['falliti'],
                    'execution_time': results['execution_time'],
                    'validation_summary': results.get('validation_summary', {})
                }
                
                self.logger.info(f"✅ Data collection completed:")
                self.logger.info(f"  📊 Total records: {results['total_records_saved']:,}")
                self.logger.info(f"  ✅ Symbols completed: {results['grand_total_symbols']['completati']}")
                self.logger.info(f"  ⏱️ Time: {results['execution_time']}")
                
                self.log_phase_end("data_collection", True)
                return True
            else:
                self.logger.error("❌ Data collection failed - no results returned")
                self.log_phase_end("data_collection", False)
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Data collection failed: {e}")
            self.logger.error(traceback.format_exc())
            self.log_phase_end("data_collection", False)
            return False
    
    async def _run_data_collection_subprocess(self) -> bool:
        """Run data collection as subprocess if direct import fails."""
        import subprocess
        
        # Look for data collection scripts
        data_scripts = ['data_collector.py', 'real_time.py']
        
        for script in data_scripts:
            if Path(script).exists():
                self.logger.info(f"🔄 Trying data collection with: {script}")
                try:
                    # Build command line arguments
                    cmd = [
                        sys.executable, script,
                        '--num-symbols', str(self.config['num_symbols']),
                        '--days', str(self.config['days_of_data']),
                        '--batch-size', str(self.config['batch_size']),
                        '--concurrency', str(self.config['concurrency'])
                    ]
                    
                    # Add timeframes
                    for tf in self.config['timeframes']:
                        cmd.extend(['--timeframes', tf])
                    
                    # Add flags
                    if not self.config['enable_validation']:
                        cmd.append('--skip-validation')
                    if self.config['generate_ml_datasets']:
                        cmd.append('--generate-ml-datasets')
                    if self.config['force_rebuild']:
                        cmd.append('--force-ml-dataset')
                    
                    self.logger.info(f"Running: {' '.join(cmd)}")
                    
                    # Run the command
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
                    
                    if result.returncode == 0:
                        self.logger.info(f"✅ Data collection completed with {script}")
                        
                        # Parse basic results from output
                        self.results['data_collection'] = {
                            'status': 'completed_subprocess',
                            'script_used': script,
                            'return_code': result.returncode
                        }
                        
                        self.log_phase_end("data_collection", True)
                        return True
                    else:
                        self.logger.warning(f"⚠️ Data collection failed with {script}")
                        self.logger.warning(f"Error: {result.stderr}")
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ Error running {script}: {e}")
        
        return False
    
    def run_feature_engineering(self) -> bool:
        """
        Execute feature engineering phase.
        
        Returns:
            True if successful, False otherwise
        """
        self.log_phase_start("feature_engineering", "Generating enhanced features for ML datasets")
        
        try:
            # Check if enhanced_feature_engineer_v2.py exists
            if not Path("enhanced_feature_engineer_v2.py").exists():
                self.logger.warning("⚠️ enhanced_feature_engineer_v2.py not found, trying alternative...")
                
                # Try alternative feature engineering files
                for alt_file in ["feature_engineer.py", "enhanced_feature_engineer.py"]:
                    if Path(alt_file).exists():
                        self.logger.info(f"🔄 Using alternative: {alt_file}")
                        try:
                            if alt_file == "feature_engineer.py":
                                from feature_engineer import enhance_ml_datasets_v2
                            else:
                                from enhanced_feature_engineer import enhance_ml_datasets_v2
                            break
                        except ImportError:
                            self.logger.warning(f"⚠️ Failed to import from {alt_file}")
                            continue
                else:
                    self.logger.error("❌ No feature engineering script found")
                    self.log_phase_end("feature_engineering", False)
                    return False
            else:
                # Import feature engineering function
                from enhanced_feature_engineer_v2 import enhance_ml_datasets_v2
            
            self.logger.info("🔧 Starting enhanced feature engineering...")
            
            # Run feature engineering
            result = enhance_ml_datasets_v2()
            
            if result is not False:  # Function returns None on success, False on failure
                self.results['feature_engineering'] = {
                    'status': 'completed',
                    'enhanced_datasets_created': True
                }
                
                self.logger.info("✅ Feature engineering completed successfully")
                self.log_phase_end("feature_engineering", True)
                return True
            else:
                self.logger.error("❌ Feature engineering failed")
                self.log_phase_end("feature_engineering", False)
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Feature engineering failed: {e}")
            self.logger.error(traceback.format_exc())
            self.log_phase_end("feature_engineering", False)
            return False
    
    def run_model_training(self) -> bool:
        """
        Execute model training phase.
        
        Returns:
            True if successful, False otherwise
        """
        self.log_phase_start("model_training", "Training ensemble models with hyperparameter optimization")
        
        try:
            # Check if train.py exists and has the required function
            if Path("train.py").exists():
                try:
                    from train import train_all_enhanced_models
                    self.logger.info("🤖 Starting model training with train.py...")
                    result = train_all_enhanced_models()
                except (ImportError, AttributeError):
                    self.logger.warning("⚠️ train_all_enhanced_models not found in train.py")
                    return self._try_alternative_training()
            else:
                self.logger.warning("⚠️ train.py not found, trying alternatives...")
                return self._try_alternative_training()
            
            # Check if models were created
            models_dir = Path("ml_system/models")
            if models_dir.exists() and any(models_dir.iterdir()):
                self.results['model_training'] = {
                    'status': 'completed',
                    'models_created': True,
                    'models_directory': str(models_dir.absolute())
                }
                
                self.logger.info("✅ Model training completed successfully")
                self.log_phase_end("model_training", True)
                return True
            else:
                self.logger.error("❌ Model training failed - no models found")
                self.log_phase_end("model_training", False)
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Model training failed: {e}")
            self.logger.error(traceback.format_exc())
            self.log_phase_end("model_training", False)
            return False
    
    def _try_alternative_training(self) -> bool:
        """Try alternative training scripts."""
        alternative_scripts = [
            "train_enhanced_binary_model.py",
            "train_binary_model.py", 
            "train_models.py"
        ]
        
        for script in alternative_scripts:
            if Path(script).exists():
                self.logger.info(f"🔄 Trying alternative training script: {script}")
                try:
                    # Run the script as subprocess
                    import subprocess
                    result = subprocess.run([sys.executable, script], 
                                          capture_output=True, text=True, timeout=3600)
                    
                    if result.returncode == 0:
                        self.logger.info(f"✅ Alternative training completed: {script}")
                        return True
                    else:
                        self.logger.warning(f"⚠️ Alternative training failed: {script}")
                        self.logger.warning(f"Error: {result.stderr}")
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ Error running {script}: {e}")
        
        return False
    
    def run_prediction_testing(self) -> bool:
        """
        Execute prediction testing phase.
        
        Returns:
            True if successful, False otherwise
        """
        self.log_phase_start("prediction_testing", "Testing model predictions and generating sample signals")
        
        try:
            from predict import PredictionEngine
            
            self.logger.info("🎯 Starting prediction testing...")
            
            # Look for available models
            models_dir = Path("ml_system/models")
            if not models_dir.exists():
                models_dir.mkdir(parents=True, exist_ok=True)
            
            model_files = list(models_dir.glob("**/best_binary_model.pkl"))
            
            if not model_files:
                # Look for enhanced models
                model_files = list(models_dir.glob("**/ensemble_model.joblib"))
            
            if not model_files:
                # Look for any model files
                model_files = list(models_dir.glob("**/*.pkl")) + list(models_dir.glob("**/*.joblib"))
            
            if not model_files:
                self.logger.warning("⚠️ No trained models found for testing")
                self.results['prediction_testing'] = {
                    'status': 'skipped',
                    'reason': 'no_models_found'
                }
                self.log_phase_end("prediction_testing", True)
                return True
            
            # Test with first available model
            model_path = model_files[0]
            self.logger.info(f"🔍 Testing model: {model_path}")
            
            # Initialize prediction engine
            config = {
                'model_path': str(model_path),
                'confidence_threshold': 0.7,
                'enable_detailed_logging': True
            }
            
            engine = PredictionEngine(config)
            
            # Get current signal state
            current_signal = engine.get_current_signal()
            
            self.results['prediction_testing'] = {
                'status': 'completed',
                'model_tested': str(model_path),
                'current_signal': current_signal,
                'engine_initialized': True
            }
            
            self.logger.info(f"✅ Prediction testing completed")
            self.logger.info(f"  🎯 Current signal: {current_signal.get('signal', 'N/A')}")
            self.logger.info(f"  📊 Confidence: {current_signal.get('confidence', 0.0):.3f}")
            
            self.log_phase_end("prediction_testing", True)
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Prediction testing failed: {e}")
            self.logger.error(traceback.format_exc())
            self.log_phase_end("prediction_testing", False)
            return False
    
    def generate_pipeline_report(self) -> None:
        """Generate comprehensive pipeline execution report."""
        total_time = datetime.now() - self.start_time
        
        print(f"\n{Back.GREEN}{Fore.BLACK}  PIPELINE EXECUTION REPORT  {Style.RESET_ALL}")
        print("=" * 80)
        
        # Phase summary
        print(f"\n📊 PHASE EXECUTION SUMMARY:")
        print("-" * 50)
        
        for phase_name, timing in self.phase_times.items():
            status = "✅ SUCCESS" if timing.get('success', False) else "❌ FAILED"
            duration = timing.get('duration', timedelta(0))
            print(f"{phase_name:20} | {status:12} | {str(duration):>12}")
        
        print("-" * 50)
        print(f"{'TOTAL PIPELINE':20} | {'':12} | {str(total_time):>12}")
        
        # Results summary
        if self.results:
            print(f"\n📈 RESULTS SUMMARY:")
            print("-" * 50)
            
            for phase, result in self.results.items():
                print(f"\n🔸 {phase.upper().replace('_', ' ')}:")
                for key, value in result.items():
                    if isinstance(value, dict):
                        print(f"  {key}: {len(value)} items")
                    else:
                        print(f"  {key}: {value}")
        
        # Database info
        try:
            if Path(DB_FILE).exists():
                db_size = Path(DB_FILE).stat().st_size / 1024 / 1024  # MB
                print(f"\n💾 DATABASE INFO:")
                print(f"  File: {DB_FILE}")
                print(f"  Size: {db_size:.2f} MB")
        except:
            pass
        
        # Models info
        models_dir = Path("ml_system/models")
        if models_dir.exists():
            model_count = len(list(models_dir.glob("**/*.pkl"))) + len(list(models_dir.glob("**/*.joblib")))
            print(f"\n🤖 MODELS INFO:")
            print(f"  Directory: {models_dir}")
            print(f"  Models created: {model_count}")
        
        # ML datasets info
        datasets_dir = Path("ml_datasets")
        if datasets_dir.exists():
            dataset_count = len(list(datasets_dir.glob("**/merged*.csv")))
            print(f"\n📊 DATASETS INFO:")
            print(f"  Directory: {datasets_dir}")
            print(f"  Datasets created: {dataset_count}")
        
        print(f"\n🎯 PIPELINE COMPLETED in {total_time}")
        print("=" * 80)
        
        # Save detailed report
        report_file = self.results_dir / f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        import json
        report_data = {
            'execution_time': str(total_time),
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'config': self.config,
            'phase_times': {
                phase: {
                    'start': timing['start'].isoformat(),
                    'end': timing.get('end', datetime.now()).isoformat(),
                    'duration': str(timing.get('duration', timedelta(0))),
                    'success': timing.get('success', False)
                } for phase, timing in self.phase_times.items()
            },
            'results': self.results
        }
        
        try:
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2)
            self.logger.info(f"📋 Detailed report saved: {report_file}")
        except Exception as e:
            self.logger.error(f"Failed to save report: {e}")
    
    async def run_full_pipeline(self, skip_phases: List[str] = None) -> bool:
        """
        Execute the complete TradingJii pipeline.
        
        Args:
            skip_phases: List of phases to skip
            
        Returns:
            True if pipeline completed successfully, False otherwise
        """
        skip_phases = skip_phases or []
        
        print(f"\n{Back.MAGENTA}{Fore.WHITE}  TRADINGJII FULL PIPELINE  {Style.RESET_ALL}")
        print("=" * 80)
        print(f"🕐 Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⚙️ Configuration: {self.config}")
        print(f"📂 Working directory: {os.getcwd()}")
        print("=" * 80)
        
        success = True
        
        # Phase 1: Data Collection
        if 'data_collection' not in skip_phases:
            if not await self.run_data_collection():
                success = False
                if not self._should_continue_on_error("data_collection"):
                    return False
        else:
            self.logger.info("⏭️ Skipping data collection phase")
        
        # Phase 2: Feature Engineering
        if 'feature_engineering' not in skip_phases:
            if not self.run_feature_engineering():
                success = False
                if not self._should_continue_on_error("feature_engineering"):
                    return False
        else:
            self.logger.info("⏭️ Skipping feature engineering phase")
        
        # Phase 3: Model Training
        if 'model_training' not in skip_phases:
            if not self.run_model_training():
                success = False
                if not self._should_continue_on_error("model_training"):
                    return False
        else:
            self.logger.info("⏭️ Skipping model training phase")
        
        # Phase 4: Prediction Testing
        if 'prediction_testing' not in skip_phases:
            if not self.run_prediction_testing():
                success = False
                # Don't fail pipeline if prediction testing fails
        else:
            self.logger.info("⏭️ Skipping prediction testing phase")
        
        # Generate final report
        self.generate_pipeline_report()
        
        return success
    
    def _should_continue_on_error(self, failed_phase: str) -> bool:
        """
        Determine if pipeline should continue after a phase failure.
        
        Args:
            failed_phase: Name of the failed phase
            
        Returns:
            True to continue, False to stop
        """
        # For now, stop on any critical phase failure
        critical_phases = ['data_collection', 'feature_engineering']
        
        if failed_phase in critical_phases:
            self.logger.error(f"❌ Critical phase {failed_phase} failed. Stopping pipeline.")
            return False
        
        self.logger.warning(f"⚠️ Non-critical phase {failed_phase} failed. Continuing pipeline.")
        return True

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="TradingJii Full Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_full_pipeline.py                           # Run complete pipeline with defaults
  python run_full_pipeline.py --symbols 10 --days 30   # 10 symbols, 30 days of data
  python run_full_pipeline.py --skip-data-collection    # Skip data collection phase
  python run_full_pipeline.py --timeframes 1h 4h       # Only 1h and 4h timeframes
  python run_full_pipeline.py --force-rebuild           # Force rebuild of all datasets
        """
    )
    
    # Data collection arguments
    parser.add_argument('--symbols', type=int, default=DEFAULT_CONFIG['num_symbols'],
                       help=f"Number of top symbols to process (default: {DEFAULT_CONFIG['num_symbols']})")
    
    parser.add_argument('--timeframes', nargs='+', default=DEFAULT_CONFIG['timeframes'],
                       help=f"Timeframes to process (default: {' '.join(DEFAULT_CONFIG['timeframes'])})")
    
    parser.add_argument('--days', type=int, default=DEFAULT_CONFIG['days_of_data'],
                       help=f"Days of historical data to download (default: {DEFAULT_CONFIG['days_of_data']})")
    
    parser.add_argument('--batch-size', type=int, default=DEFAULT_CONFIG['batch_size'],
                       help=f"Batch size for data processing (default: {DEFAULT_CONFIG['batch_size']})")
    
    parser.add_argument('--concurrency', type=int, default=DEFAULT_CONFIG['concurrency'],
                       help=f"Concurrency level for downloads (default: {DEFAULT_CONFIG['concurrency']})")
    
    # Phase control arguments
    parser.add_argument('--skip-data-collection', action='store_true',
                       help="Skip data collection phase")
    
    parser.add_argument('--skip-feature-engineering', action='store_true',
                       help="Skip feature engineering phase")
    
    parser.add_argument('--skip-model-training', action='store_true',
                       help="Skip model training phase")
    
    parser.add_argument('--skip-prediction-testing', action='store_true',
                       help="Skip prediction testing phase")
    
    # Options
    parser.add_argument('--force-rebuild', action='store_true',
                       help="Force rebuild of all datasets and models")
    
    parser.add_argument('--no-validation', action='store_true',
                       help="Skip data validation")
    
    parser.add_argument('--verbose', action='store_true',
                       help="Enable verbose logging")
    
    return parser.parse_args()

async def main():
    """Main entry point for the pipeline orchestrator."""
    try:
        # Parse arguments (use defaults if no args provided)
        if len(sys.argv) == 1:
            # No arguments provided, use all defaults
            print(f"{Fore.YELLOW}🔧 No arguments provided, using DEFAULT configuration:{Style.RESET_ALL}")
            print(f"   📊 Symbols: {DEFAULT_CONFIG['num_symbols']}")
            print(f"   📈 Timeframes: {DEFAULT_CONFIG['timeframes']}")
            print(f"   📅 Days: {DEFAULT_CONFIG['days_of_data']}")
            print(f"   ⚙️ Use --help for all options")
            
            config = DEFAULT_CONFIG.copy()
            skip_phases = []
        else:
            args = parse_arguments()
            
            # Build configuration
            config = {
                'num_symbols': args.symbols,
                'timeframes': args.timeframes,
                'days_of_data': args.days,
                'batch_size': args.batch_size,
                'concurrency': args.concurrency,
                'enable_validation': not args.no_validation,
                'generate_ml_datasets': True,
                'force_rebuild': args.force_rebuild
            }
            
            # Determine phases to skip
            skip_phases = []
            if args.skip_data_collection:
                skip_phases.append('data_collection')
            if args.skip_feature_engineering:
                skip_phases.append('feature_engineering')
            if args.skip_model_training:
                skip_phases.append('model_training')
            if args.skip_prediction_testing:
                skip_phases.append('prediction_testing')
            
            # Configure logging
            log_level = logging.DEBUG if args.verbose else logging.INFO
            logger = setup_logging(log_level)
        
        # Initialize and run pipeline
        pipeline = TradingJiiPipeline(config)
        success = await pipeline.run_full_pipeline(skip_phases)
        
        if success:
            print(f"\n{Fore.GREEN}🎉 TradingJii pipeline completed successfully!{Style.RESET_ALL}")
            sys.exit(0)
        else:
            print(f"\n{Fore.RED}💥 TradingJii pipeline failed!{Style.RESET_ALL}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}⚠️ Pipeline interrupted by user{Style.RESET_ALL}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Fore.RED}💥 Pipeline failed with error: {e}{Style.RESET_ALL}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # Set event loop policy for Windows if needed
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Run the pipeline
    asyncio.run(main())
