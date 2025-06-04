#!/usr/bin/env python3
"""
ML Signal Scanner CLI for TradingJii

Main CLI interface for the ML trading signal system:
- Train models on existing datasets
- Generate real-time trading signals
- System validation and health checks  
- Batch processing and output formatting
- Performance monitoring and diagnostics
"""

import os
import sys
import argparse
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Import ML modules
from modules.ml.config import (
    MODEL_CONFIG, 
    FALLBACK_CONFIG,
    LOGGING_CONFIG,
    get_available_models,
    get_ensemble_models,
    validate_config
)
from modules.ml.model_trainer import ModelTrainer, generate_datasets_and_train
from modules.ml.predictor import ModelPredictor, batch_predict
from modules.ml.feature_extractor import assess_feature_quality

# Import existing modules
from modules.utils.logging_setup import setup_logging
from modules.utils.config import REALTIME_CONFIG


def setup_cli_logging():
    """Setup logging for CLI operations."""
    import logging as log
    import logging.handlers
    
    # Use the existing logging setup but ensure console output
    logger = setup_logging(level=log.INFO)
    
    # Add file logging if configured
    if LOGGING_CONFIG["enable_file_logging"]:
        file_handler = log.handlers.RotatingFileHandler(
            LOGGING_CONFIG["predictor_log"],
            maxBytes=LOGGING_CONFIG["max_log_size_mb"] * 1024 * 1024,
            backupCount=3
        )
        file_handler.setFormatter(log.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)
    
    return logger


def validate_system_setup() -> Dict[str, bool]:
    """
    Validate that the ML system is properly set up.
    
    Returns:
        Dictionary with validation results
    """
    print(f"\n{Fore.CYAN}=== SYSTEM VALIDATION ==={Style.RESET_ALL}")
    
    validation_results = {
        "config_valid": False,
        "directories_exist": False,
        "models_available": False,
        "dependencies_available": False,
        "database_accessible": False
    }
    
    try:
        # 1. Validate configuration
        print(f"🔧 Checking ML configuration...")
        try:
            validate_config()
            validation_results["config_valid"] = True
            print(f"   ✅ Configuration valid")
        except Exception as e:
            print(f"   ❌ Configuration error: {e}")
        
        # 2. Check directories
        print(f"📁 Checking directories...")
        required_dirs = ["models", "logs", "datasets"]
        missing_dirs = []
        
        for dir_name in required_dirs:
            if not os.path.exists(dir_name):
                missing_dirs.append(dir_name)
                try:
                    os.makedirs(dir_name, exist_ok=True)
                    print(f"   ➕ Created directory: {dir_name}")
                except Exception as e:
                    print(f"   ❌ Could not create {dir_name}: {e}")
            else:
                print(f"   ✅ Directory exists: {dir_name}")
        
        validation_results["directories_exist"] = len(missing_dirs) == 0
        
        # 3. Check for available models
        print(f"🤖 Checking available models...")
        models_dir = "models"
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
            if model_files:
                validation_results["models_available"] = True
                print(f"   ✅ Found {len(model_files)} model files")
                for model_file in model_files[:3]:  # Show first 3
                    print(f"      - {model_file}")
                if len(model_files) > 3:
                    print(f"      ... and {len(model_files) - 3} more")
            else:
                print(f"   ⚠️  No trained models found - train models first")
        else:
            print(f"   ❌ Models directory not found")
        
        # 4. Check dependencies
        print(f"📦 Checking dependencies...")
        try:
            import sklearn
            import joblib
            import pandas as pd
            import numpy as np
            validation_results["dependencies_available"] = True
            print(f"   ✅ All required dependencies available")
            print(f"      - scikit-learn: {sklearn.__version__}")
            print(f"      - pandas: {pd.__version__}")
            print(f"      - numpy: {np.__version__}")
        except ImportError as e:
            print(f"   ❌ Missing dependencies: {e}")
            print(f"   💡 Install with: pip install scikit-learn pandas numpy joblib")
        
        # 5. Check database accessibility
        print(f"🗄️  Checking database...")
        try:
            from modules.data.db_manager import DB_FILE
            import sqlite3
            
            if os.path.exists(DB_FILE):
                with sqlite3.connect(DB_FILE) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    tables = cursor.fetchall()
                    
                if tables:
                    validation_results["database_accessible"] = True
                    print(f"   ✅ Database accessible with {len(tables)} tables")
                else:
                    print(f"   ⚠️  Database exists but no tables found")
            else:
                print(f"   ⚠️  Database not found - run data_collector.py first")
                
        except Exception as e:
            print(f"   ❌ Database error: {e}")
        
        # Summary
        print(f"\n{Fore.CYAN}=== VALIDATION SUMMARY ==={Style.RESET_ALL}")
        total_checks = len(validation_results)
        passed_checks = sum(validation_results.values())
        
        for check, result in validation_results.items():
            status = f"{Fore.GREEN}✅ PASS{Style.RESET_ALL}" if result else f"{Fore.RED}❌ FAIL{Style.RESET_ALL}"
            print(f"  {check.replace('_', ' ').title()}: {status}")
        
        overall_status = "READY" if passed_checks >= total_checks - 1 else "NEEDS SETUP"
        status_color = Fore.GREEN if passed_checks >= total_checks - 1 else Fore.YELLOW
        
        print(f"\n📊 Overall Status: {status_color}{overall_status}{Style.RESET_ALL} ({passed_checks}/{total_checks} checks passed)")
        
        if overall_status == "NEEDS SETUP":
            print(f"\n💡 {Fore.YELLOW}Setup recommendations:{Style.RESET_ALL}")
            if not validation_results["dependencies_available"]:
                print(f"   - Install dependencies: pip install scikit-learn pandas numpy joblib")
            if not validation_results["database_accessible"]:
                print(f"   - Run data collector: python data_collector.py")
            if not validation_results["models_available"]:
                print(f"   - Train models: python ml_signal_scanner.py --train")
        
        return validation_results
        
    except Exception as e:
        print(f"❌ System validation failed: {e}")
        return validation_results


def train_models_command(args) -> bool:
    """
    Execute model training command.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        True if training was successful
    """
    print(f"\n{Fore.CYAN}=== MODEL TRAINING ==={Style.RESET_ALL}")
    
    try:
        # Get symbols and timeframes
        symbols = REALTIME_CONFIG['specific_symbols'] if REALTIME_CONFIG['use_specific_symbols'] else []
        timeframes = REALTIME_CONFIG['timeframes']
        
        if not symbols:
            symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]
            print(f"⚠️  Using default symbols: {', '.join(symbols)}")
        
        print(f"🎯 Training models for:")
        print(f"   Symbols: {Fore.YELLOW}{', '.join(symbols)}{Style.RESET_ALL}")
        print(f"   Timeframes: {Fore.CYAN}{', '.join(timeframes)}{Style.RESET_ALL}")
        print(f"   Model types: {Fore.GREEN}{', '.join(args.model_types)}{Style.RESET_ALL}")
        
        if args.generate_datasets:
            print(f"\n📊 Generating datasets and training models...")
            results = generate_datasets_and_train(
                symbols=symbols,
                timeframes=timeframes,
                force_regeneration=args.force_regeneration
            )
        else:
            print(f"\n🤖 Training models on existing datasets...")
            trainer = ModelTrainer()
            results = trainer.train_all_models(symbols, timeframes, args.model_types)
        
        # Display results
        if results:
            print(f"\n{Fore.GREEN}✅ Training completed!{Style.RESET_ALL}")
            print(f"\n📈 Training Results:")
            
            total_models = 0
            for key, model_paths in results.items():
                symbol_tf = key.replace('_', ' - ')
                if model_paths:
                    print(f"   {Fore.YELLOW}{symbol_tf}{Style.RESET_ALL}: {len(model_paths)} models")
                    for path in model_paths:
                        model_name = os.path.basename(path)
                        print(f"      ➤ {model_name}")
                        total_models += 1
                else:
                    print(f"   {Fore.RED}{symbol_tf}{Style.RESET_ALL}: Failed")
            
            print(f"\n🎉 Total models trained: {Fore.GREEN}{total_models}{Style.RESET_ALL}")
            return True
        else:
            print(f"\n{Fore.RED}❌ No models were trained{Style.RESET_ALL}")
            return False
            
    except Exception as e:
        print(f"\n{Fore.RED}❌ Training failed: {e}{Style.RESET_ALL}")
        logging.error(f"Model training error: {e}")
        return False


def predict_signals_command(args) -> bool:
    """
    Execute signal prediction command.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        True if prediction was successful
    """
    print(f"\n{Fore.CYAN}=== SIGNAL PREDICTION ==={Style.RESET_ALL}")
    
    try:
        # Determine symbols and timeframes
        if args.symbol and args.timeframe:
            # Single symbol prediction
            symbols = [args.symbol]
            timeframes = [args.timeframe]
        elif args.scan_all:
            # Scan all configured symbols
            symbols = REALTIME_CONFIG['specific_symbols'] if REALTIME_CONFIG['use_specific_symbols'] else []
            timeframes = REALTIME_CONFIG['timeframes']
            
            if not symbols:
                symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]
        else:
            print(f"{Fore.RED}❌ Must specify either --symbol and --timeframe, or --scan-all{Style.RESET_ALL}")
            return False
        
        print(f"🔍 Scanning signals for:")
        print(f"   Symbols: {Fore.YELLOW}{', '.join(symbols)}{Style.RESET_ALL}")
        print(f"   Timeframes: {Fore.CYAN}{', '.join(timeframes)}{Style.RESET_ALL}")
        print(f"   Model: {Fore.GREEN}{args.model or 'default'}{Style.RESET_ALL}")
        
        # Make predictions
        predictor = ModelPredictor()
        results = []
        
        for symbol in symbols:
            for timeframe in timeframes:
                try:
                    result = predictor.predict_with_fallbacks(symbol, timeframe, args.model)
                    results.append(result)
                    
                    # Display result immediately
                    signal_color = {
                        "BUY": Fore.GREEN,
                        "SELL": Fore.RED,
                        "HOLD": Fore.YELLOW
                    }.get(result["signal"], Fore.WHITE)
                    
                    confidence_bar = "█" * int(result["confidence"] * 10)
                    
                    print(f"\n{signal_color}[{result['signal']}]{Style.RESET_ALL} {Fore.YELLOW}{symbol}{Style.RESET_ALL} ({timeframe})")
                    print(f"   Confidence: {confidence_bar} {result['confidence']:.1%}")
                    
                    if result.get("fallback_used"):
                        print(f"   {Fore.BLUE}ℹ️  Used fallback model{Style.RESET_ALL}")
                    
                    if result.get("error"):
                        print(f"   {Fore.RED}⚠️  Error: {result['error']}{Style.RESET_ALL}")
                    
                except Exception as e:
                    print(f"{Fore.RED}❌ Prediction failed for {symbol} ({timeframe}): {e}{Style.RESET_ALL}")
        
        # Save results if requested
        if args.save_to:
            save_results(results, args.save_to, args.output_format)
        
        # Display summary
        successful = len([r for r in results if r["success"]])
        total = len(results)
        
        print(f"\n{Fore.CYAN}=== PREDICTION SUMMARY ==={Style.RESET_ALL}")
        print(f"📊 Success rate: {successful}/{total} ({successful/total*100:.1f}%)")
        
        # Show signal distribution
        signal_counts = {}
        for result in results:
            if result["success"]:
                signal = result["signal"]
                signal_counts[signal] = signal_counts.get(signal, 0) + 1
        
        if signal_counts:
            print(f"📈 Signal distribution:")
            for signal, count in signal_counts.items():
                signal_color = {
                    "BUY": Fore.GREEN,
                    "SELL": Fore.RED,
                    "HOLD": Fore.YELLOW
                }.get(signal, Fore.WHITE)
                print(f"   {signal_color}{signal}{Style.RESET_ALL}: {count}")
        
        # Show performance stats
        stats = predictor.get_prediction_stats()
        if stats["fallback_rate"] > 0:
            print(f"⚠️  Fallback rate: {stats['fallback_rate']:.1%}")
        
        return successful > 0
        
    except Exception as e:
        print(f"\n{Fore.RED}❌ Prediction failed: {e}{Style.RESET_ALL}")
        logging.error(f"Signal prediction error: {e}")
        return False


def save_results(results: List[Dict], filepath: str, output_format: str):
    """
    Save prediction results to file.
    
    Args:
        results: List of prediction results
        filepath: Output file path
        output_format: Output format ('json' or 'csv')
    """
    try:
        if output_format.lower() == 'json':
            output_data = {
                "timestamp": datetime.now().isoformat(),
                "total_predictions": len(results),
                "successful_predictions": len([r for r in results if r["success"]]),
                "results": results
            }
            
            with open(filepath, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
                
        elif output_format.lower() == 'csv':
            import pandas as pd
            
            # Flatten results for CSV
            csv_data = []
            for result in results:
                csv_row = {
                    "timestamp": result["timestamp"],
                    "symbol": result["symbol"],
                    "timeframe": result["timeframe"],
                    "signal": result["signal"],
                    "confidence": result["confidence"],
                    "success": result["success"],
                    "model_used": result.get("model_used"),
                    "fallback_used": result.get("fallback_used", False),
                    "error": result.get("error", "")
                }
                csv_data.append(csv_row)
            
            df = pd.DataFrame(csv_data)
            df.to_csv(filepath, index=False)
        
        print(f"💾 Results saved to: {Fore.BLUE}{filepath}{Style.RESET_ALL}")
        
    except Exception as e:
        print(f"{Fore.RED}❌ Failed to save results: {e}{Style.RESET_ALL}")


def show_system_stats():
    """Display system statistics and health information."""
    print(f"\n{Fore.CYAN}=== SYSTEM STATISTICS ==={Style.RESET_ALL}")
    
    try:
        # Model information
        print(f"🤖 Available Models:")
        available_models = get_available_models()
        ensemble_models = get_ensemble_models()
        
        for model_name in available_models:
            config = MODEL_CONFIG[model_name]
            print(f"   • {Fore.GREEN}{model_name}{Style.RESET_ALL} v{config.get('version', 'unknown')}")
            print(f"     Type: {config['type']}, Accuracy: {config.get('accuracy', 'N/A')}")
        
        for ensemble_name in ensemble_models:
            config = MODEL_CONFIG[ensemble_name]
            print(f"   • {Fore.BLUE}{ensemble_name}{Style.RESET_ALL} v{config.get('version', 'unknown')} (ensemble)")
            print(f"     Models: {', '.join(config['models'])}")
        
        # Database information
        print(f"\n🗄️  Database Information:")
        try:
            from modules.data.db_manager import DB_FILE
            import sqlite3
            
            if os.path.exists(DB_FILE):
                file_size = os.path.getsize(DB_FILE) / (1024 * 1024)  # MB
                print(f"   • Database size: {file_size:.1f} MB")
                
                with sqlite3.connect(DB_FILE) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    tables = [row[0] for row in cursor.fetchall()]
                    print(f"   • Tables: {len(tables)}")
                    
                    # Count records in main tables
                    for table in tables[:5]:  # Show first 5 tables
                        try:
                            cursor.execute(f"SELECT COUNT(*) FROM {table}")
                            count = cursor.fetchone()[0]
                            print(f"     - {table}: {count:,} records")
                        except Exception:
                            pass
            else:
                print(f"   • Database not found")
                
        except Exception as e:
            print(f"   • Database error: {e}")
        
        # Configuration summary
        print(f"\n⚙️  Configuration:")
        print(f"   • Default model: {FALLBACK_CONFIG['default_model']}")
        print(f"   • Fallback enabled: {FALLBACK_CONFIG['enable_graceful_degradation']}")
        print(f"   • Symbols configured: {len(REALTIME_CONFIG.get('specific_symbols', []))}")
        print(f"   • Timeframes: {', '.join(REALTIME_CONFIG.get('timeframes', []))}")
        
    except Exception as e:
        print(f"{Fore.RED}❌ Error getting system stats: {e}{Style.RESET_ALL}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="TradingJii ML Signal Scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --validate-setup                    # Check system setup
  %(prog)s --train --generate-datasets         # Generate datasets and train models  
  %(prog)s --train --model-types random_forest # Train specific model type
  %(prog)s --predict --scan-all                # Scan all configured symbols
  %(prog)s --predict --symbol BTC/USDT:USDT --timeframe 4h  # Single prediction
  %(prog)s --predict --scan-all --save-to signals.json     # Save results
  %(prog)s --system-stats                      # Show system information
        """
    )
    
    # Main actions
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument('--validate-setup', action='store_true',
                              help='Validate system setup and dependencies')
    action_group.add_argument('--train', action='store_true',
                              help='Train ML models')
    action_group.add_argument('--predict', action='store_true',
                              help='Generate trading signals')
    action_group.add_argument('--system-stats', action='store_true',
                              help='Show system statistics')
    
    # Training options
    train_group = parser.add_argument_group('training options')
    train_group.add_argument('--generate-datasets', action='store_true',
                             help='Generate datasets before training')
    train_group.add_argument('--force-regeneration', action='store_true',
                             help='Force dataset regeneration even if they exist')
    train_group.add_argument('--model-types', nargs='+', 
                             choices=['random_forest', 'gradient_boosting', 'svm', 'logistic_regression'],
                             default=['random_forest'],
                             help='Model types to train')
    
    # Prediction options
    predict_group = parser.add_argument_group('prediction options')
    predict_group.add_argument('--scan-all', action='store_true',
                               help='Scan all configured symbols and timeframes')
    predict_group.add_argument('--symbol', type=str,
                               help='Specific symbol to predict (requires --timeframe)')
    predict_group.add_argument('--timeframe', type=str,
                               help='Specific timeframe to predict (requires --symbol)')
    predict_group.add_argument('--model', type=str,
                               help='Specific model to use (default: configured default)')
    
    # Output options
    output_group = parser.add_argument_group('output options')
    output_group.add_argument('--save-to', type=str,
                              help='Save results to file')
    output_group.add_argument('--output-format', choices=['json', 'csv'], default='json',
                              help='Output format for saved results')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_cli_logging()
    
    # Print header
    print(f"{Fore.BLUE}╔══════════════════════════════════════════════════════════════╗{Style.RESET_ALL}")
    print(f"{Fore.BLUE}║                    TradingJii ML Signal Scanner              ║{Style.RESET_ALL}")
    print(f"{Fore.BLUE}║                   Production-Ready ML System                 ║{Style.RESET_ALL}")
    print(f"{Fore.BLUE}╚══════════════════════════════════════════════════════════════╝{Style.RESET_ALL}")
    
    success = False
    
    try:
        if args.validate_setup:
            validation_results = validate_system_setup()
            success = sum(validation_results.values()) >= len(validation_results) - 1
            
        elif args.train:
            success = train_models_command(args)
            
        elif args.predict:
            success = predict_signals_command(args)
            
        elif args.system_stats:
            show_system_stats()
            success = True
            
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}⚠️  Operation cancelled by user{Style.RESET_ALL}")
        success = False
    except Exception as e:
        print(f"\n{Fore.RED}❌ Unexpected error: {e}{Style.RESET_ALL}")
        logger.error(f"CLI error: {e}")
        success = False
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
