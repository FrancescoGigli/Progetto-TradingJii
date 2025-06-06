#!/usr/bin/env python3
"""
Logging setup module for TradingJii

Contains custom logging configuration, including colored console output and file logging.
"""

import logging
import os
from pathlib import Path
from datetime import datetime
from colorama import init, Fore, Style, Back

# Initialize colorama
init(autoreset=True)

class ColoredFormatter(logging.Formatter):
    """
    Custom formatter for console logging with colored output.
    
    Different logging levels get different colors:
    - DEBUG: Cyan
    - INFO: Green
    - WARNING: Yellow
    - ERROR: Red
    - CRITICAL: Red on White background
    """
    FORMATS = {
        logging.DEBUG: Fore.CYAN + "%(asctime)s %(levelname)s: %(message)s" + Style.RESET_ALL,
        logging.INFO: Fore.GREEN + "%(asctime)s %(levelname)s: %(message)s" + Style.RESET_ALL,
        logging.WARNING: Fore.YELLOW + "%(asctime)s %(levelname)s: %(message)s" + Style.RESET_ALL,
        logging.ERROR: Fore.RED + "%(asctime)s %(levelname)s: %(message)s" + Style.RESET_ALL,
        logging.CRITICAL: Fore.RED + Back.WHITE + "%(asctime)s %(levelname)s: %(message)s" + Style.RESET_ALL
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)

class FileFormatter(logging.Formatter):
    """
    Custom formatter for file logging without colors.
    """
    def __init__(self):
        super().__init__(
            fmt="%(asctime)s %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

def setup_logging(level=logging.INFO, log_to_file=False, log_dir="logs", script_name=None):
    """
    Set up logging with colored console output and optional file logging.
    
    Args:
        level: The logging level (default: INFO)
        log_to_file: Whether to also log to a file (default: False)
        log_dir: Directory to store log files (default: "logs")
        script_name: Name of the script for log filename (default: None)
    
    Returns:
        The configured logger
    """
    # Clear existing handlers
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.handlers.clear()
    
    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter())
    logger.addHandler(console_handler)
    
    # File handler (if requested)
    if log_to_file:
        # Create logs directory if it doesn't exist
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)
        
        # Generate log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if script_name:
            log_filename = f"{script_name}_{timestamp}.log"
        else:
            log_filename = f"training_{timestamp}.log"
        
        log_file = log_path / log_filename
        
        # File handler without colors
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(FileFormatter())
        logger.addHandler(file_handler)
        
        # Log the file path
        logger.info(f"📁 Logging to file: {log_file}")
    
    return logger

def setup_training_logging(level=logging.INFO, timeframe=None, symbol=None):
    """
    Set up logging specifically for training scripts with automatic file logging.
    
    Args:
        level: The logging level (default: INFO)
        timeframe: Training timeframe (e.g., "1h", "4h")
        symbol: Symbol being trained (optional)
    
    Returns:
        The configured logger and log file path
    """
    # Create logs directory structure
    log_dir = Path("logs") / "training"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate descriptive filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_parts = ["training"]
    
    if timeframe:
        filename_parts.append(timeframe)
    if symbol:
        filename_parts.append(symbol)
    
    filename_parts.append(timestamp)
    log_filename = "_".join(filename_parts) + ".log"
    log_file = log_dir / log_filename
    
    # Clear existing handlers
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.handlers.clear()
    
    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter())
    logger.addHandler(console_handler)
    
    # File handler without colors
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(FileFormatter())
    logger.addHandler(file_handler)
    
    # Log the file path
    logger.info(f"📁 Training log saved to: {log_file}")
    
    return logger, log_file
