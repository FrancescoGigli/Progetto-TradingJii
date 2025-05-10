#!/usr/bin/env python3
"""
Logging setup module for TradingJii

Contains custom logging configuration, including colored console output.
"""

import logging
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

def setup_logging(level=logging.INFO):
    """
    Set up logging with colored output.
    
    Args:
        level: The logging level (default: INFO)
    
    Returns:
        The configured logger
    """
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter())
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.handlers.clear()
    logger.addHandler(handler)
    return logger
