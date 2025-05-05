# logging_config.py

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
import datetime

# Configure the main logger
LOG_LEVEL = logging.INFO  # Change to logging.DEBUG for more verbose output

# Configure logger - this will be used by all modules
def configure_logging():
    """Configure and set up the logger for the application."""
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(LOG_LEVEL)
    
    # Remove any existing handlers to avoid duplicate messages
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s %(levelname)s %(message)s'
    )
    
    # Define the log file path
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"crypto_bot.log"
    
    # Create file handler for detailed logging
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5  # 10MB max size, keep 5 backups
    )
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(detailed_formatter)
    
    # Create console handler with a higher log level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(LOG_LEVEL)
    
    # Use colored output for console
    class ColoredFormatter(logging.Formatter):
        """Colored formatter for terminal output."""
        COLORS = {
            'DEBUG': '\033[94m',  # Blue
            'INFO': '\033[92m',   # Green
            'WARNING': '\033[93m', # Yellow
            'ERROR': '\033[91m',  # Red
            'CRITICAL': '\033[91m\033[1m',  # Bold Red
            'RESET': '\033[0m'    # Reset
        }
        
        EMOJIS = {
            'DEBUG': '🔧 ',
            'INFO': 'ℹ️ ',
            'WARNING': '⚠️ ',
            'ERROR': '❌ ',
            'CRITICAL': '🔥 '
        }
        
        def format(self, record):
            log_level = record.levelname
            emoji = self.EMOJIS.get(log_level, '')
            color = self.COLORS.get(log_level, self.COLORS['RESET'])
            reset = self.COLORS['RESET']
            
            message = f"{record.asctime} {log_level} {emoji}{color}{record.getMessage()}{reset}"
            return message
    
    console_formatter = ColoredFormatter(
        '%(asctime)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    
    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Setup logger for backward compatibility
def setup_logger():
    """Legacy function that calls configure_logging for backward compatibility."""
    return configure_logging()

# Initialize a logger when this module is imported
logger = setup_logger()
