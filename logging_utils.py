# logging_utils.py

import logging
import sys
from termcolor import colored
import os
from datetime import datetime
from typing import Optional

# Formatter colorato per console
class ColorFormatter(logging.Formatter):
    LEVEL_COLOR = {
        'DEBUG': 'blue',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'magenta'
    }

    def format(self, record):
        color = self.LEVEL_COLOR.get(record.levelname, 'white')
        record.levelname = colored(record.levelname, color)
        return super().format(record)

# Handler console
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(ColorFormatter("%(asctime)s %(levelname)s %(message)s"))

# Handler file (no colori, append mode)
file_handler = logging.FileHandler("trading_bot_derivatives.log", mode="a", encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))

logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler],
)

# Wrapper di comodo per i log
def log_info(message: str):
    logging.info(colored(message, "green"))

def log_warning(message: str):
    logging.warning(colored(message, "yellow"))

def log_error(message: str):
    logging.error(colored(message, "red"))

def log_debug(message: str):
    logging.debug(colored(message, "blue"))

def log_success(message: str):
    log_info(f"[OK] {message}")

def log_failure(message: str, error: Exception = None):
    if error:
        log_error(f"[FAIL] {message}: {error}")
    else:
        log_error(f"[FAIL] {message}")

def log_trade_open(trade_id: str):
    log_info(f"[TRADE OPEN] {trade_id}")

def log_trade_close(trade_id: str):
    log_info(f"[TRADE CLOSE] {trade_id}")

def log_position_size(symbol: str, size: float, margin: float):
    log_info(f"[SIZE] {symbol}: {size} contracts (margin={margin})")

def setup_logging(
    log_level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> logging.Logger:
    """
    Configura il logging per l'applicazione
    
    Args:
        log_level: Livello di logging (default: INFO)
        log_file: Percorso del file di log (opzionale)
        log_format: Formato del messaggio di log
        
    Returns:
        Logger configurato
    """
    # Crea il logger
    logger = logging.getLogger("trading_bot")
    logger.setLevel(log_level)
    
    # Formatta il messaggio
    formatter = logging.Formatter(log_format)
    
    # Handler per la console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler per il file se specificato
    if log_file:
        # Crea la directory dei log se non esiste
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger() -> logging.Logger:
    """
    Restituisce il logger configurato
    
    Returns:
        Logger configurato
    """
    return logging.getLogger("trading_bot")

# Configurazione di default
default_log_file = os.path.join(
    "logs",
    f"trading_bot_{datetime.now().strftime('%Y%m%d')}.log"
)

# Crea il logger di default
logger = setup_logging(log_file=default_log_file)
