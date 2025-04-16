#!/usr/bin/env python3
import logging
import sys
import os
from termcolor import colored

# Importazione di colorama per il supporto dei colori su Windows
try:
    import colorama
    colorama.init()
    COLORED_OUTPUT = True
except ImportError:
    COLORED_OUTPUT = False
    print("Per avere output colorato, installa colorama: pip install colorama")

# Configura il logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Crea un formatter personalizzato per i log colorati
class ColoredFormatter(logging.Formatter):
    """Un formattatore personalizzato per mostrare i log con colori."""
    
    # Mappatura dei livelli di log ai colori
    COLORS = {
        logging.DEBUG: 'blue',
        logging.INFO: 'cyan',
        logging.WARNING: 'yellow',
        logging.ERROR: 'red',
        logging.CRITICAL: 'magenta',
    }
    
    def format(self, record):
        # Formatta il messaggio normalmente
        message = super().format(record)
        
        # Colora in base al livello di log
        if hasattr(record, 'levelno'):
            color = self.COLORS.get(record.levelno, 'white')
            return colored(message, color)
        return message

# Configura il logger root
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Rimuovi gli handler esistenti
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Crea un nuovo handler per la console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Imposta il formattatore personalizzato
if COLORED_OUTPUT:
    formatter = ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
else:
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Se specificato, aggiungi anche un handler per i log su file
log_file = os.environ.get('TRADING_BOT_LOG_FILE')
if log_file:
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

# Funzione di utilità per avere log colorati programmaticamente
def log_colored(level, message, color=None):
    """
    Registra un messaggio colorato.
    
    Args:
        level: Il livello di log (info, warning, error, ecc.)
        message: Il messaggio da registrare
        color: Il colore da utilizzare (opzionale, se non specificato usa il colore predefinito per il livello)
    """
    if color:
        colored_message = colored(message, color)
    else:
        colored_message = message
    
    if level == 'info':
        logging.info(colored_message)
    elif level == 'warning':
        logging.warning(colored_message)
    elif level == 'error':
        logging.error(colored_message)
    elif level == 'debug':
        logging.debug(colored_message)
    elif level == 'critical':
        logging.critical(colored_message)