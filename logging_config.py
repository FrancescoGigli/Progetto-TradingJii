import logging
import sys
from termcolor import colored

class ColorFormatter(logging.Formatter):
    # Mappa dei livelli di log con colori
    LEVEL_COLOR = {
        'DEBUG': 'blue',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'magenta'
    }
    
    def format(self, record):
        # Seleziona il colore in base al livello del log
        color = self.LEVEL_COLOR.get(record.levelname, 'white')
        
        # Colora il livello del log
        record.levelname = colored(record.levelname, color)
        return super().format(record)

# Handler per la console con il formatter personalizzato
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(ColorFormatter("%(asctime)s %(levelname)s %(message)s"))

# Handler per il file (senza colori)
file_handler = logging.FileHandler("trading_bot_derivatives.log", mode='w', encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))

logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler],
)