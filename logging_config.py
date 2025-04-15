import logging
import sys
from termcolor import colored

class LevelFormatter(logging.Formatter):
    # Mappa dei livelli di log con prefissi e colori
    LEVEL_PREFIX = {
        'DEBUG': '[DEBUG]',
        'INFO': '[INFO]',
        'WARNING': '[WARN]',
        'ERROR': '[ERROR]',
        'CRITICAL': '[CRIT]'
    }
    LEVEL_COLOR = {
        'DEBUG': 'blue',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'magenta'
    }
    
    def format(self, record):
        # Seleziona il prefisso e il colore in base al livello del log
        prefix = self.LEVEL_PREFIX.get(record.levelname, '')
        color = self.LEVEL_COLOR.get(record.levelname, 'white')
        
        # Non colora tutto il messaggio: colora solo il prefisso
        record.msg = f"{colored(prefix, color)} {record.msg}"
        return super().format(record)

# Handler per la console con il formatter personalizzato
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(LevelFormatter("%(asctime)s %(message)s"))

# Handler per il file (senza colori)
file_handler = logging.FileHandler("trading_bot_derivatives.log", mode='w', encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))

logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler],
)