import logging
import sys
from termcolor import colored

class EmojiFormatter(logging.Formatter):
    # Mappa dei livelli di log con emoji e colori
    LEVEL_EMOJI = {
        'DEBUG': '🐛',
        'INFO': 'ℹ️',
        'WARNING': '⚠️',
        'ERROR': '❌',
        'CRITICAL': '🚨'
    }
    LEVEL_COLOR = {
        'DEBUG': 'blue',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'magenta'
    }
    
    def format(self, record):
        # Seleziona l'emoji e il colore in base al livello del log
        emoji = self.LEVEL_EMOJI.get(record.levelname, '')
        color = self.LEVEL_COLOR.get(record.levelname, 'white')
        
        # Non colora tutto il messaggio: colora solo l'emoji
        record.msg = f"{colored(emoji, color)} {record.msg}"
        return super().format(record)

# Handler per la console con il formatter personalizzato
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(EmojiFormatter("%(asctime)s %(levelname)s %(message)s"))

# Handler per il file (senza colori)
file_handler = logging.FileHandler("trading_bot_derivatives.log", mode='w', encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))

logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler],
)