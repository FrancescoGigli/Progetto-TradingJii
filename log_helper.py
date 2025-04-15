import logging
from termcolor import colored

# Funzioni di log sicure senza emoji
def log_info(message, color="green"):
    """Wrapper sicuro per logging.info senza emoji"""
    logging.info(colored(message, color))
    
def log_warning(message, color="yellow"):
    """Wrapper sicuro per logging.warning senza emoji"""
    logging.warning(colored(message, color))
    
def log_error(message, color="red"):
    """Wrapper sicuro per logging.error senza emoji"""
    logging.error(colored(message, color))
    
def log_debug(message, color="blue"):
    """Wrapper sicuro per logging.debug senza emoji"""
    logging.debug(colored(message, color))

# Sostituzioni per messaggi comuni
def log_success(message):
    """Log di un'operazione completata con successo"""
    log_info(f"[OK] {message}", "green")
    
def log_failure(message, error=None):
    """Log di un'operazione fallita"""
    if error:
        log_error(f"[FAIL] {message}: {error}", "red")
    else:
        log_error(f"[FAIL] {message}", "red")
        
def log_trade_open(trade):
    """Log di un trade aperto"""
    log_info(f"[TRADE] Trade aperto: {trade}", "green")
    
def log_trade_close(trade):
    """Log di un trade chiuso"""
    log_info(f"[TRADE] Trade chiuso: {trade}", "green")
    
def log_position_size(symbol, size, margin):
    """Log della dimensione posizione"""
    log_info(f"[SIZE] Dimensione posizione per {symbol}: {size} contratti (Margine = {margin})", "cyan") 