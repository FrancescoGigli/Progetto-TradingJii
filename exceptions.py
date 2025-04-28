# exceptions.py

class TradingError(Exception):
    """Classe base per gli errori del trading bot"""
    pass

class ExchangeError(TradingError):
    """Errori relativi all'exchange"""
    pass

class ModelError(TradingError):
    """Errori relativi ai modelli di ML"""
    pass

class ConfigurationError(TradingError):
    """Errori relativi alla configurazione"""
    pass

class DataError(TradingError):
    """Errori relativi ai dati"""
    pass

class ValidationError(TradingError):
    """Errori di validazione"""
    pass

class AuthenticationError(TradingError):
    """Errori di autenticazione"""
    pass

class InsufficientBalanceError(TradingError):
    """Errore per saldo insufficiente"""
    pass

class PositionError(TradingError):
    """Errori relativi alle posizioni"""
    pass
