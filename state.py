# state.py

from typing import List, Dict, Optional
import ccxt.async_support as ccxt_async

class AppState:
    """Classe singleton per gestire lo stato globale dell'applicazione"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AppState, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        # Configurazioni di base
        self.enabled_timeframes: List[str] = ["5m", "15m", "30m", "1h", "4h"]
        self.timeframe_default: str = "15m"
        self.leverage: int = 5
        self.margin_usdt: float = 10.0
        self.top_train_crypto: int = 10
        self.top_analysis_crypto: int = 5
        self.trade_cycle_interval: int = 300
        self.train_if_not_found: bool = True
        self.excluded_symbols: List[str] = ["USDC", "BUSD", "DAI", "TUSD"]

        # Configurazioni dei modelli
        self.model_rates: Dict[str, float] = {
            "lstm": 0.4,
            "rf":   0.3,
            "xgb":  0.3
        }
        self.time_steps: int = 60
        self.expected_columns: List[str] = [
            "timestamp", "open", "high", "low", "close", "volume",
            "ema5", "ema10", "ema20",
            "macd", "macd_signal", "macd_histogram",
            "rsi_fast", "stoch_rsi",
            "atr",
            "bollinger_hband", "bollinger_lband", "bollinger_pband",
            "vwap", "adx",
            "roc", "log_return",
            "tenkan_sen", "kijun_sen", "senkou_span_a", "senkou_span_b", "chikou_span",
            "williams_r", "obv",
            "sma_fast", "sma_slow", "sma_fast_trend", "sma_slow_trend", "sma_cross",
            "close_lag_1", "volume_lag_1",
            "weekday_sin", "weekday_cos", "hour_sin", "hour_cos"
        ]

        # Stato runtime
        self.async_exchange: Optional[ccxt_async.Exchange] = None
        self.bot_running: bool = False
        self.initialized: bool = False
        self.selected_models: List[str] = ["lstm", "rf", "xgb"]
        self.current_config = None

    def update_from_config(self, config: dict):
        """Aggiorna lo stato con i valori dalla configurazione"""
        if "timeframes" in config:
            self.enabled_timeframes = config["timeframes"]
            self.timeframe_default  = config["timeframes"][0]
        if "models" in config:
            self.selected_models   = config["models"]
        if "trading_params" in config:
            params = config["trading_params"]
            if "leverage"           in params: self.leverage           = params["leverage"]
            if "margin_usdt"        in params: self.margin_usdt        = params["margin_usdt"]
            if "top_analysis_crypto" in params: self.top_analysis_crypto = params["top_analysis_crypto"]

        self.current_config = config

    def reset(self):
        """Resetta lo stato ai valori di default"""
        self._initialize()

# Istanza globale
app_state = AppState()
