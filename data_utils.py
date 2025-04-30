#data_utils.py
import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
from ta.trend import EMAIndicator, MACD, ADXIndicator, IchimokuIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice
from ta.momentum import StochRSIIndicator, WilliamsRIndicator
from ta.volume import OnBalanceVolumeIndicator

# Import per i nuovi indicatori
from ta.volume import MFIIndicator
from ta.trend import CCIIndicator

from config import EXPECTED_COLUMNS

def add_technical_indicators(df):
    # Creare una copia del DataFrame per evitare warning o modifiche indesiderate
    df = df.copy()
    
    # Verifica se l'indice è già di tipo datetime
    if isinstance(df.index, pd.DatetimeIndex):
        # Usa l'indice esistente
        timestamp_index = df.index
    else:
        # Se l'indice non è datetime e c'è una colonna timestamp, prova a usarla
        if 'timestamp' in df.columns:
            try:
                timestamp_index = pd.to_datetime(df['timestamp'])
                # Non impostiamo l'indice del DataFrame, ma usiamo timestamp_index per le operazioni
            except Exception as e:
                print(f"Errore nella conversione della colonna timestamp a datetime: {e}")
                # Creiamo un indice fittizio
                timestamp_index = pd.date_range(start='2023-01-01', periods=len(df), freq='1H')
        else:
            # Nessuna colonna timestamp, creiamo un indice fittizio
            timestamp_index = pd.date_range(start='2023-01-01', periods=len(df), freq='1H')
    
    # Calcolo delle EMA
    try:
        df['ema5'] = EMAIndicator(df['close'], window=5).ema_indicator()
    except Exception:
        df['ema5'] = 0.0
    try:
        df['ema10'] = EMAIndicator(df['close'], window=10).ema_indicator()
    except Exception:
        df['ema10'] = 0.0
    try:
        df['ema20'] = EMAIndicator(df['close'], window=20).ema_indicator()
    except Exception:
        df['ema20'] = 0.0

    # Calcolo del MACD e del suo segnale
    try:
        macd = MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
    except Exception:
        df['macd'] = 0.0
        df['macd_signal'] = 0.0
        df['macd_histogram'] = 0.0

    # Calcolo degli indicatori di momentum
    try:
        df['rsi_fast'] = momentum.RSIIndicator(df['close'], window=7).rsi()
    except Exception:
        df['rsi_fast'] = 0.0

    try:
        df['stoch_rsi'] = StochRSIIndicator(df['close'], window=14).stochrsi()
    except Exception:
        df['stoch_rsi'] = 0.0

    # Calcolo dell'ATR
    try:
        df['atr'] = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    except Exception:
        df['atr'] = 0.0

    # Bollinger Bands e Bollinger %B
    try:
        bollinger = BollingerBands(df['close'], window=20, window_dev=2)
        df['bollinger_hband'] = bollinger.bollinger_hband()
        df['bollinger_lband'] = bollinger.bollinger_lband()
        df['bollinger_pband'] = bollinger.bollinger_pband()
    except Exception:
        df['bollinger_hband'] = 0.0
        df['bollinger_lband'] = 0.0
        df['bollinger_pband'] = 0.0

    # VWAP
    try:
        df['vwap'] = VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume'], window=14).volume_weighted_average_price()
    except Exception:
        df['vwap'] = 0.0

    # ADX
    try:
        df['adx'] = ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
    except Exception:
        df['adx'] = 0.0

    # Rate of Change e log return
    try:
        df['roc'] = df['close'].pct_change(periods=1)
    except Exception:
        df['roc'] = 0.0
    try:
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    except Exception:
        df['log_return'] = 0.0

    # Ichimoku
    try:
        # Manual implementation of Ichimoku components
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        period9_high = df['high'].rolling(window=9).max()
        period9_low = df['low'].rolling(window=9).min()
        df['tenkan_sen'] = (period9_high + period9_low) / 2
        
        # Kijun-sen (Base Line): (26-period high + 26-period low)/2
        period26_high = df['high'].rolling(window=26).max()
        period26_low = df['low'].rolling(window=26).min()
        df['kijun_sen'] = (period26_high + period26_low) / 2
        
        # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
        
        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
        period52_high = df['high'].rolling(window=52).max()
        period52_low = df['low'].rolling(window=52).min()
        df['senkou_span_b'] = ((period52_high + period52_low) / 2).shift(26)
        
        # Chikou Span (Lagging Span): Close price shifted back 26 periods
        df['chikou_span'] = df['close'].shift(-26)
            
    except Exception as e:
        print(f"Error calculating Ichimoku indicators: {e}")
        df['tenkan_sen'] = 0.0
        df['kijun_sen'] = 0.0
        df['senkou_span_a'] = 0.0
        df['senkou_span_b'] = 0.0
        df['chikou_span'] = 0.0

    # Altri indicatori
    try:
        df['williams_r'] = WilliamsRIndicator(df['high'], df['low'], df['close'], lbp=14).williams_r()
    except Exception:
        df['williams_r'] = 0.0

    try:
        df['obv'] = OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
    except Exception:
        df['obv'] = 0.0

    # Nuovi indicatori: Money Flow Index (MFI) e Commodity Channel Index (CCI)
    try:
        mfi = MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=14)
        df['mfi'] = mfi.money_flow_index()
    except Exception:
        df['mfi'] = 0.0

    try:
        cci = CCIIndicator(high=df['high'], low=df['low'], close=df['close'], window=20, constant=0.015)
        df['cci'] = cci.cci()
    except Exception:
        df['cci'] = 0.0

    # Calcolo delle SMA: veloce (window=10) e lenta (window=50)
    df['sma_fast'] = df['close'].rolling(window=10, min_periods=10).mean()
    df['sma_slow'] = df['close'].rolling(window=50, min_periods=50).mean()

    # Calcolo del trend per SMA
    df['sma_fast_trend'] = 0
    df['sma_slow_trend'] = 0
    df.loc[df['sma_fast'] > df['sma_fast'].shift(1), 'sma_fast_trend'] = 1
    df.loc[df['sma_fast'] < df['sma_fast'].shift(1), 'sma_fast_trend'] = -1
    df.loc[df['sma_slow'] > df['sma_slow'].shift(1), 'sma_slow_trend'] = 1
    df.loc[df['sma_slow'] < df['sma_slow'].shift(1), 'sma_slow_trend'] = -1

    # Crossover tra SMA veloce e lenta
    df['sma_cross'] = 0
    df.loc[df['sma_fast'] > df['sma_slow'], 'sma_cross'] = 1
    df.loc[df['sma_fast'] < df['sma_slow'], 'sma_cross'] = -1

    # Lag features: close e volume con ritardo di 1 periodo
    df['close_lag_1'] = df['close'].shift(1)
    df['volume_lag_1'] = df['volume'].shift(1)

    # Codifica ciclica delle informazioni temporali
    # Usiamo timestamp_index che è stata creata all'inizio
    try:
        df['weekday_sin'] = np.sin(2 * np.pi * timestamp_index.dayofweek / 7)
        df['weekday_cos'] = np.cos(2 * np.pi * timestamp_index.dayofweek / 7)
        df['hour_sin'] = np.sin(2 * np.pi * timestamp_index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * timestamp_index.hour / 24)
    except Exception as e:
        print(f"Errore nel calcolo delle feature cicliche: {e}")
        df['weekday_sin'] = 0
        df['weekday_cos'] = 0
        df['hour_sin'] = 0
        df['hour_cos'] = 0

    # Propagazione forward e backward per gestire eventuali NaN
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    
    # Verifica quali colonne sono effettivamente presenti in EXPECTED_COLUMNS
    missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns and col != 'timestamp']
    if missing_columns:
        print(f"Attenzione: le seguenti colonne sono mancanti: {missing_columns}")
        # Aggiungi colonne mancanti con valore 0
        for col in missing_columns:
            df[col] = 0.0
    
    # Restituisci il DataFrame con le colonne attese
    columns_to_return = [col for col in EXPECTED_COLUMNS if col != 'timestamp' and col in df.columns]
    return df[columns_to_return].round(4)

def prepare_data(df):
    required_initial = ['open', 'high', 'low', 'close', 'volume']
    for col in required_initial:
        if col not in df.columns:
            raise ValueError(f"Column {col} missing in input data.")
    
    # Assicuriamoci che 'timestamp' non causi problemi
    if 'timestamp' in df.columns:
        # Salva timestamp solo se necessario per altri calcoli
        timestamp_col = df['timestamp']
        df = df.drop(columns=['timestamp'])
        
    df = add_technical_indicators(df)
    
    # Verifica quali colonne sono in EXPECTED_COLUMNS e presenti in df
    valid_cols = [col for col in EXPECTED_COLUMNS if col != 'timestamp' and col in df.columns]
    
    # Restituisci solo le colonne valide
    return df[valid_cols].values

