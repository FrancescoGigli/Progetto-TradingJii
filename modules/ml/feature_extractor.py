#!/usr/bin/env python3
"""
Feature Extractor Module for TradingJii ML

This module extracts and validates features for ML predictions by integrating with:
- Existing volatility data from volatility_processor
- Technical indicators from indicator_processor  
- OHLCV data from db_manager
- Schema validation and error handling
"""

import sqlite3
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from colorama import Fore, Style

# Import existing TradingJii modules
from modules.data.db_manager import DB_FILE
from modules.data.series_segmenter import categorize_series
from modules.utils.config import REALTIME_CONFIG

# Import ML specific modules
from .config import (
    FEATURE_CONFIG, 
    MODEL_CONFIG, 
    VALIDATION_CONFIG,
    generate_schema_hash,
    FeatureValidationError,
    SchemaValidationError,
    InsufficientDataError
)

def load_volatility_series(symbol: str, timeframe: str) -> pd.DataFrame:
    """
    Load volatility data from the volatility table.
    
    Args:
        symbol: Cryptocurrency symbol
        timeframe: Timeframe
        
    Returns:
        DataFrame with timestamp and volatility columns
    """
    table_name = f"volatility_{timeframe}"
    
    try:
        with sqlite3.connect(DB_FILE) as conn:
            query = f"""
                SELECT timestamp, volatility
                FROM {table_name}
                WHERE symbol = ?
                ORDER BY timestamp ASC
            """
            
            df = pd.read_sql_query(query, conn, params=(symbol,))
            
            if df.empty:
                logging.warning(f"No volatility data found for {symbol} in timeframe {timeframe}")
                return pd.DataFrame(columns=['timestamp', 'volatility'])
                
            return df
            
    except Exception as e:
        logging.error(f"Error loading volatility series for {symbol} ({timeframe}): {e}")
        return pd.DataFrame(columns=['timestamp', 'volatility'])

def extract_volatility_features(symbol: str, timeframe: str, window_size: int = 7) -> pd.DataFrame:
    """
    Extract volatility-based features using existing volatility processor.
    
    Args:
        symbol: Cryptocurrency symbol (e.g., 'BTC/USDT:USDT')
        timeframe: Timeframe (e.g., '1h', '4h')
        window_size: Size of volatility window for pattern analysis
        
    Returns:
        DataFrame with volatility features
    """
    try:
        # Use existing volatility data loader
        df = load_volatility_series(symbol, timeframe)
        
        if df.empty:
            logging.warning(f"No volatility data available for {symbol} ({timeframe})")
            return pd.DataFrame()
        
        # Create sliding windows like in the existing dataset_generator
        features_list = []
        
        if len(df) >= window_size:
            for i in range(len(df) - window_size + 1):
                window = df['volatility'].iloc[i:i+window_size].values
                timestamp = df['timestamp'].iloc[i+window_size-1]
                
                # Create feature row with window values
                feature_row = {f'x_{j+1}': window[j] for j in range(window_size)}
                feature_row['timestamp'] = timestamp
                
                # Add derived features
                feature_row['vol_trend'] = np.mean(np.diff(window))  # Trend direction
                feature_row['vol_std'] = np.std(window)              # Volatility of volatility
                feature_row['vol_min'] = np.min(window)              # Minimum in window
                feature_row['vol_max'] = np.max(window)              # Maximum in window
                feature_row['vol_range'] = feature_row['vol_max'] - feature_row['vol_min']
                
                # Pattern classification (using existing categorize_series)
                pattern = categorize_series(window.tolist(), threshold=0.0)
                feature_row['vol_pattern'] = pattern
                
                features_list.append(feature_row)
        
        if features_list:
            features_df = pd.DataFrame(features_list)
            features_df['timestamp'] = pd.to_datetime(features_df['timestamp'])
            
            logging.debug(f"Extracted {len(features_df)} volatility feature rows for {symbol} ({timeframe})")
            return features_df
        else:
            logging.warning(f"Insufficient data to create volatility features for {symbol} ({timeframe})")
            return pd.DataFrame()
            
    except Exception as e:
        logging.error(f"Error extracting volatility features for {symbol} ({timeframe}): {e}")
        return pd.DataFrame()

def extract_technical_analysis_features(symbol: str, timeframe: str, lookback_periods: int = 50) -> pd.DataFrame:
    """
    Extract technical analysis features from existing indicator tables.
    
    Args:
        symbol: Cryptocurrency symbol
        timeframe: Timeframe
        lookback_periods: Number of periods to look back
        
    Returns:
        DataFrame with TA features
    """
    try:
        table_name = f"ta_{timeframe}".replace('-', '_')
        
        with sqlite3.connect(DB_FILE) as conn:
            # Get the latest technical analysis data
            query = f"""
                SELECT timestamp, rsi14, macd, macd_signal, macd_hist,
                       ema20, ema50, ema200, sma20, sma50,
                       atr14, bbands_upper, bbands_middle, bbands_lower,
                       stoch_k, stoch_d, adx14, obv, vwap, volume_sma20
                FROM {table_name}
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """
            
            df = pd.read_sql_query(query, conn, params=(symbol, lookback_periods))
            
            if df.empty:
                logging.warning(f"No technical analysis data available for {symbol} ({timeframe})")
                return pd.DataFrame()
            
            # Sort by timestamp ascending for feature calculation
            df = df.sort_values('timestamp')
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Add derived TA features
            if 'ema20' in df.columns and not df['ema20'].isna().all():
                # EMA slope (trend strength)
                df['ema20_slope'] = df['ema20'].diff()
                df['ema50_slope'] = df['ema50'].diff() if 'ema50' in df.columns else 0
                
                # EMA crossovers
                if 'ema50' in df.columns:
                    df['ema_cross'] = (df['ema20'] > df['ema50']).astype(int)
            
            # RSI momentum features
            if 'rsi14' in df.columns and not df['rsi14'].isna().all():
                df['rsi_oversold'] = (df['rsi14'] < 30).astype(int)
                df['rsi_overbought'] = (df['rsi14'] > 70).astype(int)
                df['rsi_momentum'] = df['rsi14'].diff()
            
            # MACD features
            if 'macd' in df.columns and 'macd_signal' in df.columns:
                df['macd_bullish'] = (df['macd'] > df['macd_signal']).astype(int)
                df['macd_cross'] = ((df['macd'] > df['macd_signal']) != 
                                   (df['macd'].shift(1) > df['macd_signal'].shift(1))).astype(int)
            
            # Bollinger Bands features
            if all(col in df.columns for col in ['bbands_upper', 'bbands_middle', 'bbands_lower']):
                # We'll need price data for this - skip for now or get from OHLCV table
                pass
            
            logging.debug(f"Extracted {len(df)} TA feature rows for {symbol} ({timeframe})")
            return df
            
    except Exception as e:
        logging.error(f"Error extracting TA features for {symbol} ({timeframe}): {e}")
        return pd.DataFrame()

def combine_features(
    volatility_features: pd.DataFrame, 
    ta_features: pd.DataFrame,
    merge_tolerance_minutes: int = 30
) -> pd.DataFrame:
    """
    Combine volatility and technical analysis features by timestamp.
    
    Args:
        volatility_features: DataFrame with volatility features
        ta_features: DataFrame with TA features  
        merge_tolerance_minutes: Tolerance for timestamp matching
        
    Returns:
        Combined DataFrame with all features
    """
    try:
        if volatility_features.empty and ta_features.empty:
            return pd.DataFrame()
        
        if volatility_features.empty:
            return ta_features
        
        if ta_features.empty:
            return volatility_features
        
        # Ensure timestamp columns exist and are datetime
        for df in [volatility_features, ta_features]:
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Merge on timestamp with nearest matching
        tolerance = pd.Timedelta(minutes=merge_tolerance_minutes)
        
        combined_df = pd.merge_asof(
            volatility_features.sort_values('timestamp'),
            ta_features.sort_values('timestamp'),
            on='timestamp',
            tolerance=tolerance,
            direction='nearest'
        )
        
        logging.debug(f"Combined features: {len(combined_df)} rows from "
                     f"{len(volatility_features)} volatility + {len(ta_features)} TA features")
        
        return combined_df
        
    except Exception as e:
        logging.error(f"Error combining features: {e}")
        return pd.DataFrame()

def extract_and_validate_features(
    symbol: str, 
    timeframe: str, 
    model_config: Dict,
    latest_only: bool = True
) -> Optional[pd.DataFrame]:
    """
    Extract and validate features for a specific model configuration.
    
    Args:
        symbol: Cryptocurrency symbol
        timeframe: Timeframe  
        model_config: Model configuration from MODEL_CONFIG
        latest_only: If True, return only the latest feature row
        
    Returns:
        Validated DataFrame with features or None if validation fails
    """
    try:
        model_name = model_config.get("path", "unknown").split("/")[-1]
        logging.debug(f"Extracting features for {symbol} ({timeframe}) using model {model_name}")
        
        # Determine what features are needed
        required_features = model_config.get("features", [])
        feature_columns = model_config.get("feature_columns", [])
        min_samples = model_config.get("min_samples", 10)
        
        # Extract features based on model requirements
        vol_features = pd.DataFrame()
        ta_features = pd.DataFrame()
        
        # Check if volatility features are needed
        vol_feature_types = ["vol_window", "vol_trend", "vol_std", "x_"]
        needs_volatility = any(any(vf in rf for vf in vol_feature_types) for rf in required_features)
        
        if needs_volatility:
            window_size = FEATURE_CONFIG["volatility_window_size"]
            vol_features = extract_volatility_features(symbol, timeframe, window_size)
            
            if vol_features.empty and needs_volatility:
                raise InsufficientDataError(f"No volatility features available for {symbol}")
        
        # Check if TA features are needed  
        ta_feature_types = ["rsi", "macd", "ema", "sma", "bbands", "atr", "stoch", "adx"]
        needs_ta = any(any(tf in rf for tf in ta_feature_types) for rf in required_features)
        
        if needs_ta:
            lookback = FEATURE_CONFIG["ta_lookback_periods"] 
            ta_features = extract_technical_analysis_features(symbol, timeframe, lookback)
            
            if ta_features.empty and needs_ta:
                logging.warning(f"No TA features available for {symbol} - continuing with volatility only")
        
        # Combine features
        combined_features = combine_features(vol_features, ta_features)
        
        if combined_features.empty:
            raise InsufficientDataError(f"No features could be extracted for {symbol}")
        
        # Validate minimum samples
        if len(combined_features) < min_samples:
            raise InsufficientDataError(
                f"Insufficient samples: need {min_samples}, got {len(combined_features)}"
            )
        
        # Select and validate required columns
        if feature_columns:
            # Use explicit column order from model config
            missing_cols = [col for col in feature_columns if col not in combined_features.columns]
            if missing_cols:
                raise FeatureValidationError(f"Missing required columns: {missing_cols}")
            
            # Select columns in exact order
            feature_matrix = combined_features[feature_columns].copy()
        else:
            # Use all available numeric columns except timestamp
            numeric_columns = combined_features.select_dtypes(include=[np.number]).columns
            feature_matrix = combined_features[numeric_columns].copy()
        
        # Validate schema if configured
        if VALIDATION_CONFIG["enable_schema_validation"] and model_config.get("feature_schema_hash"):
            expected_hash = model_config["feature_schema_hash"]
            current_hash = generate_schema_hash(feature_matrix.dtypes, feature_matrix.columns)
            
            if current_hash != expected_hash:
                raise SchemaValidationError(
                    f"Feature schema mismatch for {model_name}: {current_hash} != {expected_hash}"
                )
        
        # Handle NaN values
        if feature_matrix.isnull().any().any():
            logging.warning(f"Found NaN values in features for {symbol} - forward filling")
            feature_matrix = feature_matrix.fillna(method='ffill').fillna(0)
        
        # Return latest sample only if requested
        if latest_only:
            feature_matrix = feature_matrix.tail(1)
        
        # Add metadata
        if 'timestamp' in combined_features.columns:
            feature_matrix['timestamp'] = combined_features['timestamp'].tail(len(feature_matrix)).values
        
        feature_matrix['symbol'] = symbol
        feature_matrix['timeframe'] = timeframe
        
        logging.debug(f"Successfully extracted and validated {len(feature_matrix)} feature samples for {symbol}")
        return feature_matrix
        
    except (InsufficientDataError, FeatureValidationError, SchemaValidationError) as e:
        logging.error(f"Feature validation error for {symbol} ({timeframe}): {e}")
        return None
        
    except Exception as e:
        logging.error(f"Unexpected error extracting features for {symbol} ({timeframe}): {e}")
        return None

def get_latest_features_for_model(symbol: str, timeframe: str, model_name: str) -> Optional[pd.DataFrame]:
    """
    Get latest features for a specific model.
    
    Args:
        symbol: Cryptocurrency symbol
        timeframe: Timeframe
        model_name: Name of the model from MODEL_CONFIG
        
    Returns:
        DataFrame with latest features or None if extraction fails
    """
    try:
        from .config import get_model_config
        
        model_config = get_model_config(model_name)
        return extract_and_validate_features(symbol, timeframe, model_config, latest_only=True)
        
    except Exception as e:
        logging.error(f"Error getting latest features for model {model_name}: {e}")
        return None

# Utility functions for feature quality assessment
def assess_feature_quality(features_df: pd.DataFrame) -> Dict[str, float]:
    """
    Assess the quality of extracted features.
    
    Args:
        features_df: DataFrame with features
        
    Returns:
        Dictionary with quality metrics
    """
    if features_df.empty:
        return {"quality_score": 0.0, "completeness": 0.0, "freshness": 0.0}
    
    # Calculate completeness (non-null ratio)
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        completeness = 1 - (features_df[numeric_cols].isnull().sum().sum() / 
                           (len(features_df) * len(numeric_cols)))
    else:
        completeness = 0.0
    
    # Calculate freshness (how recent is the latest data)
    freshness = 1.0  # Default to fresh
    if 'timestamp' in features_df.columns:
        latest_time = pd.to_datetime(features_df['timestamp']).max()
        time_diff = datetime.now() - latest_time
        # Decay freshness over time (1.0 for < 1 hour, 0.0 for > 24 hours)
        freshness = max(0.0, 1.0 - (time_diff.total_seconds() / (24 * 3600)))
    
    # Overall quality score
    quality_score = (completeness * 0.7) + (freshness * 0.3)
    
    return {
        "quality_score": quality_score,
        "completeness": completeness,
        "freshness": freshness,
        "sample_count": len(features_df)
    }
