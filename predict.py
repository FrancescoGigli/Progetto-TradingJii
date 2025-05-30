#!/usr/bin/env python3
"""
Binary Prediction Engine with Dynamic Confidence Management
=========================================================

Core prediction module that:
1. Loads binary model and preprocessing pipeline
2. Calculates predict_proba on new data
3. Generates BUY/SELL/HOLD signals based on confidence threshold
4. Manages signal state transitions and confidence updates
5. Provides detailed logging and state tracking
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import warnings
import joblib
import json

# Custom modules
from modules.utils.logging_setup import setup_logging

warnings.filterwarnings('ignore')

# ====================================================================
# PREDICTION CONFIGURATION
# ====================================================================

DEFAULT_CONFIG = {
    'confidence_threshold': 0.7,           # Minimum confidence for BUY/SELL
    'confidence_improvement_threshold': 0.05,  # Min improvement to update signal
    'model_path': 'ml_system/models/binary_models/best_binary_model.pkl',
    'state_file': 'ml_system/logs/predictions/signal_state.json',
    'log_file': 'signal_log.json',
    'max_log_entries': 1000,              # Rotate logs after this many entries
    'enable_detailed_logging': True
}

class PredictionEngine:
    """
    Binary prediction engine with confidence management.
    
    Handles model loading, prediction, confidence thresholding,
    and signal state management.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize prediction engine.
        
        Args:
            config: Configuration dictionary
        """
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.logger = setup_logging(logging.INFO)
        
        # Model components
        self.model_bundle = None
        self.model = None
        self.preprocessor = None
        self.feature_engineer = None
        self.conversion_info = None
        
        # Signal state
        self.current_signal = None
        self.current_confidence = 0.0
        self.last_prediction_time = None
        self.signal_history = []
        
        # Initialize
        self._create_directories()
        self._load_model()
        self._load_signal_state()
    
    def _create_directories(self) -> None:
        """Create necessary directories."""
        directories = [
            Path(self.config['state_file']).parent,
            Path(self.config['log_file']).parent
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _load_model(self) -> None:
        """Load the binary model and preprocessing pipeline."""
        model_path = Path(self.config['model_path'])
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        try:
            self.logger.info(f"Loading model bundle from {model_path}")
            self.model_bundle = joblib.load(model_path)
            
            # Extract components
            self.model = self.model_bundle['model']
            self.preprocessor = self.model_bundle['preprocessor']
            self.feature_engineer = self.model_bundle['feature_engineer']
            self.conversion_info = self.model_bundle['conversion_info']
            
            model_info = {
                'model_name': self.model_bundle['model_name'],
                'symbol': self.model_bundle.get('symbol', 'Unknown'),
                'timeframe': self.model_bundle.get('timeframe', 'Unknown'),
                'training_timestamp': self.model_bundle.get('training_timestamp', 'Unknown'),
                'mapping_strategy': self.conversion_info['mapping_strategy']
            }
            
            self.logger.info(f"Model loaded successfully: {model_info}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_signal_state(self) -> None:
        """Load previous signal state if exists."""
        state_file = Path(self.config['state_file'])
        
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                self.current_signal = state.get('current_signal')
                self.current_confidence = state.get('current_confidence', 0.0)
                self.last_prediction_time = state.get('last_prediction_time')
                
                self.logger.info(f"Loaded signal state: {self.current_signal} (confidence: {self.current_confidence:.3f})")
                
            except Exception as e:
                self.logger.warning(f"Failed to load signal state: {e}")
        else:
            self.logger.info("No previous signal state found, starting fresh")
    
    def _save_signal_state(self) -> None:
        """Save current signal state."""
        state = {
            'current_signal': self.current_signal,
            'current_confidence': self.current_confidence,
            'last_prediction_time': self.last_prediction_time,
            'last_update': datetime.now(timezone.utc).isoformat()
        }
        
        try:
            with open(self.config['state_file'], 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save signal state: {e}")
    
    def _log_signal_event(self, event_data: Dict) -> None:
        """Log signal event to signal_log.json."""
        log_file = Path(self.config['log_file'])
        
        # Load existing log
        signal_log = []
        if log_file.exists():
            try:
                with open(log_file, 'r') as f:
                    signal_log = json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load signal log: {e}")
        
        # Add new event
        signal_log.append(event_data)
        
        # Rotate logs if too many entries
        if len(signal_log) > self.config['max_log_entries']:
            signal_log = signal_log[-self.config['max_log_entries']:]
        
        # Save updated log
        try:
            with open(log_file, 'w') as f:
                json.dump(signal_log, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save signal log: {e}")
    
    def _preprocess_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing and feature engineering pipeline."""
        try:
            # Apply preprocessing
            X_processed = self.preprocessor.transform(X)
            
            # Apply feature engineering
            X_final = self.feature_engineer.transform(X_processed)
            
            return X_final
            
        except Exception as e:
            self.logger.error(f"Feature preprocessing failed: {e}")
            raise
    
    def _get_class_label(self, binary_prediction: int) -> str:
        """Convert binary prediction to class label."""
        mapping_strategy = self.conversion_info['mapping_strategy']
        
        if mapping_strategy == 'BUY_AS_1':
            return 'BUY' if binary_prediction == 1 else 'SELL'
        else:  # SELL_AS_1
            return 'SELL' if binary_prediction == 1 else 'BUY'
    
    def predict_single(self, X: pd.DataFrame, 
                      timestamp: Optional[str] = None,
                      symbol: Optional[str] = None,
                      timeframe: Optional[str] = None) -> Dict[str, Any]:
        """
        Make prediction on single sample with confidence management.
        
        Args:
            X: Feature DataFrame (single row)
            timestamp: Prediction timestamp
            symbol: Trading symbol
            timeframe: Timeframe
            
        Returns:
            Dictionary with prediction results
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()
        
        try:
            # Preprocess features
            X_processed = self._preprocess_features(X)
            
            # Get model predictions
            binary_prediction = self.model.predict(X_processed)[0]
            prediction_proba = self.model.predict_proba(X_processed)[0]
            
            # Calculate confidence (max probability)
            confidence = float(np.max(prediction_proba))
            predicted_class = self._get_class_label(binary_prediction)
            
            # Decision logic based on confidence threshold
            decision_reason = ""
            final_signal = ""
            
            if confidence < self.config['confidence_threshold']:
                final_signal = "HOLD"
                decision_reason = f"confidence_too_low ({confidence:.3f} < {self.config['confidence_threshold']})"
            else:
                # Check if we should update existing signal
                if self.current_signal == predicted_class:
                    # Same direction: check if confidence improved significantly
                    confidence_improvement = confidence - self.current_confidence
                    
                    if confidence_improvement >= self.config['confidence_improvement_threshold']:
                        final_signal = predicted_class
                        decision_reason = f"confidence_improved (+{confidence_improvement:.3f})"
                    else:
                        final_signal = self.current_signal  # Keep existing
                        decision_reason = f"same_direction_insufficient_improvement ({confidence_improvement:.3f})"
                else:
                    # New direction
                    final_signal = predicted_class
                    decision_reason = "new_direction"
            
            # Update signal state if needed
            signal_changed = final_signal != self.current_signal
            if signal_changed or (final_signal == self.current_signal and confidence > self.current_confidence):
                self.current_signal = final_signal
                self.current_confidence = confidence
                self.last_prediction_time = timestamp
                self._save_signal_state()
            
            # Prepare result
            result = {
                'timestamp': timestamp,
                'signal': final_signal,
                'confidence': confidence,
                'predicted_class': predicted_class,
                'binary_prediction': int(binary_prediction),
                'model_proba': prediction_proba.tolist(),
                'decision_reason': decision_reason,
                'signal_changed': signal_changed,
                'previous_signal': self.current_signal if not signal_changed else None,
                'symbol': symbol,
                'timeframe': timeframe,
                'model_info': {
                    'model_name': self.model_bundle['model_name'],
                    'mapping_strategy': self.conversion_info['mapping_strategy']
                }
            }
            
            # Log event
            if self.config['enable_detailed_logging']:
                log_event = {
                    'timestamp': timestamp,
                    'signal': final_signal,
                    'confidence': confidence,
                    'reason': decision_reason,
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'model_proba': prediction_proba.tolist(),
                    'signal_changed': signal_changed
                }
                self._log_signal_event(log_event)
            
            # Console logging
            if signal_changed:
                self.logger.info(f"SIGNAL CHANGE: {final_signal} (confidence: {confidence:.3f}, reason: {decision_reason})")
            elif self.config['enable_detailed_logging']:
                self.logger.debug(f"Signal: {final_signal} (confidence: {confidence:.3f}, reason: {decision_reason})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return {
                'timestamp': timestamp,
                'signal': 'HOLD',
                'confidence': 0.0,
                'error': str(e),
                'decision_reason': 'prediction_error',
                'signal_changed': False
            }
    
    def predict_batch(self, X: pd.DataFrame,
                     timestamps: Optional[List[str]] = None,
                     symbol: Optional[str] = None,
                     timeframe: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Make predictions on batch of samples.
        
        Args:
            X: Feature DataFrame (multiple rows)
            timestamps: List of timestamps
            symbol: Trading symbol
            timeframe: Timeframe
            
        Returns:
            List of prediction results
        """
        if timestamps is None:
            timestamps = [datetime.now(timezone.utc).isoformat()] * len(X)
        
        results = []
        for i, (_, row) in enumerate(X.iterrows()):
            row_df = pd.DataFrame([row])
            result = self.predict_single(
                row_df, timestamps[i], symbol, timeframe
            )
            results.append(result)
        
        return results
    
    def get_current_signal(self) -> Dict[str, Any]:
        """Get current signal state."""
        return {
            'signal': self.current_signal,
            'confidence': self.current_confidence,
            'last_prediction_time': self.last_prediction_time,
            'model_info': {
                'model_name': self.model_bundle['model_name'] if self.model_bundle else None,
                'mapping_strategy': self.conversion_info['mapping_strategy'] if self.conversion_info else None
            }
        }
    
    def get_signal_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get signal history from log file.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of signal events
        """
        log_file = Path(self.config['log_file'])
        
        if not log_file.exists():
            return []
        
        try:
            with open(log_file, 'r') as f:
                signal_log = json.load(f)
            
            if limit:
                signal_log = signal_log[-limit:]
            
            return signal_log
            
        except Exception as e:
            self.logger.error(f"Failed to load signal history: {e}")
            return []
    
    def reset_signal_state(self) -> None:
        """Reset signal state to initial values."""
        self.current_signal = None
        self.current_confidence = 0.0
        self.last_prediction_time = None
        self._save_signal_state()
        self.logger.info("Signal state reset")

# ====================================================================
# CONVENIENCE FUNCTIONS
# ====================================================================

def create_prediction_engine(config: Optional[Dict] = None) -> PredictionEngine:
    """Create and return a configured prediction engine."""
    return PredictionEngine(config)

def predict_signal(X: pd.DataFrame, 
                  config: Optional[Dict] = None,
                  timestamp: Optional[str] = None,
                  symbol: Optional[str] = None,
                  timeframe: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function for single prediction.
    
    Args:
        X: Feature DataFrame (single row)
        config: Engine configuration
        timestamp: Prediction timestamp
        symbol: Trading symbol
        timeframe: Timeframe
        
    Returns:
        Prediction result
    """
    engine = PredictionEngine(config)
    return engine.predict_single(X, timestamp, symbol, timeframe)

# ====================================================================
# COMMAND LINE INTERFACE
# ====================================================================

def main():
    """Command line interface for testing predictions."""
    import argparse
    
    parser = argparse.ArgumentParser(description="TradingJii Binary Prediction Engine")
    parser.add_argument('--model-path', type=str,
                       default='ml_system/models/binary_models/best_binary_model.pkl',
                       help='Path to model bundle')
    parser.add_argument('--confidence-threshold', type=float, default=0.7,
                       help='Confidence threshold for BUY/SELL signals')
    parser.add_argument('--test-features', type=str,
                       help='CSV file with test features')
    parser.add_argument('--reset-state', action='store_true',
                       help='Reset signal state')
    parser.add_argument('--show-history', type=int,
                       help='Show last N signal events')
    parser.add_argument('--show-current', action='store_true',
                       help='Show current signal state')
    
    args = parser.parse_args()
    
    # Configure engine
    config = {
        'model_path': args.model_path,
        'confidence_threshold': args.confidence_threshold,
        'enable_detailed_logging': True
    }
    
    try:
        engine = PredictionEngine(config)
        
        if args.reset_state:
            engine.reset_signal_state()
            print("Signal state reset")
            return
        
        if args.show_current:
            state = engine.get_current_signal()
            print(f"Current Signal: {state['signal']}")
            print(f"Confidence: {state['confidence']:.3f}")
            print(f"Last Update: {state['last_prediction_time']}")
            return
        
        if args.show_history:
            history = engine.get_signal_history(args.show_history)
            print(f"Last {len(history)} signal events:")
            for event in history:
                print(f"{event['timestamp']}: {event['signal']} (conf: {event['confidence']:.3f}, reason: {event['reason']})")
            return
        
        if args.test_features:
            test_file = Path(args.test_features)
            if not test_file.exists():
                print(f"Test file not found: {test_file}")
                return
            
            df = pd.read_csv(test_file)
            print(f"Testing with {len(df)} samples from {test_file}")
            
            results = engine.predict_batch(df)
            
            for i, result in enumerate(results):
                print(f"Sample {i+1}: {result['signal']} (conf: {result['confidence']:.3f}, reason: {result['decision_reason']})")
        
        else:
            print("Prediction engine loaded successfully!")
            print("Use --test-features to test with data")
            print("Use --show-current to see current signal")
            print("Use --show-history N to see last N events")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
