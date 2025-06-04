#!/usr/bin/env python3
"""
Predictor Module for TradingJii ML

This module loads trained models and performs real-time predictions:
- Loads models with validation and fallback mechanisms
- Extracts features from live data  
- Makes predictions with confidence scoring
- Handles errors gracefully with fallback strategies
- Supports multiple model types and ensemble predictions
"""

import os
import logging
import pickle
import pandas as pd
import numpy as np
import glob
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from colorama import Fore, Style

# ML Libraries
try:
    import joblib
    sklearn_available = True
except ImportError:
    logging.warning("Scikit-learn/joblib not available for model loading")
    sklearn_available = False

# Import ML modules
from .config import (
    MODEL_CONFIG,
    FALLBACK_CONFIG,
    VALIDATION_CONFIG,
    LOGGING_CONFIG,
    get_model_config,
    get_threshold_config,
    ModelNotFoundError,
    FeatureValidationError,
    SchemaValidationError,
    InsufficientDataError,
    generate_schema_hash
)

from .feature_extractor import (
    extract_and_validate_features,
    get_latest_features_for_model,
    assess_feature_quality
)

class ModelPredictor:
    """
    Main class for loading models and making predictions.
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize the predictor.
        
        Args:
            models_dir: Directory containing trained models
        """
        self.models_dir = models_dir
        self.loaded_models = {}  # Cache for loaded models
        self.model_metadata = {}  # Cache for model metadata
        self.discovered_models = {}  # Cache for discovered model files
        
        # Performance tracking
        self.prediction_stats = {
            "total_predictions": 0,
            "successful_predictions": 0,
            "fallback_used": 0,
            "errors": []
        }
        
        # Recursion prevention
        self._recursion_depth = 0
        self._max_recursion_depth = FALLBACK_CONFIG.get("recursion_depth_limit", 3)
        
        # Setup logging
        self.setup_logging()
        
        # Discover available models on initialization
        self.discover_available_models()

    def setup_logging(self):
        """Setup logging for predictions."""
        import logging as log
        import logging.handlers
        
        # Create a logger specific to predictions
        self.logger = log.getLogger('predictor')
        self.logger.setLevel(getattr(log, LOGGING_CONFIG["log_level"]))
        
        # Avoid duplicate handlers
        if not self.logger.handlers:
            formatter = log.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            # Always add console handler
            if LOGGING_CONFIG.get("enable_console_logging", True):
                console_handler = log.StreamHandler()
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)
            
            # Add file handler if enabled
            if LOGGING_CONFIG["enable_file_logging"]:
                log_file = LOGGING_CONFIG["predictor_log"]
                
                file_handler = log.handlers.RotatingFileHandler(
                    log_file,
                    maxBytes=LOGGING_CONFIG["max_log_size_mb"] * 1024 * 1024,
                    backupCount=5
                )
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)

    def discover_available_models(self):
        """Discover available model files in the models directory."""
        try:
            if not os.path.exists(self.models_dir):
                self.logger.warning(f"Models directory {self.models_dir} does not exist")
                return
            
            # Find all .pkl files in models directory
            model_files = glob.glob(os.path.join(self.models_dir, "*.pkl"))
            
            self.discovered_models = {}
            
            for file_path in model_files:
                filename = os.path.basename(file_path)
                
                # Parse volatility classifier models
                if filename.startswith("volatility_classifier_"):
                    parts = filename.replace("volatility_classifier_", "").replace(".pkl", "").split("_")
                    if len(parts) >= 3:
                        symbol = "_".join(parts[:-2])  # Handle symbols with underscores
                        timestamp = "_".join(parts[-2:])
                        
                        if "volatility_classifier" not in self.discovered_models:
                            self.discovered_models["volatility_classifier"] = {}
                        
                        if symbol not in self.discovered_models["volatility_classifier"]:
                            self.discovered_models["volatility_classifier"][symbol] = []
                        
                        self.discovered_models["volatility_classifier"][symbol].append({
                            "path": file_path,
                            "timestamp": timestamp,
                            "filename": filename
                        })
                
                # Add other model patterns here as needed
                
            # Sort models by timestamp (newest first)
            for model_type in self.discovered_models:
                for symbol in self.discovered_models[model_type]:
                    self.discovered_models[model_type][symbol].sort(
                        key=lambda x: x["timestamp"], reverse=True
                    )
            
            self.logger.info(f"Discovered models: {len(model_files)} files")
            for model_type, symbols in self.discovered_models.items():
                self.logger.info(f"  {model_type}: {list(symbols.keys())}")
                
        except Exception as e:
            self.logger.error(f"Error discovering models: {e}")

    def find_best_model_for_symbol(self, model_name: str, symbol: str) -> Optional[str]:
        """
        Find the best available model file for a given symbol.
        
        Args:
            model_name: Name of the model type
            symbol: Trading symbol
            
        Returns:
            Path to the best model file or None if not found
        """
        try:
            # Clean symbol for matching (remove :USDT suffix if present)
            clean_symbol = symbol.replace("/USDT:USDT", "_USDTUSDT").replace("/", "_")
            
            if model_name in self.discovered_models:
                symbol_models = self.discovered_models[model_name]
                
                # Try exact match first
                if clean_symbol in symbol_models:
                    best_model = symbol_models[clean_symbol][0]  # First is newest
                    self.logger.debug(f"Found exact model match for {symbol}: {best_model['filename']}")
                    return best_model["path"]
                
                # Try partial matches (e.g., BTC for BTC/USDT:USDT)
                symbol_base = clean_symbol.split("_")[0]
                for available_symbol in symbol_models:
                    if available_symbol.startswith(symbol_base):
                        best_model = symbol_models[available_symbol][0]
                        self.logger.debug(f"Found partial model match for {symbol}: {best_model['filename']}")
                        return best_model["path"]
                
                # Fallback to any available model of this type
                if symbol_models:
                    first_symbol = list(symbol_models.keys())[0]
                    best_model = symbol_models[first_symbol][0]
                    self.logger.warning(f"Using fallback model for {symbol}: {best_model['filename']}")
                    return best_model["path"]
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding model for {symbol}: {e}")
            return None

    def load_model_safe(self, model_path: str, expected_version: str = None) -> Optional[Dict]:
        """
        Safely load a model with validation.
        
        Args:
            model_path: Path to the model file
            expected_version: Expected model version for validation
            
        Returns:
            Model package dictionary or None if loading failed
        """
        try:
            if not sklearn_available:
                raise ModelNotFoundError("Scikit-learn/joblib not available for model loading")
            
            if not os.path.exists(model_path):
                raise ModelNotFoundError(f"Model file not found: {model_path}")
            
            # Load model package
            self.logger.debug(f"Loading model from {model_path}")
            model_package = joblib.load(model_path)
            
            # Validate model package structure
            required_keys = ["model", "metadata", "feature_names"]
            missing_keys = [key for key in required_keys if key not in model_package]
            if missing_keys:
                raise ValueError(f"Invalid model package. Missing keys: {missing_keys}")
            
            # Validate version if specified
            if expected_version:
                actual_version = model_package["metadata"].get("version")
                if actual_version != expected_version:
                    self.logger.warning(f"Version mismatch: expected {expected_version}, got {actual_version}")
            
            # Validate model is not None
            if model_package["model"] is None:
                raise ValueError("Model object is None")
            
            self.logger.info(f"Successfully loaded model: {model_package['metadata'].get('model_name', 'unknown')}")
            return model_package
            
        except Exception as e:
            self.logger.error(f"Error loading model from {model_path}: {e}")
            return None

    def get_model_for_config(self, model_name: str, symbol: str = None) -> Optional[Dict]:
        """
        Get or load model for a given configuration.
        
        Args:
            model_name: Name of the model from MODEL_CONFIG
            symbol: Trading symbol (for symbol-specific models)
            
        Returns:
            Model package or None if not available
        """
        try:
            # Create cache key with symbol if applicable
            cache_key = f"{model_name}_{symbol}" if symbol else model_name
            
            # Check if model is already loaded
            if cache_key in self.loaded_models:
                return self.loaded_models[cache_key]
            
            # Get model configuration
            model_config = get_model_config(model_name)
            
            # Handle ensemble models
            if model_config["type"] == "ensemble":
                return self._load_ensemble_models(model_name, model_config)
            
            # Check if model is disabled
            if not model_config.get("enabled", True):
                self.logger.warning(f"Model {model_name} is disabled in configuration")
                return None
            
            # Try to find symbol-specific model first
            model_path = None
            if symbol and model_config.get("symbol_specific", False):
                model_path = self.find_best_model_for_symbol(model_name, symbol)
            
            # Fallback to configured path if no symbol-specific model found
            if not model_path:
                configured_path = model_config["path"]
                if "*" in configured_path:
                    # Pattern-based path, try to find any model of this type
                    if model_name in self.discovered_models and self.discovered_models[model_name]:
                        # Use first available model
                        first_symbol = list(self.discovered_models[model_name].keys())[0]
                        first_model = self.discovered_models[model_name][first_symbol][0]
                        model_path = first_model["path"]
                        self.logger.info(f"Using available model for {model_name}: {first_model['filename']}")
                else:
                    # Fixed path
                    model_path = configured_path
                    if not os.path.isabs(model_path):
                        model_path = os.path.join(self.models_dir, model_path)
            
            if not model_path or not os.path.exists(model_path):
                self.logger.error(f"No valid model file found for {model_name}")
                return None
            
            # Load the model
            expected_version = model_config.get("version")
            model_package = self.load_model_safe(model_path, expected_version)
            
            if model_package:
                # Cache the loaded model
                self.loaded_models[cache_key] = model_package
                self.model_metadata[cache_key] = model_package["metadata"]
                return model_package
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting model for {model_name}: {e}")
            return None

    def _load_ensemble_models(self, ensemble_name: str, ensemble_config: Dict) -> Optional[Dict]:
        """
        Load models for ensemble prediction.
        
        Args:
            ensemble_name: Name of the ensemble
            ensemble_config: Ensemble configuration
            
        Returns:
            Ensemble package or None if loading failed
        """
        try:
            sub_models = {}
            weights = ensemble_config.get("weights", [])
            model_names = ensemble_config.get("models", [])
            
            if len(weights) != len(model_names):
                self.logger.warning(f"Ensemble {ensemble_name}: weights and models count mismatch")
                # Default to equal weights
                weights = [1.0 / len(model_names)] * len(model_names)
            
            # Load each sub-model
            for i, sub_model_name in enumerate(model_names):
                sub_model = self.get_model_for_config(sub_model_name)
                if sub_model:
                    sub_models[sub_model_name] = {
                        "model_package": sub_model,
                        "weight": weights[i]
                    }
                else:
                    self.logger.warning(f"Failed to load sub-model {sub_model_name} for ensemble {ensemble_name}")
            
            # Check minimum models requirement
            min_models = ensemble_config.get("min_models_required", 1)
            if len(sub_models) < min_models:
                self.logger.error(f"Ensemble {ensemble_name}: only {len(sub_models)} models loaded, need {min_models}")
                return None
            
            # Create ensemble package
            ensemble_package = {
                "type": "ensemble",
                "sub_models": sub_models,
                "metadata": {
                    "ensemble_name": ensemble_name,
                    "version": ensemble_config.get("version", "1.0.0"),
                    "models_loaded": list(sub_models.keys()),
                    "weights": weights[:len(sub_models)]
                }
            }
            
            # Cache ensemble
            self.loaded_models[ensemble_name] = ensemble_package
            
            self.logger.info(f"Loaded ensemble {ensemble_name} with {len(sub_models)} models")
            return ensemble_package
            
        except Exception as e:
            self.logger.error(f"Error loading ensemble {ensemble_name}: {e}")
            return None

    def predict_single_model(
        self,
        model_package: Dict,
        features: pd.DataFrame
    ) -> Optional[Dict]:
        """
        Make prediction using a single model.
        
        Args:
            model_package: Loaded model package
            features: Feature DataFrame
            
        Returns:
            Prediction results or None if failed
        """
        try:
            model = model_package["model"]
            scaler = model_package.get("scaler")
            feature_names = model_package["feature_names"]
            metadata = model_package["metadata"]
            
            # Prepare features for prediction
            if not features.empty:
                # Select and order features according to model training
                feature_data = features[feature_names].values
                
                # Apply scaling if model was trained with scaling
                if scaler is not None:
                    feature_data = scaler.transform(feature_data)
                
                # Make prediction
                prediction = model.predict(feature_data)[0]  # Get first (and only) prediction
                
                # Get prediction probabilities if available
                probabilities = None
                confidence = 0.5  # Default confidence
                
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(feature_data)[0]
                    probabilities = proba.tolist()
                    confidence = max(proba)  # Highest probability as confidence
                
                return {
                    "prediction": int(prediction),
                    "probabilities": probabilities,
                    "confidence": float(confidence),
                    "model_info": {
                        "name": metadata.get("model_name", "unknown"),
                        "version": metadata.get("version", "unknown"),
                        "accuracy": model_package.get("metrics", {}).get("accuracy", 0.0)
                    }
                }
            else:
                raise ValueError("Empty features provided")
                
        except Exception as e:
            self.logger.error(f"Error in single model prediction: {e}")
            return None

    def predict_ensemble(
        self,
        ensemble_package: Dict,
        symbol: str,
        timeframe: str
    ) -> Optional[Dict]:
        """
        Make prediction using ensemble of models.
        
        Args:
            ensemble_package: Loaded ensemble package
            symbol: Cryptocurrency symbol
            timeframe: Timeframe
            
        Returns:
            Ensemble prediction results or None if failed
        """
        try:
            sub_models = ensemble_package["sub_models"]
            predictions = []
            total_weight = 0.0
            
            for model_name, model_info in sub_models.items():
                try:
                    # Get features for this specific model
                    model_package = model_info["model_package"]
                    weight = model_info["weight"]
                    
                    # Extract features using the model's configuration
                    model_config = get_model_config(model_name)
                    features = extract_and_validate_features(symbol, timeframe, model_config, latest_only=True)
                    
                    if features is not None and not features.empty:
                        # Make prediction with this model
                        pred_result = self.predict_single_model(model_package, features)
                        
                        if pred_result:
                            pred_result["weight"] = weight
                            pred_result["model_name"] = model_name
                            predictions.append(pred_result)
                            total_weight += weight
                        else:
                            self.logger.warning(f"Prediction failed for sub-model {model_name}")
                    else:
                        self.logger.warning(f"No features available for sub-model {model_name}")
                        
                except Exception as e:
                    self.logger.error(f"Error in ensemble sub-model {model_name}: {e}")
                    continue
            
            if not predictions:
                return None
            
            # Combine predictions using weighted average
            weighted_prediction = 0.0
            weighted_confidence = 0.0
            combined_probabilities = None
            
            for pred in predictions:
                weight = pred["weight"] / total_weight  # Normalize weights
                weighted_prediction += pred["prediction"] * weight
                weighted_confidence += pred["confidence"] * weight
                
                # Combine probabilities if available
                if pred["probabilities"] and combined_probabilities is None:
                    combined_probabilities = [0.0] * len(pred["probabilities"])
                
                if pred["probabilities"] and combined_probabilities:
                    for i, prob in enumerate(pred["probabilities"]):
                        combined_probabilities[i] += prob * weight
            
            # Final prediction (round to nearest integer for classification)
            final_prediction = int(round(weighted_prediction))
            
            return {
                "prediction": final_prediction,
                "probabilities": combined_probabilities,
                "confidence": float(weighted_confidence),
                "ensemble_info": {
                    "name": ensemble_package["metadata"]["ensemble_name"],
                    "models_used": len(predictions),
                    "total_models": len(sub_models),
                    "individual_predictions": [
                        {
                            "model": pred["model_name"],
                            "prediction": pred["prediction"],
                            "confidence": pred["confidence"],
                            "weight": pred["weight"]
                        }
                        for pred in predictions
                    ]
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in ensemble prediction: {e}")
            return None

    def predict_with_fallbacks(
        self,
        symbol: str,
        timeframe: str,
        model_name: str = None
    ) -> Dict:
        """
        Make prediction with fallback mechanisms.
        
        Args:
            symbol: Cryptocurrency symbol
            timeframe: Timeframe
            model_name: Specific model to use (None for default)
            
        Returns:
            Prediction result dictionary
        """
        # Initialize result with safe defaults
        result = {
            "symbol": symbol,
            "timeframe": timeframe,
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "prediction": 0,  # HOLD as safe default
            "signal": "HOLD",
            "confidence": 0.0,
            "model_used": None,
            "fallback_used": False,
            "error": None,
            "feature_quality": {},
            "prediction_meta": {}
        }
        
        # Update prediction stats
        self.prediction_stats["total_predictions"] += 1
        
        try:
            # Determine which model to use
            target_model = model_name or FALLBACK_CONFIG["default_model"]
            
            self.logger.debug(f"Making prediction for {symbol} ({timeframe}) using {target_model}")
            
            # Load model
            model_package = self.get_model_for_config(target_model, symbol)
            if not model_package:
                raise ModelNotFoundError(f"Could not load model: {target_model}")
            
            # Make prediction based on model type
            prediction_result = None
            
            if model_package.get("type") == "ensemble":
                prediction_result = self.predict_ensemble(model_package, symbol, timeframe)
            else:
                # Single model prediction
                model_config = get_model_config(target_model)
                features = extract_and_validate_features(symbol, timeframe, model_config, latest_only=True)
                
                if features is not None and not features.empty:
                    # Assess feature quality
                    quality_metrics = assess_feature_quality(features)
                    result["feature_quality"] = quality_metrics
                    
                    # Check minimum quality threshold
                    min_quality = VALIDATION_CONFIG.get("min_data_quality_score", 0.8)
                    if quality_metrics["quality_score"] < min_quality:
                        self.logger.warning(f"Low feature quality ({quality_metrics['quality_score']:.2f}) for {symbol}")
                    
                    prediction_result = self.predict_single_model(model_package, features)
                else:
                    raise InsufficientDataError(f"No valid features extracted for {symbol}")
            
            if prediction_result:
                # Convert prediction to signal
                signal = self._prediction_to_signal(
                    prediction_result["prediction"],
                    prediction_result["confidence"],
                    target_model
                )
                
                # Update result with successful prediction
                result.update({
                    "success": True,
                    "prediction": prediction_result["prediction"],
                    "signal": signal,
                    "confidence": prediction_result["confidence"],
                    "model_used": target_model,
                    "prediction_meta": prediction_result
                })
                
                self.prediction_stats["successful_predictions"] += 1
                self.logger.debug(f"Prediction successful: {signal} (confidence: {prediction_result['confidence']:.3f})")
                
            else:
                raise ValueError("Prediction returned None")
                
        except ModelNotFoundError as e:
            self.logger.error(f"Model error for {symbol}: {e}")
            result = self._try_fallback_model(symbol, timeframe, result, str(e))
            
        except (FeatureValidationError, SchemaValidationError, InsufficientDataError) as e:
            self.logger.error(f"Feature error for {symbol}: {e}")
            result["error"] = str(e)
            
        except Exception as e:
            self.logger.error(f"Unexpected prediction error for {symbol}: {e}")
            result["error"] = str(e)
        
        # Log error if prediction failed
        if not result["success"]:
            self.prediction_stats["errors"].append({
                "symbol": symbol,
                "timeframe": timeframe,
                "error": result["error"],
                "timestamp": result["timestamp"]
            })
        
        return result

    def _try_fallback_model(self, symbol: str, timeframe: str, original_result: Dict, error_msg: str) -> Dict:
        """
        Try fallback model if primary model fails.
        
        Args:
            symbol: Cryptocurrency symbol
            timeframe: Timeframe
            original_result: Original result dict to update
            error_msg: Error message from primary model
            
        Returns:
            Updated result with fallback attempt
        """
        if not FALLBACK_CONFIG["enable_graceful_degradation"]:
            # Return safe neutral fallback
            return self._create_safe_fallback_result(symbol, timeframe, error_msg)
        
        fallback_model = FALLBACK_CONFIG.get("fallback_model")
        if fallback_model and fallback_model != original_result.get("model_attempted"):
            try:
                self.logger.info(f"Trying fallback model {fallback_model} for {symbol}")
                
                fallback_result = self.predict_with_fallbacks(symbol, timeframe, fallback_model)
                if fallback_result["success"]:
                    fallback_result["fallback_used"] = True
                    fallback_result["primary_error"] = error_msg
                    self.prediction_stats["fallback_used"] += 1
                    return fallback_result
                    
            except Exception as e:
                self.logger.error(f"Fallback model also failed for {symbol}: {e}")
        
        # If fallback fails or not available, return safe neutral default
        return self._create_safe_fallback_result(symbol, timeframe, error_msg)

    def _create_safe_fallback_result(self, symbol: str, timeframe: str, error_msg: str) -> Dict:
        """
        Create a safe neutral fallback result when all models fail.
        
        Args:
            symbol: Cryptocurrency symbol
            timeframe: Timeframe  
            error_msg: Error message from failed predictions
            
        Returns:
            Safe fallback result with HOLD signal and 0.0 confidence
        """
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "timestamp": datetime.now().isoformat(),
            "success": True,  # Mark as success since we're providing a safe signal
            "prediction": 0,  # 0 = HOLD in classification
            "signal": "HOLD",
            "confidence": 0.0,  # Zero confidence indicates fallback
            "model_used": "safe_fallback",
            "fallback_used": True,
            "error": error_msg,
            "feature_quality": {},
            "prediction_meta": {
                "prediction": 0,
                "confidence": 0.0,
                "model_info": {
                    "name": "safe_fallback",
                    "version": "1.0.0",
                    "accuracy": 0.0
                },
                "fallback_reason": "All models failed - returning safe neutral signal"
            }
        }

    def _prediction_to_signal(self, prediction: int, confidence: float, model_name: str) -> str:
        """
        Convert model prediction to trading signal.
        
        Args:
            prediction: Raw model prediction (-1, 0, 1)
            confidence: Prediction confidence (0.0 to 1.0)
            model_name: Name of the model used
            
        Returns:
            Trading signal string
        """
        try:
            # Get threshold configuration
            threshold_config = get_threshold_config()
            
            # Apply confidence thresholds
            if prediction == 1 and confidence >= threshold_config["buy"]:
                return "BUY"
            elif prediction == -1 and confidence >= threshold_config["sell"]:
                return "SELL"
            else:
                return "HOLD"
                
        except Exception as e:
            self.logger.error(f"Error converting prediction to signal: {e}")
            return "HOLD"

    def get_prediction_stats(self) -> Dict:
        """
        Get prediction performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        total = self.prediction_stats["total_predictions"]
        successful = self.prediction_stats["successful_predictions"]
        fallback = self.prediction_stats["fallback_used"]
        
        success_rate = (successful / total) if total > 0 else 0.0
        fallback_rate = (fallback / total) if total > 0 else 0.0
        
        return {
            "total_predictions": total,
            "successful_predictions": successful,
            "success_rate": success_rate,
            "fallback_used": fallback,
            "fallback_rate": fallback_rate,
            "error_count": len(self.prediction_stats["errors"]),
            "recent_errors": self.prediction_stats["errors"][-5:],  # Last 5 errors
            "models_loaded": list(self.loaded_models.keys()),
            "health_status": "healthy" if success_rate > 0.8 and fallback_rate < 0.2 else "degraded"
        }

    def clear_cache(self):
        """Clear loaded model cache."""
        self.loaded_models.clear()
        self.model_metadata.clear()
        self.logger.info("Model cache cleared")


# Utility functions for CLI usage
def predict_symbol(
    symbol: str,
    timeframe: str,
    model_name: str = None,
    models_dir: str = "models"
) -> Dict:
    """
    Make a prediction for a single symbol (CLI wrapper).
    
    Args:
        symbol: Cryptocurrency symbol
        timeframe: Timeframe
        model_name: Model to use (None for default)
        models_dir: Directory containing models
        
    Returns:
        Prediction result dictionary
    """
    try:
        predictor = ModelPredictor(models_dir)
        return predictor.predict_with_fallbacks(symbol, timeframe, model_name)
    except Exception as e:
        logging.error(f"CLI prediction failed for {symbol}: {e}")
        return {
            "symbol": symbol,
            "success": False,
            "error": str(e),
            "signal": "HOLD"
        }


def batch_predict(
    symbols: List[str],
    timeframes: List[str],
    model_name: str = None,
    models_dir: str = "models"
) -> List[Dict]:
    """
    Make predictions for multiple symbols and timeframes.
    
    Args:
        symbols: List of symbols
        timeframes: List of timeframes
        model_name: Model to use (None for default)
        models_dir: Directory containing models
        
    Returns:
        List of prediction results
    """
    predictor = ModelPredictor(models_dir)
    results = []
    
    for symbol in symbols:
        for timeframe in timeframes:
            try:
                result = predictor.predict_with_fallbacks(symbol, timeframe, model_name)
                results.append(result)
            except Exception as e:
                logging.error(f"Batch prediction failed for {symbol} ({timeframe}): {e}")
                results.append({
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "success": False,
                    "error": str(e),
                    "signal": "HOLD"
                })
    
    return results


if __name__ == "__main__":
    # Example usage
    import sys
    from modules.utils.logging_setup import setup_logging
    
    # Setup logging
    setup_logging(level=logging.INFO)
    
    if len(sys.argv) >= 3:
        symbol = sys.argv[1]
        timeframe = sys.argv[2]
        model_name = sys.argv[3] if len(sys.argv) > 3 else None
        
        result = predict_symbol(symbol, timeframe, model_name)
        
        print(f"Prediction for {symbol} ({timeframe}):")
        print(f"  Signal: {result.get('signal', 'UNKNOWN')}")
        print(f"  Confidence: {result.get('confidence', 0.0):.3f}")
        print(f"  Success: {result.get('success', False)}")
        
        if result.get("error"):
            print(f"  Error: {result['error']}")
    else:
        print("Usage: python predictor.py SYMBOL TIMEFRAME [MODEL_NAME]")
        print("Example: python predictor.py BTC/USDT:USDT 4h volatility_classifier")
