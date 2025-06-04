#!/usr/bin/env python3
"""
Smart Model Selector for TradingJii ML

This module implements intelligent model selection with strict asset compatibility checks:
- Prevents cross-asset model contamination (e.g., ETH models for SOL)
- Enforces symbol-specific model matching
- Safe fallback strategies without compromising prediction quality
- Asset class compatibility matrix for proper model selection
"""

import os
import logging
import glob
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class AssetClass(Enum):
    """Asset classification for compatibility checks"""
    BITCOIN = "bitcoin"
    ETHEREUM = "ethereum"
    LAYER1 = "layer1"
    LAYER2 = "layer2"
    DEFI = "defi"
    STABLE = "stable"
    UNKNOWN = "unknown"

class MarketCapTier(Enum):
    """Market cap tiers for model compatibility"""
    LARGE_CAP = "large_cap"  # >$100B
    MID_CAP = "mid_cap"      # $10B-$100B  
    SMALL_CAP = "small_cap"  # <$10B

@dataclass
class SymbolCompatibility:
    """Symbol compatibility configuration"""
    symbol: str
    asset_class: AssetClass
    market_cap_tier: MarketCapTier
    compatible_symbols: List[str]
    fallback_allowed: bool
    cross_asset_forbidden: bool

# Asset Compatibility Matrix - CRITICAL: Prevents cross-contamination
SYMBOL_COMPATIBILITY = {
    "BTC/USDT:USDT": SymbolCompatibility(
        symbol="BTC/USDT:USDT",
        asset_class=AssetClass.BITCOIN,
        market_cap_tier=MarketCapTier.LARGE_CAP,
        compatible_symbols=["BTC/USDT:USDT", "BTC-USDT"],  # Only BTC variants
        fallback_allowed=False,  # NEVER use other asset models for BTC
        cross_asset_forbidden=True
    ),
    "ETH/USDT:USDT": SymbolCompatibility(
        symbol="ETH/USDT:USDT", 
        asset_class=AssetClass.ETHEREUM,
        market_cap_tier=MarketCapTier.LARGE_CAP,
        compatible_symbols=["ETH/USDT:USDT", "ETH-USDT"],  # Only ETH variants
        fallback_allowed=False,  # NEVER use BTC/SOL models for ETH
        cross_asset_forbidden=True
    ),
    "SOL/USDT:USDT": SymbolCompatibility(
        symbol="SOL/USDT:USDT",
        asset_class=AssetClass.LAYER1,
        market_cap_tier=MarketCapTier.MID_CAP,
        compatible_symbols=["SOL/USDT:USDT", "SOL-USDT"],  # Only SOL variants
        fallback_allowed=False,  # NEVER use ETH/BTC models for SOL
        cross_asset_forbidden=True
    ),
    "ADA/USDT:USDT": SymbolCompatibility(
        symbol="ADA/USDT:USDT",
        asset_class=AssetClass.LAYER1,
        market_cap_tier=MarketCapTier.MID_CAP,
        compatible_symbols=["ADA/USDT:USDT", "ADA-USDT"],
        fallback_allowed=False,
        cross_asset_forbidden=True
    ),
    "MATIC/USDT:USDT": SymbolCompatibility(
        symbol="MATIC/USDT:USDT",
        asset_class=AssetClass.LAYER2,
        market_cap_tier=MarketCapTier.MID_CAP,
        compatible_symbols=["MATIC/USDT:USDT", "MATIC-USDT"],
        fallback_allowed=False,
        cross_asset_forbidden=True
    )
}

# Model-Symbol Restrictions
MODEL_SYMBOL_RESTRICTIONS = {
    "volatility_classifier": {
        "symbol_specific": True,
        "cross_asset_forbidden": True,  # CRITICAL: no cross-asset usage
        "require_exact_match": True,
        "min_confidence_for_fallback": 0.9,  # Very high bar for any fallback
        "allowed_fallback_types": []  # No fallbacks allowed
    },
    "trend_predictor": {
        "symbol_specific": True,
        "cross_asset_forbidden": True,
        "require_exact_match": True,
        "min_confidence_for_fallback": 0.85,
        "allowed_fallback_types": []
    },
    "ensemble_v1": {
        "symbol_specific": False,  # Ensemble can be more flexible
        "cross_asset_forbidden": False,  # But still not recommended
        "require_exact_match": False,
        "min_confidence_for_fallback": 0.7,
        "allowed_fallback_types": ["same_asset_class"]
    }
}

class ModelCompatibilityError(Exception):
    """Raised when model-symbol compatibility check fails"""
    pass

class SmartModelSelector:
    """
    Intelligent model selector that prevents cross-asset contamination.
    
    Key Features:
    - Strict symbol-model compatibility checking
    - Asset class enforcement (BTC models only for BTC)
    - Safe failure instead of wrong model selection
    - Detailed logging for model selection decisions
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize the smart model selector.
        
        Args:
            models_dir: Directory containing model files
        """
        self.models_dir = models_dir
        self.logger = logging.getLogger(__name__)
        
        # Cache for discovered models
        self.discovered_models = {}
        self.selection_history = []  # Track selection decisions for debugging
        
        # Discover available models on initialization
        self._discover_models()
    
    def _discover_models(self):
        """Discover and categorize available model files."""
        try:
            if not os.path.exists(self.models_dir):
                self.logger.warning(f"Models directory {self.models_dir} does not exist")
                return
            
            model_files = glob.glob(os.path.join(self.models_dir, "*.pkl"))
            self.discovered_models = {}
            
            for file_path in model_files:
                filename = os.path.basename(file_path)
                model_info = self._parse_model_filename(filename, file_path)
                
                if model_info:
                    model_type = model_info["model_type"]
                    symbol = model_info["symbol"]
                    
                    if model_type not in self.discovered_models:
                        self.discovered_models[model_type] = {}
                    
                    if symbol not in self.discovered_models[model_type]:
                        self.discovered_models[model_type][symbol] = []
                    
                    self.discovered_models[model_type][symbol].append(model_info)
            
            # Sort models by timestamp (newest first)
            for model_type in self.discovered_models:
                for symbol in self.discovered_models[model_type]:
                    self.discovered_models[model_type][symbol].sort(
                        key=lambda x: x.get("timestamp", ""), reverse=True
                    )
            
            self.logger.info(f"Discovered {len(model_files)} model files across "
                           f"{len(self.discovered_models)} model types")
            
        except Exception as e:
            self.logger.error(f"Error discovering models: {e}")
    
    def _parse_model_filename(self, filename: str, file_path: str) -> Optional[Dict]:
        """
        Parse model filename to extract metadata.
        
        Args:
            filename: Model filename
            file_path: Full path to model file
            
        Returns:
            Dictionary with model metadata or None if parsing fails
        """
        try:
            # Parse volatility classifier models: volatility_classifier_SYMBOL_TIMESTAMP.pkl
            if filename.startswith("volatility_classifier_"):
                parts = filename.replace("volatility_classifier_", "").replace(".pkl", "").split("_")
                if len(parts) >= 3:
                    # Extract symbol (handle symbols with underscores)
                    symbol_parts = parts[:-2]  # All except last 2 (timestamp)
                    timestamp_parts = parts[-2:]  # Last 2 are timestamp
                    
                    symbol_raw = "_".join(symbol_parts)
                    timestamp = "_".join(timestamp_parts)
                    
                    # Convert symbol format: BTC_USDTUSDT -> BTC/USDT:USDT
                    symbol = self._normalize_symbol(symbol_raw)
                    
                    return {
                        "model_type": "volatility_classifier",
                        "symbol": symbol,
                        "symbol_raw": symbol_raw,
                        "timestamp": timestamp,
                        "filename": filename,
                        "file_path": file_path,
                        "file_size": os.path.getsize(file_path)
                    }
            
            # Parse other model types here as needed
            elif filename.startswith("trend_model_"):
                # trend_model_SYMBOL_VERSION.pkl
                parts = filename.replace("trend_model_", "").replace(".pkl", "").split("_")
                if len(parts) >= 2:
                    symbol_raw = "_".join(parts[:-1])
                    version = parts[-1]
                    symbol = self._normalize_symbol(symbol_raw)
                    
                    return {
                        "model_type": "trend_predictor",
                        "symbol": symbol,
                        "symbol_raw": symbol_raw,
                        "version": version,
                        "filename": filename,
                        "file_path": file_path,
                        "file_size": os.path.getsize(file_path)
                    }
            
            # Generic fallback parsing
            else:
                return {
                    "model_type": "unknown",
                    "symbol": "unknown",
                    "filename": filename,
                    "file_path": file_path,
                    "file_size": os.path.getsize(file_path)
                }
                
        except Exception as e:
            self.logger.error(f"Error parsing model filename {filename}: {e}")
            return None
    
    def _normalize_symbol(self, symbol_raw: str) -> str:
        """
        Normalize symbol from filename format to standard format.
        
        Args:
            symbol_raw: Raw symbol from filename (e.g., BTC_USDTUSDT)
            
        Returns:
            Normalized symbol (e.g., BTC/USDT:USDT)
        """
        try:
            # Handle common patterns
            if symbol_raw.endswith("_USDTUSDT"):
                base = symbol_raw.replace("_USDTUSDT", "")
                return f"{base}/USDT:USDT"
            elif symbol_raw.endswith("_USDT"):
                base = symbol_raw.replace("_USDT", "")
                return f"{base}/USDT"
            elif "_" in symbol_raw:
                # Generic fallback
                parts = symbol_raw.split("_")
                if len(parts) == 2:
                    return f"{parts[0]}/{parts[1]}"
            
            # Return as-is if no pattern matches
            return symbol_raw
            
        except Exception:
            return symbol_raw
    
    def get_symbol_compatibility(self, symbol: str) -> Optional[SymbolCompatibility]:
        """
        Get compatibility configuration for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            SymbolCompatibility object or None if not configured
        """
        return SYMBOL_COMPATIBILITY.get(symbol)
    
    def is_model_compatible_with_symbol(self, model_type: str, model_symbol: str, target_symbol: str) -> Tuple[bool, str]:
        """
        Check if a model trained on model_symbol is compatible with target_symbol.
        
        Args:
            model_type: Type of model (e.g., 'volatility_classifier')
            model_symbol: Symbol the model was trained on
            target_symbol: Symbol we want to predict for
            
        Returns:
            Tuple of (is_compatible, reason)
        """
        try:
            # Get model restrictions
            restrictions = MODEL_SYMBOL_RESTRICTIONS.get(model_type, {})
            
            # If cross-asset is forbidden, check strict compatibility
            if restrictions.get("cross_asset_forbidden", False):
                if model_symbol != target_symbol:
                    return False, f"Cross-asset usage forbidden: {model_symbol} model cannot be used for {target_symbol}"
            
            # Check exact match requirement
            if restrictions.get("require_exact_match", False):
                if model_symbol != target_symbol:
                    return False, f"Exact match required: {model_symbol} != {target_symbol}"
            
            # Get compatibility configurations
            target_compat = self.get_symbol_compatibility(target_symbol)
            model_compat = self.get_symbol_compatibility(model_symbol)
            
            if not target_compat or not model_compat:
                return False, f"Compatibility configuration missing for {target_symbol} or {model_symbol}"
            
            # Check if target symbol allows the model symbol
            if model_symbol in target_compat.compatible_symbols:
                return True, f"Direct compatibility: {model_symbol} in {target_compat.compatible_symbols}"
            
            # Check asset class compatibility (only if not forbidden)
            if not target_compat.cross_asset_forbidden:
                if target_compat.asset_class == model_compat.asset_class:
                    return True, f"Same asset class: {target_compat.asset_class}"
            
            return False, f"No compatibility found between {model_symbol} and {target_symbol}"
            
        except Exception as e:
            self.logger.error(f"Error checking compatibility: {e}")
            return False, f"Compatibility check error: {str(e)}"
    
    def get_compatible_models(self, target_symbol: str, model_type: str) -> List[Dict]:
        """
        Get all models compatible with the target symbol.
        
        Args:
            target_symbol: Symbol to predict for
            model_type: Type of model needed
            
        Returns:
            List of compatible model dictionaries
        """
        compatible_models = []
        
        try:
            if model_type not in self.discovered_models:
                self.logger.warning(f"No models of type {model_type} discovered")
                return compatible_models
            
            for model_symbol, model_list in self.discovered_models[model_type].items():
                is_compatible, reason = self.is_model_compatible_with_symbol(
                    model_type, model_symbol, target_symbol
                )
                
                if is_compatible:
                    for model_info in model_list:
                        model_info["compatibility_reason"] = reason
                        compatible_models.append(model_info)
                        
                        self.logger.debug(f"Compatible model found: {model_info['filename']} - {reason}")
                else:
                    self.logger.debug(f"Incompatible model rejected: {model_symbol} for {target_symbol} - {reason}")
            
            # Sort by timestamp (newest first)
            compatible_models.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            
            return compatible_models
            
        except Exception as e:
            self.logger.error(f"Error getting compatible models: {e}")
            return compatible_models
    
    def select_best_model(self, symbol: str, model_type: str) -> Optional[str]:
        """
        Select the best compatible model for a symbol.
        
        Args:
            symbol: Target symbol
            model_type: Type of model needed
            
        Returns:
            Path to best model file or None if no compatible model found
        """
        try:
            self.logger.debug(f"Selecting best model for {symbol} (type: {model_type})")
            
            compatible_models = self.get_compatible_models(symbol, model_type)
            
            if not compatible_models:
                self.logger.error(f"No compatible models found for {symbol} (type: {model_type})")
                
                # Log available models for debugging
                if model_type in self.discovered_models:
                    available_symbols = list(self.discovered_models[model_type].keys())
                    self.logger.info(f"Available models for {model_type}: {available_symbols}")
                
                return None
            
            # Select the newest (first) compatible model
            best_model = compatible_models[0]
            
            # Record selection decision
            selection_record = {
                "target_symbol": symbol,
                "model_type": model_type,
                "selected_model": best_model["filename"],
                "model_symbol": best_model["symbol"],
                "compatibility_reason": best_model["compatibility_reason"],
                "alternatives_count": len(compatible_models) - 1,
                "timestamp": str(os.path.getctime(best_model["file_path"]))
            }
            
            self.selection_history.append(selection_record)
            
            self.logger.info(f"Selected model for {symbol}: {best_model['filename']} - {best_model['compatibility_reason']}")
            
            return best_model["file_path"]
            
        except Exception as e:
            self.logger.error(f"Error selecting best model for {symbol}: {e}")
            return None
    
    def validate_model_selection(self, symbol: str, model_path: str) -> bool:
        """
        Validate that a model selection is safe and appropriate.
        
        Args:
            symbol: Target symbol
            model_path: Path to selected model
            
        Returns:
            True if selection is valid
        """
        try:
            if not os.path.exists(model_path):
                self.logger.error(f"Model file does not exist: {model_path}")
                return False
            
            filename = os.path.basename(model_path)
            model_info = self._parse_model_filename(filename, model_path)
            
            if not model_info:
                self.logger.error(f"Could not parse model filename: {filename}")
                return False
            
            model_symbol = model_info["symbol"]
            model_type = model_info["model_type"]
            
            is_compatible, reason = self.is_model_compatible_with_symbol(
                model_type, model_symbol, symbol
            )
            
            if not is_compatible:
                self.logger.error(f"Model validation failed: {reason}")
                return False
            
            self.logger.info(f"Model validation passed: {filename} for {symbol} - {reason}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating model selection: {e}")
            return False
    
    def get_selection_stats(self) -> Dict:
        """
        Get statistics about model selections made.
        
        Returns:
            Dictionary with selection statistics
        """
        if not self.selection_history:
            return {"total_selections": 0}
        
        stats = {
            "total_selections": len(self.selection_history),
            "symbols_served": len(set(record["target_symbol"] for record in self.selection_history)),
            "models_used": len(set(record["selected_model"] for record in self.selection_history)),
            "cross_asset_selections": 0,
            "exact_match_selections": 0
        }
        
        for record in self.selection_history:
            if record["target_symbol"] != record["model_symbol"]:
                stats["cross_asset_selections"] += 1
            else:
                stats["exact_match_selections"] += 1
        
        return stats
    
    def clear_cache(self):
        """Clear discovery cache and selection history."""
        self.discovered_models.clear()
        self.selection_history.clear()
        self.logger.info("Model selector cache cleared")


# Utility functions for easy integration
def get_smart_model_path(symbol: str, model_type: str, models_dir: str = "models") -> Optional[str]:
    """
    Utility function to get model path using smart selection.
    
    Args:
        symbol: Target symbol
        model_type: Type of model needed
        models_dir: Models directory
        
    Returns:
        Path to best compatible model or None
    """
    selector = SmartModelSelector(models_dir)
    return selector.select_best_model(symbol, model_type)

def validate_symbol_model_compatibility(symbol: str, model_path: str) -> bool:
    """
    Utility function to validate symbol-model compatibility.
    
    Args:
        symbol: Target symbol
        model_path: Path to model file
        
    Returns:
        True if compatible
    """
    selector = SmartModelSelector()
    return selector.validate_model_selection(symbol, model_path)


if __name__ == "__main__":
    # Example usage and testing
    import sys
    
    if len(sys.argv) >= 3:
        symbol = sys.argv[1]
        model_type = sys.argv[2]
        
        selector = SmartModelSelector()
        model_path = selector.select_best_model(symbol, model_type)
        
        if model_path:
            print(f"[SUCCESS] Selected model for {symbol}: {os.path.basename(model_path)}")
            
            # Validate selection
            is_valid = selector.validate_model_selection(symbol, model_path)
            print(f"[VALIDATE] Validation: {'PASSED' if is_valid else 'FAILED'}")
        else:
            print(f"[ERROR] No compatible model found for {symbol} (type: {model_type})")
            
        # Show stats
        stats = selector.get_selection_stats()
        print(f"[STATS] Selection stats: {stats}")
    else:
        print("Usage: python model_selector.py SYMBOL MODEL_TYPE")
        print("Example: python model_selector.py BTC/USDT:USDT volatility_classifier")
