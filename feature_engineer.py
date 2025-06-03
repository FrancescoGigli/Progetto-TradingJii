#!/usr/bin/env python3
"""
FIXED Enhanced Feature Engineering - SENZA DATA LEAKAGE
Aggiunge features avanzate ai dataset SENZA usare il target 'y'
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class FixedEnhancedFeatureEngineer:
    """
    Feature Engineer CORRETTO senza data leakage.
    USA SOLO le features x_1...x_7, MAI il target 'y'.
    """
    
    def __init__(self):
        self.lag_periods = [1, 2, 3, 6, 12]  # lag periods
        self.rolling_windows = [3, 5, 10]    # rolling windows
        
    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea lag features SOLO per x_1...x_7 (MAI per y)"""
        print("  🔧 Creazione lag features (SENZA y)...")
        
        # SOLO x_1...x_7, MAI y!
        lag_cols = [f'x_{i}' for i in range(1, 8)]
        features_added = 0
        
        for col in lag_cols:
            if col in df.columns:
                for lag in self.lag_periods:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
                    features_added += 1
        
        print(f"    ✅ Aggiunte {features_added} lag features (SENZA data leakage)")
        return df
    
    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea rolling statistics SOLO per x_1...x_7 (MAI per y)"""
        print("  🔧 Creazione rolling features (SENZA y)...")
        
        # SOLO x_1...x_7, MAI y!
        rolling_cols = [f'x_{i}' for i in range(1, 8)]
        features_added = 0
        
        for col in rolling_cols:
            if col in df.columns:
                for window in self.rolling_windows:
                    df[f'{col}_rolling_{window}_mean'] = df[col].rolling(window=window, min_periods=1).mean()
                    df[f'{col}_rolling_{window}_std'] = df[col].rolling(window=window, min_periods=1).std()
                    df[f'{col}_rolling_{window}_min'] = df[col].rolling(window=window, min_periods=1).min()
                    df[f'{col}_rolling_{window}_max'] = df[col].rolling(window=window, min_periods=1).max()
                    features_added += 4
                    
                    # Rolling range
                    df[f'{col}_rolling_{window}_range'] = df[f'{col}_rolling_{window}_max'] - df[f'{col}_rolling_{window}_min']
                    features_added += 1
        
        print(f"    ✅ Aggiunte {features_added} rolling features (SENZA data leakage)")
        return df
    
    def create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea momentum features SOLO per x_1...x_7 (MAI per y)"""
        print("  🔧 Creazione momentum features (SENZA y)...")
        
        features_added = 0
        
        # SOLO x_1...x_7, MAI y!
        for col in [f'x_{i}' for i in range(1, 8)]:
            if col in df.columns:
                # Change over multiple periods
                for period in [1, 2, 3, 6]:
                    df[f'{col}_change_{period}'] = df[col].diff(period)
                    df[f'{col}_pct_change_{period}'] = df[col].pct_change(period)
                    features_added += 2
                
                # Acceleration (2nd derivative)
                df[f'{col}_acceleration'] = df[col].diff().diff()
                features_added += 1
        
        print(f"    ✅ Aggiunte {features_added} momentum features (SENZA data leakage)")
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea interaction features tra x_i"""
        print("  🔧 Creazione interaction features...")
        
        features_added = 0
        
        # Interazioni tra x_i e x_j
        for i in range(1, 8):
            for j in range(i+1, 8):
                col1, col2 = f'x_{i}', f'x_{j}'
                if col1 in df.columns and col2 in df.columns:
                    # Product interaction
                    df[f'{col1}_{col2}_product'] = df[col1] * df[col2]
                    features_added += 1
                    
                    # Ratio interaction (avoid division by zero)
                    df[f'{col1}_{col2}_ratio'] = df[col1] / (df[col2] + 1e-10)
                    features_added += 1
                    
                    # Difference
                    df[f'{col1}_{col2}_diff'] = df[col1] - df[col2]
                    features_added += 1
        
        # Sum features
        x_cols = [f'x_{i}' for i in range(1, 8)]
        if all(col in df.columns for col in x_cols):
            df['x_sum'] = df[x_cols].sum(axis=1)
            df['x_mean'] = df[x_cols].mean(axis=1)
            df['x_std'] = df[x_cols].std(axis=1)
            df['x_min'] = df[x_cols].min(axis=1)
            df['x_max'] = df[x_cols].max(axis=1)
            df['x_range'] = df['x_max'] - df['x_min']
            features_added += 6
        
        print(f"    ✅ Aggiunte {features_added} interaction features")
        return df
    
    def create_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea features basate sul pattern"""
        print("  🔧 Creazione pattern features...")
        
        features_added = 0
        
        if 'pattern' in df.columns:
            # Convert pattern to string if not already
            df['pattern'] = df['pattern'].astype(str)
            
            # Pattern length
            df['pattern_length'] = df['pattern'].str.len()
            features_added += 1
            
            # Count of 1s and 0s in pattern
            df['pattern_ones'] = df['pattern'].str.count('1')
            df['pattern_zeros'] = df['pattern'].str.count('0')
            features_added += 2
            
            # Pattern ratio
            df['pattern_ratio'] = df['pattern_ones'] / (df['pattern_length'] + 1e-10)
            features_added += 1
            
            # Pattern changes (alternations)
            df['pattern_changes'] = df['pattern'].apply(
                lambda x: sum(1 for i in range(1, len(str(x))) if str(x)[i] != str(x)[i-1])
            )
            features_added += 1
        
        print(f"    ✅ Aggiunte {features_added} pattern features")
        return df
    
    def create_sequence_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea features di sequenza temporale SOLO per x_1...x_7"""
        print("  🔧 Creazione sequence features (SENZA y)...")
        
        features_added = 0
        
        # Trend detection SOLO su x_1...x_7
        for col in [f'x_{i}' for i in range(1, 8)]:
            if col in df.columns:
                # Simple trend (3-period)
                df[f'{col}_trend_3'] = np.where(
                    df[col] > df[col].shift(1), 1,
                    np.where(df[col] < df[col].shift(1), -1, 0)
                )
                features_added += 1
                
                # Cumulative trend
                df[f'{col}_cum_trend'] = df[f'{col}_trend_3'].cumsum()
                features_added += 1
        
        # Distance from moving averages SOLO per x_1...x_7
        for col in [f'x_{i}' for i in range(1, 8)]:
            if col in df.columns:
                ma_5 = df[col].rolling(window=5, min_periods=1).mean()
                df[f'{col}_dist_ma5'] = df[col] - ma_5
                features_added += 1
        
        print(f"    ✅ Aggiunte {features_added} sequence features (SENZA data leakage)")
        return df
    
    def clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pulisce features da valori infiniti e NaN"""
        print("  🧹 Pulizia features...")
        
        # Replace infinite values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN with forward fill, then backward fill, then 0
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Remove constant features (zero variance)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        constant_cols = []
        for col in numeric_cols:
            if df[col].var() == 0:
                constant_cols.append(col)
        
        if constant_cols:
            df = df.drop(columns=constant_cols)
            print(f"    🗑️ Rimosse {len(constant_cols)} features costanti")
        
        return df
    
    def enhance_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main function per enhancement del dataset SENZA data leakage"""
        print("🚀 FIXED ENHANCED FEATURE ENGINEERING (SENZA DATA LEAKAGE)")
        print(f"  📊 Dataset input: {df.shape}")
        
        # VERIFICA: Il target y NON deve essere usato per features!
        if 'y' in df.columns:
            print(f"  ⚠️ ATTENZIONE: Target 'y' presente ma NON verrà usato per features!")
        
        original_features = df.shape[1]
        
        # Apply all feature engineering steps (SENZA y!)
        df = self.create_lag_features(df)
        df = self.create_rolling_features(df)
        df = self.create_momentum_features(df)
        df = self.create_interaction_features(df)
        df = self.create_pattern_features(df)
        df = self.create_sequence_features(df)
        df = self.clean_features(df)
        
        new_features = df.shape[1]
        added_features = new_features - original_features
        
        print(f"  📊 Dataset output: {df.shape}")
        print(f"  ✅ Features aggiunte: {added_features}")
        print(f"  🛡️ NESSUN DATA LEAKAGE: Target 'y' mai usato per features!")
        print(f"  🎯 Accuracy realistica attesa: 55-70% (non più 100%)")
        
        return df

def enhance_ml_datasets_fixed():
    """Enhance tutti i dataset ML con versione CORRETTA senza data leakage"""
    print("\n" + "="*60)
    print("🛡️ FIXED ENHANCED FEATURE ENGINEERING - SENZA DATA LEAKAGE")
    print("="*60)
    
    enhancer = FixedEnhancedFeatureEngineer()
    ml_datasets_path = Path("ml_datasets")
    
    if not ml_datasets_path.exists():
        print("❌ Directory ml_datasets non trovata")
        return
    
    enhanced_count = 0
    
    for symbol_dir in ml_datasets_path.iterdir():
        if symbol_dir.is_dir():
            symbol = symbol_dir.name
            
            for timeframe_dir in symbol_dir.iterdir():
                if timeframe_dir.is_dir():
                    timeframe = timeframe_dir.name
                    merged_file = timeframe_dir / "merged.csv"
                    
                    if merged_file.exists():
                        try:
                            print(f"\n🔧 Processing {symbol} ({timeframe})...")
                            
                            # Load dataset
                            df = pd.read_csv(merged_file)
                            original_shape = df.shape
                            
                            # Enhance features SENZA data leakage
                            df_enhanced = enhancer.enhance_dataset(df)
                            
                            # Save enhanced dataset
                            enhanced_file = timeframe_dir / "merged_enhanced_fixed.csv"
                            df_enhanced.to_csv(enhanced_file, index=False)
                            
                            print(f"  💾 Salvato: {enhanced_file}")
                            print(f"  📊 {original_shape} -> {df_enhanced.shape}")
                            
                            enhanced_count += 1
                            
                        except Exception as e:
                            print(f"  ❌ Errore processing {symbol} ({timeframe}): {e}")
                            import traceback
                            traceback.print_exc()
    
    print(f"\n✅ FIXED Enhancement completato! Dataset processati: {enhanced_count}")
    print("🛡️ Usa 'merged_enhanced_fixed.csv' per training senza data leakage")
    print("🎯 Accuracy attesa: 55-70% (realistico per trading)")
    print("="*60)

# Alias for compatibility with pipeline
def enhance_ml_datasets_v2():
    """Alias for enhance_ml_datasets_fixed for pipeline compatibility"""
    return enhance_ml_datasets_fixed()

if __name__ == "__main__":
    enhance_ml_datasets_fixed()
