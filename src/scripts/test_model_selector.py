# test_model_selector.py

import logging
import pandas as pd
import numpy as np
import sys
import os
import argparse
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.volatility_utils import calculate_volatility_rate
from src.data.subseries_utils import binary_to_pattern, get_all_subseries_with_categories
from src.utils.config import DB_FILE
from src.data.db_manager import get_symbol_data, init_data_tables
from src.data.model_selector import get_model_selector

def generate_training_data(n_samples=100, save_to_db=True):
    """
    Generate training data and save to database.
    
    Args:
        n_samples: Number of random data points to generate
        save_to_db: Whether to save data to database
        
    Returns:
        DataFrame with generated data and volatility
    """
    logging.info(f"Generating {n_samples} random data points for training...")
    
    # Create random price data
    dates = pd.date_range(start='2023-01-01', periods=n_samples)
    
    # Create simulated price pattern with random walk
    close_prices = [100]
    for i in range(1, n_samples):
        # Random walk with slight upward bias
        change = np.random.normal(0.1, 1.0)
        close_prices.append(close_prices[-1] * (1 + change/100))
    
    # Generate OHLCV data with simulated price pattern
    data = {
        'close': close_prices,
        'open': [p * (1 + np.random.normal(0, 0.01)) for p in close_prices],
        'high': [p * (1 + abs(np.random.normal(0, 0.015))) for p in close_prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.015))) for p in close_prices],
        'volume': np.random.normal(1000, 200, n_samples)
    }
    df = pd.DataFrame(data, index=dates)
    
    # Calculate volatility
    df_vol = calculate_volatility_rate(df)
    print(f"\n==== Generated {len(df_vol)} data points with volatility ====")
    print(df_vol[['close', 'close_volatility']].head())
    
    # Get all subseries and categorize them
    categorized = get_all_subseries_with_categories(df_vol)
    print(f"\n==== Found {len(categorized)} categories in generated data ====")
    
    for category, subseries_list in list(categorized.items())[:5]:
        pattern = binary_to_pattern(category)
        print(f"Category: {category}, Pattern: {pattern}, Count: {len(subseries_list)}")
    
    # Save to database if requested
    if save_to_db:
        from src.data.db_manager import save_subseries_data
        
        print("\n==== Saving subseries to database ====")
        symbol = "SAMPLE"
        timeframe = "train"
        
        # Save to database
        count = save_subseries_data(symbol, timeframe, categorized)
        print(f"Saved {count} subseries to database.")
    
    return df_vol
    
def test_with_random_data(n_samples=30):
    """
    Test the model selector with random data.
    
    Args:
        n_samples: Number of random data points to generate
        
    Returns:
        DataFrame with generated data and volatility
    """
    logging.info(f"Testing model selector with {n_samples} random data points...")
    
    # Create random price data
    dates = pd.date_range(start='2023-01-01', periods=n_samples)
    data = {
        'open': np.random.normal(100, 2, n_samples),
        'high': np.random.normal(102, 2, n_samples),
        'low': np.random.normal(98, 2, n_samples),
        'close': np.random.normal(101, 2, n_samples),
        'volume': np.random.normal(1000, 200, n_samples)
    }
    df = pd.DataFrame(data, index=dates)
    
    # Calculate volatility
    df_vol = calculate_volatility_rate(df)
    print(f"\n==== Generated {len(df_vol)} data points with volatility ====")
    print(df_vol[['close', 'close_volatility']].head())
    
    # Get model selector
    selector = get_model_selector()
    
    # Get current category
    current_category, pattern = selector.categorize_current_data(df_vol)
    print(f"\n==== Current pattern: {pattern} ({current_category}) ====")
    
    if current_category:
        # Predict next category
        next_cat, next_pattern, prob = selector.predict_next_category(current_category)
        if next_cat:
            print(f"\n==== Prediction ====")
            print(f"Current: {pattern} ({current_category})")
            print(f"Next: {next_pattern} ({next_cat}) - Probability: {prob:.4f}")
            
            # Get multi-step prediction
            predictions = selector.predict_multiple_steps(current_category, steps=3)
            if predictions:
                print(f"\n==== Multi-step prediction ====")
                for i, (cat, pat, p) in enumerate(predictions):
                    print(f"Step {i+1}: {pat} ({cat}) - Prob: {p:.4f}")
            else:
                print("\nNo multi-step predictions available.")
        else:
            print(f"\nNo prediction available for {current_category}")
    
    return df_vol

def test_with_real_data(symbol, timeframe, limit=100):
    """
    Test the model selector with real data from the database.
    
    Args:
        symbol: Symbol to use for testing
        timeframe: Timeframe to use
        limit: Limit on number of data points
        
    Returns:
        DataFrame with data
    """
    logging.info(f"Testing model selector with real data: {symbol} ({timeframe})...")
    
    # Get data from the database
    df = get_symbol_data(symbol, timeframe, limit)
    
    if df.empty:
        print(f"No data found for {symbol} ({timeframe})")
        return None
    
    print(f"\n==== Retrieved {len(df)} data points for {symbol} ({timeframe}) ====")
    
    # Calculate volatility if needed
    if 'close_volatility' not in df.columns:
        df_vol = calculate_volatility_rate(df)
    else:
        df_vol = df
    
    # Get model selector
    selector = get_model_selector()
    
    # Get current category
    current_category, pattern = selector.categorize_current_data(df_vol)
    print(f"\n==== Current pattern: {pattern} ({current_category}) ====")
    
    # Get category info
    cat_info = selector.get_category_info(current_category)
    print(f"\n==== Category Information ====")
    print(f"Category: {cat_info['category_id']}")
    print(f"Pattern: {cat_info['pattern']}")
    print(f"Count: {cat_info['count']}")
    
    # Print top transitions
    print(f"\n==== Top Transitions ====")
    transitions = sorted(cat_info['transitions'].items(), key=lambda x: x[1], reverse=True)
    for to_cat, prob in transitions[:3]:
        to_pattern = binary_to_pattern(to_cat)
        print(f"To {to_pattern} ({to_cat}): {prob:.4f}")
    
    # Predict next category
    next_cat, next_pattern, prob = selector.predict_next_category(current_category)
    if next_cat:
        print(f"\n==== Prediction ====")
        print(f"Current: {pattern} ({current_category})")
        print(f"Next: {next_pattern} ({next_cat}) - Probability: {prob:.4f}")
        
        # Get multi-step prediction
        predictions = selector.predict_multiple_steps(current_category, steps=3)
        if predictions:
            print(f"\n==== Multi-step prediction ====")
            for i, (cat, pat, p) in enumerate(predictions):
                print(f"Step {i+1}: {pat} ({cat}) - Prob: {p:.4f}")
        else:
            print("\nNo multi-step predictions available.")
    else:
        print(f"\nNo prediction available for {current_category}")
    
    return df_vol

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Test the model selector with sample or real data")
    parser.add_argument('--symbol', type=str, default="BTC/USDT", help="Symbol to use for testing")
    parser.add_argument('--timeframe', type=str, default="1h", help="Timeframe to use for testing")
    parser.add_argument('--limit', type=int, default=100, help="Limit for querying data")
    parser.add_argument('--random', action='store_true', help="Test with random data")
    parser.add_argument('--samples', type=int, default=30, help="Number of random samples")
    parser.add_argument('--train', action='store_true', help="Generate training data before testing")
    parser.add_argument('--train-size', type=int, default=200, help="Number of training samples to generate")
    parser.add_argument('--no-save', action='store_true', help="Do not save training data to database")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("\n===== MODEL SELECTOR TEST =====")
    
    # Initialize database tables if they don't exist
    init_data_tables()
    
    # Generate training data if requested
    if args.train:
        print("\n----- GENERATING TRAINING DATA -----")
        generate_training_data(n_samples=args.train_size, save_to_db=not args.no_save)
    
    print("\n----- TESTING MODEL PREDICTIONS -----")
    
    if args.random:
        # Test with random data
        test_with_random_data(args.samples)
    else:
        # Check if database exists
        if os.path.exists(DB_FILE):
            # Test with real data if available
            test_with_real_data(args.symbol, args.timeframe, args.limit)
        else:
            print(f"Database file {DB_FILE} not found. Testing with random data instead...")
            test_with_random_data(args.samples)
    
    print("\n===== DONE =====")
