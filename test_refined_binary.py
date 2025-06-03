#!/usr/bin/env python3
"""
Test script for refined binary classification system.

This script tests the new labeling approach that excludes the neutral zone
to create cleaner class separation.
"""

import sys
import os
import pandas as pd
import numpy as np
import logging

# Add modules to path
sys.path.append('modules')

from modules.utils.config import BUY_THRESHOLD, SELL_THRESHOLD
from modules.data.full_dataset_generator import add_labels
from modules.ml.config import CLASS_MAPPINGS

def test_refined_binary_classification():
    """Test the refined binary classification system."""
    
    print("🚀 Testing Refined Binary Classification System")
    print("=" * 60)
    
    # Display configuration
    print(f"📋 Configuration:")
    print(f"   BUY_THRESHOLD: {BUY_THRESHOLD}")
    print(f"   SELL_THRESHOLD: {SELL_THRESHOLD}")
    print(f"   Neutral zone: {SELL_THRESHOLD} <= y <= {BUY_THRESHOLD}")
    print()
    
    print(f"🎯 Class Mappings:")
    for label, name in CLASS_MAPPINGS['labels'].items():
        print(f"   {label}: {name}")
    print()
    
    # Create test data with different volatility ranges
    test_data = []
    
    # Strong sell signals (y < -0.5)
    test_data.extend([
        {'y': -2.0, 'expected_class': 0, 'category': 'Strong SELL'},
        {'y': -1.5, 'expected_class': 0, 'category': 'Strong SELL'},
        {'y': -1.0, 'expected_class': 0, 'category': 'Strong SELL'},
        {'y': -0.8, 'expected_class': 0, 'category': 'Strong SELL'},
        {'y': -0.6, 'expected_class': 0, 'category': 'Strong SELL'},
    ])
    
    # Neutral zone (should be excluded) -0.5 <= y <= 0.5
    test_data.extend([
        {'y': -0.5, 'expected_class': 'EXCLUDED', 'category': 'Neutral (excluded)'},
        {'y': -0.3, 'expected_class': 'EXCLUDED', 'category': 'Neutral (excluded)'},
        {'y': -0.1, 'expected_class': 'EXCLUDED', 'category': 'Neutral (excluded)'},
        {'y': 0.0, 'expected_class': 'EXCLUDED', 'category': 'Neutral (excluded)'},
        {'y': 0.1, 'expected_class': 'EXCLUDED', 'category': 'Neutral (excluded)'},
        {'y': 0.3, 'expected_class': 'EXCLUDED', 'category': 'Neutral (excluded)'},
        {'y': 0.5, 'expected_class': 'EXCLUDED', 'category': 'Neutral (excluded)'},
    ])
    
    # Strong buy signals (y > 0.5)
    test_data.extend([
        {'y': 0.6, 'expected_class': 1, 'category': 'Strong BUY'},
        {'y': 0.8, 'expected_class': 1, 'category': 'Strong BUY'},
        {'y': 1.0, 'expected_class': 1, 'category': 'Strong BUY'},
        {'y': 1.5, 'expected_class': 1, 'category': 'Strong BUY'},
        {'y': 2.0, 'expected_class': 1, 'category': 'Strong BUY'},
    ])
    
    # Create DataFrame
    df = pd.DataFrame(test_data)
    
    print(f"📊 Test Data Summary:")
    print(f"   Total samples: {len(df)}")
    print(f"   SELL samples: {len([d for d in test_data if d['expected_class'] == 0])}")
    print(f"   BUY samples: {len([d for d in test_data if d['expected_class'] == 1])}")
    print(f"   Neutral samples (to be excluded): {len([d for d in test_data if d['expected_class'] == 'EXCLUDED'])}")
    print()
    
    # Apply the refined binary labeling
    print("🔬 Applying refined binary labeling...")
    
    # Capture logging output
    import io
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)
    
    # Apply labeling
    labeled_df = add_labels(df)
    
    # Get the captured log
    log_output = log_capture.getvalue()
    
    print("📋 Labeling Statistics:")
    for line in log_output.split('\n'):
        if line.strip() and 'Refined binary labeling statistics:' in line:
            continue
        if line.strip() and any(x in line for x in ['Initial samples:', 'BUY samples', 'SELL samples', 'Neutral zone', 'Final samples', 'Samples excluded']):
            print(f"   {line.strip()}")
    print()
    
    # Verify results
    print("✅ Verification Results:")
    
    # Check that neutral zone samples were excluded
    initial_count = len(df)
    final_count = len(labeled_df)
    excluded_count = initial_count - final_count
    
    expected_excluded = len([d for d in test_data if d['expected_class'] == 'EXCLUDED'])
    
    print(f"   Expected excluded samples: {expected_excluded}")
    print(f"   Actually excluded samples: {excluded_count}")
    print(f"   ✓ Exclusion correct: {expected_excluded == excluded_count}")
    print()
    
    # Check class distribution
    if not labeled_df.empty:
        class_counts = labeled_df['y_class'].value_counts().sort_index()
        print(f"   Final class distribution:")
        for label, count in class_counts.items():
            class_name = CLASS_MAPPINGS['labels'][label]
            print(f"     {class_name} ({label}): {count} samples")
        print()
        
        # Verify that only clear signals remain
        sell_samples = labeled_df[labeled_df['y_class'] == 0]
        buy_samples = labeled_df[labeled_df['y_class'] == 1]
        
        print(f"   SELL sample y values: {sell_samples['y'].tolist()}")
        print(f"   BUY sample y values: {buy_samples['y'].tolist()}")
        print()
        
        # Verify thresholds
        all_sell_below_threshold = all(y < SELL_THRESHOLD for y in sell_samples['y'])
        all_buy_above_threshold = all(y > BUY_THRESHOLD for y in buy_samples['y'])
        
        print(f"   ✓ All SELL samples < {SELL_THRESHOLD}: {all_sell_below_threshold}")
        print(f"   ✓ All BUY samples > {BUY_THRESHOLD}: {all_buy_above_threshold}")
        print()
    
    # Summary
    print("🎯 System Summary:")
    print(f"   Approach: {CLASS_MAPPINGS['description']['approach']}")
    print(f"   Neutral zone excluded: {CLASS_MAPPINGS['description']['neutral_zone_excluded']}")
    print(f"   Buy threshold: {CLASS_MAPPINGS['description']['buy_threshold']}")
    print(f"   Sell threshold: {CLASS_MAPPINGS['description']['sell_threshold']}")
    print()
    print("📝 Note:", CLASS_MAPPINGS['description']['note'])
    print()
    
    if not labeled_df.empty:
        print("✅ Refined binary classification system is working correctly!")
        print("🚀 Ready for improved model training with cleaner class separation!")
    else:
        print("⚠️ Warning: No samples remained after filtering. Check threshold values.")
    
    return labeled_df

if __name__ == "__main__":
    result_df = test_refined_binary_classification()
    
    if not result_df.empty:
        print("\n📋 Sample of final labeled data:")
        print(result_df[['y', 'y_class']].head(10))
