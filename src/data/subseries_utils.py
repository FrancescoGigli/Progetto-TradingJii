# subseries_utils.py

import pandas as pd
import numpy as np
import logging
from src.utils.config import SUBSERIES_LENGTH, SUBSERIES_MIN_SAMPLES

def create_subseries(df, length=SUBSERIES_LENGTH):
    """
    Split a time series into subseries of fixed length using sliding window.
    
    Args:
        df: DataFrame with time series data (volatility or price)
        length: Length of each subseries
        
    Returns:
        List of DataFrames, each containing a subseries
    """
    if df.empty:
        logging.warning("Empty DataFrame provided to create_subseries")
        return []
    
    if len(df) < length:
        logging.warning(f"DataFrame length {len(df)} is less than subseries length {length}")
        return []
    
    subseries_list = []
    for i in range(len(df) - length + 1):
        subseries_list.append(df.iloc[i:i+length])
    
    logging.debug(f"Created {len(subseries_list)} subseries of length {length}")
    return subseries_list

def categorize_subseries(subseries, column='close_volatility'):
    """
    Categorize a subseries based on up/down behavior of price volatility.
    
    Args:
        subseries: DataFrame representing a subseries
        column: The column to use for determining up/down behavior
        
    Returns:
        Category ID (string)
    """
    if column not in subseries.columns:
        raise ValueError(f"subseries must contain {column} column")
    
    # Extract the direction (up/down) of each time frame in the subseries
    # 1 for up (or neutral), 0 for down
    directions = (subseries[column] >= 0).astype(int)
    
    # Convert the binary pattern to a category ID
    # Example: [1,0,1,1,0,1,0] -> "1011010" -> category ID
    category_id = ''.join(directions.astype(str))
    
    return category_id

def get_all_subseries_with_categories(df, column='close_volatility', length=SUBSERIES_LENGTH):
    """
    Extract all subseries from a DataFrame and categorize them.
    
    Args:
        df: DataFrame with time series data
        column: Column to use for categorization
        length: Length of each subseries
        
    Returns:
        Dictionary mapping category IDs to lists of subseries
    """
    subseries_list = create_subseries(df, length)
    categorized = {}
    
    for subseries in subseries_list:
        try:
            category = categorize_subseries(subseries, column)
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(subseries)
        except Exception as e:
            logging.warning(f"Could not categorize subseries: {e}")
            continue
    
    # Filter categories with too few samples
    filtered_categories = {k: v for k, v in categorized.items() 
                         if len(v) >= SUBSERIES_MIN_SAMPLES}
    
    discarded = len(categorized) - len(filtered_categories)
    if discarded > 0:
        logging.info(f"Discarded {discarded} categories with fewer than {SUBSERIES_MIN_SAMPLES} samples")
    
    return filtered_categories

def get_category_distribution(categorized_subseries):
    """
    Get the distribution of subseries across categories.
    
    Args:
        categorized_subseries: Dictionary mapping category IDs to lists of subseries
        
    Returns:
        DataFrame with category IDs and their counts, sorted by count
    """
    categories = []
    counts = []
    
    for category, subseries_list in categorized_subseries.items():
        categories.append(category)
        counts.append(len(subseries_list))
    
    distribution = pd.DataFrame({
        'category': categories,
        'count': counts
    }).sort_values('count', ascending=False).reset_index(drop=True)
    
    return distribution

def binary_to_pattern(category_id):
    """
    Convert a binary category ID to a human-readable pattern.
    
    Args:
        category_id: Binary string like "1011010"
        
    Returns:
        Human-readable pattern like "↑↓↑↑↓↑↓"
    """
    pattern = ''
    for bit in category_id:
        if bit == '1':
            pattern += '↑'  # Up arrow for 1 (bullish)
        else:
            pattern += '↓'  # Down arrow for 0 (bearish)
    
    return pattern

def predict_next_category(current_category, transition_matrix):
    """
    Predict the next category based on the current category and transition probabilities.
    
    Args:
        current_category: The current category ID
        transition_matrix: Dictionary mapping current categories to {next_category: probability}
        
    Returns:
        Most likely next category ID
    """
    if current_category not in transition_matrix:
        logging.warning(f"Category {current_category} not found in transition matrix")
        return None
    
    # Get transition probabilities for current category
    transitions = transition_matrix[current_category]
    
    # Return the category with the highest probability
    return max(transitions, key=transitions.get)
