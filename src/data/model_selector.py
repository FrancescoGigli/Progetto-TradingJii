# model_selector.py

import logging
import numpy as np
import pandas as pd
from src.data.db_manager import get_category_transitions, get_top_categories
from src.data.subseries_utils import binary_to_pattern, categorize_subseries
from src.utils.config import SUBSERIES_LENGTH

class MarkovModelSelector:
    """
    Model selector that uses Markov chains to predict the next category.
    This is the first step in the approach described in the paper, where
    we identify patterns in the volatility series and select models based
    on these patterns.
    """
    
    def __init__(self):
        """Initialize the model selector."""
        self.transitions = None
        self.top_categories = None
        self.load_transition_probabilities()
    
    def load_transition_probabilities(self):
        """Load transition probabilities from the database."""
        try:
            self.transitions = get_category_transitions()
            logging.info(f"Loaded {len(self.transitions)} category transitions")
            
            self.top_categories = {cat_id: (pattern, count) for cat_id, pattern, count in get_top_categories(limit=20)}
            logging.info(f"Loaded {len(self.top_categories)} top categories")
            
            return True
        except Exception as e:
            logging.error(f"Error loading transition probabilities: {e}")
            self.transitions = {}
            self.top_categories = {}
            return False
    
    def predict_next_category(self, current_category):
        """
        Predict the next category based on the current category.
        
        Args:
            current_category: The current category ID
            
        Returns:
            Tuple of (next_category_id, pattern, probability)
        """
        if not self.transitions:
            self.load_transition_probabilities()
        
        if not self.transitions or current_category not in self.transitions:
            logging.warning(f"No transition data for category {current_category}")
            return None, None, 0.0
        
        # Get transition probabilities for this category
        transitions = self.transitions[current_category]
        
        # Get the most likely next category
        next_category = max(transitions.items(), key=lambda x: x[1])
        next_category_id, probability = next_category
        
        # Get the pattern for this category
        pattern = None
        if next_category_id in self.top_categories:
            pattern = self.top_categories[next_category_id][0]
        else:
            pattern = binary_to_pattern(next_category_id)
        
        return next_category_id, pattern, probability
    
    def predict_multiple_steps(self, current_category, steps=5):
        """
        Predict a sequence of categories for multiple steps ahead.
        
        Args:
            current_category: The starting category ID
            steps: Number of steps to predict
            
        Returns:
            List of tuples (category_id, pattern, probability)
        """
        if not self.transitions:
            self.load_transition_probabilities()
        
        if not self.transitions or current_category not in self.transitions:
            logging.warning(f"No transition data for category {current_category}")
            return []
        
        predictions = []
        current = current_category
        
        for _ in range(steps):
            next_cat, pattern, prob = self.predict_next_category(current)
            if not next_cat:
                break
                
            predictions.append((next_cat, pattern, prob))
            current = next_cat
        
        return predictions
    
    def get_category_info(self, category_id):
        """
        Get information about a category.
        
        Args:
            category_id: The category ID to get info for
            
        Returns:
            Dictionary with category information
        """
        if category_id in self.top_categories:
            pattern, count = self.top_categories[category_id]
        else:
            pattern = binary_to_pattern(category_id)
            count = 0
        
        return {
            'category_id': category_id,
            'pattern': pattern,
            'count': count,
            'transitions': self.transitions.get(category_id, {})
        }
    
    def categorize_current_data(self, df):
        """
        Categorize the most recent data to get the current category.
        
        Args:
            df: DataFrame with volatility data
            
        Returns:
            Current category ID and pattern
        """
        if len(df) < SUBSERIES_LENGTH:
            logging.warning(f"Not enough data points ({len(df)}) for categorization (need {SUBSERIES_LENGTH})")
            return None, None
        
        # Get the most recent subseries
        current_subseries = df.iloc[-SUBSERIES_LENGTH:]
        
        try:
            # Categorize the current subseries
            category_id = categorize_subseries(current_subseries)
            pattern = binary_to_pattern(category_id)
            
            return category_id, pattern
        except Exception as e:
            logging.error(f"Error categorizing current data: {e}")
            return None, None

def get_model_selector():
    """Factory function to create and return a model selector."""
    return MarkovModelSelector()

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    selector = get_model_selector()
    
    # Get the top categories
    top_categories = get_top_categories(limit=5)
    
    if top_categories:
        print("\n===== Testing Model Selector =====")
        
        # Try prediction with the top category
        top_category_id = top_categories[0][0]
        print(f"\nTop category: {top_category_id} ({binary_to_pattern(top_category_id)})")
        
        next_cat, pattern, prob = selector.predict_next_category(top_category_id)
        if next_cat:
            print(f"Next predicted category: {next_cat} ({pattern})")
            print(f"Probability: {prob:.4f}")
            
            # Get multi-step prediction
            predictions = selector.predict_multiple_steps(top_category_id, steps=3)
            print("\nMulti-step prediction:")
            for i, (cat, pat, p) in enumerate(predictions):
                print(f"Step {i+1}: {cat} ({pat}) - Prob: {p:.4f}")
        else:
            print("No prediction available.")
    else:
        print("No categories found. Make sure to run the volatility test with --save first.")
