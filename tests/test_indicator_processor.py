#!/usr/bin/env python3
"""
Tests for indicator_processor.py
"""

import unittest
import sqlite3
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to the path to import modules properly
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data.indicator_processor import (
    init_indicator_tables,
    calculate_indicators,
    save_indicators,
    _validate_indicators
)

# Test database file path
TEST_DB_FILE = 'test_indicators.db'

class TestIndicatorProcessor(unittest.TestCase):
    """Test cases for indicator_processor functionality"""
    
    def setUp(self):
        """Set up test environment before each test"""
        # Create test database
        self.conn = sqlite3.connect(TEST_DB_FILE)
        self.cursor = self.conn.cursor()
        
        # Override the DB_FILE in the module
        import modules.data.indicator_processor as ip
        ip.DB_FILE = TEST_DB_FILE
        
        # Create test data tables
        self._create_test_data_table()
        
        # Initialize indicator tables
        init_indicator_tables(['5m'])
        
    def tearDown(self):
        """Clean up after each test"""
        self.conn.close()
        # Delete test database file
        if os.path.exists(TEST_DB_FILE):
            os.remove(TEST_DB_FILE)
    
    def _create_test_data_table(self):
        """Create test data tables and insert sample data"""
        # Create data table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS data_5m (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                UNIQUE(symbol, timestamp)
            )
        """)
        
        # Insert sample data
        base_time = datetime.now() - timedelta(days=10)
        symbol = "BTC/USDT"
        
        # Generate 300 sample data points (enough for all indicators including EMA200)
        for i in range(300):
            timestamp = base_time + timedelta(minutes=5 * i)
            close = 30000 + np.sin(i/10) * 2000  # Sine wave around $30,000
            open_price = close - np.random.normal(0, 100)
            high = max(close, open_price) + np.random.normal(50, 20)
            low = min(close, open_price) - np.random.normal(50, 20)
            volume = 10000 + np.random.normal(0, 1000)
            
            self.cursor.execute("""
                INSERT INTO data_5m (symbol, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (symbol, timestamp.strftime('%Y-%m-%dT%H:%M:%S'), 
                 open_price, high, low, close, volume))
            
        self.conn.commit()
    
    def test_table_creation(self):
        """Test that indicator tables are properly created"""
        # Check if the table exists
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ta_5m'")
        table_exists = self.cursor.fetchone() is not None
        self.assertTrue(table_exists, "Indicator table should be created")
        
        # Check if table has the correct columns
        if table_exists:
            self.cursor.execute("PRAGMA table_info(ta_5m)")
            columns = self.cursor.fetchall()
            column_names = [col[1] for col in columns]
            
            # Check for essential columns
            for expected_col in ["symbol", "timestamp", "sma9", "ema20", "rsi14", "bbands_upper", "adx14"]:
                self.assertIn(expected_col, column_names, f"Column {expected_col} should exist in ta_5m")
    
    def test_calculate_indicators(self):
        """Test indicator calculation with dummy data"""
        # Load test data
        query = """SELECT timestamp, open, high, low, close, volume
                  FROM data_5m WHERE symbol = 'BTC/USDT' ORDER BY timestamp ASC"""
        df = pd.read_sql_query(query, self.conn)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Run indicator calculation
        result_df = calculate_indicators(df)
        
        # Check that we have results
        self.assertFalse(result_df.empty, "Result DataFrame should not be empty")
        
        # Check that we have the expected columns
        expected_cols = ['timestamp', 'sma9', 'sma20', 'ema20', 'rsi14']
        for col in expected_cols:
            self.assertIn(col, result_df.columns, f"Column {col} should exist in result")
        
        # Check that indicators are mostly non-null
        # We expect some NaN values at the start of the series due to lookback periods
        for col in result_df.columns:
            if col != 'timestamp':
                non_null_pct = result_df[col].count() / len(result_df) * 100
                self.assertTrue(non_null_pct > 70, f"Column {col} should be mostly non-null (got {non_null_pct:.1f}%)")
    
    def test_save_indicators(self):
        """Test saving indicators to database"""
        # Load test data
        query = """SELECT timestamp, open, high, low, close, volume
                  FROM data_5m WHERE symbol = 'BTC/USDT' ORDER BY timestamp ASC"""
        df = pd.read_sql_query(query, self.conn)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate indicators
        result_df = calculate_indicators(df)
        
        # Save indicators
        symbol = "BTC/USDT"
        timeframe = "5m"
        success = save_indicators(symbol, timeframe, result_df)
        
        # Check save was successful
        self.assertTrue(success, "Indicator save should succeed")
        
        # Check we have rows in the indicators table
        self.cursor.execute("SELECT COUNT(*) FROM ta_5m WHERE symbol = ?", (symbol,))
        count = self.cursor.fetchone()[0]
        self.assertGreater(count, 0, "Should have saved some indicator records")
        
        # Verify that saved data matches original calculation
        # Get a sample record from the database
        self.cursor.execute("""
            SELECT timestamp, rsi14, sma20, ema20
            FROM ta_5m
            WHERE symbol = ? AND rsi14 IS NOT NULL
            LIMIT 1
        """, (symbol,))
        db_row = self.cursor.fetchone()
        
        if db_row:
            # Find the corresponding row in the calculated DataFrame
            timestamp = db_row[0]
            db_timestamp = datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S')
            orig_row = result_df[result_df['timestamp'] == pd.Timestamp(db_timestamp)]
            
            if not orig_row.empty:
                # Check RSI value matches (with small tolerance for floating point differences)
                db_rsi = db_row[1]
                calc_rsi = orig_row['rsi14'].values[0]
                self.assertAlmostEqual(db_rsi, calc_rsi, places=4, 
                                      msg="RSI value in DB should match calculated value")

if __name__ == '__main__':
    unittest.main()
