# check_database.py
import sqlite3
import os
import sys
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.config import DB_FILE

def check_database():
    """Check the database structure and available data."""
    if not os.path.exists(DB_FILE):
        print(f"Database file {DB_FILE} not found.")
        return
    
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Get list of tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print("\n==== Database Tables ====")
        for table in tables:
            print(table[0])
        
        # Check data tables
        for table in [t[0] for t in tables if t[0].startswith('data_')]:
            timeframe = table.split('_')[1]
            
            # Get symbol count
            cursor.execute(f"SELECT COUNT(DISTINCT symbol) FROM {table}")
            symbol_count = cursor.fetchone()[0]
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            row_count = cursor.fetchone()[0]
            
            # Get sample symbols
            cursor.execute(f"SELECT DISTINCT symbol FROM {table} LIMIT 5")
            sample_symbols = [s[0] for s in cursor.fetchall()]
            
            print(f"\n==== Table: {table} ====")
            print(f"Timeframe: {timeframe}")
            print(f"Symbol count: {symbol_count}")
            print(f"Row count: {row_count}")
            print(f"Sample symbols: {', '.join(sample_symbols) if sample_symbols else 'None'}")
            
            # If we have symbols, get a sample of data for the first symbol
            if sample_symbols:
                symbol = sample_symbols[0]
                print(f"\n==== Sample Data for {symbol} ({timeframe}) ====")
                
                # Check if we have volatility columns
                cursor.execute(f"PRAGMA table_info({table})")
                columns = [col[1] for col in cursor.fetchall()]
                has_volatility = 'close_volatility' in columns
                
                if has_volatility:
                    print("Volatility columns present.")
                else:
                    print("Volatility columns NOT present.")
                
                # Get sample data
                query = f"SELECT timestamp, open, high, low, close, volume FROM {table} WHERE symbol = ? ORDER BY timestamp DESC LIMIT 5"
                cursor.execute(query, (symbol,))
                rows = cursor.fetchall()
                
                if rows:
                    for row in rows:
                        print(f"{row[0]} - O:{row[1]:.2f} H:{row[2]:.2f} L:{row[3]:.2f} C:{row[4]:.2f} V:{row[5]:.2f}")
        
        # Check subseries tables
        subseries_tables = ['subseries_categories', 'subseries_occurrences', 'category_transitions']
        for table in subseries_tables:
            if table in [t[0] for t in tables]:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"\n==== Table: {table} ====")
                print(f"Row count: {count}")
                
                if table == 'subseries_categories' and count > 0:
                    cursor.execute("SELECT category_id, pattern, count FROM subseries_categories ORDER BY count DESC LIMIT 5")
                    categories = cursor.fetchall()
                    print("\nTop categories:")
                    for cat in categories:
                        print(f"{cat[0]} ({cat[1]}) - Count: {cat[2]}")
                
                elif table == 'subseries_occurrences' and count > 0:
                    cursor.execute("SELECT DISTINCT symbol, timeframe, COUNT(*) FROM subseries_occurrences GROUP BY symbol, timeframe")
                    occurrences = cursor.fetchall()
                    print("\nOccurrences by symbol and timeframe:")
                    for occ in occurrences:
                        print(f"{occ[0]} ({occ[1]}) - Count: {occ[2]}")
                
                elif table == 'category_transitions' and count > 0:
                    cursor.execute("SELECT from_category, to_category, probability FROM category_transitions ORDER BY probability DESC LIMIT 5")
                    transitions = cursor.fetchall()
                    print("\nTop transitions:")
                    for trans in transitions:
                        print(f"{trans[0]} -> {trans[1]} - Probability: {trans[2]:.4f}")
        
        conn.close()
        
    except Exception as e:
        print(f"Error checking database: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    check_database()
