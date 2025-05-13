import sqlite3
import os

db_file = 'crypto_data.db'

# Check if database file exists
if not os.path.exists(db_file):
    print(f"Database file {db_file} does not exist.")
    print("You need to run real_time.py first to create and populate the database.")
    exit(1)

try:
    # Connect to the database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # List all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    if tables:
        print(f"Found {len(tables)} tables in the database:")
        for table in tables:
            print(f"- {table[0]}")
            
        # Check at least one of the data tables
        timeframes = ['5m', '15m']
        for tf in timeframes:
            try:
                # Check if data table exists
                cursor.execute(f"SELECT COUNT(*) FROM data_{tf}")
                count = cursor.fetchone()[0]
                print(f"Data table for {tf} timeframe has {count} records")
                
                # Get unique symbols
                cursor.execute(f"SELECT DISTINCT symbol FROM data_{tf}")
                symbols = cursor.fetchall()
                print(f"Found {len(symbols)} different cryptocurrencies in {tf} timeframe")
                
                # Sample of symbols
                if symbols:
                    sample = [s[0] for s in symbols[:5]]
                    print(f"Sample symbols: {', '.join(sample)}")
                
            except sqlite3.OperationalError:
                print(f"Table data_{tf} does not exist or has errors")
    else:
        print("No tables found in the database.")
        print("The database exists but is empty. Run real_time.py to populate it.")
    
    conn.close()
    
except sqlite3.Error as e:
    print(f"SQLite error: {e}")
