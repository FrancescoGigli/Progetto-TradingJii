import sqlite3

# Connect to the database
conn = sqlite3.connect('crypto_data.db')
cursor = conn.cursor()

# Get all table names
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

print("Tables in crypto_data.db:")
for table in tables:
    table_name = table[0]
    print(f"\n- {table_name}")
    
    # Get column names for each table
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    print("  Columns:")
    for col in columns:
        print(f"    - {col[1]} ({col[2]})")
    
    # Get row count
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    print(f"  Row count: {count}")
    
    # If it's a market data table, get date range
    if 'market_data' in table_name:
        try:
            cursor.execute(f"SELECT MIN(timestamp), MAX(timestamp) FROM {table_name}")
            date_range = cursor.fetchone()
            if date_range[0] and date_range[1]:
                print(f"  Date range: {date_range[0]} to {date_range[1]}")
        except:
            pass

# Close the connection
conn.close()
