import sqlite3

# Connect to the database
conn = sqlite3.connect('crypto_data.db')
cursor = conn.cursor()

# Get the date range
cursor.execute('SELECT MIN(timestamp), MAX(timestamp) FROM market_data_1h')
date_range = cursor.fetchone()
print(f"Date range: {date_range[0]} to {date_range[1]}")

# Get available symbols
cursor.execute('SELECT DISTINCT symbol FROM market_data_1h')
symbols = cursor.fetchall()
print("\nAvailable symbols:")
for symbol in symbols:
    print(f"- {symbol[0]}")

# Get count of records per symbol
cursor.execute('''
    SELECT symbol, COUNT(*) as count 
    FROM market_data_1h 
    GROUP BY symbol 
    ORDER BY count DESC
''')
counts = cursor.fetchall()
print("\nRecord counts per symbol:")
for symbol, count in counts:
    print(f"- {symbol}: {count} records")

# Close the connection
conn.close()
