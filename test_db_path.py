import os
import sys

# Add backend path
sys.path.append('trading-backtest-app/backend')

# Simulate what happens in the backend
current_dir = os.path.dirname(os.path.abspath('trading-backtest-app/backend/app.py'))
project_root = os.path.dirname(os.path.dirname(current_dir))
db_path = os.path.join(project_root, 'crypto_data.db')

print(f'Current dir: {current_dir}')
print(f'Project root: {project_root}')
print(f'DB path: {db_path}')
print(f'DB exists: {os.path.exists(db_path)}')

# Check the actual db location
actual_db_path = 'crypto_data.db'
print(f'\nActual DB path: {actual_db_path}')
print(f'Actual DB exists: {os.path.exists(actual_db_path)}')
print(f'Absolute path: {os.path.abspath(actual_db_path)}')
