# frontend_server.py
from flask import Flask, send_from_directory
from flask_cors import CORS
import os

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)  # Abilita CORS se necessario

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

if __name__ == "__main__":
    # Avvia il server Flask in modalità produzione senza debug
    app.run(host='0.0.0.0', port=5000)
