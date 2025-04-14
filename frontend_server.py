from flask import Flask, render_template, send_from_directory
from flask_cors import CORS
import os

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)  # Abilita CORS per tutte le rotte

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

if __name__ == "__main__":
    # Avvia il server Flask
    app.run(debug=True, port=5000) 