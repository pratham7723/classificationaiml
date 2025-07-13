from flask import Flask, render_template, redirect, url_for
import os
from classification_logic import run_classification_demo

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run')
def run():
    # Run the classification demo and get results
    results = run_classification_demo()
    return render_template('results.html', **results)

if __name__ == '__main__':
    # Ensure static directory exists for plots
    os.makedirs('static', exist_ok=True)
    app.run(debug=True) 