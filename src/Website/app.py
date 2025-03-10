from flask import Flask, render_template, jsonify
import random
import os
from datetime import datetime, timedelta

app = Flask(__name__)

# Mock data for charts - this would be replaced with real data in production
@app.route('/api/data')
def get_data():
    # Generate some random data for the charts
    timestamps = [(datetime.now() - timedelta(minutes=i)).strftime('%H:%M:%S') for i in range(30, 0, -1)]
    accelerometer_data = [random.uniform(-2, 2) for _ in range(30)]
    gyroscope_data = [random.uniform(-180, 180) for _ in range(30)]
    
    return jsonify({
        'timestamps': timestamps,
        'accelerometer': accelerometer_data,
        'gyroscope': gyroscope_data
    })

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/data')
def data_view():
    return render_template('data.html')

if __name__ == '__main__':
    app.run(debug=True)