# Phase 4: Flask-HTML Dashboard (flask_dashboard.py)

from flask import Flask, render_template, request
import joblib
import numpy as np
import requests

app = Flask(__name__)

# API URL (Ensure it points to the API's endpoint)
API_URL = 'http://127.0.0.1:5000/predict'

# Dashboard Route
@app.route('/', methods=['GET', 'POST'])
def dashboard():
    prediction = None
    probability = None

    if request.method == 'POST':
        try:
            # Collect input data from form
            input_data = {
                f"V{i}": float(request.form.get(f'V{i}', 0)) for i in range(1, 29)
            }
            input_data['scaled_amount'] = float(request.form.get('scaled_amount', 0))
            input_data['scaled_time'] = float(request.form.get('scaled_time', 0))

            # Send data to the API
            response = requests.post(API_URL, json=input_data)

            if response.status_code == 200:
                result = response.json()
                prediction = result.get('fraud_prediction')
                probability = result.get('fraud_probability')
            else:
                print(f"API Error: {response.text}")

        except Exception as e:
            print(f"Error during prediction: {e}")

    return render_template('index.html', prediction=prediction, probability=probability)

if __name__ == '__main__':
    app.run(port=5001, debug=True)  # Dashboard runs on port 5001
