# Phase 3: Flask API Development (flask_api.py)

from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('fraud_detection_model.pkl')

# API Home Route
@app.route('/')
def home():
    return "Fraud Detection API is running!"

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON input
        data = request.get_json()

        # Required features (V1-V28, scaled_amount, scaled_time)
        features = ['V' + str(i) for i in range(1, 29)] + ['scaled_amount', 'scaled_time']
        input_data = [data.get(feature, 0) for feature in features]

        # Convert input to numpy array (2D for model)
        input_array = np.array(input_data).reshape(1, -1)

        # Perform prediction
        prediction = model.predict(input_array)[0]
        probability = model.predict_proba(input_array)[0][1]

        return jsonify({
            'fraud_prediction': int(prediction),
            'fraud_probability': float(probability)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)  # API runs on port 5000
