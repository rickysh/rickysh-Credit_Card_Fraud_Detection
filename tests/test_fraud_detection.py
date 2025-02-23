# tests/test_fraud_detection.py

import pytest
import requests
from flask import Flask

# Define endpoints
API_URL = "http://127.0.0.1:5000/predict"
DASHBOARD_URL = "http://127.0.0.1:5001/"

# Sample valid input data
valid_input = {
    "V1": -1.23, "V2": 2.45, "V3": 1.56, "V4": -0.65, "V5": 0.45, "V6": -0.12,
    "V7": 0.56, "V8": -1.32, "V9": 0.23, "V10": -0.65, "V11": 0.34, "V12": -0.98,
    "V13": 0.45, "V14": -1.23, "V15": 2.34, "V16": 0.12, "V17": -0.45, "V18": 1.45,
    "V19": -0.78, "V20": 0.89, "V21": -1.12, "V22": 0.67, "V23": -0.34, "V24": 1.23,
    "V25": -0.45, "V26": 0.98, "V27": -1.11, "V28": 0.67, "scaled_amount": 123.45, "scaled_time": 456.78
}

# Test API - Happy Path
def test_api_predict_valid():
    response = requests.post(API_URL, json=valid_input)
    assert response.status_code == 200

    data = response.json()
    assert "fraud_prediction" in data
    assert "fraud_probability" in data

# Test API - Invalid Input
def test_api_invalid_input():
    invalid_input = valid_input.copy()
    invalid_input["V1"] = "invalid"

    response = requests.post(API_URL, json=invalid_input)
    assert response.status_code == 400

# Test Dashboard - Load Page
def test_dashboard_load():
    response = requests.get(DASHBOARD_URL)
    assert response.status_code == 200
    assert "Fraud Detection System" in response.text

# Test Dashboard - Form Submission
def test_dashboard_predict():
    form_data = {f"V{i}": str(valid_input[f"V{i}"]) for i in range(1, 29)}
    form_data["scaled_amount"] = str(valid_input["scaled_amount"])
    form_data["scaled_time"] = str(valid_input["scaled_time"])

    response = requests.post(DASHBOARD_URL, data=form_data)
    assert response.status_code == 200
    assert "Prediction Result" in response.text

# Test API - Missing Data
def test_api_missing_data():
    incomplete_input = valid_input.copy()
    incomplete_input.pop("V1")

    response = requests.post(API_URL, json=incomplete_input)
    assert response.status_code == 400

# Run tests via: pytest tests/
if __name__ == "__main__":
    pytest.main(["-v", "--tb=short", "--disable-warnings", "--html=report.html", "tests"])
