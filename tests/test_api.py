
"""Automated tests for the customer churn prediction API."""

import pytest
from fastapi.testclient import TestClient

from src.api import app


SAMPLE_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.7,
    "TotalCharges": 171.4,
}


@pytest.fixture
def client():
    # Entering the context starts the API and loads the trained model.
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "Customer Churn Prediction API"


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "model_loaded": True,
    }


def test_valid_prediction(client):
    response = client.post("/predict", json=SAMPLE_CUSTOMER)

    assert response.status_code == 200

    result = response.json()
    assert result["churn_prediction"] in ["Yes", "No"]
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert 0.0 <= result["no_churn_probability"] <= 1.0
    assert result["model"] == "best_model.joblib"


def test_probabilities_sum_to_one(client):
    response = client.post("/predict", json=SAMPLE_CUSTOMER)

    assert response.status_code == 200

    result = response.json()
    total = (
        result["churn_probability"]
        + result["no_churn_probability"]
    )

    assert total == pytest.approx(1.0, abs=0.0002)


def test_invalid_gender_is_rejected(client):
    customer = SAMPLE_CUSTOMER.copy()
    customer["gender"] = "Unknown"

    response = client.post("/predict", json=customer)

    assert response.status_code == 422


def test_missing_required_field_is_rejected(client):
    customer = SAMPLE_CUSTOMER.copy()
    del customer["tenure"]

    response = client.post("/predict", json=customer)

    assert response.status_code == 422


def test_negative_monthly_charges_are_rejected(client):
    customer = SAMPLE_CUSTOMER.copy()
    customer["MonthlyCharges"] = -10

    response = client.post("/predict", json=customer)

    assert response.status_code == 422


def test_unexpected_field_is_rejected(client):
    customer = SAMPLE_CUSTOMER.copy()
    customer["unexpected_field"] = "test"

    response = client.post("/predict", json=customer)

    assert response.status_code == 422
