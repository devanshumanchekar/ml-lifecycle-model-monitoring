from fastapi.testclient import TestClient

from api.app import app


client = TestClient(app)


VALID_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
}


def test_health_endpoint():
    response = client.get("/api/v1/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model"] == "CustomerChurnModel"
    assert data["version"] == "1"


def test_prediction_endpoint():
    response = client.post(
        "/api/v1/predict",
        json=VALID_CUSTOMER,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in ["Yes", "No"]
    assert 0 <= data["churn_probability"] <= 1
    assert data["model"] == "CustomerChurnModel"
    assert data["version"] == "1"


def test_invalid_customer_input():
    invalid_customer = VALID_CUSTOMER.copy()

    invalid_customer["SeniorCitizen"] = 2

    response = client.post(
        "/api/v1/predict",
        json=invalid_customer,
    )

    assert response.status_code == 422