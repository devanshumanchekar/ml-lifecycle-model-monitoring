from pathlib import Path
from typing import Literal

import pandas as pd
import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]

    tenure: int = Field(ge=0)

    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]

    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]

    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]

    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]

    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

# Project root: ML-Lifecycle-Model-Monitoring/
# Project root: ML-Lifecycle-Model-Monitoring/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_NAME = "CustomerChurnModel"
MODEL_VERSION = "1"

MODEL_PATH = PROJECT_ROOT / "models" / "customer_churn_model_v1"

model = mlflow.sklearn.load_model(
    str(MODEL_PATH)
)


app = FastAPI(
    title="Customer Churn ML API",
    description="Production-oriented API for customer churn prediction.",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "message": "Customer Churn ML API is running"
    }



@app.post("/predict")
def predict(customer: CustomerInput):
    try:
        input_data = customer.model_dump()
        input_df = pd.DataFrame([input_data])

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0, 1]

        return {
            "prediction": "Yes" if prediction == 1 else "No",
            "churn_probability": round(float(probability), 4),
            "model": MODEL_NAME,
            "version": MODEL_VERSION,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}"
        )
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "version": MODEL_VERSION,
    }
    