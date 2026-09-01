# Customer Churn ML Lifecycle & Monitoring

An end-to-end machine learning engineering project for predicting telecom customer churn and monitoring the model after deployment.

## What I Built

- Data cleaning and exploratory analysis
- Feature preprocessing pipeline
- Logistic Regression, Random Forest and XGBoost comparison
- Stratified 5-fold cross-validation
- MLflow experiment tracking
- MLflow Model Registry with versioned model
- FastAPI prediction API
- API input validation and versioning
- Dockerized ML service with health checks
- Data drift detection using KS and Chi-Square tests
- Model performance monitoring
- Streamlit monitoring dashboard
- Automated testing with pytest
- GitHub Actions CI

## Model Selection

Three models were evaluated:

| Model | CV ROC-AUC |
|---|---:|
| Logistic Regression | 0.8462 |
| XGBoost | 0.8437 |
| Random Forest | 0.8203 |

**Selected model:** Logistic Regression

**Test ROC-AUC:** 0.8421

## Architecture

```text
Customer Data
      ↓
Preprocessing
      ↓
Model Training
      ↓
MLflow Tracking
      ↓
Model Registry
      ↓
FastAPI
      ↓
Docker
      ↓
Production Monitoring
      ├── Data Drift
      └── Model Performance
MLflow

Registered model:

CustomerChurnModel
Version 1
API
Health
GET /api/v1/health
Prediction
POST /api/v1/predict
Swagger
http://127.0.0.1:8000/docs
Monitoring

The monitoring system checks:

Numerical feature drift
Categorical feature drift
Accuracy
Precision
Recall
F1 Score
ROC-AUC

Run monitoring:

python monitoring\run_monitoring.py
Dashboard

Run locally:

python -m streamlit run monitoring/dashboard.py

Dashboard:

http://localhost:8501
Docker

Build:

docker build -t customer-churn-api .

Run:

docker run -d --name customer-churn-api-container -p 8000:8000 customer-churn-api
Testing

Run all tests:

python -m pytest -v
CI

GitHub Actions automatically runs the test suite on pushes and pull requests.

Project Structure
api/            FastAPI service
data/           Dataset
models/         Versioned production model
monitoring/     Drift and performance monitoring
notebooks/      ML development notebook
tests/          Automated tests
.github/        CI workflow
Dockerfile      Container configuration
Cost

Built and tested locally using free/open-source tooling.

No paid cloud deployment required.