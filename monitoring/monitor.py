from pathlib import Path

import mlflow.sklearn
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from monitoring.drift_detection import (
    detect_categorical_drift,
    detect_numerical_drift,
)


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REFERENCE_PATH = PROJECT_ROOT / "data" / "reference_data.csv"

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "customer_churn_model_v1"
)


# =========================================================
# FEATURES TO MONITOR
# =========================================================

NUMERICAL_COLUMNS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]


CATEGORICAL_COLUMNS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


# =========================================================
# DATA DRIFT MONITORING
# =========================================================

def run_drift_monitoring(
    current_data: pd.DataFrame,
) -> dict:
    """
    Compare current production data against the
    reference dataset and detect feature drift.
    """

    # Load reference data.
    reference_data = pd.read_csv(
        REFERENCE_PATH
    )

    # Numerical feature drift.
    numerical_results = detect_numerical_drift(
        reference_data,
        current_data,
        NUMERICAL_COLUMNS,
    )

    # Categorical feature drift.
    categorical_results = detect_categorical_drift(
        reference_data,
        current_data,
        CATEGORICAL_COLUMNS,
    )

    # Combine all feature-level results.
    all_results = pd.concat(
        [
            numerical_results,
            categorical_results,
        ],
        ignore_index=True,
    )

    # Keep only features where drift was detected.
    drifted_results = all_results[
        all_results["drift_detected"]
    ]

    drift_count = int(
        len(drifted_results)
    )

    return {
        "total_features_checked": int(
            len(all_results)
        ),
        "drifted_features": drift_count,
        "drift_detected": drift_count > 0,
        "drifted_feature_names": (
            drifted_results["feature"].tolist()
        ),
        "results": all_results,
    }


# =========================================================
# DATA DRIFT REPORT
# =========================================================

def create_monitoring_report(
    current_data_path: Path,
    output_path: Path,
) -> dict:
    """
    Run feature drift monitoring and save
    a summary report as JSON.
    """

    current_data = pd.read_csv(
        current_data_path
    )

    result = run_drift_monitoring(
        current_data
    )

    report = {
        "monitoring_status": (
            "DRIFT_DETECTED"
            if result["drift_detected"]
            else "NO_DRIFT"
        ),
        "total_features_checked": (
            result["total_features_checked"]
        ),
        "drifted_features": (
            result["drifted_features"]
        ),
        "drifted_feature_names": (
            result["drifted_feature_names"]
        ),
    }

    # Ensure output directory exists.
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save JSON report.
    pd.DataFrame([report]).to_json(
        output_path,
        orient="records",
        indent=2,
    )

    return report


# =========================================================
# MODEL PERFORMANCE MONITORING
# =========================================================

def evaluate_model_performance(
    labeled_data: pd.DataFrame,
) -> dict:
    """
    Evaluate the deployed model against
    labeled production data.
    """

    data = labeled_data.copy()

    # Separate features from target.
    X_current = data.drop(
        columns=[
            "Churn",
            "customerID",
        ]
    )

    # Convert target to 0/1.
    y_current = data["Churn"].map(
        {
            "No": 0,
            "Yes": 1,
        }
    )

    # Load the exact deployed model artifact.
    production_model = (
        mlflow.sklearn.load_model(
            str(MODEL_PATH)
        )
    )

    # Generate predictions.
    predictions = production_model.predict(
        X_current
    )

    # Generate probabilities.
    probabilities = (
        production_model
        .predict_proba(X_current)[:, 1]
    )

    # Calculate performance metrics.
    metrics = {
        "accuracy": float(
            accuracy_score(
                y_current,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                y_current,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_current,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_current,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_current,
                probabilities,
            )
        ),
    }

    return metrics


# =========================================================
# MODEL PERFORMANCE REPORT
# =========================================================

def create_performance_report(
    labeled_data_path: Path,
    output_path: Path,
) -> dict:
    """
    Evaluate model performance on labeled
    production data and save a JSON report.
    """

    labeled_data = pd.read_csv(
        labeled_data_path
    )

    metrics = evaluate_model_performance(
        labeled_data
    )

    report = {
        "model": "CustomerChurnModel",
        "version": "1",
        "metrics": metrics,
    }

    # Ensure output directory exists.
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save performance report.
    pd.DataFrame([report]).to_json(
        output_path,
        orient="records",
        indent=2,
    )

    return report