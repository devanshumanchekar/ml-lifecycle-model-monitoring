from pathlib import Path

import pandas as pd

from monitor import (
    create_monitoring_report,
    create_performance_report,
)


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PRODUCTION_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "production_data.csv"
)

LABELED_PRODUCTION_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "production_labeled_data.csv"
)

DRIFT_REPORT_PATH = (
    PROJECT_ROOT
    / "monitoring"
    / "drift_report.json"
)

PERFORMANCE_REPORT_PATH = (
    PROJECT_ROOT
    / "monitoring"
    / "performance_report.json"
)


def run_monitoring_pipeline() -> None:
    """
    Run the complete production monitoring workflow.

    1. Detect feature drift.
    2. Save drift report.
    3. Evaluate model performance.
    4. Save performance report.
    5. Print an overall monitoring summary.
    """

    print("=" * 60)
    print("CUSTOMER CHURN ML MONITORING")
    print("=" * 60)

    # -----------------------------------------------------
    # Validate required input files.
    # -----------------------------------------------------

    if not PRODUCTION_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Production data not found: "
            f"{PRODUCTION_DATA_PATH}"
        )

    if not LABELED_PRODUCTION_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Labeled production data not found: "
            f"{LABELED_PRODUCTION_DATA_PATH}"
        )

    # -----------------------------------------------------
    # Data drift monitoring.
    # -----------------------------------------------------

    print("\n[1/2] Running data drift monitoring...")

    drift_report = create_monitoring_report(
        current_data_path=PRODUCTION_DATA_PATH,
        output_path=DRIFT_REPORT_PATH,
    )

    print(
        f"Features checked: "
        f"{drift_report['total_features_checked']}"
    )

    print(
        f"Drifted features: "
        f"{drift_report['drifted_features']}"
    )

    print(
        f"Drift status: "
        f"{drift_report['monitoring_status']}"
    )

    # -----------------------------------------------------
    # Model performance monitoring.
    # -----------------------------------------------------

    print("\n[2/2] Running model performance monitoring...")

    performance_report = create_performance_report(
        labeled_data_path=LABELED_PRODUCTION_DATA_PATH,
        output_path=PERFORMANCE_REPORT_PATH,
    )

    metrics = performance_report["metrics"]

    print(
        f"Accuracy:  {metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: {metrics['precision']:.4f}"
    )

    print(
        f"Recall:    {metrics['recall']:.4f}"
    )

    print(
        f"F1 Score:  {metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC:   {metrics['roc_auc']:.4f}"
    )

    # -----------------------------------------------------
    # Final status.
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("MONITORING PIPELINE COMPLETE")
    print("=" * 60)

    print(
        f"Drift report: "
        f"{DRIFT_REPORT_PATH}"
    )

    print(
        f"Performance report: "
        f"{PERFORMANCE_REPORT_PATH}"
    )


if __name__ == "__main__":
    run_monitoring_pipeline()