from pathlib import Path

import pandas as pd

from monitoring.monitor import (
    run_drift_monitoring,
    evaluate_model_performance,
)


def test_run_drift_monitoring():

    current_data = pd.read_csv(
        "data/production_data.csv"
    )

    result = run_drift_monitoring(
        current_data
    )

    assert result["total_features_checked"] == 19
    assert "drift_detected" in result
    assert "drifted_feature_names" in result


def test_evaluate_model_performance():

    labeled_data = pd.read_csv(
        "data/production_labeled_data.csv"
    )

    result = evaluate_model_performance(
        labeled_data
    )

    assert "accuracy" in result
    assert "precision" in result
    assert "recall" in result
    assert "f1" in result
    assert "roc_auc" in result

    assert 0 <= result["accuracy"] <= 1
    assert 0 <= result["precision"] <= 1
    assert 0 <= result["recall"] <= 1
    assert 0 <= result["f1"] <= 1
    assert 0 <= result["roc_auc"] <= 1