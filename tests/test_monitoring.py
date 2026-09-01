import pandas as pd

from monitoring import monitor


DATASET_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"


def load_clean_dataset():
    data = pd.read_csv(DATASET_PATH)

    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"],
        errors="coerce",
    )

    data = data.dropna(
        subset=["TotalCharges"]
    )

    return data


def test_run_drift_monitoring(tmp_path, monkeypatch):
    data = load_clean_dataset()

    reference_data = data.drop(
        columns=["customerID", "Churn"]
    ).head(500).copy()

    current_data = data.drop(
        columns=["customerID", "Churn"]
    ).head(500).copy()

    reference_path = (
        tmp_path / "reference_data.csv"
    )

    reference_data.to_csv(
        reference_path,
        index=False,
    )

    monkeypatch.setattr(
        monitor,
        "REFERENCE_PATH",
        reference_path,
    )

    result = monitor.run_drift_monitoring(
        current_data
    )

    assert result["total_features_checked"] == 19
    assert "drift_detected" in result
    assert "drifted_feature_names" in result


def test_evaluate_model_performance():
    data = load_clean_dataset()

    labeled_data = data.sample(
        n=100,
        random_state=42,
    ).copy()

    result = monitor.evaluate_model_performance(
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