import pandas as pd

from monitoring.drift_detection import (
    detect_numerical_drift,
)


def test_numerical_drift_detects_shift():
    reference_data = pd.DataFrame(
        {
            "value": [1, 2, 3, 4, 5] * 20
        }
    )

    current_data = pd.DataFrame(
        {
            "value": [20, 21, 22, 23, 24] * 20
        }
    )

    result = detect_numerical_drift(
        reference_data=reference_data,
        current_data=current_data,
        columns=["value"],
    )

    assert len(result) == 1
    assert result.iloc[0]["drift_detected"] == True


def test_numerical_drift_does_not_flag_identical_data():
    reference_data = pd.DataFrame(
        {
            "value": [1, 2, 3, 4, 5] * 20
        }
    )

    current_data = reference_data.copy()

    result = detect_numerical_drift(
        reference_data=reference_data,
        current_data=current_data,
        columns=["value"],
    )

    assert len(result) == 1
    assert result.iloc[0]["drift_detected"] == False