from typing import List

import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp


def detect_numerical_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    columns: List[str],
    significance_level: float = 0.05,
) -> pd.DataFrame:
    results = []

    for column in columns:
        statistic, p_value = ks_2samp(
            reference_data[column].dropna(),
            current_data[column].dropna(),
        )

        results.append(
            {
                "feature": column,
                "test": "KS",
                "statistic": statistic,
                "p_value": p_value,
                "drift_detected": p_value < significance_level,
            }
        )

    return pd.DataFrame(results)


def detect_categorical_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    columns: List[str],
    significance_level: float = 0.05,
) -> pd.DataFrame:
    results = []

    for column in columns:
        reference_counts = reference_data[column].value_counts()
        current_counts = current_data[column].value_counts()

        categories = reference_counts.index.union(current_counts.index)

        reference = reference_counts.reindex(
            categories,
            fill_value=0,
        )

        current = current_counts.reindex(
            categories,
            fill_value=0,
        )

        contingency_table = pd.DataFrame(
            [reference.values, current.values],
            index=["reference", "current"],
            columns=categories,
        )

        statistic, p_value, _, _ = chi2_contingency(
            contingency_table
        )

        results.append(
            {
                "feature": column,
                "test": "Chi-Square",
                "statistic": statistic,
                "p_value": p_value,
                "drift_detected": p_value < significance_level,
            }
        )

    return pd.DataFrame(results)