from pathlib import Path
import json

import pandas as pd
import streamlit as st


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="ML Model Monitoring",
    page_icon="📊",
    layout="wide",
)


st.title("Customer Churn — ML Monitoring Dashboard")

st.caption(
    "Production monitoring for CustomerChurnModel Version 1"
)


# =========================================================
# LOAD DRIFT REPORT
# =========================================================

if not DRIFT_REPORT_PATH.exists():
    st.error(
        "Drift report not found. "
        "Run the drift monitoring pipeline first."
    )
    st.stop()


drift_data = pd.read_json(
    DRIFT_REPORT_PATH
)

drift_report = drift_data.iloc[0]


monitoring_status = drift_report[
    "monitoring_status"
]

total_features = int(
    drift_report["total_features_checked"]
)

drifted_features = int(
    drift_report["drifted_features"]
)

drifted_names = drift_report[
    "drifted_feature_names"
]


# Handle JSON string representation.
if isinstance(drifted_names, str):

    try:
        drifted_names = json.loads(
            drifted_names
        )

    except json.JSONDecodeError:
        drifted_names = [drifted_names]


# =========================================================
# DRIFT STATUS
# =========================================================

if monitoring_status == "DRIFT_DETECTED":

    st.error(
        "⚠️ Data Drift Detected"
    )

else:

    st.success(
        "✅ No Significant Data Drift Detected"
    )


# =========================================================
# TOP SUMMARY
# =========================================================

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Features Checked",
        total_features,
    )


with col2:

    st.metric(
        "Drifted Features",
        drifted_features,
    )


with col3:

    st.metric(
        "Model Version",
        "1",
    )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.subheader("Model Performance")


if PERFORMANCE_REPORT_PATH.exists():

    performance_data = pd.read_json(
        PERFORMANCE_REPORT_PATH
    )

    performance_report = (
        performance_data.iloc[0]
    )

    performance_metrics = (
        performance_report["metrics"]
    )

    if isinstance(
        performance_metrics,
        str,
    ):

        try:
            performance_metrics = json.loads(
                performance_metrics
            )

        except json.JSONDecodeError:
            performance_metrics = {}

    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(
        5
    )

    with metric_col1:

        st.metric(
            "Accuracy",
            f"{performance_metrics['accuracy']:.2%}",
        )

    with metric_col2:

        st.metric(
            "Precision",
            f"{performance_metrics['precision']:.2%}",
        )

    with metric_col3:

        st.metric(
            "Recall",
            f"{performance_metrics['recall']:.2%}",
        )

    with metric_col4:

        st.metric(
            "F1 Score",
            f"{performance_metrics['f1']:.2%}",
        )

    with metric_col5:

        st.metric(
            "ROC-AUC",
            f"{performance_metrics['roc_auc']:.2%}",
        )

else:

    st.warning(
        "Performance report not found."
    )


# =========================================================
# DRIFTED FEATURES
# =========================================================

st.subheader("Drifted Features")


if drifted_names:

    drift_table = pd.DataFrame(
        {
            "Feature": drifted_names
        }
    )

    st.dataframe(
        drift_table,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        "No features are currently flagged."
    )


# =========================================================
# MONITORING SUMMARY
# =========================================================

st.subheader("Monitoring Summary")


summary_table = pd.DataFrame(
    {
        "Metric": [
            "Monitoring Status",
            "Total Features Checked",
            "Drifted Features",
            "Model",
            "Model Version",
        ],
        "Value": [
            monitoring_status,
            total_features,
            drifted_features,
            "CustomerChurnModel",
            "1",
        ],
    }
)


st.dataframe(
    summary_table,
    use_container_width=True,
    hide_index=True,
)