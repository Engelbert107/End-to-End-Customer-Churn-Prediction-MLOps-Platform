
"""Streamlit dashboard for churn prediction and model monitoring."""

import os

import pandas as pd
import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT = 15

st.set_page_config(
    page_title="Customer Churn MLOps Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("Customer Churn Prediction & Monitoring")
st.caption(
    "Predict customer churn and monitor model activity "
    "using aggregated production metrics."
)


def get_monitoring_metrics(days):
    """Fetch aggregate monitoring metrics from the FastAPI service."""
    response = requests.get(
        f"{API_URL}/monitoring/metrics",
        params={"days": days},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def show_prediction_form():
    """Render the customer form and call the prediction endpoint."""

    st.subheader("Predict customer churn")
    st.write("Enter customer details to generate a churn prediction.")

    with st.form("customer_prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox(
                "Senior citizen", [0, 1],
                format_func=lambda value: "Yes" if value == 1 else "No",
            )
            partner = st.selectbox("Has partner?", ["Yes", "No"])
            dependents = st.selectbox("Has dependents?", ["Yes", "No"])
            tenure = st.number_input(
                "Tenure (months)", min_value=0, max_value=1000, value=5
            )
            phone_service = st.selectbox(
                "Phone service", ["Yes", "No"]
            )
            multiple_lines = st.selectbox(
                "Multiple lines",
                ["No", "Yes", "No phone service"],
            )

        with col2:
            internet_service = st.selectbox(
                "Internet service",
                ["Fiber optic", "DSL", "No"],
            )
            online_security = st.selectbox(
                "Online security",
                ["No", "Yes", "No internet service"],
            )
            online_backup = st.selectbox(
                "Online backup",
                ["No", "Yes", "No internet service"],
            )
            device_protection = st.selectbox(
                "Device protection",
                ["No", "Yes", "No internet service"],
            )
            tech_support = st.selectbox(
                "Tech support",
                ["No", "Yes", "No internet service"],
            )
            streaming_tv = st.selectbox(
                "Streaming TV",
                ["Yes", "No", "No internet service"],
            )
            streaming_movies = st.selectbox(
                "Streaming movies",
                ["Yes", "No", "No internet service"],
            )

        with col3:
            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"],
            )
            paperless_billing = st.selectbox(
                "Paperless billing", ["Yes", "No"]
            )
            payment_method = st.selectbox(
                "Payment method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
            monthly_charges = st.number_input(
                "Monthly charges",
                min_value=0.0,
                value=85.0,
                step=1.0,
            )
            total_charges = st.number_input(
                "Total charges",
                min_value=0.0,
                value=400.0,
                step=10.0,
            )

        submitted = st.form_submit_button(
            "Predict churn",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        customer = {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        try:
            with st.spinner("Generating prediction..."):
                response = requests.post(
                    f"{API_URL}/predict",
                    json=customer,
                    timeout=REQUEST_TIMEOUT,
                )
                response.raise_for_status()
                result = response.json()

            st.session_state["latest_prediction"] = result

            # The API records monitoring metrics during each successful
            # prediction. Fetch fresh metrics when the monitoring tab opens.

        except requests.exceptions.RequestException as exc:
            st.error(f"Could not get a prediction from the API: {exc}")

    result = st.session_state.get("latest_prediction")

    if result:
        st.divider()
        st.subheader("Prediction result")

        prediction_col, probability_col, model_col = st.columns(3)

        with prediction_col:
            st.metric(
                "Predicted churn",
                result["churn_prediction"],
            )

        with probability_col:
            st.metric(
                "Churn probability",
                f'{result["churn_probability"]:.1%}',
            )

        with model_col:
            st.metric("Model", result.get("model", "Unknown"))

        st.progress(
            max(0.0, min(1.0, result["churn_probability"])),
            text="Estimated churn probability",
        )

        st.caption(
            "This is a model estimate, not a confirmed customer outcome."
        )


def show_monitoring_dashboard():
    """Render monitoring metrics and visualizations."""

    st.subheader("Model monitoring")
    st.write(
        "Review prediction activity, predicted outcomes, "
        "probability patterns, and aggregate input distributions."
    )

    period_col, refresh_col = st.columns([2, 1])

    with period_col:
        days = st.selectbox(
            "Monitoring period",
            options=[7, 30, 90, 180, 365],
            index=1,
            format_func=lambda value: f"Last {value} days",
            key="monitoring_period",
        )

    with refresh_col:
        st.write("")
        st.write("")
        refresh = st.button(
            "Refresh metrics",
            use_container_width=True,
        )

    # A button press causes Streamlit to rerun this script; metrics are
    # fetched on every render, so they remain up to date.
    try:
        with st.spinner("Loading monitoring metrics..."):
            metrics = get_monitoring_metrics(days)

    except requests.exceptions.RequestException as exc:
        st.error(
            f"Unable to load monitoring metrics from {API_URL}. "
            f"Check that the API is running. Details: {exc}"
        )
        return

    total = metrics.get("total_predictions", 0)
    churn_yes = metrics.get("churn_yes", 0)
    churn_no = metrics.get("churn_no", 0)
    average_probability = metrics.get("average_churn_probability")

    st.caption(
        f"Showing metrics for the last {metrics.get('period_days', days)} "
        "days. Use Refresh metrics after making new predictions."
    )

    # KPI cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    kpi1.metric("Total predictions", f"{total:,}")
    kpi2.metric("Predicted churn", f"{churn_yes:,}")
    kpi3.metric("Predicted no churn", f"{churn_no:,}")

    kpi4.metric(
        "Average churn probability",
        f"{average_probability:.1%}"
        if average_probability is not None
        else "N/A",
    )

    if total == 0:
        st.info(
            "No predictions have been recorded in this period yet. "
            "Make a prediction in the Prediction tab to populate "
            "the monitoring charts."
        )

    st.divider()

    # Daily prediction volume
    st.markdown("#### Prediction volume over time")
    daily = metrics.get("daily", [])

    if daily:
        daily_df = pd.DataFrame(daily)
        daily_df["date"] = pd.to_datetime(daily_df["date"])
        daily_df = daily_df.set_index("date")

        volume_columns = [
            column
            for column in ["total_predictions", "churn_yes", "churn_no"]
            if column in daily_df.columns
        ]

        if volume_columns:
            chart_df = daily_df[volume_columns].rename(
                columns={
                    "total_predictions": "Total predictions",
                    "churn_yes": "Predicted churn",
                    "churn_no": "Predicted no churn",
                }
            )
            st.line_chart(chart_df)
        else:
            st.info("Daily prediction counts are not available.")
    else:
        st.info("Daily activity will appear here after predictions are made.")

    left_chart, right_chart = st.columns(2)

    # Predicted churn vs retention
    with left_chart:
        st.markdown("#### Predicted outcomes")

        outcome_df = pd.DataFrame(
            {
                "Predictions": [churn_yes, churn_no],
            },
            index=["Predicted churn", "Predicted no churn"],
        )

        st.bar_chart(outcome_df)

        if total:
            st.caption(
                f"{churn_yes / total:.1%} of recorded predictions "
                "were classified as churn."
            )

    # Churn probability distribution
    with right_chart:
        st.markdown("#### Churn probability distribution")

        probability_bins = metrics.get("probability_distribution", {})
        bin_order = [
            "0-20%",
            "20-40%",
            "40-60%",
            "60-80%",
            "80-100%",
        ]

        probability_df = pd.DataFrame(
            {
                "Predictions": [
                    probability_bins.get(label, 0)
                    for label in bin_order
                ]
            },
            index=bin_order,
        )

        st.bar_chart(probability_df)
        st.caption(
            "These bars show how many predictions fell into each "
            "probability range, not actual churn rates."
        )

    st.divider()

    # Aggregated feature distributions
    st.markdown("#### Input feature distributions")
    st.write(
        "Explore the feature categories and numeric ranges submitted "
        "to the model during the selected period."
    )

    feature_buckets = metrics.get("input_feature_buckets", {})

    if feature_buckets:
        parsed_buckets = []

        for label, count in feature_buckets.items():
            feature, separator, bucket = label.partition(":")
            if separator:
                parsed_buckets.append(
                    {
                        "Feature": feature,
                        "Category or range": bucket,
                        "Count": count,
                    }
                )

        if parsed_buckets:
            feature_df = pd.DataFrame(parsed_buckets)
            available_features = sorted(feature_df["Feature"].unique())

            selected_feature = st.selectbox(
                "Choose a feature",
                options=available_features,
                key="monitoring_feature",
            )

            selected_df = feature_df[
                feature_df["Feature"] == selected_feature
            ].copy()

            selected_df = selected_df.set_index("Category or range")
            selected_df = selected_df[["Count"]].sort_values(
                "Count",
                ascending=False,
            )

            st.bar_chart(selected_df)
            st.dataframe(
                selected_df.reset_index(),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No usable feature buckets were returned.")
    else:
        st.info(
            "Feature distributions will appear after customer predictions "
            "have been recorded."
        )

    st.divider()
    st.markdown("#### Daily metrics table")

    if daily:
        display_df = pd.DataFrame(daily).rename(
            columns={
                "date": "Date",
                "total_predictions": "Total predictions",
                "churn_yes": "Predicted churn",
                "churn_no": "Predicted no churn",
                "average_churn_probability": "Average churn probability",
            }
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

    st.caption(metrics.get("note", ""))
    
    st.divider()
    st.markdown("#### Training data drift detection")

    try:
        drift_response = requests.get(
            f"{API_URL}/monitoring/drift",
            params={"days": days},
            timeout=REQUEST_TIMEOUT,
        )
        drift_response.raise_for_status()
        drift = drift_response.json()

        if drift.get("status") == "baseline_missing":
            st.warning(drift.get("message", "Drift baseline is missing."))
        else:
            st.caption(
                f"Baseline: {drift.get('baseline_source', 'unknown')} | "
                f"Training rows: {drift.get('baseline_rows', 'unknown')} | "
                f"Recent predictions: {drift.get('total_predictions', 0)}"
            )

            high, moderate, low = st.columns(3)
            high.metric(
                "High drift",
                drift.get("high_drift_features", 0),
            )
            moderate.metric(
                "Moderate drift",
                drift.get("moderate_drift_features", 0),
            )

            feature_results = drift.get("features", [])
            low_count = sum(
                item.get("status") == "low"
                for item in feature_results
            )
            low.metric("Low drift", low_count)

            drift_df = pd.DataFrame(feature_results)

            if not drift_df.empty:
                display_df = drift_df.rename(
                    columns={
                        "feature": "Feature",
                        "psi": "PSI score",
                        "status": "Drift status",
                        "baseline_count": "Baseline observations",
                        "observed_count": "Recent observations",
                    }
                )

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True,
                )

                scored_df = drift_df.dropna(subset=["psi"]).copy()

                if not scored_df.empty:
                    scored_df = scored_df.set_index("feature")
                    st.markdown("##### Population Stability Index (PSI) scores by feature")
                    st.bar_chart(scored_df[["psi"]])

            if drift.get("total_predictions", 0) < 100:
                st.info(
                    "The recent sample is small. Treat PSI scores as "
                    "preliminary and avoid drawing strong conclusions."
                )

            st.caption(
                "Rule-of-thumb interpretation: PSI below 0.10 is low, "
                "0.10–0.20 is moderate, and 0.20 or higher is high. "
                "These are heuristics, not universal thresholds."
            )

    except requests.exceptions.RequestException as exc:
        st.error(f"Could not load drift results: {exc}")


    st.warning(
        "Monitoring limitation: feature distributions are aggregate "
        "summaries, not a confirmed data-drift test. To detect drift, "
        "compare them with a saved training-data baseline. To measure "
        "model accuracy over time, collect actual customer outcomes "
        "through an appropriate, privacy-conscious process."
    )


# Keep prediction and monitoring in separate tabs.
prediction_tab, monitoring_tab = st.tabs(
    ["Customer prediction", "Model monitoring"]
)

with prediction_tab:
    show_prediction_form()

with monitoring_tab:
    show_monitoring_dashboard()
