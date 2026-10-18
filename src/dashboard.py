
"""Streamlit dashboard for customer churn prediction."""

import os
import requests
import streamlit as st



API_URL = os.getenv("API_URL", "http://127.0.0.1:8000") + "/predict"

# API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Customer Churn Prediction")
st.caption("End-to-end ML platform | Powered by your trained model")

st.info(
    "Enter a customer's account and service details to estimate "
    "their likelihood of leaving the company."
)

with st.form("customer_form"):
    st.subheader("Customer information")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior citizen", [0, 1])
        partner = st.selectbox("Has partner?", ["Yes", "No"])
        dependents = st.selectbox("Has dependents?", ["Yes", "No"])
        tenure = st.number_input(
            "Tenure (months)", min_value=0, max_value=1000, value=2
        )
        phone = st.selectbox("Phone service", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple lines",
            ["No", "Yes", "No phone service"],
        )

    with col2:
        internet = st.selectbox(
            "Internet service", ["DSL", "Fiber optic", "No"]
        )
        online_security = st.selectbox(
            "Online security", ["No", "Yes", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online backup", ["No", "Yes", "No internet service"]
        )
        device_protection = st.selectbox(
            "Device protection", ["No", "Yes", "No internet service"]
        )
        tech_support = st.selectbox(
            "Tech support", ["No", "Yes", "No internet service"]
        )
        streaming_tv = st.selectbox(
            "Streaming TV", ["No", "Yes", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming movies", ["No", "Yes", "No internet service"]
        )

    with col3:
        contract = st.selectbox(
            "Contract", ["Month-to-month", "One year", "Two year"]
        )
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        payment = st.selectbox(
            "Payment method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        monthly_charges = st.number_input(
            "Monthly charges ($)", min_value=0.0, value=85.70, step=1.0
        )
        total_charges = st.number_input(
            "Total charges ($)", min_value=0.0, value=171.40, step=10.0
        )

    submitted = st.form_submit_button(
        "Predict churn", type="primary", use_container_width=True
    )

if submitted:
    customer = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    try:
        with st.spinner("Getting prediction..."):
            response = requests.post(
                API_URL, json=customer, timeout=15
            )

        if response.ok:
            result = response.json()
            churn = result["churn_prediction"]
            probability = result["churn_probability"]

            st.divider()
            st.subheader("Prediction result")

            result_col, probability_col = st.columns(2)

            with result_col:
                if churn == "Yes":
                    st.error("Prediction: Customer likely to churn")
                else:
                    st.success("Prediction: Customer likely to stay")

            with probability_col:
                st.metric("Estimated churn probability", f"{probability:.2%}")
                st.progress(probability)

            st.caption(
                "This is a model estimate, not a guarantee of customer behavior."
            )
        else:
            st.error(
                f"API returned HTTP {response.status_code}: "
                f"{response.text[:300]}"
            )

    except requests.RequestException:
        st.error(
            "Could not connect to the prediction API. "
            "Make sure the FastAPI server is running on port 8000."
        )
