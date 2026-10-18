
"""Make a prediction with the saved churn model."""

import joblib
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
model = joblib.load(ROOT / "models" / "best_model.joblib")

customer = pd.DataFrame(
    [
        {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 12,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 50.0,
            "TotalCharges": 600.0,
        }
    ]
)

prediction = int(model.predict(customer)[0])
probability = float(model.predict_proba(customer)[0, 1])

print("Predicted churn:", "Yes" if prediction == 1 else "No")
print(f"Estimated churn probability: {probability:.2%}")
