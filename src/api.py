
"""FastAPI service for customer churn prediction."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"

# The trained model is loaded during application startup.
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained model when the API starts."""
    global model

    if not MODEL_PATH.is_file():
        raise RuntimeError(
            f"Model file not found: {MODEL_PATH}. "
            "Run python scripts/train_models.py first."
        )

    # Only load model artifacts created by your trusted project.
    model = joblib.load(MODEL_PATH)

    yield


app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Predict customer churn using the trained "
        "Telco Customer Churn classification model."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class CustomerData(BaseModel):
    """Validated customer features expected by the model."""

    model_config = ConfigDict(extra="forbid")

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=1000)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


@app.get("/")
def root():
    """Return basic API information."""
    return {
        "message": "Customer Churn Prediction API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    """Check whether the model is loaded."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    return {
        "status": "healthy",
        "model_loaded": True,
    }


@app.post("/predict")
def predict(customer: CustomerData):
    """Predict churn for one customer."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    try:
        # Preserve the feature names expected by the training pipeline.
        features = pd.DataFrame([customer.model_dump()])

        prediction = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]

        churn_probability = float(probabilities[1])
        no_churn_probability = float(probabilities[0])

        return {
            "churn_prediction": "Yes" if prediction == 1 else "No",
            "churn_probability": round(churn_probability, 4),
            "no_churn_probability": round(no_churn_probability, 4),
            "model": "best_model.joblib",
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the API logs.",
        ) from exc
