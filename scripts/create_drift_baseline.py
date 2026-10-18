
"""Build a feature-distribution baseline from the training dataset."""

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.monitoring import _bucket_features

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)
OUTPUT_PATH = PROJECT_ROOT / "reports" / "drift_baseline.json"



def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Training dataset not found: {DATA_PATH}\n"
            "Run python scripts/download_data.py first."
        )

    data = pd.read_csv(DATA_PATH)

    required_features = {
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Contract",
        "PaymentMethod",
        "InternetService",
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "PaperlessBilling",
    }

    missing = required_features - set(data.columns)
    if missing:
        raise ValueError(
            f"Training dataset is missing required columns: {sorted(missing)}"
        )

    feature_distributions = {}

    for row in data[list(required_features)].to_dict(orient="records"):
        for label in _bucket_features(row):
            feature, _, category = label.partition(":")
            feature_distributions.setdefault(feature, Counter())
            feature_distributions[feature][category] += 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    baseline = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(DATA_PATH.relative_to(PROJECT_ROOT)),
        "total_rows": int(len(data)),
        "feature_distributions": {
            feature: dict(counts)
            for feature, counts in feature_distributions.items()
        },
        "method": (
            "Numeric ranges and categorical counts using the same "
            "feature bucketing function as online monitoring."
        ),
    }

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(baseline, file, indent=2)

    print(f"Baseline saved to: {OUTPUT_PATH}")
    print(f"Training rows: {len(data)}")
    print(f"Features included: {len(feature_distributions)}")


if __name__ == "__main__":
    main()
