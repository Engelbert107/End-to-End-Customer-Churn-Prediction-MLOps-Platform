
"""Run the customer churn data pipeline."""

import sys
from pathlib import Path

# Add the project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline import run_pipeline


RAW_DATA = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

PROCESSED_DATA = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "telco_customer_churn_clean.csv"
)


if __name__ == "__main__":
    run_pipeline(
        input_path=RAW_DATA,
        output_path=PROCESSED_DATA,
    )
