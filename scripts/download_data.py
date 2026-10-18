
import sys
import subprocess
import pandas as pd
from pathlib import Path





"""  
Configure Kaggle authentication
1. Sign in to Kaggle.
2. Open your account settings and find the API token option.
3. Generate a token and follow Kaggle's current instructions for configuring it locally.
4. Make sure the credentials file is excluded from your Git repository.
"""

# Kaggle dataset identifier
DATASET = "blastchar/telco-customer-churn"

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def download_dataset():
    """Download and validate the Kaggle Telco Churn dataset."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset: {DATASET}")

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "kaggle",
                "datasets",
                "download",
                "-d",
                DATASET,
                "-p",
                str(DATA_DIR),
                "--unzip",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Kaggle download failed. Check your API credentials "
            "and internet connection."
        ) from exc

    # Locate the downloaded CSV
    csv_files = list(DATA_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV file found in {DATA_DIR}"
        )

    # Read the dataset
    csv_path = csv_files[0]
    df = pd.read_csv(csv_path)

    # Basic validation
    required_columns = {"customerID", "Churn"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    print("\nDownload successful!")
    print(f"File: {csv_path}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nChurn distribution:")
    print(df["Churn"].value_counts(dropna=False))

    print("\nMissing values by column:")
    print(df.isna().sum()[lambda x: x > 0])

    print("\nPreview:")
    print(df.head())

    return csv_path


if __name__ == "__main__":
    download_dataset()
