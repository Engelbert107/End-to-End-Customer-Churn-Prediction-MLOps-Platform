
"""Data validation and cleaning for the churn project."""

import json
from pathlib import Path

import pandas as pd

from src.schema import (
    CATEGORICAL_COLUMNS,
    EXPECTED_COLUMNS,
    NUMERIC_COLUMNS,
    REQUIRED_COLUMNS,
)


def validate_schema(df: pd.DataFrame) -> None:
    """Raise an error if required dataset columns are missing or unexpected."""
    actual = list(df.columns)

    missing = sorted(set(EXPECTED_COLUMNS) - set(actual))
    unexpected = sorted(set(actual) - set(EXPECTED_COLUMNS))

    if missing or unexpected:
        raise ValueError(
            "Dataset schema mismatch. "
            f"Missing columns: {missing}. "
            f"Unexpected columns: {unexpected}."
        )

    if len(actual) != len(set(actual)):
        raise ValueError("Dataset contains duplicate column names.")

    # Keep the dataset's canonical column order.
    if actual != EXPECTED_COLUMNS:
        df = df[EXPECTED_COLUMNS]


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Validate and clean a copy of the input DataFrame."""
    df = df.copy()

    # Normalize column names and string values.
    df.columns = df.columns.str.strip()
    validate_schema(df)

    for column in CATEGORICAL_COLUMNS:
        df[column] = df[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )

    # Treat whitespace-only cells as missing values.
    df = df.replace(r"^\s*$", pd.NA, regex=True)

    rows_before = len(df)

    # Remove exact duplicate records.
    exact_duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    # Customer IDs should identify individual customers.
    duplicate_ids = int(df["customerID"].duplicated().sum())
    if duplicate_ids:
        raise ValueError(
            f"Found {duplicate_ids} duplicate customerID values. "
            "Investigate these records rather than silently deleting them."
        )

    # IDs and target labels must not be missing.
    for column in REQUIRED_COLUMNS:
        if df[column].isna().any():
            raise ValueError(
                f"Required column '{column}' contains missing values."
            )

    # Convert numeric fields; invalid values become missing.
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Numeric columns must contain valid values.
    for column in NUMERIC_COLUMNS:
        if df[column].isna().any():
            if column == "TotalCharges":
                # Median imputation is a simple baseline strategy.
                median = df[column].median()
                if pd.isna(median):
                    raise ValueError(
                        f"Cannot impute '{column}': all values are missing."
                    )
                df[column] = df[column].fillna(median)
            else:
                raise ValueError(
                    f"Numeric column '{column}' contains missing or "
                    "invalid values. Investigate before continuing."
                )

    # Validate the target values before modeling.
    valid_churn_labels = {"Yes", "No"}
    observed_labels = set(df["Churn"].dropna().unique())
    if not observed_labels.issubset(valid_churn_labels):
        raise ValueError(
            f"Unexpected Churn labels: {observed_labels - valid_churn_labels}"
        )

    # Fill missing categorical features with an explicit sentinel.
    for column in CATEGORICAL_COLUMNS:
        if column in REQUIRED_COLUMNS:
            continue
        df[column] = df[column].fillna("__MISSING__")

    report = {
        "rows_before": rows_before,
        "rows_after": len(df),
        "exact_duplicates_removed": exact_duplicates,
        "duplicate_customer_ids": duplicate_ids,
        "missing_values_after_cleaning": {
            column: int(count)
            for column, count in df.isna().sum().items()
        },
        "columns": len(df.columns),
        "churn_distribution": {
            str(label): int(count)
            for label, count in df["Churn"].value_counts().items()
        },
    }

    return df, report


def run_pipeline(
    input_path: Path,
    output_path: Path,
) -> dict:
    """Load, validate, clean, and save the dataset and its report."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)

    cleaned_df, report = clean_data(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)

    report_path = output_path.with_name("data_quality_report.json")
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"Raw rows: {report['rows_before']}")
    print(f"Clean rows: {report['rows_after']}")
    print(f"Exact duplicates removed: {report['exact_duplicates_removed']}")
    print(f"Clean CSV saved to: {output_path}")
    print(f"Quality report saved to: {report_path}")

    return report
