
"""Tests for the customer churn data pipeline."""

import pandas as pd
import pytest

from src.data_pipeline import clean_data
from src.schema import EXPECTED_COLUMNS


@pytest.fixture
def sample_data():
    """Return a small dataset with the expected schema."""
    return pd.DataFrame(
        [
            [
                "C001", "Female", 0, "Yes", "No", 12,
                "Yes", "No", "DSL", "Yes", "No", "No",
                "No", "No", "No", "Month-to-month",
                "Yes", "Electronic check", 50.0, "600.0", "No",
            ],
            [
                "C002", "Male", 0, "No", "No", 2,
                "Yes", "No", "Fiber optic", "No", "No", "No",
                "No", "Yes", "Yes", "Month-to-month",
                "Yes", "Electronic check", 70.0, "140.0", "Yes",
            ],
        ],
        columns=EXPECTED_COLUMNS,
    )


def test_valid_data_passes(sample_data):
    cleaned, report = clean_data(sample_data)

    assert len(cleaned) == 2
    assert len(cleaned.columns) == 21
    assert report["rows_after"] == 2


def test_exact_duplicates_are_removed(sample_data):
    duplicated = pd.concat(
        [sample_data, sample_data.iloc[[0]]],
        ignore_index=True,
    )

    cleaned, report = clean_data(duplicated)

    assert len(cleaned) == 2
    assert report["exact_duplicates_removed"] == 1


def test_missing_total_charges_are_imputed(sample_data):
    sample_data.loc[0, "TotalCharges"] = "   "

    cleaned, _ = clean_data(sample_data)

    assert cleaned["TotalCharges"].notna().all()
    assert pd.api.types.is_numeric_dtype(cleaned["TotalCharges"])


def test_missing_required_target_raises_error(sample_data):
    sample_data.loc[0, "Churn"] = None

    with pytest.raises(ValueError, match="Churn"):
        clean_data(sample_data)


def test_missing_column_raises_error(sample_data):
    invalid_data = sample_data.drop(columns=["Contract"])

    with pytest.raises(ValueError, match="schema mismatch"):
        clean_data(invalid_data)


def test_duplicate_customer_ids_raise_error(sample_data):
    sample_data.loc[1, "customerID"] = "C001"

    with pytest.raises(ValueError, match="duplicate customerID"):
        clean_data(sample_data)
