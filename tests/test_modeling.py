
"""Basic tests for the modeling components."""

import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from src.train import (
    build_preprocessor,
    evaluate_model,
)


def test_preprocessor_handles_unknown_categories():
    train = pd.DataFrame(
        {
            "tenure": [1, 12, 24, 36],
            "MonthlyCharges": [20.0, 40.0, 60.0, 80.0],
            "TotalCharges": [20.0, 480.0, 1440.0, 2880.0],
            "SeniorCitizen": [0, 0, 1, 0],
            "Contract": ["Month-to-month", "One year", "Two year", "One year"],
        }
    )

    test = pd.DataFrame(
        {
            "tenure": [6],
            "MonthlyCharges": [50.0],
            "TotalCharges": [300.0],
            "SeniorCitizen": [0],
            "Contract": ["New contract type"],
        }
    )

    numeric_columns = [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]

    preprocessor = build_preprocessor(["Contract"])
    transformed_train = preprocessor.fit_transform(train)
    transformed_test = preprocessor.transform(test)

    assert transformed_train.shape[0] == len(train)
    assert transformed_test.shape[0] == len(test)
    assert transformed_train.shape[1] == transformed_test.shape[1]


def test_evaluation_returns_expected_metrics():
    X_train = pd.DataFrame({"feature": [0, 1, 2, 3]})
    y_train = [0, 0, 1, 1]

    model = Pipeline(
        [
            (
                "classifier",
                DummyClassifier(strategy="most_frequent"),
            )
        ]
    )
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_train, y_train)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert len(metrics["confusion_matrix"]) == 2
