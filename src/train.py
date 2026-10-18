
"""Train, compare, and track customer churn models with MLflow."""

import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier


# ---------------------------------------------------------------------
# Project configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

# Use an absolute SQLite database path for consistent local tracking.
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"

EXPERIMENT_NAME = "Customer Churn Prediction"

TARGET = "Churn"
ID_COLUMN = "customerID"
RANDOM_STATE = 42
TEST_SIZE = 0.20

NUMERIC_COLUMNS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

METRIC_NAMES = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
]


# ---------------------------------------------------------------------
# Data loading and preparation
# ---------------------------------------------------------------------

def load_data(path: Path = RAW_DATA_PATH):
    """Load raw data and prepare features and target."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. "
            "Run scripts/download_data.py first."
        )

    df = pd.read_csv(path)

    # Normalize column names and whitespace-only strings.
    df.columns = df.columns.str.strip()
    df = df.replace(r"^\s*$", pd.NA, regex=True)

    required_columns = {
        TARGET,
        ID_COLUMN,
        *NUMERIC_COLUMNS,
    }
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if df[TARGET].isna().any():
        raise ValueError("The target column contains missing values.")

    expected_target_values = {"Yes", "No"}
    if not set(df[TARGET].unique()).issubset(expected_target_values):
        raise ValueError("Unexpected values in the Churn target.")

    if df[ID_COLUMN].isna().any() or df[ID_COLUMN].duplicated().any():
        raise ValueError("customerID must be present and unique.")

    # Convert numeric columns; invalid values become missing.
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Customer IDs are identifiers, not predictive features.
    X = df.drop(columns=[TARGET, ID_COLUMN])
    y = df[TARGET].map({"No": 0, "Yes": 1}).astype(int)

    categorical_columns = [
        column
        for column in X.columns
        if column not in NUMERIC_COLUMNS
    ]

    return X, y, categorical_columns


# ---------------------------------------------------------------------
# Preprocessing and model definitions
# ---------------------------------------------------------------------

def build_preprocessor(categorical_columns):
    """Build preprocessing fitted only on training data."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="__MISSING__",
                ),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def build_models():
    """Return baseline and candidate classifiers."""
    return {
        "DummyBaseline": DummyClassifier(
            strategy="most_frequent"
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    }


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate_model(model, X_test, y_test):
    """Calculate binary classification metrics."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "precision": float(
            precision_score(
                y_test, predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_test, predictions, zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_test, predictions, zero_division=0
            )
        ),
        "roc_auc": float(
            roc_auc_score(y_test, probabilities)
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist(),
    }


def log_classifier_params(classifier):
    """Log classifier hyperparameters to the active MLflow run."""
    params = classifier.get_params(deep=False)

    # MLflow parameters are stored as strings. Skip unset values.
    mlflow_params = {
        key: str(value)
        for key, value in params.items()
        if value is not None
    }

    mlflow.log_params(mlflow_params)


# ---------------------------------------------------------------------
# Training, MLflow tracking, and artifact saving
# ---------------------------------------------------------------------

def train_and_compare():
    """Train models, track experiments, compare results, and save outputs."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Configure SQLite-backed MLflow tracking.
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"MLflow experiment: {EXPERIMENT_NAME}")

    X, y, categorical_columns = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    results = {}
    fitted_models = {}

    for name, classifier in build_models().items():
        print(f"\nTraining {name}...")

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(categorical_columns),
                ),
                ("classifier", classifier),
            ]
        )

        # Every model gets its own independently tracked MLflow run.
        with mlflow.start_run(run_name=name):
            mlflow.set_tags(
                {
                    "model_name": name,
                    "project": "customer-churn-prediction",
                    "stage": "model-training",
                }
            )

            mlflow.log_params(
                {
                    "test_size": TEST_SIZE,
                    "random_state": RANDOM_STATE,
                    "numeric_feature_count": len(NUMERIC_COLUMNS),
                    "categorical_feature_count": len(
                        categorical_columns
                    ),
                }
            )

            log_classifier_params(classifier)

            pipeline.fit(X_train, y_train)

            metrics = evaluate_model(
                pipeline, X_test, y_test
            )

            results[name] = metrics
            fitted_models[name] = pipeline

            # Log numeric metrics to MLflow.
            mlflow.log_metrics(
                {
                    metric: metrics[metric]
                    for metric in METRIC_NAMES
                }
            )

            # Save the confusion matrix as a run artifact.
            mlflow.log_dict(
                {
                    "labels": {
                        "0": "No Churn",
                        "1": "Churn",
                    },
                    "matrix": metrics["confusion_matrix"],
                },
                "confusion_matrix.json",
            )

            # Save the complete preprocessing + model pipeline.
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                serialization_format="cloudpickle",
            )

            print(
                f"ROC-AUC: {metrics['roc_auc']:.4f} | "
                f"Recall: {metrics['recall']:.4f} | "
                f"F1: {metrics['f1']:.4f}"
            )

    # -------------------------------------------------------------
    # Save detailed metrics and comparison report.
    # -------------------------------------------------------------

    metrics_path = REPORT_DIR / "model_metrics.json"
    metrics_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    comparison = pd.DataFrame(
        [
            {
                "model": name,
                **{
                    metric: values[metric]
                    for metric in METRIC_NAMES
                },
            }
            for name, values in results.items()
        ]
    ).sort_values("roc_auc", ascending=False)

    comparison.to_csv(
        REPORT_DIR / "model_comparison.csv",
        index=False,
    )

    # -------------------------------------------------------------
    # Select the best real model using ROC-AUC.
    # -------------------------------------------------------------

    candidate_names = [
        "LogisticRegression",
        "RandomForest",
        "XGBoost",
    ]

    best_name = max(
        candidate_names,
        key=lambda name: results[name]["roc_auc"],
    )

    best_model = fitted_models[best_name]
    best_model_path = MODEL_DIR / "best_model.joblib"

    joblib.dump(best_model, best_model_path)

    summary = {
        "selection_metric": "roc_auc",
        "best_model": best_name,
        "best_roc_auc": results[best_name]["roc_auc"],
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "random_state": RANDOM_STATE,
    }

    summary_path = REPORT_DIR / "best_model_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    # -------------------------------------------------------------
    # Log the selected model in a dedicated MLflow run.
    # -------------------------------------------------------------

    with mlflow.start_run(
        run_name=f"Best Model - {best_name}"
    ):
        mlflow.set_tags(
            {
                "model_name": best_name,
                "run_type": "best_model_selection",
                "selection_metric": "roc_auc",
            }
        )

        mlflow.log_params(
            {
                "selected_model": best_name,
                "selection_metric": "roc_auc",
                "random_state": RANDOM_STATE,
                "test_size": TEST_SIZE,
            }
        )

        mlflow.log_metrics(
            {
                metric: results[best_name][metric]
                for metric in METRIC_NAMES
            }
        )

        mlflow.log_dict(
            summary,
            "best_model_summary.json",
        )

        mlflow.sklearn.log_model(
            sk_model=best_model,
            name="selected_model",
            serialization_format="cloudpickle",
        )

    # -------------------------------------------------------------
    # Print final results.
    # -------------------------------------------------------------

    print("\n=== Model comparison ===")
    print(comparison.to_string(index=False))

    print(f"\nBest model: {best_name}")
    print(f"Saved model: {best_model_path}")
    print(f"Saved reports: {REPORT_DIR}")
    print(f"MLflow database: {MLFLOW_DB_PATH}")
    print(f"MLflow experiment: {EXPERIMENT_NAME}")

    return results, best_name


if __name__ == "__main__":
    train_and_compare()
