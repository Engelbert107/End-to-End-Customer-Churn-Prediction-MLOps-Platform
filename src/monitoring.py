
"""SQLite-backed prediction monitoring and PSI data-drift detection."""

import json
import logging
import math
import os
import sqlite3
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DB_PATH = Path(
    os.getenv(
        "MONITORING_DB_PATH",
        str(PROJECT_ROOT / "monitoring-data" / "monitoring.db"),
    )
)
BASELINE_PATH = Path(
    os.getenv(
        "DRIFT_BASELINE_PATH",
        str(PROJECT_ROOT / "reports" / "drift_baseline.json"),
    )
)

logger = logging.getLogger(__name__)
_db_lock = Lock()

NUMERIC_RANGES = {
    "tenure": [(0, 12), (12, 24), (24, 48), (48, float("inf"))],
    "MonthlyCharges": [
        (0, 25), (25, 50), (50, 75), (75, float("inf"))
    ],
    "TotalCharges": [
        (0, 500), (500, 1500), (1500, 3000), (3000, float("inf"))
    ],
}

CATEGORICAL_FEATURES = [
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
]


def _connect():
    """Create the monitoring database and table if needed."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(DB_PATH), timeout=10)

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_metrics (
            day TEXT PRIMARY KEY,
            total_predictions INTEGER NOT NULL DEFAULT 0,
            churn_yes INTEGER NOT NULL DEFAULT 0,
            churn_no INTEGER NOT NULL DEFAULT 0,
            probability_sum REAL NOT NULL DEFAULT 0,
            probability_bins TEXT NOT NULL DEFAULT '{}',
            feature_buckets TEXT NOT NULL DEFAULT '{}'
        )
        """
    )
    return connection


def _bucket_features(features):
    """Convert one customer's features to aggregate bucket labels."""
    buckets = {}

    for feature, ranges in NUMERIC_RANGES.items():
        value = features.get(feature)

        # The training CSV may contain missing TotalCharges values.
        if value is None:
            continue

        try:
            value = float(value)
        except (TypeError, ValueError):
            continue

        if not math.isfinite(value):
            continue

        for lower, upper in ranges:
            if lower <= value < upper:
                label = f"{feature}:{lower:g}-{upper:g}"
                buckets[label] = 1
                break

    for feature in CATEGORICAL_FEATURES:
        value = features.get(feature)

        if value is None:
            continue

        try:
            if bool(value != value):  # Handles NaN without requiring pandas.
                continue
        except (TypeError, ValueError):
            continue

        label = f"{feature}:{value}"
        buckets[label] = 1

    return buckets


def record_prediction(features, prediction, churn_probability):
    """Record daily aggregates without storing individual predictions."""
    now = datetime.now(timezone.utc)
    day = now.date().isoformat()

    probability = float(churn_probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("churn_probability must be between 0 and 1")

    probability_bin = min(int(probability * 5), 4)
    probability_label = (
        f"{probability_bin * 20}-{(probability_bin + 1) * 20}%"
    )
    incoming_features = _bucket_features(features)

    with _db_lock:
        connection = _connect()
        try:
            connection.execute(
                "INSERT OR IGNORE INTO daily_metrics (day) VALUES (?)",
                (day,),
            )

            row = connection.execute(
                """
                SELECT total_predictions, churn_yes, churn_no,
                       probability_sum, probability_bins, feature_buckets
                FROM daily_metrics WHERE day = ?
                """,
                (day,),
            ).fetchone()

            total, yes, no, probability_sum, bins_json, features_json = row
            probability_bins = json.loads(bins_json)
            feature_buckets = json.loads(features_json)

            probability_bins[probability_label] = (
                probability_bins.get(probability_label, 0) + 1
            )

            for label, count in incoming_features.items():
                feature_buckets[label] = (
                    feature_buckets.get(label, 0) + count
                )

            is_churn = int(prediction) == 1

            connection.execute(
                """
                UPDATE daily_metrics
                SET total_predictions = ?,
                    churn_yes = ?,
                    churn_no = ?,
                    probability_sum = ?,
                    probability_bins = ?,
                    feature_buckets = ?
                WHERE day = ?
                """,
                (
                    total + 1,
                    yes + int(is_churn),
                    no + int(not is_churn),
                    probability_sum + probability,
                    json.dumps(probability_bins),
                    json.dumps(feature_buckets),
                    day,
                ),
            )
            connection.commit()
        finally:
            connection.close()


def get_metrics(days=30):
    """Return aggregate prediction metrics for the selected period."""
    if not 1 <= days <= 365:
        raise ValueError("days must be between 1 and 365")

    start_day = (
        datetime.now(timezone.utc).date() - timedelta(days=days - 1)
    ).isoformat()

    with _db_lock:
        connection = _connect()
        try:
            rows = connection.execute(
                """
                SELECT day, total_predictions, churn_yes, churn_no,
                       probability_sum, probability_bins, feature_buckets
                FROM daily_metrics
                WHERE day >= ?
                ORDER BY day
                """,
                (start_day,),
            ).fetchall()
        finally:
            connection.close()

    daily = []
    total_predictions = total_yes = total_no = 0
    probability_sum = 0.0
    probability_bins = Counter()
    feature_buckets = Counter()

    for day, total, yes, no, prob_sum, bins_json, features_json in rows:
        probability_bins.update(json.loads(bins_json))
        feature_buckets.update(json.loads(features_json))

        daily.append(
            {
                "date": day,
                "total_predictions": total,
                "churn_yes": yes,
                "churn_no": no,
                "average_churn_probability": (
                    round(prob_sum / total, 4) if total else None
                ),
            }
        )

        total_predictions += total
        total_yes += yes
        total_no += no
        probability_sum += prob_sum

    return {
        "period_days": days,
        "total_predictions": total_predictions,
        "churn_yes": total_yes,
        "churn_no": total_no,
        "average_churn_probability": (
            round(probability_sum / total_predictions, 4)
            if total_predictions else None
        ),
        "probability_distribution": dict(probability_bins),
        "input_feature_buckets": dict(feature_buckets),
        "daily": daily,
        "note": (
            "Aggregated metrics only; individual customer records "
            "and prediction events are not stored."
        ),
    }


def _psi(expected_counts, actual_counts):
    """Calculate PSI from expected and observed category counts."""
    expected_total = sum(expected_counts.values())
    actual_total = sum(actual_counts.values())

    if expected_total <= 0 or actual_total <= 0:
        return None

    categories = set(expected_counts) | set(actual_counts)
    if not categories:
        return None

    # Small smoothing prevents logarithms of zero. Scores can still be
    # unstable when the observed sample size is very small.
    epsilon = 1e-6
    category_count = len(categories)
    score = 0.0

    for category in categories:
        expected = (
            expected_counts.get(category, 0) + epsilon
        ) / (expected_total + epsilon * category_count)
        actual = (
            actual_counts.get(category, 0) + epsilon
        ) / (actual_total + epsilon * category_count)

        score += (actual - expected) * math.log(actual / expected)

    return round(score, 6)


def get_drift_metrics(days=30):
    """Compare recent input distributions with the training baseline."""
    if not 1 <= days <= 365:
        raise ValueError("days must be between 1 and 365")

    if not BASELINE_PATH.is_file():
        return {
            "status": "baseline_missing",
            "message": (
                "Training baseline not found. Run "
                "python scripts/create_drift_baseline.py first."
            ),
            "baseline_path": str(BASELINE_PATH),
            "period_days": days,
            "features": [],
        }

    try:
        with BASELINE_PATH.open("r", encoding="utf-8") as file:
            baseline = json.load(file)
    except (OSError, json.JSONDecodeError):
        logger.exception("Could not read the drift baseline.")
        raise

    metrics = get_metrics(days=days)
    observed_labels = metrics["input_feature_buckets"]
    baseline_features = baseline.get("feature_distributions", {})

    observed_by_feature = {}
    for label, count in observed_labels.items():
        feature, separator, category = label.partition(":")
        if separator:
            observed_by_feature.setdefault(feature, {})
            observed_by_feature[feature][category] = count

    results = []

    for feature, expected_counts in baseline_features.items():
        actual_counts = observed_by_feature.get(feature, {})
        expected_total = sum(expected_counts.values())
        actual_total = sum(actual_counts.values())

        if metrics["total_predictions"] == 0 or actual_total == 0:
            results.append(
                {
                    "feature": feature,
                    "psi": None,
                    "status": "insufficient_data",
                    "baseline_count": expected_total,
                    "observed_count": actual_total,
                }
            )
            continue

        score = _psi(expected_counts, actual_counts)

        if score is None:
            status = "insufficient_data"
        elif score < 0.1:
            status = "low"
        elif score < 0.2:
            status = "moderate"
        else:
            status = "high"

        results.append(
            {
                "feature": feature,
                "psi": score,
                "status": status,
                "baseline_count": expected_total,
                "observed_count": actual_total,
            }
        )

    status_order = {"high": 0, "moderate": 1, "low": 2,
                    "insufficient_data": 3}
    results.sort(
        key=lambda item: (
            status_order.get(item["status"], 4),
            -(item["psi"] or 0),
        )
    )

    scored = [item for item in results if item["psi"] is not None]

    return {
        "status": "ok",
        "period_days": days,
        "baseline_source": baseline.get("source", "unknown"),
        "baseline_rows": baseline.get("total_rows"),
        "total_predictions": metrics["total_predictions"],
        "thresholds": {
            "low": "PSI < 0.10",
            "moderate": "0.10 <= PSI < 0.20",
            "high": "PSI >= 0.20",
        },
        "high_drift_features": sum(
            item["status"] == "high" for item in scored
        ),
        "moderate_drift_features": sum(
            item["status"] == "moderate" for item in scored
        ),
        "features": results,
        "note": (
            "Population Stability Index (PSI) thresholds are rules of thumb, not universal guarantees. "
            "Interpret scores cautiously when prediction sample sizes "
            "are small. Drift does not directly measure model accuracy."
        ),
    }
