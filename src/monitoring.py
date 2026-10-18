
"""SQLite-backed monitoring for customer churn predictions."""

import json
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

_db_lock = Lock()


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
    """Convert feature values into aggregate buckets."""
    buckets = {}

    numeric_ranges = {
        "tenure": [(0, 12), (12, 24), (24, 48), (48, 1001)],
        "MonthlyCharges": [
            (0, 25),
            (25, 50),
            (50, 75),
            (75, float("inf")),
        ],
        "TotalCharges": [
            (0, 500),
            (500, 1500),
            (1500, 3000),
            (3000, float("inf")),
        ],
    }

    for feature, ranges in numeric_ranges.items():
        value = float(features[feature])

        for lower, upper in ranges:
            if lower <= value < upper:
                label = f"{feature}:{lower:g}-{upper:g}"
                buckets[label] = buckets.get(label, 0) + 1
                break

    categorical_features = [
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

    for feature in categorical_features:
        label = f"{feature}:{features[feature]}"
        buckets[label] = buckets.get(label, 0) + 1

    return buckets


def record_prediction(features, prediction, churn_probability):
    """Record daily aggregates without saving individual customer records."""
    now = datetime.now(timezone.utc)
    day = now.date().isoformat()

    probability = float(churn_probability)
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
                FROM daily_metrics
                WHERE day = ?
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
    """Return aggregate monitoring metrics for the requested period."""
    if not 1 <= days <= 365:
        raise ValueError("days must be between 1 and 365")

    start_day = (
        datetime.now(timezone.utc).date()
        - timedelta(days=days - 1)
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
    total_predictions = 0
    total_yes = 0
    total_no = 0
    probability_sum = 0.0
    probability_bins = Counter()
    feature_buckets = Counter()

    for day, total, yes, no, prob_sum, bins_json, features_json in rows:
        bins = json.loads(bins_json)
        buckets = json.loads(features_json)

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
        probability_bins.update(bins)
        feature_buckets.update(buckets)

    return {
        "period_days": days,
        "total_predictions": total_predictions,
        "churn_yes": total_yes,
        "churn_no": total_no,
        "average_churn_probability": (
            round(probability_sum / total_predictions, 4)
            if total_predictions
            else None
        ),
        "probability_distribution": dict(probability_bins),
        "input_feature_buckets": dict(feature_buckets),
        "daily": daily,
        "note": (
            "Aggregated metrics only; individual customer records "
            "and prediction events are not stored."
        ),
    }
