#!/usr/bin/env python
"""
Training script for HVAC failure prediction.

This mirrors the logic from the Colab notebook but is written so it can be
called from:
- the command line
- an Azure ML pipeline step
- a CI job (GitHub Actions / Azure DevOps)

Usage (locally or in a container):

    python ml/training/train_anomaly_model.py \
        --n-units 25 \
        --days 60 \
        --freq-minutes 15 \
        --output-model-path ml/models/hvac_failure_rf_model.pkl \
        --output-metrics-path ml/models/hvac_failure_metrics.json
"""

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
import joblib


# -----------------------------
# Config / dataclasses
# -----------------------------

@dataclass
class TrainConfig:
    n_units: int = 25
    days: int = 60
    freq_minutes: int = 15
    random_seed: int = 42
    output_model_path: Path = Path("ml/models/hvac_failure_rf_model.pkl")
    output_metrics_path: Path = Path("ml/models/hvac_failure_metrics.json")


# -----------------------------
# Data generation
# -----------------------------

def generate_unit_data(
    unit_id: int,
    start_time: datetime,
    days: int,
    freq_minutes: int,
) -> pd.DataFrame:
    periods = int((days * 24 * 60) / freq_minutes)
    time_index = pd.date_range(start=start_time, periods=periods, freq=f"{freq_minutes}min")

    # Baseline signals
    ambient_temp = 20 + 10 * np.sin(np.linspace(0, 10 * np.pi, periods))
    load_factor = np.clip(np.random.normal(0.7, 0.1, periods), 0, 1)
    supply_temp = ambient_temp - 5 + 8 * load_factor
    vibration = np.random.normal(0.3, 0.05, periods) + 0.1 * load_factor
    power_kw = 5 + 10 * load_factor + np.random.normal(0, 0.5, periods)
    humidity = np.clip(np.random.normal(45, 8, periods), 20, 80)

    failures = np.zeros(periods, dtype=int)
    n_fail_events = np.random.randint(1, 4)

    for _ in range(n_fail_events):
        failure_start = np.random.randint(periods // 4, periods - periods // 10)
        window = int(24 * 60 / freq_minutes)  # 24h window
        end = min(periods, failure_start + window)

        # Degrading behavior before failure
        vibration[failure_start:end] += np.linspace(0.1, 0.5, end - failure_start)
        supply_temp[failure_start:end] += np.linspace(1.0, 3.0, end - failure_start)
        power_kw[failure_start:end] += np.linspace(0.5, 2.0, end - failure_start)

        failures[end - 1] = 1  # mark failure at end of window

    df = pd.DataFrame(
        {
            "timestamp": time_index,
            "unit_id": unit_id,
            "ambient_temp": ambient_temp,
            "supply_temp": supply_temp,
            "load_factor": load_factor,
            "vibration": vibration,
            "power_kw": power_kw,
            "humidity": humidity,
            "failure_event": failures,
        }
    )
    return df


def add_future_failure_label(
    df: pd.DataFrame,
    horizon_hours: int,
    freq_minutes: int,
) -> pd.DataFrame:
    horizon_steps = int(horizon_hours * 60 / freq_minutes)
    df = df.sort_values("timestamp")
    failure_future = (
        df["failure_event"]
        .rolling(window=horizon_steps, min_periods=1)
        .max()
        .shift(-horizon_steps + 1)
    )
    df["failure_in_24h"] = failure_future.fillna(0).astype(int)
    return df


def generate_dataset(cfg: TrainConfig) -> pd.DataFrame:
    np.random.seed(cfg.random_seed)
    start_time = datetime.now() - timedelta(days=cfg.days)

    all_units = [
        generate_unit_data(i, start_time, cfg.days, cfg.freq_minutes)
        for i in range(cfg.n_units)
    ]
    data = pd.concat(all_units).reset_index(drop=True)

    data = (
        data.groupby("unit_id", group_keys=False)
        .apply(lambda d: add_future_failure_label(d, 24, cfg.freq_minutes))
        .reset_index(drop=True)
    )
    return data


# -----------------------------
# Model training
# -----------------------------

FEATURE_COLS = [
    "ambient_temp",
    "supply_temp",
    "load_factor",
    "vibration",
    "power_kw",
    "humidity",
]


def train_model(data: pd.DataFrame, cfg: TrainConfig) -> Tuple[RandomForestClassifier, dict]:
    X = data[FEATURE_COLS]
    y = data["failure_in_24h"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=cfg.random_seed
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        n_jobs=-1,
        random_state=cfg.random_seed,
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_proba)

    metrics = {
        "roc_auc": roc_auc,
        "classification_report": report,
        "n_samples": int(len(data)),
        "positive_rate": float(y.mean()),
    }
    return clf, metrics


# -----------------------------
# Entrypoint
# -----------------------------

def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train HVAC failure prediction model.")
    parser.add_argument("--n-units", type=int, default=25)
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--freq-minutes", type=int, default=15)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--output-model-path",
        type=str,
        default="ml/models/hvac_failure_rf_model.pkl",
    )
    parser.add_argument(
        "--output-metrics-path",
        type=str,
        default="ml/models/hvac_failure_metrics.json",
    )

    args = parser.parse_args()

    return TrainConfig(
        n_units=args.n_units,
        days=args.days,
        freq_minutes=args.freq_minutes,
        random_seed=args.random_seed,
        output_model_path=Path(args.output_model_path),
        output_metrics_path=Path(args.output_metrics_path),
    )


def main():
    cfg = parse_args()

    print("Generating synthetic dataset...")
    data = generate_dataset(cfg)
    print(f"Dataset shape: {data.shape}, positive rate={data['failure_in_24h'].mean():.4f}")

    # Azure ML automatically wires MLflow to the current run.
    # We still call start_run() so metrics are properly associated.
    with mlflow.start_run():
        print("Training model...")
        model, metrics = train_model(data, cfg)

        # Ensure output directories exist
        cfg.output_model_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.output_metrics_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Saving model to {cfg.output_model_path}")
        joblib.dump(model, cfg.output_model_path)

        print(f"Saving metrics to {cfg.output_metrics_path}")
        with cfg.output_metrics_path.open("w") as f:
            json.dump(metrics, f, indent=2)

        # ---- MLflow logging ----
        print("Logging metrics to MLflow...")
        mlflow.log_metric("roc_auc", metrics["roc_auc"])
        mlflow.log_metric("n_samples", metrics["n_samples"])
        mlflow.log_metric("positive_rate", metrics["positive_rate"])
        # store the full classification report as an artifact
        mlflow.log_dict(metrics["classification_report"], "classification_report.json")

    print("Done.")


if __name__ == "__main__":
    main()

