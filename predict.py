#!/usr/bin/env python3
"""
Prediction CLI for Musanze Cooperative models.
Accepts a single JSON record and returns predictions.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data_pipeline import FEATURE_COLS, GROUP_CODE, SEED
from src.regression import load_regression_model, predict_regression


MODEL_VERSION = "1.0.0"
REQUIRED_FIELDS = FEATURE_COLS


def validate_input(record: dict) -> tuple[bool, str]:
    """Validate input record has all required fields with correct types."""
    missing = [f for f in REQUIRED_FIELDS if f not in record]
    if missing:
        return False, f"Missing required fields: {missing}"

    for field in REQUIRED_FIELDS:
        value = record[field]
        if not isinstance(value, (int, float)):
            return False, f"Field '{field}' must be numeric, got {type(value).__name__}"

    return True, ""


def load_classification_model(models_dir: Path):
    """Load classification model with lazy import."""
    try:
        import joblib
        return joblib.load(models_dir / "classification_model.joblib")
    except Exception:
        return None


def load_clustering_model(models_dir: Path):
    """Load clustering model with lazy import."""
    try:
        data = np.load(models_dir / "clustering_model.npz")
        return {
            "cluster_centers": data["cluster_centers"],
            "scaler_mean": data["scaler_mean"],
            "scaler_std": data["scaler_std"],
            "k": int(data["k"]),
        }
    except Exception:
        return None


def predict_classification(X: np.ndarray, model, scaler) -> tuple[np.ndarray, np.ndarray]:
    """Make classification predictions with fallback."""
    if model is None or scaler is None:
        return np.array([0]), np.array([0.5])
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]
    return y_pred, y_prob


def predict_clustering(
    X: np.ndarray,
    cluster_centers: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_std: np.ndarray,
) -> np.ndarray:
    """Assign cluster labels to new data."""
    X_scaled = (X - scaler_mean) / scaler_std
    distances = np.linalg.norm(X_scaled[:, np.newaxis] - cluster_centers, axis=2)
    return np.argmin(distances, axis=1)


def main():
    parser = argparse.ArgumentParser(description="Predict harvest yield, dispatch attention, and cluster")
    parser.add_argument("--record", required=True, help="JSON record with feature values")
    parser.add_argument("--models", default="models", help="Models directory")
    args = parser.parse_args()

    models_dir = Path(args.models)

    try:
        record = json.loads(args.record)
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON: {e}"}), file=sys.stderr)
        sys.exit(1)

    valid, error_msg = validate_input(record)
    if not valid:
        print(json.dumps({"error": error_msg}), file=sys.stderr)
        sys.exit(1)

    X = np.array([[record[f] for f in REQUIRED_FIELDS]], dtype=float)

    try:
        reg_weights, reg_mean, reg_std = load_regression_model(models_dir / "regression_model.npz")
    except FileNotFoundError as e:
        print(json.dumps({"error": f"Regression model not found: {e}"}), file=sys.stderr)
        sys.exit(1)

    clf_data = load_classification_model(models_dir)
    cluster_data = load_clustering_model(models_dir)

    reg_pred = predict_regression(X, reg_weights, reg_mean, reg_std)[0]

    clf_model = clf_data["model"] if clf_data else None
    clf_scaler = clf_data["scaler"] if clf_data else None
    clf_pred, clf_prob = predict_classification(X, clf_model, clf_scaler)

    if cluster_data:
        cluster_label = predict_clustering(
            X,
            cluster_data["cluster_centers"],
            cluster_data["scaler_mean"],
            cluster_data["scaler_std"],
        )[0]
    else:
        cluster_label = 0

    result = {
        "regression_prediction_kg": float(reg_pred),
        "classification_prediction": int(clf_pred[0]),
        "classification_probability": float(clf_prob[0]),
        "cluster_label": int(cluster_label),
        "group_code": GROUP_CODE,
        "model_version": MODEL_VERSION,
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()