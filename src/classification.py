import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
)
from src.data_pipeline import SEED, GROUP_CODE, FEATURE_COLS, TARGET_CLASSIFICATION


def train_classification(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
) -> Dict[str, Any]:
    """Train logistic regression classifier."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=SEED, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(random_state=SEED, max_iter=1000)
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    y_prob = clf.predict_proba(X_test_scaled)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "seed": SEED,
        "group_code": GROUP_CODE,
        "test_size": test_size,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "class_distribution_train": np.bincount(y_train).tolist(),
        "class_distribution_test": np.bincount(y_test).tolist(),
    }

    return {
        "model": clf,
        "scaler": scaler,
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "y_prob": y_prob.tolist(),
        "feature_names": FEATURE_COLS,
    }


def save_classification_artifacts(
    result: Dict[str, Any],
    metrics_path: Path,
    cm_plot_path: Path,
    model_path: Path,
):
    """Save classification metrics, confusion matrix plot, and model."""
    import joblib

    with open(metrics_path, "w") as f:
        json.dump(
            {
                "metrics": result["metrics"],
                "confusion_matrix": result["confusion_matrix"],
                "feature_names": result["feature_names"],
            },
            f,
            indent=2,
        )

    plt.figure(figsize=(6, 5))
    cm = np.array(result["confusion_matrix"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Attention", "Attention"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Classification Confusion Matrix")
    plt.tight_layout()
    plt.savefig(cm_plot_path, dpi=150)
    plt.close()

    joblib.dump({"model": result["model"], "scaler": result["scaler"]}, model_path)


def load_classification_model(model_path: Path):
    """Load classification model and scaler."""
    import joblib
    return joblib.load(model_path)


def predict_classification(
    X: np.ndarray,
    model,
    scaler,
) -> Tuple[np.ndarray, np.ndarray]:
    """Make classification predictions on new data."""
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]
    return y_pred, y_prob


def error_cost_analysis() -> str:
    """Explain which error is more costly in this scenario."""
    return (
        "In the Musanze Cooperative context, a False Negative (predicting 0 when actual is 1) "
        "is more costly than a False Positive. A False Negative means a consignment needing "
        "dispatch attention is missed, potentially leading to spoilage, quality issues, or "
        "farmer dissatisfaction. A False Positive only triggers unnecessary attention, "
        "which wastes some resources but prevents the worse outcome."
    )


if __name__ == "__main__":
    from src.data_pipeline import load_and_validate, separate_features_targets

    csv_path = Path("data/AI_A1_G01.csv")
    df = load_and_validate(csv_path)
    _, X, _, y_clf = separate_features_targets(df)

    result = train_classification(X, y_clf)
    print(f"Classification metrics: {result['metrics']}")
    print(f"Error cost analysis: {error_cost_analysis()}")

    save_classification_artifacts(
        result,
        Path("artifacts/classification_metrics.json"),
        Path("artifacts/confusion_matrix.png"),
        Path("models/classification_model.joblib"),
    )
    print("Classification artifacts saved")