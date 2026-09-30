import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict, Any
from src.data_pipeline import (
    SEED,
    GROUP_CODE,
    train_test_split_manual,
    standardize_fit,
    standardize_transform,
    save_scaler,
)


def add_bias(X: np.ndarray) -> np.ndarray:
    """Add bias column (intercept) to feature matrix."""
    return np.hstack([np.ones((X.shape[0], 1)), X])


def initialize_weights(n_features: int) -> np.ndarray:
    """Initialize weights with small random values."""
    np.random.seed(SEED)
    return np.random.randn(n_features + 1) * 0.01


def predict_linear(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Linear prediction: X @ weights."""
    return X @ weights


def compute_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error loss."""
    return float(np.mean((y_true - y_pred) ** 2))


def compute_gradient(X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute gradient of MSE loss."""
    n = X.shape[0]
    return (-2 / n) * X.T @ (y_true - y_pred)


def gradient_descent(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    learning_rate: float = 0.01,
    max_epochs: int = 2000,
    tolerance: float = 1e-6,
) -> Tuple[np.ndarray, list, list]:
    """Batch gradient descent for linear regression."""
    X_train_bias = add_bias(X_train)
    X_val_bias = add_bias(X_val)

    weights = initialize_weights(X_train.shape[1])
    train_losses = []
    val_losses = []

    prev_loss = float("inf")

    for epoch in range(max_epochs):
        y_pred_train = predict_linear(X_train_bias, weights)
        train_loss = compute_loss(y_train, y_pred_train)
        train_losses.append(train_loss)

        y_pred_val = predict_linear(X_val_bias, weights)
        val_loss = compute_loss(y_val, y_pred_val)
        val_losses.append(val_loss)

        gradient = compute_gradient(X_train_bias, y_train, y_pred_train)
        weights -= learning_rate * gradient

        if abs(prev_loss - train_loss) < tolerance:
            break
        prev_loss = train_loss

    return weights, train_losses, val_losses


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE, R-squared."""
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0
    return {"mae": mae, "rmse": rmse, "r2": r2}


def train_regression(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    learning_rate: float = 0.01,
    max_epochs: int = 2000,
) -> Dict[str, Any]:
    """Train linear regression with gradient descent."""
    X_train, X_test, y_train, y_test = train_test_split_manual(X, y, test_size, SEED)

    mean, std = standardize_fit(X_train)
    X_train_scaled = standardize_transform(X_train, mean, std)
    X_test_scaled = standardize_transform(X_test, mean, std)

    weights, train_losses, val_losses = gradient_descent(
        X_train_scaled, y_train, X_test_scaled, y_test, learning_rate, max_epochs
    )

    X_test_bias = add_bias(X_test_scaled)
    y_pred = predict_linear(X_test_bias, weights)

    metrics = compute_metrics(y_test, y_pred)
    metrics["seed"] = SEED
    metrics["group_code"] = GROUP_CODE
    metrics["learning_rate"] = learning_rate
    metrics["max_epochs"] = max_epochs
    metrics["final_train_loss"] = train_losses[-1]
    metrics["final_val_loss"] = val_losses[-1]

    return {
        "weights": weights.tolist(),
        "scaler_mean": mean.tolist(),
        "scaler_std": std.tolist(),
        "metrics": metrics,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist(),
    }


def save_regression_artifacts(
    result: Dict[str, Any],
    metrics_path: Path,
    loss_plot_path: Path,
    model_path: Path,
):
    """Save regression metrics, loss plot, and model."""
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "weights": result["weights"],
                "scaler_mean": result["scaler_mean"],
                "scaler_std": result["scaler_std"],
                "metrics": result["metrics"],
            },
            f,
            indent=2,
        )

    plt.figure(figsize=(8, 5))
    plt.plot(result["train_losses"], label="Train Loss (MSE)")
    plt.plot(result["val_losses"], label="Validation Loss (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Linear Regression Gradient Descent Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(loss_plot_path, dpi=150)
    plt.close()

    np.savez(
        model_path,
        weights=np.array(result["weights"]),
        scaler_mean=np.array(result["scaler_mean"]),
        scaler_std=np.array(result["scaler_std"]),
    )


def load_regression_model(model_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load regression model weights and scaler."""
    data = np.load(model_path)
    return data["weights"], data["scaler_mean"], data["scaler_std"]


def predict_regression(
    X: np.ndarray,
    weights: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_std: np.ndarray,
) -> np.ndarray:
    """Make regression predictions on new data."""
    X_scaled = standardize_transform(X, scaler_mean, scaler_std)
    X_bias = add_bias(X_scaled)
    return predict_linear(X_bias, weights)


if __name__ == "__main__":
    from src.data_pipeline import load_and_validate, separate_features_targets

    csv_path = Path("data/AI_A1_G01.csv")
    df = load_and_validate(csv_path)
    _, X, y_reg, _ = separate_features_targets(df)

    result = train_regression(X, y_reg)
    print(f"Regression metrics: {result['metrics']}")

    save_regression_artifacts(
        result,
        Path("artifacts/regression_metrics.json"),
        Path("artifacts/regression_loss.png"),
        Path("models/regression_model.npz"),
    )
    print("Regression artifacts saved")