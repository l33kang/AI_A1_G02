import pandas as pd
import numpy as np
import json
import hashlib
from pathlib import Path
from typing import Tuple, Dict, Any


SEED = 42
GROUP_CODE = "AI-G01"
FEATURE_COLS = [
    "plot_area_ha",
    "rainfall_mm",
    "soil_ph",
    "seed_kg",
    "distance_km",
    "arrival_hour",
]
TARGET_REGRESSION = "actual_yield_kg"
TARGET_CLASSIFICATION = "dispatch_attention"
ID_COL = "record_id"


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 fingerprint of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_and_validate(csv_path: Path) -> pd.DataFrame:
    """Load CSV and validate schema."""
    df = pd.read_csv(csv_path)

    expected_cols = [ID_COL] + FEATURE_COLS + [TARGET_REGRESSION, TARGET_CLASSIFICATION]
    missing_cols = set(expected_cols) - set(df.columns)
    extra_cols = set(df.columns) - set(expected_cols)

    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    if extra_cols:
        raise ValueError(f"Unexpected columns: {extra_cols}")

    for col in FEATURE_COLS + [TARGET_REGRESSION]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column {col} must be numeric")

    if not set(df[TARGET_CLASSIFICATION].unique()).issubset({0, 1}):
        raise ValueError("dispatch_attention must contain only 0 and 1")

    return df


def check_missing_values(df: pd.DataFrame) -> Dict[str, int]:
    """Report missing values per column."""
    return df.isnull().sum().to_dict()


def check_duplicates(df: pd.DataFrame) -> int:
    """Report duplicate rows (excluding record_id)."""
    check_cols = FEATURE_COLS + [TARGET_REGRESSION, TARGET_CLASSIFICATION]
    return int(df.duplicated(subset=check_cols).sum())


def separate_features_targets(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Separate identifiers, features, and targets."""
    ids = df[ID_COL].values
    X = df[FEATURE_COLS].values.astype(float)
    y_reg = df[TARGET_REGRESSION].values.astype(float)
    y_clf = df[TARGET_CLASSIFICATION].values.astype(int)
    return ids, X, y_reg, y_clf


def descriptive_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute descriptive statistics for features and targets."""
    stats = {}
    for col in FEATURE_COLS + [TARGET_REGRESSION, TARGET_CLASSIFICATION]:
        col_data = df[col]
        if pd.api.types.is_numeric_dtype(col_data):
            stats[col] = {
                "mean": float(col_data.mean()),
                "std": float(col_data.std()),
                "min": float(col_data.min()),
                "max": float(col_data.max()),
                "median": float(col_data.median()),
            }
    return stats


def generate_data_report(
    df: pd.DataFrame,
    csv_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    """Generate and save data report JSON."""
    missing = check_missing_values(df)
    duplicates = check_duplicates(df)
    stats = descriptive_statistics(df)
    sha256 = compute_sha256(csv_path)

    report = {
        "group_code": GROUP_CODE,
        "dataset_fingerprint_sha256": sha256,
        "row_count": int(len(df)),
        "feature_count": len(FEATURE_COLS),
        "missing_values": missing,
        "duplicate_rows": duplicates,
        "descriptive_statistics": stats,
        "columns": list(df.columns),
        "seed": SEED,
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    return report


def train_test_split_manual(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Manual train/test split with fixed seed."""
    np.random.seed(random_state)
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    test_count = int(n_samples * test_size)
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit scaler on training data."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def standardize_transform(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Transform data using fitted scaler."""
    return (X - mean) / std


def save_scaler(mean: np.ndarray, std: np.ndarray, path: Path):
    """Save scaler parameters."""
    np.savez(path, mean=mean, std=std)


def load_scaler(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load scaler parameters."""
    data = np.load(path)
    return data["mean"], data["std"]


if __name__ == "__main__":
    csv_path = Path("data/AI_A1_G01.csv")
    df = load_and_validate(csv_path)
    print("Data loaded and validated successfully")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")