#!/usr/bin/env python3
"""
Complete pipeline for Musanze Cooperative Harvest and Dispatch Decision Lab.
Runs data validation, regression, classification, and clustering.
"""
import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data_pipeline import (
    load_and_validate,
    separate_features_targets,
    generate_data_report,
    FEATURE_COLS,
)
from src.regression import train_regression, save_regression_artifacts
from src.classification import train_classification, save_classification_artifacts, error_cost_analysis
from src.clustering import train_clustering, save_clustering_artifacts, clustering_interpretation


def main():
    parser = argparse.ArgumentParser(description="Run complete ML pipeline")
    parser.add_argument("--data", required=True, help="Path to input CSV")
    parser.add_argument("--output", required=True, help="Output artifacts directory")
    parser.add_argument("--group", required=True, help="Group code (e.g., AI-G01)")
    args = parser.parse_args()

    data_path = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {data_path}...")
    df = load_and_validate(data_path)
    print(f"Data shape: {df.shape}")

    print("Generating data report...")
    report = generate_data_report(df, data_path, output_dir / "data_report.json")
    print(f"Data report saved. SHA-256: {report['dataset_fingerprint_sha256']}")

    ids, X, y_reg, y_clf = separate_features_targets(df)

    print("Training regression model...")
    reg_result = train_regression(X, y_reg)
    save_regression_artifacts(
        reg_result,
        output_dir / "regression_metrics.json",
        output_dir / "regression_loss.png",
        models_dir / "regression_model.npz",
    )
    print(f"Regression metrics: MAE={reg_result['metrics']['mae']:.2f}, "
          f"RMSE={reg_result['metrics']['rmse']:.2f}, R2={reg_result['metrics']['r2']:.4f}")

    print("Training classification model...")
    clf_result = train_classification(X, y_clf)
    save_classification_artifacts(
        clf_result,
        output_dir / "classification_metrics.json",
        output_dir / "confusion_matrix.png",
        models_dir / "classification_model.joblib",
    )
    print(f"Classification metrics: Acc={clf_result['metrics']['accuracy']:.4f}, "
          f"F1={clf_result['metrics']['f1']:.4f}")
    print(f"Error cost: {error_cost_analysis()}")

    print("Training clustering model...")
    cluster_result = train_clustering(X)
    save_clustering_artifacts(
        cluster_result,
        output_dir / "clustering_metrics.json",
        output_dir / "clusters.csv",
        output_dir / "cluster_plot.png",
        ids,
    )
    print(f"Clustering: k={cluster_result['k']}, Silhouette={cluster_result['silhouette_score']:.4f}")
    print(f"Interpretation: {clustering_interpretation()}")

    print("\nPipeline complete! Artifacts saved to:", output_dir)
    print("Models saved to:", models_dir)


if __name__ == "__main__":
    main()