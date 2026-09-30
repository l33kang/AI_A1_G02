import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from src.data_pipeline import SEED, GROUP_CODE, FEATURE_COLS


def evaluate_k_range(
    X: np.ndarray,
    k_range: List[int] = None,
) -> Dict[str, Any]:
    """Evaluate silhouette scores for different k values."""
    if k_range is None:
        k_range = [2, 3, 4, 5]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = {}
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        results[k] = {
            "silhouette_score": float(score),
            "inertia": float(kmeans.inertia_),
            "labels": labels.tolist(),
        }

    best_k = max(results.keys(), key=lambda k: results[k]["silhouette_score"])
    return {
        "k_results": results,
        "best_k": best_k,
        "best_score": results[best_k]["silhouette_score"],
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_std": scaler.scale_.tolist(),
    }


def train_clustering(
    X: np.ndarray,
    k: int = None,
) -> Dict[str, Any]:
    """Train KMeans clustering with optimal k."""
    eval_result = evaluate_k_range(X)
    if k is None:
        k = eval_result["best_k"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    silhouette = silhouette_score(X_scaled, labels)

    return {
        "k": k,
        "silhouette_score": float(silhouette),
        "inertia": float(kmeans.inertia_),
        "labels": labels.tolist(),
        "cluster_centers": kmeans.cluster_centers_.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_std": scaler.scale_.tolist(),
        "k_evaluation": eval_result["k_results"],
        "seed": SEED,
        "group_code": GROUP_CODE,
        "feature_names": FEATURE_COLS,
    }


def save_clustering_artifacts(
    result: Dict[str, Any],
    metrics_path: Path,
    clusters_csv_path: Path,
    plot_path: Path,
    ids: np.ndarray,
):
    """Save clustering metrics, cluster assignments CSV, and cluster plot."""
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "k": result["k"],
                "silhouette_score": result["silhouette_score"],
                "inertia": result["inertia"],
                "k_evaluation": result["k_evaluation"],
                "seed": result["seed"],
                "group_code": result["group_code"],
                "feature_names": result["feature_names"],
            },
            f,
            indent=2,
        )

    import pandas as pd
    clusters_df = pd.DataFrame({"record_id": ids, "cluster_label": result["labels"]})
    clusters_df.to_csv(clusters_csv_path, index=False)

    scaler = StandardScaler()
    scaler.mean_ = np.array(result["scaler_mean"])
    scaler.scale_ = np.array(result["scaler_std"])
    
    # Reconstruct X from saved data for plotting
    from src.data_pipeline import load_and_validate, separate_features_targets
    df = load_and_validate(Path("data/AI_A1_G01.csv"))
    _, X, _, _ = separate_features_targets(df)
    X_scaled = scaler.transform(X)

    pca = PCA(n_components=2, random_state=SEED)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=result["labels"], cmap="tab10", alpha=0.7, s=60)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"K-Means Clustering (k={result['k']}) - PCA Projection")
    plt.colorbar(scatter, label="Cluster")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    np.savez(
        Path("models/clustering_model.npz"),
        cluster_centers=np.array(result["cluster_centers"]),
        scaler_mean=np.array(result["scaler_mean"]),
        scaler_std=np.array(result["scaler_std"]),
        k=result["k"],
    )


def load_clustering_model(model_path: Path):
    """Load clustering model and scaler."""
    data = np.load(model_path)
    return {
        "cluster_centers": data["cluster_centers"],
        "scaler_mean": data["scaler_mean"],
        "scaler_std": data["scaler_std"],
        "k": int(data["k"]),
    }


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


def clustering_interpretation() -> str:
    """Cautious interpretation of clusters."""
    return (
        "Clusters represent groups of collection points with similar operating profiles "
        "based on input features (plot area, rainfall, soil pH, seed quantity, distance, "
        "arrival hour). These are statistical groupings, not verified real-world categories. "
        "Domain experts should validate whether clusters correspond to meaningful operational "
        "segments before making decisions based on cluster membership."
    )


if __name__ == "__main__":
    from src.data_pipeline import load_and_validate, separate_features_targets

    csv_path = Path("data/AI_A1_G01.csv")
    df = load_and_validate(csv_path)
    ids, X, _, _ = separate_features_targets(df)

    np.save("data/X_features.npy", X)

    result = train_clustering(X)
    print(f"Best k: {result['k']}, Silhouette: {result['silhouette_score']:.4f}")
    print(f"Interpretation: {clustering_interpretation()}")

    save_clustering_artifacts(
        result,
        Path("artifacts/clustering_metrics.json"),
        Path("artifacts/clusters.csv"),
        Path("artifacts/cluster_plot.png"),
        ids,
    )
    print("Clustering artifacts saved")