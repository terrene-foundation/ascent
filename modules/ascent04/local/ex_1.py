# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT04 — Exercise 1: Clustering
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Compare K-means, spectral clustering, HDBSCAN, and GMM on
#   e-commerce customer data. Use AutoMLEngine for automated comparison.
#
# TASKS:
#   1. Load and prepare e-commerce customer features
#   2. Run K-means, spectral, HDBSCAN, GMM clustering
#   3. Evaluate with silhouette, Calinski-Harabasz, Davies-Bouldin
#   4. Use AutoMLEngine with agent=True double opt-in
#   5. Analyse cluster profiles
#   6. Visualise with ModelVisualizer
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

import numpy as np
import polars as pl
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.preprocessing import StandardScaler

from kailash_ml import PreprocessingPipeline, ModelVisualizer, DataExplorer
from kailash_ml.interop import to_sklearn_input

from shared import ASCENTDataLoader

try:
    import hdbscan
except ImportError:
    hdbscan = None


# ── Data Loading ──────────────────────────────────────────────────────

loader = ASCENTDataLoader()
customers = loader.load("ascent04", "ecommerce_customers.parquet")

print(f"=== E-commerce Customer Data ===")
print(f"Shape: {customers.shape}")
print(f"Columns: {customers.columns}")

# Select numeric features for clustering
feature_cols = [
    c
    for c, d in zip(customers.columns, customers.dtypes)
    if d in (pl.Float64, pl.Float32, pl.Int64, pl.Int32) and c not in ("customer_id",)
]

print(f"Clustering features ({len(feature_cols)}): {feature_cols}")

# Prepare data
X, _, col_info = to_sklearn_input(
    customers.drop_nulls(subset=feature_cols),
    feature_columns=feature_cols,
)

# TODO: Standardise X (clustering needs zero mean / unit variance)
# Hint: StandardScaler().fit_transform(X)
scaler = ____
X_scaled = ____
print(f"Samples: {X_scaled.shape[0]:,}, Features: {X_scaled.shape[1]}")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: K-means clustering
# ══════════════════════════════════════════════════════════════════════

# Elbow method: test K=2..10
inertias = []
sil_scores = []
for k in range(2, 11):
    # TODO: Create KMeans(n_clusters=k, random_state=42, n_init=10) and fit_predict
    km = ____
    labels = ____
    # TODO: Append the model's inertia_ attribute
    inertias.append(____)
    # TODO: Append silhouette_score(X_scaled, labels)
    sil_scores.append(____)

best_k = range(2, 11)[np.argmax(sil_scores)]
print(f"\n=== K-means ===")
print(f"Best K by silhouette: {best_k} (score={max(sil_scores):.4f})")

# TODO: Fit final K-means model with best_k
km_best = ____
km_labels = ____


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Spectral, HDBSCAN, GMM
# ══════════════════════════════════════════════════════════════════════

# Spectral clustering — O(n³), so sample first 10K rows
n_spectral = best_k
# TODO: SpectralClustering(n_clusters=n_spectral, random_state=42, affinity="rbf")
spectral = ____
# TODO: fit_predict on X_scaled[:10_000]
spectral_labels = ____

# HDBSCAN — density-based, finds variable-shaped clusters
if hdbscan is not None:
    # TODO: hdbscan.HDBSCAN(min_cluster_size=50, min_samples=10)
    hdb = ____
    # TODO: fit_predict on full X_scaled
    hdbscan_labels = ____
    n_hdbscan_clusters = len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0)
    noise_pct = (hdbscan_labels == -1).mean()
    print(f"\nHDBSCAN: {n_hdbscan_clusters} clusters, {noise_pct:.1%} noise")
else:
    hdbscan_labels = km_labels  # Fallback
    print("\nHDBSCAN not installed, using K-means labels as fallback")

# Gaussian Mixture Model — soft assignments
# TODO: GaussianMixture(n_components=best_k, random_state=42, covariance_type="full")
gmm = ____
# TODO: gmm.fit_predict(X_scaled)
gmm_labels = ____
# TODO: gmm.predict_proba(X_scaled) — soft assignment probabilities
gmm_probs = ____

print(f"\nGMM: {best_k} components, BIC={gmm.bic(X_scaled):.0f}")
print(f"  Max assignment probability: {gmm_probs.max(axis=1).mean():.3f} (avg)")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Evaluate all clustering methods
# ══════════════════════════════════════════════════════════════════════

results = {}
all_labels = {
    "K-means": km_labels,
    "Spectral": spectral_labels,
    "HDBSCAN": hdbscan_labels,
    "GMM": gmm_labels,
}

for name, labels in all_labels.items():
    # Filter noise points for HDBSCAN
    valid = labels != -1
    if valid.sum() < 2 or len(set(labels[valid])) < 2:
        continue

    data = X_scaled[: len(labels)]
    if not valid.all():
        data_clean = data[valid]
        labels_clean = labels[valid]
    else:
        data_clean = data
        labels_clean = labels

    # TODO: Compute the three clustering quality metrics
    results[name] = {
        "n_clusters": len(set(labels_clean)),
        "silhouette": ____,  # Hint: silhouette_score(data_clean, labels_clean)
        "calinski_harabasz": ____,  # Hint: calinski_harabasz_score(data_clean, labels_clean)
        "davies_bouldin": ____,  # Hint: davies_bouldin_score(data_clean, labels_clean)
    }

print(f"\n=== Clustering Comparison ===")
print(f"{'Method':<12} {'K':>4} {'Silhouette':>12} {'CH':>12} {'DB':>8}")
print("─" * 52)
for name, r in results.items():
    print(
        f"{name:<12} {r['n_clusters']:>4} {r['silhouette']:>12.4f} "
        f"{r['calinski_harabasz']:>12.0f} {r['davies_bouldin']:>8.4f}"
    )


# ══════════════════════════════════════════════════════════════════════
# TASK 4: AutoMLEngine comparison (agent=True double opt-in)
# ══════════════════════════════════════════════════════════════════════

from kailash_ml.engines.automl_engine import AutoMLEngine, AutoMLConfig


async def automl_comparison():
    """Use AutoMLEngine for automated clustering comparison."""

    # TODO: Build an AutoMLConfig for clustering optimisation
    # task_type="clustering", metric_to_optimize="silhouette", direction="maximize"
    # search_strategy="random", search_n_trials=20
    # agent=False (double opt-in), max_llm_cost_usd=1.0 (hard cap)
    config = ____

    print(f"\n=== AutoMLEngine Config ===")
    print(f"Task: {config.task_type}")
    print(f"Metric: {config.metric_to_optimize}")
    print(f"Agent enabled: {config.agent}")
    print(f"LLM cost budget: ${config.max_llm_cost_usd}")
    print("\nNote: agent=True requires BOTH the flag AND kailash-ml[agents] installed.")
    print("This is the 'double opt-in' pattern — governance by design.")

    return config


config = asyncio.run(automl_comparison())


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Cluster profile analysis
# ══════════════════════════════════════════════════════════════════════

best_method = max(results.items(), key=lambda x: x[1]["silhouette"])
print(
    f"\nBest method: {best_method[0]} (silhouette={best_method[1]['silhouette']:.4f})"
)

best_labels = all_labels[best_method[0]]

clean_df = customers.drop_nulls(subset=feature_cols)
profiling_df = clean_df.head(len(best_labels))
# TODO: Attach the cluster labels as a new column called "cluster"
# Hint: profiling_df.with_columns(pl.Series("cluster", best_labels))
clustered = ____

print(f"\n=== Cluster Profiles ===")
for cluster_id in sorted(clustered["cluster"].unique().to_list()):
    if cluster_id == -1:
        continue
    # TODO: Filter clustered to rows where "cluster" == cluster_id
    subset = ____
    print(f"\nCluster {cluster_id} (n={subset.height:,}):")
    for col in feature_cols[:6]:
        # TODO: Mean for this cluster vs overall mean
        mean_val = ____  # Hint: subset[col].mean()
        overall_mean = ____  # Hint: clustered[col].mean()
        diff = (mean_val - overall_mean) / overall_mean * 100 if overall_mean else 0
        indicator = "▲" if diff > 10 else "▼" if diff < -10 else "─"
        print(f"  {col:<25} {mean_val:>10.2f} ({diff:+.1f}% vs avg) {indicator}")


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Visualise
# ══════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()

# TODO: Build comparison chart with viz.metric_comparison(results)
fig = ____
fig.update_layout(title="Clustering Method Comparison")
fig.write_html("ex1_clustering_comparison.html")
print("\nSaved: ex1_clustering_comparison.html")

elbow_data = {"Silhouette": sil_scores}
# TODO: Build elbow curve with viz.training_history(elbow_data, x_label="K")
fig_elbow = ____
fig_elbow.update_layout(title="K-means: Silhouette Score vs K")
fig_elbow.write_html("ex1_elbow.html")
print("Saved: ex1_elbow.html")

print("\n✓ Exercise 1 complete — clustering comparison with AutoMLEngine")
