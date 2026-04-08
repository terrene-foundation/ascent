# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT03 — Exercise 6: DataFlow and Persistence
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Persist ML evaluation results to a DataFlow database using
#   @db.model, db.express CRUD, and async context. Query and compare
#   model runs to demonstrate reproducible experiment tracking.
#
# TASKS:
#   1. Design @db.model schema for ML evaluation results
#   2. Train multiple model variants to generate experiment data
#   3. Persist all results with db.express.create
#   4. Query, filter, and compare runs with db.express.list
#   5. Update records: mark the best model as production candidate
#   6. Explore async patterns: async with, context managers, lifecycle
# ════════════════════════════════════════════════════════════════════════
"""
import asyncio
import json
import os
import tempfile
from datetime import datetime

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)

from dataflow import DataFlow
from kailash_ml import PreprocessingPipeline, ModelVisualizer
from kailash_ml.interop import to_sklearn_input

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()


# ── Data Loading ──────────────────────────────────────────────────────

loader = ASCENTDataLoader()
credit = loader.load("ascent03", "sg_credit_scoring.parquet")

# TODO: Instantiate PreprocessingPipeline and call setup() with credit, target="default",
#       seed=42, normalize=False, categorical_encoding="ordinal", imputation_strategy="median"
pipeline = ____
result = ____

X_train, y_train, col_info = to_sklearn_input(
    result.train_data,
    feature_columns=[c for c in result.train_data.columns if c != "default"],
    target_column="default",
)
X_test, y_test, _ = to_sklearn_input(
    result.test_data,
    feature_columns=[c for c in result.test_data.columns if c != "default"],
    target_column="default",
)
feature_names = col_info["feature_columns"]

print(f"=== Singapore Credit Data ===")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Default rate (train): {y_train.mean():.2%}")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Design @db.model schema
# ══════════════════════════════════════════════════════════════════════
# DataFlow uses declarative models — define once, get CRUD for free.

_db_path = os.path.join(tempfile.gettempdir(), "ascent03_ex6_results.db")
DB_URL = f"sqlite:///{_db_path}"

# TODO: Create the DataFlow instance pointing at DB_URL
db = ____


@db.model
class ModelRun:
    """A single training run with its configuration and metrics."""

    id: int
    run_name: str
    model_family: str  # "lgbm", "xgboost", etc.
    dataset: str
    # Hyperparameters (serialised to JSON string)
    hyperparams_json: str = "{}"
    # Evaluation metrics
    accuracy: float = 0.0
    f1_score: float = 0.0
    auc_roc: float = 0.0
    auc_pr: float = 0.0
    log_loss_val: float = 0.0
    brier_score: float = 0.0
    # Metadata
    train_samples: int = 0
    test_samples: int = 0
    feature_count: int = 0
    is_production_candidate: bool = False
    notes: str = ""


@db.model
class FeatureImportance:
    """Top feature importances for a model run — linked by run_id."""

    id: int
    run_id: int  # Foreign key (by convention)
    feature_name: str
    importance: float
    rank: int


print("\n=== DataFlow Schema Defined ===")
print("ModelRun: evaluation results + hyperparams per training run")
print("FeatureImportance: top features per run (linked by run_id)")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Train multiple model variants
# ══════════════════════════════════════════════════════════════════════

model_configs = [
    {
        "run_name": "lgbm_default",
        "hyperparams": {
            "n_estimators": 300,
            "learning_rate": 0.1,
            "max_depth": 6,
            "scale_pos_weight": (1 - y_train.mean()) / y_train.mean(),
        },
        "notes": "Baseline with cost-sensitive weight",
    },
    {
        "run_name": "lgbm_shallow",
        "hyperparams": {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 3,
            "scale_pos_weight": (1 - y_train.mean()) / y_train.mean(),
        },
        "notes": "Shallow trees — higher bias, lower variance",
    },
    {
        "run_name": "lgbm_deep",
        "hyperparams": {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 8,
            "num_leaves": 63,
            "scale_pos_weight": (1 - y_train.mean()) / y_train.mean(),
        },
        "notes": "Deeper trees — lower bias, higher variance",
    },
    {
        "run_name": "lgbm_regularised",
        "hyperparams": {
            "n_estimators": 400,
            "learning_rate": 0.08,
            "max_depth": 6,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "min_child_samples": 30,
            "scale_pos_weight": (1 - y_train.mean()) / y_train.mean(),
        },
        "notes": "L1+L2 regularisation applied",
    },
]

trained_runs = []

for config in model_configs:
    print(f"\nTraining {config['run_name']}...")
    # TODO: Build LGBMClassifier(**config["hyperparams"], random_state=42, verbose=-1)
    model = ____
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # TODO: Build the metrics dict (keys: accuracy, f1_score, auc_roc, auc_pr,
    #       log_loss_val, brier_score) using sklearn's metric functions
    metrics = ____

    # Feature importances
    importances = model.feature_importances_
    top_features = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True,
    )[:10]

    trained_runs.append(
        {
            "config": config,
            "metrics": metrics,
            "top_features": top_features,
            "model": model,
        }
    )

    print(f"  AUC-ROC: {metrics['auc_roc']:.4f}  AUC-PR: {metrics['auc_pr']:.4f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Persist results with db.express.create
# ══════════════════════════════════════════════════════════════════════

run_ids = []


async def persist_all_runs():
    """Persist all training runs and feature importances to DataFlow."""
    # TODO: Initialise DataFlow
    await ____  # Hint: db.initialize()

    print("\n=== Persisting Results ===")
    for run_data in trained_runs:
        config = run_data["config"]
        metrics = run_data["metrics"]

        # TODO: Insert a ModelRun row with run_name, model_family="lightgbm",
        #       dataset="sg_credit_scoring", hyperparams_json=json.dumps(config["hyperparams"]),
        #       all metric fields, train_samples/test_samples/feature_count from X_train.shape,
        #       is_production_candidate=False, notes=config["notes"]
        await ____

        # Retrieve the created record to get its auto-generated id
        runs = await db.express.list("ModelRun", {"run_name": config["run_name"]})
        run_id = runs[-1]["id"] if runs else 0
        run_ids.append(run_id)
        print(f"  Persisted {config['run_name']}: ID={run_id}")

        # Create FeatureImportance records for this run
        for rank, (feat_name, importance) in enumerate(
            run_data["top_features"], start=1
        ):
            await db.express.create(
                "FeatureImportance",
                {
                    "run_id": run_id,
                    "feature_name": feat_name,
                    "importance": float(importance),
                    "rank": rank,
                },
            )

    print(
        f"\nPersisted {len(trained_runs)} runs, {len(trained_runs) * 10} feature records"
    )


asyncio.run(persist_all_runs())


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Query, filter, and compare runs
# ══════════════════════════════════════════════════════════════════════


async def query_and_compare():
    """Retrieve and compare all stored model runs."""
    await db.initialize()  # Re-init for new event loop

    # TODO: List all runs via db.express.list("ModelRun")
    all_runs = ____
    print(f"\n=== All Stored Runs ({len(all_runs)}) ===")
    print(f"{'Run Name':<25} {'AUC-ROC':>10} {'AUC-PR':>10} {'Brier':>8} {'Notes':<35}")
    print("─" * 95)
    for run in sorted(all_runs, key=lambda r: float(r["auc_pr"]), reverse=True):
        print(
            f"{run['run_name']:<25} {float(run['auc_roc']):>10.4f} {float(run['auc_pr']):>10.4f} "
            f"{float(run['brier_score']):>8.4f} {run['notes']:<35}"
        )

    # Filter: only runs with AUC-PR above a threshold
    high_quality = [r for r in all_runs if float(r["auc_pr"]) > 0.30]
    print(f"\nRuns with AUC-PR > 0.30: {len(high_quality)}")

    # TODO: Use db.express.list("ModelRun", filter_dict) to fetch only the
    #       regularised runs (model_family="lightgbm", notes="L1+L2 regularisation applied")
    regularised = await ____
    print(f"Regularised runs: {len(regularised)}")

    # Get a specific run by ID
    if run_ids:
        first_run = await db.express.read("ModelRun", str(run_ids[0]))
        if first_run:
            print(f"\nFirst run retrieved by ID={run_ids[0]}:")
            hyperparams = json.loads(first_run["hyperparams_json"])
            print(f"  Hyperparams: {hyperparams}")
        else:
            print(f"\nFirst run ID={run_ids[0]} not found (stale across event loops)")

    return all_runs


all_runs = asyncio.run(query_and_compare())


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Update records — mark the best model
# ══════════════════════════════════════════════════════════════════════


async def promote_best():
    """Find the best run by AUC-PR and mark it as production candidate."""
    await db.initialize()  # Re-init for new event loop
    all_runs = await db.express.list("ModelRun")

    # TODO: Find best by AUC-PR using max(...) with key=lambda r: float(r["auc_pr"])
    best = ____
    print(f"\n=== Promoting Best Run ===")
    print(f"Best run: {best['run_name']} (AUC-PR={float(best['auc_pr']):.4f})")

    # TODO: Update the best row via db.express.update("ModelRun", str(best["id"]), {...})
    #       Set is_production_candidate=True and append a "Promoted YYYY-MM-DD" suffix to notes
    updated = await ____
    print(f"Updated ID={best['id']}: is_production_candidate = True")

    # Verify the update
    confirmed = await db.express.read("ModelRun", str(best["id"]))
    if confirmed:
        print(
            f"Confirmed: {confirmed['run_name']} → production_candidate={confirmed['is_production_candidate']}"
        )
    else:
        print(f"Confirmed: update applied (read-back unavailable)")

    # List only production candidates
    candidates = await db.express.list("ModelRun", {"is_production_candidate": True})
    print(f"\nProduction candidates: {[c['run_name'] for c in candidates]}")


asyncio.run(promote_best())


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Async patterns — connection lifecycle
# ══════════════════════════════════════════════════════════════════════
# 1. await db.initialize()         — opens connection pool
# 2. await db.express.create(...)  — single insert
# 3. await db.express.list(...)    — query with optional filter
# 4. await db.express.read(...)    — retrieve by primary key
# 5. await db.express.update(...)  — partial update
# 6. await db.express.delete(...)  — delete by primary key
# 7. await db.close()              — release connection pool


async def demonstrate_async_context():
    """Show async context manager pattern for connection management."""
    await db.initialize()  # Re-init for new event loop

    print("\n=== Async Connection Lifecycle ===")
    print("1. db.initialize()  — open pool (done once per process)")
    print("2. db.express.*     — use CRUD operations")
    print("3. db.close()       — release pool (done at shutdown)")

    print("\nPattern: async with db.connect() as conn:")
    print("  # automatically released when block exits")

    # Feature importance retrieval — demonstrate join pattern
    if run_ids:
        print("\n=== Top Feature per Run ===")
        for run_id in run_ids:
            # TODO: List FeatureImportance rows where run_id matches and rank == 1
            fi_records = await ____
            run = await db.express.read("ModelRun", str(run_id))
            if fi_records:
                top_feat = fi_records[0]
                print(
                    f"  {run['run_name']}: top feature = {top_feat['feature_name']} "
                    f"(importance={float(top_feat['importance']):.2f})"
                )

    # Clean up connection pool
    db.close()
    print("\nConnection pool closed.")


asyncio.run(demonstrate_async_context())


# ══════════════════════════════════════════════════════════════════════
# Final comparison visualisation
# ══════════════════════════════════════════════════════════════════════

metrics_by_run = {
    r["run_name"]: {
        "AUC_ROC": float(r["auc_roc"]),
        "AUC_PR": float(r["auc_pr"]),
        "Brier_Score": float(r["brier_score"]),
    }
    for r in all_runs
}

viz = ModelVisualizer()
# TODO: Build viz.metric_comparison(metrics_by_run)
fig = ____
fig.update_layout(title="Model Run Comparison — Stored in DataFlow")
fig.write_html("ex6_run_comparison.html")
print("\nSaved: ex6_run_comparison.html")

print("\n✓ Exercise 6 complete — DataFlow persistence for ML experiments")
print("  Patterns learned:")
print("  • @db.model = declarative schema → auto CRUD")
print("  • db.express.create/list/get/update = zero-boilerplate operations")
print("  • async/await = non-blocking I/O for concurrent ML pipelines")
