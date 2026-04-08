# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT03 — Exercise 7: Model Registry and Hyperparameter Search
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Use HyperparameterSearch for Bayesian optimization, then
#   register the best model in ModelRegistry with staging → production
#   promotion as a governance gate.
#
# TASKS:
#   1. Define search space with SearchSpace and ParamDistribution
#   2. Run HyperparameterSearch with Bayesian optimization
#   3. Analyse search results and convergence
#   4. Register best model in ModelRegistry
#   5. Promote from staging to production (governance gate)
#   6. Retrieve and compare model versions
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os
import pickle
import tempfile

import lightgbm as lgb
import numpy as np
import polars as pl
from kailash.db.connection import ConnectionManager
from kailash_ml import PreprocessingPipeline, ModelVisualizer
from kailash_ml.interop import to_sklearn_input
from kailash_ml.engines.hyperparameter_search import (
    HyperparameterSearch,
    SearchSpace,
    SearchConfig,
    ParamDistribution,
)
from kailash_ml.engines.model_registry import ModelRegistry
from kailash_ml.engines.training_pipeline import TrainingPipeline, ModelSpec, EvalSpec
from kailash_ml.types import (
    FeatureSchema,
    FeatureField,
    MetricSpec,
    ModelSignature,
)
from sklearn.metrics import roc_auc_score, average_precision_score

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()


# ── Data Loading ──────────────────────────────────────────────────────

loader = ASCENTDataLoader()
credit = loader.load("ascent03", "sg_credit_scoring.parquet")

pipeline = PreprocessingPipeline()
result = pipeline.setup(
    credit, target="default", seed=42, normalize=False, categorical_encoding="ordinal"
)

# Exclude entity ID and target from feature columns
non_feature_cols = {"default", "customer_id"}
feature_cols_train = [c for c in result.train_data.columns if c not in non_feature_cols]

X_train, y_train, col_info = to_sklearn_input(
    result.train_data,
    feature_columns=feature_cols_train,
    target_column="default",
)
X_test, y_test, _ = to_sklearn_input(
    result.test_data,
    feature_columns=feature_cols_train,
    target_column="default",
)
feature_names = col_info["feature_columns"]

# Cast string columns to Categorical so TrainingPipeline.train() can handle them
string_cols = [
    c
    for c in credit.columns
    if credit[c].dtype in (pl.Utf8, pl.String) and c != "customer_id"
]
credit = credit.with_columns([pl.col(c).cast(pl.Categorical) for c in string_cols])

# Build FeatureSchema for the search API (exclude customer_id — it's an entity ID, not a feature)
schema = FeatureSchema(
    name="credit_scoring_features",
    features=[FeatureField(name=f, dtype="float64") for f in feature_names],
    entity_id_column="customer_id",
)


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Define search space
# ══════════════════════════════════════════════════════════════════════

# TODO: Build a SearchSpace with ParamDistribution entries for each
#       hyperparameter. Required params and ranges:
#         n_estimators       int_uniform   [100, 1000]
#         learning_rate      log_uniform   [0.01, 0.5]
#         max_depth          int_uniform   [3, 10]
#         num_leaves         int_uniform   [15, 127]
#         min_child_samples  int_uniform   [5, 100]
#         subsample          uniform       [0.5, 1.0]
#         colsample_bytree   uniform       [0.5, 1.0]
#         reg_alpha          log_uniform   [1e-8, 10.0]
#         reg_lambda         log_uniform   [1e-8, 10.0]
search_space = ____

print("=== Search Space ===")
for p in search_space.params:
    print(f"  {p.name}: {p.type} [{p.low}, {p.high}]")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Run HyperparameterSearch
# ══════════════════════════════════════════════════════════════════════

# TODO: Build SearchConfig with strategy="bayesian", metric_to_optimize="auc",
#       direction="maximize", n_trials=20, early_stopping_patience=10,
#       timeout_seconds=120
config = ____

# TODO: Build a base ModelSpec for lightgbm.LGBMClassifier with
#       hyperparameters={"random_state": 42, "verbose": -1}, framework="sklearn"
base_model_spec = ____

# TODO: Build EvalSpec with metrics=["auc","f1","accuracy"],
#       split_strategy="stratified_kfold", n_splits=5, test_size=0.2
eval_spec = ____


async def run_search():
    _db_path = os.path.join(tempfile.gettempdir(), "ascent03_models.db")
    conn = ConnectionManager(f"sqlite:///{_db_path}")
    await conn.initialize()

    # TODO: Create a ModelRegistry over conn
    registry = ____

    # TODO: Create a TrainingPipeline with feature_store=None and the registry
    training_pipeline = ____

    # TODO: Create the HyperparameterSearch wrapper around the training pipeline
    search = ____

    # TODO: Run the Bayesian search via search.search(...) with all the
    #       data/schema/spec/config arguments and experiment_name="credit_default_bayesian"
    search_result = await ____

    print(f"\n=== Search Results ===")
    print(f"Best params:")
    for k, v in search_result.best_params.items():
        print(f"  {k}: {v}")
    print(f"Best metrics: {search_result.best_metrics}")
    print(f"Best trial: #{search_result.best_trial_number}")
    print(f"Total trials: {len(search_result.all_trials)}")
    print(f"Time: {search_result.total_time_seconds:.1f}s")

    # Trial history
    print(f"\nTop 5 trials:")
    sorted_trials = sorted(
        search_result.all_trials,
        key=lambda t: t.metrics.get("auc", 0),
        reverse=True,
    )
    for i, trial in enumerate(sorted_trials[:5]):
        score = trial.metrics.get("auc", 0)
        lr = trial.params.get("learning_rate", "?")
        depth = trial.params.get("max_depth", "?")
        lr_str = f"{lr:.4f}" if isinstance(lr, float) else str(lr)
        print(f"  #{i+1}: score={score:.4f}, lr={lr_str}, depth={depth}")

    return search_result, conn, registry


search_result, conn, registry = asyncio.run(run_search())


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Analyse convergence
# ══════════════════════════════════════════════════════════════════════

# Plot optimisation convergence
scores = [t.metrics.get("auc", 0) for t in search_result.all_trials]
# TODO: Compute the running maximum across trials
best_so_far = ____  # Hint: np.maximum.accumulate(scores)

viz = ModelVisualizer()
fig = viz.training_history(
    {"Best Score": best_so_far.tolist()},
    x_label="Trial",
)
fig.update_layout(title="Hyperparameter Search Convergence")
fig.write_html("ex7_search_convergence.html")
print("\nSaved: ex7_search_convergence.html")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Register best model in ModelRegistry
# ══════════════════════════════════════════════════════════════════════


async def register_model():
    # Train final model with best params
    best_params = {**search_result.best_params}
    best_params["random_state"] = 42
    best_params["verbose"] = -1

    # TODO: Build LGBMClassifier(**best_params) and fit on (X_train, y_train)
    best_model = ____
    best_model.fit(X_train, y_train)

    # Serialize the model to bytes
    model_bytes = pickle.dumps(best_model)
    best_score = search_result.best_metrics.get("auc", 0)

    # TODO: Register the model in the registry via registry.register_model(
    #         name="credit_default_lgbm", artifact=model_bytes,
    #         metrics=[MetricSpec(name="auc", value=best_score)])
    model_version = await ____
    model_id = model_version.version

    print(f"\n=== Model Registered ===")
    print(f"Model: {model_version.name} v{model_id}")
    print(f"Stage: {model_version.stage} (not yet production)")

    return model_id, best_model


model_id, best_model = asyncio.run(register_model())


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Promote staging → production (governance gate)
# ══════════════════════════════════════════════════════════════════════


async def promote_model():
    """Promotion is a governance gate: explicit, audited, threshold-checked."""

    # Verify model quality before promotion
    y_proba = best_model.predict_proba(X_test)[:, 1]
    auc_pr = average_precision_score(y_test, y_proba)
    auc_roc = roc_auc_score(y_test, y_proba)

    print(f"\n=== Pre-Promotion Validation ===")
    print(f"Test AUC-PR:  {auc_pr:.4f}")
    print(f"Test AUC-ROC: {auc_roc:.4f}")

    # Quality gate: must exceed thresholds
    min_auc_pr = 0.30  # Reasonable for 12% default rate
    if auc_pr >= min_auc_pr:
        # TODO: Promote via registry.promote_model(name="credit_default_lgbm",
        #       version=model_id, target_stage="production",
        #       reason=f"Passed quality gate: AUC-PR={auc_pr:.4f} >= {min_auc_pr}")
        promoted = await ____
        print(f"Model promoted to PRODUCTION (stage={promoted.stage})")
        print(f"  Reason: AUC-PR={auc_pr:.4f} >= {min_auc_pr} threshold")
    else:
        print(f"Model REJECTED: AUC-PR={auc_pr:.4f} < {min_auc_pr} threshold")

    return auc_pr, auc_roc


auc_pr, auc_roc = asyncio.run(promote_model())


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Query and compare model versions
# ══════════════════════════════════════════════════════════════════════


async def compare_versions():
    """List all registered models and their stages."""
    # TODO: List all registered models via registry.list_models()
    models = await ____
    print(f"\n=== Model Registry ===")
    for m in models:
        print(
            f"  {m.get('name', '?')} (v{m.get('version', '?')}): "
            f"stage={m.get('stage', '?')}"
        )

    # Get production model
    try:
        # TODO: Use registry.get_model("credit_default_lgbm", stage="production")
        prod_model = await ____
        print(f"\nProduction model: {prod_model.name} v{prod_model.version}")
        print(f"  Stage: {prod_model.stage}")
        for metric in prod_model.metrics:
            print(f"  {metric.name}: {metric.value:.4f}")
    except Exception:
        print("\nNo production model found")

    # List all versions
    versions = await registry.get_model_versions("credit_default_lgbm")
    print(f"\nAll versions ({len(versions)}):")
    for v in versions:
        print(f"  v{v.version}: stage={v.stage}")

    await conn.close()


asyncio.run(compare_versions())

print("\n✓ Exercise 7 complete — HyperparameterSearch + ModelRegistry promotion")
print("  Key concept: staging → production promotion as governance gate")
