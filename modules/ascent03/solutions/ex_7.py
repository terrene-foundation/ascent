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
# (the search API passes the raw DataFrame to train(), not the preprocessed one)
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

search_space = SearchSpace(
    params=[
        ParamDistribution(
            name="n_estimators",
            type="int_uniform",
            low=100,
            high=1000,
        ),
        ParamDistribution(
            name="learning_rate",
            type="log_uniform",
            low=0.01,
            high=0.5,
        ),
        ParamDistribution(
            name="max_depth",
            type="int_uniform",
            low=3,
            high=10,
        ),
        ParamDistribution(
            name="num_leaves",
            type="int_uniform",
            low=15,
            high=127,
        ),
        ParamDistribution(
            name="min_child_samples",
            type="int_uniform",
            low=5,
            high=100,
        ),
        ParamDistribution(
            name="subsample",
            type="uniform",
            low=0.5,
            high=1.0,
        ),
        ParamDistribution(
            name="colsample_bytree",
            type="uniform",
            low=0.5,
            high=1.0,
        ),
        ParamDistribution(
            name="reg_alpha",
            type="log_uniform",
            low=1e-8,
            high=10.0,
        ),
        ParamDistribution(
            name="reg_lambda",
            type="log_uniform",
            low=1e-8,
            high=10.0,
        ),
    ]
)

print("=== Search Space ===")
for p in search_space.params:
    print(f"  {p.name}: {p.type} [{p.low}, {p.high}]")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Run HyperparameterSearch
# ══════════════════════════════════════════════════════════════════════

config = SearchConfig(
    strategy="bayesian",
    metric_to_optimize="auc",  # ROC-AUC (supported by kailash-ml)
    direction="maximize",
    n_trials=20,
    early_stopping_patience=10,
    timeout_seconds=120,
)

base_model_spec = ModelSpec(
    model_class="lightgbm.LGBMClassifier",
    hyperparameters={"random_state": 42, "verbose": -1},
    framework="sklearn",
)

eval_spec = EvalSpec(
    metrics=["auc", "f1", "accuracy"],
    split_strategy="stratified_kfold",
    n_splits=5,
    test_size=0.2,
)


async def run_search():
    _db_path = os.path.join(tempfile.gettempdir(), "ascent03_models.db")
    conn = ConnectionManager(f"sqlite:///{_db_path}")
    await conn.initialize()

    registry = ModelRegistry(conn)

    training_pipeline = TrainingPipeline(feature_store=None, registry=registry)

    search = HyperparameterSearch(pipeline=training_pipeline)

    search_result = await search.search(
        data=credit,
        schema=schema,
        base_model_spec=base_model_spec,
        search_space=search_space,
        config=config,
        eval_spec=eval_spec,
        experiment_name="credit_default_bayesian",
    )

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
best_so_far = np.maximum.accumulate(scores)

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

    best_model = lgb.LGBMClassifier(**best_params)
    best_model.fit(X_train, y_train)

    # Register in ModelRegistry (serialize model to bytes)
    model_bytes = pickle.dumps(best_model)
    best_score = search_result.best_metrics.get("auc", 0)
    model_version = await registry.register_model(
        name="credit_default_lgbm",
        artifact=model_bytes,
        metrics=[MetricSpec(name="auc", value=best_score)],
    )
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
    """
    Promotion is a governance gate:
    - A model cannot reach production without explicit promotion
    - This creates an audit trail: who promoted, when, why
    - EATP concept: model provenance chain
    """

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
        promoted = await registry.promote_model(
            name="credit_default_lgbm",
            version=model_id,
            target_stage="production",
            reason=f"Passed quality gate: AUC-PR={auc_pr:.4f} >= {min_auc_pr}",
        )
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
    models = await registry.list_models()
    print(f"\n=== Model Registry ===")
    for m in models:
        print(
            f"  {m.get('name', '?')} (v{m.get('version', '?')}): "
            f"stage={m.get('stage', '?')}"
        )

    # Get production model
    try:
        prod_model = await registry.get_model("credit_default_lgbm", stage="production")
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
