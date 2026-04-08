# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT03 — Exercise 5: Workflow Orchestration and Custom Nodes
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Build an ML pipeline using Kailash WorkflowBuilder — load,
#   preprocess, train, evaluate, persist to DataFlow. Learn the
#   runtime.execute(workflow.build()) pattern.
#
# TASKS:
#   1. Build a Kailash workflow for the ML pipeline
#   2. Define @db.model for evaluation results (DataFlow)
#   3. Execute the workflow with LocalRuntime
#   4. Persist results with db.express
#   5. Define ModelSignature (input/output schema for trained models)
#   6. Query persisted results
# ════════════════════════════════════════════════════════════════════════
"""
import asyncio
import os
import tempfile

import polars as pl
from dotenv import load_dotenv

from kailash.workflow.builder import WorkflowBuilder
from kailash.runtime import LocalRuntime
from dataflow import DataFlow
from kailash_ml import PreprocessingPipeline, ModelVisualizer
from kailash_ml.interop import to_sklearn_input
from kailash_ml.engines.training_pipeline import TrainingPipeline, ModelSpec, EvalSpec

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()


# ── Data Loading ──────────────────────────────────────────────────────

loader = ASCENTDataLoader()
credit = loader.load("ascent03", "sg_credit_scoring.parquet")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Build a Kailash ML workflow
# ══════════════════════════════════════════════════════════════════════
# WorkflowBuilder: nodes, connections, runtime.execute(workflow.build())

# TODO: Create a WorkflowBuilder named "credit_scoring_pipeline"
workflow = ____

# Node 1: Data preprocessing (PythonCodeNode executes arbitrary Python)
# TODO: Add a PythonCodeNode with id "preprocess" containing code that builds
#       a result dict with target/train_size/seed/status keys
workflow.add_node(
    "PythonCodeNode",
    "preprocess",
    {
        "code": ____,
    },
)

# Node 2: Model training step
# TODO: Add a PythonCodeNode "train" with code building result dict
#       (model_class, n_estimators, learning_rate, status) and connections=["preprocess"]
workflow.add_node(
    "PythonCodeNode",
    "train",
    {
        "code": """
result = {
    "model_class": "lightgbm.LGBMClassifier",
    "n_estimators": 500,
    "learning_rate": 0.1,
    "status": "trained"
}
""",
    },
    connections=["preprocess"],
)

# Node 3: Evaluation step
workflow.add_node(
    "PythonCodeNode",
    "evaluate",
    {
        "code": """
result = {
    "metrics": ["accuracy", "f1", "auc_roc", "auc_pr", "log_loss"],
    "status": "evaluated"
}
""",
    },
    connections=["train"],
)

# Build and execute
runtime = LocalRuntime()
print("=== Executing Workflow ===")
# TODO: Execute via runtime.execute(workflow.build())  — MUST call .build()
results, run_id = ____

print(f"Run ID: {run_id}")
print(f"Node results: {list(results.keys())}")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Define @db.model for evaluation results
# ══════════════════════════════════════════════════════════════════════

_db_path = os.path.join(tempfile.gettempdir(), "ascent03_models.db")
# TODO: Create the DataFlow instance pointing at sqlite:///{_db_path}
db = ____


@db.model
class ModelEvaluation:
    """Stores evaluation results for trained models."""

    id: int
    model_name: str
    dataset: str
    accuracy: float
    f1_score: float
    auc_roc: float
    auc_pr: float
    log_loss: float
    train_size: int
    test_size: int
    feature_count: int


@db.model
class ModelArtifact:
    """Stores model metadata and serialisation path."""

    id: int
    model_name: str
    version: int
    artifact_path: str
    is_production: bool = False
    created_by: str = "ascent03"


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Train and evaluate manually (parallel to workflow)
# ══════════════════════════════════════════════════════════════════════

import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
)

pipeline = PreprocessingPipeline()
result = pipeline.setup(
    credit, target="default", seed=42, normalize=False, categorical_encoding="ordinal"
)

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

# TODO: Build LGBMClassifier with n_estimators=500, learning_rate=0.1, max_depth=6,
#       scale_pos_weight=(1 - y_train.mean()) / y_train.mean(),
#       random_state=42, verbose=-1
model = ____
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# TODO: Build the eval_metrics dict with accuracy, f1, auc_roc, auc_pr, log_loss
eval_metrics = ____

print(f"\n=== Manual Evaluation ===")
for metric, value in eval_metrics.items():
    print(f"  {metric}: {value:.4f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Persist results with db.express
# ══════════════════════════════════════════════════════════════════════


async def persist_results():
    """Store evaluation results and model metadata in DataFlow."""
    # TODO: Initialise the DataFlow connection pool
    await ____  # Hint: db.initialize()

    # TODO: Use db.express.create("ModelEvaluation", {...}) to insert the metrics row
    #       Fields: model_name="lgbm_credit_v1", dataset="sg_credit_scoring",
    #       accuracy/f1_score/auc_roc/auc_pr/log_loss from eval_metrics,
    #       train_size, test_size, feature_count from X_train.shape
    await ____

    # Retrieve the persisted record
    evals = await db.express.list("ModelEvaluation", {"model_name": "lgbm_credit_v1"})
    eval_record = evals[0] if evals else {}
    print(f"\nPersisted evaluation: ID={eval_record.get('id', 'N/A')}")

    # Store model artifact metadata
    await db.express.create(
        "ModelArtifact",
        {
            "model_name": "lgbm_credit_v1",
            "version": 1,
            "artifact_path": "models/lgbm_credit_v1.pkl",
            "is_production": False,
            "created_by": "ascent03_ex4",
        },
    )
    artifacts = await db.express.list("ModelArtifact", {"model_name": "lgbm_credit_v1"})
    artifact_record = artifacts[0] if artifacts else {}
    print(f"Persisted artifact: ID={artifact_record.get('id', 'N/A')}")

    return eval_record, artifact_record


eval_record, artifact_record = asyncio.run(persist_results())


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Define ModelSignature
# ══════════════════════════════════════════════════════════════════════
# ModelSignature is the input/output contract for a trained model.

from kailash_ml.types import ModelSignature, FeatureSchema, FeatureField

# TODO: Build a FeatureSchema named "credit_model_input" whose features list is
#       [FeatureField(name=f, dtype="float64") for f in col_info["feature_columns"]]
#       and entity_id_column="application_id"
input_schema = ____

# TODO: Build a ModelSignature with input_schema, output_columns=["default_probability","default_label"],
#       output_dtypes=["float64","int64"], model_type="classifier"
signature = ____

print(f"\n=== ModelSignature ===")
print(f"Input features: {len(signature.input_schema.features)}")
print(f"Output: {signature.output_columns}")
print(f"Model type: {signature.model_type}")
print(f"\nModelSignature is the contract between model and deployment.")
print("InferenceServer (M4) validates inputs against this signature.")


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Query persisted results
# ══════════════════════════════════════════════════════════════════════


async def query_results():
    """Query stored evaluations to compare models."""
    await db.initialize()  # Re-init for new event loop

    # TODO: List all ModelEvaluation rows via db.express.list("ModelEvaluation")
    evals = ____
    print(f"\n=== Persisted Evaluations ({len(evals)}) ===")
    for e in evals:
        print(
            f"  {e['model_name']}: AUC-ROC={float(e['auc_roc']):.4f}, AUC-PR={float(e['auc_pr']):.4f}"
        )

    artifacts = await db.express.list("ModelArtifact")
    print(f"\nModel Artifacts ({len(artifacts)}):")
    for a in artifacts:
        status = "🟢 PRODUCTION" if a["is_production"] else "staging"
        print(f"  {a['model_name']} v{a['version']}: {status}")

    db.close()


asyncio.run(query_results())

print("\n✓ Exercise 5 complete — Kailash workflow orchestration + DataFlow persistence")
print("  Pattern: WorkflowBuilder → runtime.execute(workflow.build()) → db.express")
