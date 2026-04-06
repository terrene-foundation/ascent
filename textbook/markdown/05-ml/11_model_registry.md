# Chapter 11: ModelRegistry

## Overview

`ModelRegistry` manages versioned model artifacts with a full lifecycle: staging, shadow, production, and archived. It stores serialized model bytes alongside metrics and signatures, supports version comparison, and provides MLflow-compatible export/import. This chapter covers registration, version management, lifecycle promotion, artifact retrieval, comparison, serialization, and edge cases.

## Prerequisites

- Python 3.10+ installed
- Kailash ML installed (`pip install kailash-ml`)
- Completed [Chapter 10: EnsembleEngine](10_ensemble_engine.md)
- Understanding of ConnectionManager from [Core Chapter 10](../00-core/10_connection_manager.md)
- Familiarity with sklearn model training and `pickle` serialization

## Concepts

### Concept 1: Model Versions and Stages

Every registered model gets an auto-incrementing version number and starts at the `"staging"` stage. The lifecycle stages are: `staging` (newly registered), `shadow` (receiving production traffic for comparison), `production` (serving live predictions), and `archived` (retired). Promoting a new version to production automatically archives the current production version.

- **What**: A versioning and lifecycle system for trained model artifacts
- **Why**: Production ML requires controlled rollouts -- you do not swap models without shadow testing
- **How**: `register_model()` creates a new version at staging; `promote_model()` transitions between stages
- **When**: Use for any model that serves predictions -- from internal analytics to customer-facing APIs

### Concept 2: Artifact Storage

`register_model()` takes `artifact=bytes`, not a model object. You serialize the model (e.g., with `pickle.dumps()`) before registering. Artifacts are stored via a pluggable `ArtifactStore` -- `LocalFileArtifactStore` for local development, with cloud backends available for production.

- **What**: Byte-level storage of serialized model artifacts
- **Why**: Byte storage is framework-agnostic -- sklearn, PyTorch, XGBoost models all reduce to bytes
- **How**: `pickle.dumps(model)` produces bytes; `registry.load_artifact()` returns bytes for `pickle.loads()`
- **When**: Always serialize before registration; deserialize after loading

### Concept 3: Metrics and Signatures

Each model version carries `MetricSpec` values (name/value pairs like accuracy, F1) and an optional `ModelSignature` that documents the input schema, output columns, and model type. These enable version comparison and serve as documentation for downstream consumers.

- **What**: Structured metadata attached to each model version
- **Why**: Metrics enable automated comparison; signatures document the model's data contract
- **How**: Pass `metrics=[MetricSpec(name="accuracy", value=0.92)]` and `signature=ModelSignature(...)` to `register_model()`
- **When**: Always attach metrics for any model you plan to compare or promote

### Concept 4: Version Comparison and MLflow Export

`compare()` computes metric deltas between two versions and identifies the better one. `export_mlflow()` writes MLmodel metadata and the pickled artifact in MLflow-compatible format for interoperability. `import_mlflow()` reads it back.

- **What**: Built-in comparison and interoperability features
- **Why**: Comparison automates the "is v2 better than v1?" decision; MLflow export enables tool integration
- **How**: `compare("model", 1, 2)` returns a dict with `deltas` and `better_version`; `export_mlflow()` writes to disk
- **When**: Before promotion decisions (compare) and when integrating with MLflow-based tooling (export)

## Key API

| Method                 | Parameters                                                 | Returns              | Description                               |
| ---------------------- | ---------------------------------------------------------- | -------------------- | ----------------------------------------- |
| `ModelRegistry()`      | `conn: ConnectionManager`, `artifact_store: ArtifactStore` | `ModelRegistry`      | Create with database and artifact storage |
| `register_model()`     | `name`, `artifact: bytes`, `metrics`, `signature`          | `ModelVersion`       | Register a new version at staging         |
| `get_model()`          | `name`, `version=None`, `stage=None`                       | `ModelVersion`       | Retrieve by name, version, or stage       |
| `list_models()`        | --                                                         | `list[dict]`         | List all registered model names           |
| `get_model_versions()` | `name`                                                     | `list[ModelVersion]` | All versions (newest first)               |
| `promote_model()`      | `name`, `version`, `target_stage`, `reason=None`           | `ModelVersion`       | Transition to a new lifecycle stage       |
| `compare()`            | `name`, `version_a`, `version_b`                           | `dict`               | Metric deltas and better version          |
| `load_artifact()`      | `name`, `version`                                          | `bytes`              | Retrieve serialized model bytes           |
| `export_mlflow()`      | `name`, `version`, `output_dir`                            | `Path`               | Export in MLflow format                   |
| `import_mlflow()`      | `path`                                                     | `ModelVersion`       | Import from MLflow format                 |

## Code Walkthrough

```python
from __future__ import annotations

import asyncio
import pickle
import tempfile

from sklearn.ensemble import RandomForestClassifier

from kailash.db.connection import ConnectionManager
from kailash_ml import FeatureField, FeatureSchema, MetricSpec, ModelSignature
from kailash_ml.engines.model_registry import (
    LocalFileArtifactStore,
    ModelNotFoundError,
    ModelRegistry,
    ModelVersion,
)


async def main() -> None:
    # ── 1. Set up ConnectionManager + ModelRegistry ─────────────────

    conn = ConnectionManager("sqlite:///:memory:")
    await conn.initialize()

    with tempfile.TemporaryDirectory() as artifact_dir:
        artifact_store = LocalFileArtifactStore(artifact_dir)
        registry = ModelRegistry(conn, artifact_store)

        # ── 2. Train and serialize a model ──────────────────────────
        # register_model() takes artifact=bytes, NOT the model object.

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            [0, 1, 0, 1],
        )
        artifact_bytes = pickle.dumps(model)

        # ── 3. Define metrics and signature ─────────────────────────

        metrics = [
            MetricSpec(name="accuracy", value=0.92),
            MetricSpec(name="f1", value=0.89),
        ]

        signature = ModelSignature(
            input_schema=FeatureSchema(
                name="churn_model_input",
                features=[
                    FeatureField(name="feature_a", dtype="float64"),
                    FeatureField(name="feature_b", dtype="float64"),
                ],
                entity_id_column="entity_id",
            ),
            output_columns=["prediction"],
            output_dtypes=["float64"],
            model_type="classifier",
        )

        # ── 4. register_model() — version 1 at STAGING ─────────────

        mv1 = await registry.register_model(
            "churn_model",
            artifact_bytes,
            metrics=metrics,
            signature=signature,
        )

        assert mv1.version == 1
        assert mv1.stage == "staging"
        assert len(mv1.metrics) == 2

        # ── 5. register_model() — version 2 auto-increments ────────

        better_model = RandomForestClassifier(n_estimators=50, random_state=42)
        better_model.fit(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            [0, 1, 0, 1],
        )

        mv2 = await registry.register_model(
            "churn_model",
            pickle.dumps(better_model),
            metrics=[MetricSpec(name="accuracy", value=0.95)],
            signature=signature,
        )

        assert mv2.version == 2

        # ── 6. get_model() — various lookups ────────────────────────

        latest = await registry.get_model("churn_model")
        assert latest.version == 2

        v1 = await registry.get_model("churn_model", version=1)
        assert v1.version == 1

        staging = await registry.get_model("churn_model", stage="staging")
        assert staging.stage == "staging"

        # ── 7. list_models() and get_model_versions() ───────────────

        models = await registry.list_models()
        assert models[0]["name"] == "churn_model"
        assert models[0]["latest_version"] == 2

        versions = await registry.get_model_versions("churn_model")
        assert len(versions) == 2
        assert versions[0].version == 2  # Newest first

        # ── 8. promote_model() — staging -> shadow -> production ────

        promoted = await registry.promote_model(
            "churn_model",
            version=1,
            target_stage="shadow",
            reason="Shadow testing before production",
        )
        assert promoted.stage == "shadow"

        prod = await registry.promote_model(
            "churn_model",
            version=1,
            target_stage="production",
            reason="Shadow tests passed",
        )
        assert prod.stage == "production"

        # ── 9. Promote v2 to production — auto-archives v1 ─────────

        prod_v2 = await registry.promote_model(
            "churn_model",
            version=2,
            target_stage="production",
            reason="v2 outperforms v1",
        )
        assert prod_v2.stage == "production"

        v1_after = await registry.get_model("churn_model", version=1)
        assert v1_after.stage == "archived"

        # ── 10. compare() — compare two versions ───────────────────

        comparison = await registry.compare("churn_model", 1, 2)
        assert comparison["version_a"] == 1
        assert comparison["version_b"] == 2
        assert "deltas" in comparison
        assert "better_version" in comparison

        # ── 11. load_artifact() — retrieve model bytes ──────────────

        loaded_bytes = await registry.load_artifact("churn_model", 1)
        loaded_model = pickle.loads(loaded_bytes)
        assert hasattr(loaded_model, "predict")

        # ── 12. MLflow export/import ────────────────────────────────

        with tempfile.TemporaryDirectory() as export_dir:
            export_path = await registry.export_mlflow(
                "churn_model", 1, export_dir
            )
            assert (export_path / "MLmodel").exists()
            assert (export_path / "model.pkl").exists()

            imported_mv = await registry.import_mlflow(export_path)
            assert imported_mv.stage == "staging"

        # ── 13. Edge cases ──────────────────────────────────────────

        # Invalid stage transition
        try:
            await registry.promote_model(
                "churn_model", version=2, target_stage="staging"
            )
        except ValueError as e:
            assert "invalid transition" in str(e).lower()

        # Model not found
        try:
            await registry.get_model("nonexistent_model")
        except ModelNotFoundError:
            pass  # Expected

        # MetricSpec rejects NaN/Inf
        try:
            MetricSpec(name="bad", value=float("nan"))
        except ValueError:
            pass  # Expected

    await conn.close()

asyncio.run(main())
```

### Step-by-Step Explanation

1. **Setup**: ModelRegistry requires a `ConnectionManager` for metadata storage and an `ArtifactStore` for model bytes. `LocalFileArtifactStore` writes to a local directory.

2. **Serialization**: `pickle.dumps(model)` converts the sklearn model to bytes. The registry stores these bytes, not the model object.

3. **Metrics and signature**: `MetricSpec` captures name/value pairs for evaluation metrics. `ModelSignature` documents the input schema, output columns, and model type.

4. **Registration**: `register_model()` creates version 1 at the `"staging"` stage. The returned `ModelVersion` includes version number, stage, metrics, signature, UUID, and timestamp.

5. **Auto-incrementing versions**: Registering another model with the same name creates version 2. Version numbers increment automatically.

6. **Flexible lookups**: `get_model()` supports lookup by latest (default), specific version, or stage.

7. **Listing**: `list_models()` returns summary info. `get_model_versions()` returns all versions, newest first.

8. **Promotion lifecycle**: `promote_model()` transitions between stages. Valid transitions: staging to shadow/production/archived, shadow to production/archived/staging, production to archived/shadow.

9. **Auto-archiving**: When v2 is promoted to production, v1 (the current production version) is automatically archived. Only one version can be in production at a time.

10. **Comparison**: `compare()` computes metric deltas and identifies the better version based on metric improvements.

11. **Artifact retrieval**: `load_artifact()` returns the original bytes. `pickle.loads()` reconstructs the model.

12. **MLflow interop**: `export_mlflow()` writes MLmodel metadata and model.pkl. `import_mlflow()` reads them back as a new staging version.

13. **Edge cases**: Invalid stage transitions, non-existent models, and non-finite metric values are all caught with clear errors.

## Common Mistakes

| Mistake                                       | Correct Pattern                                     | Why                                                                     |
| --------------------------------------------- | --------------------------------------------------- | ----------------------------------------------------------------------- |
| Passing the model object instead of bytes     | `registry.register_model("m", pickle.dumps(model))` | The registry stores bytes, not Python objects                           |
| Promoting directly from staging to production | Go staging -> shadow -> production                  | Shadow testing catches issues before they affect live traffic           |
| Forgetting to initialize ConnectionManager    | `await conn.initialize()` before creating registry  | Registry operations fail with RuntimeError on uninitialized connections |
| Using NaN/Inf in MetricSpec                   | Validate metrics before creating MetricSpec         | Non-finite values are rejected to prevent corrupt comparisons           |

## Exercises

1. Register three versions of a model with increasing accuracy (0.85, 0.90, 0.95). Use `compare()` to compare v1 vs v3 and verify the delta.

2. Promote v1 to production, then promote v2 to production. Verify v1 was automatically archived. Then try promoting v1 back to staging from archived -- what transition is this?

3. Export a model to MLflow format, import it back, and verify the imported version's metrics match the original.

## Key Takeaways

- ModelRegistry manages versioned model artifacts with a staging/shadow/production/archived lifecycle
- Models are stored as bytes (`pickle.dumps()`), not as Python objects
- Version numbers auto-increment; new registrations start at staging
- Promoting to production automatically archives the current production version
- `compare()` computes metric deltas for data-driven promotion decisions
- MLflow export/import enables interoperability with external ML tooling
- MetricSpec rejects NaN/Inf; ModelNotFoundError is raised for missing models

## Next Chapter

[Chapter 12: DriftMonitor](12_drift_monitor.md) -- Detect feature drift using PSI and KS-test, monitor performance degradation, and configure scheduled monitoring.
