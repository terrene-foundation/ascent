# Chapter 1: ML to Registry Pipeline

## Overview

The most common Kailash integration pattern connects **kailash-ml** training to the **ModelRegistry** for versioned model management. You train a model with `TrainingPipeline`, serialize it, register it in the registry with metrics, and promote it through governance stages to production. This chapter teaches the end-to-end pattern: train, serialize, register, and promote.

## Prerequisites

- Kailash ML installed (`pip install kailash-ml`)
- Kailash Core SDK installed (`pip install kailash`)
- Familiarity with TrainingPipeline from the ML section
- Understanding of model serialization (pickle)

## Concepts

### Concept 1: The ML Lifecycle in Kailash

Kailash's ML lifecycle follows four stages:

| Stage    | Framework        | What Happens                            |
| -------- | ---------------- | --------------------------------------- |
| Train    | TrainingPipeline | Fit a model on data, produce metrics    |
| Register | ModelRegistry    | Store versioned artifact with metrics   |
| Promote  | ModelRegistry    | Governance gate: staging -> production  |
| Serve    | InferenceServer  | Load model, serve predictions via Nexus |

- **What**: A four-stage pipeline connecting training to serving through versioned governance
- **Why**: Without registration and promotion, models go directly from training to production without review, creating untracked production deployments
- **How**: `TrainingPipeline.train()` -> `pickle.dumps()` -> `ModelRegistry.register_model()` -> `ModelRegistry.promote_model()`
- **When**: Every time a model is trained and intended for production use

### Concept 2: Artifact Serialization

ModelRegistry stores model artifacts as **bytes**, not model objects. You must serialize the trained model (typically with `pickle.dumps()`) before registration. This decouples the registry from any specific ML framework.

- **What**: Converting a trained model object to a byte array for storage
- **Why**: The registry is framework-agnostic -- it stores bytes, not sklearn/xgboost/torch objects
- **How**: `model_bytes = pickle.dumps(trained_model)`
- **When**: Between training and registration

### Concept 3: MetricSpec

Metrics are registered alongside the model artifact using `MetricSpec` objects. Each spec has a `name` and a `value`. These metrics are used for promotion decisions -- you compare versions by their registered metrics.

### Key API

| Method / Class              | Parameters                                             | Returns         | Description                       |
| --------------------------- | ------------------------------------------------------ | --------------- | --------------------------------- |
| `TrainingPipeline()`        | --                                                     | `Pipeline`      | Create a training pipeline        |
| `ModelRegistry(conn)`       | `conn: ConnectionManager`                              | `ModelRegistry` | Create a registry backed by DB    |
| `registry.register_model()` | `name`, `artifact: bytes`, `metrics: list[MetricSpec]` | `ModelVersion`  | Register a versioned model        |
| `registry.promote_model()`  | `name`, `version`, `target_stage`, `reason`            | `None`          | Promote through governance stages |
| `MetricSpec()`              | `name: str`, `value: float`                            | `MetricSpec`    | A single metric name-value pair   |

## Code Walkthrough

```python
from __future__ import annotations

import asyncio
import os
import pickle
import tempfile

import polars as pl
from kailash_ml import (
    TrainingPipeline,
    ModelRegistry,
)
from kailash_ml.types import MetricSpec, ModelSignature
from kailash.db.connection import ConnectionManager
```

### Create Training Data

```python
df = pl.DataFrame(
    {
        "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "feature_2": [0.1, 0.4, 0.9, 1.6, 2.5, 3.6, 4.9, 6.4, 8.1, 10.0],
        "target": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    }
)

assert df.shape == (10, 3)
```

A minimal binary classification dataset. In production, you would load real data via `ASCENTDataLoader` or a DataFlow query.

### Train and Serialize

```python
# Pattern: train -> get model bytes -> register
# pipeline = TrainingPipeline()
# model, metrics = pipeline.train(data=df, target="target", spec=ModelSpec(...))
# model_bytes = pickle.dumps(model)

# For demonstration, create a minimal model artifact
trained_model = {"type": "logistic_regression", "weights": [0.5, 0.3]}
model_bytes = pickle.dumps(trained_model)

assert isinstance(model_bytes, bytes), "Model must be serialized to bytes"
```

The critical step is `pickle.dumps()` -- ModelRegistry expects bytes, not a model object. In production, the full `TrainingPipeline.train()` call produces a real sklearn/xgboost model.

### Register in ModelRegistry

```python
async def main():
    db_path = os.path.join(tempfile.gettempdir(), "textbook_integration_01.db")
    conn = ConnectionManager(f"sqlite:///{db_path}")
    await conn.initialize()

    registry = ModelRegistry(conn)
    await registry.initialize()

    # Register with metrics
    version = await registry.register_model(
        name="credit_scorer",
        artifact=model_bytes,
        metrics=[
            MetricSpec(name="accuracy", value=0.85),
            MetricSpec(name="auc", value=0.92),
        ],
    )

    assert version is not None, "Registration returns a version"
    print(f"Registered version: {version}")
```

ModelRegistry is backed by a database via `ConnectionManager`. The `register_model()` call stores the artifact bytes and associated metrics, returning a version object.

### Promote to Production

```python
    await registry.promote_model(
        name="credit_scorer",
        version=version.version,
        target_stage="production",
        reason="Passed all validation checks",
    )

    print("Model promoted to production")
```

Promotion is the governance gate. The `reason` field documents why this model was approved for production -- essential for audit trails.

### The Complete Integration Pattern

The full pipeline from training to serving:

```
TrainingPipeline.train()
    -> pickle.dumps()
        -> ModelRegistry.register_model()
            -> ModelRegistry.promote_model()
                -> InferenceServer.predict()
```

1. **Train** with kailash-ml TrainingPipeline
2. **Register** the serialized model with metrics in ModelRegistry
3. **Promote** through the governance gate (staging -> production)
4. **Serve** via InferenceServer deployed through Nexus

## Exercises

1. Create a ModelRegistry backed by an in-memory SQLite database. Register two versions of the same model with different accuracy metrics. Promote the one with higher accuracy to production.
2. Why does ModelRegistry require `bytes` instead of a model object? What advantages does this give for multi-framework support?
3. Extend the pattern to include evaluation: after registering, load the model bytes back with `pickle.loads()`, run predictions on a test set, and only promote if accuracy exceeds a threshold.

## Key Takeaways

- The ML lifecycle in Kailash is: Train -> Register -> Promote -> Serve
- ModelRegistry stores model artifacts as bytes -- always use `pickle.dumps()` before registration
- MetricSpec attaches named metrics to registered versions for comparison
- Promotion is a governance gate that requires explicit approval with a documented reason
- ModelRegistry is database-backed via ConnectionManager for production persistence

## Next Chapter

[Chapter 2: ML to Nexus Deployment](02_ml_to_nexus.md) -- Deploy trained models via ONNX export and Nexus multi-channel serving.
