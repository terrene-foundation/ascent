# Chapter 13: InferenceServer

## Overview

`InferenceServer` serves predictions from models stored in ModelRegistry. It supports single-record prediction, batch prediction, model caching for low-latency serving, version pinning, and model introspection via the `MLToolProtocol`. This chapter covers the full prediction lifecycle: registering a model, creating a server, predicting, caching, querying metrics, and handling multiple versions.

## Prerequisites

- Python 3.10+ installed
- Kailash ML installed (`pip install kailash-ml`)
- Completed [Chapter 11: ModelRegistry](11_model_registry.md)
- Understanding of ConnectionManager and model registration workflow
- Familiarity with sklearn classifiers

## Concepts

### Concept 1: Prediction via Registry

InferenceServer does not load models directly -- it reads from ModelRegistry. When you call `predict()`, the server fetches the model artifact from the registry, deserializes it, and runs inference. This means every model served through InferenceServer has full version tracking, metrics, and lifecycle management.

- **What**: A prediction server backed by ModelRegistry for model resolution
- **Why**: Decouples the serving layer from model storage, enabling version management, A/B testing, and rollback
- **How**: `server.predict("model_name", features={...})` resolves the model from the registry, loads it, and calls `.predict()`
- **When**: Use for any model that needs versioned, managed serving -- which is every production model

### Concept 2: Single vs Batch Prediction

`predict()` handles one record at a time, returning a `PredictionResult` with the prediction, probabilities (for classifiers), model version, and inference time. `predict_batch()` handles multiple records efficiently, returning a list of `PredictionResult` objects.

- **What**: Two prediction APIs for different throughput needs
- **Why**: Single prediction suits real-time APIs; batch prediction suits offline scoring and bulk processing
- **How**: Both resolve the model the same way; batch prediction passes all records through the model in one call
- **When**: `predict()` for API endpoints; `predict_batch()` for scoring datasets, generating reports, or backfilling predictions

### Concept 3: Model Caching

`warm_cache()` pre-loads models into memory so subsequent predictions avoid the deserialization overhead. The `cache_size` parameter controls how many models are kept in memory. Cached models serve predictions significantly faster.

- **What**: An in-memory LRU cache of deserialized model objects
- **Why**: Deserializing a model from bytes on every prediction is slow; caching amortizes the cost
- **How**: `warm_cache(["model_name"])` loads and caches; subsequent `predict()` calls hit the cache
- **When**: Call during server startup for models that serve frequent traffic

### Concept 4: MLToolProtocol

InferenceServer implements the `MLToolProtocol`, which provides `get_metrics()` and `get_model_info()` for model introspection. These methods return structured information about model performance and configuration without running inference.

- **What**: A protocol for querying model metadata and metrics
- **Why**: Enables monitoring dashboards, health checks, and model selection without prediction calls
- **How**: `get_metrics("model")` returns metrics dict; `get_model_info("model")` returns version list and signature
- **When**: For health check endpoints, admin dashboards, and automated model selection

## Key API

| Method              | Parameters                                          | Returns                  | Description                              |
| ------------------- | --------------------------------------------------- | ------------------------ | ---------------------------------------- |
| `InferenceServer()` | `registry: ModelRegistry`, `cache_size: int = 10`   | `InferenceServer`        | Create with registry and cache size      |
| `predict()`         | `model_name`, `features: dict`, `version=None`      | `PredictionResult`       | Single-record prediction                 |
| `predict_batch()`   | `model_name`, `records: list[dict]`, `version=None` | `list[PredictionResult]` | Multi-record prediction                  |
| `warm_cache()`      | `model_names: list[str]`                            | `None`                   | Pre-load models into memory              |
| `get_metrics()`     | `model_name`, `version=None`                        | `dict`                   | Retrieve model metrics (MLToolProtocol)  |
| `get_model_info()`  | `model_name`                                        | `dict`                   | Retrieve model metadata (MLToolProtocol) |

## Code Walkthrough

```python
from __future__ import annotations

import asyncio
import pickle
import tempfile

from sklearn.ensemble import RandomForestClassifier

from kailash.db.connection import ConnectionManager
from kailash_ml import FeatureField, FeatureSchema, MetricSpec, ModelSignature
from kailash_ml.engines.inference_server import InferenceServer, PredictionResult
from kailash_ml.engines.model_registry import LocalFileArtifactStore, ModelRegistry


async def main() -> None:
    # ── 1. Set up registry with a trained model ─────────────────────

    conn = ConnectionManager("sqlite:///:memory:")
    await conn.initialize()

    with tempfile.TemporaryDirectory() as artifact_dir:
        artifact_store = LocalFileArtifactStore(artifact_dir)
        registry = ModelRegistry(conn, artifact_store)

        model = RandomForestClassifier(n_estimators=20, random_state=42)
        model.fit(
            [
                [1.0, 2.0, 0.5], [3.0, 4.0, 1.5],
                [5.0, 6.0, 2.5], [7.0, 8.0, 3.5],
                [2.0, 1.0, 0.2], [4.0, 3.0, 1.2],
                [6.0, 5.0, 2.2], [8.0, 7.0, 3.2],
            ],
            [0, 1, 0, 1, 0, 1, 0, 1],
        )

        signature = ModelSignature(
            input_schema=FeatureSchema(
                name="inference_input",
                features=[
                    FeatureField(name="feat_a", dtype="float64"),
                    FeatureField(name="feat_b", dtype="float64"),
                    FeatureField(name="feat_c", dtype="float64"),
                ],
                entity_id_column="entity_id",
            ),
            output_columns=["prediction"],
            output_dtypes=["float64"],
            model_type="classifier",
        )

        mv = await registry.register_model(
            "serving_model",
            pickle.dumps(model),
            metrics=[
                MetricSpec(name="accuracy", value=0.95),
                MetricSpec(name="f1", value=0.93),
            ],
            signature=signature,
        )

        # ── 2. Create InferenceServer ───────────────────────────────

        server = InferenceServer(registry, cache_size=5)

        # ── 3. predict() — single-record prediction ────────────────

        result = await server.predict(
            "serving_model",
            features={"feat_a": 3.0, "feat_b": 4.0, "feat_c": 1.5},
        )

        assert isinstance(result, PredictionResult)
        assert result.prediction in (0, 1)
        assert result.model_name == "serving_model"
        assert result.model_version == mv.version
        assert result.inference_time_ms >= 0
        assert result.inference_path in ("native", "onnx")

        # Probabilities for classification models
        assert result.probabilities is not None
        assert len(result.probabilities) == 2
        assert abs(sum(result.probabilities) - 1.0) < 0.01

        # ── 4. predict() — with specific version ───────────────────

        versioned_result = await server.predict(
            "serving_model",
            features={"feat_a": 5.0, "feat_b": 6.0, "feat_c": 2.5},
            version=1,
        )
        assert versioned_result.model_version == 1

        # ── 5. predict_batch() — multiple records at once ───────────

        records = [
            {"feat_a": 1.0, "feat_b": 2.0, "feat_c": 0.5},
            {"feat_a": 3.0, "feat_b": 4.0, "feat_c": 1.5},
            {"feat_a": 5.0, "feat_b": 6.0, "feat_c": 2.5},
            {"feat_a": 7.0, "feat_b": 8.0, "feat_c": 3.5},
        ]

        batch_results = await server.predict_batch("serving_model", records)

        assert len(batch_results) == 4
        for r in batch_results:
            assert isinstance(r, PredictionResult)
            assert r.prediction in (0, 1)

        # Empty list returns empty results
        empty_results = await server.predict_batch("serving_model", [])
        assert empty_results == []

        # ── 6. warm_cache() — pre-load models ──────────────────────

        await server.warm_cache(["serving_model"])

        cached_result = await server.predict(
            "serving_model",
            features={"feat_a": 2.0, "feat_b": 3.0, "feat_c": 1.0},
        )
        assert isinstance(cached_result, PredictionResult)

        # ── 7. get_metrics() — MLToolProtocol ───────────────────────

        metrics_info = await server.get_metrics("serving_model")
        assert metrics_info["metrics"]["accuracy"] == 0.95
        assert metrics_info["metrics"]["f1"] == 0.93

        versioned_metrics = await server.get_metrics(
            "serving_model", version="1"
        )
        assert versioned_metrics["version"] == 1

        # ── 8. get_model_info() — MLToolProtocol ────────────────────

        model_info = await server.get_model_info("serving_model")
        assert model_info["name"] == "serving_model"
        assert isinstance(model_info["versions"], list)
        assert model_info["signature"] is not None

        # ── 9. Multiple versions — latest-serving ───────────────────

        model_v2 = RandomForestClassifier(n_estimators=50, random_state=42)
        model_v2.fit(
            [
                [1.0, 2.0, 0.5], [3.0, 4.0, 1.5],
                [5.0, 6.0, 2.5], [7.0, 8.0, 3.5],
                [2.0, 1.0, 0.2], [4.0, 3.0, 1.2],
                [6.0, 5.0, 2.2], [8.0, 7.0, 3.2],
            ],
            [0, 1, 0, 1, 0, 1, 0, 1],
        )

        mv2 = await registry.register_model(
            "serving_model",
            pickle.dumps(model_v2),
            metrics=[MetricSpec(name="accuracy", value=0.97)],
            signature=signature,
        )

        # predict() without version uses latest
        latest_result = await server.predict(
            "serving_model",
            features={"feat_a": 3.0, "feat_b": 4.0, "feat_c": 1.5},
        )
        assert latest_result.model_version == 2

        # Can still target v1 explicitly
        v1_result = await server.predict(
            "serving_model",
            features={"feat_a": 3.0, "feat_b": 4.0, "feat_c": 1.5},
            version=1,
        )
        assert v1_result.model_version == 1

        # ── 10. Serialization round-trip ────────────────────────────

        pr_dict = result.to_dict()
        pr_restored = PredictionResult.from_dict(pr_dict)
        assert pr_restored.prediction == result.prediction
        assert pr_restored.model_name == result.model_name

    await conn.close()

asyncio.run(main())
```

### Step-by-Step Explanation

1. **Setup**: Train a classifier, register it in ModelRegistry with metrics and a signature. The registry provides versioned storage that InferenceServer reads from.

2. **Server creation**: `InferenceServer(registry, cache_size=5)` creates a server backed by the registry with an LRU cache holding up to 5 deserialized models.

3. **Single prediction**: `predict()` returns a `PredictionResult` with the prediction value, class probabilities (for classifiers), model name, version, inference time in milliseconds, and the inference path (native sklearn or ONNX).

4. **Version pinning**: `predict(..., version=1)` targets a specific model version. Without `version`, the server uses the latest version.

5. **Batch prediction**: `predict_batch()` processes multiple records and returns a list of results. Empty input returns an empty list.

6. **Cache warming**: `warm_cache()` pre-loads models to avoid deserialization latency on the first prediction. Call this at server startup.

7. **Metrics query**: `get_metrics()` returns the registered metrics for a model version -- useful for monitoring dashboards.

8. **Model info**: `get_model_info()` returns the model name, all version numbers, and the signature -- useful for admin endpoints and health checks.

9. **Multi-version serving**: Registering v2 makes it the latest. `predict()` without version uses v2; `predict(..., version=1)` still serves v1. This enables gradual rollout and A/B testing.

10. **Serialization**: `PredictionResult.to_dict()` and `from_dict()` support logging and API responses.

## Common Mistakes

| Mistake                                       | Correct Pattern                                   | Why                                                                                      |
| --------------------------------------------- | ------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| Creating InferenceServer without a registry   | Always back with ModelRegistry                    | InferenceServer resolves models from the registry; direct model loading is not supported |
| Not warming cache for high-traffic models     | Call `warm_cache()` at startup                    | Cold predictions incur deserialization overhead on the first request                     |
| Passing model object instead of features dict | `predict("model", features={"a": 1.0, "b": 2.0})` | The server accepts feature dictionaries, not raw numpy arrays                            |
| Ignoring `inference_path` in PredictionResult | Check if ONNX path is being used                  | ONNX serving can be faster; knowing the path helps debug performance                     |

## Exercises

1. Register two versions of a model with different hyperparameters. Use InferenceServer to predict the same record with both versions and compare the results.

2. Benchmark cold vs warm prediction: time a `predict()` call before and after `warm_cache()`. How much faster is the cached path?

3. Build a simple prediction API pattern: a function that takes a model name and feature dict, calls `predict()`, and returns a JSON-serializable response using `to_dict()`.

## Key Takeaways

- InferenceServer serves predictions from models stored in ModelRegistry
- `predict()` returns a PredictionResult with prediction, probabilities, version, and timing
- `predict_batch()` handles multiple records efficiently
- `warm_cache()` pre-loads models for low-latency serving
- Version pinning enables A/B testing and gradual rollout
- `get_metrics()` and `get_model_info()` provide model introspection without inference
- PredictionResult supports serialization for logging and API responses

## Next Chapter

[Chapter 14: OnnxBridge](14_onnx_bridge.md) -- Export trained models to ONNX format for cross-runtime serving.
