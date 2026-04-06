# Chapter 2: ML to Nexus Deployment

## Overview

After training and registering a model, the next step is making it available to consumers -- web applications, APIs, and AI agents. This chapter teaches the deployment pipeline: export a trained model to ONNX format with **OnnxBridge**, load it into **InferenceServer** for predictions, and expose it through **Nexus** for multi-channel access (REST API + MCP tools simultaneously).

## Prerequisites

- [Chapter 1: ML to Registry Pipeline](01_ml_to_registry.md)
- Kailash Nexus installed (`pip install kailash-nexus`)
- Understanding of ONNX format (helpful)

## Concepts

### Concept 1: The Deployment Pipeline

The full path from a trained model to a production endpoint:

```
TrainingPipeline.train()
    -> OnnxBridge.export(model)           # Convert to ONNX
    -> OnnxBridge.validate(onnx_model)    # Verify correctness
    -> InferenceServer(model_path)        # Load for serving
    -> InferenceServer.predict(input)     # Single prediction
    -> Nexus.register("predict", workflow) # Expose via API+MCP
    -> Nexus.start()                      # Go live
```

- **What**: A six-step pipeline from training artifacts to live multi-channel endpoints
- **Why**: Each step adds a capability -- ONNX for portability, InferenceServer for caching, Nexus for multi-channel access
- **How**: Chain the output of each step into the next
- **When**: Whenever a model needs to serve live traffic

### Concept 2: OnnxBridge

OnnxBridge converts trained models from their native format (sklearn, xgboost, etc.) to ONNX (Open Neural Network Exchange). ONNX models are portable across runtimes and languages, and enable hardware-specific optimizations.

- **What**: A converter from ML framework models to ONNX format with validation
- **Why**: ONNX decouples training framework from inference runtime, enabling optimized serving
- **How**: `onnx_model = OnnxBridge.export(trained_model, input_schema)` then `OnnxBridge.validate(onnx_model)`
- **When**: After training, before loading into InferenceServer

### Concept 3: Multi-Channel Exposure via Nexus

When you register a handler with Nexus, it becomes available on multiple channels simultaneously:

| Channel | Access Pattern                    | Consumer         |
| ------- | --------------------------------- | ---------------- |
| HTTP    | `POST /workflows/predict/execute` | Web apps, mobile |
| MCP     | Tool: `workflow_predict`          | AI agents        |

- **What**: Single registration produces endpoints on all channels
- **Why**: AI agents and web applications can both access the same model without separate integrations
- **How**: `app.handler("predict")` registers a handler that Nexus exposes on all channels
- **When**: Whenever the model needs to be accessible to both human and AI consumers

### Key API

| Method / Class           | Parameters                    | Returns           | Description                        |
| ------------------------ | ----------------------------- | ----------------- | ---------------------------------- |
| `OnnxBridge.export()`    | `model`, `input_schema`       | `onnx_model`      | Convert to ONNX format             |
| `OnnxBridge.validate()`  | `onnx_model`                  | `bool`            | Verify ONNX correctness            |
| `InferenceServer()`      | `model_path`                  | `InferenceServer` | Create a prediction server         |
| `server.predict()`       | `input_data`                  | `result`          | Single prediction                  |
| `server.predict_batch()` | `batch_data`                  | `results`         | Batch prediction                   |
| `Nexus()`                | `enable_durability`, `preset` | `Nexus`           | Create multi-channel platform      |
| `@app.handler()`         | `name`, `description`         | decorator         | Register a handler on all channels |

## Code Walkthrough

```python
from __future__ import annotations

from kailash_ml import InferenceServer, OnnxBridge
from nexus import Nexus
```

### The Deployment Pipeline

The full production path involves six steps. Here we demonstrate the Nexus registration pattern:

```python
# InferenceServer configuration:
# server = InferenceServer(model_path="model.onnx")
# await server.initialize()
# result = await server.predict(input_data)
# results = await server.predict_batch(batch_data)
# await server.warm_cache()  # Pre-load for low-latency

# OnnxBridge configuration:
# onnx_model = OnnxBridge.export(trained_model, input_schema)
# is_valid = OnnxBridge.validate(onnx_model)
```

InferenceServer handles model loading, warm caching (pre-loading into memory for low-latency first requests), and both single and batch predictions. OnnxBridge handles the conversion from native ML formats to ONNX.

### Nexus Handler Registration

```python
app = Nexus(enable_durability=False)


@app.handler("predict", description="Run model prediction")
async def predict(features: str) -> dict:
    """Handler that wraps InferenceServer.predict()."""
    # In production:
    # result = await server.predict(json.loads(features))
    # return {"prediction": result.prediction, "confidence": result.confidence}
    return {"prediction": 1, "confidence": 0.92}


# Verify registration
assert app._registry is not None
```

The `@app.handler` decorator registers the function with Nexus. After `app.start()`, this handler is available as both a REST endpoint (`POST /workflows/predict/execute`) and an MCP tool (`workflow_predict`).

### Multi-Channel Access

After registration, consumers access the model through their preferred channel:

- **Web applications** call `POST /workflows/predict/execute` with feature data
- **AI agents** call the `workflow_predict` MCP tool, enabling agents to use ML models as reasoning tools
- **CLI users** interact through the Nexus CLI channel

This single registration, multi-channel deployment pattern eliminates the need to build separate API and agent integrations.

### Production Pattern

```python
# In production, combine all engines:
#
#   from kailash_ml import TrainingPipeline, ModelRegistry, InferenceServer, OnnxBridge
#   from nexus import Nexus
#
#   # Train
#   model, metrics = pipeline.train(data, target="y")
#
#   # Register
#   version = await registry.register_model("model", artifact=pickle.dumps(model))
#   await registry.promote_model("model", version.version, "production")
#
#   # Export and serve
#   onnx = OnnxBridge.export(model, schema)
#   server = InferenceServer(onnx_path)
#   await server.initialize()
#
#   # Deploy
#   app = Nexus(preset="production")
#   app.register("predict", prediction_workflow)
#   app.start()
```

The production pattern chains all four Kailash packages: kailash-ml (training + registry), OnnxBridge (export), InferenceServer (serving), and Nexus (multi-channel deployment).

## Exercises

1. Create a Nexus app with two handlers: one for single predictions and one for batch predictions. Both should wrap an InferenceServer. What HTTP endpoints and MCP tools does this produce?
2. Why is ONNX export important for production serving? What problems would arise if you served the pickle-serialized model directly?
3. Describe the difference between `server.predict()` and `server.predict_batch()`. When would you use each in a Nexus handler?

## Key Takeaways

- The deployment pipeline is: Train -> ONNX Export -> InferenceServer -> Nexus
- OnnxBridge converts models to portable ONNX format with validation
- InferenceServer handles model loading, caching, and prediction (single + batch)
- Nexus exposes handlers on all channels simultaneously (HTTP + MCP)
- Single registration produces both REST API and MCP tool access
- AI agents can call ML models via MCP tools, enabling agent-driven data science

## Next Chapter

[Chapter 3: Agent with ML Tools](03_agent_with_tools.md) -- Give a Kaizen AI agent access to ML engines as tools.
