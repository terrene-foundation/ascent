# Chapter 14: OnnxBridge

## Overview

`OnnxBridge` exports trained sklearn models to ONNX format for cross-runtime serving -- "train in Python, serve anywhere." It provides pre-flight compatibility checks, export with automatic feature detection, and post-export numerical validation that compares native predictions against ONNX runtime predictions. Export is non-fatal: failures return a result object instead of raising exceptions, making ONNX support an optional optimization rather than a hard requirement.

## Prerequisites

- Python 3.10+ installed
- Kailash ML installed (`pip install kailash-ml`)
- Optional: `skl2onnx` and `onnxruntime` for full export and validation
- Completed [Chapter 13: InferenceServer](13_inference_server.md)
- Familiarity with sklearn model training

## Concepts

### Concept 1: Pre-Flight Compatibility Check

Before exporting, `check_compatibility()` verifies whether a model can be converted to ONNX. It returns an `OnnxCompatibility` object with a `compatible` flag, a `confidence` level ("guaranteed", "likely", "unsupported"), the framework name, and the model type. This avoids wasted export attempts on unsupported models.

- **What**: A pre-export check that determines if a model can be converted to ONNX
- **Why**: Not all model types support ONNX export; checking first saves time and provides clear diagnostics
- **How**: Maps the model's class name against a registry of known-compatible sklearn types
- **When**: Before calling `export()`, especially when handling models of unknown type

### Concept 2: Non-Fatal Export

`export()` returns an `OnnxExportResult` with `success`, `onnx_status`, `onnx_path`, `model_size_bytes`, `export_time_seconds`, and `error_message`. On failure, `success` is `False` and `error_message` explains why. The method never raises exceptions -- this makes ONNX support a graceful optimization path.

- **What**: An export method that returns a result object rather than raising on failure
- **Why**: ONNX export depends on optional libraries (`skl2onnx`); missing dependencies should not crash the application
- **How**: Internally catches all export errors and wraps them in the result object
- **When**: Always -- even with guaranteed compatibility, export can fail due to missing libraries

### Concept 3: Numerical Validation

After export, `validate()` compares native sklearn predictions against ONNX runtime predictions on the same input data. It reports `max_diff`, `mean_diff`, and whether the differences fall within a specified tolerance. This catches numerical precision issues that could affect production predictions.

- **What**: A post-export check that verifies ONNX predictions match native predictions
- **Why**: ONNX conversion can introduce numerical differences due to floating-point representation
- **How**: Runs the same inputs through both the native model and the ONNX model, computing element-wise differences
- **When**: After every successful export, before deploying the ONNX model

### Concept 4: Feature Detection

OnnxBridge can determine the number of input features from three sources (in priority order): an explicit `schema` parameter, an explicit `n_features` parameter, or the model's `n_features_in_` attribute (set by sklearn after fitting). If none are available, export fails with a clear error.

- **What**: Automatic inference of model input dimensions for ONNX graph construction
- **Why**: ONNX requires fixed input dimensions at export time
- **How**: Checks `schema.input_schema.features`, then `n_features`, then `model.n_features_in_`
- **When**: During `export()` -- you can omit explicit dimensions if the model has been fitted

## Key API

| Method                  | Parameters                                                                 | Returns                | Description                        |
| ----------------------- | -------------------------------------------------------------------------- | ---------------------- | ---------------------------------- |
| `OnnxBridge()`          | --                                                                         | `OnnxBridge`           | Create the bridge                  |
| `check_compatibility()` | `model`, `framework: str`                                                  | `OnnxCompatibility`    | Pre-flight compatibility check     |
| `export()`              | `model`, `framework`, `schema=None`, `output_path=None`, `n_features=None` | `OnnxExportResult`     | Export to ONNX (non-fatal)         |
| `validate()`            | `model`, `onnx_path`, `X_test`, `tolerance=1e-4`                           | `OnnxValidationResult` | Compare native vs ONNX predictions |

## Code Walkthrough

```python
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge

from kailash_ml import FeatureField, FeatureSchema, ModelSignature
from kailash_ml.bridge.onnx_bridge import (
    OnnxBridge,
    OnnxCompatibility,
    OnnxExportResult,
    OnnxValidationResult,
)

# ── 1. Instantiate OnnxBridge ───────────────────────────────────────

bridge = OnnxBridge()

# ── 2. Train models for export ──────────────────────────────────────

X_train = np.array(
    [
        [1.0, 2.0, 0.5], [3.0, 4.0, 1.5],
        [5.0, 6.0, 2.5], [7.0, 8.0, 3.5],
        [2.0, 1.0, 0.2], [4.0, 3.0, 1.2],
        [6.0, 5.0, 2.2], [8.0, 7.0, 3.2],
    ]
)
y_cls = np.array([0, 1, 0, 1, 0, 1, 0, 1])
y_reg = np.array([10.0, 20.0, 30.0, 40.0, 15.0, 25.0, 35.0, 45.0])

rf = RandomForestClassifier(n_estimators=10, random_state=42)
rf.fit(X_train, y_cls)

gb = GradientBoostingClassifier(n_estimators=10, random_state=42)
gb.fit(X_train, y_cls)

lr = LogisticRegression(max_iter=200, random_state=42)
lr.fit(X_train, y_cls)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_reg)

# ── 3. check_compatibility() — pre-flight check ────────────────────

rf_compat = bridge.check_compatibility(rf, "sklearn")
assert isinstance(rf_compat, OnnxCompatibility)
assert rf_compat.compatible is True
assert rf_compat.confidence == "guaranteed"
assert rf_compat.framework == "sklearn"
assert rf_compat.model_type == "RandomForestClassifier"

# Unknown framework
unknown_compat = bridge.check_compatibility(rf, "unknown_framework")
assert unknown_compat.compatible is False
assert unknown_compat.confidence == "unsupported"

# ── 4. export() — sklearn RandomForest ──────────────────────────────

schema = ModelSignature(
    input_schema=FeatureSchema(
        name="export_test",
        features=[
            FeatureField(name="a", dtype="float64"),
            FeatureField(name="b", dtype="float64"),
            FeatureField(name="c", dtype="float64"),
        ],
        entity_id_column="entity_id",
    ),
    output_columns=["prediction"],
    output_dtypes=["float64"],
    model_type="classifier",
)

with tempfile.TemporaryDirectory() as tmpdir:
    output_path = Path(tmpdir) / "rf_model.onnx"

    result = bridge.export(
        rf, "sklearn", schema=schema, output_path=output_path,
    )

    assert isinstance(result, OnnxExportResult)

    if result.success:
        assert result.onnx_path == output_path
        assert output_path.exists()
        assert result.model_size_bytes > 0

        # ── 5. validate() — numerical correctness ──────────────

        validation = bridge.validate(
            rf, output_path, X_train, tolerance=1e-4,
        )

        assert isinstance(validation, OnnxValidationResult)
        if validation.valid:
            assert validation.max_diff <= 1e-4

    else:
        # skl2onnx not installed — export gracefully skipped
        assert result.error_message is not None

# ── 6. export() — infer n_features from model ──────────────────────

with tempfile.TemporaryDirectory() as tmpdir:
    auto_result = bridge.export(
        rf, "sklearn", output_path=Path(tmpdir) / "auto.onnx",
    )
    assert isinstance(auto_result, OnnxExportResult)

# ── 7. export() — unsupported framework ─────────────────────────────

with tempfile.TemporaryDirectory() as tmpdir:
    unsupported_result = bridge.export(
        rf, "unknown_framework",
        output_path=Path(tmpdir) / "unsupported.onnx",
        n_features=3,
    )
    assert unsupported_result.success is False
    assert unsupported_result.onnx_status == "skipped"

# ── 8. export() — cannot determine features ─────────────────────────

class BareModel:
    """Model without n_features_in_ attribute."""
    pass

with tempfile.TemporaryDirectory() as tmpdir:
    bare_result = bridge.export(
        BareModel(), "sklearn",
        output_path=Path(tmpdir) / "bare.onnx",
    )
    assert bare_result.success is False
    assert "features" in bare_result.error_message.lower()

# ── 9. export() returns result, never raises ─────────────────────────

with tempfile.TemporaryDirectory() as tmpdir:
    bad_result = bridge.export(
        "not_a_model", "sklearn",
        n_features=3,
        output_path=Path(tmpdir) / "bad.onnx",
    )
    assert isinstance(bad_result, OnnxExportResult)
    assert bad_result.success is False

print("PASS: 05-ml/14_onnx_bridge")
```

### Step-by-Step Explanation

1. **Bridge creation**: `OnnxBridge()` is stateless -- it does not hold any model references. Each method call is independent.

2. **Model training**: Four sklearn models (RandomForest, GradientBoosting, LogisticRegression, Ridge) are trained on the same data. All are compatible with ONNX export.

3. **Compatibility check**: `check_compatibility(rf, "sklearn")` returns `compatible=True, confidence="guaranteed"` for RandomForest. Unknown frameworks return `compatible=False, confidence="unsupported"`.

4. **Export with schema**: `export()` takes the model, framework name, optional schema (for feature dimensions), and output path. The result includes success status, file size, and export time. If `skl2onnx` is not installed, export fails gracefully.

5. **Numerical validation**: `validate()` runs both native and ONNX predictions on the same data and compares them. `max_diff` and `mean_diff` quantify the precision loss. Validation requires `onnxruntime`.

6. **Automatic feature inference**: Without a schema, OnnxBridge reads `model.n_features_in_` (set by sklearn during `.fit()`).

7. **Unsupported framework**: Export for an unknown framework returns `success=False, onnx_status="skipped"` with an error message.

8. **Missing features**: A model without `n_features_in_` and no schema/n_features parameter fails with a clear error about features.

9. **Non-fatal guarantee**: Even passing a string instead of a model does not raise an exception. `export()` always returns an `OnnxExportResult`.

## Common Mistakes

| Mistake                                        | Correct Pattern                                       | Why                                                                         |
| ---------------------------------------------- | ----------------------------------------------------- | --------------------------------------------------------------------------- |
| Expecting `export()` to raise on failure       | Check `result.success` and `result.error_message`     | Export is non-fatal by design; failures are reported in the result object   |
| Skipping `check_compatibility()` before export | Always check first for unknown model types            | Saves time and provides clear diagnostics for unsupported models            |
| Not validating after export                    | Always call `validate()` before deploying ONNX models | ONNX conversion can introduce numerical differences that affect predictions |
| Passing unfitted models                        | Call `model.fit()` before export                      | Unfitted models lack `n_features_in_` and trained parameters                |

## Exercises

1. Train a RandomForest and a Ridge model. Export both to ONNX, validate both, and compare their `max_diff` values. Which model type has better numerical precision through ONNX conversion?

2. Try exporting a model without `skl2onnx` installed (or simulate by checking the error path). Verify the result object contains a helpful error message.

3. Create a wrapper function that attempts ONNX export, falls back to native serving if export fails, and logs the decision. This is the production pattern for optional ONNX acceleration.

## Key Takeaways

- OnnxBridge exports sklearn models to ONNX for cross-runtime serving
- `check_compatibility()` verifies export feasibility before attempting conversion
- `export()` is non-fatal -- failures return a result object, never raise exceptions
- `validate()` compares native vs ONNX predictions to catch numerical precision issues
- Feature dimensions are inferred from schema, explicit `n_features`, or `model.n_features_in_`
- ONNX support is an optional optimization -- applications work without `skl2onnx` or `onnxruntime`

## Next Chapter

[Chapter 15: ML Agents](15_ml_agents.md) -- Discover the 6 ML Kaizen agents, the AgentInfusionProtocol double opt-in pattern, and the 5 mandatory guardrails.
