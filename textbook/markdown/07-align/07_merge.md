# Chapter 7: Adapter Merge

## Overview

After training a LoRA adapter, you need to **merge** it into the base model before serving. LoRA keeps adapter weights separate during training, but deployment tools (vLLM, GGUF converters, Ollama) expect a standard model with all weights baked in. The **AdapterMerger** handles this merge process and tracks the lifecycle status in the AdapterRegistry. This chapter teaches you the merge lifecycle, the merger API, and the end-to-end path from training to production.

## Prerequisites

- [Chapter 2: LoRA Configuration](02_lora_config.md)
- [Chapter 6: Adapter Registry](06_adapter_registry.md)
- Understanding of PEFT merge_and_unload concepts (helpful)

## Concepts

### Concept 1: Merge Lifecycle

Every adapter progresses through three merge statuses:

| Status     | Meaning                                 | What exists on disk           |
| ---------- | --------------------------------------- | ----------------------------- |
| `separate` | LoRA weights only (default after train) | adapter_config.json + weights |
| `merged`   | Base model + adapter combined           | Full model directory          |
| `exported` | GGUF file generated from merged model   | .gguf file                    |

- **What**: A three-stage progression tracking what form the adapter's weights are in
- **Why**: Different deployment targets need different formats -- vLLM can load merged HF models directly, Ollama needs GGUF
- **How**: `registry.update_merge_status()` advances the status; `registry.update_gguf_path()` moves to exported
- **When**: After training completes, merge before deployment

### Concept 2: AdapterMerger

The `AdapterMerger` wraps PEFT's `merge_and_unload()` operation. It requires an `AdapterRegistry` to look up adapter metadata and update merge status after completion. Internally, it loads the base model, loads the LoRA adapter, calls `merge_and_unload()`, and saves the result.

- **What**: A class that combines LoRA adapter weights with the base model
- **Why**: Deployment tools expect a standard HuggingFace model, not a base model + separate adapter
- **How**: `merger = AdapterMerger(adapter_registry=registry)` then `await merger.merge("my-adapter")`
- **When**: After training and before GGUF export or vLLM serving

### Concept 3: Why Merge Is Required

Three deployment scenarios require merged models:

1. **GGUF export**: GGUF conversion tools expect a full model, not base + adapter
2. **vLLM serving**: vLLM loads HuggingFace models directly for high-throughput inference
3. **Distribution**: Merged models are simpler to share -- one directory instead of base model + adapter

### Key API

| Method / Class    | Parameters                          | Returns         | Description                          |
| ----------------- | ----------------------------------- | --------------- | ------------------------------------ |
| `AdapterMerger()` | `adapter_registry: AdapterRegistry` | `AdapterMerger` | Create a merger with registry access |
| `merger.merge()`  | `adapter_name`, `version`           | `str` (path)    | Merge and return output path         |
| `merge_adapter()` | `adapter_name`, `adapter_registry`  | `str` (path)    | Standalone convenience function      |

## Code Walkthrough

```python
from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

from kailash_align import AdapterMerger, AdapterRegistry, AdapterSignature
from kailash_align.exceptions import MergeError
from kailash_align.registry import AdapterVersion

model_id = os.environ.get("DEFAULT_LLM_MODEL", "meta-llama/Llama-3.1-8B")
```

### Creating an AdapterMerger

```python
registry = AdapterRegistry()
merger = AdapterMerger(adapter_registry=registry)

assert merger._registry is registry
```

The merger must have access to a registry to look up adapter paths and update merge status.

### Merger Without Registry Fails

```python
no_registry_merger = AdapterMerger()

try:
    await no_registry_merger.merge("any-adapter")
    assert False, "Should have raised MergeError"
except MergeError as e:
    assert "AdapterRegistry" in str(e)
```

Attempting to merge without a registry raises `MergeError` because the merger cannot look up the adapter's path or signature.

### The Full Merge Lifecycle

```python
sig = AdapterSignature(base_model_id=model_id)
adapter = await registry.register_adapter(
    name="merge-test",
    adapter_path="/tmp/adapters/merge-test/v1",
    signature=sig,
    training_metrics={"train_loss": 0.3},
)

assert adapter.merge_status == "separate", "New adapters start as 'separate'"
assert adapter.merged_model_path is None
assert adapter.gguf_path is None
```

A freshly registered adapter is in `"separate"` status with no merged model path or GGUF path.

```python
# Stage 1: Merge (base model + LoRA adapter -> merged model)
merged = await registry.update_merge_status(
    "merge-test", "1", "merged", "/tmp/merged/merge-test"
)
assert merged.merge_status == "merged"
assert merged.merged_model_path == "/tmp/merged/merge-test"
```

After merging, the status advances to `"merged"` and the path to the merged model is recorded.

```python
# Stage 2: Export to GGUF
exported = await registry.update_gguf_path(
    "merge-test",
    "1",
    "/tmp/gguf/merge-test.gguf",
    quantization_config={"method": "q4_k_m"},
)
assert exported.merge_status == "exported"
assert exported.gguf_path == "/tmp/gguf/merge-test.gguf"
assert exported.quantization_config == {"method": "q4_k_m"}
```

GGUF export advances the status to `"exported"` and records the quantization configuration used.

### Valid Merge Status Values

```python
for valid_status in ("separate", "merged", "exported"):
    adapter2 = await registry.register_adapter(
        name=f"status-test-{valid_status}",
        adapter_path=f"/tmp/adapters/status-test-{valid_status}",
        signature=sig,
    )
    result = await registry.update_merge_status(
        f"status-test-{valid_status}", "1", valid_status
    )
    assert result.merge_status == valid_status

# Invalid merge_status
try:
    await registry.update_merge_status("merge-test", "1", "invalid")
    assert False, "Should have raised AlignmentError"
except Exception as e:
    assert "invalid" in str(e).lower()
```

Only `"separate"`, `"merged"`, and `"exported"` are valid merge statuses.

### Convenience Function

```python
from kailash_align.merge import merge_adapter

# Without registry, raises MergeError
try:
    await merge_adapter("any-adapter", adapter_registry=None)
    assert False, "Should have raised MergeError"
except MergeError:
    pass
```

`merge_adapter()` is a standalone function that wraps `AdapterMerger` for simple use cases.

### End-to-End Lifecycle: Train -> Merge -> Export -> Promote

```python
lifecycle_adapter = await registry.register_adapter(
    name="lifecycle-demo",
    adapter_path="/tmp/adapters/lifecycle",
    signature=sig,
)
assert lifecycle_adapter.merge_status == "separate"

# Stage 1: Merge
await registry.update_merge_status(
    "lifecycle-demo", "1", "merged", "/tmp/merged/lifecycle"
)
merged_adapter = await registry.get_adapter("lifecycle-demo")
assert merged_adapter.merge_status == "merged"

# Stage 2: Export to GGUF
await registry.update_gguf_path("lifecycle-demo", "1", "/tmp/gguf/lifecycle.gguf")
exported_adapter = await registry.get_adapter("lifecycle-demo")
assert exported_adapter.merge_status == "exported"
assert exported_adapter.gguf_path == "/tmp/gguf/lifecycle.gguf"

# Stage 3: Promote to production
await registry.promote("lifecycle-demo", "1", "shadow")
await registry.promote("lifecycle-demo", "1", "production")
prod = await registry.get_adapter("lifecycle-demo", version="1")
assert prod.stage == "production"
assert prod.merge_status == "exported"
```

This is the complete path: register (separate) -> merge -> export GGUF -> promote through staging gates to production. The final adapter in production has merge_status="exported" and stage="production".

## Exercises

1. Register an adapter, merge it, and export to GGUF. At each step, call `get_adapter()` and verify both `merge_status` and `stage`. What is the stage at each point?
2. Why does `AdapterMerger` require an `AdapterRegistry`? What information does it need from the registry to perform a merge?
3. In the end-to-end lifecycle, could you promote an adapter to production before merging it? Would that make sense operationally? Why or why not?

## Key Takeaways

- Merge converts separate LoRA weights + base model into a standard HuggingFace model
- The merge lifecycle has three statuses: separate -> merged -> exported
- AdapterMerger requires an AdapterRegistry to look up adapter metadata and track status
- GGUF export is a post-merge step that creates a quantized file for Ollama/llama.cpp
- The full production path is: train -> register -> merge -> export -> promote
- Merge is required for vLLM serving, GGUF export, and model distribution

## Next Chapter

[Chapter 8: Evaluation](08_evaluation.md) -- Evaluate aligned models against standard benchmarks using AlignmentEvaluator.
