# Chapter 6: Adapter Registry

## Overview

After training a LoRA adapter, you need to track it -- its version, training metrics, evaluation results, and lifecycle stage. The **AdapterRegistry** provides versioned adapter management with stage promotion (staging, shadow, production, archived), merge status tracking, and capacity bounds. This chapter teaches you how to register, query, promote, and delete adapters through the full lifecycle.

## Prerequisites

- [Chapter 1: AlignmentConfig Basics](01_alignment_config.md)
- [Chapter 2: LoRA Configuration](02_lora_config.md)
- Understanding of model versioning concepts

## Concepts

### Concept 1: AdapterSignature

An `AdapterSignature` describes the characteristics of an adapter: which base model it was trained on, the LoRA configuration (rank, alpha, target modules), the training method, and the task type. Signatures are attached to every registered version.

- **What**: A descriptor dataclass capturing the adapter's provenance and configuration
- **Why**: Without signatures, you cannot verify compatibility when loading an adapter onto a base model
- **How**: `AdapterSignature(base_model_id="meta-llama/Llama-3.1-8B", adapter_type="lora", rank=16)`
- **When**: Create one before registering any adapter version

### Concept 2: Version Auto-Increment

Each call to `register_adapter()` with the same name auto-increments the version number. Version 1 is assigned on first registration; subsequent registrations produce 2, 3, and so on. Versions are string-typed for flexibility.

- **What**: Automatic sequential versioning within each adapter name
- **Why**: Manual version management is error-prone and creates gaps or collisions
- **How**: Call `register_adapter("my-adapter", path, signature)` multiple times -- versions auto-increment
- **When**: Every time you train a new iteration of the same adapter

### Concept 3: Stage Promotion (Monotonic)

Adapters progress through stages: `staging` (default) -> `shadow` -> `production` -> `archived`. Promotion is **monotonic** -- you can only move forward, never backward. This prevents production models from being accidentally demoted.

- **What**: A one-way lifecycle progression through four stages
- **Why**: Monotonic promotion ensures that once an adapter reaches production, it cannot silently revert to an earlier state
- **How**: `registry.promote("my-adapter", "1", "shadow")` then `promote(..., "production")`
- **When**: After validation passes at each gate

### Concept 4: Capacity Bounds

AdapterRegistry accepts `max_adapters` and `max_versions_per_adapter` to prevent unbounded growth. Exceeding these limits raises `AlignmentError`.

### Key API

| Method                  | Parameters                                                      | Returns           | Description                      |
| ----------------------- | --------------------------------------------------------------- | ----------------- | -------------------------------- |
| `AdapterRegistry()`     | `max_adapters`, `max_versions_per_adapter`                      | `AdapterRegistry` | Create a registry with bounds    |
| `register_adapter()`    | `name`, `adapter_path`, `signature`, `training_metrics`, `tags` | `AdapterVersion`  | Register a new version           |
| `get_adapter()`         | `name`, `version=None`                                          | `AdapterVersion`  | Get latest or specific version   |
| `list_adapters()`       | `base_model_id`, `tags`, `stage`                                | `list`            | Filter adapters                  |
| `promote()`             | `name`, `version`, `target_stage`                               | `AdapterVersion`  | Move to next stage (monotonic)   |
| `update_merge_status()` | `name`, `version`, `status`, `merged_model_path`                | `AdapterVersion`  | Track merge lifecycle            |
| `update_gguf_path()`    | `name`, `version`, `gguf_path`, `quantization_config`           | `AdapterVersion`  | Record GGUF export               |
| `update_eval_results()` | `name`, `version`, `results`                                    | `AdapterVersion`  | Attach benchmark scores          |
| `delete_adapter()`      | `name`, `version=None`                                          | `None`            | Delete version or entire adapter |

## Code Walkthrough

```python
from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

from kailash_align import AdapterRegistry, AdapterSignature
from kailash_align.exceptions import AdapterNotFoundError, AlignmentError
from kailash_align.registry import AdapterVersion

model_id = os.environ.get("DEFAULT_LLM_MODEL", "meta-llama/Llama-3.1-8B")
```

The registry API is async because production registries may back onto databases or remote storage.

### Creating a Registry

```python
registry = AdapterRegistry()

# Registry accepts optional capacity bounds
bounded_registry = AdapterRegistry(
    max_adapters=100,
    max_versions_per_adapter=10,
)
```

An unbounded registry has no limits. In production, always set bounds to prevent runaway storage growth.

### AdapterSignature

```python
signature = AdapterSignature(
    base_model_id=model_id,
    adapter_type="lora",
    rank=16,
    alpha=32,
    target_modules=("q_proj", "v_proj"),
    task_type="CAUSAL_LM",
    training_method="sft",
)

assert signature.base_model_id == model_id
assert signature.rank == 16
assert signature.alpha == 32
assert signature.adapter_type == "lora"
```

Signatures validate their fields: `base_model_id` must be non-empty, `adapter_type` must be `"lora"` or `"qlora"`, `rank` must be >= 1, `target_modules` must be non-empty, and `training_method` must be in the METHOD_REGISTRY.

### Registering Adapter Versions

```python
version1 = await registry.register_adapter(
    name="my-sft-adapter",
    adapter_path="/tmp/adapters/my-sft-adapter/v1",
    signature=signature,
    training_metrics={"train_loss": 0.45, "eval_loss": 0.52},
    tags=["sft", "experiment-1"],
)

assert isinstance(version1, AdapterVersion)
assert version1.adapter_name == "my-sft-adapter"
assert version1.version == "1"
assert version1.stage == "staging"
assert version1.merge_status == "separate"
assert version1.training_metrics["train_loss"] == 0.45
```

New adapters start in `"staging"` with merge status `"separate"`. Training metrics and tags are optional metadata.

```python
version2 = await registry.register_adapter(
    name="my-sft-adapter",
    adapter_path="/tmp/adapters/my-sft-adapter/v2",
    signature=signature,
    training_metrics={"train_loss": 0.38, "eval_loss": 0.44},
)

assert version2.version == "2", "Second version auto-incremented"
```

Registering the same name again auto-increments to version "2".

### Querying Adapters

```python
latest = await registry.get_adapter("my-sft-adapter")
assert latest.version == "2", "Default: returns latest version"

specific = await registry.get_adapter("my-sft-adapter", version="1")
assert specific.version == "1"
```

`get_adapter()` returns the latest version by default. Pass `version` to retrieve a specific one. Non-existent adapters or versions raise `AdapterNotFoundError`.

### Listing with Filters

```python
all_adapters = await registry.list_adapters()
assert len(all_adapters) == 1, "One adapter registered"

by_model = await registry.list_adapters(base_model_id=model_id)
assert len(by_model) == 1

by_tags = await registry.list_adapters(tags=["sft"])
assert len(by_tags) == 1
```

Filter by base model, tags, or stage to find matching adapters.

### Stage Promotion

```python
promoted = await registry.promote("my-sft-adapter", "1", "shadow")
assert promoted.stage == "shadow"

promoted = await registry.promote("my-sft-adapter", "1", "production")
assert promoted.stage == "production"

# Cannot go backward
try:
    await registry.promote("my-sft-adapter", "1", "staging")
    assert False, "Should have raised AlignmentError"
except AlignmentError:
    pass  # Expected: backward transition not allowed
```

Promotion is monotonic: staging -> shadow -> production -> archived. Backward transitions and same-stage promotions are both rejected.

### Merge Status and GGUF Export

```python
merged = await registry.update_merge_status(
    "my-sft-adapter", "1", "merged", "/tmp/merged/my-sft-adapter"
)
assert merged.merge_status == "merged"
assert merged.merged_model_path == "/tmp/merged/my-sft-adapter"

exported = await registry.update_gguf_path(
    "my-sft-adapter",
    "1",
    "/tmp/gguf/my-sft-adapter.gguf",
    quantization_config={"method": "q4_k_m"},
)
assert exported.gguf_path == "/tmp/gguf/my-sft-adapter.gguf"
assert exported.merge_status == "exported"
```

The merge lifecycle tracks an adapter from `separate` (LoRA weights only) through `merged` (base + adapter combined) to `exported` (GGUF file ready for serving).

### Evaluation Results

```python
evaled = await registry.update_eval_results(
    "my-sft-adapter",
    "2",
    {"arc_easy": 0.72, "hellaswag": 0.65},
)
assert evaled.eval_results == {"arc_easy": 0.72, "hellaswag": 0.65}
```

Attach benchmark scores to any version. These are used for promotion decisions.

### Deletion

```python
await registry.delete_adapter("my-sft-adapter", version="2")
# Version 2 is gone, version 1 still exists

await registry.delete_adapter("my-sft-adapter")
# Entire adapter deleted (all versions)
```

Delete a specific version or an entire adapter with all its versions.

### Capacity Bounds

```python
small_registry = AdapterRegistry(max_adapters=2, max_versions_per_adapter=2)

sig = AdapterSignature(base_model_id=model_id)

await small_registry.register_adapter("a1", "/tmp/a1", sig)
await small_registry.register_adapter("a2", "/tmp/a2", sig)

# Third adapter exceeds max_adapters
try:
    await small_registry.register_adapter("a3", "/tmp/a3", sig)
    assert False, "Should have raised AlignmentError"
except AlignmentError as e:
    assert "maximum" in str(e).lower()
```

Capacity bounds prevent unbounded growth. Both total adapters and versions per adapter are capped.

## Exercises

1. Create a registry with `max_adapters=5`. Register three adapters with different names and promote one to production. List adapters filtered by `stage="production"` and verify you get exactly one result.
2. Register two versions of the same adapter, update eval results on both, and compare their scores. Which version would you promote to production?
3. Walk an adapter through the full lifecycle: register -> merge -> export GGUF -> promote to production. Verify the merge_status and stage at each step.

## Key Takeaways

- AdapterRegistry provides versioned adapter management with auto-incrementing versions
- AdapterSignature captures provenance (base model, LoRA config, training method)
- Stage promotion is monotonic: staging -> shadow -> production -> archived
- Merge status tracks the adapter from separate LoRA weights through merged model to GGUF export
- Capacity bounds prevent unbounded storage growth
- All operations are async for production database compatibility

## Next Chapter

[Chapter 7: Adapter Merge](07_merge.md) -- Learn how to merge LoRA adapters into base models using the AdapterMerger.
