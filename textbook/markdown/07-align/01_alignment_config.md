# Chapter 1: AlignmentConfig Basics

## Overview

Every fine-tuning job in Kailash Align starts with an **AlignmentConfig** -- a dataclass that declares which alignment method to use, which base model to tune, and how to configure the training run. This chapter teaches you how to create, validate, and inspect alignment configurations for all 12 built-in methods.

## Prerequisites

- Python 3.10+ installed
- Kailash Align installed (`pip install kailash-align`)
- Familiarity with LLM fine-tuning concepts (SFT, DPO, LoRA)

## Concepts

### Concept 1: AlignmentConfig as the Root Configuration

AlignmentConfig is the **top-level container** for an entire alignment job. It specifies the method (e.g., `"sft"`, `"dpo"`, `"grpo"`), the base model to fine-tune, and sub-configurations for LoRA, SFT, DPO, and other method-specific settings.

- **What**: A dataclass that holds the method name, base model ID, and all method-specific sub-configs
- **Why**: A single configuration object makes it easy to serialize, validate, and reproduce training runs
- **How**: Instantiate with `method` and `base_model_id` at minimum; sub-configs (LoRA, SFT, DPO, etc.) are auto-created with sensible defaults
- **When**: Use AlignmentConfig as the first step before calling `AlignmentPipeline.train()`

### Concept 2: Method Registry

Kailash Align ships with 12+ built-in alignment methods, tracked in `METHOD_REGISTRY`. Each method has metadata describing its category (offline, online, unpaired, monolithic), required dataset columns, and whether it needs a reward function or generation backend.

- **What**: A dictionary mapping method names to their metadata descriptors
- **Why**: The registry validates method names at config creation time, preventing typos from reaching the training loop
- **How**: `AlignmentConfig.__post_init__()` checks that the method name exists in `METHOD_REGISTRY` (with a special case for `sft_then_dpo`)
- **When**: Consult the registry when choosing between alignment approaches

### Concept 3: Auto-Created Sub-Configs

When you set `method="kto"` but do not pass a `kto=KTOConfig(...)`, the AlignmentConfig's `__post_init__` creates a default `KTOConfig()` for you. This auto-creation means you only need to override the parameters you care about.

### Concept 4: Validation

`validate()` returns a list of warning strings. An empty list means the config is ready for training. Warnings flag missing but recommended settings -- for example, GRPO without `reward_funcs` produces a warning because GRPO requires reward functions to score completions.

### Key API

| Method / Class               | Parameters                                                    | Returns           | Description                                   |
| ---------------------------- | ------------------------------------------------------------- | ----------------- | --------------------------------------------- |
| `AlignmentConfig()`          | `method: str`, `base_model_id: str`, `lora`, `sft`, `dpo` ... | `AlignmentConfig` | Create a root alignment configuration         |
| `config.validate()`          | --                                                            | `list[str]`       | Return warnings (empty = valid)               |
| `config.get_method_config()` | `method: str`                                                 | sub-config        | Dispatch to the correct method sub-config     |
| `METHOD_REGISTRY`            | --                                                            | `dict`            | All registered alignment methods and metadata |

## Code Walkthrough

```python
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

from kailash_align import AlignmentConfig
from kailash_align.config import LoRAConfig, SFTConfig, DPOConfig
from kailash_align.method_registry import METHOD_REGISTRY
```

We import the core config classes and the method registry. `load_dotenv()` loads environment variables so we can use `DEFAULT_LLM_MODEL` for the base model.

### Creating a Basic Config

```python
model_id = os.environ.get("DEFAULT_LLM_MODEL", "meta-llama/Llama-3.1-8B")

config = AlignmentConfig(
    method="sft",
    base_model_id=model_id,
)

assert config.method == "sft"
assert config.base_model_id == model_id
assert isinstance(config.lora, LoRAConfig), "LoRA config auto-populated"
assert isinstance(config.sft, SFTConfig), "SFT config auto-populated"
```

Even though we only specified `method` and `base_model_id`, the config automatically creates default `LoRAConfig` and `SFTConfig` instances. This is the auto-creation behavior.

### Validation: Missing base_model_id

```python
try:
    AlignmentConfig(method="sft", base_model_id="")
    assert False, "Should have raised ValueError"
except ValueError:
    pass  # Expected: base_model_id is required
```

An empty `base_model_id` is rejected at creation time -- you must specify which model to fine-tune.

### Validation: Invalid Method

```python
try:
    AlignmentConfig(method="nonexistent_method", base_model_id=model_id)
    assert False, "Should have raised ValueError"
except ValueError:
    pass  # Expected: unknown training method
```

Method names are validated against `METHOD_REGISTRY`. Typos are caught immediately.

### Iterating Over All Registered Methods

```python
registered_methods = sorted(METHOD_REGISTRY.keys())
assert len(registered_methods) >= 11, f"Expected at least 11 methods, got {len(registered_methods)}"

for method_name in ["sft", "dpo", "kto", "orpo", "grpo", "rloo", "online_dpo"]:
    cfg = AlignmentConfig(method=method_name, base_model_id=model_id)
    assert cfg.method == method_name
```

All registered methods produce valid configs. The registry contains at least 11 methods including SFT, DPO, KTO, ORPO, GRPO, RLOO, and OnlineDPO.

### The sft_then_dpo Combo Method

```python
combo_config = AlignmentConfig(
    method="sft_then_dpo",
    base_model_id=model_id,
)
assert combo_config.method == "sft_then_dpo"
assert isinstance(combo_config.sft, SFTConfig)
assert isinstance(combo_config.dpo, DPOConfig)
```

The special `sft_then_dpo` method runs SFT first, then DPO on the SFT output. It auto-creates both sub-configs.

### validate() Checks for Missing Configs

```python
grpo_no_reward = AlignmentConfig(
    method="grpo",
    base_model_id=model_id,
)
warnings = grpo_no_reward.validate()
assert any("reward_funcs" in w for w in warnings), "GRPO without reward_funcs should produce a warning"
```

GRPO requires reward functions. Without them, `validate()` returns a warning rather than raising an error -- this lets you build configs incrementally.

```python
sft_config = AlignmentConfig(
    method="sft",
    base_model_id=model_id,
)
warnings = sft_config.validate()
assert warnings == [], f"SFT config should be valid, got warnings: {warnings}"
```

A fully-specified SFT config validates cleanly with no warnings.

### get_method_config() Dispatches to Sub-Configs

```python
cfg = AlignmentConfig(
    method="dpo",
    base_model_id=model_id,
    dpo=DPOConfig(beta=0.2, learning_rate=3e-5),
)
dpo_sub = cfg.get_method_config("dpo")
assert dpo_sub is cfg.dpo
assert dpo_sub.beta == 0.2

# For unknown methods, falls back to SFT config
sft_fallback = cfg.get_method_config("cpo")
assert sft_fallback is cfg.sft, "Experimental methods fall back to SFT config"
```

`get_method_config()` returns the sub-config for the given method. Experimental methods that do not have a dedicated sub-config fall back to the SFT config.

### Additional Options

```python
config_opts = AlignmentConfig(
    method="sft",
    base_model_id=model_id,
    experiment_dir="/tmp/my-experiment",
    local_files_only=True,
    base_model_revision="abc123",
)
assert config_opts.experiment_dir == "/tmp/my-experiment"
assert config_opts.local_files_only is True
assert config_opts.base_model_revision == "abc123"
```

AlignmentConfig also accepts `experiment_dir` (where to store checkpoints), `local_files_only` (skip HuggingFace Hub downloads), and `base_model_revision` (pin a specific model commit).

## Exercises

1. Create an AlignmentConfig for the `"kto"` method and verify that a KTOConfig is auto-created. Then call `validate()` and confirm it returns an empty list.
2. Loop over all keys in `METHOD_REGISTRY` and create an AlignmentConfig for each. Which methods produce validation warnings with default settings?
3. Create an `sft_then_dpo` config with custom SFT learning rate of `1e-4` and DPO beta of `0.2`. Use `get_method_config()` to retrieve each sub-config and verify your custom values.

## Key Takeaways

- `AlignmentConfig` is the root configuration for all fine-tuning jobs in Kailash Align
- `base_model_id` is required; `method` must be a registered method name or `sft_then_dpo`
- Sub-configs (LoRA, SFT, DPO, KTO, etc.) are auto-created with sensible defaults
- `validate()` returns warnings without raising errors, enabling incremental config building
- `get_method_config()` dispatches to the right sub-config, with SFT as the fallback

## Next Chapter

[Chapter 2: LoRA Configuration](02_lora_config.md) -- Learn how to configure Low-Rank Adaptation parameters including rank, alpha, target modules, and dropout.
