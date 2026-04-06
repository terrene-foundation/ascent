# Chapter 5: Advanced Alignment Methods

## Overview

Beyond SFT and DPO, Kailash Align supports five additional alignment methods, each addressing different data availability and optimization constraints. **KTO** works with unpaired binary feedback. **ORPO** combines SFT and preference alignment in one pass. **GRPO** and **RLOO** are online RL methods that generate completions and score them with reward functions. **OnlineDPO** applies DPO loss to online-generated pairs. This chapter teaches you how to configure each method and when to choose one over another.

## Prerequisites

- [Chapter 1: AlignmentConfig Basics](01_alignment_config.md)
- [Chapter 4: DPO Configuration](04_dpo_config.md)
- Understanding of online vs offline RL concepts (helpful)

## Concepts

### Concept 1: Method Categories

Kailash Align organizes methods into four categories:

| Category     | Methods               | Data Requirement                     | Reward Function? |
| ------------ | --------------------- | ------------------------------------ | ---------------- |
| `offline`    | DPO                   | Preference pairs (chosen/rejected)   | No               |
| `unpaired`   | KTO                   | Binary labels (good/bad)             | No               |
| `monolithic` | ORPO                  | Preference pairs (one-pass SFT+pref) | No               |
| `online`     | GRPO, RLOO, OnlineDPO | Prompts only (generates completions) | GRPO/RLOO: Yes   |

- **What**: A taxonomy that determines what data and infrastructure each method needs
- **Why**: Choosing the right category depends on your data availability and compute budget
- **How**: Check `METHOD_REGISTRY[method].category` for any method's category
- **When**: Use offline methods when you have preference data; online methods when you have reward functions and prompts

### Concept 2: Reward Functions (Online Methods)

GRPO and RLOO require reward functions to score generated completions. These functions are registered in the `RewardRegistry` and called during training to evaluate each completion. OnlineDPO does not require reward functions -- it uses the model's own log probabilities.

### Concept 3: Generation Backend

Online methods (GRPO, RLOO, OnlineDPO) generate completions during training. They can use either vLLM (`use_vllm=True`) for faster generation or HuggingFace's generate function. The `num_generations` parameter controls how many completions are generated per prompt.

### Key API

| Config Class      | Category   | Key Parameters                                          |
| ----------------- | ---------- | ------------------------------------------------------- |
| `KTOConfig`       | unpaired   | `beta`, `desirable_weight`, `undesirable_weight`        |
| `ORPOConfig`      | monolithic | `beta`, `max_length`, `max_prompt_length`               |
| `GRPOConfig`      | online     | `num_generations`, `temperature`, `kl_coef`, `use_vllm` |
| `RLOOConfig`      | online     | `num_generations`, `temperature`, `kl_coef`, `use_vllm` |
| `OnlineDPOConfig` | online     | `beta`, `max_completion_length`, `use_vllm`             |

## Code Walkthrough

```python
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

from kailash_align import (
    AlignmentConfig,
    GRPOConfig,
    KTOConfig,
    OnlineDPOConfig,
    ORPOConfig,
    RLOOConfig,
)
from kailash_align.method_registry import METHOD_REGISTRY

model_id = os.environ.get("DEFAULT_LLM_MODEL", "meta-llama/Llama-3.1-8B")
```

### KTO -- Unpaired Binary Feedback

```python
kto = KTOConfig()

assert kto.learning_rate == 5e-7, "KTO paper recommends very low LR"
assert kto.beta == 0.1
assert kto.desirable_weight == 1.0
assert kto.undesirable_weight == 1.0
assert kto.max_length == 1024
assert kto.max_prompt_length == 512
```

KTO works with `(prompt, completion, label)` triples where label is binary (good/bad). This dramatically lowers the data barrier -- you do not need paired preferences (chosen vs rejected), just a thumbs-up or thumbs-down per completion. Note the very low default learning rate (5e-7), which is 10x lower than DPO's default.

```python
# Custom KTO with asymmetric weighting
kto_custom = KTOConfig(
    desirable_weight=1.5,
    undesirable_weight=0.5,
    learning_rate=1e-6,
    beta=0.05,
)
assert kto_custom.desirable_weight == 1.5
assert kto_custom.undesirable_weight == 0.5
```

Asymmetric weighting lets you emphasize learning from good examples (higher desirable_weight) or bad examples (higher undesirable_weight). Weights must be positive.

```python
kto_method = METHOD_REGISTRY["kto"]
assert kto_method.category == "unpaired"
assert kto_method.requires_preference_data is False
assert "prompt" in kto_method.dataset_required_columns
assert "completion" in kto_method.dataset_required_columns
assert "label" in kto_method.dataset_required_columns
```

KTO's dataset requires `prompt`, `completion`, and `label` columns -- not the `chosen`/`rejected` columns that DPO needs.

### ORPO -- Monolithic SFT + Preference

```python
orpo = ORPOConfig()

assert orpo.learning_rate == 8e-6, "ORPO paper recommends 8e-6"
assert orpo.beta == 0.1
assert orpo.max_length == 1024
assert orpo.max_prompt_length == 512
```

ORPO combines SFT and preference alignment in one training pass, eliminating the need for a two-stage `sft_then_dpo` pipeline. It uses the same preference data format as DPO but learns the task format and preferences simultaneously.

```python
orpo_method = METHOD_REGISTRY["orpo"]
assert orpo_method.category == "monolithic"
assert orpo_method.requires_preference_data is True
assert "prompt" in orpo_method.dataset_required_columns
assert "chosen" in orpo_method.dataset_required_columns
assert "rejected" in orpo_method.dataset_required_columns
```

### GRPO -- Online RL (DeepSeek-R1 Method)

```python
grpo = GRPOConfig()

assert grpo.num_generations == 4, "Default: 4 completions per prompt"
assert grpo.temperature == 0.7
assert grpo.max_completion_length == 2048
assert grpo.learning_rate == 1e-5
assert grpo.kl_coef == 0.001
assert grpo.use_vllm is False
assert grpo.vllm_gpu_utilization == 0.5
```

GRPO (Group Relative Policy Optimization) generates multiple completions per prompt and scores them with reward functions. The `num_generations` parameter controls how many completions are generated. `kl_coef` penalizes divergence from the reference policy (similar to beta in DPO). `use_vllm=True` enables faster generation via vLLM.

```python
# Custom GRPO for single-GPU training
grpo_custom = GRPOConfig(
    num_generations=8,
    temperature=1.0,
    kl_coef=0.01,
    max_completion_length=1024,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
)
assert grpo_custom.num_generations == 8
assert grpo_custom.temperature == 1.0
assert grpo_custom.kl_coef == 0.01
```

On a single GPU, use batch size 1 with high accumulation to fit the multiple generations in memory.

```python
grpo_method = METHOD_REGISTRY["grpo"]
assert grpo_method.category == "online"
assert grpo_method.requires_reward_func is True
assert grpo_method.requires_generation_backend is True
assert "prompt" in grpo_method.dataset_required_columns
```

GRPO requires reward functions and a generation backend, but only `prompt` in the dataset -- completions are generated during training.

### RLOO -- REINFORCE Leave-One-Out

```python
rloo = RLOOConfig()

assert rloo.num_generations == 4
assert rloo.temperature == 0.7
assert rloo.kl_coef == 0.001
assert rloo.use_vllm is False
```

RLOO uses the same infrastructure as GRPO but with a leave-one-out baseline for variance reduction. The configuration parameters are identical.

```python
rloo_method = METHOD_REGISTRY["rloo"]
assert rloo_method.category == "online"
assert rloo_method.requires_reward_func is True
```

### OnlineDPO -- DPO with Online Generation

```python
online_dpo = OnlineDPOConfig()

assert online_dpo.beta == 0.1
assert online_dpo.max_length == 2048
assert online_dpo.max_prompt_length == 512
assert online_dpo.max_completion_length == 512
assert online_dpo.use_vllm is False
```

OnlineDPO generates completions online and applies DPO loss. Unlike GRPO/RLOO, it does not require reward functions -- it uses the model's own log probabilities to create preference pairs.

```python
online_dpo_method = METHOD_REGISTRY["online_dpo"]
assert online_dpo_method.category == "online"
assert online_dpo_method.requires_reward_func is False
assert online_dpo_method.requires_generation_backend is True
```

### Shared Validation Rules

All five configs enforce bf16/fp16 mutual exclusion and frozen immutability:

```python
for ConfigClass in (KTOConfig, ORPOConfig, GRPOConfig, RLOOConfig, OnlineDPOConfig):
    try:
        ConfigClass(bf16=True, fp16=True)
        assert False, f"{ConfigClass.__name__} should reject bf16+fp16"
    except ValueError as e:
        assert "bf16" in str(e) and "fp16" in str(e)

for config_obj in (kto, orpo, grpo, rloo, online_dpo):
    try:
        config_obj.learning_rate = 999.0  # type: ignore[misc]
        assert False, f"{type(config_obj).__name__} should be frozen"
    except AttributeError:
        pass  # Expected: all frozen
```

### Auto-Creation in AlignmentConfig

```python
for method in ("kto", "orpo", "grpo", "rloo", "online_dpo"):
    cfg = AlignmentConfig(method=method, base_model_id=model_id)
    sub = getattr(cfg, method)
    assert sub is not None, f"{method} config should be auto-created"
```

All method-specific configs are auto-created when you set the method on AlignmentConfig, just like SFT and DPO.

## Exercises

1. You have a dataset of 10,000 customer support responses, each labeled "helpful" or "unhelpful" (no paired preferences). Which method should you use? Create the config.
2. Create a GRPOConfig for a single T4 GPU (16GB). What values would you choose for `num_generations`, `per_device_train_batch_size`, and `gradient_accumulation_steps`?
3. Compare the dataset requirements for DPO, KTO, and GRPO by inspecting their `dataset_required_columns` in METHOD_REGISTRY. Which method requires the least annotation effort?

## Key Takeaways

- KTO lowers the data barrier by using binary feedback instead of preference pairs
- ORPO eliminates the two-stage pipeline by combining SFT and preference alignment
- GRPO and RLOO are online RL methods that generate and score completions during training
- OnlineDPO generates completions online but does not need reward functions
- All configs share validation rules: bf16/fp16 mutual exclusion, frozen immutability, positive learning rates
- Method-specific configs are auto-created in AlignmentConfig

## Next Chapter

[Chapter 6: Adapter Registry](06_adapter_registry.md) -- Learn how to version, manage, and promote LoRA adapters through their lifecycle.
