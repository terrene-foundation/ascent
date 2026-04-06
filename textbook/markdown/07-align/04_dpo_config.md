# Chapter 4: DPO Configuration

## Overview

Direct Preference Optimization (DPO) aligns models using pairs of preferred and rejected responses. Instead of training a separate reward model, DPO directly optimizes the policy using a closed-form loss derived from the Bradley-Terry preference model. The **beta** parameter controls how much the fine-tuned model is allowed to deviate from the reference policy. This chapter teaches you how to configure DPO training, understand beta's role, manage prompt/completion length budgets, and use loss type variants.

## Prerequisites

- [Chapter 1: AlignmentConfig Basics](01_alignment_config.md)
- [Chapter 3: SFT Configuration](03_sft_config.md)
- Understanding of preference data format (prompt, chosen, rejected)

## Concepts

### Concept 1: Beta -- The KL Divergence Penalty

Beta controls how far the trained model can deviate from the reference (base) model. Lower beta (e.g., 0.01) allows more aggressive optimization -- the model can diverge significantly from its starting behavior. Higher beta (e.g., 0.5) keeps the model closer to the reference, producing more conservative changes.

- **What**: A positive float that scales the KL divergence penalty in the DPO loss
- **Why**: Without beta, the model could overfit to the preference data and lose its general capabilities
- **How**: `DPOConfig(beta=0.1)` is the default. Lower for stronger alignment, higher for safety
- **When**: Start with 0.1; decrease if the model is not learning preferences, increase if it degrades on general tasks

### Concept 2: Prompt and Completion Length Budgets

`max_length` is the total token budget for the entire sequence (prompt + completion). `max_prompt_length` caps the prompt portion. The difference (`max_length - max_prompt_length`) is the completion budget -- the space available for the chosen and rejected completions.

- **What**: Two integers that partition the sequence length budget between prompt and completion
- **Why**: Prompts that consume the entire budget leave no room for completions, making training useless
- **How**: `DPOConfig(max_length=2048, max_prompt_length=512)` gives 1536 tokens for completions
- **When**: Set based on your dataset's prompt and completion length distributions

### Concept 3: Loss Type Variants

Kailash Align supports DPO loss variants through the `loss_type` field on AlignmentConfig. Standard DPO uses the original loss. IPO (Identity Preference Optimization) and SimPO are alternative formulations with different theoretical properties.

### Key API

| Field                    | Type    | Default | Description                                        |
| ------------------------ | ------- | ------- | -------------------------------------------------- |
| `beta`                   | `float` | `0.1`   | KL penalty strength (must be positive, finite)     |
| `num_train_epochs`       | `int`   | `1`     | Epochs (DPO typically needs fewer than SFT)        |
| `learning_rate`          | `float` | `5e-5`  | Lower than SFT default (preference signal is weak) |
| `max_length`             | `int`   | `2048`  | Total sequence length budget                       |
| `max_prompt_length`      | `int`   | `512`   | Maximum prompt tokens                              |
| `warmup_ratio`           | `float` | `0.1`   | Higher warmup than SFT (preference stability)      |
| `gradient_checkpointing` | `bool`  | `True`  | DPO holds two models in memory, so this helps      |

## Code Walkthrough

```python
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

from kailash_align import AlignmentConfig, DPOConfig
from kailash_align.method_registry import METHOD_REGISTRY
```

### Default DPO Configuration

```python
default = DPOConfig()

assert default.num_train_epochs == 1
assert default.per_device_train_batch_size == 4
assert default.gradient_accumulation_steps == 4
assert default.learning_rate == 5e-5
assert default.warmup_ratio == 0.1
assert default.max_length == 2048
assert default.max_prompt_length == 512
assert default.beta == 0.1
assert default.gradient_checkpointing is True
assert default.bf16 is True
assert default.fp16 is False
```

Notice the differences from SFTConfig: DPO defaults to 1 epoch (not 3), lower learning rate (5e-5 vs 2e-4), and higher warmup ratio (0.1 vs 0.03). These reflect the fact that DPO's preference signal is weaker and more sensitive to hyperparameters.

### Beta: Conservative vs Aggressive

```python
conservative = DPOConfig(beta=0.5)
assert conservative.beta == 0.5

aggressive = DPOConfig(beta=0.01)
assert aggressive.beta == 0.01
```

Conservative beta (0.5) keeps the model close to its starting behavior -- good when you want subtle adjustments. Aggressive beta (0.01) allows large behavioral changes -- use when you have high-quality preference data and want strong alignment.

### Beta Validation

```python
try:
    DPOConfig(beta=0.0)
    assert False, "Should have raised ValueError"
except ValueError:
    pass  # Expected: beta must be positive

try:
    DPOConfig(beta=-0.1)
    assert False, "Should have raised ValueError"
except ValueError:
    pass  # Expected: beta must be positive

try:
    DPOConfig(beta=float("nan"))
    assert False, "Should have raised ValueError"
except ValueError:
    pass  # Expected: NaN rejected
```

Beta must be positive and finite. Zero beta would eliminate the KL penalty entirely, and negative beta would invert the optimization objective.

### Prompt and Completion Length Budget

```python
cfg = DPOConfig(max_length=2048, max_prompt_length=512)
assert cfg.max_prompt_length < cfg.max_length, "Prompt length should be less than total max length"
completion_budget = cfg.max_length - cfg.max_prompt_length
assert completion_budget == 1536, "Completion budget = max_length - max_prompt_length"
```

With max_length=2048 and max_prompt_length=512, you have 1536 tokens for completions. If your preference pairs have long completions, increase max_length or decrease max_prompt_length.

### DPO Loss Type Variants

```python
model_id = os.environ.get("DEFAULT_LLM_MODEL", "meta-llama/Llama-3.1-8B")

# Standard DPO
standard_dpo = AlignmentConfig(
    method="dpo",
    base_model_id=model_id,
    dpo=DPOConfig(beta=0.1),
)
assert standard_dpo.loss_type is None, "Default: standard DPO loss"

# IPO variant (Identity Preference Optimization)
ipo_config = AlignmentConfig(
    method="dpo",
    base_model_id=model_id,
    dpo=DPOConfig(beta=0.1),
    loss_type="ipo",
)
assert ipo_config.loss_type == "ipo"

# SimPO variant
simpo_config = AlignmentConfig(
    method="dpo",
    base_model_id=model_id,
    loss_type="simpo",
)
assert simpo_config.loss_type == "simpo"
```

Loss type variants are set on AlignmentConfig (not DPOConfig) because they affect how the DPO trainer computes the loss. `None` means standard DPO.

### DPO Method Registry Metadata

```python
dpo_method = METHOD_REGISTRY["dpo"]
assert dpo_method.requires_preference_data is True
assert dpo_method.supports_loss_type is True
assert dpo_method.requires_reward_func is False
assert dpo_method.category == "offline"
assert "prompt" in dpo_method.dataset_required_columns
assert "chosen" in dpo_method.dataset_required_columns
assert "rejected" in dpo_method.dataset_required_columns
```

DPO is an offline method -- it uses pre-collected preference data (prompt, chosen, rejected) rather than generating completions online. It does not require a reward function.

### The sft_then_dpo Combo

```python
from kailash_align import SFTConfig

combo = AlignmentConfig(
    method="sft_then_dpo",
    base_model_id=model_id,
    sft=SFTConfig(num_train_epochs=2, learning_rate=2e-4),
    dpo=DPOConfig(num_train_epochs=1, beta=0.1),
)
assert combo.sft.num_train_epochs == 2
assert combo.dpo.beta == 0.1
```

The `sft_then_dpo` method runs SFT first to teach the model the task format, then DPO to align its preferences. This is the most common two-stage alignment pipeline.

## Common Mistakes

| Mistake                         | Correct Pattern                         | Why                                                                                 |
| ------------------------------- | --------------------------------------- | ----------------------------------------------------------------------------------- |
| Beta too low (< 0.01)           | Start with `beta=0.1`                   | Very low beta causes reward hacking -- the model exploits the preference signal     |
| max_prompt_length >= max_length | Ensure `max_prompt_length < max_length` | No room for completions means the model cannot learn from the chosen/rejected pairs |
| DPO learning rate same as SFT   | Use lower LR for DPO (5e-5 default)     | DPO's preference signal is weaker; high LR causes training instability              |
| Skipping SFT before DPO         | Use `sft_then_dpo` for best results     | DPO works best when the model already understands the task format from SFT          |

## Exercises

1. Create a DPOConfig with `max_length=4096` and `max_prompt_length=1024`. What is the completion budget? When would you need this configuration?
2. The default DPO learning rate (5e-5) is 4x lower than SFT's default (2e-4). Create both configs and verify this ratio. Why is DPO more sensitive to learning rate?
3. Create an `sft_then_dpo` AlignmentConfig where SFT trains for 3 epochs with learning rate 2e-4, and DPO trains for 1 epoch with beta 0.2. Use `get_method_config()` to retrieve each sub-config.

## Key Takeaways

- DPO aligns models using preference pairs (prompt, chosen, rejected) without a separate reward model
- Beta controls KL divergence penalty: lower = more aggressive, higher = more conservative
- `max_length - max_prompt_length` defines the completion budget
- DPO defaults to lower learning rate and fewer epochs than SFT
- Loss type variants (IPO, SimPO) are set on AlignmentConfig, not DPOConfig
- `sft_then_dpo` is the recommended two-stage alignment pipeline

## Next Chapter

[Chapter 5: Advanced Alignment Methods](05_advanced_methods.md) -- Explore KTO, ORPO, GRPO, RLOO, and OnlineDPO for specialized alignment scenarios.
