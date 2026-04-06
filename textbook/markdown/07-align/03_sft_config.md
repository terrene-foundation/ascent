# Chapter 3: SFT Configuration

## Overview

Supervised Fine-Tuning (SFT) is the most common alignment method -- you show the model examples of desired behavior and train it to reproduce them. `SFTConfig` controls the training hyperparameters: learning rate, batch size, sequence length, precision, and gradient management. This chapter teaches you how to configure SFT runs and understand the tradeoffs between memory, speed, and quality.

## Prerequisites

- [Chapter 1: AlignmentConfig Basics](01_alignment_config.md)
- [Chapter 2: LoRA Configuration](02_lora_config.md)
- Basic understanding of gradient descent and batch training

## Concepts

### Concept 1: Effective Batch Size

The actual number of examples the optimizer sees per step is `per_device_train_batch_size * gradient_accumulation_steps * num_gpus`. Gradient accumulation lets you simulate large batches on memory-constrained hardware by accumulating gradients over multiple forward passes before updating weights.

- **What**: The product of per-device batch size, gradient accumulation steps, and GPU count
- **Why**: Larger effective batch sizes produce more stable gradients but require more memory (or more accumulation steps)
- **How**: `SFTConfig(per_device_train_batch_size=4, gradient_accumulation_steps=8)` gives an effective batch of 32 on one GPU
- **When**: Start with the default (4 \* 4 = 16), increase accumulation if you need larger batches without more memory

### Concept 2: Precision (bf16 vs fp16)

`bf16=True` uses BFloat16 precision (broader dynamic range, less precision), while `fp16=True` uses Float16 (narrower range, higher precision). They are **mutually exclusive** -- enabling both raises `ValueError`. Disabling both trains in Float32 (2x memory, more stable).

- **What**: The numerical precision used for training computations
- **Why**: Half-precision halves memory usage and speeds up training on modern GPUs
- **How**: `bf16=True` is the default (best for A100/H100). Use `fp16=True` for older GPUs (V100, T4)
- **When**: bf16 for Ampere+ GPUs; fp16 for pre-Ampere; float32 only when debugging numerical issues

### Concept 3: Gradient Checkpointing

When `gradient_checkpointing=True`, the model discards intermediate activations during the forward pass and recomputes them during backpropagation. This trades compute time (~20% slower) for significant memory savings.

### Key API

| Field                         | Type    | Default  | Description                                       |
| ----------------------------- | ------- | -------- | ------------------------------------------------- |
| `num_train_epochs`            | `int`   | `3`      | Number of full passes through the dataset         |
| `per_device_train_batch_size` | `int`   | `4`      | Samples per GPU per forward pass                  |
| `gradient_accumulation_steps` | `int`   | `4`      | Steps before optimizer update                     |
| `learning_rate`               | `float` | `2e-4`   | Peak learning rate (must be positive, finite)     |
| `warmup_ratio`                | `float` | `0.03`   | Fraction of training for LR warmup [0, 1)         |
| `max_seq_length`              | `int`   | `2048`   | Maximum token sequence length                     |
| `gradient_checkpointing`      | `bool`  | `True`   | Trade compute for memory                          |
| `bf16`                        | `bool`  | `True`   | BFloat16 precision (mutually exclusive with fp16) |
| `fp16`                        | `bool`  | `False`  | Float16 precision (mutually exclusive with bf16)  |
| `dataset_text_field`          | `str`   | `"text"` | Column name containing training text              |

## Code Walkthrough

```python
from __future__ import annotations

from kailash_align import SFTConfig
```

### Default Configuration

```python
default = SFTConfig()

assert default.num_train_epochs == 3
assert default.per_device_train_batch_size == 4
assert default.gradient_accumulation_steps == 4
assert default.learning_rate == 2e-4
assert default.warmup_ratio == 0.03
assert default.max_seq_length == 2048
assert default.logging_steps == 10
assert default.save_steps == 100
assert default.gradient_checkpointing is True
assert default.bf16 is True
assert default.fp16 is False
assert default.dataset_text_field == "text"
```

The defaults target a single-GPU setup with moderate memory: bf16 precision, gradient checkpointing enabled, effective batch size of 16 (4 \* 4).

### Custom Configuration

```python
custom = SFTConfig(
    num_train_epochs=5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    warmup_ratio=0.1,
    max_seq_length=4096,
    logging_steps=5,
    save_steps=200,
    gradient_checkpointing=False,
    bf16=False,
    fp16=True,
    dataset_text_field="content",
)

assert custom.num_train_epochs == 5
assert custom.per_device_train_batch_size == 8
assert custom.learning_rate == 1e-4
assert custom.max_seq_length == 4096
assert custom.fp16 is True
assert custom.bf16 is False
assert custom.dataset_text_field == "content"
```

This config targets a V100 GPU (fp16 instead of bf16), with a larger batch size and longer sequences. The dataset uses a `"content"` column instead of the default `"text"`.

### Immutability

```python
try:
    default.learning_rate = 1e-3  # type: ignore[misc]
    assert False, "Should have raised FrozenInstanceError"
except AttributeError:
    pass  # Expected: frozen dataclass
```

Like all Align config classes, SFTConfig is frozen after creation.

### Learning Rate Validation

```python
try:
    SFTConfig(learning_rate=0.0)
    assert False, "Should have raised ValueError"
except ValueError:
    pass  # Expected: learning_rate must be positive

try:
    SFTConfig(learning_rate=-1e-4)
    assert False, "Should have raised ValueError"
except ValueError:
    pass  # Expected: learning_rate must be positive

try:
    SFTConfig(learning_rate=float("nan"))
    assert False, "Should have raised ValueError"
except ValueError:
    pass  # Expected: NaN rejected

try:
    SFTConfig(learning_rate=float("inf"))
    assert False, "Should have raised ValueError"
except ValueError:
    pass  # Expected: Inf rejected
```

Learning rate must be positive and finite. Zero, negative, NaN, and infinity are all rejected.

### bf16/fp16 Mutual Exclusion

```python
try:
    SFTConfig(bf16=True, fp16=True)
    assert False, "Should have raised ValueError"
except ValueError as e:
    assert "bf16" in str(e) and "fp16" in str(e)

# Both False is valid (uses float32)
both_off = SFTConfig(bf16=False, fp16=False)
assert both_off.bf16 is False
assert both_off.fp16 is False
```

You cannot enable both precision modes simultaneously. Disabling both is valid -- the model trains in float32.

### Effective Batch Size Calculation

```python
cfg = SFTConfig(per_device_train_batch_size=4, gradient_accumulation_steps=8)
effective_single_gpu = cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps
assert effective_single_gpu == 32, "4 * 8 = 32 effective batch size on one GPU"
```

On a single GPU, effective batch size = per_device \* gradient_accumulation. On multi-GPU setups, multiply by the number of GPUs.

### Common Configurations

```python
# Memory-constrained: small batch, high accumulation
constrained = SFTConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    max_seq_length=1024,
)
assert constrained.per_device_train_batch_size == 1
assert constrained.gradient_checkpointing is True

# Speed-focused: large batch, less accumulation
fast = SFTConfig(
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    gradient_checkpointing=False,
)
assert fast.per_device_train_batch_size == 16
```

Memory-constrained setups use batch size 1 with high accumulation and gradient checkpointing. Speed-focused setups on large GPUs maximize batch size and disable checkpointing.

## Exercises

1. Calculate the effective batch size for `SFTConfig(per_device_train_batch_size=2, gradient_accumulation_steps=16)` on a 4-GPU setup. What is the total effective batch size?
2. Create two SFTConfig instances: one for a T4 GPU (16GB) and one for an A100 (80GB). Which fields would you change and why?
3. What happens if you set `warmup_ratio=1.0`? Why is this value rejected? What would it mean mathematically if it were allowed?

## Key Takeaways

- SFTConfig controls all hyperparameters for supervised fine-tuning
- Effective batch size = per_device _ gradient_accumulation _ num_gpus
- bf16 and fp16 are mutually exclusive; both off means float32
- Gradient checkpointing trades ~20% speed for significant memory savings
- Learning rate must be positive and finite; warmup_ratio must be in [0, 1)
- All configs are frozen (immutable) for reproducibility

## Next Chapter

[Chapter 4: DPO Configuration](04_dpo_config.md) -- Configure Direct Preference Optimization with beta control, prompt/completion length budgets, and loss type variants.
