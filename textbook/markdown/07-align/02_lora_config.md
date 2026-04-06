# Chapter 2: LoRA Configuration

## Overview

LoRA (Low-Rank Adaptation) is the parameter-efficient fine-tuning strategy used by Kailash Align. Instead of updating all model weights, LoRA injects small trainable matrices into specific layers, dramatically reducing memory and compute requirements. This chapter teaches you how to configure LoRA parameters -- rank, alpha, target modules, dropout, and bias -- and understand their impact on training.

## Prerequisites

- [Chapter 1: AlignmentConfig Basics](01_alignment_config.md)
- Understanding of matrix decomposition concepts (helpful but not required)

## Concepts

### Concept 1: Rank and Alpha

**Rank** (`r`) controls the dimensionality of the low-rank matrices. Lower rank means fewer trainable parameters (faster, less memory) but less capacity to learn new behavior. **Alpha** (`alpha`) is a scaling factor applied to the LoRA update. The effective learning rate scales as `alpha / rank`, so the convention `alpha = 2 * rank` keeps the scaling constant as you change rank.

- **What**: Two integers that control the size and scaling of LoRA adapter matrices
- **Why**: They let you trade off between training speed/memory and model capacity
- **How**: `LoRAConfig(rank=16, alpha=32)` -- the default. Higher rank for more capacity, proportionally higher alpha to maintain scaling
- **When**: Start with defaults; increase rank if the model underfits, decrease if you are memory-constrained

### Concept 2: Target Modules

Target modules specify which layers in the base model receive LoRA adapters. The default targets attention projection layers (`q_proj`, `v_proj`, `k_proj`, `o_proj`). Adding MLP layers (`gate_proj`, `up_proj`) increases capacity at the cost of more parameters.

- **What**: A tuple of layer name strings identifying which model components get LoRA matrices
- **Why**: Not all layers benefit equally from fine-tuning; targeting attention projections is the most common and well-validated approach
- **How**: `LoRAConfig(target_modules=("q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj"))`
- **When**: Extend beyond attention layers only when default targets underfit

### Concept 3: Frozen Dataclass

`LoRAConfig` is a **frozen** dataclass -- once created, it cannot be modified. This prevents accidental mid-training config mutation and ensures reproducibility. To change a value, create a new `LoRAConfig`.

### Key API

| Field            | Type             | Default                                    | Description                         |
| ---------------- | ---------------- | ------------------------------------------ | ----------------------------------- |
| `rank`           | `int`            | `16`                                       | Low-rank matrix dimension (>= 1)    |
| `alpha`          | `int`            | `32`                                       | Scaling factor (>= 1)               |
| `target_modules` | `tuple[str,...]` | `("q_proj", "v_proj", "k_proj", "o_proj")` | Which layers get LoRA adapters      |
| `dropout`        | `float`          | `0.05`                                     | Dropout rate on LoRA layers [0, 1)  |
| `bias`           | `str`            | `"none"`                                   | Bias training: none, all, lora_only |
| `task_type`      | `str`            | `"CAUSAL_LM"`                              | PEFT task type                      |

## Code Walkthrough

```python
from __future__ import annotations

import math

from kailash_align import LoRAConfig
```

### Default Configuration

```python
default = LoRAConfig()

assert default.rank == 16, f"Default rank should be 16, got {default.rank}"
assert default.alpha == 32, f"Default alpha should be 32, got {default.alpha}"
assert default.target_modules == ("q_proj", "v_proj", "k_proj", "o_proj")
assert default.dropout == 0.05
assert default.bias == "none"
assert default.task_type == "CAUSAL_LM"
```

The defaults are well-tested starting points: rank 16 with alpha 32 (2x scaling), attention projections only, 5% dropout, no bias training.

### Custom Configuration

```python
custom = LoRAConfig(
    rank=64,
    alpha=128,
    target_modules=("q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj"),
    dropout=0.1,
    bias="lora_only",
)

assert custom.rank == 64
assert custom.alpha == 128
assert len(custom.target_modules) == 6
assert "gate_proj" in custom.target_modules
assert custom.dropout == 0.1
assert custom.bias == "lora_only"
```

This high-capacity config targets 6 layers with rank 64. The `"lora_only"` bias setting trains bias terms only on LoRA-adapted layers.

### Immutability

```python
try:
    default.rank = 32  # type: ignore[misc]
    assert False, "Should have raised FrozenInstanceError"
except AttributeError:
    pass  # Expected: frozen dataclass cannot be modified
```

Attempting to modify any field after creation raises `AttributeError`. Create a new config instead.

### Rank Validation

```python
try:
    LoRAConfig(rank=0)
    assert False, "Should have raised ValueError"
except ValueError as e:
    assert "rank" in str(e)

try:
    LoRAConfig(rank=-1)
    assert False, "Should have raised ValueError"
except ValueError as e:
    assert "rank" in str(e)
```

Rank must be at least 1. Zero or negative values are rejected at creation time.

### Dropout Validation

```python
zero_dropout = LoRAConfig(dropout=0.0)
assert zero_dropout.dropout == 0.0, "Zero dropout is valid"

try:
    LoRAConfig(dropout=1.0)
    assert False, "Should have raised ValueError"
except ValueError as e:
    assert "dropout" in str(e)

try:
    LoRAConfig(dropout=-0.1)
    assert False, "Should have raised ValueError"
except ValueError as e:
    assert "dropout" in str(e)
```

Dropout must be in the half-open interval `[0, 1)`. Zero is valid (no dropout). Values of 1.0 or above, and negative values, are rejected. `NaN` and `Inf` are also rejected.

### Target Modules Must Not Be Empty

```python
try:
    LoRAConfig(target_modules=())
    assert False, "Should have raised ValueError"
except ValueError as e:
    assert "target_modules" in str(e)
```

At least one target module is required -- LoRA with no target layers would be a no-op.

### Bias Validation

```python
for valid_bias in ("none", "all", "lora_only"):
    cfg = LoRAConfig(bias=valid_bias)
    assert cfg.bias == valid_bias

try:
    LoRAConfig(bias="invalid")
    assert False, "Should have raised ValueError"
except ValueError as e:
    assert "bias" in str(e)
```

Only three bias strategies are supported: `"none"` (freeze all biases), `"all"` (train all biases), and `"lora_only"` (train biases only on LoRA-adapted layers).

### Common Patterns

```python
# Low-rank: fewer parameters, faster training, less capacity
low_rank = LoRAConfig(rank=4, alpha=8)
assert low_rank.rank == 4
assert low_rank.alpha == 8

# High-rank: more parameters, slower training, more capacity
high_rank = LoRAConfig(rank=128, alpha=256)
assert high_rank.rank == 128

# Typical convention: alpha = 2 * rank
assert default.alpha == default.rank * 2, "Convention: alpha = 2 * rank"
```

The `alpha = 2 * rank` convention keeps the effective LoRA scaling constant regardless of rank. This means changing rank only affects capacity, not the magnitude of updates.

## Exercises

1. Create three LoRAConfig instances: low-capacity (rank=4), medium (rank=16, default), and high-capacity (rank=128). For each, calculate the ratio `alpha / rank` and verify it equals 2.
2. What happens if you try to create a LoRAConfig with `dropout=float("nan")`? Why does the validator reject it?
3. Create a LoRAConfig that targets all six common layers (`q_proj`, `v_proj`, `k_proj`, `o_proj`, `gate_proj`, `up_proj`) with `bias="lora_only"`. In what scenario would you choose this over the default four-layer config?

## Key Takeaways

- `LoRAConfig` controls parameter-efficient fine-tuning through rank, alpha, target modules, dropout, and bias
- The `alpha = 2 * rank` convention maintains consistent update scaling across different rank values
- The config is frozen (immutable) after creation for reproducibility
- All fields are validated at creation time: rank >= 1, alpha >= 1, dropout in [0, 1), bias in {none, all, lora_only}
- Target modules default to attention projections; add MLP layers for more capacity

## Next Chapter

[Chapter 3: SFT Configuration](03_sft_config.md) -- Configure Supervised Fine-Tuning parameters including learning rate, batch size, sequence length, and precision settings.
