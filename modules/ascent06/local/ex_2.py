# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT06 — Exercise 2: DPO / QLoRA Alignment
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Compare DPO (preference optimization) vs QLoRA (quantized
#   fine-tuning). Evaluate with LLM-as-judge and human rubric.
#
# TASKS:
#   1. Load preference pairs dataset
#   2. Configure and run DPO alignment
#   3. Configure and run QLoRA fine-tuning
#   4. Evaluate both with LLM-as-judge
#   5. Compare methods: quality, cost, speed
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv

from kailash_align import AlignmentConfig, AlignmentPipeline, AdapterRegistry
from kailash_align.config import SFTConfig, DPOConfig, LoRAConfig
from kailash_align.registry import AdapterSignature

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()

base_model = os.environ.get("SFT_BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Load preference pairs
# ══════════════════════════════════════════════════════════════════════

loader = ASCENTDataLoader()
preferences = loader.load("ascent06", "preference_pairs.parquet")

print(f"=== Preference Pairs ===")
print(f"Shape: {preferences.shape}")
print(f"Columns: {preferences.columns}")
# Expected: prompt, chosen, rejected
print(preferences.head(2))


# ══════════════════════════════════════════════════════════════════════
# TASK 2: DPO alignment
# ══════════════════════════════════════════════════════════════════════
# DPO: Direct Preference Optimization
# Derives from Bradley-Terry preference model:
#   P(y_w > y_l | x) = σ(β⁻¹(r(x,y_w) - r(x,y_l)))
# Key insight: eliminates the reward model entirely
#   π*(y|x) ∝ π_ref(y|x) · exp(β⁻¹ · r*(x,y))
# Loss: L_DPO = -E[log σ(β(log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))]

# TODO: Create an AlignmentConfig for DPO.
# Use DPOConfig(beta=0.1, num_train_epochs=2, per_device_train_batch_size=2,
# learning_rate=5e-5, max_length=512) and LoRAConfig(rank=16, alpha=32,
# dropout=0.05, target_modules=("q_proj", "v_proj")).
# Set experiment_dir="./dpo_output".
____


def demonstrate_dpo():
    """Demonstrate DPO alignment pipeline setup."""
    # TODO: Create AdapterRegistry and AlignmentPipeline for DPO.
    # Print the method, beta value, and production usage pattern.
    ____
    ____
    ____
    ____

    return pipeline, registry


dpo_pipeline, dpo_registry = demonstrate_dpo()


# ══════════════════════════════════════════════════════════════════════
# TASK 3: QLoRA fine-tuning
# ══════════════════════════════════════════════════════════════════════
# QLoRA: Quantized LoRA
# 1. Quantize base model to NF4 (4-bit NormalFloat)
# 2. Apply LoRA adapters on top of quantized model
# 3. Train adapters in full precision while base stays 4-bit
# 4. Double quantization: quantize the quantization constants too

# QLoRA config: same as SFT but with use_qlora=True
# Requires bitsandbytes (GPU-only library), so we show the config pattern
# without instantiating it. On a GPU machine:
#   qlora_config = AlignmentConfig(
#       method="sft",
#       base_model_id=base_model,
#       use_qlora=True,  # NF4 quantization via bitsandbytes
#       lora=LoRAConfig(rank=16, alpha=32, dropout=0.05,
#                        target_modules=("q_proj", "v_proj", "k_proj", "o_proj")),
#       sft=SFTConfig(num_train_epochs=3, per_device_train_batch_size=4,
#                      learning_rate=2e-4, max_seq_length=512),
#       experiment_dir="./qlora_output",
#   )

# TODO: Create an AlignmentConfig for QLoRA (without use_qlora=True since we
# have no GPU here). Use method="sft", LoRAConfig targeting all 4 attention
# projections (q_proj, v_proj, k_proj, o_proj), and standard SFTConfig.
# Set experiment_dir="./qlora_output".
____


def demonstrate_qlora():
    """Demonstrate QLoRA fine-tuning setup."""
    import polars as pl

    # TODO: Convert preference pairs to instruction format (use chosen responses).
    # Select prompt → instruction, chosen → response.
    # Create AdapterRegistry. Print: method, use_qlora flag, memory savings,
    # converted row count, and production usage pattern.
    ____
    ____
    ____
    ____

    return qlora_registry


qlora_registry = demonstrate_qlora()


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Register adapters and demonstrate evaluation pattern
# ══════════════════════════════════════════════════════════════════════


async def register_both_adapters():
    """Register DPO and QLoRA adapters for comparison."""
    # TODO: Register both adapters in their respective registries.
    # For each: build AdapterSignature from the config, call register_adapter()
    # with name, adapter_path, signature, training_metrics, and tags.
    # Print registered adapters with method and stage.
    # Then print the LLM-as-judge evaluation pattern and its three known biases.
    ____
    ____
    ____
    ____
    ____
    ____

    return dpo_adapter, qlora_adapter


dpo_adapter, qlora_adapter = asyncio.run(register_both_adapters())


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Method comparison
# ══════════════════════════════════════════════════════════════════════

# TODO: Print a side-by-side comparison table of DPO vs QLoRA.
# Rows: data requirement, reward model needed, memory efficiency, alignment quality,
# training loss (from registered adapter metrics). Then print when to choose each.
____
____
____
____
____

print("\n✓ Exercise 2 complete — DPO vs QLoRA with LLM-as-judge evaluation")
