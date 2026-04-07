# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT06 — Exercise 1: SFT Fine-Tuning with Kailash Align
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Configure AlignmentConfig and run AlignmentPipeline for
#   supervised fine-tuning on a small model. Track adapter in
#   AdapterRegistry. Dataset: Singapore domain Q&A pairs.
#
# TASKS:
#   1. Load SFT dataset (instruction-response pairs)
#   2. Configure AlignmentConfig for SFT
#   3. Run AlignmentPipeline
#   4. Register adapter in AdapterRegistry
#   5. Evaluate fine-tuned vs base model
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv

from kailash_align import AlignmentConfig, AlignmentPipeline, AdapterRegistry
from kailash_align.config import SFTConfig, LoRAConfig
from kailash_align.registry import AdapterSignature

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Load SFT dataset
# ══════════════════════════════════════════════════════════════════════

loader = ASCENTDataLoader()
sft_data = loader.load("ascent06", "sg_domain_qa.parquet")

print(f"=== SFT Dataset ===")
print(f"Shape: {sft_data.shape}")
print(f"Columns: {sft_data.columns}")
print(f"\nSample:")
print(sft_data.head(3))

# SFT expects instruction-response format
# Columns: instruction, response (and optionally: context, category)
n_train = int(sft_data.height * 0.9)
train_data = sft_data[:n_train]
eval_data = sft_data[n_train:]
print(f"\nTrain: {train_data.height}, Eval: {eval_data.height}")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Configure AlignmentConfig
# ══════════════════════════════════════════════════════════════════════

base_model = os.environ.get("SFT_BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# TODO: Create an AlignmentConfig for supervised fine-tuning.
# Configure LoRAConfig (rank=16, alpha=32, dropout=0.05, target_modules q_proj+v_proj)
# and SFTConfig (3 epochs, batch_size=4, lr=2e-4, warmup_ratio=0.1,
# max_seq_length=512, gradient_accumulation_steps=4).
# Set experiment_dir="./sft_output".
____

print(f"\n=== AlignmentConfig ===")
print(f"Method: {config.method}")
print(f"Base model: {config.base_model_id}")
print(f"LoRA rank: {config.lora.rank} (W = W₀ + BA where B∈ℝ^{{d×r}}, A∈ℝ^{{r×k}})")
print(f"Target modules: {config.lora.target_modules}")
print(f"Training: {config.sft.num_train_epochs} epochs, lr={config.sft.learning_rate}")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Run AlignmentPipeline
# ══════════════════════════════════════════════════════════════════════


def demonstrate_pipeline():
    """Demonstrate AlignmentPipeline API.

    In production (with GPU), training runs as:
        pipeline = AlignmentPipeline(config, adapter_registry=registry)
        result = await pipeline.train(dataset=hf_dataset, adapter_name="sg_domain_sft_v1")

    The pipeline:
    1. Loads the base model and applies LoRA adapters
    2. Trains on the dataset for the configured epochs
    3. Saves adapter weights to experiment_dir
    4. Registers the adapter in AdapterRegistry (if provided)
    5. Returns AlignmentResult with adapter_name, adapter_path, metrics
    """
    # TODO: Create AdapterRegistry and AlignmentPipeline.
    # Attach the registry to the pipeline so training auto-registers the adapter.
    # Print the pipeline's method and confirm the registry is attached.
    ____
    ____
    ____
    ____
    ____

    return pipeline, registry


pipeline, adapter_registry = demonstrate_pipeline()


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Register adapter in AdapterRegistry
# ══════════════════════════════════════════════════════════════════════


async def register_adapter():
    """Register an adapter in AdapterRegistry (simulating post-training registration)."""
    # TODO: Build an AdapterSignature from the config fields, then call
    # adapter_registry.register_adapter() with name, path, signature,
    # training_metrics, and tags. Then list all adapters and print their details.
    ____
    ____
    ____

    print(f"\n=== Adapter Registered ===")
    print(f"  Name: {adapter.adapter_name}")
    print(f"  Version: {adapter.version}")
    print(f"  Stage: {adapter.stage}")
    print(f"  Base model: {adapter.base_model_id}")
    print(f"  Path: {adapter.adapter_path}")
    print(f"  LoRA config: {adapter.lora_config}")

    # TODO: List all adapters and print each one.
    ____
    print(f"\nAll registered adapters: {len(adapters)}")
    for a in adapters:
        print(
            f"  {a.adapter_name} v{a.version}: base={a.base_model_id} stage={a.stage}"
        )

    return adapter


sft_adapter = asyncio.run(register_adapter())


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Evaluate fine-tuned vs base
# ══════════════════════════════════════════════════════════════════════


def evaluate():
    # TODO: Implement evaluate().
    # Print 3 Singapore-domain evaluation questions.
    # Explain why LoRA parameter count is small vs full fine-tuning:
    # compute lora_params_per_layer = 2 * d_model * rank * len(target_modules)
    # where d_model=2048 for TinyLlama. Print the result.
    ____
    ____
    ____
    ____
    ____
    ____


evaluate()

print("\n✓ Exercise 1 complete — SFT fine-tuning with LoRA + AdapterRegistry")
