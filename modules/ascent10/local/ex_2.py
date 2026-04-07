# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT10 — Exercise 2: DPO Preference Alignment
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Align a model using DPO — construct preference pairs,
#   configure DPO training, compare aligned vs base model on safety
#   and helpfulness metrics.
#
# TASKS:
#   1. Load preference dataset (chosen/rejected pairs)
#   2. Configure AlignmentConfig for DPO with beta parameter
#   3. Train DPO pipeline
#   4. Evaluate aligned model on safety prompts
#   5. Compare DPO vs SFT-only model outputs
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os

import polars as pl

from kailash_align import AdapterRegistry, AlignmentConfig, AlignmentPipeline
from kailash_align.config import LoRAConfig

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()

loader = ASCENTDataLoader()
pref_data = loader.load("ascent10", "preference_pairs.parquet")
base_model = os.environ.get("SFT_BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

assert (
    "prompt" in pref_data.columns
    and "chosen" in pref_data.columns
    and "rejected" in pref_data.columns
)

# ══════════════════════════════════════════════════════════════════════
# TASK 1: Split preference dataset 90/10. Print sizes and sample prompt.
# ══════════════════════════════════════════════════════════════════════
____
____
____

# ══════════════════════════════════════════════════════════════════════
# TASK 2: Build AlignmentConfig for DPO. Key hyperparameter is beta
#   (~0.1 for moderate preference strength). Use same LoRA settings as
#   Exercise 1. Print config and the DPO loss formula.
# ══════════════════════════════════════════════════════════════════════
____
____
____


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Async DPO training — AlignmentPipeline(dpo_config), train on
#   train_pref with adapter name "sg_domain_dpo_v1". Handle GPU errors.
#   Return (pipeline, result_or_None).
# ══════════════════════════════════════════════════════════════════════
async def run_dpo():
    ____
    ____


dpo_pipeline, dpo_result = asyncio.run(run_dpo())

# ══════════════════════════════════════════════════════════════════════
# TASK 4: Safety evaluation — if dpo_result is not None, test harmful
#   prompts against base and aligned model, compare refusal rates.
#   If no result, explain what the evaluation would measure.
# ══════════════════════════════════════════════════════════════════════
____
____

# ══════════════════════════════════════════════════════════════════════
# TASK 5: Print SFT vs DPO comparison summary. Print beta sensitivity
#   table for betas [0.01, 0.1, 0.5, 1.0] with descriptive labels.
# ══════════════════════════════════════════════════════════════════════
____
____

print("\n✓ Exercise 2 complete — DPO preference alignment with safety evaluation")
