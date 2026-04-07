# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT10 — Exercise 1: LoRA Fine-Tuning with AlignmentPipeline
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Configure AlignmentConfig for LoRA-based SFT, train on
#   Singapore domain Q&A, register the adapter, and calculate parameter
#   savings vs full fine-tuning.
#
# TASKS:
#   1. Load SFT dataset (instruction-response pairs)
#   2. Configure AlignmentConfig with LoRA parameters
#   3. Run AlignmentPipeline training
#   4. Register adapter in AdapterRegistry
#   5. Calculate and verify LoRA parameter reduction vs full fine-tuning
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
sft_data = loader.load("ascent10", "sg_domain_qa.parquet")
base_model = os.environ.get("SFT_BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# ══════════════════════════════════════════════════════════════════════
# TASK 1: Split dataset 90/10 and print sizes.
# ══════════════════════════════════════════════════════════════════════
____
____
____

# ══════════════════════════════════════════════════════════════════════
# TASK 2: Build AlignmentConfig for SFT with LoRA (rank=16, alpha=32,
#   dropout=0.05, target q_proj + v_proj). Print key config fields.
# ══════════════════════════════════════════════════════════════════════
____
____
____


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Async training — instantiate AlignmentPipeline, call train()
#   with train_data and adapter name "sg_domain_sft_v1". Handle
#   GPU-unavailable with try/except. Return (pipeline, result_or_None).
# ══════════════════════════════════════════════════════════════════════
async def run_training():
    ____
    ____


pipeline, sft_result = asyncio.run(run_training())

# ══════════════════════════════════════════════════════════════════════
# TASK 4: If sft_result is not None, register adapter in AdapterRegistry
#   with metadata {base_model, method, final_loss}. Print outcome.
# ══════════════════════════════════════════════════════════════════════
____
____

# ══════════════════════════════════════════════════════════════════════
# TASK 5: Parameter reduction analysis — TinyLlama hidden_dim=2048,
#   num_layers=22. Compute full params vs LoRA params (B+A matrices),
#   reduction ratio, percentage. Print a rank comparison for r in
#   [4, 8, 16, 32, 64].
# ══════════════════════════════════════════════════════════════════════
____
____
____
____
____

print("\n✓ Exercise 1 complete — LoRA SFT with parameter reduction analysis")
