# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT06 — Exercise 4: LoRA Adapter Merging and Evaluation
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Register SFT and DPO adapters, merge a LoRA adapter into its
#   base model using AdapterMerger, and evaluate merged vs adapter-loaded
#   inference. Understand the adapter lifecycle: train → register → merge.
#
# TASKS:
#   1. Register SFT and DPO adapters in AdapterRegistry
#   2. Demonstrate AdapterMerger (LoRA merge into base model)
#   3. Understand merging strategies (TIES, DARE) conceptually
#   4. Compare adapter-loaded vs merged inference trade-offs
#   5. Manage adapter lifecycle: staging → production → merged
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os

import polars as pl

from kailash_align import (
    AlignmentConfig,
    AlignmentPipeline,
    AdapterRegistry,
)
from kailash_align.config import SFTConfig, DPOConfig, LoRAConfig
from kailash_align.registry import AdapterSignature
from kailash_align.merge import AdapterMerger

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()

base_model = os.environ.get("SFT_BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")


# ── Evaluation Dataset ────────────────────────────────────────────────

eval_prompts = [
    {
        "prompt": "What is the HDB resale flat application procedure in Singapore?",
        "category": "housing",
        "reference": (
            "The HDB resale process involves: (1) register intent to buy/sell on HDB portal, "
            "(2) grant of option (14 days), (3) exercise option (21 days), (4) submit resale "
            "application (both parties), (5) HDB approval (8 weeks), (6) completion appointment. "
            "Buyers must secure an HLE or bank loan before exercising the option."
        ),
    },
    {
        "prompt": "Explain CPF contribution rates for Singapore employees aged 35-45.",
        "category": "cpf",
        "reference": (
            "For employees aged 35-45, CPF contributions are: employer 16%, employee 20%, "
            "total 36%. Allocation: OA 23%, SA 6%, MA 8% (approximate). Contributions apply "
            "to ordinary wages up to $6,000/month and additional wages up to the annual limit."
        ),
    },
    {
        "prompt": "How does MAS regulate AI systems used in financial services?",
        "category": "regulation",
        "reference": (
            "MAS regulates AI in financial services through FEAT principles (Fairness, Ethics, "
            "Accountability, Transparency). Key requirements: model risk management framework, "
            "ongoing monitoring, explainability for customer-facing decisions (especially credit), "
            "board oversight of AI governance, and regular model validation by independent parties."
        ),
    },
    {
        "prompt": "What is Singapore's SingPass MyInfo system and how does it work?",
        "category": "digital_gov",
        "reference": (
            "MyInfo is a government-managed personal data platform. Citizens store verified "
            "personal data once; services retrieve it with consent. Data includes NRIC details, "
            "income records, CPF balances, and property ownership. Authentication via SingPass "
            "with 2FA. Pre-fills forms for banking, insurance, and government applications."
        ),
    },
    {
        "prompt": "Explain the key differences between HDB BTO, SBF, and resale flats.",
        "category": "housing",
        "reference": (
            "BTO (Build-To-Order): new flats built to demand, 3-5 year wait, subsidised price. "
            "SBF (Sale of Balance Flats): unsold BTO flats, shorter wait, similar subsidy. "
            "Resale: existing flats on open market, immediate occupancy, market price + CPF grant "
            "if eligible. BTO/SBF require citizenship and income ceiling; resale has fewer restrictions."
        ),
    },
]

print(f"=== LoRA Adapter Merging Exercise ===")
print(f"Base model: {base_model}")
print(f"Evaluation prompts: {len(eval_prompts)}")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Register SFT and DPO adapters in AdapterRegistry
# ══════════════════════════════════════════════════════════════════════


async def register_adapters():
    """Register SFT and DPO adapters (simulating output from Ex1 and Ex2)."""
    # TODO: Implement register_adapters.
    # 1. Create AdapterRegistry().
    # 2. Register an SFT adapter: AdapterSignature(base_model_id, adapter_type="lora",
    #    rank=16, alpha=32, target_modules=("q_proj","v_proj"), training_method="sft").
    #    Call registry.register_adapter(name="sg_domain_sft_v1",
    #    adapter_path="./sft_output/sg_domain_sft_v1", signature=sft_sig,
    #    training_metrics={"train_loss":0.42,"eval_loss":0.51}, tags=[...]).
    # 3. Register a DPO adapter with training_method="dpo",
    #    metrics {"train_loss":0.35,"eval_loss":0.43}.
    # 4. List all adapters and print name, version, method, stage, merge_status.
    # 5. Return (registry, sft_adapter, dpo_adapter).
    ____
    ____
    ____
    ____
    ____
    ____

    return registry, sft_adapter, dpo_adapter


registry, sft_adapter, dpo_adapter = asyncio.run(register_adapters())


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Demonstrate AdapterMerger (LoRA merge into base model)
# ══════════════════════════════════════════════════════════════════════
#
# AdapterMerger merges LoRA adapter weights INTO the base model:
#   W_merged = W_base + B × A   (where B, A are LoRA matrices)
#
# After merging:
#   - No adapter loading overhead at inference time
#   - Model is a standard HuggingFace model (no PEFT dependency)
#   - Cannot un-merge (the adapter is baked in)
#   - Suitable for GGUF export and Ollama deployment
#
# Pipeline: train → register → [optional: evaluate] → merge → export → deploy


def demonstrate_merger():
    """Show the AdapterMerger API without requiring GPU."""
    # TODO: Implement demonstrate_merger.
    # Create AdapterMerger(adapter_registry=registry).
    # Print the merge equation (W_merged = W_base + B x A), the SDK API
    # (merger.merge("sg_domain_sft_v1")), the convenience function pattern,
    # and the merge lifecycle states: separate → merged → exported.
    ____
    ____
    ____
    ____

    return merger


merger = demonstrate_merger()


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Merging strategies — conceptual overview
# ══════════════════════════════════════════════════════════════════════
#
# Multi-adapter merging strategies (not in SDK — research frontier):
#
# TIES (Task-Interference Elimination Strategy):
#   1. Trim: zero out low-magnitude delta weights per adapter
#   2. Elect sign: resolve sign conflicts via majority vote
#   3. Disjoint merge: only merge where adapters agree on sign
#   Key parameter: density (fraction kept after trimming)
#
# DARE (Drop And REscale):
#   1. Drop: randomly zero out delta weights with probability p
#   2. Rescale: multiply remaining by 1/(1-p) to preserve expectation
#   Simpler than TIES but often competitive
#
# Linear merge: W_merged = α·W_adapter1 + (1-α)·W_adapter2
#   Simplest approach; works when adapters are similar

# TODO: Print the conceptual overview of multi-adapter merging strategies.
# Explain: single-adapter merge (SDK), TIES, DARE, and linear merge.
# Describe when multi-adapter merge matters and the current production pattern.
____


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Compare adapter-loaded vs merged inference
# ══════════════════════════════════════════════════════════════════════

# TODO: Print a comparison table of adapter-loaded vs merged inference.
# Rows: inference latency, memory overhead, flexibility, PEFT dependency,
# GGUF export, Ollama deployment, un-merge possible, multi-adapter support.
____
____
____


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Adapter lifecycle management
# ══════════════════════════════════════════════════════════════════════


async def demonstrate_lifecycle():
    """Show the full adapter lifecycle: staging → production → merged."""
    # TODO: Implement demonstrate_lifecycle.
    # 1. Promote the SFT adapter to "production" via registry.promote().
    # 2. Retrieve the updated adapter and print its stage and merge_status.
    # 3. Print the 6-step full lifecycle: Train → Register → Stage → Merge → Export → Deploy.
    # 4. List all adapters and print their stage and merge_status.
    ____
    ____
    ____
    ____
    ____
    ____


asyncio.run(demonstrate_lifecycle())

# TODO: Print the 4 key takeaways:
# 1. AdapterMerger merge equation and benefits.
# 2. Adapter lifecycle states in AdapterRegistry.
# 3. Multi-adapter merging (TIES/DARE) is research, not yet in SDK.
# 4. AdapterRegistry as auditable source of truth.
____

print("\n✓ Exercise 4 complete — LoRA adapter merging with AdapterMerger")
