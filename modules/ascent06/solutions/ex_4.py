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
    registry = AdapterRegistry()

    # SFT adapter from Exercise 1
    sft_sig = AdapterSignature(
        base_model_id=base_model,
        adapter_type="lora",
        rank=16,
        alpha=32,
        target_modules=("q_proj", "v_proj"),
        training_method="sft",
    )
    sft_adapter = await registry.register_adapter(
        name="sg_domain_sft_v1",
        adapter_path="./sft_output/sg_domain_sft_v1",
        signature=sft_sig,
        training_metrics={"train_loss": 0.42, "eval_loss": 0.51},
        tags=["singapore", "domain-qa", "lora-r16"],
    )

    # DPO adapter from Exercise 2
    dpo_sig = AdapterSignature(
        base_model_id=base_model,
        adapter_type="lora",
        rank=16,
        alpha=32,
        target_modules=("q_proj", "v_proj"),
        training_method="dpo",
    )
    dpo_adapter = await registry.register_adapter(
        name="sg_domain_dpo_v1",
        adapter_path="./dpo_output/sg_domain_dpo_v1",
        signature=dpo_sig,
        training_metrics={"train_loss": 0.35, "eval_loss": 0.43},
        tags=["singapore", "domain-qa", "dpo"],
    )

    print(f"\n=== AdapterRegistry ===")
    adapters = await registry.list_adapters()
    print(f"Registered adapters: {len(adapters)}")
    for a in adapters:
        method = a.lora_config.get("training_method", "?")
        print(
            f"  {a.adapter_name} v{a.version}: method={method}, "
            f"stage={a.stage}, merge_status={a.merge_status}"
        )

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
    merger = AdapterMerger(adapter_registry=registry)

    print(f"\n=== AdapterMerger ===")
    print(f"AdapterMerger merges LoRA adapter weights into the base model.")
    print(f"After merge: W_merged = W_base + B x A (standard HF model)")
    print()
    print(f"API:")
    print(f"  merger = AdapterMerger(adapter_registry=registry)")
    print(f'  merged_path = await merger.merge("sg_domain_sft_v1")')
    print(f"  # Saves merged model to adapter_path/../merged/")
    print(f'  # Updates registry: merge_status="merged", merged_model_path=...')
    print()
    print(f"Or use the convenience function:")
    print(f"  from kailash_align.merge import merge_adapter")
    print(
        f'  merged_path = await merge_adapter("sg_domain_sft_v1", adapter_registry=registry)'
    )
    print()
    print(f"Merge lifecycle in registry:")
    print(f"  separate → merged → exported (GGUF)")
    print(
        f"  Each stage is tracked; merge is idempotent (re-merge returns existing path)"
    )

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

print(f"\n=== Multi-Adapter Merging Strategies (Conceptual) ===")
print(
    """
Kailash Align provides AdapterMerger for single-adapter merge (LoRA → base).
Multi-adapter merging (combining SFT + DPO adapters) is a research technique:

Single-adapter merge (SDK: AdapterMerger):
  W_merged = W_base + B × A
  Use case: deploy fine-tuned model as standard HF model

Multi-adapter merge (research — not yet in SDK):
  TIES:  Trim → Elect sign → Disjoint merge
  DARE:  Drop random weights → Rescale
  Linear: α · adapter_1 + (1-α) · adapter_2

When multi-adapter merge matters:
  - Combining SFT (task format) + DPO (alignment) into one model
  - Merging domain adapters (finance + legal) for multi-domain models
  - Ensemble effect without running multiple models at inference time

For now, the production pattern is:
  1. SFT adapter for instruction following
  2. DPO adapter on top of SFT for alignment
  3. Merge final adapter into base model with AdapterMerger
  4. Export to GGUF for local deployment via Ollama
"""
)


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Compare adapter-loaded vs merged inference
# ══════════════════════════════════════════════════════════════════════

print(f"=== Adapter-Loaded vs Merged Inference ===")
print(f"{'Aspect':<30} {'Adapter-Loaded':>18} {'Merged':>18}")
print("─" * 70)
print(f"{'Inference latency':<30} {'Higher (PEFT)':>18} {'Lower (native)':>18}")
print(f"{'Memory overhead':<30} {'Adapter in memory':>18} {'None':>18}")
print(f"{'Flexibility':<30} {'Swap adapters':>18} {'Fixed':>18}")
print(f"{'PEFT dependency':<30} {'Required':>18} {'Not required':>18}")
print(f"{'GGUF export':<30} {'Not possible':>18} {'Possible':>18}")
print(f"{'Ollama deployment':<30} {'Not possible':>18} {'Possible':>18}")
print(f"{'Un-merge possible':<30} {'N/A (separate)':>18} {'No':>18}")
print(f"{'Multi-adapter':<30} {'Load multiple':>18} {'One only':>18}")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Adapter lifecycle management
# ══════════════════════════════════════════════════════════════════════


async def demonstrate_lifecycle():
    """Show the full adapter lifecycle: staging → production → merged."""

    # Promote SFT adapter to production
    await registry.promote(
        "sg_domain_sft_v1",
        version=str(sft_adapter.version),
        stage="production",
    )

    # Check updated status
    promoted = await registry.get_adapter("sg_domain_sft_v1")
    print(f"\n=== Adapter Lifecycle ===")
    print(f"SFT adapter promoted to: {promoted.stage}")
    print(f"Merge status: {promoted.merge_status}")

    print(f"\nFull lifecycle:")
    print(f"  1. Train  → AlignmentPipeline.train() produces adapter weights")
    print(f"  2. Register → AdapterRegistry.register_adapter() tracks metadata")
    print(f"  3. Stage  → staging (default) → promote('production')")
    print(f"  4. Merge  → AdapterMerger.merge() bakes LoRA into base model")
    print(f"  5. Export → GGUF conversion for local deployment")
    print(f"  6. Deploy → Ollama, vLLM, or InferenceServer")

    # Show all adapters with their status
    adapters = await registry.list_adapters()
    print(f"\nAdapter registry state:")
    for a in adapters:
        print(
            f"  {a.adapter_name} v{a.version}: stage={a.stage}, "
            f"merge_status={a.merge_status}"
        )


asyncio.run(demonstrate_lifecycle())

print(f"\n=== Key Takeaways ===")
print(
    """
1. AdapterMerger merges LoRA weights INTO the base model:
   - W_merged = W_base + B x A
   - No adapter loading overhead at inference
   - Enables GGUF export for local deployment

2. Adapter lifecycle is tracked in AdapterRegistry:
   - staging → production → merged → exported
   - Every state change is auditable

3. Multi-adapter merging (TIES/DARE) is a research technique:
   - Not yet in SDK; would combine SFT + DPO adapters
   - For now: use sequential training (SFT → DPO on top)

4. AdapterRegistry as source of truth:
   - Every adapter tagged with provenance (method, base model, metrics)
   - Production adapter promoted explicitly — no silent overwrites
   - Merge status tracked: separate → merged → exported
"""
)

print("\n✓ Exercise 4 complete — LoRA adapter merging with AdapterMerger")
