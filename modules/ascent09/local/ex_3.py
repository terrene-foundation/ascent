# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT09 — Exercise 3: GRPO and Group Relative Policy Optimization
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Understand GRPO (the algorithm behind DeepSeek-R1) —
#   implement group-relative advantage estimation, verifiable reward
#   functions, and the clipped GRPO update. Compare against PPO.
#
# TASKS:
#   1. Implement GRPO objective (group-relative advantages)
#   2. Implement verifiable reward function for SG regulatory Q&A
#   3. Implement clipped GRPO update
#   4. Compare GRPO vs PPO on regulatory Q&A
#   5. Use AlignmentPipeline(method="grpo") at scale
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os

import numpy as np
import polars as pl

from kaizen_agents import Delegate

from kailash_align import (
    AlignmentConfig,
    AlignmentPipeline,
    AdapterRegistry,
    GRPOConfig,
)
from kailash_align.config import LoRAConfig
from kailash_align.registry import AdapterSignature

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()

base_model = os.environ.get("SFT_BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
llm_model = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))

# ── Data Loading ─────────────────────────────────────────────────────

loader = ASCENTDataLoader()
regulations = loader.load("ascent09", "sg_regulations.parquet")

print(f"=== SG Regulations Dataset ===")
print(f"Shape: {regulations.shape}")
print(f"Columns: {regulations.columns}")
print(regulations.head(3))


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Implement GRPO objective (group-relative advantages)
# ══════════════════════════════════════════════════════════════════════
# GRPO generates G completions per prompt, uses group statistics as baseline.
# A_i = (r_i - mean(r_group)) / std(r_group) — no value network needed.

rng = np.random.default_rng(42)


def compute_grpo_advantages(
    rewards: np.ndarray,
    group_size: int,
) -> np.ndarray:
    """Compute group-relative advantages for GRPO.

    Args:
        rewards: (n_prompts * group_size,) flat array
        group_size: completions per prompt (G)

    Returns:
        advantages: (n_prompts * group_size,) normalized
    """
    n_total = len(rewards)
    n_prompts = n_total // group_size

    # TODO: Reshape to (n_prompts, group_size), compute per-group mean/std,
    #   normalize: (grouped - means) / max(stds, 1e-8), flatten
    ____
    ____
    ____
    ____
    return ____


# TODO: Simulate 20 prompts x 8 completions with varying difficulty
n_prompts = 20
group_size = 8
n_total = ____
prompt_difficulty = ____
raw_rewards = np.zeros(n_total)
for i in range(n_prompts):
    ____

advantages = ____

print(f"\n=== GRPO Advantages ===")
print(f"  Prompts: {n_prompts}, Group size: {group_size}")
print(f"  Raw rewards: mean={raw_rewards.mean():.3f}, std={raw_rewards.std():.3f}")
print(f"  Advantages: mean={advantages.mean():.6f}, std={advantages.std():.3f}")

# TODO: Verify zero-mean property per group
grouped_adv = ____
print(f"  Per-group max|mean|={np.max(np.abs(grouped_adv.mean(axis=1))):.2e}")

print(f"\n  Example group (prompt 0):")
print(f"    Rewards:    {raw_rewards[:group_size].round(3)}")
print(f"    Advantages: {advantages[:group_size].round(3)}")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Implement verifiable reward function for SG regulatory Q&A
# ══════════════════════════════════════════════════════════════════════
# Verifiable rewards: programmatically computable, no learned reward model.

reg_ids = regulations.select("regulation_id").to_series().to_list()

# TODO: Build keyword sets per regulatory category (general, financial, data_protection, employment)
# Hint: financial includes ["MAS", "capital", "liquidity", ...], data_protection includes ["PDPA", ...]
regulatory_keywords = {
    "general": ____,
    "financial": ____,
    "data_protection": ____,
    "employment": ____,
}


def regulatory_reward(
    completion: str,
    prompt: str,
    regulation_ids: list[str],
    keywords: dict[str, list[str]],
) -> dict:
    """Verifiable reward: citation + format + keywords + length (each 0-1)."""
    scores = {}

    # TODO: Citation — count reg IDs found in completion, min(hits/2, 1.0)
    ____
    scores["citation"] = ____

    # TODO: Format — count signals [":", "\n-", "\n1.", "\n2.", "section", "article"]
    ____
    scores["format"] = ____

    # TODO: Keywords — flatten all keyword lists, count matches, min(hits/5, 1.0)
    ____
    scores["keywords"] = ____

    # TODO: Length — 1.0 for 50-500 words, linear decay outside
    ____
    scores["length"] = ____

    scores["total"] = sum(scores.values())
    scores["max_possible"] = 4.0
    return scores


# TODO: Write a good completion (cites PDPA, structured, references MAS) and a poor one
good_completion = ____
poor_completion = ____
sample_prompt = "What are the key data protection requirements in Singapore?"

good_scores = ____
poor_scores = ____

print(f"\n=== Verifiable Reward Function ===")
print(f"  Good: {good_scores['total']:.2f}/{good_scores['max_possible']:.0f}")
print(f"  Poor: {poor_scores['total']:.2f}/{poor_scores['max_possible']:.0f}")
print(f"  Gap: {good_scores['total'] - poor_scores['total']:.2f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Implement clipped GRPO update
# ══════════════════════════════════════════════════════════════════════
# L^CLIP = min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)
# r_t = pi_theta(y|x) / pi_old(y|x) is the importance ratio.


def grpo_clipped_loss(
    log_probs_new: np.ndarray,
    log_probs_old: np.ndarray,
    advantages: np.ndarray,
    clip_epsilon: float = 0.2,
    kl_coef: float = 0.001,
    log_probs_ref: np.ndarray | None = None,
) -> dict:
    """Clipped GRPO objective with optional KL penalty."""
    # TODO: ratio = exp(log_new - log_old)
    ratio = ____
    # TODO: surr1 = ratio * advantages
    surr1 = ____
    # TODO: clipped_ratio = clip(ratio, 1-eps, 1+eps); surr2 = clipped * advantages
    clipped_ratio = ____
    surr2 = ____
    # TODO: policy_loss = -mean(min(surr1, surr2))
    policy_loss = ____

    # TODO: KL penalty if reference provided: mean(log_new - log_ref)
    kl_div = ____

    total_loss = ____
    clip_fraction = ____

    # TODO: Return dict with total_loss, policy_loss, kl_penalty, kl_divergence,
    #   clip_fraction, mean_ratio, mean_advantage
    return ____


# TODO: Simulate GRPO update — old log-probs, shift by advantages * 0.1
log_probs_old = ____
log_probs_ref = ____
log_probs_new = ____

grpo_result = ____

print(f"\n=== Clipped GRPO Update ===")
for key, val in grpo_result.items():
    print(f"  {key}: {val:.6f}")

# TODO: Show effect of different clip_epsilon values
print(f"\n  Effect of clip_epsilon:")
for eps in [0.05, 0.1, 0.2, 0.3, 0.5]:
    result = ____
    print(
        f"    eps={eps:.2f} -> loss={result['total_loss']:.6f}, clip={result['clip_fraction']:.4f}"
    )


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Compare GRPO vs PPO on regulatory Q&A
# ══════════════════════════════════════════════════════════════════════
# GRPO: no value network, group-relative advantages
# PPO: learned value baseline, value function loss


def simulate_training_comparison(
    n_steps: int = 50,
    n_prompts_per_step: int = 10,
    group_size: int = 8,
) -> dict:
    """Simulate GRPO vs PPO training dynamics."""
    grpo_losses, ppo_losses = [], []

    for step in range(n_steps):
        n = ____
        # TODO: Simulate improving rewards: improvement = step/n_steps * 0.5
        rewards = ____

        # TODO: GRPO — group advantages, policy update, loss
        grpo_adv = ____
        grpo_log_old = ____
        grpo_log_new = ____
        grpo_res = ____

        # TODO: PPO — noisy value baseline, normalize, policy update, loss
        value_baseline = ____
        ppo_adv = ____
        ppo_adv = ____
        ppo_log_old = ____
        ppo_log_new = ____
        ppo_res = ____

        grpo_losses.append(____)
        ppo_losses.append(____)

    return {"grpo_losses": grpo_losses, "ppo_losses": ppo_losses}


comparison = simulate_training_comparison()

# TODO: Compute final loss stats (average + variance of last 10 steps)
grpo_final = ____
ppo_final = ____
grpo_var = ____
ppo_var = ____

print(f"\n=== GRPO vs PPO Comparison ===")
print(f"  GRPO final loss: {grpo_final:.6f} (var={grpo_var:.6f})")
print(f"  PPO  final loss: {ppo_final:.6f} (var={ppo_var:.6f})")
print(f"  GRPO: no value network, group normalization, verifiable rewards")
print(f"  PPO:  learned V(s), works with any reward, more sample efficient")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Use AlignmentPipeline(method="grpo") at scale
# ══════════════════════════════════════════════════════════════════════

registry = AdapterRegistry()

# TODO: AlignmentConfig with method="grpo",
#   GRPOConfig(num_generations=4, temperature=0.7, kl_coef=0.001,
#     max_completion_length=512, batch=2, grad_accum=4, lr=1e-5),
#   LoRAConfig(rank=16, alpha=32, dropout=0.05, target_modules=("q_proj","v_proj")),
#   reward_funcs=["accuracy"]
grpo_config = AlignmentConfig(
    method=____,
    base_model_id=____,
    grpo=GRPOConfig(____),
    lora=LoRAConfig(____),
    reward_funcs=____,
    experiment_dir=____,
)

pipeline = ____

print(f"\n=== AlignmentPipeline (GRPO) ===")
print(f"  Method: {grpo_config.method}")
print(f"  Generations/prompt: {grpo_config.grpo.num_generations}")
print(f"  Temperature: {grpo_config.grpo.temperature}")
print(f"  Reward functions: {grpo_config.reward_funcs}")

print(f"\nProduction: register reward function, then pipeline.train()")
print(f"  GRPO incentivizes step-by-step reasoning (DeepSeek-R1 pattern)")

print("\n=== Exercise 3 complete ===")
