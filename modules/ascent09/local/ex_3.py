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
# GRPO generates G completions per prompt and uses group statistics
# as the baseline — no value network needed.
#
# For prompt x, generate completions {y_1,...,y_G}, score with R(y_i,x).
# Group-relative advantage: A_i = (r_i - mean(r_group)) / std(r_group)

rng = np.random.default_rng(42)


def compute_grpo_advantages(
    rewards: np.ndarray,
    group_size: int,
) -> np.ndarray:
    """Compute group-relative advantages for GRPO.

    For each group of G completions from the same prompt:
      A_i = (r_i - mean(r_group)) / max(std(r_group), eps)

    Args:
        rewards: (n_prompts * group_size,) flat array of rewards
        group_size: number of completions per prompt (G)

    Returns:
        advantages: (n_prompts * group_size,) normalized advantages
    """
    n_total = len(rewards)
    n_prompts = n_total // group_size
    assert n_total == n_prompts * group_size, "Rewards must be divisible by group_size"

    # TODO: Reshape rewards to (n_prompts, group_size), normalize within each group
    # Hint: grouped = rewards.reshape(n_prompts, group_size)
    #   group_means = grouped.mean(axis=1, keepdims=True)
    #   group_stds = grouped.std(axis=1, keepdims=True)
    #   advantages = (grouped - group_means) / np.maximum(group_stds, 1e-8)
    ____
    ____
    ____
    ____
    ____


n_prompts = 20
group_size = 8
n_total = n_prompts * group_size

prompt_difficulty = rng.uniform(0.2, 0.8, n_prompts)
raw_rewards = np.zeros(n_total)
for i in range(n_prompts):
    base = prompt_difficulty[i]
    raw_rewards[i * group_size : (i + 1) * group_size] = rng.normal(
        base, 0.2, group_size
    )

advantages = compute_grpo_advantages(raw_rewards, group_size)

print(f"\n=== GRPO Advantages ===")
print(f"  Prompts: {n_prompts}, Completions per prompt: {group_size}")
print(f"  Raw rewards:  mean={raw_rewards.mean():.3f}, std={raw_rewards.std():.3f}")
print(f"  Advantages:   mean={advantages.mean():.6f}, std={advantages.std():.3f}")
print(f"  Advantage range: [{advantages.min():.3f}, {advantages.max():.3f}]")

grouped_adv = advantages.reshape(n_prompts, group_size)
group_means = grouped_adv.mean(axis=1)
print(f"  Per-group mean (should be ~0): max|mean|={np.max(np.abs(group_means)):.2e}")

print(f"\n  Example group (prompt 0):")
print(f"    Rewards:    {raw_rewards[:group_size].round(3)}")
print(f"    Advantages: {advantages[:group_size].round(3)}")
print(f"    Best completion: index {np.argmax(advantages[:group_size])}")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Implement verifiable reward function for SG regulatory Q&A
# ══════════════════════════════════════════════════════════════════════
# GRPO works best with verifiable rewards — programmatically computable
# without a learned reward model (math correctness, code execution, format).

reg_ids = regulations.select("regulation_id").to_series().to_list()
reg_categories = regulations.select("category").to_series().unique().to_list()
reg_texts = regulations.select("text").to_series().to_list()

regulatory_keywords = {
    "general": ["compliance", "regulation", "authority", "enforcement", "penalty"],
    "financial": ["MAS", "capital", "liquidity", "risk", "prudential", "reporting"],
    "data_protection": ["PDPA", "consent", "personal data", "breach", "notification"],
    "employment": ["MOM", "employment", "CPF", "leave", "workplace", "safety"],
}


def regulatory_reward(
    completion: str,
    prompt: str,
    regulation_ids: list[str],
    keywords: dict[str, list[str]],
) -> dict:
    """Compute verifiable reward for a regulatory Q&A response.

    Components (each 0-1, summed):
      1. Citation reward: mentions a valid regulation ID
      2. Format reward: has structured sections
      3. Keyword reward: covers relevant regulatory terms
      4. Length reward: reasonable length (100-500 words)
    """
    scores = {}

    # TODO: Compute citation score (fraction of regulation IDs found, capped at 1.0)
    # Hint: citation_hits = sum(1 for reg_id in regulation_ids if reg_id in completion)
    #   scores["citation"] = min(1.0, citation_hits / 2)
    citation_hits = ____
    scores["citation"] = ____

    # TODO: Compute format score (structured answer signals, capped at 1.0)
    # Hint: format_signals = [":", "\n-", "\n1.", "\n2.", "section", "article"]
    #   scores["format"] = min(1.0, format_hits / 3)
    format_signals = [":", "\n-", "\n1.", "\n2.", "section", "article"]
    format_hits = ____
    scores["format"] = ____

    # TODO: Compute keyword score (regulatory term coverage, capped at 1.0)
    # Hint: all_keywords = [kw for kws in keywords.values() for kw in kws]
    #   scores["keywords"] = min(1.0, keyword_hits / 5)
    all_keywords = ____
    keyword_hits = ____
    scores["keywords"] = ____

    # Length reward: prefer 100-500 word responses
    word_count = len(completion.split())
    if word_count < 50:
        scores["length"] = word_count / 50
    elif word_count <= 500:
        scores["length"] = 1.0
    else:
        scores["length"] = max(0.0, 1.0 - (word_count - 500) / 500)

    scores["total"] = sum(scores.values())
    scores["max_possible"] = 4.0

    return scores


good_completion = (
    "Under the Personal Data Protection Act (PDPA), organisations must obtain "
    "consent before collecting personal data. Key requirements:\n"
    "1. Purpose limitation: data collected for stated purposes only\n"
    "2. Notification obligation: individuals must be informed of purposes\n"
    "3. Breach notification: PDPC must be notified within 3 days\n"
    "Reference: PDPA Section 13, MAS Guidelines on Technology Risk Management"
)

poor_completion = "Singapore has many regulations. They are important for business."

sample_prompt = "What are the key data protection requirements in Singapore?"

good_scores = regulatory_reward(
    good_completion, sample_prompt, reg_ids, regulatory_keywords
)
poor_scores = regulatory_reward(
    poor_completion, sample_prompt, reg_ids, regulatory_keywords
)

print(f"\n=== Verifiable Reward Function ===")
print(f"\n  Good response scores:")
for component, score in good_scores.items():
    if component != "max_possible":
        print(f"    {component}: {score:.2f}")

print(f"\n  Poor response scores:")
for component, score in poor_scores.items():
    if component != "max_possible":
        print(f"    {component}: {score:.2f}")

print(f"\n  Reward gap: {good_scores['total'] - poor_scores['total']:.2f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Implement clipped GRPO update
# ══════════════════════════════════════════════════════════════════════
# Same clipping as PPO but with group-relative advantages:
#   L^CLIP = min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)
# where r_t = pi_theta(y|x) / pi_old(y|x) is the importance ratio.


def grpo_clipped_loss(
    log_probs_new: np.ndarray,
    log_probs_old: np.ndarray,
    advantages: np.ndarray,
    clip_epsilon: float = 0.2,
    kl_coef: float = 0.001,
    log_probs_ref: np.ndarray | None = None,
) -> dict:
    """Compute the clipped GRPO surrogate objective.

    L = -E[min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)]
      + kl_coef * KL(pi || pi_ref)
    """
    # TODO: Compute importance ratio and both surrogate objectives
    # Hint: ratio = np.exp(log_probs_new - log_probs_old)
    #   surr1 = ratio * advantages
    #   clipped_ratio = np.clip(ratio, 1-clip_epsilon, 1+clip_epsilon)
    #   surr2 = clipped_ratio * advantages
    #   policy_loss = -np.mean(np.minimum(surr1, surr2))
    ratio = ____
    surr1 = ____
    clipped_ratio = ____
    surr2 = ____
    policy_loss = ____

    # KL penalty against reference policy
    kl_div = 0.0
    if log_probs_ref is not None:
        kl_div = float(np.mean(log_probs_new - log_probs_ref))

    total_loss = policy_loss + kl_coef * kl_div
    clip_fraction = float(np.mean(np.abs(ratio - 1.0) > clip_epsilon))

    return {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "kl_penalty": kl_coef * kl_div,
        "kl_divergence": kl_div,
        "clip_fraction": clip_fraction,
        "mean_ratio": float(np.mean(ratio)),
        "mean_advantage": float(np.mean(advantages)),
    }


log_probs_old = rng.normal(-3.0, 0.5, n_total)
log_probs_ref = log_probs_old.copy()

shift = advantages * 0.1
log_probs_new = log_probs_old + shift

grpo_result = grpo_clipped_loss(
    log_probs_new,
    log_probs_old,
    advantages,
    clip_epsilon=0.2,
    kl_coef=0.001,
    log_probs_ref=log_probs_ref,
)

print(f"\n=== Clipped GRPO Update ===")
for key, val in grpo_result.items():
    print(f"  {key}: {val:.6f}")

print(f"\n  Effect of clip_epsilon:")
for eps in [0.05, 0.1, 0.2, 0.3, 0.5]:
    result = grpo_clipped_loss(
        log_probs_new,
        log_probs_old,
        advantages,
        clip_epsilon=eps,
        kl_coef=0.001,
        log_probs_ref=log_probs_ref,
    )
    print(
        f"    eps={eps:.2f} -> loss={result['total_loss']:.6f}, "
        f"clip_frac={result['clip_fraction']:.4f}"
    )


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Compare GRPO vs PPO on regulatory Q&A
# ══════════════════════════════════════════════════════════════════════
# GRPO eliminates the value network.
# PPO: L = L^CLIP(policy) + c1*L^VF(value) + c2*S[pi](entropy)
# GRPO: L = L^CLIP(policy) + kl_coef*KL(pi || pi_ref)


def simulate_training_comparison(
    n_steps: int = 50,
    n_prompts_per_step: int = 10,
    group_size: int = 8,
) -> dict:
    """Simulate GRPO vs PPO training dynamics."""
    grpo_losses = []
    ppo_losses = []
    grpo_rewards = []
    ppo_rewards = []

    for step in range(n_steps):
        n = n_prompts_per_step * group_size

        improvement = step / n_steps * 0.5
        rewards = rng.normal(0.3 + improvement, 0.3, n).clip(0, 1)

        # TODO: Compute GRPO advantages and loss for this step
        # Hint: grpo_adv = compute_grpo_advantages(rewards, group_size)
        #   grpo_log_old = rng.normal(-3, 0.3, n)
        #   grpo_log_new = grpo_log_old + grpo_adv * 0.05
        #   grpo_res = grpo_clipped_loss(grpo_log_new, grpo_log_old, grpo_adv, kl_coef=0.001)
        grpo_adv = ____
        grpo_log_old = ____
        grpo_log_new = ____
        grpo_res = ____

        # PPO: noisy value-function baseline
        value_baseline = rewards.mean() + rng.normal(0, 0.1)
        ppo_adv = rewards - value_baseline
        ppo_adv = (ppo_adv - ppo_adv.mean()) / (ppo_adv.std() + 1e-8)
        ppo_log_old = rng.normal(-3, 0.3, n)
        ppo_log_new = ppo_log_old + ppo_adv * 0.05
        ppo_res = grpo_clipped_loss(ppo_log_new, ppo_log_old, ppo_adv, kl_coef=0.001)

        grpo_losses.append(grpo_res["total_loss"])
        ppo_losses.append(ppo_res["total_loss"])
        grpo_rewards.append(rewards.mean())
        ppo_rewards.append(rewards.mean())

    return {
        "grpo_losses": grpo_losses,
        "ppo_losses": ppo_losses,
        "grpo_rewards": grpo_rewards,
        "ppo_rewards": ppo_rewards,
    }


comparison = simulate_training_comparison()

print(f"\n=== GRPO vs PPO Comparison ===")
print(f"{'Metric':<25} {'GRPO':>12} {'PPO':>12}")
print("-" * 50)

grpo_final_loss = np.mean(comparison["grpo_losses"][-10:])
ppo_final_loss = np.mean(comparison["ppo_losses"][-10:])
grpo_loss_var = np.var(comparison["grpo_losses"][-10:])
ppo_loss_var = np.var(comparison["ppo_losses"][-10:])

print(
    f"{'Final loss (avg last 10)':<25} {grpo_final_loss:>12.6f} {ppo_final_loss:>12.6f}"
)
print(f"{'Loss variance':<25} {grpo_loss_var:>12.6f} {ppo_loss_var:>12.6f}")
print(f"{'Value network params':<25} {'0':>12} {'d_model*4':>12}")
print(f"{'Baseline type':<25} {'group stats':>12} {'learned V(s)':>12}")
print(f"{'Reward requirement':<25} {'verifiable':>12} {'any':>12}")

print(f"\n  GRPO advantages over PPO:")
print(f"    1. No value network (saves ~25% memory)")
print(f"    2. Group normalization is unbiased (no value function approximation error)")
print(f"    3. More stable with verifiable rewards (math, code, structured QA)")
print(f"    4. Simpler implementation (fewer hyperparameters)")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Use AlignmentPipeline(method="grpo") at scale
# ══════════════════════════════════════════════════════════════════════

registry = AdapterRegistry()

# TODO: Create AlignmentConfig with method="grpo" and GRPOConfig
# Hint: AlignmentConfig(method="grpo", base_model_id=base_model,
#   grpo=GRPOConfig(num_generations=4, temperature=0.7, kl_coef=0.001,
#                   max_completion_length=512, per_device_train_batch_size=2,
#                   gradient_accumulation_steps=4, learning_rate=1e-5),
#   lora=LoRAConfig(rank=16, alpha=32, dropout=0.05, target_modules=("q_proj","v_proj")),
#   reward_funcs=["accuracy"], experiment_dir="./grpo_output")
grpo_config = AlignmentConfig(____)

pipeline = AlignmentPipeline(grpo_config, adapter_registry=registry)

print(f"\n=== AlignmentPipeline (GRPO) ===")
print(f"  Method: {grpo_config.method}")
print(f"  Generations per prompt: {grpo_config.grpo.num_generations}")
print(f"  Temperature: {grpo_config.grpo.temperature}")
print(f"  KL coefficient: {grpo_config.grpo.kl_coef}")
print(f"  Reward functions: {grpo_config.reward_funcs}")
print(f"  Pipeline created: {pipeline is not None}")

print(f"\nProduction usage (requires GPU):")
print(f"  @reward_registry.register('sg_regulatory')")
print(f"  def sg_reward(completions, prompts, **kwargs):")
print(f"      return [regulatory_reward(c, p, reg_ids, keywords)['total']")
print(f"              for c, p in zip(completions, prompts)]")
print(f"")
print(
    f"  result = await pipeline.train(dataset=prompt_dataset, adapter_name='sg_grpo_v1')"
)

print(f"\n=== GRPO Reasoning Traces (DeepSeek-R1 Pattern) ===")
print(f"  GRPO incentivizes step-by-step reasoning when reward is verifiable:")
print(f"  <think>")
print(f"    The question asks about PDPA breach notification requirements.")
print(f"    Under Section 26D, organisations must notify PDPC...")
print(f"  </think>")
print(f"  Under the PDPA, organisations must notify the PDPC within 3 calendar days")
print(f"  of assessing that a data breach is notifiable (Section 26D).")

print("\n=== Exercise 3 complete -- GRPO and group relative policy optimization ===")
