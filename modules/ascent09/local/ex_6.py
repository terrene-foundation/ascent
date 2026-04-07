# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT09 — Exercise 6: Reward Hacking and Goodhart's Law
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Study reward misspecification and Goodhart's Law — when a
#   measure becomes a target, it ceases to be a good measure. Implement
#   proxy rewards, demonstrate gaming, and build robust reward defenses.
#
# TASKS:
#   1. Define proxy rewards and show optimization against them
#   2. Demonstrate how policies game the proxy metric
#   3. Implement KL-regularization to prevent gaming
#   4. Implement reward ensemble (multiple independent proxies)
#   5. Design robust reward for SG regulatory Q&A
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os

import numpy as np
import polars as pl

from kaizen_agents import Delegate
from kaizen import Signature, InputField, OutputField

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()

llm_model = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))

# ── Data Loading ─────────────────────────────────────────────────────

loader = ASCENTDataLoader()
regulations = loader.load("ascent09", "sg_regulations.parquet")

print(f"=== SG Regulations Dataset ===")
print(f"Shape: {regulations.shape}")
print(f"Columns: {regulations.columns}")
print(regulations.head(3))

rng = np.random.default_rng(42)


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Define proxy rewards and show optimization against them
# ══════════════════════════════════════════════════════════════════════
# Goodhart's Law: "When a measure becomes a target, it ceases to be
# a good measure."
# Common proxy rewards that get gamed: length, keyword density, formatting.


def proxy_reward_length(response: str) -> float:
    """Proxy reward: longer responses are better (gets gamed by padding)."""
    # TODO: Count words and return min(1.0, word_count / 200)
    word_count = ____
    return ____


def proxy_reward_keywords(response: str, keywords: list[str]) -> float:
    """Proxy reward: more domain keywords = more knowledgeable (gets gamed by stuffing)."""
    # TODO: Count case-insensitive keyword hits
    # Hint: hits = sum(1 for kw in keywords if kw.lower() in response_lower)
    #   return min(1.0, hits / max(len(keywords), 1))
    response_lower = ____
    hits = ____
    return ____


def proxy_reward_format(response: str) -> float:
    """Proxy reward: structured formatting (gets gamed by empty structures)."""
    # TODO: Score format: +0.2 for newlines, +0.3 for numbered lists,
    #   +0.2 for colons, +0.3 for bullet markers. Return min(1.0, score)
    score = 0.0
    ____
    return ____


def true_quality_reward(response: str, reference: str) -> float:
    """The 'true' reward we actually care about (but can't easily compute)."""
    # TODO: Combine length factor (penalize >300 words) with reference word overlap
    # Hint: length_factor = min(1.0, len(response.split()) / 150) * 0.3
    #   if len(response.split()) > 300: length_factor *= 0.5
    #   overlap = len(ref_words & resp_words) / max(len(ref_words), 1)
    #   return min(1.0, length_factor + overlap * 0.7)
    ____


regulatory_keywords = [
    "MAS",
    "PDPA",
    "compliance",
    "regulation",
    "enforcement",
    "penalty",
    "consent",
    "reporting",
    "KYC",
    "AML",
]

reference_answer = (
    "Under Singapore's PDPA, organisations must obtain consent before collecting "
    "personal data. Key obligations include purpose limitation, notification, "
    "and the right to access and correct data. Non-compliance may result in "
    "fines up to $1 million imposed by the PDPC."
)

good_response = (
    "Singapore's PDPA requires organisations to obtain consent before collecting "
    "personal data. Key requirements:\n"
    "1. Purpose limitation: collect only for stated purposes\n"
    "2. Notification: inform individuals of collection purposes\n"
    "3. Access: individuals can request their data\n"
    "Penalties: fines up to $1 million for non-compliance (PDPC)."
)

gamed_response = (
    "Understanding Singapore's regulatory framework regarding compliance and regulation "
    "is essential for all organisations. The MAS and PDPA provide comprehensive guidance "
    "on enforcement and penalty structures.\n\n"
    "Key Points:\n"
    "1. Compliance with MAS regulations is mandatory\n"
    "2. PDPA consent requirements must be followed\n"
    "3. Reporting obligations under KYC and AML frameworks\n"
    "4. Enforcement actions and penalty implications\n"
    "5. Additional compliance considerations\n"
    "6. Further regulatory guidance\n"
    "7. Supplementary compliance measures\n\n"
    "In conclusion, compliance with Singapore's regulatory requirements including "
    "MAS guidelines, PDPA obligations, KYC protocols, AML frameworks, and enforcement "
    "mechanisms is of paramount importance for all regulated entities."
)

print(f"\n=== Proxy Reward Comparison ===")
print(
    f"  {'Reward':<20} {'Good Response':>15} {'Gamed Response':>15} {'Gamed Wins?':>12}"
)
print(f"  {'-' * 64}")

for name, reward_fn in [
    ("Length", lambda r: proxy_reward_length(r)),
    ("Keywords", lambda r: proxy_reward_keywords(r, regulatory_keywords)),
    ("Format", lambda r: proxy_reward_format(r)),
    ("True Quality", lambda r: true_quality_reward(r, reference_answer)),
]:
    good_score = reward_fn(good_response)
    gamed_score = reward_fn(gamed_response)
    gamed_wins = "YES" if gamed_score > good_score else "no"
    print(f"  {name:<20} {good_score:>15.3f} {gamed_score:>15.3f} {gamed_wins:>12}")

print(f"\n  The gamed response wins on proxies but loses on true quality!")
print(f"  This is Goodhart's Law in action.")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Demonstrate how policies game the proxy metric
# ══════════════════════════════════════════════════════════════════════


def simulate_reward_hacking(
    n_steps: int = 100,
    proxy_weight: float = 1.0,
) -> dict:
    """Simulate a policy optimizing against a proxy reward.

    Proxy rises monotonically; true quality peaks early then degrades.
    """
    proxy_scores = []
    true_scores = []

    for step in range(n_steps):
        progress = step / n_steps

        # TODO: Compute proxy (monotonically rising) and true quality (Goodhart curve)
        # Hint: proxy = 0.3 + 0.7 * (1 - np.exp(-3 * progress * proxy_weight))
        #   if progress < 0.3: true = 0.3 + 1.5 * progress
        #   elif progress < 0.6: true = 0.75 + 0.05 * np.sin(progress * 10)
        #   else: true = 0.75 - 0.3 * (progress - 0.6)
        proxy = ____

        if progress < 0.3:
            true = ____
        elif progress < 0.6:
            true = ____
        else:
            true = ____

        proxy_scores.append(proxy)
        true_scores.append(true)

    return {
        "proxy_scores": proxy_scores,
        "true_scores": true_scores,
        "max_proxy": max(proxy_scores),
        "max_true": max(true_scores),
        "final_proxy": proxy_scores[-1],
        "final_true": true_scores[-1],
        "divergence_step": next(
            (
                i
                for i in range(1, n_steps)
                if true_scores[i] < true_scores[i - 1] and i > n_steps * 0.3
            ),
            n_steps,
        ),
    }


hack_result = simulate_reward_hacking()

print(f"\n=== Reward Hacking Simulation ===")
print(f"  Peak proxy score: {hack_result['max_proxy']:.3f} (at step 100)")
print(f"  Peak true quality: {hack_result['max_true']:.3f} (at step ~30)")
print(f"  Final true quality: {hack_result['final_true']:.3f} (DEGRADED)")
print(f"  Divergence begins at step: ~{hack_result['divergence_step']}")
print(f"")
print(f"  Proxy-True Quality Over Training:")
for step in [0, 10, 25, 50, 75, 100]:
    idx = min(step, 99)
    proxy = hack_result["proxy_scores"][idx]
    true = hack_result["true_scores"][idx]
    marker = " <-- divergence" if abs(step - hack_result["divergence_step"]) < 5 else ""
    print(f"    Step {step:>3}: proxy={proxy:.3f}, true={true:.3f}{marker}")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Implement KL-regularization to prevent gaming
# ══════════════════════════════════════════════════════════════════════
# KL penalty: L = -E[R(y)] + beta * KL[pi || pi_ref]
# Constrains how far the policy can deviate from the reference.


def simulate_kl_regularized(
    n_steps: int = 100,
    kl_coef: float = 0.1,
) -> dict:
    """Simulate training with KL regularization."""
    proxy_scores = []
    true_scores = []
    kl_penalties = []

    for step in range(n_steps):
        progress = step / n_steps

        # TODO: Compute deviation, KL penalty, effective deviation, then proxy and true scores
        # Hint: deviation = progress**1.5
        #   kl_penalty = kl_coef * deviation * 10
        #   effective_deviation = deviation / (1 + kl_coef * 5)
        #   proxy = 0.3 + 0.7 * (1 - np.exp(-3 * effective_deviation))
        #   if effective_deviation < 0.3: true = 0.3 + 1.5 * effective_deviation
        #   elif effective_deviation < 0.6: true = 0.75 + 0.02 * np.sin(effective_deviation * 10)
        #   else: true = 0.75 - 0.05 * (effective_deviation - 0.6)
        deviation = ____
        kl_penalty = ____
        effective_deviation = ____

        proxy = ____

        if effective_deviation < 0.3:
            true = ____
        elif effective_deviation < 0.6:
            true = ____
        else:
            true = ____

        proxy_scores.append(proxy)
        true_scores.append(true)
        kl_penalties.append(kl_penalty)

    return {
        "proxy_scores": proxy_scores,
        "true_scores": true_scores,
        "kl_penalties": kl_penalties,
        "final_proxy": proxy_scores[-1],
        "final_true": true_scores[-1],
    }


print(f"\n=== KL Regularization ===")
print(f"  {'KL Coef':>10} {'Final Proxy':>14} {'Final True':>14} {'Quality Delta':>14}")
print(f"  {'-' * 55}")

no_kl = simulate_reward_hacking()
for kl_coef in [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]:
    if kl_coef == 0.0:
        result = no_kl
        final_proxy = result["final_proxy"]
        final_true = result["final_true"]
    else:
        result = simulate_kl_regularized(kl_coef=kl_coef)
        final_proxy = result["final_proxy"]
        final_true = result["final_true"]

    delta = final_true - no_kl["final_true"]
    print(
        f"  {kl_coef:>10.2f} {final_proxy:>14.3f} {final_true:>14.3f} {delta:>+14.3f}"
    )

print(f"\n  Higher KL coef -> less reward hacking but also less alignment improvement")
print(f"  Sweet spot: kl_coef=0.1 gives best true quality")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Implement reward ensemble (two independent proxies)
# ══════════════════════════════════════════════════════════════════════


def reward_ensemble(
    response: str,
    reference: str,
    keywords: list[str],
    weights: dict[str, float] | None = None,
) -> dict:
    """Ensemble reward combining multiple independent signals.

    Uses weighted average, minimum (pessimistic), and geometric mean.
    """
    if weights is None:
        weights = {"length": 0.15, "keywords": 0.20, "format": 0.15, "quality": 0.50}

    # TODO: Compute all four individual rewards
    # Hint: rewards = {"length": proxy_reward_length(response),
    #   "keywords": proxy_reward_keywords(response, keywords),
    #   "format": proxy_reward_format(response),
    #   "quality": true_quality_reward(response, reference)}
    rewards = ____

    # TODO: Compute weighted average, minimum, and geometric mean
    # Hint: weighted = sum(rewards[k] * weights.get(k, 0) for k in rewards)
    #   minimum = min(rewards.values())
    #   geo_mean = float(np.exp(np.mean(np.log(np.array(list(rewards.values())) + 1e-10))))
    weighted = ____
    minimum = ____
    geo_mean = ____

    return {
        "individual": rewards,
        "weighted_avg": weighted,
        "minimum": minimum,
        "geometric_mean": geo_mean,
    }


good_ensemble = reward_ensemble(good_response, reference_answer, regulatory_keywords)
gamed_ensemble = reward_ensemble(gamed_response, reference_answer, regulatory_keywords)

print(f"\n=== Reward Ensemble ===")
print(
    f"  {'Aggregation':<20} {'Good Response':>15} {'Gamed Response':>15} {'Correct?':>10}"
)
print(f"  {'-' * 62}")
for agg in ["weighted_avg", "minimum", "geometric_mean"]:
    good_val = good_ensemble[agg]
    gamed_val = gamed_ensemble[agg]
    correct = "YES" if good_val > gamed_val else "no"
    print(f"  {agg:<20} {good_val:>15.3f} {gamed_val:>15.3f} {correct:>10}")

print(f"\n  Ensemble benefits:")
print(f"    1. Minimum aggregation catches responses that game individual proxies")
print(f"    2. Geometric mean penalizes any single weak dimension")
print(f"    3. Independent proxies reduce shared failure modes")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Design robust reward for SG regulatory Q&A
# ══════════════════════════════════════════════════════════════════════


def robust_regulatory_reward(
    response: str,
    prompt: str,
    reference: str,
    regulations_df: pl.DataFrame,
) -> dict:
    """Robust reward for SG regulatory Q&A with anti-gaming defenses.

    Signals: factual grounding, specificity, conciseness, hedging, reference overlap.
    Aggregated with geometric mean to prevent single-proxy gaming.
    """
    import re

    scores = {}

    # TODO: Grounding — count genuine regulation ID citations
    # Hint: reg_ids = regulations_df.select("regulation_id").to_series().to_list()
    #   genuine_citations = sum(1 for rid in reg_ids if rid in response)
    #   scores["grounding"] = min(1.0, genuine_citations / 2)
    reg_ids = ____
    genuine_citations = ____
    scores["grounding"] = ____

    # TODO: Specificity — count concrete signals (section numbers, dollar amounts, years, %)
    # Hint: check re.search(r"[Ss]ection\s+\d+", response) etc., cap at min(1.0, signals / 3)
    specificity_signals = 0
    ____
    scores["specificity"] = ____

    # TODO: Conciseness — sweet spot at 30-200 words, penalize >200
    # Hint: word_count < 30: word_count/30; <=200: 1.0; else: max(0.1, 1.0-(wc-200)/200)
    word_count = ____
    scores["conciseness"] = ____

    # TODO: Hedging — count appropriate uncertainty phrases, cap at min(1.0, count / 2)
    # Hint: hedging_phrases = ["please consult", "seek professional", "this is general",
    #   "may vary", "subject to change", "as of", "according to"]
    hedging_phrases = ____
    hedge_count = ____
    scores["hedging"] = ____

    # TODO: Reference overlap — word overlap between response and reference
    # Hint: overlap = len(ref_words & resp_words) / max(len(ref_words), 1)
    #   scores["reference_overlap"] = min(1.0, overlap * 1.5)
    ref_words = ____
    resp_words = ____
    overlap = ____
    scores["reference_overlap"] = ____

    # TODO: Aggregate with geometric mean (anti-gaming)
    # Hint: values = np.clip(np.array(list(scores.values())), 0.01, 1.0)
    #   geometric_mean = float(np.exp(np.mean(np.log(values))))
    values = ____
    geometric_mean = ____
    scores["ensemble_score"] = geometric_mean

    return scores


good_robust = robust_regulatory_reward(
    good_response,
    "What are Singapore's PDPA requirements?",
    reference_answer,
    regulations,
)
gamed_robust = robust_regulatory_reward(
    gamed_response,
    "What are Singapore's PDPA requirements?",
    reference_answer,
    regulations,
)

print(f"\n=== Robust Regulatory Reward ===")
print(f"  {'Component':<25} {'Good Response':>15} {'Gamed Response':>15}")
print(f"  {'-' * 57}")
for key in good_robust:
    print(f"  {key:<25} {good_robust[key]:>15.3f} {gamed_robust[key]:>15.3f}")

good_wins = good_robust["ensemble_score"] > gamed_robust["ensemble_score"]
print(f"\n  Ensemble correctly ranks good > gamed: {good_wins}")

print(f"\n  Anti-gaming defenses:")
print(f"    1. Conciseness penalty prevents length gaming")
print(f"    2. Specificity rewards substance over keyword stuffing")
print(f"    3. Grounding requires real regulation citations")
print(f"    4. Geometric mean prevents any single proxy from dominating")

print(f"\n  Production integration with AlignmentPipeline:")
print(f"    @reward_registry.register('sg_robust')")
print(f"    def sg_robust_reward(completions, prompts, **kwargs):")
print(f"        return [robust_regulatory_reward(c, p, ref, regs)['ensemble_score']")
print(f"                for c, p in zip(completions, prompts)]")
print(f"")
print(f"    config = AlignmentConfig(method='grpo', reward_funcs=['sg_robust'], ...)")

print("\n=== Exercise 6 complete -- reward hacking and Goodhart's Law ===")
