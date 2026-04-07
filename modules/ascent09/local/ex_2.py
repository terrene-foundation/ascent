# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT09 — Exercise 2: DPO Derivation and Reward Modeling
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Derive DPO from RLHF first principles — implement the
#   Bradley-Terry preference model, derive and implement the DPO loss,
#   and validate against AlignmentPipeline at multiple beta values.
#
# TASKS:
#   1. Implement Bradley-Terry preference model
#   2. Derive DPO loss from RLHF (show reward model elimination)
#   3. Implement DPO loss numerically and verify gradients
#   4. Run AlignmentPipeline(method="dpo") at beta=0.01, 0.1, 1.0
#   5. Evaluate DPO vs base model using LLM-as-judge
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os

import numpy as np
import polars as pl

from kaizen_agents import Delegate
from kaizen import Signature, InputField, OutputField

from kailash_align import AlignmentConfig, AlignmentPipeline, AdapterRegistry
from kailash_align.config import DPOConfig, LoRAConfig
from kailash_align.registry import AdapterSignature

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()

base_model = os.environ.get("SFT_BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
_raw_llm = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))
llm_model = _raw_llm if os.environ.get("OPENAI_API_KEY") else None

# ── Data Loading ─────────────────────────────────────────────────────

loader = ASCENTDataLoader()
preferences = loader.load("ascent10", "preference_pairs.parquet")

print(f"=== Preference Pairs Dataset ===")
print(f"Shape: {preferences.shape}")
print(f"Columns: {preferences.columns}")
print(preferences.head(3))

# TODO: Split into 80% train, 20% eval
n_train = ____
train_prefs = ____
eval_prefs = ____
print(f"\nTrain: {train_prefs.height}, Eval: {eval_prefs.height}")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Implement Bradley-Terry preference model
# ══════════════════════════════════════════════════════════════════════
# P(y_w > y_l | x) = sigma(r(x, y_w) - r(x, y_l))


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    # TODO: np.where(x >= 0, 1/(1+exp(-x)), exp(x)/(1+exp(x)))
    return ____


def bradley_terry_log_likelihood(
    rewards_chosen: np.ndarray,
    rewards_rejected: np.ndarray,
) -> float:
    """L = sum(log(sigma(r_chosen - r_rejected)))"""
    # TODO: diffs, log_probs = log(sigmoid(diffs) + 1e-10), return sum
    ____
    ____
    return ____


def bradley_terry_accuracy(
    rewards_chosen: np.ndarray,
    rewards_rejected: np.ndarray,
) -> float:
    """Fraction where reward model prefers chosen."""
    return ____


rng = np.random.default_rng(42)
n_pairs = preferences.height

# TODO: Simulate rewards — chosen mean=1.5 std=0.8, rejected mean=0.5 std=0.8
r_chosen = ____
r_rejected = ____

bt_ll = ____
bt_acc = ____

print(f"\n=== Bradley-Terry Preference Model ===")
print(f"  Log-likelihood: {bt_ll:.2f}")
print(f"  Accuracy: {bt_acc:.4f}")
print(f"  Mean reward gap: {np.mean(r_chosen - r_rejected):.3f}")

# TODO: Show how accuracy varies with reward gap
print(f"\n  Reward Gap vs Preference Accuracy:")
for gap in [0.0, 0.5, 1.0, 2.0, 3.0]:
    acc = ____
    print(f"    Gap={gap:.1f} -> P(chosen)={acc:.4f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Derive DPO loss (eliminate the reward model)
# ══════════════════════════════════════════════════════════════════════
# RLHF: train reward model, then PPO.
# DPO: r(x,y) = beta*log(pi/pi_ref) + beta*log(Z(x))
# Substitute into Bradley-Terry, Z(x) cancels:
#   L_DPO = -E[log sigma(beta*(log_ratio_w - log_ratio_l))]

print(f"\n=== DPO Derivation ===")
print(f"  RLHF: Train reward model, then PPO to maximize r(x,y) - beta*KL")
print(f"  DPO:  Skip the reward model — use policy log-ratios directly")
print(f"  L_DPO = -E[log sigma(beta * (log_ratio_w - log_ratio_l))]")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Implement DPO loss numerically and verify gradients
# ══════════════════════════════════════════════════════════════════════


def dpo_loss(
    log_pi_chosen: np.ndarray,
    log_pi_rejected: np.ndarray,
    log_ref_chosen: np.ndarray,
    log_ref_rejected: np.ndarray,
    beta: float = 0.1,
) -> dict:
    """Compute DPO loss and implicit rewards."""
    # TODO: log_ratio = log_pi - log_ref for chosen and rejected
    log_ratio_chosen = ____
    log_ratio_rejected = ____

    # TODO: implicit rewards = beta * log_ratio
    implicit_r_chosen = ____
    implicit_r_rejected = ____

    # TODO: logits = beta * (ratio_w - ratio_l); loss = -mean(log(sigmoid(logits)))
    logits = ____
    loss = ____

    accuracy = ____

    # TODO: Return dict with loss, implicit_rewards_chosen/rejected, accuracy, mean_logit
    return ____


# TODO: Simulate reference log-probs (chosen ~ N(-2.0, 0.5), rejected ~ N(-2.5, 0.5))
n_sim = 200
log_ref_chosen = ____
log_ref_rejected = ____

# Policy = reference initially
log_pi_chosen = ____
log_pi_rejected = ____

result_init = ____
print(f"\n=== DPO Loss (policy = reference) ===")
print(f"  Loss: {result_init['loss']:.4f}, Accuracy: {result_init['accuracy']:.4f}")

# TODO: Simulate training — chosen +0.5, rejected -0.3
log_pi_chosen_trained = ____
log_pi_rejected_trained = ____

result_trained = ____
print(f"\n=== DPO Loss (after training) ===")
print(f"  Loss: {result_trained['loss']:.4f} (from {result_init['loss']:.4f})")
print(f"  Accuracy: {result_trained['accuracy']:.4f}")

assert result_trained["loss"] < result_init["loss"], "Loss should decrease"
print(f"  Gradient verified: pushing chosen up, rejected down reduces loss")

# TODO: Print implicit reward distribution stats
ir_chosen = result_trained["implicit_rewards_chosen"]
ir_rejected = result_trained["implicit_rewards_rejected"]
print(f"  Implicit reward gap: {np.mean(ir_chosen) - np.mean(ir_rejected):.4f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Run AlignmentPipeline(method="dpo") at beta=0.01, 0.1, 1.0
# ══════════════════════════════════════════════════════════════════════

registry = AdapterRegistry()
beta_configs = {}

for beta in [0.01, 0.1, 1.0]:
    # TODO: AlignmentConfig with method="dpo",
    #   DPOConfig(beta, epochs=2, batch=2, lr=5e-5, max_length=512),
    #   LoRAConfig(rank=16, alpha=32, dropout=0.05, target_modules=("q_proj","v_proj"))
    config = AlignmentConfig(
        method=____,
        base_model_id=____,
        dpo=DPOConfig(____),
        lora=LoRAConfig(____),
        experiment_dir=____,
    )
    beta_configs[beta] = config
    pipeline = ____
    print(f"  Beta={beta}: pipeline created")

# TODO: Show simulated DPO loss across beta values
print(f"\n  Simulated DPO loss:")
for beta in [0.01, 0.1, 0.5, 1.0, 5.0]:
    result = ____
    print(
        f"    beta={beta:<5} -> loss={result['loss']:.4f}, acc={result['accuracy']:.4f}"
    )

print(f"\nProduction: result = await pipeline.train(dataset=..., adapter_name='...')")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Evaluate DPO vs base model using LLM-as-judge
# ══════════════════════════════════════════════════════════════════════


class SafetyJudge(Signature):
    """LLM-as-judge for response quality evaluation."""

    prompt: str = InputField(desc="The original question")
    response_a: str = InputField(desc="Response from model A")
    response_b: str = InputField(desc="Response from model B")
    verdict: str = OutputField(desc="Which is better: A, B, or tie")
    reasoning: str = OutputField(desc="Why one response is better")


async def evaluate_with_judge():
    """Evaluate DPO-aligned vs base model using LLM-as-judge."""
    if not llm_model:
        print("\n=== Evaluation (API key not set -- showing pattern) ===")
        print(f"  delegate = Delegate(model=os.environ['DEFAULT_LLM_MODEL'])")
        print(f"  verdict = await judge.run(SafetyJudge(...))")
        return

    # TODO: Create Delegate, evaluate 5 samples from eval_prefs
    delegate = ____
    # TODO: Get 5 eval samples, extract prompt/chosen/rejected columns
    eval_samples = ____
    prompts = ____
    chosen = ____
    rejected = ____

    print(f"\n=== LLM-as-Judge Evaluation ({len(prompts)} samples) ===")
    verdicts = ____

    for i, (prompt, resp_chosen, resp_rejected) in enumerate(
        zip(prompts, chosen, rejected)
    ):
        # TODO: Random swap to mitigate position bias
        swap = ____
        resp_a, resp_b = ____

        # TODO: Build judge prompt, stream delegate.run(), parse A/B/tie
        judge_prompt = ____
        response_text = ""
        async for event in delegate.run(judge_prompt):
            if hasattr(event, "text"):
                response_text += event.text

        # TODO: Parse verdict, account for swap
        raw_verdict = ____
        actual_verdict = ____
        verdicts[actual_verdict] = verdicts.get(actual_verdict, 0) + 1

    # TODO: Print results — agreement rate with preference labels
    total = ____
    print(f"\n  Agreement with labels: {____:.0%}")


asyncio.run(evaluate_with_judge())

print("\n=== Exercise 2 complete ===")
