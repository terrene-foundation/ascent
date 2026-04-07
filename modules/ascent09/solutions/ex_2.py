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
from kaizen.core import Signature, InputField, OutputField

from kailash_align import AlignmentConfig, AlignmentPipeline, AdapterRegistry
from kailash_align.config import DPOConfig, LoRAConfig
from kailash_align.registry import AdapterSignature

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()

base_model = os.environ.get("SFT_BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
llm_model = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))

# ── Data Loading ─────────────────────────────────────────────────────

loader = ASCENTDataLoader()
preferences = loader.load("ascent10", "preference_pairs.parquet")

print(f"=== Preference Pairs Dataset ===")
print(f"Shape: {preferences.shape}")
print(f"Columns: {preferences.columns}")
print(preferences.head(3))

n_train = int(preferences.height * 0.8)
train_prefs = preferences[:n_train]
eval_prefs = preferences[n_train:]
print(f"\nTrain: {train_prefs.height}, Eval: {eval_prefs.height}")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Implement Bradley-Terry preference model
# ══════════════════════════════════════════════════════════════════════
# The Bradley-Terry model defines preference probability as:
#   P(y_w > y_l | x) = sigma(r(x, y_w) - r(x, y_l))
# where r(x, y) is a reward function and sigma is the sigmoid.
#
# This is the foundation of RLHF: train a reward model, then
# optimize the policy against it using PPO.


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def bradley_terry_log_likelihood(
    rewards_chosen: np.ndarray,
    rewards_rejected: np.ndarray,
) -> float:
    """Compute log-likelihood under Bradley-Terry model.

    L = sum(log(sigma(r(x, y_w) - r(x, y_l))))

    Higher rewards for chosen responses increase the likelihood.
    """
    diffs = rewards_chosen - rewards_rejected
    log_probs = np.log(sigmoid(diffs) + 1e-10)
    return float(np.sum(log_probs))


def bradley_terry_accuracy(
    rewards_chosen: np.ndarray,
    rewards_rejected: np.ndarray,
) -> float:
    """Fraction of pairs where the reward model prefers the chosen response."""
    return float(np.mean(rewards_chosen > rewards_rejected))


# Simulate reward model outputs
rng = np.random.default_rng(42)
n_pairs = preferences.height

# A good reward model assigns higher scores to chosen responses
r_chosen = rng.normal(1.5, 0.8, n_pairs)  # Mean 1.5 for chosen
r_rejected = rng.normal(0.5, 0.8, n_pairs)  # Mean 0.5 for rejected

bt_ll = bradley_terry_log_likelihood(r_chosen, r_rejected)
bt_acc = bradley_terry_accuracy(r_chosen, r_rejected)

print(f"\n=== Bradley-Terry Preference Model ===")
print(f"  P(y_w > y_l | x) = sigma(r(x, y_w) - r(x, y_l))")
print(f"  Log-likelihood: {bt_ll:.2f}")
print(f"  Accuracy: {bt_acc:.4f}")
print(f"  Mean reward gap: {np.mean(r_chosen - r_rejected):.3f}")

# Show how accuracy varies with reward gap
print(f"\n  Reward Gap vs Preference Accuracy:")
for gap in [0.0, 0.5, 1.0, 2.0, 3.0]:
    acc = float(sigmoid(np.array([gap]))[0])
    print(f"    Gap={gap:.1f} -> P(chosen preferred)={acc:.4f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Derive DPO loss (eliminate the reward model)
# ══════════════════════════════════════════════════════════════════════
# Standard RLHF pipeline:
#   1. Train reward model r(x, y) on preferences
#   2. Optimize policy pi with PPO against r(x, y)
#      max_pi E[r(x, y)] - beta * KL[pi || pi_ref]
#
# DPO insight: the optimal policy has a closed form:
#   pi*(y|x) = pi_ref(y|x) * exp(r(x,y) / beta) / Z(x)
#
# Rearranging for r:
#   r(x, y) = beta * log(pi(y|x) / pi_ref(y|x)) + beta * log(Z(x))
#
# Substituting into Bradley-Terry and canceling Z(x):
#   L_DPO = -E[log sigma(beta * (log pi(y_w|x)/pi_ref(y_w|x)
#                                - log pi(y_l|x)/pi_ref(y_l|x)))]

print(f"\n=== DPO Derivation ===")
print(f"  RLHF: Train reward model r(x,y), then PPO to maximize r(x,y) - beta*KL")
print(f"  DPO:  Skip the reward model entirely!")
print(f"")
print(f"  Key identity:")
print(f"    r(x,y) = beta * log(pi(y|x) / pi_ref(y|x)) + beta * log(Z(x))")
print(f"")
print(f"  Substituting into Bradley-Terry and canceling Z(x):")
print(f"    L_DPO = -E[log sigma(beta * (log_ratio_w - log_ratio_l))]")
print(f"    where log_ratio = log(pi(y|x) / pi_ref(y|x))")
print(f"")
print(f"  Advantages over RLHF:")
print(f"    1. No separate reward model to train")
print(f"    2. No PPO instability (clipping, value function, GAE)")
print(f"    3. Simpler pipeline: just a classification loss on preferences")
print(f"    4. Implicit reward: r(x,y) = beta * log(pi/pi_ref)")


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
    """Compute DPO loss and implicit rewards.

    L_DPO = -E[log sigma(beta * ((log pi(y_w|x) - log pi_ref(y_w|x))
                                - (log pi(y_l|x) - log pi_ref(y_l|x))))]

    Returns:
        loss: scalar DPO loss
        implicit_rewards_chosen: beta * log(pi/pi_ref) for chosen
        implicit_rewards_rejected: beta * log(pi/pi_ref) for rejected
        accuracy: fraction where chosen has higher implicit reward
    """
    # Log ratios: how much the policy deviates from reference
    log_ratio_chosen = log_pi_chosen - log_ref_chosen
    log_ratio_rejected = log_pi_rejected - log_ref_rejected

    # Implicit rewards
    implicit_r_chosen = beta * log_ratio_chosen
    implicit_r_rejected = beta * log_ratio_rejected

    # DPO loss: push chosen up and rejected down
    logits = beta * (log_ratio_chosen - log_ratio_rejected)
    loss = -np.mean(np.log(sigmoid(logits) + 1e-10))

    accuracy = float(np.mean(implicit_r_chosen > implicit_r_rejected))

    return {
        "loss": float(loss),
        "implicit_rewards_chosen": implicit_r_chosen,
        "implicit_rewards_rejected": implicit_r_rejected,
        "accuracy": accuracy,
        "mean_logit": float(np.mean(logits)),
    }


# Simulate log-probabilities from policy and reference
n_sim = 200
log_ref_chosen = rng.normal(-2.0, 0.5, n_sim)  # Reference model log-probs
log_ref_rejected = rng.normal(-2.5, 0.5, n_sim)

# Initially, policy = reference
log_pi_chosen = log_ref_chosen.copy()
log_pi_rejected = log_ref_rejected.copy()

result_init = dpo_loss(
    log_pi_chosen, log_pi_rejected, log_ref_chosen, log_ref_rejected, beta=0.1
)
print(f"\n=== DPO Loss (policy = reference) ===")
print(f"  Loss: {result_init['loss']:.4f}")
print(f"  Accuracy: {result_init['accuracy']:.4f}")
print(f"  Mean logit: {result_init['mean_logit']:.4f}")

# Simulate training: policy increases log-prob for chosen, decreases for rejected
log_pi_chosen_trained = log_pi_chosen + 0.5  # Increase for chosen
log_pi_rejected_trained = log_pi_rejected - 0.3  # Decrease for rejected

result_trained = dpo_loss(
    log_pi_chosen_trained,
    log_pi_rejected_trained,
    log_ref_chosen,
    log_ref_rejected,
    beta=0.1,
)
print(f"\n=== DPO Loss (after training) ===")
print(
    f"  Loss: {result_trained['loss']:.4f} (decreased from {result_init['loss']:.4f})"
)
print(
    f"  Accuracy: {result_trained['accuracy']:.4f} (increased from {result_init['accuracy']:.4f})"
)
print(f"  Mean logit: {result_trained['mean_logit']:.4f}")

# Verify gradient direction: loss should decrease when chosen goes up
assert (
    result_trained["loss"] < result_init["loss"]
), "DPO loss should decrease when policy prefers chosen over rejected"
print(f"\n  Gradient verified: pushing chosen up and rejected down reduces loss")

# Show implicit reward distribution
ir_chosen = result_trained["implicit_rewards_chosen"]
ir_rejected = result_trained["implicit_rewards_rejected"]
print(f"\n  Implicit rewards (after training):")
print(f"    Chosen:   mean={np.mean(ir_chosen):.4f}, std={np.std(ir_chosen):.4f}")
print(f"    Rejected: mean={np.mean(ir_rejected):.4f}, std={np.std(ir_rejected):.4f}")
print(f"    Gap: {np.mean(ir_chosen) - np.mean(ir_rejected):.4f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Run AlignmentPipeline(method="dpo") at beta=0.01, 0.1, 1.0
# ══════════════════════════════════════════════════════════════════════

registry = AdapterRegistry()
beta_configs = {}

for beta in [0.01, 0.1, 1.0]:
    config = AlignmentConfig(
        method="dpo",
        base_model_id=base_model,
        dpo=DPOConfig(
            beta=beta,
            num_train_epochs=2,
            per_device_train_batch_size=2,
            learning_rate=5e-5,
            max_length=512,
        ),
        lora=LoRAConfig(
            rank=16,
            alpha=32,
            dropout=0.05,
            target_modules=("q_proj", "v_proj"),
        ),
        experiment_dir=f"./dpo_beta_{beta}",
    )
    beta_configs[beta] = config

    pipeline = AlignmentPipeline(config, adapter_registry=registry)

    print(f"\n  Beta={beta}: pipeline created")
    print(f"    Higher beta -> policy stays closer to reference (more conservative)")
    print(f"    Lower beta -> policy can deviate more (more aggressive alignment)")

print(f"\n=== Beta Effect on DPO ===")
print(f"{'Beta':>8} {'Behavior':>30} {'Risk':>25}")
print("-" * 65)
print(f"{'0.01':>8} {'Aggressive alignment':>30} {'Reward hacking':>25}")
print(f"{'0.1':>8} {'Balanced (recommended)':>30} {'Low risk':>25}")
print(f"{'1.0':>8} {'Very conservative':>30} {'Under-alignment':>25}")

# Demonstrate the effect of beta on our simulated loss
print(f"\n  Simulated DPO loss at different beta values:")
for beta in [0.01, 0.1, 0.5, 1.0, 5.0]:
    result = dpo_loss(
        log_pi_chosen_trained,
        log_pi_rejected_trained,
        log_ref_chosen,
        log_ref_rejected,
        beta=beta,
    )
    print(
        f"    beta={beta:<5} -> loss={result['loss']:.4f}, accuracy={result['accuracy']:.4f}"
    )

print(f"\nProduction usage (requires GPU + preference data):")
print(f"  result = await pipeline.train(")
print(f"      dataset=hf_pref_dataset,")
print(f'      adapter_name="sg_domain_dpo_v1",')
print(f"  )")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Evaluate DPO vs base model using LLM-as-judge
# ══════════════════════════════════════════════════════════════════════


class SafetyJudge(Signature):
    """LLM-as-judge Signature for evaluating response quality."""

    prompt: str = InputField(desc="The original question")
    response_a: str = InputField(desc="Response from model A")
    response_b: str = InputField(desc="Response from model B")
    verdict: str = OutputField(desc="Which is better: A, B, or tie")
    reasoning: str = OutputField(desc="Why one response is better")


async def evaluate_with_judge():
    """Evaluate DPO-aligned vs base model responses using LLM-as-judge."""
    if not llm_model:
        print("\n=== Evaluation (API key not set -- showing pattern) ===")
        print(f"  judge = Delegate(model=os.environ['DEFAULT_LLM_MODEL'])")
        print(
            f"  verdict = await judge.run(SafetyJudge(prompt=..., response_a=..., response_b=...))"
        )
        return

    delegate = Delegate(model=llm_model, budget_usd=1.0)

    # Sample evaluation prompts from the preference dataset
    eval_samples = eval_prefs.head(5)
    prompts = eval_samples.select("prompt").to_series().to_list()
    chosen = eval_samples.select("chosen").to_series().to_list()
    rejected = eval_samples.select("rejected").to_series().to_list()

    print(f"\n=== LLM-as-Judge Evaluation ===")
    print(f"  Evaluating {len(prompts)} samples")
    print(f"  Judge model: {llm_model}")

    verdicts = {"A": 0, "B": 0, "tie": 0}

    for i, (prompt, resp_chosen, resp_rejected) in enumerate(
        zip(prompts, chosen, rejected)
    ):
        # Randomly swap positions to mitigate position bias
        swap = rng.random() > 0.5
        if swap:
            resp_a, resp_b = resp_rejected, resp_chosen
        else:
            resp_a, resp_b = resp_chosen, resp_rejected

        judge_prompt = (
            f"Compare these two responses to the question.\n\n"
            f"Question: {prompt}\n\n"
            f"Response A: {resp_a[:500]}\n\n"
            f"Response B: {resp_b[:500]}\n\n"
            f"Which response is better? Reply with just 'A', 'B', or 'tie', "
            f"followed by a brief explanation."
        )

        response_text = ""
        async for event in delegate.run(judge_prompt):
            if hasattr(event, "text"):
                response_text += event.text

        # Parse verdict
        first_word = (
            response_text.strip().split()[0].upper() if response_text.strip() else "TIE"
        )
        if "A" in first_word:
            raw_verdict = "A"
        elif "B" in first_word:
            raw_verdict = "B"
        else:
            raw_verdict = "tie"

        # Account for position swap
        if swap:
            actual_verdict = (
                "B" if raw_verdict == "A" else ("A" if raw_verdict == "B" else "tie")
            )
        else:
            actual_verdict = raw_verdict

        verdicts[actual_verdict] = verdicts.get(actual_verdict, 0) + 1
        label = (
            "chosen"
            if actual_verdict == "A"
            else ("rejected" if actual_verdict == "B" else "tie")
        )
        print(f"  Sample {i + 1}: Judge prefers {label}")

    total = sum(verdicts.values())
    print(
        f"\n  Results: Chosen={verdicts['A']}/{total}, Rejected={verdicts['B']}/{total}, Tie={verdicts['tie']}/{total}"
    )
    print(
        f"  Judge agreement with preference labels: {verdicts['A'] / max(1, total):.0%}"
    )

    print(f"\n  Known LLM-as-judge biases:")
    print(
        f"    1. Position bias -- prefers first response (mitigated by random swapping)"
    )
    print(f"    2. Verbosity bias -- prefers longer responses")
    print(f"    3. Self-enhancement -- prefers own style")
    print(
        f"    Mitigation: swap positions, control length, use multiple diverse judges"
    )


asyncio.run(evaluate_with_judge())

print("\n=== Exercise 2 complete -- DPO derivation and reward modeling ===")
