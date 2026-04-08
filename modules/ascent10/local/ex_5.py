# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT10 — Exercise 5: Adversarial Robustness and Attacks
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Implement adversarial attacks (FGSM, PGD) and defences
#   (adversarial training, randomised smoothing) on a tabular fraud
#   detection model. Measure attack success rates and certified robustness.
#
# TASKS:
#   1. Implement FGSM on tabular fraud model — measure ASR at varying epsilon
#   2. Implement PGD (iterative FGSM with projection) — compare ASR vs FGSM
#   3. Adversarial training: augment data with FGSM/PGD examples
#   4. Data poisoning attack: detect via influence function approximation
#   5. Certified robustness via randomised smoothing
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os

import numpy as np
import polars as pl
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()


# ── Data Loading ─────────────────────────────────────────────────────

loader = ASCENTDataLoader()
fraud_data = loader.load("ascent04", "credit_card_fraud.parquet")

print("=== Credit Card Fraud Dataset ===")
print(f"Shape: {fraud_data.shape}")
print(f"Fraud rate: {fraud_data['is_fraud'].mean():.4%}")

# TODO: Build feature matrix from V-features + amount; target = is_fraud.
feature_cols = ____
X = ____
y = ____

# TODO: Stratified 80/20 split (random_state=42).
X_train, X_test, y_train, y_test = ____

# TODO: Train baseline GradientBoostingClassifier (n_estimators=100, max_depth=4).
baseline_model = ____
____

baseline_acc = ____
baseline_auc = ____
print(f"\nBaseline model: accuracy={baseline_acc:.4f}, AUC={baseline_auc:.4f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Implement FGSM on tabular fraud model
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print(f"=== FGSM (Fast Gradient Sign Method) ===")
print(f"{'=' * 70}")


def compute_numerical_gradient(
    model: GradientBoostingClassifier,
    X_sample: np.ndarray,
    y_sample: np.ndarray,
    delta: float = 1e-5,
) -> np.ndarray:
    """Approximate gradient of NLL loss w.r.t. inputs via finite differences."""
    # TODO: For each feature column, build X_plus and X_minus perturbed by +/- delta.
    # TODO: Get predict_proba for both, compute NLL of true class (clip to 1e-10).
    # TODO: Gradient[j, i] = (loss_plus - loss_minus) / (2 * delta).
    ____


def fgsm_attack(
    model: GradientBoostingClassifier,
    X_samples: np.ndarray,
    y_samples: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """FGSM: x_adv = x + epsilon * sign(grad_x L(x, y))."""
    # TODO: Use compute_numerical_gradient + epsilon * sign() perturbation.
    ____


def attack_success_rate(
    model: GradientBoostingClassifier,
    X_clean: np.ndarray,
    X_adv: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """ASR: fraction of correctly-classified samples that flip after attack."""
    # TODO: Restrict to samples that were correct before attack, then count flips.
    ____


# TODO: Use a 500-sample subset of the test set for the attack experiments.
n_attack = 500
X_attack = ____
y_attack = ____

print(f"\nFGSM attack on {n_attack} test samples:")
print(f"{'Epsilon':<12} {'ASR':>8} {'Adv Accuracy':>14} {'Clean Accuracy':>16}")
print("-" * 55)

# TODO: Sweep epsilons in [0.01, 0.05, 0.1, 0.3, 0.5]; record ASR + adv accuracy.
epsilons = [0.01, 0.05, 0.1, 0.3, 0.5]
fgsm_results = []
____


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Implement PGD (iterative FGSM with projection)
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print(f"=== PGD (Projected Gradient Descent) ===")
print(f"{'=' * 70}")


def pgd_attack(
    model: GradientBoostingClassifier,
    X_samples: np.ndarray,
    y_samples: np.ndarray,
    epsilon: float,
    step_size: float,
    n_steps: int,
) -> np.ndarray:
    """PGD: iterative FGSM with projection back into epsilon-ball."""
    # TODO: Random init within +/- epsilon of X_samples.
    # TODO: For n_steps: gradient sign step, then clip perturbation to [-eps, eps].
    ____


print(f"\nPGD attack (10 steps, step_size=epsilon/4):")
print(f"{'Epsilon':<12} {'PGD ASR':>10} {'FGSM ASR':>10} {'Difference':>12}")
print("-" * 50)

# TODO: For each epsilon, run pgd_attack(step_size=eps/4, n_steps=10),
# TODO: compare PGD ASR vs FGSM ASR, append to pgd_results, print row.
pgd_results = []
____

print(f"\nPGD is strictly stronger than FGSM (iterative refinement).")
print(f"PGD-robust models are also FGSM-robust (but not vice versa).")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Adversarial training — augment with FGSM/PGD examples
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print(f"=== Adversarial Training ===")
print(f"{'=' * 70}")

# TODO: Generate FGSM examples from a 2000-sample subset of train data
# TODO: at adv_epsilon=0.1.
adv_epsilon = 0.1
X_train_subset = ____
y_train_subset = ____

X_adv_train = ____

# TODO: Augment training set: vstack X_train + X_adv_train; concat labels.
X_augmented = ____
y_augmented = ____

# TODO: Train robust_model on the augmented set.
robust_model = ____
____

robust_clean_acc = ____
robust_clean_auc = ____

print(f"\nAdversarial training (augmented with FGSM at eps={adv_epsilon}):")
print(f"  Clean accuracy:  {robust_clean_acc:.4f} (baseline: {baseline_acc:.4f})")
print(f"  Clean AUC:       {robust_clean_auc:.4f} (baseline: {baseline_auc:.4f})")

print(f"\nRobust model vs attacks:")
print(f"{'Epsilon':<12} {'Baseline ASR':>14} {'Robust ASR':>12} {'Reduction':>12}")
print("-" * 55)

# TODO: For each epsilon, FGSM-attack baseline + robust models and compute ASR.
____

print(f"\nAdversarial training reduces ASR at moderate epsilon values.")
print(f"Trade-off: slight clean accuracy decrease for improved robustness.")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Data poisoning attack + influence function detection
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print(f"=== Data Poisoning Attack & Detection ===")
print(f"{'=' * 70}")

# TODO: Flip labels on a random 5% of training data.
rng = np.random.default_rng(42)
n_poison = ____
poison_indices = ____

X_poisoned = ____
y_poisoned = ____
____

# TODO: Train a fresh GBM on the poisoned data and report accuracy delta.
poisoned_model = ____
____

poisoned_acc = ____
poisoned_auc = ____

print(
    f"\nPoisoning: flipped {n_poison} labels ({n_poison/len(y_train):.1%} of training data)"
)
print(f"  Clean model accuracy:    {baseline_acc:.4f}")
print(f"  Poisoned model accuracy: {poisoned_acc:.4f}")
print(f"  Accuracy drop:           {baseline_acc - poisoned_acc:.4f}")


def estimate_influence(
    model: GradientBoostingClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    candidate_indices: np.ndarray,
) -> np.ndarray:
    """Approximate influence: low confidence on own label = suspicious."""
    # TODO: For each candidate idx, compute the model's predicted probability of
    # TODO: y_train[idx]. Influence = -log(clip(confidence, 1e-10, 1.0)).
    ____


print(f"\nInfluence-based poison detection:")
candidate_indices = np.arange(min(2000, len(y_poisoned)))
# TODO: Compute influences and flag the top 10% as suspicious.
influences = ____

threshold = ____
flagged = ____

# TODO: Compute precision and recall vs the actual poison_indices set.
poison_set = ____
flagged_set = ____

true_positives = ____
precision = ____
recall = ____

print(f"  Flagged as suspicious: {len(flagged)} samples")
print(f"  True positives: {true_positives}")
print(f"  Detection precision: {precision:.4f}")
print(f"  Detection recall: {recall:.4f}")
print(f"  (Random baseline precision: {len(poison_set)/len(candidate_indices):.4f})")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Certified robustness via randomised smoothing
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print(f"=== Certified Robustness (Randomised Smoothing) ===")
print(f"{'=' * 70}")


def randomised_smoothing_predict(
    model: GradientBoostingClassifier,
    x: np.ndarray,
    sigma: float,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[int, float]:
    """Predict with randomised smoothing — majority vote over noisy samples."""
    # TODO: Sample n_samples Gaussian perturbations of x at scale sigma.
    # TODO: Predict, take majority class, compute confidence = max_count / n_samples.
    ____


def certify_radius(
    confidence: float,
    sigma: float,
    delta: float = 0.05,
) -> float:
    """Certified L2 radius: R = sigma * Phi^{-1}(p_lower)."""
    # TODO: Use scipy.stats.norm.ppf on a confidence lower bound.
    # TODO: Return 0.0 if confidence <= 0.5 or p_lower <= 0.5.
    ____


print(f"\nRandomised smoothing: add Gaussian noise, take majority vote.")
print(f"Provides a CERTIFIED radius R within which predictions cannot change.\n")

sigma_values = [0.1, 0.25, 0.5, 1.0]
n_certify = 50
n_noise_samples = 200

rng = np.random.default_rng(42)

print(f"{'Sigma':<8} {'Avg Radius':>12} {'Certified %':>13} {'Smooth Acc':>12}")
print("-" * 50)

# TODO: For each sigma, certify n_certify test points: collect radii,
# TODO: certified percentage (radius > 0), and smoothed accuracy.
____

print(f"\nInterpretation:")
print(f"  Higher sigma -> larger certified radius (stronger guarantee)")
print(f"  Higher sigma -> lower accuracy (more noise = less precise)")
print(f"  Trade-off: robustness guarantee vs prediction quality")
print(
    f"\nRandomised smoothing is the ONLY method with formal L2 robustness certificates."
)
print(
    f"Unlike adversarial training, the guarantee holds against ANY attack within radius R."
)

print(f"\n=== Adversarial Robustness Summary ===")
print(f"  FGSM:     Single-step, fast, weaker attack")
print(f"  PGD:      Multi-step, stronger attack, standard benchmark")
print(f"  Adv Training: Empirical defence, no formal guarantee")
print(f"  Rand Smoothing: Certified defence, formal L2 guarantee")
print(f"  Poisoning Detection: Influence functions flag suspicious training data")

print(
    "\n--- Exercise 5 complete: adversarial attacks, defences, certified robustness ---"
)
