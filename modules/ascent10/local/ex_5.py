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

feature_cols = [c for c in fraud_data.columns if c.startswith("v")]
X = fraud_data.select(feature_cols + ["amount"]).to_numpy()
y = fraud_data["is_fraud"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TODO: Train baseline GradientBoostingClassifier (n_estimators=100, max_depth=4),
#        compute baseline accuracy and AUC, and print results
# Hint: baseline_model = GradientBoostingClassifier(...); baseline_model.fit(X_train, y_train)
#        baseline_acc = accuracy_score(...); baseline_auc = roc_auc_score(...)
____
____
____
____
____


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Implement FGSM on tabular fraud model
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print(f"=== FGSM (Fast Gradient Sign Method) ===")
print(f"{'=' * 70}")


# TODO: Implement compute_numerical_gradient
# Hint: for each feature i: X_plus[:,i] += delta, X_minus[:,i] -= delta
#        compute loss = -log(p(true_class)), gradient = (loss_plus - loss_minus) / (2*delta)
def compute_numerical_gradient(
    model: GradientBoostingClassifier,
    X_sample: np.ndarray,
    y_sample: np.ndarray,
    delta: float = 1e-5,
) -> np.ndarray:
    """Compute gradient of loss w.r.t. input features using finite differences."""
    ____
    ____
    ____
    ____
    ____


# TODO: Implement fgsm_attack
# Hint: gradients = compute_numerical_gradient(model, X_samples, y_samples)
#        return X_samples + epsilon * np.sign(gradients)
def fgsm_attack(
    model: GradientBoostingClassifier,
    X_samples: np.ndarray,
    y_samples: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """FGSM: x_adv = x + epsilon * sign(grad_x L(x, y))"""
    ____
    ____


# TODO: Implement attack_success_rate
# Hint: correct_mask = (clean_preds == y_true); flipped = adv_preds[correct_mask] != y_true[correct_mask]
#        return flipped / correct_mask.sum()
def attack_success_rate(
    model: GradientBoostingClassifier,
    X_clean: np.ndarray,
    X_adv: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """ASR: fraction of correctly-classified samples that flip after attack."""
    ____
    ____
    ____
    ____
    ____
    ____


n_attack = 500
X_attack = X_test[:n_attack]
y_attack = y_test[:n_attack]

print(f"\nFGSM attack on {n_attack} test samples:")
print(f"{'Epsilon':<12} {'ASR':>8} {'Adv Accuracy':>14} {'Clean Accuracy':>16}")
print("-" * 55)

epsilons = [0.01, 0.05, 0.1, 0.3, 0.5]
# TODO: For each epsilon, run fgsm_attack, compute asr and adv_acc, collect in fgsm_results
fgsm_results = []
____
____
____
____
____


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Implement PGD (iterative FGSM with projection)
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print(f"=== PGD (Projected Gradient Descent) ===")
print(f"{'=' * 70}")


# TODO: Implement pgd_attack
# Hint: 1) random init within epsilon-ball: X_adv = X_samples + rng.uniform(-eps, eps, shape)
#        2) for n_steps: grad = compute_numerical_gradient; X_adv += step_size * sign(grad)
#        3) project back: perturbation = np.clip(X_adv - X_samples, -epsilon, epsilon)
def pgd_attack(
    model: GradientBoostingClassifier,
    X_samples: np.ndarray,
    y_samples: np.ndarray,
    epsilon: float,
    step_size: float,
    n_steps: int,
) -> np.ndarray:
    """PGD: iterative FGSM with projection back into epsilon-ball."""
    ____
    ____
    ____
    ____
    ____


print(f"\nPGD attack (10 steps, step_size=epsilon/4):")
print(f"{'Epsilon':<12} {'PGD ASR':>10} {'FGSM ASR':>10} {'Difference':>12}")
print("-" * 50)

# TODO: For each eps in epsilons, run pgd_attack (step_size=eps/4, n_steps=10),
#        compute pgd_asr and fgsm_asr from fgsm_results[i], print comparison row
pgd_results = []
____
____
____
____
____
____

print(f"\nPGD is strictly stronger than FGSM (iterative refinement).")
print(f"PGD-robust models are also FGSM-robust (but not vice versa).")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Adversarial training — augment with FGSM/PGD examples
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print(f"=== Adversarial Training ===")
print(f"{'=' * 70}")

adv_epsilon = 0.1
X_train_subset = X_train[:2000]
y_train_subset = y_train[:2000]

# TODO: Generate adversarial examples from training data, augment, train robust model
# Hint: X_adv_train = fgsm_attack(baseline_model, X_train_subset, y_train_subset, adv_epsilon)
#        X_augmented = np.vstack([X_train, X_adv_train])
#        y_augmented = np.concatenate([y_train, y_train_subset])
X_adv_train = ____
X_augmented = ____
y_augmented = ____

# TODO: Train robust_model on X_augmented/y_augmented, evaluate clean accuracy + AUC
# Then loop over epsilons: for each, compute baseline_asr (fgsm on baseline),
# robust_asr (fgsm on robust_model), print reduction table
____
____
____
____
____

print(f"\nAdversarial training reduces ASR at moderate epsilon values.")
print(f"Trade-off: slight clean accuracy decrease for improved robustness.")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Data poisoning attack + influence function detection
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print(f"=== Data Poisoning Attack & Detection ===")
print(f"{'=' * 70}")

# TODO: Inject poison: flip 5% of y_train labels, train poisoned_model, compute poisoned accuracy
# Hint: poison_indices = rng.choice(len(y_train), n_poison, replace=False)
#        y_poisoned[poison_indices] = 1 - y_poisoned[poison_indices]
____
____
____
____
____


# TODO: Implement estimate_influence
# Hint: for each candidate, compute model.predict_proba on that training point;
#        influence = -log(confidence for true class) — low confidence = suspicious
def estimate_influence(
    model: GradientBoostingClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    candidate_indices: np.ndarray,
) -> np.ndarray:
    """Approximate influence function: flag mislabelled training points."""
    ____
    ____
    ____
    ____
    ____


print(f"\nInfluence-based poison detection:")
# TODO: Run estimate_influence, flag top 10% (percentile 90), compute precision and recall
# Hint: threshold = np.percentile(influences, 90); flagged = candidate_indices[influences > threshold]
#        precision = true_positives / len(flagged_set); recall = true_positives / len(poison_set)
____
____
____
____
____


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Certified robustness via randomised smoothing
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print(f"=== Certified Robustness (Randomised Smoothing) ===")
print(f"{'=' * 70}")


# TODO: Implement randomised_smoothing_predict
# Hint: noise = rng.normal(0, sigma, (n_samples, x.shape[0])); X_noisy = x.reshape(1,-1) + noise
#        majority_class = classes[argmax(counts)]; confidence = counts.max() / n_samples
def randomised_smoothing_predict(
    model: GradientBoostingClassifier,
    x: np.ndarray,
    sigma: float,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[int, float]:
    """Predict with randomised smoothing. Returns (predicted_class, confidence)."""
    ____
    ____
    ____
    ____
    ____
    ____
    ____


# TODO: Implement certify_radius
# Hint: from scipy.stats import norm; if confidence <= 0.5 return 0.0
#        p_lower = confidence - delta; radius = sigma * norm.ppf(p_lower)
def certify_radius(
    confidence: float,
    sigma: float,
    delta: float = 0.05,
) -> float:
    """Compute certified L2 radius for randomised smoothing."""
    ____
    ____
    ____
    ____
    ____
    ____


print(f"\nRandomised smoothing: add Gaussian noise, take majority vote.")
print(f"Provides a CERTIFIED radius R within which predictions cannot change.\n")

sigma_values = [0.1, 0.25, 0.5, 1.0]
n_certify = 50
n_noise_samples = 200

rng = np.random.default_rng(42)

print(f"{'Sigma':<8} {'Avg Radius':>12} {'Certified %':>13} {'Smooth Acc':>12}")
print("-" * 50)

# TODO: For each sigma in sigma_values, run randomised_smoothing_predict on n_certify samples,
#        compute certify_radius per sample, collect avg_radius, certified_pct (radius > 0),
#        smooth_acc (correct / n_certify), and print a row per sigma
# Hint: randomised_smoothing_predict(model, X_test[i], sigma, n_samples, rng) -> (pred_class, conf)
#        certify_radius(conf, sigma) -> float radius; print row with sigma, avg_radius, certified_pct, smooth_acc
____
____
____
____
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
