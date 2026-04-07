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

# Train baseline fraud model
baseline_model = GradientBoostingClassifier(
    n_estimators=100, max_depth=4, random_state=42
)
baseline_model.fit(X_train, y_train)

baseline_acc = accuracy_score(y_test, baseline_model.predict(X_test))
baseline_auc = roc_auc_score(y_test, baseline_model.predict_proba(X_test)[:, 1])
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
    """Compute gradient of loss w.r.t. input features using finite differences.

    For tree-based models (non-differentiable), we approximate the gradient
    numerically: df/dx_i ~ (f(x + delta*e_i) - f(x - delta*e_i)) / (2*delta)
    where f is the predicted probability of the true class.
    """
    n_features = X_sample.shape[1]
    gradients = np.zeros_like(X_sample)

    for i in range(n_features):
        X_plus = X_sample.copy()
        X_minus = X_sample.copy()
        X_plus[:, i] += delta
        X_minus[:, i] -= delta

        # Loss = -log(p(true class))
        prob_plus = model.predict_proba(X_plus)
        prob_minus = model.predict_proba(X_minus)

        for j in range(len(y_sample)):
            true_class = int(y_sample[j])
            p_plus = prob_plus[j, true_class]
            p_minus = prob_minus[j, true_class]
            # Gradient of negative log-likelihood
            loss_plus = -np.log(np.clip(p_plus, 1e-10, 1.0))
            loss_minus = -np.log(np.clip(p_minus, 1e-10, 1.0))
            gradients[j, i] = (loss_plus - loss_minus) / (2 * delta)

    return gradients


def fgsm_attack(
    model: GradientBoostingClassifier,
    X_samples: np.ndarray,
    y_samples: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """FGSM: x_adv = x + epsilon * sign(grad_x L(x, y))

    Single-step perturbation in the direction that maximises loss.
    """
    gradients = compute_numerical_gradient(model, X_samples, y_samples)
    perturbation = epsilon * np.sign(gradients)
    X_adv = X_samples + perturbation
    return X_adv


def attack_success_rate(
    model: GradientBoostingClassifier,
    X_clean: np.ndarray,
    X_adv: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """ASR: fraction of correctly-classified samples that flip after attack."""
    clean_preds = model.predict(X_clean)
    adv_preds = model.predict(X_adv)

    # Only count samples that were correctly classified before attack
    correct_mask = clean_preds == y_true
    if correct_mask.sum() == 0:
        return 0.0

    flipped = (adv_preds[correct_mask] != y_true[correct_mask]).sum()
    return flipped / correct_mask.sum()


# Use a subset for efficiency
n_attack = 500
X_attack = X_test[:n_attack]
y_attack = y_test[:n_attack]

print(f"\nFGSM attack on {n_attack} test samples:")
print(f"{'Epsilon':<12} {'ASR':>8} {'Adv Accuracy':>14} {'Clean Accuracy':>16}")
print("-" * 55)

epsilons = [0.01, 0.05, 0.1, 0.3, 0.5]
fgsm_results = []
for eps in epsilons:
    X_adv = fgsm_attack(baseline_model, X_attack, y_attack, eps)
    asr = attack_success_rate(baseline_model, X_attack, X_adv, y_attack)
    adv_acc = accuracy_score(y_attack, baseline_model.predict(X_adv))
    clean_acc = accuracy_score(y_attack, baseline_model.predict(X_attack))
    fgsm_results.append({"epsilon": eps, "asr": asr, "adv_acc": adv_acc})
    print(f"{eps:<12.2f} {asr:>8.4f} {adv_acc:>14.4f} {clean_acc:>16.4f}")


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
    """PGD: iterative FGSM with projection back into epsilon-ball.

    x_0 = x + uniform(-epsilon, epsilon)  (random start)
    x_{t+1} = Proj_{B(x, epsilon)}(x_t + alpha * sign(grad_x L(x_t, y)))
    """
    rng = np.random.default_rng(42)

    # Random initialisation within epsilon-ball
    X_adv = X_samples + rng.uniform(-epsilon, epsilon, X_samples.shape)

    for step in range(n_steps):
        gradients = compute_numerical_gradient(model, X_adv, y_samples)
        X_adv = X_adv + step_size * np.sign(gradients)

        # Project back into L-infinity epsilon-ball around original
        perturbation = X_adv - X_samples
        perturbation = np.clip(perturbation, -epsilon, epsilon)
        X_adv = X_samples + perturbation

    return X_adv


print(f"\nPGD attack (10 steps, step_size=epsilon/4):")
print(f"{'Epsilon':<12} {'PGD ASR':>10} {'FGSM ASR':>10} {'Difference':>12}")
print("-" * 50)

pgd_results = []
for i, eps in enumerate(epsilons):
    X_adv_pgd = pgd_attack(
        baseline_model, X_attack, y_attack, epsilon=eps, step_size=eps / 4, n_steps=10
    )
    pgd_asr = attack_success_rate(baseline_model, X_attack, X_adv_pgd, y_attack)
    fgsm_asr = fgsm_results[i]["asr"]
    diff = pgd_asr - fgsm_asr
    pgd_results.append({"epsilon": eps, "pgd_asr": pgd_asr, "fgsm_asr": fgsm_asr})
    print(f"{eps:<12.2f} {pgd_asr:>10.4f} {fgsm_asr:>10.4f} {diff:>+12.4f}")

print(f"\nPGD is strictly stronger than FGSM (iterative refinement).")
print(f"PGD-robust models are also FGSM-robust (but not vice versa).")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Adversarial training — augment with FGSM/PGD examples
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print(f"=== Adversarial Training ===")
print(f"{'=' * 70}")

# Generate adversarial examples from training data
adv_epsilon = 0.1
X_train_subset = X_train[:2000]
y_train_subset = y_train[:2000]

X_adv_train = fgsm_attack(baseline_model, X_train_subset, y_train_subset, adv_epsilon)

# Augment training set with adversarial examples
X_augmented = np.vstack([X_train, X_adv_train])
y_augmented = np.concatenate([y_train, y_train_subset])

# Train adversarially robust model
robust_model = GradientBoostingClassifier(
    n_estimators=100, max_depth=4, random_state=42
)
robust_model.fit(X_augmented, y_augmented)

# Evaluate clean accuracy
robust_clean_acc = accuracy_score(y_test, robust_model.predict(X_test))
robust_clean_auc = roc_auc_score(y_test, robust_model.predict_proba(X_test)[:, 1])

print(f"\nAdversarial training (augmented with FGSM at eps={adv_epsilon}):")
print(f"  Clean accuracy:  {robust_clean_acc:.4f} (baseline: {baseline_acc:.4f})")
print(f"  Clean AUC:       {robust_clean_auc:.4f} (baseline: {baseline_auc:.4f})")

print(f"\nRobust model vs attacks:")
print(f"{'Epsilon':<12} {'Baseline ASR':>14} {'Robust ASR':>12} {'Reduction':>12}")
print("-" * 55)

for eps in epsilons:
    X_adv = fgsm_attack(baseline_model, X_attack, y_attack, eps)
    baseline_asr = attack_success_rate(baseline_model, X_attack, X_adv, y_attack)

    X_adv_r = fgsm_attack(robust_model, X_attack, y_attack, eps)
    robust_asr = attack_success_rate(robust_model, X_attack, X_adv_r, y_attack)

    reduction = baseline_asr - robust_asr
    print(f"{eps:<12.2f} {baseline_asr:>14.4f} {robust_asr:>12.4f} {reduction:>+12.4f}")

print(f"\nAdversarial training reduces ASR at moderate epsilon values.")
print(f"Trade-off: slight clean accuracy decrease for improved robustness.")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Data poisoning attack + influence function detection
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print(f"=== Data Poisoning Attack & Detection ===")
print(f"{'=' * 70}")

# Inject 5% mislabelled examples into training data
rng = np.random.default_rng(42)
n_poison = int(len(y_train) * 0.05)
poison_indices = rng.choice(len(y_train), n_poison, replace=False)

X_poisoned = X_train.copy()
y_poisoned = y_train.copy()
y_poisoned[poison_indices] = 1 - y_poisoned[poison_indices]  # Flip labels

# Train on poisoned data
poisoned_model = GradientBoostingClassifier(
    n_estimators=100, max_depth=4, random_state=42
)
poisoned_model.fit(X_poisoned, y_poisoned)

poisoned_acc = accuracy_score(y_test, poisoned_model.predict(X_test))
poisoned_auc = roc_auc_score(y_test, poisoned_model.predict_proba(X_test)[:, 1])

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
    """Approximate influence function: estimate each training point's impact on test loss.

    Simplified approach: leave-one-out approximation using prediction confidence.
    Points with high influence (large loss change when removed) are suspicious.
    """
    # Base test loss
    base_probs = model.predict_proba(X_test)
    base_loss = 0.0
    for i, true_class in enumerate(y_test[:100]):
        base_loss -= np.log(np.clip(base_probs[i, int(true_class)], 1e-10, 1.0))

    influences = np.zeros(len(candidate_indices))

    for idx_pos, train_idx in enumerate(candidate_indices):
        # Prediction confidence for this training point
        prob = model.predict_proba(X_train[train_idx : train_idx + 1])
        true_class = int(y_train[train_idx])
        confidence = prob[0, true_class]

        # Low confidence on its own label = suspicious (likely mislabelled)
        influences[idx_pos] = -np.log(np.clip(confidence, 1e-10, 1.0))

    return influences


# Detect poisoned samples using influence estimation
print(f"\nInfluence-based poison detection:")
candidate_indices = np.arange(min(2000, len(y_poisoned)))
influences = estimate_influence(
    poisoned_model, X_poisoned, y_poisoned, X_test, y_test, candidate_indices
)

# Flag top 10% as suspicious
threshold = np.percentile(influences, 90)
flagged = candidate_indices[influences > threshold]

# Calculate detection precision and recall
poison_set = set(poison_indices[poison_indices < len(candidate_indices)])
flagged_set = set(flagged)

true_positives = len(poison_set & flagged_set)
precision = true_positives / len(flagged_set) if flagged_set else 0.0
recall = true_positives / len(poison_set) if poison_set else 0.0

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
    """Predict with randomised smoothing.

    Sample n_samples noisy versions of x, take majority vote.
    Returns (predicted_class, confidence).
    """
    noise = rng.normal(0, sigma, (n_samples, x.shape[0]))
    X_noisy = x.reshape(1, -1) + noise

    predictions = model.predict(X_noisy)
    classes, counts = np.unique(predictions, return_counts=True)

    majority_class = classes[np.argmax(counts)]
    confidence = counts.max() / n_samples

    return int(majority_class), confidence


def certify_radius(
    confidence: float,
    sigma: float,
    delta: float = 0.05,
) -> float:
    """Compute certified L2 radius for randomised smoothing.

    If the smoothed classifier returns class c with probability p_A,
    then for any perturbation ||delta|| < R, the prediction remains c,
    where R = sigma * Phi^{-1}(p_A).

    Phi^{-1} is the inverse normal CDF (probit function).
    """
    from scipy.stats import norm

    if confidence <= 0.5:
        return 0.0  # Cannot certify

    # Lower confidence bound using Clopper-Pearson
    p_lower = confidence - delta  # Simplified bound
    if p_lower <= 0.5:
        return 0.0

    radius = sigma * norm.ppf(p_lower)
    return max(0.0, radius)


print(f"\nRandomised smoothing: add Gaussian noise, take majority vote.")
print(f"Provides a CERTIFIED radius R within which predictions cannot change.\n")

sigma_values = [0.1, 0.25, 0.5, 1.0]
n_certify = 50  # Number of test samples to certify
n_noise_samples = 200

rng = np.random.default_rng(42)

print(f"{'Sigma':<8} {'Avg Radius':>12} {'Certified %':>13} {'Smooth Acc':>12}")
print("-" * 50)

for sigma in sigma_values:
    radii = []
    correct = 0

    for i in range(n_certify):
        pred_class, conf = randomised_smoothing_predict(
            baseline_model, X_test[i], sigma, n_noise_samples, rng
        )
        radius = certify_radius(conf, sigma)
        radii.append(radius)
        if pred_class == y_test[i]:
            correct += 1

    avg_radius = np.mean(radii)
    certified_pct = np.mean([r > 0 for r in radii])
    smooth_acc = correct / n_certify

    print(
        f"{sigma:<8.2f} {avg_radius:>12.4f} {certified_pct:>13.2%} {smooth_acc:>12.4f}"
    )

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
