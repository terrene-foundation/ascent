# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT10 — Exercise 3: Privacy-Preserving ML and Data Minimisation
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Implement k-anonymity, membership inference attacks and
#   defences, synthetic data generation, and PACT-governed inference
#   with clearance-based access control.
#
# TASKS:
#   1. Implement k-anonymity on ICU patients — measure utility at k=5,10,50
#   2. Membership inference attack via shadow model
#   3. Defend with L2 regularisation + early stopping, measure AUC reduction
#   4. Generate synthetic data using Gaussian copula, verify properties
#   5. PACT-governed inference with ClearanceManager checking permissions
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from kailash.trust import ConfidentialityLevel, TrustPosture
from pact import GovernanceEngine, PactGovernedAgent, compile_org, load_org_yaml
from pact.governance import RoleClearance, RoleEnvelope, KnowledgeItem
from kailash.trust.pact.config import (
    ConstraintEnvelopeConfig,
    OperationalConstraintConfig,
    TemporalConstraintConfig,
    DataAccessConstraintConfig,
    CommunicationConstraintConfig,
)

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Implement k-anonymity — generalise quasi-identifiers
# ══════════════════════════════════════════════════════════════════════

loader = ASCENTDataLoader()
patients = loader.load("ascent02", "icu_patients.parquet")

print("=== ICU Patient Dataset ===")
print(f"Shape: {patients.shape}")
print(f"Columns: {patients.columns}")
print(patients.head(5))


def generalise_age(age: int, bin_size: int) -> str:
    """Generalise age into bins (e.g., 25 -> '20-29' for bin_size=10)."""
    # TODO: Compute lower = (age // bin_size) * bin_size, upper = lower + bin_size - 1.
    ____


def apply_k_anonymity(
    df: pl.DataFrame,
    quasi_identifiers: list[str],
    age_bin_size: int,
    k: int,
) -> tuple[pl.DataFrame, dict]:
    """Apply k-anonymity by generalising quasi-identifiers."""
    # TODO: Drop direct identifier patient_id.
    # TODO: Generalise age via map_elements -> age_group, drop age.
    # TODO: Group by quasi_identifiers and count records per equivalence class.
    # TODO: Suppress (anti-join) records in classes with count < k.
    # TODO: Return the suppressed DataFrame and a stats dict.
    ____


print(f"\n=== k-Anonymity Analysis ===")
print(f"Quasi-identifiers: age, gender")
print(f"Direct identifier removed: patient_id\n")

# TODO: Sweep k in [5, 10, 50] with bin_size = max(5, k // 2);
# TODO: print rows kept, suppression rate, equivalence class count.
k_values = [5, 10, 50]
____

print(f"\nTrade-off: higher k = stronger privacy, but more data loss (suppression).")
print(f"k=5 is the Singapore PDPA recommended minimum for de-identification.")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Membership inference attack via shadow model
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Membership Inference Attack ===")
print(f"Goal: determine whether a specific record was in the training set.")
print(
    f"Method: train shadow model, use confidence scores to train attack classifier.\n"
)

# TODO: Build a synthetic feature matrix and binary label
# TODO: (label = age > median age). Use n = patients.height rows.
rng = np.random.default_rng(42)
n = patients.height
features = ____
median_age = ____
labels = ____

# TODO: Split data 50/50 into target_data vs shadow_data via train_test_split.
X_target, X_shadow, y_target, y_shadow = ____

# TODO: Split target_data 50/50 again into train and holdout.
X_train, X_holdout, y_train, y_holdout = ____

# TODO: Train target_model = RandomForestClassifier on (X_train, y_train).
target_model = ____
____

# TODO: Split shadow_data 50/50 into shadow_train + shadow_test.
X_shadow_train, X_shadow_test, y_shadow_train, y_shadow_test = ____
# TODO: Train shadow_model on (X_shadow_train, y_shadow_train).
shadow_model = ____
____

# TODO: Build attack training set:
# TODO:   members = predict_proba(shadow_train) (label=1)
# TODO:   non-members = predict_proba(shadow_test) (label=0)
shadow_train_probs = ____
shadow_test_probs = ____

attack_X = ____
attack_y = ____

# TODO: Fit attack_model = LogisticRegression(random_state=42) on (attack_X, attack_y).
attack_model = ____
____

# TODO: Apply the attack to the target model: predict_proba(X_train) vs
# TODO: predict_proba(X_holdout); evaluate attack accuracy + AUC.
target_train_probs = ____
target_holdout_probs = ____

eval_X = ____
eval_y = ____

attack_preds = ____
attack_probs = ____

attack_acc = ____
attack_auc = ____

print(
    f"Target model accuracy: {accuracy_score(y_holdout, target_model.predict(X_holdout)):.4f}"
)
print(f"Membership inference attack:")
print(f"  Attack accuracy: {attack_acc:.4f} (random baseline = 0.5)")
print(f"  Attack AUC: {attack_auc:.4f}")
print(f"  {'Vulnerable!' if attack_auc > 0.6 else 'Reasonably private'}")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Defend with L2 regularisation + early stopping
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Defence: L2 Regularisation + Early Stopping ===")

# TODO: Build a defended model: RandomForestClassifier with fewer trees,
# TODO: shallow depth, larger min_samples_leaf to fight overfitting.
defended_model = ____
____

# TODO: Re-run the shadow attack on the defended model and report
# TODO: the accuracy / AUC reduction vs the original attack.
defended_train_probs = ____
defended_holdout_probs = ____

defended_eval_X = ____
defended_attack_probs = ____
defended_attack_preds = ____

defended_acc = ____
defended_auc = ____

model_acc = ____

print(f"Defended model accuracy: {model_acc:.4f}")
print(f"Attack on defended model:")
print(f"  Attack accuracy: {defended_acc:.4f} (was {attack_acc:.4f})")
print(f"  Attack AUC: {defended_auc:.4f} (was {attack_auc:.4f})")
print(f"  AUC reduction: {attack_auc - defended_auc:.4f}")
print(f"\nRegularisation reduces overfitting, which reduces the gap between")
print(f"member and non-member confidence distributions -- making MI harder.")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Synthetic data via Gaussian copula
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Synthetic Data Generation (Gaussian Copula) ===")


def fit_gaussian_copula(
    data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a Gaussian copula by estimating marginal CDFs and correlation."""
    # TODO: Step 1 - rank-normalise each column to uniform marginals (clip to [0.001, 0.999]).
    # TODO: Step 2 - probit transform via stats.norm.ppf -> standard normals.
    # TODO: Step 3 - compute corr_matrix via np.corrcoef on the normals.
    # TODO: Also compute marginal_means and marginal_stds for inverse transform.
    ____


def sample_gaussian_copula(
    n_samples: int,
    corr_matrix: np.ndarray,
    marginal_means: np.ndarray,
    marginal_stds: np.ndarray,
    generator: np.random.Generator,
) -> np.ndarray:
    """Generate synthetic data from fitted Gaussian copula."""
    # TODO: Sample from multivariate_normal with the corr matrix.
    # TODO: Rescale by marginal_stds + marginal_means.
    ____


# TODO: Fit copula on first 200 rows of features; sample 200 synthetic rows.
real_data = ____
corr_mat, means, stds = ____

synthetic_data = ____

print(f"Real data shape: {real_data.shape}")
print(f"Synthetic data shape: {synthetic_data.shape}")

print(f"\nMean comparison (real vs synthetic):")
# TODO: Print real vs synthetic mean for each of the first 6 features.
____

print(f"\nStd comparison (real vs synthetic):")
# TODO: Print real vs synthetic std for each of the first 6 features.
____

# TODO: Compare correlation matrices via mean absolute difference.
real_corr = ____
synth_corr = ____
corr_diff = ____
print(f"\nMean absolute correlation difference: {corr_diff:.4f}")
print(f"Correlation structure {'preserved' if corr_diff < 0.1 else 'divergent'}")

print(f"\nSynthetic data provides inherent privacy: no real individual's record")
print(f"appears in the synthetic dataset (generated from learned distribution).")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: PACT-governed inference with clearance checking
# ══════════════════════════════════════════════════════════════════════

# TODO: Define a YAML org with service_admin, researcher, external_user,
# TODO: inference_agent (agent: true), and dpo. Grant clearances at
# TODO: secret/confidential/restricted/top_secret on patient_data,
# TODO: model_outputs, audit_logs, anonymised_data compartments.
privacy_org_yaml = ____

# TODO: Write to temp file, load, compile, build GovernanceEngine.
priv_file = ____
____
priv_loaded = ____
priv_compiled = ____
priv_engine = ____

# TODO: Build priv_roles (role_id -> address) for non-vacant nodes.
priv_roles = ____
____

level_map = ____

# TODO: For each yaml clearance, build RoleClearance + grant on the engine.
____

# TODO: Build a RoleEnvelope for inference_agent restricted to predict,
# TODO: get_model_info, log_request. Apply with priv_engine.set_role_envelope.
inf_env = ____
____

# TODO: Build 3 KnowledgeItems at SECRET / CONFIDENTIAL / RESTRICTED levels
# TODO: with patient_data / model_outputs / anonymised_data compartments.
inference_items = ____

print(f"\n=== PACT-Governed Inference ===")
print(f"Access control matrix:\n")

requesters = ["researcher", "external_user", "inference_agent", "dpo"]
# TODO: For each item and each requester role, run check_access with
# TODO: TrustPosture.SUPERVISED and print ALLOW/DENY rows.
____

# TODO: Wrap inference_agent in a PactGovernedAgent and print its
# TODO: clearance, allowed_actions and compartments.
governed_inf = ____

print(f"Governed inference agent:")
print(f"  Clearance: {governed_inf.context.effective_clearance_level}")
print(f"  Actions: {sorted(governed_inf.context.allowed_actions)}")
print(f"  Compartments: {sorted(governed_inf.context.compartments)}")

# TODO: For each action in [predict, get_model_info, export_raw_data, modify_model],
# TODO: call priv_engine.verify_action and print ALLOW/BLOCK.
____

# TODO: Verify audit integrity and clean up the temp YAML file.
integrity = ____
print(f"\nAudit integrity: {'VALID' if integrity else 'BROKEN'}")

____

print(
    "\n--- Exercise 3 complete: privacy-preserving ML with k-anonymity, MI defence, PACT ---"
)
