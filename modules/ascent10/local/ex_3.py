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


# TODO: Implement generalise_age helper
# Hint: lower = (age // bin_size) * bin_size; upper = lower + bin_size - 1; return f"{lower}-{upper}"
def generalise_age(age: int, bin_size: int) -> str:
    """Generalise age into bins (e.g., 25 -> '20-29' for bin_size=10)."""
    ____
    ____
    ____


# TODO: Implement apply_k_anonymity
# Hint: 1) drop "patient_id", 2) generalise age via map_elements(generalise_age, bin_size)
#        3) group_by QI columns, count equivalence classes, filter violations (<k)
#        4) suppress violating records with anti-join, return (df, stats_dict)
def apply_k_anonymity(
    df: pl.DataFrame,
    quasi_identifiers: list[str],
    age_bin_size: int,
    k: int,
) -> tuple[pl.DataFrame, dict]:
    """Apply k-anonymity by generalising quasi-identifiers."""
    ____
    ____
    ____
    ____
    ____


print(f"\n=== k-Anonymity Analysis ===")
print(f"Quasi-identifiers: age, gender")
print(f"Direct identifier removed: patient_id\n")

# TODO: For k in [5, 10, 50], call apply_k_anonymity with bin_size=max(5, k//2) and print results
# Hint: anon_df, s = apply_k_anonymity(patients, quasi_identifiers=["age","gender"], age_bin_size=..., k=k)
#        print: k, bin_size, s['anonymised_rows'], s['original_rows'], s['suppression_rate'], s['equivalence_classes']
k_values = [5, 10, 50]
____
____
____
____
____
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

rng = np.random.default_rng(42)
n = patients.height
features = rng.normal(0, 1, (n, 6))
median_age = patients["age"].median()
labels = (patients["age"].to_numpy() > median_age).astype(int)

X_target, X_shadow, y_target, y_shadow = train_test_split(
    features, labels, test_size=0.5, random_state=42
)

X_train, X_holdout, y_train, y_holdout = train_test_split(
    X_target, y_target, test_size=0.5, random_state=42
)

# TODO: Implement membership inference attack
# Hint: 1) train target_model (RandomForestClassifier) on X_train
#        2) train shadow_model on X_shadow_train
#        3) build attack dataset: shadow_train_probs (label=1), shadow_test_probs (label=0)
#        4) train attack_model (LogisticRegression) on attack dataset
#        5) evaluate on target train/holdout, compute accuracy_score + roc_auc_score
____
____
____
____
____

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

# TODO: Retrain with strong regularisation (fewer trees, shallow depth, min_samples_leaf)
# Hint: RandomForestClassifier(n_estimators=50, max_depth=3, min_samples_leaf=10, random_state=42)
defended_model = ____
defended_model.fit(X_train, y_train)

# TODO: Re-run membership inference attack on defended_model
# Hint: defended_train_probs = defended_model.predict_proba(X_train)
#        defended_holdout_probs = defended_model.predict_proba(X_holdout)
#        defended_eval_X = np.vstack([...]), run attack_model on it, compute acc + auc
____
____
____
____
____
____
____
____
____

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


# TODO: Implement fit_gaussian_copula
# Hint: 1) empirical CDF via stats.rankdata / (n+1), clip to [0.001, 0.999]
#        2) probit transform: stats.norm.ppf(uniforms)
#        3) np.corrcoef(normals, rowvar=False) for correlation matrix
#        4) return (corr_matrix, marginal_means, marginal_stds)
def fit_gaussian_copula(
    data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a Gaussian copula by estimating marginal CDFs and correlation."""
    ____
    ____
    ____
    ____
    ____


# TODO: Implement sample_gaussian_copula
# Hint: generator.multivariate_normal(zeros, corr_matrix, size=n_samples)
#       then transform: synthetic = normals * marginal_stds + marginal_means
def sample_gaussian_copula(
    n_samples: int,
    corr_matrix: np.ndarray,
    marginal_means: np.ndarray,
    marginal_stds: np.ndarray,
    generator: np.random.Generator,
) -> np.ndarray:
    """Generate synthetic data from fitted Gaussian copula."""
    ____
    ____
    ____
    ____
    ____


real_data = features[:200]
corr_mat, means, stds = fit_gaussian_copula(real_data)
synthetic_data = sample_gaussian_copula(200, corr_mat, means, stds, rng)

print(f"Real data shape: {real_data.shape}")
print(f"Synthetic data shape: {synthetic_data.shape}")

print(f"\nMean comparison (real vs synthetic):")
for i in range(min(6, real_data.shape[1])):
    print(
        f"  Feature {i}: {real_data[:, i].mean():.4f} vs {synthetic_data[:, i].mean():.4f}"
    )

print(f"\nStd comparison (real vs synthetic):")
for i in range(min(6, real_data.shape[1])):
    print(
        f"  Feature {i}: {real_data[:, i].std():.4f} vs {synthetic_data[:, i].std():.4f}"
    )

real_corr = np.corrcoef(real_data, rowvar=False)
synth_corr = np.corrcoef(synthetic_data, rowvar=False)
corr_diff = np.abs(real_corr - synth_corr).mean()
print(f"\nMean absolute correlation difference: {corr_diff:.4f}")
print(f"Correlation structure {'preserved' if corr_diff < 0.1 else 'divergent'}")

print(f"\nSynthetic data provides inherent privacy: no real individual's record")
print(f"appears in the synthetic dataset (generated from learned distribution).")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: PACT-governed inference with clearance checking
# ══════════════════════════════════════════════════════════════════════

privacy_org_yaml = """
org_id: privacy_governed_inference
name: "Privacy-Governed ML Service"

departments:
  - id: ml_service
    name: "ML Inference Service"
  - id: data_governance
    name: "Data Governance"

roles:
  - id: service_admin
    name: "Service Administrator"
    is_primary_for_unit: ml_service

  - id: researcher
    name: "Research Analyst"
    reports_to: service_admin
    agent: false

  - id: external_user
    name: "External API User"
    reports_to: service_admin
    agent: false

  - id: inference_agent
    name: "Inference Agent"
    reports_to: service_admin
    agent: true

  - id: dpo
    name: "Data Protection Officer"
    is_primary_for_unit: data_governance

clearances:
  - role: service_admin
    level: secret
    compartments: [patient_data, model_outputs, audit_logs]
  - role: researcher
    level: confidential
    compartments: [model_outputs, anonymised_data]
  - role: external_user
    level: restricted
    compartments: [model_outputs]
  - role: inference_agent
    level: confidential
    compartments: [patient_data, model_outputs]
  - role: dpo
    level: top_secret
    compartments: [patient_data, model_outputs, audit_logs, anonymised_data]
"""

# TODO: Write privacy_org_yaml to a temp file, load_org_yaml, compile_org,
#        create GovernanceEngine, and build priv_roles dict (node_id -> address)
# Hint: priv_file = Path(tempfile.mktemp(suffix=".yaml")); priv_file.write_text(...)
#        priv_loaded = load_org_yaml(str(priv_file)); priv_compiled = compile_org(priv_loaded.org_definition)
#        priv_engine = GovernanceEngine(priv_loaded.org_definition)
____
____
____
____
____

# TODO: Build level_map (restricted/confidential/secret/top_secret -> ConfidentialityLevel)
#        and grant clearances to all roles in priv_engine
____
____
____
____
____

# TODO: Create RoleEnvelope for inference_agent with allowed_actions=["predict","get_model_info","log_request"]
# Hint: defining_role_address=priv_roles["service_admin"], target=priv_roles["inference_agent"]
inf_env = ____
priv_engine.set_role_envelope(inf_env)

# TODO: Create inference_items list of 3 KnowledgeItems:
#   raw_patient_record (SECRET, patient_data compartment, owned by service_admin)
#   model_prediction_output (CONFIDENTIAL, model_outputs, owned by inference_agent)
#   anonymised_dataset (RESTRICTED, anonymised_data, owned by dpo)
# Hint: KnowledgeItem(item_id=..., classification=ConfidentialityLevel.SECRET,
#         owning_unit_address=priv_roles.get("service_admin", "R1"), compartments=frozenset([...]))
____
____
____
____
____

print(f"\n=== PACT-Governed Inference ===")
print(f"Access control matrix:\n")

# TODO: Print access control matrix: for each inference_item, for each requester,
#        call priv_engine.check_access(priv_roles[req_id], item, TrustPosture.SUPERVISED)
#        and print ALLOW or DENY
# Hint: requesters = ["researcher", "external_user", "inference_agent", "dpo"]
____
____
____
____
____
____

# TODO: Create PactGovernedAgent for inference_agent
# Hint: PactGovernedAgent(engine=priv_engine, role_address=priv_roles["inference_agent"])
governed_inf = ____

print(f"Governed inference agent:")
print(f"  Clearance: {governed_inf.context.effective_clearance_level}")
print(f"  Actions: {sorted(governed_inf.context.allowed_actions)}")
print(f"  Compartments: {sorted(governed_inf.context.compartments)}")

for action in ["predict", "get_model_info", "export_raw_data", "modify_model"]:
    verdict = priv_engine.verify_action(priv_roles["inference_agent"], action)
    status = "ALLOW" if verdict.level == "auto_approved" else "BLOCK"
    print(f"  Action '{action}': {status}")

integrity = priv_engine.verify_audit_integrity()
print(f"\nAudit integrity: {'VALID' if integrity else 'BROKEN'}")

priv_file.unlink()

print(
    "\n--- Exercise 3 complete: privacy-preserving ML with k-anonymity, MI defence, PACT ---"
)
