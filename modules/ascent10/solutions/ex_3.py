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

# Quasi-identifiers: age, gender (these can re-identify individuals)
# Direct identifiers: patient_id (must be removed)


def generalise_age(age: int, bin_size: int) -> str:
    """Generalise age into bins (e.g., 25 -> '20-29' for bin_size=10)."""
    lower = (age // bin_size) * bin_size
    upper = lower + bin_size - 1
    return f"{lower}-{upper}"


def apply_k_anonymity(
    df: pl.DataFrame,
    quasi_identifiers: list[str],
    age_bin_size: int,
    k: int,
) -> tuple[pl.DataFrame, dict]:
    """Apply k-anonymity by generalising quasi-identifiers.

    Returns:
        Tuple of (anonymised DataFrame, statistics dict).
    """
    # Step 1: Remove direct identifiers
    anon_df = df.drop("patient_id")

    # Step 2: Generalise age into bins
    anon_df = anon_df.with_columns(
        pl.col("age")
        .map_elements(lambda a: generalise_age(a, age_bin_size), return_dtype=pl.Utf8)
        .alias("age_group")
    ).drop("age")

    # Step 3: Check k-anonymity — each equivalence class must have >= k records
    qi_cols = [c if c != "age" else "age_group" for c in quasi_identifiers]
    equiv_classes = anon_df.group_by(qi_cols).agg(pl.len().alias("count"))
    violations = equiv_classes.filter(pl.col("count") < k)

    # Step 4: Suppress records in violating equivalence classes
    if violations.height > 0:
        violating_groups = violations.select(qi_cols)
        suppressed = anon_df.join(violating_groups, on=qi_cols, how="anti")
    else:
        suppressed = anon_df

    statistics = {
        "original_rows": df.height,
        "anonymised_rows": suppressed.height,
        "suppressed_rows": df.height - suppressed.height,
        "suppression_rate": (df.height - suppressed.height) / df.height,
        "equivalence_classes": equiv_classes.height,
        "min_class_size": equiv_classes["count"].min(),
        "violations_before_suppression": violations.height,
        "k_achieved": k,
        "age_bin_size": age_bin_size,
    }

    return suppressed, statistics


print(f"\n=== k-Anonymity Analysis ===")
print(f"Quasi-identifiers: age, gender")
print(f"Direct identifier removed: patient_id\n")

k_values = [5, 10, 50]
for k in k_values:
    # Increase bin size with k to achieve anonymity
    bin_size = max(5, k // 2)
    anon_df, s = apply_k_anonymity(
        patients,
        quasi_identifiers=["age", "gender"],
        age_bin_size=bin_size,
        k=k,
    )
    print(
        f"k={k:<4} bin_size={bin_size:<4} | "
        f"rows={s['anonymised_rows']}/{s['original_rows']} "
        f"suppressed={s['suppression_rate']:.1%} "
        f"equiv_classes={s['equivalence_classes']}"
    )

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

# Create synthetic features for the classification task
rng = np.random.default_rng(42)
n = patients.height
features = rng.normal(0, 1, (n, 6))
median_age = patients["age"].median()
labels = (patients["age"].to_numpy() > median_age).astype(int)

# Split into target model data and shadow model data
X_target, X_shadow, y_target, y_shadow = train_test_split(
    features, labels, test_size=0.5, random_state=42
)

# Further split target data into train and hold-out
X_train, X_holdout, y_train, y_holdout = train_test_split(
    X_target, y_target, test_size=0.5, random_state=42
)

# Train target model (the model we are attacking)
target_model = RandomForestClassifier(n_estimators=100, random_state=42)
target_model.fit(X_train, y_train)

# Train shadow model (attacker's model to mimic target)
X_shadow_train, X_shadow_test, y_shadow_train, y_shadow_test = train_test_split(
    X_shadow, y_shadow, test_size=0.5, random_state=42
)
shadow_model = RandomForestClassifier(n_estimators=100, random_state=42)
shadow_model.fit(X_shadow_train, y_shadow_train)

# Build attack dataset from shadow model
# Members: shadow training data (label=1)
# Non-members: shadow test data (label=0)
shadow_train_probs = shadow_model.predict_proba(X_shadow_train)
shadow_test_probs = shadow_model.predict_proba(X_shadow_test)

attack_X = np.vstack([shadow_train_probs, shadow_test_probs])
attack_y = np.concatenate(
    [
        np.ones(len(shadow_train_probs)),  # Members
        np.zeros(len(shadow_test_probs)),  # Non-members
    ]
)

# Train attack classifier
attack_model = LogisticRegression(random_state=42)
attack_model.fit(attack_X, attack_y)

# Evaluate attack on target model
target_train_probs = target_model.predict_proba(X_train)
target_holdout_probs = target_model.predict_proba(X_holdout)

eval_X = np.vstack([target_train_probs, target_holdout_probs])
eval_y = np.concatenate(
    [
        np.ones(len(target_train_probs)),  # Actual members
        np.zeros(len(target_holdout_probs)),  # Actual non-members
    ]
)

attack_preds = attack_model.predict(eval_X)
attack_probs = attack_model.predict_proba(eval_X)[:, 1]

attack_acc = accuracy_score(eval_y, attack_preds)
attack_auc = roc_auc_score(eval_y, attack_probs)

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

# Retrain target with strong regularisation (reduces overfitting -> reduces MI leakage)
defended_model = RandomForestClassifier(
    n_estimators=50,  # Fewer trees (early stopping proxy)
    max_depth=3,  # Shallow trees (L2-like constraint)
    min_samples_leaf=10,  # Minimum samples per leaf (regularisation)
    random_state=42,
)
defended_model.fit(X_train, y_train)

# Re-run membership inference attack on defended model
defended_train_probs = defended_model.predict_proba(X_train)
defended_holdout_probs = defended_model.predict_proba(X_holdout)

defended_eval_X = np.vstack([defended_train_probs, defended_holdout_probs])
defended_attack_probs = attack_model.predict_proba(defended_eval_X)[:, 1]
defended_attack_preds = attack_model.predict(defended_eval_X)

defended_acc = accuracy_score(eval_y, defended_attack_preds)
defended_auc = roc_auc_score(eval_y, defended_attack_probs)

model_acc = accuracy_score(y_holdout, defended_model.predict(X_holdout))

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
    """Fit a Gaussian copula by estimating marginal CDFs and correlation.

    Steps:
    1. Transform each column to uniform marginals via empirical CDF
    2. Transform uniforms to standard normal via inverse CDF (probit)
    3. Estimate correlation matrix of the normal-transformed data
    """
    n_samples, n_features = data.shape

    # Step 1: Empirical CDF -> uniform marginals
    uniforms = np.zeros_like(data)
    for j in range(n_features):
        ranks = stats.rankdata(data[:, j])
        # Clip to avoid infinity in probit transform
        uniforms[:, j] = np.clip(ranks / (n_samples + 1), 0.001, 0.999)

    # Step 2: Probit transform (inverse normal CDF)
    normals = stats.norm.ppf(uniforms)

    # Step 3: Correlation matrix
    corr_matrix = np.corrcoef(normals, rowvar=False)

    # Store marginal statistics for inverse transform
    marginal_means = np.mean(data, axis=0)
    marginal_stds = np.std(data, axis=0)

    return corr_matrix, marginal_means, marginal_stds


def sample_gaussian_copula(
    n_samples: int,
    corr_matrix: np.ndarray,
    marginal_means: np.ndarray,
    marginal_stds: np.ndarray,
    generator: np.random.Generator,
) -> np.ndarray:
    """Generate synthetic data from fitted Gaussian copula."""
    n_features = len(marginal_means)

    # Sample from multivariate normal with estimated correlation
    normals = generator.multivariate_normal(
        np.zeros(n_features), corr_matrix, size=n_samples
    )

    # Transform back using marginal statistics
    synthetic = normals * marginal_stds + marginal_means

    return synthetic


# Fit copula on real data
real_data = features[:200]
corr_mat, means, stds = fit_gaussian_copula(real_data)

# Generate synthetic data
synthetic_data = sample_gaussian_copula(200, corr_mat, means, stds, rng)

# Compare statistical properties
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

# Correlation preservation
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

priv_file = Path(tempfile.mktemp(suffix=".yaml"))
priv_file.write_text(privacy_org_yaml)
priv_loaded = load_org_yaml(str(priv_file))
priv_compiled = compile_org(priv_loaded.org_definition)
priv_engine = GovernanceEngine(priv_loaded.org_definition)

priv_roles = {}
for addr, node in priv_compiled.nodes.items():
    if node.role_definition is not None and not node.is_vacant:
        priv_roles[node.node_id] = addr

level_map = {
    "restricted": ConfidentialityLevel.RESTRICTED,
    "confidential": ConfidentialityLevel.CONFIDENTIAL,
    "secret": ConfidentialityLevel.SECRET,
    "top_secret": ConfidentialityLevel.TOP_SECRET,
}

for cs in priv_loaded.clearances:
    if cs.role_id in priv_roles:
        clearance = RoleClearance(
            role_address=priv_roles[cs.role_id],
            max_clearance=level_map[cs.level],
            compartments=frozenset(cs.compartments),
            granted_by_role_address="system_init",
        )
        priv_engine.grant_clearance(priv_roles[cs.role_id], clearance)

# Set inference agent envelope
inf_env = RoleEnvelope(
    id=f"env-{uuid.uuid4().hex[:8]}",
    defining_role_address=priv_roles["service_admin"],
    target_role_address=priv_roles["inference_agent"],
    envelope=ConstraintEnvelopeConfig(
        id="inference-agent-envelope",
        operational=OperationalConstraintConfig(
            allowed_actions=["predict", "get_model_info", "log_request"],
        ),
        temporal=TemporalConstraintConfig(),
        data_access=DataAccessConstraintConfig(),
        communication=CommunicationConstraintConfig(),
    ),
)
priv_engine.set_role_envelope(inf_env)

# Define knowledge items at different classification levels
inference_items = [
    KnowledgeItem(
        item_id="raw_patient_record",
        classification=ConfidentialityLevel.SECRET,
        owning_unit_address=priv_roles.get("service_admin", "R1"),
        compartments=frozenset(["patient_data"]),
    ),
    KnowledgeItem(
        item_id="model_prediction_output",
        classification=ConfidentialityLevel.CONFIDENTIAL,
        owning_unit_address=priv_roles.get("inference_agent", "R2"),
        compartments=frozenset(["model_outputs"]),
    ),
    KnowledgeItem(
        item_id="anonymised_dataset",
        classification=ConfidentialityLevel.RESTRICTED,
        owning_unit_address=priv_roles.get("dpo", "R3"),
        compartments=frozenset(["anonymised_data"]),
    ),
]

print(f"\n=== PACT-Governed Inference ===")
print(f"Access control matrix:\n")

requesters = ["researcher", "external_user", "inference_agent", "dpo"]
for item in inference_items:
    print(f"Resource: {item.item_id} [{item.classification.name}]")
    for req_id in requesters:
        if req_id in priv_roles:
            decision = priv_engine.check_access(
                priv_roles[req_id], item, TrustPosture.SUPERVISED
            )
            status = "ALLOW" if decision.allowed else "DENY"
            print(f"  {req_id:<20} -> {status}")
    print()

# Governed inference workflow
governed_inf = PactGovernedAgent(
    engine=priv_engine,
    role_address=priv_roles["inference_agent"],
)

print(f"Governed inference agent:")
print(f"  Clearance: {governed_inf.context.effective_clearance_level}")
print(f"  Actions: {sorted(governed_inf.context.allowed_actions)}")
print(f"  Compartments: {sorted(governed_inf.context.compartments)}")

# Verify action permissions
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
