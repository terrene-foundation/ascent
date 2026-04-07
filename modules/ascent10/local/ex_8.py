# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT10 — Exercise 8: Capstone — Regulated AI System End-to-End
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Build a complete regulated AI system combining federated
#   learning, compliance artefacts, adversarial robustness, governance
#   registry, and governed deployment -- the full M10 stack.
#
# TASKS:
#   1. Train fraud model via federated learning (3 virtual banks) with DP
#   2. Generate model card and dataset datasheet, run bias audit
#   3. Adversarial training + certify robustness at R=0.1
#   4. Register in governance registry, apply compliance policies
#   5. Deploy via Nexus with PactGovernedAgent + DriftMonitor alerts
#   6. Run ComplianceAuditAgent end-to-end, output JSON attestation
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import pickle
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from kailash.db.connection import ConnectionManager
from kailash_ml.engines.model_registry import ModelRegistry
from kailash_ml.engines.inference_server import InferenceServer
from kailash_ml.engines.drift_monitor import DriftMonitor
from kailash_ml.types import FeatureSchema, FeatureField, ModelSignature, MetricSpec
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
from nexus import Nexus

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()


# ── Data Loading ─────────────────────────────────────────────────────

loader = ASCENTDataLoader()
fraud_data = loader.load("ascent04", "credit_card_fraud.parquet")

print("=== CAPSTONE: Regulated AI System End-to-End ===")
print(f"Fraud dataset: {fraud_data.shape}")
print(f"Fraud rate: {fraud_data['is_fraud'].mean():.4%}\n")

feature_cols = [c for c in fraud_data.columns if c.startswith("v")]
X = fraud_data.select(feature_cols + ["amount"]).to_numpy()
y = fraud_data["is_fraud"].to_numpy()
feature_names = feature_cols + ["amount"]


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Federated learning with 3 virtual bank clients + DP
# ══════════════════════════════════════════════════════════════════════

print(f"{'=' * 70}")
print(f"  TASK 1: Federated Learning with Differential Privacy")
print(f"{'=' * 70}\n")

sorted_indices = np.argsort(fraud_data["amount"].to_numpy())
n = len(sorted_indices)
n_per_client = n // 3

client_indices = {
    "bank_sg": sorted_indices[:n_per_client],
    "bank_my": sorted_indices[n_per_client : 2 * n_per_client],
    "bank_id": sorted_indices[2 * n_per_client :],
}

# TODO: Build client_data dict: for each (name, idx) in client_indices,
#        train/test split with test_size=0.2, store X_train/y_train/X_test/y_test, print stats
# Hint: client_data[name] = {"X_train": X_tr, "y_train": y_tr, "X_test": X_te, "y_test": y_te}
____
____
____
____
____
____

X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# TODO: Implement fedavg_dp_train using FederatedEnsemble
# Hint: train local GradientBoostingClassifier per client per round (progressive n_estimators)
#        create FederatedEnsemble that averages predict_proba with clip + Gaussian noise
#        FederatedEnsemble needs: predict_proba (clip, average, add noise, renormalise) and predict
def fedavg_dp_train(
    client_data: dict,
    n_rounds: int,
    clip_norm: float,
    sigma: float,
) -> GradientBoostingClassifier:
    """Train a model using FedAvg with differential privacy."""
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____


fed_model = fedavg_dp_train(client_data, n_rounds=3, clip_norm=1.0, sigma=1.0)

# TODO: Evaluate fed_model on X_test_all (accuracy + AUC), train centralized baseline
#        (GradientBoostingClassifier, n_estimators=100), evaluate both, print comparison
# Hint: fed_acc = accuracy_score(y_test_all, fed_model.predict(X_test_all))
#        print utility gap: abs(fed_acc - cent_acc), abs(fed_auc - cent_auc)
____
____
____
____
____
____
____
____
____
____
____
____


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Model card, dataset datasheet, bias audit
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print(f"  TASK 2: Compliance Artefacts + Bias Audit")
print(f"{'=' * 70}\n")

# TODO: Define segments dict with 3 boolean masks (low_value, mid_value, high_value)
#        based on amount_test percentiles (33rd and 66th)
# Hint: amount_test = X_test_all[:, -1]; low=<33pct, mid=33-66pct, high=>66pct
amount_test = X_test_all[:, -1]
____
____
____
____

print(f"Bias audit (per amount segment):")
print(f"{'Segment':<15} {'N':>6} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Pos Rate':>10}")
print("-" * 60)

# TODO: Compute per-segment bias audit: for each segment, compute precision, recall, f1, pos_rate
#        Collect pos_rates list; compute demographic parity gap; set bias_pass=True/False
bias_pass = True
pos_rates = []
____
____
____
____
____
____
____
____
____
____

# TODO: Build model_card dict with model_name, version, type, intended_use, metrics, bias_audit, limitations
# Hint: metrics = {accuracy: fed_acc, auc_roc: fed_auc, dp_epsilon_approx: 1.0}
model_card = {
    "model_name": "capstone_fraud_federated_dp",
    "version": "1.0.0",
    "type": ____,
    "intended_use": ____,
    "metrics": ____,
    "bias_audit": ____,
    "limitations": ____,
}

print(f"\nModel card generated:")
print(f"  Name: {model_card['model_name']} v{model_card['version']}")
print(f"  Type: {model_card['type']}")
print(f"  Bias audit: {model_card['bias_audit']}")
for metric, value in model_card["metrics"].items():
    print(f"  {metric}: {value:.4f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Adversarial training + certified robustness
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print(f"  TASK 3: Adversarial Training + Certified Robustness")
print(f"{'=' * 70}\n")

adv_model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
adv_model.fit(X_train_all, y_train_all)


# TODO: Implement fgsm_attack using numerical gradient approximation
# Hint: for each feature i: finite difference df/dx_i = (loss(x+delta) - loss(x-delta))/(2*delta)
#        loss = -log(p(true_class)); return X_samples + epsilon * sign(gradients)
def fgsm_attack(model, X_samples, y_samples, epsilon, delta=1e-5):
    """FGSM via numerical gradient approximation."""
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____


n_adv = 2000
X_adv = fgsm_attack(adv_model, X_train_all[:n_adv], y_train_all[:n_adv], epsilon=0.1)

X_robust = np.vstack([X_train_all, X_adv])
y_robust = np.concatenate([y_train_all, y_train_all[:n_adv]])

robust_model = GradientBoostingClassifier(
    n_estimators=100, max_depth=4, random_state=42
)
robust_model.fit(X_robust, y_robust)

# TODO: Evaluate robust_model (clean accuracy + AUC), run fgsm_attack on 500 test samples,
#        compute ASR, and print comparison vs baseline
# Hint: correct_mask = (clean_preds == y_test_all[:500]); asr = flipped / correct_mask.sum()
____
____
____
____
____
____
____
____

# Certified robustness via randomised smoothing
rng_smooth = np.random.default_rng(42)
n_certify = 100
n_noise_samples = 200
sigma = 0.25
target_radius = 0.1
target_prob = 0.95

# TODO: Run randomised smoothing certification loop over n_certify test samples
#        For each: add n_noise_samples Gaussian perturbations (sigma), take majority vote,
#        compute p_lower=confidence-0.05; if p_lower > 0.5, certified radius = sigma * norm.ppf(p_lower)
#        Count how many samples certify at radius >= target_radius, print certified %
# Hint: rng_smooth.normal(0, sigma, (n_noise_samples, X_test_all.shape[1]));
#        certified_pct = certified_count / n_certify
____
____
____
____
____
____
____
____
____
____


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Governance registry + compliance policies
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print(f"  TASK 4: Governance Registry + Compliance Policies")
print(f"{'=' * 70}\n")


# TODO: Implement register_in_governance()
# Hint: 1) ConnectionManager("sqlite:///:memory:"), await conn.initialize()
#        2) ModelRegistry(conn); pickle.dumps(robust_model) for artifact
#        3) registry.register_model(name="capstone_fraud_regulated", artifact, metrics, signature)
#        4) registry.promote_model(name, version, target_stage="production", reason=...)
#        5) apply compliance policies: list of (policy_name, bool_check) tuples
#        6) return conn, registry, model_version
async def register_in_governance():
    """Register capstone model in ModelRegistry and apply compliance policies."""
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____


conn, registry, model_version = asyncio.run(register_in_governance())


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Deploy via Nexus with PactGovernedAgent + DriftMonitor
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print(f"  TASK 5: Governed Deployment + Drift Monitoring")
print(f"{'=' * 70}\n")

capstone_org_yaml = """
org_id: capstone_regulated
name: "Capstone Regulated AI System"

departments:
  - id: fraud_ops
    name: "Fraud Operations"
  - id: compliance
    name: "Compliance"

roles:
  - id: fraud_ops_lead
    name: "Fraud Operations Lead"
    is_primary_for_unit: fraud_ops

  - id: fraud_agent
    name: "Fraud Detection Agent"
    reports_to: fraud_ops_lead
    agent: true

  - id: compliance_lead
    name: "Chief Compliance Officer"
    is_primary_for_unit: compliance

  - id: audit_agent
    name: "Compliance Audit Agent"
    reports_to: compliance_lead
    agent: true

clearances:
  - role: fraud_ops_lead
    level: secret
    compartments: [transaction_data, model_predictions, model_weights, audit_logs]
  - role: fraud_agent
    level: confidential
    compartments: [transaction_data, model_predictions]
  - role: compliance_lead
    level: top_secret
    compartments: [transaction_data, model_predictions, model_weights, audit_logs]
  - role: audit_agent
    level: secret
    compartments: [model_predictions, audit_logs]
"""

# TODO: Load capstone_org_yaml, compile, create GovernanceEngine, build cap_roles dict
#        Build level_map (confidential/secret/top_secret) and grant clearances to all roles
____
____
____
____
____
____
____
____
____
____
____
____
____
____
____

# TODO: Create fraud_env (allowed_actions: predict_fraud, get_model_info, log_prediction)
#        and audit_env (allowed_actions: run_audit, read_model_card, read_drift_status, generate_report)
#        Then set both envelopes on cap_engine
fraud_env = ____
cap_engine.set_role_envelope(fraud_env)

audit_env = ____
cap_engine.set_role_envelope(audit_env)

# TODO: Create PactGovernedAgent for fraud_agent and audit_agent
governed_fraud = ____
governed_audit = ____

print(f"Fraud agent: clearance={governed_fraud.context.effective_clearance_level}")
print(f"  Actions: {sorted(governed_fraud.context.allowed_actions)}")
print(f"Audit agent: clearance={governed_audit.context.effective_clearance_level}")
print(f"  Actions: {sorted(governed_audit.context.allowed_actions)}")


# TODO: Implement deploy_and_monitor()
# Hint: 1) InferenceServer(registry, cache_size=5), await server.warm_cache(["capstone_fraud_regulated"])
#        2) cap_engine.verify_action(cap_roles["fraud_agent"], "predict_fraud")
#        3) await server.predict(model_name=..., features={...})
#        4) DriftMonitor(conn, psi_threshold=0.2); await monitor.set_reference(model_name=, reference_data=, feature_columns=)
#        5) monitor.check_drift(X_test_all[:500]); print drift status + max PSI feature
async def deploy_and_monitor():
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____


server, monitor = asyncio.run(deploy_and_monitor())


# ══════════════════════════════════════════════════════════════════════
# TASK 6: ComplianceAuditAgent — JSON attestation + audit report
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print(f"  TASK 6: Compliance Audit — End-to-End Attestation")
print(f"{'=' * 70}\n")

audit_verdict = cap_engine.verify_action(cap_roles["audit_agent"], "run_audit")
print(f"Audit agent governance check: {audit_verdict.level}")

# TODO: Build attestation dict with sections:
#   - model: name, version, type, training_method
#   - privacy: federated_learning, differential_privacy, dp_sigma, dp_epsilon_approx, n_clients
#   - fairness: bias_audit_conducted, demographic_parity_gap, threshold, status
#   - robustness: adversarial_training, fgsm_asr_eps01, certified_robustness (method, sigma, radius, %)
#   - governance: pact_governed, org_id, clearances, audit_trail_integrity, frozen_context
#   - compliance: EU AI Act, MAS TRM, PDPA, SG AI Verify (with article verdicts)
#   - deployment: inference_server, nexus_endpoint, drift_monitoring, psi_threshold
#   - overall_verdict: "COMPLIANT" (check for NON_COMPLIANT in compliance articles)
attestation = {
    "attestation_id": f"ATT-{uuid.uuid4().hex[:8]}",
    "timestamp": datetime.now().isoformat(),
    "system": "Capstone Regulated AI System",
    "model": ____,
    "privacy": ____,
    "fairness": ____,
    "robustness": ____,
    "governance": ____,
    "compliance": ____,
    "deployment": ____,
    "overall_verdict": "COMPLIANT",
}

# Check if any non-compliant items
for reg, articles in attestation["compliance"].items():
    for art, status in articles.items():
        if "NON_COMPLIANT" in status:
            attestation["overall_verdict"] = "NON_COMPLIANT"
            break

print(f"=== Machine-Readable JSON Attestation ===")
print(json.dumps(attestation, indent=2, default=str))

# TODO: Print human-readable audit report with 6 sections:
#   1. MODEL IDENTITY: name, version, type, training_method from attestation['model']
#   2. PRIVACY GUARANTEES: n_clients, dp_epsilon_approx from attestation['privacy']
#   3. FAIRNESS ASSESSMENT: bias status, demographic_parity_gap from attestation['fairness']
#   4. ROBUSTNESS CERTIFICATION: fgsm_asr_eps01, certified % from attestation['robustness']
#   5. GOVERNANCE: audit_trail_integrity from attestation['governance']
#   6. REGULATORY COMPLIANCE: per-regulation compliant/total counts from attestation['compliance']
# Hint: loop attestation['compliance'].items() to count "COMPLIANT" (not "NON_COMPLIANT") articles
____
____
____
____
____
____
____
____
____
____
____
____

print(f"\n{'=' * 70}")
print(f"  OVERALL VERDICT: {attestation['overall_verdict']}")
print(f"{'=' * 70}")

print(f"\n=== Platform Stack ===")
print(
    f"""
  Layer 7: Compliance   Automated audit + JSON attestation
  Layer 6: Nexus        Multi-channel deployment (API + CLI + MCP)
  Layer 5: PACT         D/T/R governance, envelopes, clearance, audit
  Layer 4: Robustness   FGSM/PGD defence + randomised smoothing certificates
  Layer 3: Privacy      Federated learning + differential privacy
  Layer 2: ML           InferenceServer + DriftMonitor + ModelRegistry
  Layer 1: Core SDK     WorkflowBuilder, runtime.execute(workflow.build())

  Every layer is integrated. Every action is governed. Every decision is auditable.
  This is production-grade regulated AI.
"""
)

asyncio.run(conn.close())
cap_file.unlink()
db_path = Path("ascent10_capstone.db")
if db_path.exists():
    db_path.unlink()

print("--- Exercise 8 (CAPSTONE) complete: regulated AI system end-to-end ---")
print("  Module 10 complete: AI governance, safety, and enterprise at scale")
