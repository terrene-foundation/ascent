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

# TODO: Build the design matrix using V-features + amount; target = is_fraud.
feature_cols = ____
X = ____
y = ____
feature_names = ____


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Federated learning with 3 virtual bank clients + DP
# ══════════════════════════════════════════════════════════════════════

print(f"{'=' * 70}")
print(f"  TASK 1: Federated Learning with Differential Privacy")
print(f"{'=' * 70}\n")

# TODO: Sort indices by amount and partition into 3 non-IID clients
# TODO: (bank_sg, bank_my, bank_id). For each client do an 80/20 split
# TODO: and store X_train/y_train/X_test/y_test in client_data.
sorted_indices = ____
n = ____
n_per_client = ____

client_indices = ____

client_data = {}
____

# TODO: Stratified 80/20 global split for evaluation.
X_train_all, X_test_all, y_train_all, y_test_all = ____


def fedavg_dp_train(
    client_data: dict,
    n_rounds: int,
    clip_norm: float,
    sigma: float,
) -> GradientBoostingClassifier:
    """Train a federated GBM ensemble with differential privacy."""
    # TODO: For each round, train a GradientBoostingClassifier on each client.
    # TODO: Wrap the resulting client models in a FederatedEnsemble class
    # TODO: that averages predict_proba across clients with clipping + noise.
    # TODO: Return the ensemble (must expose predict + predict_proba + classes_).
    ____


# TODO: Train fed_model = fedavg_dp_train(client_data, n_rounds=3,
# TODO: clip_norm=1.0, sigma=1.0). Score on global test set.
fed_model = ____

fed_pred = ____
fed_proba = ____
fed_acc = ____
fed_auc = ____

# TODO: Train a centralised GradientBoostingClassifier baseline (n_estimators=100).
centralized = ____
____
cent_acc = ____
cent_auc = ____

print(f"\nFederated model (DP, sigma=1.0): accuracy={fed_acc:.4f}, AUC={fed_auc:.4f}")
print(f"Centralized baseline:            accuracy={cent_acc:.4f}, AUC={cent_auc:.4f}")
print(
    f"Utility gap: accuracy={abs(fed_acc - cent_acc):.4f}, AUC={abs(fed_auc - cent_auc):.4f}"
)


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Model card, dataset datasheet, bias audit
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print(f"  TASK 2: Compliance Artefacts + Bias Audit")
print(f"{'=' * 70}\n")

# TODO: Bucket the global test set by amount (low_value, mid_value, high_value)
# TODO: into a segments dict (name -> mask).
amount_test = ____
segments = ____

print(f"Bias audit (per amount segment):")
print(f"{'Segment':<15} {'N':>6} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Pos Rate':>10}")
print("-" * 60)

# TODO: For each segment compute precision/recall/f1/positive_rate via fed_pred
# TODO: (skip groups with single class). Track positive rates for parity gap.
bias_pass = True
pos_rates = []
____

# TODO: Compute demographic parity gap = max(pos_rates) - min(pos_rates);
# TODO: PASS if <= 0.05, else set bias_pass=False.
if len(pos_rates) >= 2:
    max_gap = ____
    print(
        f"\nDemographic parity gap: {max_gap:.4f} ({'PASS' if max_gap <= 0.05 else 'FAIL'})"
    )
    if max_gap > 0.05:
        bias_pass = False

# TODO: Build a model_card dict (name, version, type, intended_use, metrics,
# TODO: bias_audit, limitations) summarising the federated DP model.
model_card = ____

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

# TODO: Train an adv_model = GradientBoostingClassifier (n_estimators=30, max_depth=4).
adv_model = ____
____


def fgsm_attack(model, X_samples, y_samples, epsilon, delta=1e-5):
    """FGSM via numerical gradient approximation."""
    # TODO: For each feature column, compute a finite-difference NLL gradient
    # TODO: and return X_samples + epsilon * sign(gradient).
    ____


# TODO: Generate 500 adversarial training examples at epsilon=0.1.
n_adv = 500
X_adv = ____

# TODO: Build augmented training set (vstack X_train_all + X_adv) and labels.
X_robust = ____
y_robust = ____

# TODO: Train robust_model = GradientBoostingClassifier on the augmented data.
robust_model = ____
____

robust_acc = ____
robust_auc = ____

# TODO: Measure FGSM ASR (eps=0.1) on the robust model with the first 500 test points.
X_adv_test = ____
clean_preds = ____
adv_preds = ____
correct_mask = ____
asr = ____

print(f"Adversarial training results:")
print(f"  Clean accuracy: {robust_acc:.4f} (baseline: {cent_acc:.4f})")
print(f"  Clean AUC:      {robust_auc:.4f} (baseline: {cent_auc:.4f})")
print(f"  FGSM ASR (eps=0.1): {asr:.4f}")

# TODO: Certified robustness: for n_certify points, sample n_noise_samples
# TODO: at sigma=0.25, take majority vote, compute confidence, derive
# TODO: certified radius via norm.ppf, count those whose radius >= 0.1.
rng_smooth = np.random.default_rng(42)
n_certify = 20
n_noise_samples = 200
sigma = 0.25
target_radius = 0.1
target_prob = 0.95

certified_count = ____
____

certified_pct = certified_count / n_certify
print(f"\nCertified robustness (R={target_radius}, sigma={sigma}):")
print(f"  Certified: {certified_count}/{n_certify} ({certified_pct:.2%})")
print(f"  Target: {target_prob:.0%}")
print(f"  Status: {'PASS' if certified_pct >= target_prob else 'BELOW TARGET'}")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Governance registry + compliance policies
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print(f"  TASK 4: Governance Registry + Compliance Policies")
print(f"{'=' * 70}\n")


async def register_in_governance():
    """Register capstone model in ModelRegistry and apply compliance policies."""
    # TODO: ConnectionManager(":memory:"), initialise, build ModelRegistry.
    # TODO: Pickle robust_model, hash a prefix as training_data_hash.
    # TODO: Build a ModelSignature using FeatureSchema/FeatureField.
    # TODO: Register the model with metrics (accuracy, auc, fgsm asr,
    # TODO: certified pct, dp_sigma) and promote to production.
    # TODO: Build model_metadata dict, define a 5-policy list (compliance,
    # TODO: drift, provenance, retired, audit) and print PASS/FAIL per policy.
    # TODO: Return (conn, registry, model_version).
    ____


conn, registry, model_version = asyncio.run(register_in_governance())


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Deploy via Nexus with PactGovernedAgent + DriftMonitor
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print(f"  TASK 5: Governed Deployment + Drift Monitoring")
print(f"{'=' * 70}\n")

# TODO: Define a YAML org with fraud_ops_lead, fraud_agent (agent: true),
# TODO: compliance_lead, audit_agent (agent: true). Grant clearances at
# TODO: secret/confidential/top_secret on relevant compartments.
capstone_org_yaml = ____

# TODO: Write to temp, load, compile, build GovernanceEngine.
cap_file = ____
____
cap_loaded = ____
cap_engine = ____
cap_compiled = ____

cap_roles = ____
____

level_map = ____

# TODO: Grant the four role clearances on cap_engine.
____

# TODO: Set envelope for fraud_agent: predict_fraud, get_model_info, log_prediction.
fraud_env = ____
____

# TODO: Set envelope for audit_agent: run_audit, read_model_card,
# TODO: read_drift_status, generate_report.
audit_env = ____
____

# TODO: Wrap fraud_agent and audit_agent in PactGovernedAgent.
governed_fraud = ____
governed_audit = ____

print(f"Fraud agent: clearance={governed_fraud.context.effective_clearance_level}")
print(f"  Actions: {sorted(governed_fraud.context.allowed_actions)}")
print(f"Audit agent: clearance={governed_audit.context.effective_clearance_level}")
print(f"  Actions: {sorted(governed_audit.context.allowed_actions)}")


async def deploy_and_monitor():
    # TODO: Build InferenceServer(registry, cache_size=5), warm cache,
    # TODO: verify_action(predict_fraud), .predict on a test row,
    # TODO: build a DriftMonitor(reference_data=X_train_all[:2000],
    # TODO: psi_threshold=0.2), check_drift on first 500 test rows,
    # TODO: report max-PSI feature. Return (server, monitor).
    ____


server, monitor = asyncio.run(deploy_and_monitor())


# ══════════════════════════════════════════════════════════════════════
# TASK 6: ComplianceAuditAgent — JSON attestation + audit report
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print(f"  TASK 6: Compliance Audit — End-to-End Attestation")
print(f"{'=' * 70}\n")

# TODO: Verify audit_agent governance check via cap_engine.verify_action.
audit_verdict = ____
print(f"Audit agent governance check: {audit_verdict.level}")

# TODO: Build the attestation dict with attestation_id, timestamp, system,
# TODO: model, privacy, fairness, robustness, governance, compliance
# TODO: (EU AI Act, MAS TRM, PDPA, SG AI Verify), deployment, overall_verdict.
attestation = ____

# TODO: Walk attestation['compliance'] and flip overall_verdict if any
# TODO: article string contains 'NON_COMPLIANT'.
____

print(f"=== Machine-Readable JSON Attestation ===")
print(json.dumps(attestation, indent=2, default=str))

print(f"\n{'=' * 70}")
print(f"  HUMAN-READABLE AUDIT REPORT")
print(f"  Attestation: {attestation['attestation_id']}")
print(f"  Date: {attestation['timestamp']}")
print(f"{'=' * 70}")

print(f"\n1. MODEL IDENTITY")
print(f"   Name: {attestation['model']['name']} v{attestation['model']['version']}")
print(f"   Type: {attestation['model']['type']}")
print(f"   Training: {attestation['model']['training_method']}")

print(f"\n2. PRIVACY GUARANTEES")
print(f"   Federated learning: {attestation['privacy']['n_clients']} clients")
print(
    f"   Differential privacy: epsilon ~{attestation['privacy']['dp_epsilon_approx']}"
)

print(f"\n3. FAIRNESS ASSESSMENT")
print(f"   Bias audit: {attestation['fairness']['status']}")
print(
    f"   Demographic parity gap: {attestation['fairness']['demographic_parity_gap']:.4f}"
)

print(f"\n4. ROBUSTNESS CERTIFICATION")
print(f"   Adversarial training: YES")
print(f"   FGSM ASR (eps=0.1): {attestation['robustness']['fgsm_asr_eps01']:.4f}")
print(
    f"   Certified at R={target_radius}: {attestation['robustness']['certified_robustness']['certified_percentage']:.2%}"
)

print(f"\n5. GOVERNANCE")
print(f"   PACT governed: YES")
print(
    f"   Audit trail integrity: {'VALID' if attestation['governance']['audit_trail_integrity'] else 'BROKEN'}"
)
print(f"   Frozen context: YES")

print(f"\n6. REGULATORY COMPLIANCE")
for reg, articles in attestation["compliance"].items():
    compliant_count = sum(
        1 for s in articles.values() if "COMPLIANT" in s and "NON" not in s
    )
    total = len(articles)
    print(f"   {reg}: {compliant_count}/{total} articles compliant")

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

# TODO: Close the connection, unlink temporary files, drop the capstone db.
____
____
db_path = Path("ascent10_capstone.db")
if db_path.exists():
    db_path.unlink()

print("--- Exercise 8 (CAPSTONE) complete: regulated AI system end-to-end ---")
print("  Module 10 complete: AI governance, safety, and enterprise at scale")
