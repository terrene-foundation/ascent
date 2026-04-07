# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT10 — Exercise 7: Enterprise ML Governance at Scale
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Build large-scale ML governance infrastructure -- model
#   governance registry via DataFlow, policy engine, automated retirement,
#   multi-tenancy isolation, and governance incident simulation.
#
# TASKS:
#   1. Design model governance registry using DataFlow (@db.model + db.express)
#   2. Implement governance policy engine with structured verdicts
#   3. Model retirement automation: DriftMonitor -> PACT clearance downgrade
#   4. Multi-tenancy isolation: shared InferenceServer, isolated governance
#   5. Governance incident simulation: bias -> alert -> audit -> remediation
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from dataflow import DataFlow
from kailash_ml.engines.drift_monitor import DriftMonitor
from kailash.trust import ConfidentialityLevel, TrustPosture
from pact import GovernanceEngine, PactGovernedAgent, compile_org, load_org_yaml
from pact.governance import RoleClearance, RoleEnvelope
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


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Model governance registry using DataFlow
# ══════════════════════════════════════════════════════════════════════

print("=== Model Governance Registry ===\n")

db = DataFlow("sqlite:///:memory:")


# TODO: Define GovernedModel @db.model with fields:
#   id (primary_key), model_id, model_name, version, training_data_hash,
#   adapter_history (JSON), clearance_requirement, drift_psi_latest (default 0.0),
#   drift_check_count (default 0), compliance_status (default "PENDING"),
#   retirement_status (default "ACTIVE"), registered_at, last_audit_at (default "")
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


# TODO: Define GovernanceEvent @db.model with fields:
#   id (primary_key), event_type, model_id, actor, details (JSON), timestamp
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


# TODO: Implement setup_registry() async function
# Hint: 1) await db.initialize()  (db is defined above as DataFlow instance)
#        2) for each model in models_to_register: await db.express.create("GovernedModel", data)
#        3) models = await db.express.list("GovernedModel", {}), print table, return (dataflow, models)
async def setup_registry():
    """Initialise DataFlow and create governance tables."""
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


dataflow, registered_models = asyncio.run(setup_registry())


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Governance policy engine with structured verdicts
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print(f"=== Governance Policy Engine ===")
print(f"{'=' * 70}\n")

# TODO: Define GOVERNANCE_POLICIES list of 5 dicts, each with:
#   policy_id, name, description, check (lambda m: ...), severity ("BLOCKING" or "WARNING")
#   Policies: POL-001 compliance_status=="COMPLIANT" (BLOCKING),
#             POL-002 drift_psi_latest < 0.25 (BLOCKING),
#             POL-003 bool(training_data_hash) (BLOCKING),
#             POL-004 retirement_status != "RETIRED" (BLOCKING),
#             POL-005 bool(last_audit_at) (WARNING)
GOVERNANCE_POLICIES = ____


# TODO: Implement apply_policies(model_metadata: dict) -> dict
# Hint: for each policy in GOVERNANCE_POLICIES: run policy["check"](model_metadata)
#        mark PASS/FAIL; add to blocking_violations if BLOCKING and failed
#        set overall = "APPROVED" | "APPROVED_WITH_WARNINGS" | "BLOCKED"
def apply_policies(model_metadata: dict) -> dict:
    """Apply all governance policies to a model and produce a structured verdict."""
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


for model_data in registered_models:
    verdict = apply_policies(model_data)
    print(f"Model: {verdict['model_id']}")
    print(f"  Overall: {verdict['overall']}")
    for p in verdict["policies"]:
        marker = "PASS" if p["status"] == "PASS" else "FAIL"
        print(f"  [{marker}] {p['policy_id']}: {p['name']} ({p['severity']})")
    if verdict["blocking_violations"]:
        print(f"  BLOCKED by: {verdict['blocking_violations']}")
    if verdict["warnings"]:
        print(f"  Warnings: {verdict['warnings']}")
    print()


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Model retirement automation — DriftMonitor -> clearance downgrade
# ══════════════════════════════════════════════════════════════════════

print(f"{'=' * 70}")
print(f"=== Automated Model Retirement ===")
print(f"{'=' * 70}\n")

rng = np.random.default_rng(42)
reference_data = rng.normal(0, 1, (1000, 10))
feature_names = [f"v{i}" for i in range(10)]

monitor = DriftMonitor(
    reference_data=reference_data,
    feature_names=feature_names,
    psi_threshold=0.25,
)

retire_org_yaml = """
org_id: model_lifecycle
name: "Model Lifecycle Management"

departments:
  - id: ml_ops
    name: "ML Operations"

roles:
  - id: ops_lead
    name: "ML Operations Lead"
    is_primary_for_unit: ml_ops

  - id: model_agent
    name: "Model Serving Agent"
    reports_to: ops_lead
    agent: true

clearances:
  - role: ops_lead
    level: secret
    compartments: [model_weights, production_logs, drift_metrics]
  - role: model_agent
    level: confidential
    compartments: [model_weights, production_logs]
"""

retire_file = Path(tempfile.mktemp(suffix=".yaml"))
retire_file.write_text(retire_org_yaml)
retire_loaded = load_org_yaml(str(retire_file))
retire_engine = GovernanceEngine(retire_loaded.org_definition)
retire_compiled = compile_org(retire_loaded.org_definition)

retire_roles = {}
for addr, node in retire_compiled.nodes.items():
    if node.role_definition is not None and not node.is_vacant:
        retire_roles[node.node_id] = addr

level_map = {
    "confidential": ConfidentialityLevel.CONFIDENTIAL,
    "secret": ConfidentialityLevel.SECRET,
    "restricted": ConfidentialityLevel.RESTRICTED,
}

for cs in retire_loaded.clearances:
    if cs.role_id in retire_roles:
        clearance = RoleClearance(
            role_address=retire_roles[cs.role_id],
            max_clearance=level_map[cs.level],
            compartments=frozenset(cs.compartments),
            granted_by_role_address="system_init",
        )
        retire_engine.grant_clearance(retire_roles[cs.role_id], clearance)


# TODO: Implement simulate_retirement_automation()
# Hint: 1) check_drift on stable, mild-drift, persistent-drift data
#        2) if two consecutive drift alerts: downgrade clearance to RESTRICTED via RoleClearance
#        3) await db.express.update("GovernedModel", id, {retirement_status, drift_psi_latest, compliance_status})
#        4) await db.express.create("GovernanceEvent", {event_type:"RETIREMENT", ...})
#        5) get_context(model_agent), apply_policies(updated_model), print verdict
async def simulate_retirement_automation():
    """Simulate: PSI > 0.25 on two consecutive checks -> downgrade + retire."""
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
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____


asyncio.run(simulate_retirement_automation())
retire_file.unlink()


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Multi-tenancy isolation — shared infrastructure, isolated governance
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print(f"=== Multi-Tenancy Isolation ===")
print(f"{'=' * 70}\n")

org_a_yaml = """
org_id: bank_alpha
name: "Bank Alpha"
departments:
  - id: ml_team
    name: "ML Team"
roles:
  - id: alpha_agent
    name: "Alpha ML Agent"
    is_primary_for_unit: ml_team
    agent: true
clearances:
  - role: alpha_agent
    level: confidential
    compartments: [alpha_models, alpha_logs]
"""

org_b_yaml = """
org_id: bank_beta
name: "Bank Beta"
departments:
  - id: ml_team
    name: "ML Team"
roles:
  - id: beta_agent
    name: "Beta ML Agent"
    is_primary_for_unit: ml_team
    agent: true
clearances:
  - role: beta_agent
    level: confidential
    compartments: [beta_models, beta_logs]
"""

file_a = Path(tempfile.mktemp(suffix=".yaml"))
file_a.write_text(org_a_yaml)
file_b = Path(tempfile.mktemp(suffix=".yaml"))
file_b.write_text(org_b_yaml)

loaded_a = load_org_yaml(str(file_a))
loaded_b = load_org_yaml(str(file_b))

# TODO: Create separate GovernanceEngines (engine_a, engine_b), compile both orgs,
#   build role address dicts (roles_a, roles_b), grant CONFIDENTIAL clearances to both
#   Then create alpha_model and beta_model KnowledgeItems (CONFIDENTIAL, their own compartments)
#   Test: engine_a.check_access(roles_a["alpha_agent"], alpha_model, TrustPosture.SUPERVISED)
#   Test: engine_b.check_access(roles_b["beta_agent"], beta_model, TrustPosture.SUPERVISED)
#   Print ALLOW/DENY + reason for each
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

print(f"Bank Alpha agent accessing Alpha model:")
# (result_a printed from your code above)

print(f"\nBank Beta agent accessing Beta model:")
# (result_b printed from your code above)

print(f"\nIsolation guarantee:")
print(f"  Separate GovernanceEngines per organisation")
print(f"  Bank Alpha roles do not exist in Bank Beta's engine (and vice versa)")
print(f"  Shared InferenceServer routes requests through org-specific governance")
print(f"  Audit trails are per-org (no cross-tenant log leakage)")

integrity_a = engine_a.verify_audit_integrity()
integrity_b = engine_b.verify_audit_integrity()
print(f"\n  Bank Alpha audit integrity: {'VALID' if integrity_a else 'BROKEN'}")
print(f"  Bank Beta audit integrity: {'VALID' if integrity_b else 'BROKEN'}")

file_a.unlink()
file_b.unlink()


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Governance incident simulation
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print(f"=== Governance Incident Simulation ===")
print(f"{'=' * 70}\n")

loader = ASCENTDataLoader()
fraud_data = loader.load("ascent04", "credit_card_fraud.parquet")

feature_cols = [c for c in fraud_data.columns if c.startswith("v")]
X = fraud_data.select(feature_cols + ["amount"]).to_numpy()
y = fraud_data["is_fraud"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

incident_model = GradientBoostingClassifier(
    n_estimators=100, max_depth=4, random_state=42
)
incident_model.fit(X_train, y_train)

amount_test = X_test[:, -1]
high_value_mask = amount_test > np.percentile(amount_test, 75)
low_value_mask = amount_test <= np.percentile(amount_test, 25)

y_pred = incident_model.predict(X_test)
high_value_fpr = (y_pred[high_value_mask] == 1).mean()
low_value_fpr = (y_pred[low_value_mask] == 1).mean()
fpr_gap = abs(high_value_fpr - low_value_fpr)

print(f"Incident: Biased fraud prediction rates detected")
print(f"  High-value transaction FPR: {high_value_fpr:.4f}")
print(f"  Low-value transaction FPR: {low_value_fpr:.4f}")
print(f"  FPR gap: {fpr_gap:.4f} ({'ALERT' if fpr_gap > 0.05 else 'OK'})")

if fpr_gap > 0.02:
    # TODO: Print full incident response chain: ALERT -> AUDIT -> ROOT CAUSE -> REMEDIATION -> D/T/R
    # Hint: 5-step chain: 1. DriftMonitor/bias alert, 2. ComplianceAuditAgent triggered,
    #        3. Root cause analysis, 4. Remediation steps (retrain, fairness constraint, etc.),
    #        5. D/T/R accountability (Delegator=ops_lead, Task=fraud_detection, Responsible=agent)
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
    print(f"     R (Responsible): Model Serving Agent executed predictions")
    print(f"     Trace: all steps logged in GovernanceEngine audit trail")

    # TODO: Log incident to GovernanceEvent table
    # Hint: await db.express.create("GovernanceEvent", {event_type:"POLICY_VIOLATION",
    #         model_id:"fraud_detector_v1", actor:"drift_monitor_system",
    #         details: json.dumps({type, high_value_fpr, low_value_fpr, fpr_gap, remediation}),
    #         timestamp: datetime.now().isoformat()})
    async def log_incident():
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

    asyncio.run(log_incident())

asyncio.run(dataflow.close())

db_path = Path("ascent10_governance.db")
if db_path.exists():
    db_path.unlink()

print("\n--- Exercise 7 complete: enterprise ML governance at scale ---")
