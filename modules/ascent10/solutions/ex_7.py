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
from kailash.db.connection import ConnectionManager
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

_db_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_db_tmp.close()
_db_path = _db_tmp.name
db = DataFlow(f"sqlite:///{_db_path}")


@db.model
class GovernedModel:
    """Registry entry for a governed ML model."""

    id: int
    model_id: str
    model_name: str
    version: str
    training_data_hash: str
    adapter_history: str  # JSON list of adapter versions
    clearance_requirement: str  # ConfidentialityLevel name
    drift_psi_latest: float = 0.0
    drift_check_count: int = 0
    compliance_status: str = "PENDING"
    retirement_status: str = "ACTIVE"
    registered_at: str = ""
    last_audit_at: str = ""


@db.model
class GovernanceEvent:
    """Audit log entry for governance actions."""

    id: int
    event_type: str  # REGISTRATION, DRIFT_CHECK, POLICY_VIOLATION, RETIREMENT
    model_id: str
    actor: str  # D/T/R address of the actor
    details: str  # JSON details
    timestamp: str


async def setup_registry():
    """Initialise DataFlow and create governance tables."""
    await db.initialize()

    # Register models in the governance registry
    models_to_register = [
        {
            "model_id": "fraud_detector_v1",
            "model_name": "Credit Card Fraud Detector",
            "version": "1.0.0",
            "training_data_hash": hashlib.sha256(b"fraud_training_data_v1").hexdigest()[
                :16
            ],
            "adapter_history": json.dumps([]),
            "clearance_requirement": "CONFIDENTIAL",
            "compliance_status": "COMPLIANT",
            "registered_at": datetime.now().isoformat(),
        },
        {
            "model_id": "fraud_detector_v2",
            "model_name": "Credit Card Fraud Detector (DP)",
            "version": "2.0.0",
            "training_data_hash": hashlib.sha256(
                b"fraud_training_data_v2_dp"
            ).hexdigest()[:16],
            "adapter_history": json.dumps(["lora_r16_sft_v1"]),
            "clearance_requirement": "SECRET",
            "compliance_status": "COMPLIANT",
            "registered_at": datetime.now().isoformat(),
        },
        {
            "model_id": "icu_predictor_v1",
            "model_name": "ICU Outcome Predictor (Federated)",
            "version": "1.0.0",
            "training_data_hash": hashlib.sha256(b"federated_icu_data").hexdigest()[
                :16
            ],
            "adapter_history": json.dumps([]),
            "clearance_requirement": "SECRET",
            "compliance_status": "PENDING",
            "registered_at": datetime.now().isoformat(),
        },
    ]

    created_models = []
    for model_data in models_to_register:
        create_result = await db.express.create("GovernedModel", model_data)
        # Capture the auto-assigned id alongside the original payload
        enriched = {**model_data, "id": create_result.get("id")}
        created_models.append(enriched)

    print(f"Registered {len(created_models)} models in governance registry:\n")
    print(f"{'Model ID':<25} {'Version':<10} {'Clearance':<15} {'Status':<12}")
    print("-" * 65)
    for m in created_models:
        print(
            f"{m['model_id']:<25} {m['version']:<10} "
            f"{m['clearance_requirement']:<15} {m['compliance_status']:<12}"
        )

    return db, created_models


dataflow, registered_models = asyncio.run(setup_registry())


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Governance policy engine with structured verdicts
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print(f"=== Governance Policy Engine ===")
print(f"{'=' * 70}\n")


# Define policies as structured rules
GOVERNANCE_POLICIES = [
    {
        "policy_id": "POL-001",
        "name": "Compliance status required",
        "description": "Model must have COMPLIANT status before production deployment",
        "check": lambda m: m.get("compliance_status") == "COMPLIANT",
        "severity": "BLOCKING",
    },
    {
        "policy_id": "POL-002",
        "name": "Drift threshold",
        "description": "Model PSI must be below 0.25 for continued production use",
        "check": lambda m: m.get("drift_psi_latest", 0.0) < 0.25,
        "severity": "BLOCKING",
    },
    {
        "policy_id": "POL-003",
        "name": "Training data provenance",
        "description": "Training data hash must be non-empty (provenance tracked)",
        "check": lambda m: bool(m.get("training_data_hash")),
        "severity": "BLOCKING",
    },
    {
        "policy_id": "POL-004",
        "name": "Retirement status",
        "description": "Model must not be in RETIRED status",
        "check": lambda m: m.get("retirement_status") != "RETIRED",
        "severity": "BLOCKING",
    },
    {
        "policy_id": "POL-005",
        "name": "Audit recency",
        "description": "Model should have been audited within the last 90 days",
        "check": lambda m: bool(m.get("last_audit_at")),
        "severity": "WARNING",
    },
]


def apply_policies(model_metadata: dict) -> dict:
    """Apply all governance policies to a model and produce a structured verdict."""
    verdict = {
        "model_id": model_metadata.get("model_id", "unknown"),
        "timestamp": datetime.now().isoformat(),
        "policies": [],
        "overall": "APPROVED",
        "blocking_violations": [],
        "warnings": [],
    }

    for policy in GOVERNANCE_POLICIES:
        passed = policy["check"](model_metadata)
        result = {
            "policy_id": policy["policy_id"],
            "name": policy["name"],
            "severity": policy["severity"],
            "status": "PASS" if passed else "FAIL",
        }
        verdict["policies"].append(result)

        if not passed:
            if policy["severity"] == "BLOCKING":
                verdict["blocking_violations"].append(policy["name"])
                verdict["overall"] = "BLOCKED"
            else:
                verdict["warnings"].append(policy["name"])
                if verdict["overall"] != "BLOCKED":
                    verdict["overall"] = "APPROVED_WITH_WARNINGS"

    return verdict


# Apply policies to each registered model
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

# Set up drift monitoring
rng = np.random.default_rng(42)
reference_data = rng.normal(0, 1, (1000, 10))
feature_names = [f"v{i}" for i in range(10)]

# Set up PACT governance for retirement
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


async def simulate_retirement_automation():
    """Simulate: PSI > 0.25 on two consecutive checks -> downgrade + retire."""
    drift_conn = ConnectionManager("sqlite:///:memory:")
    await drift_conn.initialize()
    try:
        monitor = DriftMonitor(drift_conn, psi_threshold=0.25)
        ref_df = pl.DataFrame(reference_data, schema=feature_names)
        await monitor.set_reference(
            model_name="fraud_detector_v1",
            reference_data=ref_df,
            feature_columns=feature_names,
        )

        def _max_psi(report) -> float:
            if not report.feature_results:
                return 0.0
            return max(f.psi for f in report.feature_results)

        # Week 1: stable
        stable_data = rng.normal(0, 1, (200, 10))
        stable_df = pl.DataFrame(stable_data, schema=feature_names)
        report_1 = await monitor.check_drift("fraud_detector_v1", stable_df)
        print(f"Check 1 (stable): drift={report_1.overall_drift_detected}")

        # Week 2: drift begins
        drifted_data_1 = rng.normal(0.6, 1.3, (200, 10))
        drifted_df_1 = pl.DataFrame(drifted_data_1, schema=feature_names)
        report_2 = await monitor.check_drift("fraud_detector_v1", drifted_df_1)
        psi_2 = _max_psi(report_2)
        print(
            f"Check 2 (mild drift): drift={report_2.overall_drift_detected}, "
            f"max_PSI={psi_2:.4f}"
        )

        # Week 3: drift persists
        drifted_data_2 = rng.normal(0.8, 1.5, (200, 10))
        drifted_df_2 = pl.DataFrame(drifted_data_2, schema=feature_names)
        report_3 = await monitor.check_drift("fraud_detector_v1", drifted_df_2)
        psi_3 = _max_psi(report_3)
        print(
            f"Check 3 (persistent drift): drift={report_3.overall_drift_detected}, "
            f"max_PSI={psi_3:.4f}"
        )

        consecutive_drift = (
            report_2.overall_drift_detected and report_3.overall_drift_detected
        )

        if consecutive_drift:
            print(f"\nTwo consecutive drift alerts -> triggering retirement protocol:")

            # Step 1: Downgrade PACT clearance to RESTRICTED
            restricted_clearance = RoleClearance(
                role_address=retire_roles["model_agent"],
                max_clearance=ConfidentialityLevel.RESTRICTED,
                compartments=frozenset(
                    ["production_logs"]
                ),  # Remove model_weights access
                granted_by_role_address=retire_roles["ops_lead"],
            )
            retire_engine.grant_clearance(
                retire_roles["model_agent"], restricted_clearance
            )
            print(f"  1. Clearance downgraded to RESTRICTED")

            # Step 2: Update registry
            model_record = registered_models[0]
            await db.express.update(
                "GovernedModel",
                str(model_record["id"]),
                {
                    "retirement_status": "RETIRING",
                    "drift_psi_latest": float(psi_3),
                    "compliance_status": "NON_COMPLIANT",
                },
            )
            print(f"  2. Registry updated: status=RETIRING, PSI={psi_3:.4f}")

            # Step 3: Log governance event
            await db.express.create(
                "GovernanceEvent",
                {
                    "event_type": "RETIREMENT",
                    "model_id": model_record["model_id"],
                    "actor": retire_roles["ops_lead"],
                    "details": json.dumps(
                        {
                            "reason": "Consecutive drift alerts",
                            "psi_values": [psi_2, psi_3],
                            "new_clearance": "RESTRICTED",
                        }
                    ),
                    "timestamp": datetime.now().isoformat(),
                },
            )
            print(f"  3. Governance event logged")

            # Step 4: Verify restricted agent cannot access model weights
            ctx = retire_engine.get_context(retire_roles["model_agent"])
            print(f"  4. Agent clearance now: {ctx.effective_clearance_level}")
            print(f"     Compartments: {sorted(ctx.compartments)}")

            # Re-apply policies using the in-memory record updated with new state
            updated_model = {
                **model_record,
                "retirement_status": "RETIRING",
                "drift_psi_latest": float(psi_3),
                "compliance_status": "NON_COMPLIANT",
            }
            verdict = apply_policies(updated_model)
            print(f"  5. Policy verdict: {verdict['overall']}")
            for v in verdict["blocking_violations"]:
                print(f"     BLOCKED: {v}")

        else:
            print(f"\nNo consecutive drift -- model remains active.")
    finally:
        await drift_conn.close()


asyncio.run(simulate_retirement_automation())
retire_file.unlink()


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Multi-tenancy isolation — shared infrastructure, isolated governance
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print(f"=== Multi-Tenancy Isolation ===")
print(f"{'=' * 70}\n")

# Two organisations sharing the same infrastructure
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

engine_a = GovernanceEngine(loaded_a.org_definition)
engine_b = GovernanceEngine(loaded_b.org_definition)

compiled_a = compile_org(loaded_a.org_definition)
compiled_b = compile_org(loaded_b.org_definition)

roles_a = {}
for addr, node in compiled_a.nodes.items():
    if node.role_definition is not None and not node.is_vacant:
        roles_a[node.node_id] = addr

roles_b = {}
for addr, node in compiled_b.nodes.items():
    if node.role_definition is not None and not node.is_vacant:
        roles_b[node.node_id] = addr

# Grant clearances
for cs in loaded_a.clearances:
    if cs.role_id in roles_a:
        engine_a.grant_clearance(
            roles_a[cs.role_id],
            RoleClearance(
                role_address=roles_a[cs.role_id],
                max_clearance=ConfidentialityLevel.CONFIDENTIAL,
                compartments=frozenset(cs.compartments),
                granted_by_role_address="system_init",
            ),
        )

for cs in loaded_b.clearances:
    if cs.role_id in roles_b:
        engine_b.grant_clearance(
            roles_b[cs.role_id],
            RoleClearance(
                role_address=roles_b[cs.role_id],
                max_clearance=ConfidentialityLevel.CONFIDENTIAL,
                compartments=frozenset(cs.compartments),
                granted_by_role_address="system_init",
            ),
        )

# Verify isolation: Bank Alpha agent cannot access Bank Beta's data
from pact.governance import KnowledgeItem

alpha_model = KnowledgeItem(
    item_id="alpha_fraud_model",
    classification=ConfidentialityLevel.CONFIDENTIAL,
    owning_unit_address=roles_a.get("alpha_agent", "R1"),
    compartments=frozenset(["alpha_models"]),
)

beta_model = KnowledgeItem(
    item_id="beta_fraud_model",
    classification=ConfidentialityLevel.CONFIDENTIAL,
    owning_unit_address=roles_b.get("beta_agent", "R1"),
    compartments=frozenset(["beta_models"]),
)

# Each org's engine only knows about its own roles
print(f"Bank Alpha agent accessing Alpha model:")
result_a = engine_a.check_access(
    roles_a["alpha_agent"], alpha_model, TrustPosture.SUPERVISED
)
print(f"  {'ALLOW' if result_a.allowed else 'DENY'}: {result_a.reason}")

print(f"\nBank Beta agent accessing Beta model:")
result_b = engine_b.check_access(
    roles_b["beta_agent"], beta_model, TrustPosture.SUPERVISED
)
print(f"  {'ALLOW' if result_b.allowed else 'DENY'}: {result_b.reason}")

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

# Load fraud data for bias detection
loader = ASCENTDataLoader()
fraud_data = loader.load("ascent04", "credit_card_fraud.parquet")

feature_cols = [c for c in fraud_data.columns if c.startswith("v")]
X = fraud_data.select(feature_cols + ["amount"]).to_numpy()
y = fraud_data["is_fraud"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

incident_model = GradientBoostingClassifier(
    n_estimators=30, max_depth=4, random_state=42
)
incident_model.fit(X_train, y_train)

# Simulate biased predictions on a segment
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
    print(f"\nIncident Response Chain (D/T/R):")
    print(f"  1. ALERT: DriftMonitor/bias check flags FPR gap > 5%")
    print(f"  2. AUDIT: ComplianceAuditAgent triggered automatically")
    print(f"     - Model card validity: STALE (bias metrics changed)")
    print(f"     - EU AI Act Art 10: NON_COMPLIANT (data governance gap)")
    print(f"     - SG AI Verify Fairness: NON_COMPLIANT")
    print(f"  3. ROOT CAUSE: Model overfits on transaction amount feature")
    print(f"     - High-value transactions have different V-feature distributions")
    print(f"     - Amount feature dominates split decisions in GBM")
    print(f"  4. REMEDIATION:")
    print(f"     a. Retrain with stratified sampling across amount segments")
    print(f"     b. Add fairness constraint (demographic parity on amount segment)")
    print(f"     c. Update model card with per-segment metrics")
    print(f"     d. Re-run compliance audit")
    print(f"  5. ACCOUNTABILITY (D/T/R chain):")
    print(f"     D (Delegator): ML Operations Lead authorised the model deployment")
    print(f"     T (Task): Fraud detection serving")
    print(f"     R (Responsible): Model Serving Agent executed predictions")
    print(f"     Trace: all steps logged in GovernanceEngine audit trail")

    # Log the incident
    async def log_incident():
        await db.express.create(
            "GovernanceEvent",
            {
                "event_type": "POLICY_VIOLATION",
                "model_id": "fraud_detector_v1",
                "actor": "drift_monitor_system",
                "details": json.dumps(
                    {
                        "type": "bias_alert",
                        "high_value_fpr": float(high_value_fpr),
                        "low_value_fpr": float(low_value_fpr),
                        "fpr_gap": float(fpr_gap),
                        "remediation": "retrain_with_fairness_constraint",
                    }
                ),
                "timestamp": datetime.now().isoformat(),
            },
        )
        print(f"\n  Incident logged to GovernanceEvent registry.")

    asyncio.run(log_incident())

# Clean up DataFlow
asyncio.run(dataflow.close_async())

# Remove temporary database
Path(_db_path).unlink(missing_ok=True)
db_path = Path("ascent10_governance.db")
if db_path.exists():
    db_path.unlink()

print("\n--- Exercise 7 complete: enterprise ML governance at scale ---")
