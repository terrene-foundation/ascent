# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT10 — Exercise 6: Regulatory Compliance Automation with PACT
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Build automated compliance checking -- define regulation
#   schemas as Signatures, implement a ComplianceAuditAgent, trigger
#   audits from DriftMonitor alerts, and deploy via Nexus.
#
# TASKS:
#   1. Define compliance schema using Signature for EU AI Act, MAS TRM,
#      SG AI Verify, PDPA
#   2. Implement ComplianceAuditAgent that inspects governance artefacts
#   3. Run audit on governed pipeline, generate remediation recommendations
#   4. Continuous compliance: DriftMonitor alert triggers audit agent
#   5. Deploy compliance checker as Nexus endpoint
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from kaizen.core import BaseAgent, Signature, InputField, OutputField
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
from kailash_nexus import Nexus

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Define compliance schemas using Signature
# ══════════════════════════════════════════════════════════════════════

print("=== Compliance Schemas (Regulation -> Assertion Predicates) ===\n")


class EUAIActComplianceSignature(Signature):
    """EU AI Act compliance check for high-risk AI systems."""

    # Inputs: artefacts to verify
    model_card_exists: bool = InputField(desc="Whether a model card is published")
    bias_audit_conducted: bool = InputField(desc="Whether bias audit has been run")
    human_oversight_defined: bool = InputField(
        desc="Whether human-in-the-loop is configured"
    )
    risk_category: str = InputField(
        desc="Risk category: MINIMAL, LIMITED, HIGH, UNACCEPTABLE"
    )
    audit_trail_active: bool = InputField(
        desc="Whether governance audit trail is recording"
    )
    drift_monitoring_active: bool = InputField(
        desc="Whether DriftMonitor is configured"
    )

    # Outputs: compliance verdict
    overall_status: str = OutputField(desc="COMPLIANT, PARTIAL, NON_COMPLIANT")
    article_verdicts: str = OutputField(desc="Per-article pass/fail with reasons")
    remediation_actions: str = OutputField(desc="Required actions for compliance")


class MASTRMComplianceSignature(Signature):
    """MAS Technology Risk Management compliance check."""

    governance_framework_active: bool = InputField(
        desc="PACT GovernanceEngine configured"
    )
    audit_trail_integrity: bool = InputField(desc="Audit trail passes integrity check")
    model_risk_documented: bool = InputField(
        desc="Model risks documented in model card"
    )
    change_management_logged: bool = InputField(
        desc="Model changes tracked in registry"
    )
    incident_response_defined: bool = InputField(
        desc="Incident response procedures exist"
    )

    overall_status: str = OutputField(desc="COMPLIANT, PARTIAL, NON_COMPLIANT")
    section_verdicts: str = OutputField(desc="Per-section pass/fail")
    remediation_actions: str = OutputField(desc="Required actions")


class SGAIVerifySignature(Signature):
    """Singapore AI Verify framework compliance check."""

    accountability_chain: bool = InputField(
        desc="D/T/R chain traces to human delegator"
    )
    fairness_audit: bool = InputField(
        desc="Bias metrics computed and thresholds defined"
    )
    transparency_docs: bool = InputField(desc="Model card and system card published")
    explainability_available: bool = InputField(
        desc="Feature importance or SHAP available"
    )
    data_governance: bool = InputField(desc="Dataset datasheet and lineage documented")

    overall_status: str = OutputField(desc="COMPLIANT, PARTIAL, NON_COMPLIANT")
    pillar_verdicts: str = OutputField(desc="Per-pillar pass/fail")
    remediation_actions: str = OutputField(desc="Required actions")


class PDPAComplianceSignature(Signature):
    """Singapore PDPA (Personal Data Protection Act) compliance check."""

    consent_documented: bool = InputField(desc="Data collection consent documented")
    data_minimisation: bool = InputField(desc="Only necessary data collected/used")
    access_controls: bool = InputField(desc="PACT clearance-based access enforced")
    retention_policy: bool = InputField(desc="Data retention limits defined")
    breach_notification: bool = InputField(
        desc="Breach detection and notification configured"
    )

    overall_status: str = OutputField(desc="COMPLIANT, PARTIAL, NON_COMPLIANT")
    obligation_verdicts: str = OutputField(desc="Per-obligation pass/fail")
    remediation_actions: str = OutputField(desc="Required actions")


# Define regulation schemas as structured assertion lists
REGULATION_SCHEMAS = {
    "EU AI Act": {
        "assertions": [
            (
                "Art 9 - Risk Management",
                "model_card_exists AND drift_monitoring_active",
            ),
            ("Art 10 - Data Governance", "bias_audit_conducted"),
            ("Art 12 - Record-keeping", "audit_trail_active"),
            ("Art 13 - Transparency", "model_card_exists"),
            ("Art 14 - Human Oversight", "human_oversight_defined"),
        ],
        "required_risk_categories": ["HIGH"],
    },
    "MAS TRM": {
        "assertions": [
            ("S7.2 - IT Risk Framework", "governance_framework_active"),
            ("S7.5 - Audit Trail", "audit_trail_integrity"),
            ("S8.1 - System Reliability", "drift_monitoring_active"),
        ],
    },
    "SG AI Verify": {
        "assertions": [
            ("Accountability", "accountability_chain"),
            ("Fairness", "fairness_audit"),
            ("Transparency", "transparency_docs"),
        ],
    },
    "PDPA": {
        "assertions": [
            ("Consent Obligation", "consent_documented"),
            ("Purpose Limitation", "data_minimisation"),
            ("Access Protection", "access_controls"),
            ("Retention Limit", "retention_policy"),
        ],
    },
}

for reg_name, schema in REGULATION_SCHEMAS.items():
    print(f"{reg_name}:")
    for assertion_name, predicate in schema["assertions"]:
        print(f"  {assertion_name}: {predicate}")
    print()


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Implement ComplianceAuditAgent
# ══════════════════════════════════════════════════════════════════════

print(f"{'=' * 70}")
print(f"=== ComplianceAuditAgent ===")
print(f"{'=' * 70}\n")


def run_compliance_audit(artefacts: dict) -> dict:
    """Run compliance audit against all regulation schemas.

    Args:
        artefacts: Dict of artefact states (name -> bool or value).

    Returns:
        Structured audit report.
    """
    report = {
        "audit_id": f"audit-{uuid.uuid4().hex[:8]}",
        "timestamp": datetime.now().isoformat(),
        "regulations": {},
        "overall_status": "COMPLIANT",
    }

    for reg_name, schema in REGULATION_SCHEMAS.items():
        reg_result = {
            "assertions": [],
            "status": "COMPLIANT",
            "remediation": [],
        }

        for assertion_name, predicate in schema["assertions"]:
            # Parse predicate (simple AND/OR logic)
            terms = [t.strip() for t in predicate.replace("AND", ",").split(",")]
            all_pass = all(artefacts.get(term, False) for term in terms)

            reg_result["assertions"].append(
                {
                    "name": assertion_name,
                    "predicate": predicate,
                    "status": "PASS" if all_pass else "FAIL",
                }
            )

            if not all_pass:
                reg_result["status"] = "NON_COMPLIANT"
                missing = [t for t in terms if not artefacts.get(t, False)]
                reg_result["remediation"].append(
                    f"{assertion_name}: missing {', '.join(missing)}"
                )

        report["regulations"][reg_name] = reg_result
        if reg_result["status"] != "COMPLIANT":
            report["overall_status"] = "NON_COMPLIANT"

    return report


# Simulate artefact state for a governed pipeline
pipeline_artefacts = {
    # EU AI Act
    "model_card_exists": True,
    "bias_audit_conducted": True,
    "human_oversight_defined": True,
    "risk_category": "HIGH",
    "audit_trail_active": True,
    "drift_monitoring_active": True,
    # MAS TRM
    "governance_framework_active": True,
    "audit_trail_integrity": True,
    "change_management_logged": True,
    "incident_response_defined": False,  # Gap
    # SG AI Verify
    "accountability_chain": True,
    "fairness_audit": True,
    "transparency_docs": True,
    "explainability_available": False,  # Gap
    # PDPA
    "consent_documented": True,
    "data_minimisation": True,
    "access_controls": True,
    "retention_policy": False,  # Gap
    "breach_notification": True,
}


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Run audit on governed pipeline, generate remediation
# ══════════════════════════════════════════════════════════════════════

audit_report = run_compliance_audit(pipeline_artefacts)

print(f"Audit ID: {audit_report['audit_id']}")
print(f"Timestamp: {audit_report['timestamp']}")
print(f"Overall Status: {audit_report['overall_status']}\n")

for reg_name, reg_result in audit_report["regulations"].items():
    print(f"{reg_name}: {reg_result['status']}")
    for assertion in reg_result["assertions"]:
        status_marker = "PASS" if assertion["status"] == "PASS" else "FAIL"
        print(f"  [{status_marker}] {assertion['name']}")
    if reg_result["remediation"]:
        print(f"  Remediation required:")
        for action in reg_result["remediation"]:
            print(f"    - {action}")
    print()

# Generate structured remediation plan
print(f"=== Remediation Plan ===")
all_remediation = []
for reg_name, reg_result in audit_report["regulations"].items():
    for action in reg_result["remediation"]:
        all_remediation.append({"regulation": reg_name, "action": action})

if all_remediation:
    for i, item in enumerate(all_remediation, 1):
        print(f"  {i}. [{item['regulation']}] {item['action']}")
else:
    print(f"  No remediation actions required. All checks PASS.")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Continuous compliance — DriftMonitor triggers audit
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print(f"=== Continuous Compliance Monitoring ===")
print(f"{'=' * 70}\n")

# Set up DriftMonitor
rng = np.random.default_rng(42)
reference_data = rng.normal(0, 1, (1000, 10))
feature_names = [f"feature_{i}" for i in range(10)]

monitor = DriftMonitor(
    reference_data=reference_data,
    feature_names=feature_names,
    psi_threshold=0.2,
)


def on_drift_detected(drift_report, pipeline_artefacts: dict) -> dict:
    """Callback: when DriftMonitor detects drift, trigger compliance audit.

    This models the continuous compliance loop:
    1. DriftMonitor checks PSI on production data
    2. If PSI > threshold, drift is flagged
    3. Compliance audit agent is triggered automatically
    4. Audit agent checks if model card is still valid
    5. Generates remediation if drift invalidates compliance
    """
    # Update artefacts based on drift
    updated_artefacts = pipeline_artefacts.copy()
    if drift_report.has_drift:
        # Drift invalidates the current bias audit (model may be biased on new data)
        updated_artefacts["bias_audit_conducted"] = False
        # Drift means model card performance metrics may be stale
        updated_artefacts["model_card_exists"] = True  # Card exists but may be outdated

    return run_compliance_audit(updated_artefacts)


# Simulate stable traffic (no drift)
stable_data = rng.normal(0, 1, (200, 10))
stable_report = monitor.check_drift(stable_data)
print(f"Week 1 (stable): drift={stable_report.has_drift}")
if not stable_report.has_drift:
    print(f"  No compliance audit triggered.\n")

# Simulate drifted traffic (triggers audit)
drifted_data = rng.normal(0.8, 1.5, (200, 10))
drift_report = monitor.check_drift(drifted_data)
print(f"Week 2 (drift): drift={drift_report.has_drift}")

if drift_report.has_drift:
    print(f"  Drift detected -> triggering compliance audit...\n")
    drift_audit = on_drift_detected(drift_report, pipeline_artefacts)
    print(f"  Audit Status: {drift_audit['overall_status']}")
    for reg_name, reg_result in drift_audit["regulations"].items():
        if reg_result["status"] != "COMPLIANT":
            print(f"  {reg_name}: {reg_result['status']}")
            for action in reg_result["remediation"]:
                print(f"    - {action}")

print(f"\nContinuous compliance loop:")
print(f"  DriftMonitor (PSI > 0.2) -> ComplianceAuditAgent -> Remediation Plan")
print(f"  This runs automatically; humans review remediation at approval gates.")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Deploy compliance checker as Nexus endpoint
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print(f"=== Nexus Compliance Endpoint ===")
print(f"{'=' * 70}\n")

# Define the compliance check as a Nexus-deployable handler
app = Nexus()


async def compliance_check_handler(model_id: str) -> dict:
    """Nexus endpoint: submit model_id, receive compliance report.

    In production, this handler:
    1. Looks up model metadata in ModelRegistry
    2. Checks governance state via GovernanceEngine
    3. Queries DriftMonitor for current drift status
    4. Runs compliance audit against all regulation schemas
    5. Returns structured JSON report
    """
    # Simulate model lookup
    artefact_state = pipeline_artefacts.copy()

    # Run audit
    report = run_compliance_audit(artefact_state)
    report["model_id"] = model_id

    return report


# Demonstrate the endpoint
print(f"Nexus endpoint: POST /compliance/check")
print(f"  Input: {{'model_id': 'credit_fraud_detector_v1'}}")

demo_report = asyncio.run(compliance_check_handler("credit_fraud_detector_v1"))

print(f"  Output:")
print(f"    audit_id: {demo_report['audit_id']}")
print(f"    model_id: {demo_report['model_id']}")
print(f"    overall_status: {demo_report['overall_status']}")
print(f"    regulations checked: {list(demo_report['regulations'].keys())}")

# PACT governance for the compliance endpoint itself
compliance_org_yaml = """
org_id: compliance_service
name: "Compliance Audit Service"

departments:
  - id: audit
    name: "Audit Operations"

roles:
  - id: audit_admin
    name: "Audit Administrator"
    is_primary_for_unit: audit

  - id: audit_agent
    name: "Compliance Audit Agent"
    reports_to: audit_admin
    agent: true

clearances:
  - role: audit_admin
    level: secret
    compartments: [audit_reports, model_metadata, governance_state]
  - role: audit_agent
    level: confidential
    compartments: [audit_reports, model_metadata]
"""

comp_file = Path(tempfile.mktemp(suffix=".yaml"))
comp_file.write_text(compliance_org_yaml)
comp_loaded = load_org_yaml(str(comp_file))
comp_engine = GovernanceEngine(comp_loaded.org_definition)

comp_compiled = compile_org(comp_loaded.org_definition)
comp_roles = {}
for addr, node in comp_compiled.nodes.items():
    if node.role_definition is not None and not node.is_vacant:
        comp_roles[node.node_id] = addr

level_map = {
    "confidential": ConfidentialityLevel.CONFIDENTIAL,
    "secret": ConfidentialityLevel.SECRET,
}
for cs in comp_loaded.clearances:
    if cs.role_id in comp_roles:
        clearance = RoleClearance(
            role_address=comp_roles[cs.role_id],
            max_clearance=level_map[cs.level],
            compartments=frozenset(cs.compartments),
            granted_by_role_address="system_init",
        )
        comp_engine.grant_clearance(comp_roles[cs.role_id], clearance)

# Set audit agent envelope
audit_env = RoleEnvelope(
    id=f"env-{uuid.uuid4().hex[:8]}",
    defining_role_address=comp_roles["audit_admin"],
    target_role_address=comp_roles["audit_agent"],
    envelope=ConstraintEnvelopeConfig(
        id="audit-agent-envelope",
        operational=OperationalConstraintConfig(
            allowed_actions=[
                "run_audit",
                "read_model_card",
                "read_drift_status",
                "generate_report",
            ],
        ),
        temporal=TemporalConstraintConfig(),
        data_access=DataAccessConstraintConfig(),
        communication=CommunicationConstraintConfig(),
    ),
)
comp_engine.set_role_envelope(audit_env)

governed_audit = PactGovernedAgent(
    engine=comp_engine,
    role_address=comp_roles["audit_agent"],
)

print(f"\nCompliance agent governance:")
print(f"  Clearance: {governed_audit.context.effective_clearance_level}")
print(f"  Actions: {sorted(governed_audit.context.allowed_actions)}")

# Verify the audit agent itself is governed
for action in ["run_audit", "generate_report", "modify_model", "delete_audit_log"]:
    verdict = comp_engine.verify_action(comp_roles["audit_agent"], action)
    status = "ALLOW" if verdict.level == "auto_approved" else "BLOCK"
    print(f"  audit_agent -> {action}: {status}")

integrity = comp_engine.verify_audit_integrity()
print(f"\nAudit integrity: {'VALID' if integrity else 'BROKEN'}")

comp_file.unlink()

print(
    "\n--- Exercise 6 complete: regulatory compliance automation with PACT + Nexus ---"
)
