# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT10 — Exercise 6: AI Governance with PACT
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Define an organization in YAML with D/T/R roles, compile
#   with GovernanceEngine, enforce operating envelopes, and verify access.
#
# TASKS:
#   1. Write YAML organization definition (departments, roles, clearances)
#   2. Compile with GovernanceEngine
#   3. Define operating envelopes (clearance levels, knowledge access)
#   4. Test access control decisions
#   5. Generate governance report with decision explanations
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os
import tempfile

from pact import (
    ConfidentialityLevel,
    GovernanceEngine,
    KnowledgeItem,
    RoleClearance,
    TrustPostureLevel,
    VettingStatus,
    compile_org,
    load_org_yaml,
)

from shared.kailash_helpers import setup_environment

setup_environment()


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Write YAML organization definition
# ══════════════════════════════════════════════════════════════════════

# PACT YAML requires: org_id, name, departments, roles, clearances
# Roles use reports_to for hierarchy, heads for department leadership
org_yaml = """
# Singapore FinTech AI Organisation — PACT Governance Definition
# D/T/R: Delegator / Task / Responsible
# Every agent action must trace back to a human Delegator.

org_id: sg_fintech_ai
name: SG FinTech AI Division

departments:
  - id: ml_eng
    name: ML Engineering
  - id: risk_compliance
    name: Risk and Compliance
  - id: customer_intel
    name: Customer Intelligence

roles:
  # Department heads (Delegators)
  - id: chief_ml_officer
    name: Chief ML Officer
    heads: ml_eng
  - id: chief_risk_officer
    name: Chief Risk Officer
    heads: risk_compliance
  - id: vp_customer
    name: VP Customer
    heads: customer_intel

  # ML Engineering roles (Responsible agents)
  - id: data_analyst
    name: Data Analyst
    reports_to: chief_ml_officer
  - id: model_trainer
    name: Model Trainer
    reports_to: chief_ml_officer
  - id: model_deployer
    name: Model Deployer
    reports_to: chief_ml_officer

  # Risk & Compliance roles
  - id: risk_assessor
    name: Risk Assessor
    reports_to: chief_risk_officer
  - id: bias_checker
    name: Bias Checker
    reports_to: chief_risk_officer

  # Customer Intelligence roles
  - id: customer_agent
    name: Customer Agent
    reports_to: vp_customer

clearances:
  # Heads get higher clearance
  - role: chief_ml_officer
    level: restricted
  - role: chief_risk_officer
    level: restricted
  - role: vp_customer
    level: confidential

  # Workers get clearance appropriate to their role
  - role: model_trainer
    level: confidential
  - role: model_deployer
    level: confidential
  - role: data_analyst
    level: confidential
  - role: risk_assessor
    level: restricted
  - role: bias_checker
    level: confidential
  - role: customer_agent
    level: public
"""

# Write YAML to temp file
org_yaml_path = os.path.join(tempfile.gettempdir(), "sg_fintech_org.yaml")
with open(org_yaml_path, "w") as f:
    f.write(org_yaml)

print("=== Organization Definition ===")
print("Organization: SG FinTech AI Division")
print("Departments: ML Engineering, Risk & Compliance, Customer Intelligence")
print("Roles: 9 (3 heads + 6 workers)")
print("D/T/R chains: heads delegate to their reports")
print(f"YAML written to: {org_yaml_path}")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Compile with GovernanceEngine
# ══════════════════════════════════════════════════════════════════════

# Step 1: load_org_yaml parses and validates the YAML
loaded = load_org_yaml(org_yaml_path)

# Step 2: compile_org validates structural integrity (no cycles, no dangling refs)
compiled = compile_org(loaded.org_definition)

# Step 3: GovernanceEngine wraps the org definition with access control
engine = GovernanceEngine(loaded.org_definition)

# Build role_id -> address mapping from compiled org
LEVEL_MAP = {
    "public": ConfidentialityLevel.PUBLIC,
    "restricted": ConfidentialityLevel.RESTRICTED,
    "confidential": ConfidentialityLevel.CONFIDENTIAL,
    "secret": ConfidentialityLevel.SECRET,
    "top_secret": ConfidentialityLevel.TOP_SECRET,
}

role_id_to_addr: dict[str, str] = {}
for addr, node in compiled.nodes.items():
    if node.node_id:
        role_id_to_addr[node.node_id] = addr

# Apply clearances from the YAML
for cs in loaded.clearances:
    addr = role_id_to_addr.get(cs.role_id)
    if addr:
        clearance = RoleClearance(
            role_address=addr,
            max_clearance=LEVEL_MAP[cs.level],
            compartments=frozenset(cs.compartments),
            granted_by_role_address=addr,
            vetting_status=VettingStatus.ACTIVE,
            nda_signed=cs.nda_signed,
        )
        engine.grant_clearance(addr, clearance)

print(f"\n=== Compiled Organization ===")
print(f"Org ID: {compiled.org_id}")
print(f"Total nodes: {len(compiled.nodes)}")
print(f"Clearances granted: {len(loaded.clearances)}")
print("Compilation validates:")
print("  - Every role has a valid reports_to chain (no dangling references)")
print("  - No circular delegation (A reports_to B reports_to A)")
print("  - Department heads are properly linked")
print("  - All role IDs are unique")

# Show the org tree
print("\nOrganization tree:")
for addr, node in sorted(compiled.nodes.items()):
    indent = "  " * addr.count("-")
    print(f"  {indent}{addr}: {node.name} ({node.node_type.value})")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Define operating envelopes (clearance-based access)
# ══════════════════════════════════════════════════════════════════════

print("\n=== Operating Envelopes (Clearance Levels) ===")
print("PACT uses confidentiality levels to control knowledge access:")
print(f"  PUBLIC < RESTRICTED < CONFIDENTIAL < SECRET < TOP_SECRET")
print()
print("  data_analyst:")
print("    Clearance: confidential")
print("    Can access: public + restricted + confidential data")
print()
print("  model_trainer:")
print("    Clearance: confidential")
print("    Can access: public + restricted + confidential data")
print()
print("  risk_assessor:")
print("    Clearance: restricted")
print("    Can access: public + restricted data (NOT confidential)")
print()
print("  customer_agent:")
print("    Clearance: public")
print("    Can access: public data only")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Test access control decisions
# ══════════════════════════════════════════════════════════════════════

print("\n=== Access Control Tests ===")
print("PACT access = clearance check + posture ceiling + structural path")
print("Posture ceilings: SUPERVISED->restricted, SHARED_PLANNING->confidential,")
print("  DELEGATED->top_secret (no ceiling)")

# Knowledge items owned by departments so structural path exists within dept
ml_public = KnowledgeItem(
    item_id="ml_docs",
    classification=ConfidentialityLevel.PUBLIC,
    owning_unit_address="D1",  # ML Engineering
)
ml_restricted = KnowledgeItem(
    item_id="ml_metrics",
    classification=ConfidentialityLevel.RESTRICTED,
    owning_unit_address="D1",  # ML Engineering
)
ml_confidential = KnowledgeItem(
    item_id="training_data",
    classification=ConfidentialityLevel.CONFIDENTIAL,
    owning_unit_address="D1",  # ML Engineering
)
ml_secret = KnowledgeItem(
    item_id="secret_model",
    classification=ConfidentialityLevel.SECRET,
    owning_unit_address="D1",  # ML Engineering
)
risk_restricted = KnowledgeItem(
    item_id="audit_log",
    classification=ConfidentialityLevel.RESTRICTED,
    owning_unit_address="D2",  # Risk & Compliance
)
cust_public = KnowledgeItem(
    item_id="public_faq",
    classification=ConfidentialityLevel.PUBLIC,
    owning_unit_address="D3",  # Customer Intelligence
)

# Test cases: (role_id, knowledge_item, posture, expected_allowed, description)
test_cases = [
    # Within-department access at DELEGATED posture (no ceiling)
    (
        "model_trainer",
        ml_confidential,
        TrustPostureLevel.DELEGATED,
        True,
        "Confidential role, own-dept confidential data, delegated posture",
    ),
    (
        "model_trainer",
        ml_secret,
        TrustPostureLevel.DELEGATED,
        False,
        "Confidential role, secret data exceeds clearance",
    ),
    (
        "chief_ml_officer",
        ml_restricted,
        TrustPostureLevel.DELEGATED,
        True,
        "Restricted role, own-dept restricted data, delegated posture",
    ),
    # Posture ceiling effect
    (
        "model_trainer",
        ml_confidential,
        TrustPostureLevel.SUPERVISED,
        False,
        "Confidential role but SUPERVISED caps at restricted",
    ),
    (
        "model_trainer",
        ml_restricted,
        TrustPostureLevel.SUPERVISED,
        True,
        "Confidential role, restricted data, within SUPERVISED ceiling",
    ),
    # Customer agent (public clearance)
    (
        "customer_agent",
        cust_public,
        TrustPostureLevel.DELEGATED,
        True,
        "Public role, public data in own dept",
    ),
    # Cross-department access (no KSP bridge = denied)
    (
        "risk_assessor",
        ml_restricted,
        TrustPostureLevel.DELEGATED,
        False,
        "Risk role accessing ML dept data without KSP bridge",
    ),
    # Same-department access
    (
        "risk_assessor",
        risk_restricted,
        TrustPostureLevel.DELEGATED,
        True,
        "Restricted role, own-dept restricted data",
    ),
]

for role_id, ki, posture, expected, desc in test_cases:
    role_addr = role_id_to_addr[role_id]
    decision = engine.check_access(role_addr, ki, posture)
    status = "ALLOWED" if decision.allowed else "DENIED"
    match_str = "OK" if decision.allowed == expected else "UNEXPECTED"
    print(
        f"  [{match_str}] {role_id} -> {ki.item_id} ({ki.classification.value}): {status}"
    )
    print(f"      {desc}")
    if not decision.allowed:
        print(f"      Reason: {decision.reason}")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Generate governance report with decision explanations
# ══════════════════════════════════════════════════════════════════════

print("\n=== Governance Report ===")

# Demonstrate verify_action for action-level governance
trainer_addr = role_id_to_addr["model_trainer"]
verdict = engine.verify_action(trainer_addr, "train_model")

print(f"Action verification for model_trainer -> train_model:")
print(f"  Level: {verdict.level}")
print(f"  Reason: {verdict.reason}")
print(f"  Role address: {verdict.role_address}")

# Explain a full decision chain
print(f"\nDecision trace for model_trainer -> access training_data:")
print(f"  1. Role: model_trainer (address: {trainer_addr})")
print(f"  2. Clearance: confidential (granted via YAML clearances)")
print(f"  3. Knowledge item: training_data (classification: confidential)")
print(f"  4. Access check: confidential >= confidential -> ALLOWED")
print(f"  5. Audit: logged to immutable audit chain")

# Verify audit integrity
integrity_ok, integrity_msg = engine.verify_audit_integrity()
print(f"\nAudit integrity: {'VALID' if integrity_ok else 'INVALID'}")
if integrity_msg:
    print(f"  Detail: {integrity_msg}")

print(f"\nD/T/R Accountability Grammar:")
print(f"  D (Delegator): chief_ml_officer — authorizes the task")
print(f"  T (Task): model_training — the bounded scope of work")
print(f"  R (Responsible): model_trainer — executes within envelope")
print(f"  If model_trainer exceeds clearance -> GovernanceEngine blocks")
print(f"  If task fails -> accountability traces to chief_ml_officer")

print(f"\nRegulatory mapping:")
print(f"  EU AI Act: clearance levels satisfy Art. 9 (risk management)")
print(f"  AI Verify: D/T/R chains satisfy accountability principle")
print(f"  MAS TRM: audit trails satisfy record-keeping requirements")

print(
    "\n=== Exercise 6 complete — PACT governance with D/T/R and clearance-based access ==="
)
