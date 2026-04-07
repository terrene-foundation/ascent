# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT06 — Exercise 5: AI Governance with PACT
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Define a realistic organization in YAML, compile it, and
#   create a GovernanceEngine. Grant clearances, set envelopes, and
#   verify action decisions.
#
# TASKS:
#   1. Define organization structure in YAML (departments, teams, roles)
#   2. Load and compile organization with load_org_yaml / compile_org
#   3. Create GovernanceEngine and grant clearances
#   4. Set operating envelopes and verify actions
#   5. Demonstrate frozen GovernanceContext
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from kailash.trust import ConfidentialityLevel
from pact import GovernanceEngine, GovernanceContext
from pact import compile_org, load_org_yaml
from pact.governance import (
    ClearanceSpec,
    RoleClearance,
    RoleEnvelope,
)
from kailash.trust.pact.config import (
    ConstraintEnvelopeConfig,
    OperationalConstraintConfig,
    TemporalConstraintConfig,
    DataAccessConstraintConfig,
    CommunicationConstraintConfig,
)

from shared.kailash_helpers import setup_environment

setup_environment()


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Define organization in YAML
# ══════════════════════════════════════════════════════════════════════
#
# PACT YAML schema (top-level keys):
#   org_id (str, required) — unique org identifier
#   name (str, required) — human-readable org name
#   departments (list of {id, name})
#   teams (list of {id, name})
#   roles (list of {id, name, reports_to?, is_primary_for_unit?, agent?})
#   clearances (list of {role, level, compartments?, nda_signed?})
#   envelopes (list of {target, defined_by, financial?, operational?})
#
# Clearance levels: public, restricted, confidential, secret, top_secret

org_yaml = """
org_id: ascent_credit_bureau
name: "ASCENT Credit Bureau"

departments:
  - id: data_science
    name: "Data Science"
  - id: risk
    name: "Risk Management"
  - id: operations
    name: "Operations"

teams:
  - id: modeling
    name: "Modeling Team"
  - id: mlops
    name: "MLOps Team"
  - id: model_validation
    name: "Model Validation"
  - id: serving
    name: "Serving"

roles:
  # Data Science department
  - id: senior_data_scientist
    name: "Senior Data Scientist"
    is_primary_for_unit: data_science
  - id: junior_data_scientist
    name: "Junior Data Scientist"
    reports_to: senior_data_scientist
  - id: ml_engineer
    name: "ML Engineer"
    is_primary_for_unit: mlops

  # Risk department
  - id: model_validator
    name: "Model Validator"
    is_primary_for_unit: risk
  - id: compliance_officer
    name: "Compliance Officer"
    reports_to: model_validator

  # Operations department
  - id: sre
    name: "Site Reliability Engineer"
    is_primary_for_unit: operations
  - id: customer_service
    name: "Customer Service Agent"
    reports_to: sre
    agent: true

clearances:
  - role: senior_data_scientist
    level: secret
    compartments: [credit_data, feature_store]
  - role: junior_data_scientist
    level: confidential
    compartments: [credit_data]
  - role: ml_engineer
    level: secret
    compartments: [credit_data, feature_store, model_artifacts]
  - role: model_validator
    level: top_secret
    compartments: [credit_data, audit_logs, model_artifacts]
    nda_signed: true
  - role: compliance_officer
    level: confidential
    compartments: [audit_logs]
  - role: sre
    level: secret
    compartments: [production_logs, model_artifacts]
  - role: customer_service
    level: restricted
    compartments: [credit_decisions]
"""

# Write to temp file
org_file = Path(tempfile.mktemp(suffix=".yaml"))
org_file.write_text(org_yaml)
print(f"=== Organization YAML ===")
print(f"Org ID: ascent_credit_bureau")
print(f"Departments: 3 (data_science, risk, operations)")
print(f"Teams: 4 (modeling, mlops, model_validation, serving)")
print(f"Roles: 7 unique roles")
print(f"Clearances: 7 (levels restricted → top_secret)")
print(f"Written to: {org_file}")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Load and compile organization
# ══════════════════════════════════════════════════════════════════════

loaded_org = load_org_yaml(str(org_file))

print(f"\n=== LoadedOrg ===")
print(f"Org ID: {loaded_org.org_definition.org_id}")
print(f"Name: {loaded_org.org_definition.name}")
print(f"Departments: {len(loaded_org.org_definition.departments)}")
print(f"Teams: {len(loaded_org.org_definition.teams)}")
print(f"Roles: {len(loaded_org.org_definition.roles)}")
print(f"Clearance specs: {len(loaded_org.clearances)}")

compiled = compile_org(loaded_org.org_definition)

print(f"\n=== CompiledOrg (D/T/R Addresses) ===")
print(f"Org ID: {compiled.org_id}")
print(f"Total nodes: {len(compiled.nodes)}")

# Build role_id → address lookup
role_addr = {}
for addr, node in compiled.nodes.items():
    if node.role_definition is not None and not node.is_vacant:
        role_addr[node.node_id] = addr
        print(f"  {addr:<8} {node.node_type.value} — {node.name} (id={node.node_id})")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Create GovernanceEngine and grant clearances
# ══════════════════════════════════════════════════════════════════════

engine = GovernanceEngine(loaded_org.org_definition)

# Grant clearances from the YAML-parsed specs
# ClearanceSpec has: role_id, level, compartments, nda_signed
# GovernanceEngine.grant_clearance() takes (role_address, RoleClearance)
level_map = {
    "public": ConfidentialityLevel.PUBLIC,
    "restricted": ConfidentialityLevel.RESTRICTED,
    "confidential": ConfidentialityLevel.CONFIDENTIAL,
    "secret": ConfidentialityLevel.SECRET,
    "top_secret": ConfidentialityLevel.TOP_SECRET,
}

print(f"\n=== Clearances Granted ===")
for cs in loaded_org.clearances:
    if cs.role_id in role_addr:
        addr = role_addr[cs.role_id]
        clearance = RoleClearance(
            role_address=addr,
            max_clearance=level_map[cs.level],
            compartments=frozenset(cs.compartments),
            granted_by_role_address="system_init",
            nda_signed=cs.nda_signed,
        )
        engine.grant_clearance(addr, clearance)
        print(
            f"  {cs.role_id:<25} → {cs.level:<12} compartments={sorted(cs.compartments)}"
        )

# Set operating envelopes programmatically
print(f"\n=== Envelopes Set ===")
envelope_defs = [
    (
        "senior_data_scientist",
        "model_validator",
        ["train_model", "evaluate_model", "profile_data"],
    ),
    (
        "junior_data_scientist",
        "senior_data_scientist",
        ["profile_data", "evaluate_model"],
    ),
    (
        "ml_engineer",
        "model_validator",
        ["train_model", "evaluate_model", "deploy_model", "monitor_drift"],
    ),
    ("customer_service", "sre", ["query_prediction"]),
]

for target_id, definer_id, actions in envelope_defs:
    if target_id in role_addr and definer_id in role_addr:
        import uuid

        envelope = RoleEnvelope(
            id=f"env-{uuid.uuid4().hex[:8]}",
            defining_role_address=role_addr[definer_id],
            target_role_address=role_addr[target_id],
            envelope=ConstraintEnvelopeConfig(
                id=f"constraint-{target_id}",
                operational=OperationalConstraintConfig(
                    allowed_actions=actions,
                ),
                temporal=TemporalConstraintConfig(),
                data_access=DataAccessConstraintConfig(),
                communication=CommunicationConstraintConfig(),
            ),
        )
        engine.set_role_envelope(envelope)
        print(f"  {target_id:<25} defined_by={definer_id:<25} actions={actions}")

print(f"\n=== GovernanceEngine Ready ===")
print(f"Organization: {loaded_org.org_definition.name}")
print(f"Enforcement mode: fail-closed (deny on error)")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Verify actions
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Action Verification ===")
test_cases = [
    ("senior_data_scientist", "train_model", "Senior DS can train models"),
    ("junior_data_scientist", "train_model", "Junior DS cannot train models"),
    ("ml_engineer", "deploy_model", "ML Engineer can deploy models"),
    ("customer_service", "train_model", "Customer service cannot train"),
    ("customer_service", "query_prediction", "Customer service can query"),
]

for role_id, action, description in test_cases:
    if role_id in role_addr:
        verdict = engine.verify_action(role_addr[role_id], action)
        print(f"  {description}")
        print(f"    {role_id} → {action}: {verdict.level} ({verdict.reason})")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Frozen GovernanceContext
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Governance Contexts ===")
for role_id in ["senior_data_scientist", "ml_engineer", "customer_service"]:
    if role_id in role_addr:
        ctx = engine.get_context(role_addr[role_id])
        print(f"\n  {role_id}:")
        print(f"    Address: {ctx.role_address}")
        print(f"    Posture: {ctx.posture}")
        print(f"    Clearance: {ctx.effective_clearance_level}")
        print(f"    Compartments: {ctx.compartments}")
        print(f"    Allowed actions: {ctx.allowed_actions}")

print(f"\n=== Frozen Context Properties ===")
print(
    """
GovernanceContext is a frozen dataclass:
  - Agents receive a SNAPSHOT, NOT the engine itself
  - Agents cannot call grant_clearance() or set_role_envelope()
  - Context is immutable — modifications raise FrozenInstanceError

WHY frozen?
  1. Agents cannot escalate their own privileges
  2. Context is immutable proof of what was authorized
  3. Monotonic tightening is guaranteed (child <= parent)
  4. AuditChain can verify context was not tampered with

Attack prevention:
  - Agent modifies its clearance: FrozenInstanceError
  - Agent accesses data outside scope: fail-closed DENY
  - Governance engine error: fail-closed DENY (not ALLOW)
"""
)

# Demonstrate immutability
if "senior_data_scientist" in role_addr:
    ctx = engine.get_context(role_addr["senior_data_scientist"])
    try:
        ctx.role_address = "hacked"
        print("ERROR: Should have raised!")
    except (AttributeError, TypeError) as e:
        print(f"Attempted modification blocked: {type(e).__name__}")
        print(f"  GovernanceContext is immutable — agents cannot self-modify")

# Clean up
org_file.unlink()

print("\n✓ Exercise 5 complete — PACT governance setup with YAML org definition")
