# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT10 — Exercise 1: Multi-Organisation Governance
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Build cross-organisation governance using PACT — federated
#   GovernanceEngines for a regulator and a bank, with KSP bridges,
#   delegation chaining, and breach simulation.
#
# TASKS:
#   1. Define two PACT organisations (MAS regulator + bank), compile both
#   2. Implement a KSP bridge with permitted/restricted data sharing
#   3. Model regulatory inspection: cross-org access control
#   4. Implement delegation chaining: CEO -> compliance -> AI agent
#   5. Breach simulation: agent exceeds clearance, verify block + audit
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

import polars as pl

from kailash.trust import ConfidentialityLevel, TrustPosture
from pact import GovernanceEngine, GovernanceContext, PactGovernedAgent
from pact import compile_org, load_org_yaml
from pact.governance import (
    RoleClearance,
    RoleEnvelope,
    KnowledgeItem,
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
# TASK 1: Define two PACT organisations — MAS and DBS Bank
# ══════════════════════════════════════════════════════════════════════

# --- MAS (Monetary Authority of Singapore) ---
mas_yaml = """
org_id: mas_singapore
name: "Monetary Authority of Singapore"

departments:
  - id: ai_supervision
    name: "AI & Technology Supervision"
  - id: model_risk
    name: "Model Risk Division"

teams:
  - id: inspection_team
    name: "On-Site Inspection"
  - id: review_team
    name: "Model Review"

roles:
  - id: chief_ai_officer
    name: "Chief AI Officer"
    is_primary_for_unit: ai_supervision

  - id: lead_inspector
    name: "Lead Inspector"
    reports_to: chief_ai_officer
    agent: false

  - id: ai_inspector
    name: "AI Inspector"
    reports_to: lead_inspector
    agent: false

  - id: model_reviewer
    name: "Model Risk Reviewer"
    is_primary_for_unit: model_risk
    agent: false

clearances:
  - role: chief_ai_officer
    level: top_secret
    compartments: [regulatory_data, model_cards, audit_trails, inspection_reports]
  - role: lead_inspector
    level: secret
    compartments: [regulatory_data, model_cards, audit_trails]
  - role: ai_inspector
    level: confidential
    compartments: [model_cards, audit_trails]
  - role: model_reviewer
    level: secret
    compartments: [model_cards, audit_trails, inspection_reports]
"""

# --- DBS Bank ---
bank_yaml = """
org_id: dbs_bank
name: "DBS Bank"

departments:
  - id: ai_centre
    name: "AI Centre of Excellence"
  - id: compliance
    name: "Compliance"

teams:
  - id: ml_ops
    name: "ML Operations"
  - id: compliance_ai
    name: "AI Compliance"

roles:
  - id: bank_ceo
    name: "Chief Executive Officer"
    is_primary_for_unit: ai_centre

  - id: compliance_officer
    name: "Chief Compliance Officer"
    reports_to: bank_ceo

  - id: ml_lead
    name: "ML Engineering Lead"
    reports_to: bank_ceo
    agent: false

  - id: credit_agent
    name: "Credit Underwriting Agent"
    reports_to: ml_lead
    agent: true

  - id: compliance_agent
    name: "Compliance Monitoring Agent"
    reports_to: compliance_officer
    agent: true

clearances:
  - role: bank_ceo
    level: top_secret
    compartments: [customer_data, model_internals, audit_trails, production_logs]
  - role: compliance_officer
    level: secret
    compartments: [audit_trails, model_cards, production_logs, regulatory_data]
  - role: ml_lead
    level: secret
    compartments: [customer_data, model_internals, model_cards]
  - role: credit_agent
    level: confidential
    compartments: [customer_data, model_cards]
  - role: compliance_agent
    level: confidential
    compartments: [audit_trails, model_cards]
"""

# Compile both organisations
mas_file = Path(tempfile.mktemp(suffix=".yaml"))
mas_file.write_text(mas_yaml)
bank_file = Path(tempfile.mktemp(suffix=".yaml"))
bank_file.write_text(bank_yaml)

mas_loaded = load_org_yaml(str(mas_file))
bank_loaded = load_org_yaml(str(bank_file))

mas_compiled = compile_org(mas_loaded.org_definition)
bank_compiled = compile_org(bank_loaded.org_definition)

# Build role_id -> address lookups
mas_roles = {}
for addr, node in mas_compiled.nodes.items():
    if node.role_definition is not None and not node.is_vacant:
        mas_roles[node.node_id] = addr

bank_roles = {}
for addr, node in bank_compiled.nodes.items():
    if node.role_definition is not None and not node.is_vacant:
        bank_roles[node.node_id] = addr

print("=== Organisation 1: MAS ===")
print(f"Org ID: {mas_compiled.org_id}")
print(f"Nodes: {len(mas_compiled.nodes)}")
for role_id, addr in mas_roles.items():
    print(f"  {role_id:<25} -> {addr}")

print(f"\n=== Organisation 2: DBS Bank ===")
print(f"Org ID: {bank_compiled.org_id}")
print(f"Nodes: {len(bank_compiled.nodes)}")
for role_id, addr in bank_roles.items():
    print(f"  {role_id:<25} -> {addr}")

# TODO: Create two GovernanceEngines (mas_engine, bank_engine) from loaded org definitions
# Hint: GovernanceEngine(loaded.org_definition)
mas_engine = ____
bank_engine = ____

# TODO: Build level_map dict (string -> ConfidentialityLevel) and grant clearances to both orgs
# Hint: for cs in loaded.clearances: if cs.role_id in roles: RoleClearance(role_address, max_clearance,
#         compartments=frozenset(cs.compartments), granted_by_role_address="system_init")
#       then engine.grant_clearance(addr, clearance) for each role in both mas and bank
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

print(f"\nGovernance engines created and clearances granted.")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: KSP bridge — cross-org data sharing rules
# ══════════════════════════════════════════════════════════════════════

# TODO: Define ksp_bridge dict with bridge_id, source_org, target_org,
#   permitted_compartments (model_cards, audit_trails),
#   restricted_compartments (customer_data, model_internals),
#   conditions (requires_notice=True, notice_period_days=5, max_duration_days=30,
#               requires_bank_ceo_approval=True)
ksp_bridge = ____

print(f"\n=== KSP Bridge ===")
print(f"Bridge: {ksp_bridge['bridge_id']}")
print(f"Direction: {ksp_bridge['source_org']} -> {ksp_bridge['target_org']}")
print(f"Permitted compartments: {ksp_bridge['permitted_compartments']}")
print(f"Restricted compartments: {ksp_bridge['restricted_compartments']}")
print(f"Conditions: {ksp_bridge['conditions']}")


# TODO: Implement check_cross_org_access
# Hint: 1) get requester clearance via engine.get_context(addr).effective_clearance_level
#        2) check item compartments against bridge permitted/restricted sets
#        3) compare requester clearance value vs item classification value
def check_cross_org_access(
    requester_org_engine: GovernanceEngine,
    requester_addr: str,
    target_item: KnowledgeItem,
    bridge: dict,
) -> dict:
    """Check if a cross-org access request is permitted under KSP bridge rules."""
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
# TASK 3: Regulatory inspection — MAS inspector accesses bank data
# ══════════════════════════════════════════════════════════════════════

# TODO: Define bank_items list of 4 KnowledgeItems:
#   credit_model_card (CONFIDENTIAL, model_cards), governance_audit_log (SECRET, audit_trails),
#   customer_loan_records (SECRET, customer_data), model_weights_v3 (TOP_SECRET, model_internals)
# Then test ai_inspector and lead_inspector access to each item via check_cross_org_access
# and print ALLOW/DENY results for both inspectors
bank_items = ____

print(f"\n=== Regulatory Inspection Simulation ===")
print(f"MAS inspector requesting access to DBS Bank resources:\n")

# TODO: Loop over bank_items, call check_cross_org_access for mas_roles["ai_inspector"]
# then for mas_roles["lead_inspector"], print ALLOW/DENY + reason
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
# TASK 4: Delegation chaining — CEO -> compliance -> agent
# ══════════════════════════════════════════════════════════════════════

# TODO: Create CEO->compliance envelope using RoleEnvelope + ConstraintEnvelopeConfig
# Hint: RoleEnvelope(id=..., defining_role_address=bank_roles["bank_ceo"],
#         target_role_address=bank_roles["compliance_officer"],
#         envelope=ConstraintEnvelopeConfig(operational=OperationalConstraintConfig(allowed_actions=[...])))
ceo_to_compliance_env = RoleEnvelope(
    id=f"env-{uuid.uuid4().hex[:8]}",
    defining_role_address=____,
    target_role_address=____,
    envelope=ConstraintEnvelopeConfig(
        id="compliance-envelope",
        operational=OperationalConstraintConfig(
            allowed_actions=____,
        ),
        temporal=TemporalConstraintConfig(),
        data_access=DataAccessConstraintConfig(),
        communication=CommunicationConstraintConfig(),
    ),
)
bank_engine.set_role_envelope(ceo_to_compliance_env)

# TODO: Create compliance->agent envelope (tighter — fewer actions than ceo_to_compliance)
# Hint: same pattern; allowed_actions should be a subset of compliance officer's actions
compliance_to_agent_env = RoleEnvelope(
    id=f"env-{uuid.uuid4().hex[:8]}",
    defining_role_address=bank_roles["compliance_officer"],
    target_role_address=bank_roles["compliance_agent"],
    envelope=ConstraintEnvelopeConfig(
        id="compliance-agent-envelope",
        operational=OperationalConstraintConfig(
            allowed_actions=____,
        ),
        temporal=TemporalConstraintConfig(),
        data_access=DataAccessConstraintConfig(),
        communication=CommunicationConstraintConfig(),
    ),
)
bank_engine.set_role_envelope(compliance_to_agent_env)

print(f"\n=== Delegation Chain ===")
print(f"CEO -> Compliance Officer -> Compliance Agent")
print(f"\nMonotonic tightening in action:")

chain_roles = ["bank_ceo", "compliance_officer", "compliance_agent"]
for role_id in chain_roles:
    ctx = bank_engine.get_context(bank_roles[role_id])
    actions = sorted(ctx.allowed_actions) if ctx.allowed_actions else "(no envelope)"
    print(f"  {role_id:<25} clearance={str(ctx.effective_clearance_level):<15}")
    print(f"    actions={actions}")

print(f"\nDelegation audit trail:")
print(f"  CEO granted envelope to compliance_officer")
print(f"  compliance_officer granted envelope to compliance_agent")
print(f"  Each step is logged in GovernanceEngine audit store")
integrity = bank_engine.verify_audit_integrity()
print(f"  Audit integrity: {'VALID' if integrity else 'BROKEN'}")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Breach simulation — agent exceeds clearance
# ══════════════════════════════════════════════════════════════════════

# TODO: Create a RoleEnvelope for credit_agent with allowed_actions=["predict_default","query_application"]
# Then call bank_engine.set_role_envelope(credit_agent_env)
# Hint: defining_role_address=bank_roles["ml_lead"], target_role_address=bank_roles["credit_agent"]
credit_agent_env = ____
bank_engine.set_role_envelope(credit_agent_env)

# TODO: Create PactGovernedAgent for the credit_agent
# Hint: PactGovernedAgent(engine=bank_engine, role_address=bank_roles["credit_agent"])
governed_credit = ____

print(f"\n=== Breach Simulation ===")
print(f"Credit Agent envelope: {sorted(governed_credit.context.allowed_actions)}")
print(f"Credit Agent clearance: {governed_credit.context.effective_clearance_level}")

breach_tests = [
    ("predict_default", "Permitted action — within envelope"),
    ("query_application", "Permitted action — within envelope"),
    ("export_customer_data", "BREACH — not in envelope"),
    ("modify_model_weights", "BREACH — not in envelope"),
    ("escalate_clearance", "BREACH — not in envelope"),
    ("access_audit_logs", "BREACH — not in envelope"),
]

print(f"\nAction verification:")
for action, description in breach_tests:
    verdict = bank_engine.verify_action(bank_roles["credit_agent"], action)
    status = "ALLOW" if verdict.level == "auto_approved" else "BLOCK"
    print(f"  {action:<30} {status:<6} ({description})")

classified_item = KnowledgeItem(
    item_id="production_model_weights",
    classification=ConfidentialityLevel.TOP_SECRET,
    owning_unit_address=bank_roles.get("ml_lead", "R1"),
    compartments=frozenset(["model_internals"]),
)

# TODO: Use bank_engine.check_access to verify credit_agent cannot access TOP_SECRET item
# Hint: check_access(role_address, knowledge_item, TrustPosture.SUPERVISED)
access_result = ____

print(f"\nClassified access attempt (TOP_SECRET model weights):")
print(f"  Agent clearance: {governed_credit.context.effective_clearance_level}")
print(f"  Item classification: {classified_item.classification}")
print(f"  Decision: {'ALLOW' if access_result.allowed else 'DENY'}")
print(f"  Reason: {access_result.reason}")

print(f"\nFrozen context verification:")
try:
    governed_credit.context.role_address = "escalated_role"
    print("  ERROR: Should have raised FrozenInstanceError!")
except (AttributeError, TypeError) as e:
    print(f"  Self-modification blocked: {type(e).__name__}")
    print(f"  Agents cannot escalate their own privileges")

integrity = bank_engine.verify_audit_integrity()
print(f"\nFinal audit integrity: {'VALID' if integrity else 'BROKEN'}")
print(f"All breach attempts logged in audit trail")

# Clean up
mas_file.unlink()
bank_file.unlink()

print("\n--- Exercise 1 complete: multi-org governance with KSP bridges ---")
