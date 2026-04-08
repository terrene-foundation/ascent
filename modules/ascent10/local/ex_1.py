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

# TODO: Write a YAML org definition for MAS (regulator).
# TODO: Include departments (ai_supervision, model_risk), teams,
# TODO: roles (chief_ai_officer, lead_inspector, ai_inspector,
# TODO: model_reviewer), and clearances at top_secret/secret/confidential.
mas_yaml = ____

# TODO: Write a YAML org definition for DBS Bank.
# TODO: Include bank_ceo, compliance_officer, ml_lead, credit_agent (agent: true),
# TODO: compliance_agent (agent: true). Grant clearances on customer_data,
# TODO: model_internals, model_cards, audit_trails compartments.
bank_yaml = ____

# TODO: Write each YAML to a temp file, load_org_yaml, then compile_org.
mas_file = ____
bank_file = ____

mas_loaded = ____
bank_loaded = ____

mas_compiled = ____
bank_compiled = ____

# TODO: Build role_id -> address lookup dicts for both orgs by walking
# TODO: compiled.nodes and selecting nodes with a non-vacant role_definition.
mas_roles = ____
____

bank_roles = ____
____

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

# TODO: Create a GovernanceEngine for each org from its loaded org_definition.
mas_engine = ____
bank_engine = ____

# TODO: Map yaml clearance level strings to ConfidentialityLevel enum values.
level_map = ____

# TODO: For every clearance in each loaded org, build a RoleClearance
# TODO: (role_address, max_clearance, frozenset(compartments), granted_by_role_address)
# TODO: and call engine.grant_clearance(addr, clearance).
____

____

print(f"\nGovernance engines created and clearances granted.")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: KSP bridge — cross-org data sharing rules
# ══════════════════════════════════════════════════════════════════════

# TODO: Define a Knowledge Sharing Policy (KSP) bridge as a dict with
# TODO: bridge_id, source_org, target_org, permitted_compartments
# TODO: (model_cards, audit_trails), restricted_compartments
# TODO: (customer_data, model_internals), and conditions (notice + approval).
ksp_bridge = ____

print(f"\n=== KSP Bridge ===")
print(f"Bridge: {ksp_bridge['bridge_id']}")
print(f"Direction: {ksp_bridge['source_org']} -> {ksp_bridge['target_org']}")
print(f"Permitted compartments: {ksp_bridge['permitted_compartments']}")
print(f"Restricted compartments: {ksp_bridge['restricted_compartments']}")
print(f"Conditions: {ksp_bridge['conditions']}")


def check_cross_org_access(
    requester_org_engine: GovernanceEngine,
    requester_addr: str,
    target_item: KnowledgeItem,
    bridge: dict,
) -> dict:
    """Check if a cross-org access request is permitted under KSP bridge rules."""
    # TODO: Get the requester's GovernanceContext via engine.get_context()
    # TODO: Pull effective_clearance_level from the context
    # TODO: Build sets from bridge permitted/restricted compartments
    # TODO: If item compartments intersect restricted -> deny with reason
    # TODO: If item compartments are not subset of permitted -> deny
    # TODO: If requester clearance value < item classification value -> deny
    # TODO: Otherwise return {"allowed": True, "reason": ...}
    ____


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Regulatory inspection — MAS inspector accesses bank data
# ══════════════════════════════════════════════════════════════════════

# TODO: Build 4 KnowledgeItems for the bank (credit_model_card,
# TODO: governance_audit_log, customer_loan_records, model_weights_v3)
# TODO: at varying classifications and compartments.
bank_items = ____

print(f"\n=== Regulatory Inspection Simulation ===")
print(f"MAS inspector requesting access to DBS Bank resources:\n")

# TODO: For each bank item, call check_cross_org_access using mas_engine
# TODO: and the ai_inspector address. Print ALLOW/DENY + reason.
____

print(f"\nLead Inspector (SECRET clearance) requesting same resources:")
# TODO: Re-run for the lead_inspector role; print ALLOW/DENY per item.
____


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Delegation chaining — CEO -> compliance -> agent
# ══════════════════════════════════════════════════════════════════════

# TODO: Build a RoleEnvelope where bank_ceo defines an envelope for
# TODO: compliance_officer with operational allowed_actions for review,
# TODO: report generation, regulatory checks, delegation. Use empty
# TODO: TemporalConstraintConfig / DataAccessConstraintConfig / CommunicationConstraintConfig.
ceo_to_compliance_env = ____
# TODO: Apply the envelope via bank_engine.set_role_envelope(...)
____

# TODO: Build a tighter envelope where compliance_officer defines actions
# TODO: for compliance_agent (read-only review_audit_trail, check_regulatory_status).
compliance_to_agent_env = ____
____

print(f"\n=== Delegation Chain ===")
print(f"CEO -> Compliance Officer -> Compliance Agent")
print(f"\nMonotonic tightening in action:")

# TODO: Iterate ["bank_ceo", "compliance_officer", "compliance_agent"],
# TODO: get_context for each, print clearance + sorted allowed_actions.
chain_roles = ["bank_ceo", "compliance_officer", "compliance_agent"]
____

print(f"\nDelegation audit trail:")
print(f"  CEO granted envelope to compliance_officer")
print(f"  compliance_officer granted envelope to compliance_agent")
print(f"  Each step is logged in GovernanceEngine audit store")
# TODO: Verify audit trail integrity and print VALID/BROKEN.
integrity = ____
print(f"  Audit integrity: {'VALID' if integrity else 'BROKEN'}")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Breach simulation — agent exceeds clearance
# ══════════════════════════════════════════════════════════════════════

# TODO: Build an envelope for credit_agent with allowed_actions limited to
# TODO: predict_default and query_application; defining role is ml_lead.
credit_agent_env = ____
____

# TODO: Wrap credit_agent as a PactGovernedAgent.
governed_credit = ____

print(f"\n=== Breach Simulation ===")
print(f"Credit Agent envelope: {sorted(governed_credit.context.allowed_actions)}")
print(f"Credit Agent clearance: {governed_credit.context.effective_clearance_level}")

# TODO: Build a list of (action, description) for permitted + breach actions.
breach_tests = ____

print(f"\nAction verification:")
# TODO: For each action, call bank_engine.verify_action(credit_agent_addr, action),
# TODO: classify ALLOW if verdict.level == "auto_approved" else BLOCK, print row.
____

# TODO: Build a TOP_SECRET KnowledgeItem (production_model_weights, model_internals).
classified_item = ____

# TODO: Use bank_engine.check_access with TrustPosture.SUPERVISED to attempt access.
access_result = ____
print(f"\nClassified access attempt (TOP_SECRET model weights):")
print(f"  Agent clearance: {governed_credit.context.effective_clearance_level}")
print(f"  Item classification: {classified_item.classification}")
print(f"  Decision: {'ALLOW' if access_result.allowed else 'DENY'}")
print(f"  Reason: {access_result.reason}")

print(f"\nFrozen context verification:")
# TODO: Try to mutate governed_credit.context.role_address to prove the
# TODO: dataclass is frozen; catch (AttributeError, TypeError).
try:
    ____
    print("  ERROR: Should have raised FrozenInstanceError!")
except (AttributeError, TypeError) as e:
    print(f"  Self-modification blocked: {type(e).__name__}")
    print(f"  Agents cannot escalate their own privileges")

# TODO: Verify final audit integrity.
integrity = ____
print(f"\nFinal audit integrity: {'VALID' if integrity else 'BROKEN'}")
print(f"All breach attempts logged in audit trail")

# TODO: Clean up the temporary YAML files.
____
____

print("\n--- Exercise 1 complete: multi-org governance with KSP bridges ---")
