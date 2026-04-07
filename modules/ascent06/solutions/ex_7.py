# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT06 — Exercise 7: Agent Governance at Scale
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Govern a multi-agent ML pipeline using PACT clearance levels,
#   envelope verification, and audit logging. Demonstrate how governance
#   properties propagate through a role hierarchy.
#
# TASKS:
#   1. Define a multi-tier organisation with clearance levels
#   2. Build governed role hierarchy (supervisor + 3 workers)
#   3. Demonstrate clearance-based access control
#   4. Test envelope enforcement across the fleet
#   5. Verify audit trail via GovernanceEngine
#   6. Scale governance: apply policy to multiple agents
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

import polars as pl

from kailash.trust import ConfidentialityLevel
from pact import GovernanceEngine, compile_org, load_org_yaml
from pact import GovernanceContext, PactGovernedAgent
from pact.governance import (
    ClearanceSpec,
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

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()


# ── Data Context ──────────────────────────────────────────────────────

loader = ASCENTDataLoader()
credit = loader.load("ascent03", "sg_credit_scoring.parquet")

data_context = (
    f"Singapore Credit Scoring: {credit.height:,} rows, "
    f"{len(credit.columns)} columns, default rate "
    f"{credit['default'].mean():.1%}"
)

print(f"=== Agent Governance at Scale ===")
print(f"Dataset: {data_context}")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Multi-tier organisation with clearance levels
# ══════════════════════════════════════════════════════════════════════
#
# Clearance levels (ConfidentialityLevel enum):
#   PUBLIC       — anyone
#   RESTRICTED   — basic authenticated access
#   CONFIDENTIAL — approved analysts
#   SECRET       — senior staff + ML engineers
#   TOP_SECRET   — executives + model risk only
#
# Resources are tagged with a minimum required clearance.
# An agent can only access a resource if its clearance >= resource level.

org_yaml_content = """
org_id: ascent_credit_governed
name: "ASCENT Credit Bureau (Governed)"

departments:
  - id: ml_pipeline
    name: "ML Pipeline Agents"
  - id: risk_governance
    name: "Risk Governance"

teams:
  - id: orchestration
    name: "Orchestration"
  - id: workers
    name: "Worker Agents"
  - id: oversight
    name: "Oversight"

roles:
  # Pipeline supervisor
  - id: pipeline_supervisor
    name: "Pipeline Supervisor"
    is_primary_for_unit: ml_pipeline

  # Worker agents
  - id: data_quality_worker
    name: "Data Quality Worker"
    reports_to: pipeline_supervisor
    agent: true
  - id: feature_worker
    name: "Feature Worker"
    reports_to: pipeline_supervisor
    agent: true
  - id: model_worker
    name: "Model Worker"
    reports_to: pipeline_supervisor
    agent: true

  # Risk governance
  - id: risk_officer
    name: "Risk Officer"
    is_primary_for_unit: risk_governance

clearances:
  - role: pipeline_supervisor
    level: secret
    compartments: [credit_applications, model_predictions, feature_store, production_logs]
  - role: data_quality_worker
    level: confidential
    compartments: [credit_applications, feature_store]
  - role: feature_worker
    level: confidential
    compartments: [credit_applications, feature_store]
  - role: model_worker
    level: secret
    compartments: [credit_applications, model_predictions, feature_store]
  - role: risk_officer
    level: top_secret
    compartments: [audit_logs, model_predictions, production_logs]
"""

org_file = Path(tempfile.mktemp(suffix=".yaml"))
org_file.write_text(org_yaml_content)

loaded = load_org_yaml(str(org_file))
compiled = compile_org(loaded.org_definition)

# Build role_id → address lookup
role_addr = {}
for addr, node in compiled.nodes.items():
    if node.role_definition is not None and not node.is_vacant:
        role_addr[node.node_id] = addr

print(f"\n=== Organisation Compiled ===")
print(f"Org ID: {compiled.org_id}")
print(f"Total nodes: {len(compiled.nodes)}")
print(f"Role addresses:")
for role_id, addr in role_addr.items():
    print(f"  {role_id:<25} → {addr}")

# Clearance hierarchy
print(f"\nClearance hierarchy:")
print(f"  PUBLIC       — anyone")
print(f"  RESTRICTED   — basic access")
print(f"  CONFIDENTIAL — approved analysts")
print(f"  SECRET       — senior staff, ML engineers")
print(f"  TOP_SECRET   — executives, model risk only")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Build governed role hierarchy
# ══════════════════════════════════════════════════════════════════════

engine = GovernanceEngine(loaded.org_definition)

# Grant clearances
level_map = {
    "public": ConfidentialityLevel.PUBLIC,
    "restricted": ConfidentialityLevel.RESTRICTED,
    "confidential": ConfidentialityLevel.CONFIDENTIAL,
    "secret": ConfidentialityLevel.SECRET,
    "top_secret": ConfidentialityLevel.TOP_SECRET,
}

print(f"\n=== Clearances Granted ===")
for cs in loaded.clearances:
    if cs.role_id in role_addr:
        addr = role_addr[cs.role_id]
        clearance = RoleClearance(
            role_address=addr,
            max_clearance=level_map[cs.level],
            compartments=frozenset(cs.compartments),
            granted_by_role_address="system_init",
        )
        engine.grant_clearance(addr, clearance)
        print(
            f"  {cs.role_id:<25} → {cs.level:<12} compartments={sorted(cs.compartments)}"
        )

# Set envelopes
envelope_defs = [
    (
        "pipeline_supervisor",
        "risk_officer",
        [
            "profile_data",
            "describe_column",
            "target_analysis",
            "train_model",
            "register_model",
        ],
    ),
    ("data_quality_worker", "pipeline_supervisor", ["profile_data", "describe_column"]),
    ("feature_worker", "pipeline_supervisor", ["describe_column", "target_analysis"]),
    (
        "model_worker",
        "pipeline_supervisor",
        ["train_model", "register_model"],
    ),
]

print(f"\n=== Envelopes Set ===")
for target_id, definer_id, actions in envelope_defs:
    if target_id in role_addr and definer_id in role_addr:
        env = RoleEnvelope(
            id=f"env-{uuid.uuid4().hex[:8]}",
            defining_role_address=role_addr[definer_id],
            target_role_address=role_addr[target_id],
            envelope=ConstraintEnvelopeConfig(
                id=f"constraint-{target_id}",
                operational=OperationalConstraintConfig(allowed_actions=actions),
                temporal=TemporalConstraintConfig(),
                data_access=DataAccessConstraintConfig(),
                communication=CommunicationConstraintConfig(),
            ),
        )
        engine.set_role_envelope(env)
        print(f"  {target_id:<25} defined_by={definer_id:<25} actions={actions}")

# Create governance contexts
print(f"\n=== Governance Contexts ===")
for role_id in role_addr:
    ctx = engine.get_context(role_addr[role_id])
    print(
        f"  {role_id:<25} clearance={str(ctx.effective_clearance_level):<35} "
        f"actions={sorted(ctx.allowed_actions) if ctx.allowed_actions else '(no envelope)'}"
    )


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Clearance-based access control
# ══════════════════════════════════════════════════════════════════════
#
# check_access evaluates whether a role can access a knowledge item
# based on clearance level and compartments.

print(f"\n=== Clearance-Based Access Control ===")

# Define knowledge items with classification levels
knowledge_items = [
    KnowledgeItem(
        item_id="credit_applications",
        classification=ConfidentialityLevel.SECRET,
        owning_unit_address="R1",  # Owned by pipeline supervisor
        compartments=frozenset(["credit_applications"]),
    ),
    KnowledgeItem(
        item_id="model_predictions",
        classification=ConfidentialityLevel.CONFIDENTIAL,
        owning_unit_address="R1",
        compartments=frozenset(["model_predictions"]),
    ),
    KnowledgeItem(
        item_id="audit_logs",
        classification=ConfidentialityLevel.TOP_SECRET,
        owning_unit_address="R2",  # Owned by risk officer
        compartments=frozenset(["audit_logs"]),
    ),
    KnowledgeItem(
        item_id="public_reports",
        classification=ConfidentialityLevel.PUBLIC,
        owning_unit_address="R1",
        compartments=frozenset(),
    ),
]

from kailash.trust import TrustPosture

for ki in knowledge_items:
    print(f"\n  Resource: {ki.item_id} (classification={ki.classification.value})")
    for role_id in [
        "pipeline_supervisor",
        "data_quality_worker",
        "model_worker",
        "risk_officer",
    ]:
        if role_id in role_addr:
            decision = engine.check_access(
                role_addr[role_id], ki, TrustPosture.SUPERVISED
            )
            status = "ALLOW" if decision.allowed else "DENY"
            print(f"    {role_id:<25} → {status} ({decision.reason})")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Envelope enforcement across the fleet
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Envelope Enforcement ===")
fleet_tests = [
    ("data_quality_worker", "profile_data", "DQ worker profiles data"),
    ("data_quality_worker", "train_model", "DQ worker tries to train"),
    ("feature_worker", "target_analysis", "Feature worker analyses targets"),
    ("feature_worker", "register_model", "Feature worker tries to register model"),
    ("model_worker", "train_model", "Model worker trains"),
    ("model_worker", "profile_data", "Model worker tries profiling"),
    ("pipeline_supervisor", "train_model", "Supervisor can train"),
    (
        "pipeline_supervisor",
        "deploy_model",
        "Supervisor tries deploy (not in envelope)",
    ),
]

for role_id, action, description in fleet_tests:
    if role_id in role_addr:
        verdict = engine.verify_action(role_addr[role_id], action)
        print(f"  {description}")
        print(f"    → {verdict.level}")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Audit trail
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Audit Trail ===")
print(f"Every governance operation is audited:")
print(f"  - Clearance grants: who granted what to whom")
print(f"  - Envelope assignments: who set what constraints")
print(f"  - verify_action calls: every ALLOW/BLOCK decision")
print(f"  - check_access calls: every knowledge access decision")
print()
print(f"GovernanceEngine supports:")
print(f"  - In-memory audit (default)")
print(f"  - SQLite-backed audit (store_backend='sqlite', store_url='path')")
print(f"  - EATP emission (eatp_emitter=PactEatpEmitter)")
print()
integrity = engine.verify_audit_integrity()
print(f"Audit integrity check: {'VALID' if integrity else 'BROKEN'}")


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Scale governance to multiple agents
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Governance at Scale ===")

# Create PactGovernedAgent instances for each worker
governed_agents = {}
for role_id in ["data_quality_worker", "feature_worker", "model_worker"]:
    if role_id in role_addr:
        agent = PactGovernedAgent(engine=engine, role_address=role_addr[role_id])
        governed_agents[role_id] = agent

print(f"Created {len(governed_agents)} governed agents:")
for role_id, agent in governed_agents.items():
    ctx = agent.context
    print(f"  {role_id:<25} actions={sorted(ctx.allowed_actions)}")

# Parallel governance checks
print(f"\nParallel envelope verification:")
for role_id, agent in governed_agents.items():
    for action in ["profile_data", "train_model", "deploy_model"]:
        verdict = engine.verify_action(role_addr[role_id], action)
        status = "ALLOW" if verdict.level == "auto_approved" else "BLOCK"
        print(f"  {role_id:<25} {action:<20} → {status}")

# Clean up
org_file.unlink()


# ══════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print(f"   GOVERNANCE AT SCALE — KEY PROPERTIES")
print(f"{'=' * 70}")
print(
    """
Clearance Levels (ConfidentialityLevel enum):
  PUBLIC → RESTRICTED → CONFIDENTIAL → SECRET → TOP_SECRET
  Resource classification independent of role permissions.
  Agent clearance must be >= resource classification.
  Prevents privilege escalation even with correct role.

Envelope Enforcement:
  verify_action() checks operational constraints.
  auto_approved = action is within all constraint dimensions.
  blocked = action is NOT in the allowed actions list.
  GovernanceEngine is thread-safe and fail-closed.

Governed Agents (PactGovernedAgent):
  Wraps GovernanceEngine + role_address.
  Agent receives frozen GovernanceContext (read-only).
  Tool execution goes through governance verification.
  Unregistered tools are DEFAULT-DENY.

Audit Trail:
  Every governance event is logged (clearance grants, action verdicts).
  verify_audit_integrity() checks for tampering.
  Supports in-memory, SQLite, and EATP emission backends.
"""
)

print(
    "✓ Exercise 7 complete — agent governance at scale with clearance, envelopes, audit"
)
