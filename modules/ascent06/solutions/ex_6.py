# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT06 — Exercise 6: Governed Agents
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Wrap a Kaizen agent with PactGovernedAgent. Define operating
#   envelopes and demonstrate that agents cannot modify their own
#   governance (frozen GovernanceContext).
#
# TASKS:
#   1. Create GovernanceEngine from a YAML organization
#   2. Wrap an agent with PactGovernedAgent
#   3. Test governance enforcement via verify_action
#   4. Demonstrate frozen context and tool registration
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

from dotenv import load_dotenv

from kailash.trust import ConfidentialityLevel
from pact import GovernanceEngine, GovernanceContext, PactGovernedAgent
from pact import compile_org, load_org_yaml
from pact.governance import RoleClearance, RoleEnvelope
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
# TASK 1: Set up GovernanceEngine
# ══════════════════════════════════════════════════════════════════════

# Minimal org YAML for this exercise
org_yaml = """
org_id: ascent_demo
name: "ASCENT Demo"

departments:
  - id: data_science
    name: "Data Science"

teams:
  - id: modeling
    name: "Modeling"

roles:
  - id: ml_agent
    name: "ML Analysis Agent"
    is_primary_for_unit: data_science
    agent: true

clearances:
  - role: ml_agent
    level: confidential
    compartments: [credit_data, feature_store]
"""

org_file = Path(tempfile.mktemp(suffix=".yaml"))
org_file.write_text(org_yaml)

loaded = load_org_yaml(str(org_file))
engine = GovernanceEngine(loaded.org_definition)

# Find the ml_agent role address
compiled = compile_org(loaded.org_definition)
agent_addr = None
for addr, node in compiled.nodes.items():
    if node.node_id == "ml_agent":
        agent_addr = addr
        break

# Grant clearance
for cs in loaded.clearances:
    if cs.role_id == "ml_agent" and agent_addr:
        clearance = RoleClearance(
            role_address=agent_addr,
            max_clearance=ConfidentialityLevel.CONFIDENTIAL,
            compartments=frozenset(cs.compartments),
            granted_by_role_address="system_init",
        )
        engine.grant_clearance(agent_addr, clearance)

# Set operating envelope
envelope = RoleEnvelope(
    id=f"env-{uuid.uuid4().hex[:8]}",
    defining_role_address=agent_addr,  # Self-defined for demo
    target_role_address=agent_addr,
    envelope=ConstraintEnvelopeConfig(
        id="ml-agent-envelope",
        operational=OperationalConstraintConfig(
            allowed_actions=["profile_data", "describe_column", "default_rate_by"],
        ),
        temporal=TemporalConstraintConfig(),
        data_access=DataAccessConstraintConfig(),
        communication=CommunicationConstraintConfig(),
    ),
)
engine.set_role_envelope(envelope)

# Get governance context
context = engine.get_context(agent_addr)

print(f"=== GovernanceContext ===")
print(f"Address: {agent_addr}")
print(f"Clearance: {context.effective_clearance_level}")
print(f"Compartments: {context.compartments}")
print(f"Allowed actions: {context.allowed_actions}")

org_file.unlink()


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Wrap agent with PactGovernedAgent
# ══════════════════════════════════════════════════════════════════════

# PactGovernedAgent wraps the engine, NOT the base agent directly
# It takes: engine, role_address, posture
governed = PactGovernedAgent(
    engine=engine,
    role_address=agent_addr,
)

print(f"\n=== PactGovernedAgent ===")
print(f"Role address: {agent_addr}")
print(f"Context (read-only): {governed.context is not None}")
print(f"Context clearance: {governed.context.effective_clearance_level}")
print(f"Context actions: {governed.context.allowed_actions}")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Test governance enforcement
# ══════════════════════════════════════════════════════════════════════

# Register tools with the governed agent
# Only registered tools can be executed (default-deny)
governed.register_tool("profile_data", cost=0.1, resource="credit_data")
governed.register_tool("describe_column", cost=0.05, resource="credit_data")
governed.register_tool("default_rate_by", cost=0.1, resource="credit_data")

print(f"\n=== Governance Enforcement ===")

# Test via verify_action on the engine
test_cases = [
    ("profile_data", "Use profile_data tool"),
    ("describe_column", "Use describe_column tool"),
    ("training_pipeline", "Use training_pipeline (NOT in envelope)"),
    ("deploy_model", "Deploy model (NOT in envelope)"),
    ("default_rate_by", "Use default_rate_by tool"),
]

for action, description in test_cases:
    verdict = engine.verify_action(agent_addr, action)
    print(f"  {description}")
    print(f"    → {verdict.level}: {verdict.reason}")

# Test tool execution through governed agent
print(f"\n=== Tool Execution via PactGovernedAgent ===")
print(f"Registered tools: {list(governed._registered_tools.keys())}")
print(f"Unregistered tools are DEFAULT-DENY")
print()
print(f"PactGovernedAgent.execute_tool() flow:")
print(f"  1. Check if tool is registered → if not, BLOCK")
print(f"  2. Verify action via GovernanceEngine.verify_action()")
print(f"  3. If BLOCKED → raise GovernanceBlockedError")
print(f"  4. If HELD → raise GovernanceHeldError")
print(f"  5. If FLAGGED → log warning, proceed")
print(f"  6. If AUTO_APPROVED → proceed silently")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Frozen context demonstration
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Frozen GovernanceContext ===")
print(
    """
GovernanceContext is a frozen dataclass:

  @dataclass(frozen=True)
  class GovernanceContext:
      role_address: str
      posture: TrustPostureLevel
      effective_envelope: ConstraintEnvelopeConfig | None
      clearance: RoleClearance | None
      effective_clearance_level: ConfidentialityLevel | None
      allowed_actions: frozenset[str]
      compartments: frozenset[str]
      org_id: str
      created_at: datetime

  # This RAISES FrozenInstanceError:
  context.role_address = "hacked"

WHY frozen?
  1. Agents cannot escalate their own privileges
  2. Context is immutable proof of what was authorized
  3. AuditChain can verify context was not tampered with
  4. Monotonic tightening is guaranteed (child <= parent)
"""
)

# Demonstrate immutability
try:
    governed.context.role_address = "hacked"
    print("ERROR: Should have raised FrozenInstanceError!")
except (AttributeError, TypeError) as e:
    print(f"Attempted modification blocked: {type(e).__name__}")
    print(f"  Agents RECEIVE governance but CANNOT modify it")

print("\n✓ Exercise 6 complete — governed agents with PACT enforcement")
