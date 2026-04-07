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

# TODO: Load the org YAML, create GovernanceEngine, and compile to find the
# ml_agent role address. Grant clearance (CONFIDENTIAL, compartments from YAML).
# Set an operating envelope allowing actions: profile_data, describe_column,
# default_rate_by. Get the governance context and print address, clearance,
# compartments, and allowed_actions.
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

print(f"=== GovernanceContext ===")
print(f"Address: {agent_addr}")
print(f"Clearance: {context.effective_clearance_level}")
print(f"Compartments: {context.compartments}")
print(f"Allowed actions: {context.allowed_actions}")

org_file.unlink()


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Wrap agent with PactGovernedAgent
# ══════════════════════════════════════════════════════════════════════

# TODO: Create PactGovernedAgent(engine=engine, role_address=agent_addr).
# Print: role_address, whether context is attached, context clearance,
# and context allowed_actions.
____
____

print(f"\n=== PactGovernedAgent ===")
print(f"Role address: {agent_addr}")
print(f"Context (read-only): {governed.context is not None}")
print(f"Context clearance: {governed.context.effective_clearance_level}")
print(f"Context actions: {governed.context.allowed_actions}")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Test governance enforcement
# ══════════════════════════════════════════════════════════════════════

# TODO: Register 3 tools on the governed agent:
# profile_data (cost=0.1, resource="credit_data"),
# describe_column (cost=0.05, resource="credit_data"),
# default_rate_by (cost=0.1, resource="credit_data").
____
____
____

print(f"\n=== Governance Enforcement ===")

# TODO: Run 5 test cases through engine.verify_action(agent_addr, action).
# Actions: profile_data, describe_column, training_pipeline (NOT in envelope),
# deploy_model (NOT in envelope), default_rate_by.
# Print each description and the verdict level + reason.
____
____
____

# TODO: Print the PactGovernedAgent.execute_tool() 6-step flow:
# registered check → verify_action → BLOCKED → HELD → FLAGGED → AUTO_APPROVED.
____
____


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Frozen context demonstration
# ══════════════════════════════════════════════════════════════════════

# TODO: Print the GovernanceContext frozen dataclass definition block
# (fields: role_address, posture, effective_envelope, clearance,
# effective_clearance_level, allowed_actions, compartments, org_id, created_at)
# and the 4 reasons WHY it is frozen.
____

# TODO: Demonstrate immutability — try governed.context.role_address = "hacked".
# Catch AttributeError/TypeError and print the exception type plus
# "Agents RECEIVE governance but CANNOT modify it".
____
____
____

print("\n✓ Exercise 6 complete — governed agents with PACT enforcement")
