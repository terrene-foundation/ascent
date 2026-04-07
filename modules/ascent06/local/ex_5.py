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

# TODO: Write an org_yaml string for "ASCENT Credit Bureau".
# Include 3 departments (data_science, risk, operations), 4 teams
# (modeling, mlops, model_validation, serving), 7 roles
# (senior_data_scientist, junior_data_scientist, ml_engineer,
# model_validator, compliance_officer, sre, customer_service[agent=true]).
# Set clearances: senior_ds→secret[credit_data,feature_store],
# junior_ds→confidential[credit_data], ml_engineer→secret[credit_data,feature_store,model_artifacts],
# model_validator→top_secret[credit_data,audit_logs,model_artifacts](nda_signed=true),
# compliance_officer→confidential[audit_logs], sre→secret[production_logs,model_artifacts],
# customer_service→restricted[credit_decisions].
____

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

# TODO: Load the YAML file with load_org_yaml() and compile with compile_org().
# Print: org_id, name, department count, team count, role count, clearance count.
# Then print the compiled node addresses and types.
____
____
____
____
____


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Create GovernanceEngine and grant clearances
# ══════════════════════════════════════════════════════════════════════

# TODO: Create GovernanceEngine(loaded_org.org_definition).
# Build a level_map dict mapping string levels to ConfidentialityLevel enum values.
# Iterate over loaded_org.clearances, build a RoleClearance for each role found
# in role_addr, and call engine.grant_clearance(addr, clearance).
# Print each granted clearance: role_id, level, compartments.
____
____
____
____
____
____

# TODO: Set operating envelopes for 4 role pairs using RoleEnvelope +
# ConstraintEnvelopeConfig(operational=OperationalConstraintConfig(allowed_actions=...)).
# Pairs: senior_ds→train/evaluate/profile (defined_by model_validator),
# junior_ds→profile/evaluate (defined_by senior_ds),
# ml_engineer→train/evaluate/deploy/monitor (defined_by model_validator),
# customer_service→query_prediction (defined_by sre).
# Call engine.set_role_envelope(envelope) for each.
____
____
____
____
____

print(f"\n=== GovernanceEngine Ready ===")
print(f"Organization: {loaded_org.org_definition.name}")
print(f"Enforcement mode: fail-closed (deny on error)")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Verify actions
# ══════════════════════════════════════════════════════════════════════

# TODO: Run 5 test cases through engine.verify_action(role_addr[role_id], action).
# Test cases: senior_ds can train_model, junior_ds cannot train_model,
# ml_engineer can deploy_model, customer_service cannot train_model,
# customer_service can query_prediction.
# Print each description, role, action, verdict.level, and verdict.reason.
____
____
____


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Frozen GovernanceContext
# ══════════════════════════════════════════════════════════════════════

# TODO: For senior_data_scientist, ml_engineer, customer_service:
# call engine.get_context(role_addr[role_id]) and print address, posture,
# effective_clearance_level, compartments, and allowed_actions.
____
____
____

# TODO: Print the "Frozen Context Properties" explanation block explaining:
# why GovernanceContext is a frozen dataclass, what it prevents,
# and the 4 security guarantees it provides.
____

# TODO: Demonstrate immutability — try to set ctx.role_address = "hacked"
# on the senior_data_scientist context; catch AttributeError/TypeError and print
# the exception type plus an explanation that agents cannot self-modify.
____
____
____

# Clean up
org_file.unlink()

print("\n✓ Exercise 5 complete — PACT governance setup with YAML org definition")
