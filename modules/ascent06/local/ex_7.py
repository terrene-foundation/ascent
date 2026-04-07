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

# TODO: Write org_yaml_content for "ASCENT Credit Bureau (Governed)".
# 2 departments: ml_pipeline, risk_governance.
# 3 teams: orchestration, workers, oversight.
# 5 roles: pipeline_supervisor (is_primary_for_unit: ml_pipeline),
#   data_quality_worker (reports_to: pipeline_supervisor, agent: true),
#   feature_worker (reports_to: pipeline_supervisor, agent: true),
#   model_worker (reports_to: pipeline_supervisor, agent: true),
#   risk_officer (is_primary_for_unit: risk_governance).
# Clearances: pipeline_supervisor→secret[credit_applications,model_predictions,feature_store,production_logs],
#   data_quality_worker→confidential[credit_applications,feature_store],
#   feature_worker→confidential[credit_applications,feature_store],
#   model_worker→secret[credit_applications,model_predictions,feature_store],
#   risk_officer→top_secret[audit_logs,model_predictions,production_logs].
____

org_file = Path(tempfile.mktemp(suffix=".yaml"))
org_file.write_text(org_yaml_content)

# TODO: Load and compile the org, build a role_id → address lookup dict.
# Print: org_id, total nodes, and each role_id → address mapping.
# Print the clearance hierarchy (PUBLIC through TOP_SECRET).
____
____
____
____


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Build governed role hierarchy
# ══════════════════════════════════════════════════════════════════════

# TODO: Create GovernanceEngine(loaded.org_definition).
# Grant clearances for all 5 roles using the level_map and role_addr lookup.
# Print each granted clearance.
____
____
____

# TODO: Set envelopes for 4 role pairs:
# pipeline_supervisor (defined_by risk_officer): profile_data, describe_column,
#   target_analysis, train_model, register_model.
# data_quality_worker (defined_by pipeline_supervisor): profile_data, describe_column.
# feature_worker (defined_by pipeline_supervisor): describe_column, target_analysis.
# model_worker (defined_by pipeline_supervisor): train_model, register_model.
____
____
____

# TODO: Print governance contexts for all roles (clearance + allowed_actions).
____
____


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Clearance-based access control
# ══════════════════════════════════════════════════════════════════════
#
# check_access evaluates whether a role can access a knowledge item
# based on clearance level and compartments.

print(f"\n=== Clearance-Based Access Control ===")

# TODO: Define 4 KnowledgeItems:
# credit_applications (SECRET, compartments={"credit_applications"}),
# model_predictions (CONFIDENTIAL, compartments={"model_predictions"}),
# audit_logs (TOP_SECRET, compartments={"audit_logs"}),
# public_reports (PUBLIC, compartments=frozenset()).
# For each item, check access for pipeline_supervisor, data_quality_worker,
# model_worker, risk_officer using engine.check_access(addr, ki, TrustPosture.SUPERVISED).
# Print ALLOW/DENY with reason for each combination.
____
____
____
____
____


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Envelope enforcement across the fleet
# ══════════════════════════════════════════════════════════════════════

# TODO: Run 8 test cases through engine.verify_action().
# Include: DQ worker profiles (allow), DQ worker trains (block),
# feature worker analyses targets (allow), feature worker registers model (block),
# model worker trains (allow), model worker profiles (block),
# supervisor trains (allow), supervisor deploys (block — not in envelope).
____
____
____


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Audit trail
# ══════════════════════════════════════════════════════════════════════

# TODO: Print the audit trail explanation (every governance operation is audited:
# clearance grants, envelope assignments, verify_action calls, check_access calls).
# Describe 3 GovernanceEngine audit backends (in-memory, SQLite, EATP).
# Call engine.verify_audit_integrity() and print whether audit is VALID or BROKEN.
____
____
____


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Scale governance to multiple agents
# ══════════════════════════════════════════════════════════════════════

# TODO: Create PactGovernedAgent instances for each of the 3 worker roles.
# Print each agent's role_id and allowed_actions.
# Then run parallel governance checks for all 3 workers × 3 actions
# (profile_data, train_model, deploy_model) — print ALLOW or BLOCK for each.
____
____
____
____

# Clean up
org_file.unlink()


# ══════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════

# TODO: Print the 4-section "GOVERNANCE AT SCALE — KEY PROPERTIES" summary block:
# Clearance Levels, Envelope Enforcement, Governed Agents (PactGovernedAgent),
# and Audit Trail properties.
____

print(
    "✓ Exercise 7 complete — agent governance at scale with clearance, envelopes, audit"
)
