# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT10 — Exercise 7: Governed Agents with PactGovernedAgent
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Wrap agents with PactGovernedAgent — budget cascading, tool
#   restrictions, clearance levels, audit chain — for compliant ML ops.
#
# TASKS:
#   1. Build base ReActAgent for data analysis
#   2. Wrap with PactGovernedAgent (role, budget, tools)
#   3. Test tool registration and execution
#   4. Demonstrate governance concepts
#   5. Audit trail concepts
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os
import tempfile

import polars as pl

from kaizen_agents import Delegate
from kaizen_agents.agents.specialized.react import ReActAgent
from pact import GovernanceEngine, PactGovernedAgent, compile_org, load_org_yaml

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()

model = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))
loader = ASCENTDataLoader()
reports = loader.load("ascent10", "sg_company_reports.parquet")
print(f"=== Company Reports: {reports.height} documents ===")


async def read_data(dataset: str) -> str:
    df = loader.load("ascent10", f"{dataset}.parquet")
    return f"Dataset: {dataset}, Shape: {df.shape}, Columns: {df.columns}"


async def analyze_text(text: str) -> str:
    words = text.lower().split()
    return f"Text length: {len(words)} words, Sample: {' '.join(words[:20])}..."


async def deploy_model(model_name: str) -> str:
    return f"Model {model_name} deployed to production endpoint"


async def delete_data(dataset: str) -> str:
    return f"Dataset {dataset} deleted permanently"


tools = [read_data, analyze_text, deploy_model, delete_data]

# ══════════════════════════════════════════════════════════════════════
# TASK 1: Create ReActAgent(model=model, tools=tools, max_llm_cost_usd=10.0).
#   Print a summary showing all 4 tools and what governance will restrict.
# ══════════════════════════════════════════════════════════════════════
____
____

# ══════════════════════════════════════════════════════════════════════
# TASK 2: Write minimal PACT org YAML (data_ops dept, analyst + operator
#   roles). Load, compile, create GovernanceEngine, find role addresses.
#   Create PactGovernedAgent for analyst and operator. Print constraints.
# ══════════════════════════════════════════════════════════════════════
____
____
____
____
____

# ══════════════════════════════════════════════════════════════════════
# TASK 3: Register read_data + analyze_text on governed_analyst.
#   Register read_data + deploy_model on governed_deployer.
#   Print the tool sets and note that execute_tool() enforces governance.
# ══════════════════════════════════════════════════════════════════════
____
____

# ══════════════════════════════════════════════════════════════════════
# TASK 4: Print the three guarantees PactGovernedAgent provides:
#   tool restriction, role-based access, immutable audit trail.
#   Show what each role can and cannot do.
# ══════════════════════════════════════════════════════════════════════
____

# ══════════════════════════════════════════════════════════════════════
# TASK 5: Explain what GovernanceEngine logs per decision (role, action,
#   allowed/blocked, timestamp, reason). Map to EU AI Act Art. 12,
#   MAS TRM 7.5, Singapore AI Verify accountability principle.
# ══════════════════════════════════════════════════════════════════════
____

print("\n✓ Exercise 7 complete — PactGovernedAgent with budget, tools, and audit")
