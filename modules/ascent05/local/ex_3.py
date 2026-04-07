# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT05 — Exercise 3: ReAct Agents with Tools
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Build a ReActAgent with custom tools for autonomous data
#   exploration. Observe reasoning-action traces and tool selection.
#
# TASKS:
#   1. Define custom tools (DataExplorer, FeatureEngineer, ModelVisualizer)
#   2. Build ReActAgent with tool access
#   3. Run autonomous data exploration
#   4. Observe and analyse reasoning-action trace
#   5. Safety: what happens without cost budgets?
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os

import polars as pl
from dotenv import load_dotenv

from kaizen import Signature, InputField, OutputField
from kaizen_agents.agents.specialized.react import ReActAgent

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()

model = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))
if not model or not os.environ.get("OPENAI_API_KEY"):
    print("Set OPENAI_API_KEY and DEFAULT_LLM_MODEL in .env to run this exercise")
    raise SystemExit(0)


# ── Data Loading ──────────────────────────────────────────────────────

loader = ASCENTDataLoader()
credit = loader.load("ascent03", "sg_credit_scoring.parquet")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Define tools for the ReActAgent
# ══════════════════════════════════════════════════════════════════════


async def tool_profile_data(dataset_name: str) -> str:
    """Profile a dataset using DataExplorer and return summary."""
    # TODO: Import DataExplorer from kailash_ml; profile a 10_000-row sample of credit (seed=42)
    #   Return formatted string: "Dataset: {dataset_name}\nRows: ...\nColumns: ...\nAlerts: ...\nTypes: ...\nTop alerts: ..."
    #   profile attributes: .n_rows, .n_columns, .alerts, .type_summary
    ____
    ____
    ____
    ____
    ____


async def tool_check_correlations(threshold: float = 0.8) -> str:
    """Find highly correlated features."""
    # TODO: Import DataExplorer and AlertConfig(high_correlation_threshold=threshold)
    #   Profile a 5000-row sample; iterate profile.correlation_matrix (dict[str, dict[str, float]])
    #   Collect pairs where abs(corr) > threshold, skip self-pairs and duplicates (use seen set)
    #   Return formatted string listing high-correlation pairs up to 10
    ____
    ____
    ____
    ____
    ____
    ____
    ____


def tool_describe_column(column_name: str) -> str:
    """Get statistics for a specific column."""
    # TODO: Return error string if column_name not in credit.columns
    #   For numeric (Float64/Float32/Int64/Int32): return mean, std, min, max, null_count
    #   For categorical: return n_unique, null_count, top 5 value_counts
    ____
    ____
    ____
    ____
    ____


def tool_default_rate_by(column_name: str) -> str:
    """Compute default rate grouped by a column."""
    # TODO: Return error string if column_name not in credit.columns
    #   Group credit by column_name; agg default mean (alias "default_rate") and count (alias "n")
    #   Sort descending, head(10); format rows as "  {val}: {rate:.3f} (n={n})"
    ____
    ____
    ____
    ____
    ____
    ____


# Tool registry
tools = {
    "profile_data": tool_profile_data,
    "check_correlations": tool_check_correlations,
    "describe_column": tool_describe_column,
    "default_rate_by": tool_default_rate_by,
}


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Build ReActAgent
# ══════════════════════════════════════════════════════════════════════


# TODO: Define DataExplorationResult(Signature) with:
#   InputField:  task (str)
#   OutputField: findings (list[str]), tools_used (list[str]), recommendation (str)
____
____
____
____
____
____


def run_react_agent():
    # TODO: Implement run_react_agent():
    #   1. Create ReActAgent(model=model)
    #   2. Build task string: explore credit dataset, profile, check correlations,
    #      find default predictors, recommend preprocessing
    #   3. Build tool_descriptions from tools dict (name: docstring for each)
    #   4. Build context string with tool_descriptions, dataset row count, columns
    #   5. Print task and available tool names
    #   6. Call agent.run(task=task, context=context) — sync call, not async
    #   7. Print result keys and each key: value[:200]; return result
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


react_result = run_react_agent()


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Analyse reasoning-action trace
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Reasoning-Action Trace ===")
print("ReAct loop: Thought → Action → Observation → Thought → ...")
print(f"The agent autonomously decided which tools to call and in what order.")
print(f"This is the key difference from CoT (Exercise 2):")
print(f"  CoT: reason step-by-step, produce answer")
print(f"  ReAct: reason, ACT on the world (call tools), observe results, reason again")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Safety — cost budgets and runaway agents
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Safety: Cost Budgets ===")
print(
    """
What happens if you REMOVE the cost budget?

1. The agent could loop indefinitely (calling tools → reasoning → more tools)
2. Each LLM call costs money — 100 iterations at $0.05/call = $5.00
3. What if the agent calls DataExplorer on a 100GB dataset?
   → Memory exhaustion, cluster costs, timeout failures

budget_usd is NOT optional. It is a governance requirement.

In production:
  - Set budgets proportional to task complexity
  - Monitor actual spend vs budget
  - PACT (Module 6) enforces budgets as operating envelopes
  - Agents cannot modify their own budget (frozen GovernanceContext)
"""
)

print("\n✓ Exercise 3 complete — ReActAgent with tools and safety analysis")
