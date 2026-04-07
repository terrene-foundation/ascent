# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT09 — Exercise 7: MCP Integration
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Build an MCP server exposing kailash-ml tools, then connect
#   an agent to use those tools for ML operations.
#
# TASKS:
#   1. Create MCPServer with tool definitions
#   2. Expose DataExplorer as MCP tool
#   3. Expose TrainingPipeline as MCP tool
#   4. Connect ReActAgent to MCP server
#   5. Run agent-driven ML workflow via MCP
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os

import polars as pl

from collections import Counter

from kaizen_agents.agents.specialized.react import ReActAgent
from kailash_ml import DataExplorer
from kailash.mcp_server import MCPServer

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()

if not os.environ.get("OPENAI_API_KEY"):
    print("\u26a0 OPENAI_API_KEY not set \u2014 skipping LLM exercises.")
    print("  Set it in .env to run this exercise with real LLM calls.")
    import sys

    sys.exit(0)

model = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))

loader = ASCENTDataLoader()
_data_cache: dict[str, pl.DataFrame] = {}


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Create MCPServer with tool definitions
# ══════════════════════════════════════════════════════════════════════

# TODO: Instantiate MCPServer(name="kailash-ml-tools").
# MCP decouples providers (this server) from consumers (the agent).
# Any MCP-compatible agent discovers and calls these tools at runtime.
____

print(f"=== MCP Server: kailash-ml-tools ===")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Expose DataExplorer as MCP tool
# ══════════════════════════════════════════════════════════════════════

# TODO: Implement two @server.tool() async functions:
#
#   explore_dataset(dataset_name: str) -> str
#     Load "ascent09/{dataset_name}.parquet" via loader, cache in _data_cache.
#     Create DataExplorer(), call await explorer.profile(df).
#     Return a summary string: shape, columns, numeric_columns, missing_summary, df.head(3).
#
#   get_column_stats(dataset_name: str, column: str) -> str
#     Guard: return error string if dataset not in _data_cache or column not in df.
#     Numeric dtype → return mean/std/min/max/median/null_count dict as str.
#     Other dtype → return type/unique/null_count/top_values (value_counts head 10) as str.
____
____

print(f"  explore_dataset | get_column_stats registered")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Expose TrainingPipeline as MCP tool
# ══════════════════════════════════════════════════════════════════════

# TODO: Implement two more @server.tool() async functions:
#
#   train_classifier(dataset_name, target, features, algorithm="gradient_boosting") -> str
#     Guard: return error if dataset not in _data_cache.
#     Split features string on commas. Train nearest-centroid classifier (80/20 split):
#     compute per-class centroid on train set; predict by nearest centroid on test set;
#     compute accuracy. Return string with algorithm, accuracy, features.
#
#   list_datasets() -> str
#     If _data_cache empty → return "not loaded" message.
#     Otherwise return one line per dataset: "  name: R rows × C columns".
____
____

print(f"  train_classifier | list_datasets registered")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Connect ReActAgent to MCP server
# ══════════════════════════════════════════════════════════════════════

mcp_tools = [explore_dataset, get_column_stats, train_classifier, list_datasets]


async def agent_with_mcp():
    # TODO: Create ReActAgent(model=model). Print tool count and note that the
    # agent discovers tools at runtime with no hardcoded knowledge.
    ____
    ____


agent = asyncio.run(agent_with_mcp())


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Run agent-driven ML workflow via MCP
# ══════════════════════════════════════════════════════════════════════


async def run_ml_workflow():
    """Run ReActAgent on an end-to-end ML task using the four MCP tools.
    The agent should autonomously: explore dataset → identify features and target →
    train classifier → report accuracy. If result has .trace, print each step.
    """
    # TODO: Create ReActAgent(model=model); run it with a task string describing
    # the end-to-end ML objective; print result and reasoning trace if available.
    ____
    ____


workflow_result = asyncio.run(run_ml_workflow())

print(f"\n=== MCP Architecture ===")
print(f"  Server defines tools | Agent discovers at runtime | Protocol: JSON-RPC")
print(f"  Benefits: reusable tools, no hardcoded knowledge, gated access per agent")

print("\n✓ Exercise 7 complete — MCP server with kailash-ml tools + ReActAgent")
