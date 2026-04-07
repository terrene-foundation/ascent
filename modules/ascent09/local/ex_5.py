# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT09 — Exercise 5: Building Agents with Kaizen
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Build a ReActAgent with custom tools for autonomous data
#   analysis — tool definition, reasoning loops, structured output.
#
# TASKS:
#   1. Define custom tools (data_summary, run_query, plot_chart)
#   2. Build ReActAgent with tool access
#   3. Run agent on multi-step analysis task
#   4. Inspect reasoning trace
#   5. Build custom BaseAgent with Signature for structured analysis
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import json
import os

import polars as pl

from kaizen_agents import Delegate
from kaizen import Signature, InputField, OutputField
from kaizen_agents.agents.specialized.react import ReActAgent
from kaizen_agents.agents.specialized.simple_qa import SimpleQAAgent

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()

if not os.environ.get("OPENAI_API_KEY"):
    print("\u26a0 OPENAI_API_KEY not set \u2014 skipping LLM exercises.")
    print("  Set it in .env to run this exercise with real LLM calls.")
    import sys

    sys.exit(0)

model = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))
print(f"LLM Model: {model}")

loader = ASCENTDataLoader()
reports = loader.load("ascent09", "sg_company_reports.parquet")
print(f"Loaded {reports.height:,} company reports | Columns: {reports.columns}")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Define custom tools (data_summary, run_query, plot_chart)
# ══════════════════════════════════════════════════════════════════════


def data_summary(dataset_name: str = "reports") -> str:
    """Get a statistical summary of the company reports dataset.
    Return shape, column names, and per-column stats: mean/std/min/max for
    numeric columns; unique count and avg string length for text columns.
    """
    # TODO: Iterate df.columns; branch on dtype (Int/Float vs Utf8/String);
    # compute stats via polars expressions; build and return a text summary string.
    ____
    ____


def run_query(query_expr: str) -> str:
    """Run a heuristic query on the reports dataset based on a description.
    Interpret 'top 5', 'count', and 'unique' keywords; fall back to a row-count summary.
    """
    # TODO: Check for 'top'/'5', 'count', 'unique' keywords; run the appropriate
    # polars operation; return a formatted string result.
    ____
    ____


def plot_chart(chart_type: str, x_col: str = "", y_col: str = "") -> str:
    """Return a text description of what the chart would show.
    Validate x_col/y_col against df.columns; fall back to first two columns.
    Mention chart_type, axes, and data point count.
    """
    # TODO: Validate column names, build and return a descriptive chart-spec string.
    ____
    ____


tools = [data_summary, run_query, plot_chart]
print(f"\n=== Tools: {[t.__name__ for t in tools]} ===")
print(data_summary())


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Build ReActAgent with tool access
# ══════════════════════════════════════════════════════════════════════


async def build_and_run_react():
    # TODO: Instantiate ReActAgent(model=model). Print model name and tool names.
    # Return the agent. The ReAct loop (Thought→Action→Observation) is autonomous —
    # no explicit tool dispatch in your code.
    ____
    ____


agent = asyncio.run(build_and_run_react())


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Run agent on multi-step analysis task
# ══════════════════════════════════════════════════════════════════════


async def multi_step_analysis():
    """Create a ReActAgent and run it on a task requiring multiple tool calls:
    summarise dataset → find top-5 by numeric metric → count unique values →
    suggest best chart → synthesise findings. The agent decides tool order autonomously.
    """
    # TODO: Create ReActAgent(model=model); define the multi-step task string;
    # call agent.run(task); print and return the result.
    ____
    ____


analysis_result = asyncio.run(multi_step_analysis())


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Inspect reasoning trace
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== ReAct Reasoning Trace ===")
print(f"  Thought → Action (tool + args) → Observation (tool output) → repeat")
print(f"  Agent decides WHICH tool and WHAT args via LLM reasoning, not if-else.")
print(f"  Loop terminates when agent emits a final answer instead of an action.")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Custom BaseAgent with Signature for structured analysis
# ══════════════════════════════════════════════════════════════════════

# TODO: Define DataAnalysisSignature(Signature) with:
#   InputFields: dataset_summary (str), analysis_question (str)
#   OutputFields: key_findings (list[str]), recommended_model (str),
#                 data_quality_issues (list[str]), next_steps (list[str])
# Precise descriptions drive output quality — the Signature is the typed contract.
____


async def structured_agent_analysis():
    # TODO: Call data_summary() to get the dataset summary. Create
    # SimpleQAAgent(model=model). Run it with a prompt combining the summary and
    # a question about predicting company performance. Print and return the result.
    ____
    ____


structured_result = asyncio.run(structured_agent_analysis())

print(f"\n=== ReActAgent vs Signature Agent ===")
print(
    f"  ReAct: autonomous tool use, multi-step, flexible — use when exploration needed"
)
print(
    f"  Signature: typed contract, structured output — use when pipeline requires structure"
)

print("\n✓ Exercise 5 complete — ReActAgent with tools + BaseAgent with Signature")
