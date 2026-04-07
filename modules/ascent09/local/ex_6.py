# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT09 — Exercise 6: Multi-Agent Orchestration
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Build a multi-agent system with delegation — router,
#   specialist agents, supervisor pattern — for complex document analysis.
#
# TASKS:
#   1. Build 3 specialist agents (financial, legal, technical)
#   2. Create router agent via Pipeline.router()
#   3. Implement supervisor pattern with delegation
#   4. Run multi-agent analysis on complex document
#   5. Compare single-agent vs multi-agent quality
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os

import polars as pl

from kaizen import InputField, OutputField, Signature
from kaizen_agents import Delegate, Pipeline
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

loader = ASCENTDataLoader()
reports = loader.load("ascent09", "sg_company_reports.parquet")
print(f"=== Company Reports: {reports.height} documents ===")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Build 3 specialist agents
# ══════════════════════════════════════════════════════════════════════

# TODO: Define three Signatures: FinancialAnalysisSignature, LegalAnalysisSignature,
# TechnicalAnalysisSignature. Each needs:
#   InputFields: document (str), question (str)
#   OutputFields: 2-3 domain-specific fields (str or list[str])
# The Signature docstring is the capability card — Pipeline.router() reads it to
# choose the right specialist for each query. Be precise about each specialist's domain.
# Then instantiate one SimpleQAAgent(model=model) per specialist.
____
____
____
____
____
____

print(f"\n=== Specialist Agents: financial, legal, technical ===")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Create router agent via Pipeline.router()
# ══════════════════════════════════════════════════════════════════════

# TODO: Create router = Pipeline.router(agents=[financial_agent, legal_agent, technical_agent]).
# The router uses LLM reasoning — not keyword matching — to select the best
# specialist by reading each agent's Signature docstring as a capability card.
____

print(f"\n=== Router: LLM-based specialist selection (not keyword dispatch) ===")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Supervisor pattern with delegation
# ══════════════════════════════════════════════════════════════════════

# TODO: Define SupervisorSignature(Signature) with:
#   InputFields: document (str), financial_analysis (str),
#                legal_analysis (str), technical_analysis (str)
#   OutputFields: executive_summary (str), overall_risk (str), action_items (list[str])
# Then instantiate supervisor = SimpleQAAgent(model=model).
# The supervisor is the fan-in node: it synthesises all three specialist outputs.
____
____

print(f"\n=== Supervisor: fan-out (3 specialists) → fan-in (unified assessment) ===")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Run multi-agent analysis
# ══════════════════════════════════════════════════════════════════════


async def multi_agent_analysis():
    """Run all three specialists on doc[:2000], then synthesise with the supervisor.
    Step 1: run financial_agent, legal_agent, technical_agent with domain prompts.
    Step 2: pass all three results to supervisor for unified synthesis.
    Print each specialist's truncated result and the supervisor summary.
    """
    # TODO: Extract doc excerpt, run each specialist with a domain-specific prompt,
    # collect results, pass to supervisor, print section headers at each stage.
    ____
    ____
    ____
    ____


multi_result = asyncio.run(multi_agent_analysis())


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Compare single-agent vs multi-agent
# ══════════════════════════════════════════════════════════════════════


async def single_agent_comparison():
    # TODO: Create Delegate(model=model, budget_usd=3.0). Send a single prompt
    # asking for financial, legal, and technical analysis of the same document.
    # Stream and print the response to compare depth vs multi-agent.
    ____
    ____


single_result = asyncio.run(single_agent_comparison())

print(f"\n=== Comparison ===")
print(f"  Single-agent: one LLM call, broad but shallow")
print(f"  Multi-agent: 3 specialists + supervisor, deep domain insights + audit trail")
print(
    f"  Use multi-agent when: multiple domains needed, quality > latency, auditability required"
)

print("\n✓ Exercise 6 complete — multi-agent orchestration with Pipeline.router()")
