# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT05 — Exercise 2: Chain-of-Thought Reasoning
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Build a ChainOfThoughtAgent that reasons step-by-step about
#   Module 4 clustering results. Compare CoT vs direct answering.
#
# TASKS:
#   1. Build ChainOfThoughtAgent for cluster interpretation
#   2. Provide clustering results as context
#   3. Agent explains WHY segments formed, not just WHAT
#   4. Compare CoT vs direct answer quality
#   5. Evaluate reasoning chain quality
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os

import polars as pl
from dotenv import load_dotenv

from kaizen import Signature, InputField, OutputField
from kaizen_agents import Delegate
from kaizen_agents.agents.specialized.chain_of_thought import ChainOfThoughtAgent

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()

model = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))
if not model or not os.environ.get("OPENAI_API_KEY"):
    print("Set OPENAI_API_KEY and DEFAULT_LLM_MODEL in .env to run this exercise")
    raise SystemExit(0)


# ── Prepare clustering context from M4 ───────────────────────────────

loader = ASCENTDataLoader()
customers = loader.load("ascent04", "ecommerce_customers.parquet")

cluster_summary = """
Clustering Results (K-means, K=4, Silhouette=0.42):

Cluster 0 (n=12,500 — "Casual Browsers"):
  avg_revenue: $45, avg_orders: 1.2, avg_session_duration: 3.5min
  ▼ revenue, ▼ orders, ─ sessions

Cluster 1 (n=8,200 — "Power Shoppers"):
  avg_revenue: $890, avg_orders: 15.3, avg_session_duration: 12.1min
  ▲ revenue, ▲ orders, ▲ sessions

Cluster 2 (n=18,000 — "Window Shoppers"):
  avg_revenue: $120, avg_orders: 3.1, avg_session_duration: 22.5min
  ─ revenue, ─ orders, ▲ sessions (high browse, low buy)

Cluster 3 (n=11,300 — "Bargain Hunters"):
  avg_revenue: $210, avg_orders: 8.7, avg_session_duration: 8.3min
  ─ revenue, ▲ orders (high frequency, low basket)
"""


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Define Signature for cluster interpretation
# ══════════════════════════════════════════════════════════════════════


# TODO: Define ClusterInterpretation(Signature) with:
#   InputField:  cluster_data (str), question (str)
#   OutputField: reasoning_steps (list[str]), interpretation (str),
#                actionable_insights (list[str]), confidence (float)
____
____
____
____
____
____
____
____


# ══════════════════════════════════════════════════════════════════════
# TASK 2: ChainOfThoughtAgent
# ══════════════════════════════════════════════════════════════════════


async def cot_analysis():
    # TODO: Implement cot_analysis():
    #   1. Create ChainOfThoughtAgent(model=model)
    #   2. Define three interpretation questions (why clusters differ, ROI opportunity, retention target)
    #   3. For each question: call cot_agent.run(problem=q, context=cluster_summary) — sync call
    #   4. Extract reasoning, answer, confidence from result dict (check both key variants)
    #   5. Print Q/Reasoning/Answer/Confidence; append result to list
    #   6. Return results list
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


cot_results = asyncio.run(cot_analysis())


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Compare with direct answering (no CoT)
# ══════════════════════════════════════════════════════════════════════


async def direct_analysis():
    # TODO: Implement direct_analysis():
    #   1. Create Delegate(model=model, budget_usd=1.0)
    #   2. Build prompt asking the cluster question WITHOUT step-by-step reasoning
    #   3. Stream the Delegate response into direct_answer (async for / event.text)
    #   4. Print and return direct_answer[:500]
    ____
    ____
    ____
    ____
    ____
    ____
    ____


direct_answer = asyncio.run(direct_analysis())


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Evaluate reasoning quality
# ══════════════════════════════════════════════════════════════════════

# TODO: Print a CoT vs Direct comparison showing:
#   - Extract CoT reasoning from cot_results[0] (check "reasoning_steps" or "reasoning" key)
#   - Print first 200 chars of reasoning
#   - Explain: CoT has structured output (chain, answer, confidence); Direct is free-form
#   - Explain why CoT matters: debugging, trust, reproducibility, Module 6 AuditChain
____
____
____
____
____
____
____
____
____

print("\n✓ Exercise 2 complete — ChainOfThoughtAgent with reasoning evaluation")
