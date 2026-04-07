# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT05 — Exercise 1: LLM Fundamentals and Kaizen
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Use Delegate for autonomous data analysis Q&A and build a
#   SimpleQAAgent with custom Signature for structured answers. Set
#   budget_usd from Exercise 1 — mandatory for all M5 exercises.
#
# TASKS:
#   1. Set up Delegate with cost budget governance
#   2. Run Delegate on data analysis questions
#   3. Build SimpleQAAgent with custom Signature (InputField/OutputField)
#   4. Compare Delegate vs SimpleQAAgent outputs
#   5. Track LLM costs and demonstrate budget enforcement
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os

import polars as pl
from dotenv import load_dotenv

from kaizen_agents import Delegate
from kaizen import Signature, InputField, OutputField
from kaizen_agents.agents.specialized.simple_qa import SimpleQAAgent

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()

model = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))
if not model or not os.environ.get("OPENAI_API_KEY"):
    print("Set OPENAI_API_KEY and DEFAULT_LLM_MODEL in .env to run this exercise")
    raise SystemExit(0)
print(f"LLM Model: {model}")


# ── Data Loading ──────────────────────────────────────────────────────

loader = ASCENTDataLoader()
customers = loader.load("ascent04", "ecommerce_customers.parquet")

data_summary = f"""
E-commerce Customer Dataset:
- {customers.height:,} customers
- Columns: {', '.join(customers.columns)}
- Revenue range: ${customers['total_revenue'].min():.0f} - ${customers['total_revenue'].max():.0f}
- Average orders: {customers['order_count'].mean():.1f}
"""

print(f"=== Data Context ===")
print(data_summary)


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Delegate with cost budget
# ══════════════════════════════════════════════════════════════════════


async def delegate_analysis():
    """Use Delegate for autonomous data analysis Q&A."""

    # TODO: Create Delegate with model and budget_usd=2.0 (MANDATORY for all M5 exercises)
    delegate = ____  # Hint: Delegate(model=model, budget_usd=2.0)

    questions = [
        "Based on this customer data, what are the top 3 customer segments you'd recommend for targeted marketing?",
        "What metrics should we track to predict customer churn from this dataset?",
        "If we wanted to build a recommendation system, which features would be most important?",
    ]

    print(f"\n=== Delegate Analysis (budget: $2.00) ===")
    results = []
    for i, question in enumerate(questions):
        prompt = f"{data_summary}\n\nQuestion: {question}"
        print(f"\nQ{i+1}: {question}")

        # TODO: Stream Delegate response — async for event in delegate.run(prompt);
        #   accumulate event.text if hasattr(event, "text")
        response_text = ""
        ____
        ____
        ____

        print(f"A{i+1}: {response_text[:300]}...")
        results.append({"question": question, "answer": response_text})

    return delegate, results


delegate, delegate_results = asyncio.run(delegate_analysis())


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Custom Signature for structured output
# ══════════════════════════════════════════════════════════════════════


# TODO: Define CustomerSegmentAnalysis(Signature) with:
#   InputField: data_context (str), question (str)
#   OutputField: segments (list[str]), reasoning (str),
#                confidence (float), next_steps (list[str])
____
____
____
____
____
____
____
____


# TODO: Define ChurnPrediction(Signature) with:
#   InputField: data_context (str)
#   OutputField: risk_factors (list[str]), key_metrics (list[str]),
#                model_recommendation (str)
____
____
____
____
____
____


# ══════════════════════════════════════════════════════════════════════
# TASK 3: SimpleQAAgent with custom Signature
# ══════════════════════════════════════════════════════════════════════


def structured_analysis():
    """Use SimpleQAAgent for structured output."""

    # TODO: Create SimpleQAAgent(model=model) for segment analysis;
    #   run with question="What customer segments should we target for a premium loyalty programme?"
    #   and context=data_summary; store result in segment_result
    segment_agent = ____  # Hint: SimpleQAAgent(model=model)
    segment_result = ____  # Hint: segment_agent.run(question=..., context=data_summary)

    print(f"\n=== Structured Segment Analysis ===")
    for key, value in segment_result.items():
        print(f"  {key}: {str(value)[:200]}")

    # TODO: Create a second SimpleQAAgent(model=model) for churn analysis;
    #   run with a question about top churn risk factors and metrics;
    #   store result in churn_result
    churn_agent = ____  # Hint: SimpleQAAgent(model=model)
    churn_result = ____  # Hint: churn_agent.run(question=..., context=data_summary)

    print(f"\n=== Structured Churn Analysis ===")
    for key, value in churn_result.items():
        print(f"  {key}: {str(value)[:200]}")

    return segment_result, churn_result


segment_result, churn_result = structured_analysis()


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Compare outputs
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Delegate vs SimpleQAAgent ===")
print(f"Delegate: free-form text, flexible, unpredictable structure")
print(f"SimpleQA: typed Signature, structured output, guaranteed fields")
print(f"\nWhen to use each:")
print(f"  Delegate: exploratory analysis, open-ended questions")
print(f"  SimpleQA: production pipelines requiring structured data")
print(f"  → Signature = contract. Like ModelSignature for models, but for agents.")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Cost tracking
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== LLM Cost Governance ===")
print(f"budget_usd is MANDATORY for all M5 exercises.")
print(f"CO methodology: human-on-the-loop, not in-the-loop.")
print(f"The budget cap ensures agents cannot run away with API costs.")
print(f"\nIn production:")
print(f"  1. Set budget_usd per agent based on expected task complexity")
print(f"  2. Monitor actual spend vs budget")
print(f"  3. Alert if spend approaches limit")
print(f"  4. Module 6: PACT GovernanceEngine formalises cost envelopes")

print("\n✓ Exercise 1 complete — Delegate + SimpleQAAgent with cost governance")
