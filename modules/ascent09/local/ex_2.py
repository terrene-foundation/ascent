# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT09 — Exercise 2: Prompt Engineering
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Master zero-shot, few-shot, and chain-of-thought prompting
#   via Delegate, comparing structured output quality across strategies.
#
# TASKS:
#   1. Zero-shot classification with Delegate
#   2. Few-shot with example selection
#   3. Chain-of-thought prompting
#   4. Build a custom Signature for structured extraction
#   5. Compare accuracy across prompting strategies
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os

import polars as pl

from kaizen_agents import Delegate
from kaizen import Signature, InputField, OutputField
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
sample_docs = reports.head(10)
print(f"Loaded {reports.height:,} documents")

CATEGORIES = ["Financial", "Technology", "Healthcare", "Real Estate", "Manufacturing"]


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Zero-shot classification with Delegate
# ══════════════════════════════════════════════════════════════════════


async def zero_shot_classify(text: str) -> str:
    # TODO: Create Delegate(budget_usd=0.5). Prompt it with CATEGORIES and the
    # text[:800], asking for ONLY the category name. Stream and return stripped response.
    ____
    ____


async def run_zero_shot():
    print(f"\n=== Zero-Shot Classification ===")
    results = []
    for i, text in enumerate(sample_docs.select("text").to_series().to_list()[:5]):
        category = await zero_shot_classify(text)
        print(f"  Doc {i+1}: {category}")
        results.append(category)
    return results


zero_shot_results = asyncio.run(run_zero_shot())


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Few-shot with example selection
# ══════════════════════════════════════════════════════════════════════

FEW_SHOT_EXAMPLES = [
    {
        "text": "Revenue increased 15% driven by strong loan growth and net interest margin expansion.",
        "category": "Financial",
    },
    {
        "text": "The company launched its new cloud-native SaaS platform serving enterprise clients across APAC.",
        "category": "Technology",
    },
    {
        "text": "Clinical trials for the new oncology drug showed 40% improvement in patient outcomes.",
        "category": "Healthcare",
    },
    {
        "text": "The integrated township development in Jurong added 2,500 residential units to the portfolio.",
        "category": "Real Estate",
    },
    {
        "text": "Factory automation reduced production cycle time by 30% at the Tuas semiconductor fab.",
        "category": "Manufacturing",
    },
]


async def few_shot_classify(text: str) -> str:
    # TODO: Format FEW_SHOT_EXAMPLES as 'Text: "..."\nCategory: ...' prefix.
    # Create Delegate(budget_usd=0.5), stream response, return stripped category.
    ____
    ____


async def run_few_shot():
    print(f"\n=== Few-Shot Classification ===")
    results = []
    for i, text in enumerate(sample_docs.select("text").to_series().to_list()[:5]):
        category = await few_shot_classify(text)
        print(f"  Doc {i+1}: {category}")
        results.append(category)
    return results


few_shot_results = asyncio.run(run_few_shot())


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Chain-of-thought prompting
# ══════════════════════════════════════════════════════════════════════


async def cot_classify(text: str) -> tuple[str, str]:
    # TODO: Build a CoT prompt instructing: (1) identify key terms, (2) match
    # to category, (3) state final classification. Create Delegate(budget_usd=0.5),
    # stream full response. Return (full_reasoning, last_non_empty_line).
    ____
    ____


async def run_cot():
    print(f"\n=== Chain-of-Thought Classification ===")
    results = []
    for i, text in enumerate(sample_docs.select("text").to_series().to_list()[:3]):
        reasoning, category = await cot_classify(text)
        print(f"  Doc {i+1} reasoning: {reasoning[:150]}... → {category}")
        results.append(category)
    return results


cot_results = asyncio.run(run_cot())


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Custom Signature for structured extraction
# ══════════════════════════════════════════════════════════════════════

# TODO: Define ReportExtraction(Signature) with:
#   InputField:  report_text (str)
#   OutputFields: category (str), key_entities (list[str]),
#                 financial_metrics (list[str]), sentiment (str), confidence (float)
# Precise field descriptions drive output quality — the Signature is the LLM contract.
____


async def structured_extract():
    # TODO: Create SimpleQAAgent(signature=ReportExtraction, model=model, budget_usd=1.0).
    # Run it on the first 3 documents and print the structured fields.
    ____
    ____


structured_results = asyncio.run(structured_extract())


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Compare accuracy across prompting strategies
# ══════════════════════════════════════════════════════════════════════

comparison = pl.DataFrame(
    {
        "strategy": ["Zero-Shot", "Few-Shot", "Chain-of-Thought"],
        "doc_1": [
            zero_shot_results[0] if zero_shot_results else "N/A",
            few_shot_results[0] if few_shot_results else "N/A",
            cot_results[0] if cot_results else "N/A",
        ],
        "doc_2": [
            zero_shot_results[1] if len(zero_shot_results) > 1 else "N/A",
            few_shot_results[1] if len(few_shot_results) > 1 else "N/A",
            cot_results[1] if len(cot_results) > 1 else "N/A",
        ],
        "doc_3": [
            zero_shot_results[2] if len(zero_shot_results) > 2 else "N/A",
            few_shot_results[2] if len(few_shot_results) > 2 else "N/A",
            cot_results[2] if len(cot_results) > 2 else "N/A",
        ],
    }
)
print(f"\n=== Strategy Comparison ===")
print(comparison)
print(
    f"  Zero-shot: fast; Few-shot: consistent; CoT: best for ambiguous; Signature: structured"
)

print("\n✓ Exercise 2 complete — prompt engineering strategies compared")
