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
    # TODO: Create Delegate(budget_usd=0.5). Prompt listing CATEGORIES, ask for
    # ONLY the category name. Stream and return stripped response.
    ____
    ____


# TODO: Implement run_zero_shot() — iterate sample_docs[:5], call zero_shot_classify,
# print each result, collect into a list and return it.
____
____

zero_shot_results = asyncio.run(run_zero_shot())


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Few-shot with example selection
# ══════════════════════════════════════════════════════════════════════

# TODO: Define FEW_SHOT_EXAMPLES — a list of 5 dicts, one per CATEGORY, each
# with keys 'text' (a representative Singapore business sentence) and 'category'.
____


async def few_shot_classify(text: str) -> str:
    # TODO: Format FEW_SHOT_EXAMPLES as 'Text: "..."\nCategory: ...' prefix,
    # append the new text[:800]. Delegate(budget_usd=0.5), stream, return stripped.
    ____
    ____


# TODO: Implement run_few_shot() — same loop pattern as run_zero_shot but for few-shot.
____
____

few_shot_results = asyncio.run(run_few_shot())


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Chain-of-thought prompting
# ══════════════════════════════════════════════════════════════════════


async def cot_classify(text: str) -> tuple[str, str]:
    # TODO: Build CoT prompt: (1) identify key terms, (2) match to category,
    # (3) state final classification. Delegate(budget_usd=0.5), stream full response.
    # Return (full_reasoning, last_non_empty_line).
    ____
    ____


# TODO: Implement run_cot() — iterate sample_docs[:3], call cot_classify,
# print reasoning excerpt and final category, collect categories into a list.
____
____

cot_results = asyncio.run(run_cot())


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Custom Signature for structured extraction
# ══════════════════════════════════════════════════════════════════════

# TODO: Define ReportExtraction(Signature):
#   InputField:  report_text (str)
#   OutputFields: category (str), key_entities (list[str]),
#                 financial_metrics (list[str]), sentiment (str), confidence (float)
____


async def structured_extract():
    # TODO: Create SimpleQAAgent(signature=ReportExtraction, model=model, budget_usd=1.0).
    # Run on first 3 documents, print structured fields, return results list.
    ____
    ____


structured_results = asyncio.run(structured_extract())


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Compare accuracy across prompting strategies
# ══════════════════════════════════════════════════════════════════════

# TODO: Build a pl.DataFrame comparing the three strategies across doc_1/doc_2/doc_3.
# Columns: strategy, doc_1, doc_2, doc_3. Use zero_shot_results / few_shot_results /
# cot_results (guard with empty-list checks → "N/A"). Print the DataFrame.
____
____

print(
    f"  Zero-shot: fast | Few-shot: consistent | CoT: best for ambiguous | Signature: structured"
)
print("\n✓ Exercise 2 complete — prompt engineering strategies compared")
