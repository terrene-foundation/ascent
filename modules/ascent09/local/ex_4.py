# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT09 — Exercise 4: Constitutional AI and Self-Critique
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Implement the Constitutional AI (CAI) critique-revision loop
#   with Singapore-specific principles (MAS AML, PDPA). Build a multi-stage
#   pipeline that critiques and revises LLM outputs for regulatory compliance.
#
# TASKS:
#   1. Define constitution with Singapore-specific principles
#   2. Implement critique step using Delegate
#   3. Implement revision step with sequential principle application
#   4. Build critique-revision pipeline and run on test prompts
#   5. Evaluate original vs revised responses via Signature
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os

import polars as pl

from kaizen_agents import Delegate
from kaizen import Signature, InputField, OutputField

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()

_raw_llm = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))
llm_model = _raw_llm if os.environ.get("OPENAI_API_KEY") else None

# ── Data Loading ─────────────────────────────────────────────────────

loader = ASCENTDataLoader()
reports = loader.load("ascent09", "sg_company_reports.parquet")

print(f"=== SG Company Reports Dataset ===")
print(f"Shape: {reports.shape}")
print(f"Columns: {reports.columns}")
print(reports.head(3))


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Define constitution with Singapore-specific principles
# ══════════════════════════════════════════════════════════════════════
# CAI (Anthropic, 2022): replace human feedback with principles.
# Pipeline: generate -> critique per principle -> revise if violated
# Use (original, revised) pairs as DPO training data (RLAIF).

# TODO: Define 4 principles. Each dict has "id", "name", "principle", "critique_prompt".
#   1. PDPA_COMPLIANCE: consent, purpose limitation, notification, data minimization
#   2. MAS_AML: KYC, suspicious transaction reporting, terrorism financing
#   3. FAIRNESS: non-discrimination (race, religion, nationality, gender), Tripartite Guidelines
#   4. ACCURACY: factual correctness for SG context, uncertainty qualifiers, authoritative sources
# "principle" = the rule text; "critique_prompt" = specific checklist items
CONSTITUTION = [
    {"id": ____, "name": ____, "principle": ____, "critique_prompt": ____},
    {"id": ____, "name": ____, "principle": ____, "critique_prompt": ____},
    {"id": ____, "name": ____, "principle": ____, "critique_prompt": ____},
    {"id": ____, "name": ____, "principle": ____, "critique_prompt": ____},
]

print(f"\n=== Singapore Constitutional Principles ===")
for i, p in enumerate(CONSTITUTION, 1):
    print(f"  {i}. [{p['id']}] {p['name']}")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Implement critique step using Delegate
# ══════════════════════════════════════════════════════════════════════


class CritiqueResult(Signature):
    """Structured critique against a constitutional principle."""

    response: str = InputField(desc="The response to critique")
    principle: str = InputField(desc="The principle to check")
    critique_prompt: str = InputField(desc="Specific items to check")
    violates: str = OutputField(desc="YES or NO")
    explanation: str = OutputField(desc="Detailed explanation")
    severity: str = OutputField(desc="LOW, MEDIUM, or HIGH")


async def critique_response(
    response: str,
    principle: dict,
    delegate: Delegate,
) -> dict:
    """Critique a response against one constitutional principle."""
    # TODO: Build prompt with: role (compliance reviewer), principle text,
    #   critique checklist, the response, instructions for VIOLATES/EXPLANATION/SEVERITY
    prompt = ____

    # TODO: Stream delegate.run(prompt), collect result_text
    result_text = ""
    ____

    # TODO: Parse violates (YES/NO before EXPLANATION section) and severity
    violates = ____
    severity = ____

    # TODO: Return dict with principle_id, violates, severity, explanation (truncated)
    return ____


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Implement revision step with sequential principle application
# ══════════════════════════════════════════════════════════════════════


async def revise_response(
    original_response: str,
    critique: dict,
    principle: dict,
    delegate: Delegate,
) -> str:
    """Revise response to satisfy a constitutional principle."""
    # TODO: Build prompt with original, violated principle, critique explanation.
    #   Ask to rewrite fixing compliance issues while preserving intent.
    prompt = ____

    # TODO: Stream delegate.run(prompt), return revised text
    revised = ""
    ____
    return revised


async def constitutional_pipeline(
    response: str,
    constitution: list[dict],
    delegate: Delegate,
) -> dict:
    """Full CAI critique-revision pipeline (sequential per principle)."""
    current_response = response
    critiques = []
    revisions = []

    for principle in constitution:
        # TODO: Critique current_response, append to critiques
        # If violated: revise, record before/after, update current_response
        ____
        ____
        ____

    # TODO: Return dict with original, final, critiques, revisions, n_violations, n_revisions
    return ____


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Run critique-revision pipeline on test prompts
# ══════════════════════════════════════════════════════════════════════

# TODO: Create 3 test prompts with intentionally non-compliant initial_responses:
#   1. "How can I open a bank account without ID?" -> suggests using friend's docs
#   2. "What customer data should I collect?" -> advises collecting everything without consent
#   3. "How to report suspicious transactions?" -> suggests handling internally
test_prompts = ____


async def run_pipeline():
    """Run CAI pipeline on test prompts."""
    if not llm_model:
        print("\n=== CAI Pipeline (API key not set -- showing pattern) ===")
        print(
            f"  result = await constitutional_pipeline(response, CONSTITUTION, delegate)"
        )
        return []

    # TODO: Create Delegate, loop over test_prompts, run constitutional_pipeline on each,
    #   print violations/revisions per test, return list of results
    delegate = ____
    results = []
    ____
    return results


pipeline_results = asyncio.run(run_pipeline())


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Evaluate original vs revised responses via Signature
# ══════════════════════════════════════════════════════════════════════


class ComplianceEval(Signature):
    """Evaluate regulatory compliance (1-5 scale)."""

    prompt: str = InputField(desc="The original question")
    response: str = InputField(desc="The response to evaluate")
    compliance_score: str = OutputField(desc="Score 1-5")
    issues: str = OutputField(desc="Compliance issues found")


async def evaluate_compliance():
    """Score original vs revised responses."""
    if not llm_model or not pipeline_results:
        print("\n=== Compliance Evaluation (scoring rubric) ===")
        print(f"  1=major violations, 2=significant, 3=minor, 4=mostly, 5=fully")
        return

    # TODO: Create Delegate, for each (test, result) pair:
    #   build scoring prompt for original and revised, stream, parse 1-5 score
    #   print improvement (orig_score -> rev_score)
    delegate = ____
    ____


asyncio.run(evaluate_compliance())

print(f"\n=== Constitutional AI Summary ===")
print(f"  1. Define principles (constitution) for your domain")
print(f"  2. Critique responses against each principle")
print(f"  3. Revise violated responses sequentially")
print(f"  4. Train with DPO on (original, revised) pairs")
print(f"  Principles: {[p['id'] for p in CONSTITUTION]}")

print("\n=== Exercise 4 complete ===")
