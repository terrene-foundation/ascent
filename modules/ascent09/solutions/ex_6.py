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


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Build 3 specialist agents
# ══════════════════════════════════════════════════════════════════════

loader = ASCENTDataLoader()
reports = loader.load("ascent09", "sg_company_reports.parquet")

print(f"=== Company Reports: {reports.height} documents ===")
sample_doc = reports["text"][0][:500]
print(f"Sample (first 500 chars): {sample_doc}...")


class FinancialAnalysisSignature(Signature):
    """Analyse financial aspects of a business document."""

    document: str = InputField(description="Business document text")
    question: str = InputField(description="Financial analysis question")
    revenue_insights: str = OutputField(
        description="Revenue and profitability analysis"
    )
    risk_factors: list[str] = OutputField(
        description="Financial risk factors identified"
    )
    recommendation: str = OutputField(description="Financial recommendation")


class LegalAnalysisSignature(Signature):
    """Analyse legal and compliance aspects of a business document."""

    document: str = InputField(description="Business document text")
    question: str = InputField(description="Legal analysis question")
    compliance_issues: list[str] = OutputField(description="Compliance issues found")
    regulatory_references: list[str] = OutputField(description="Relevant regulations")
    legal_risk: str = OutputField(description="Legal risk assessment")


class TechnicalAnalysisSignature(Signature):
    """Analyse technical aspects of a business document."""

    document: str = InputField(description="Business document text")
    question: str = InputField(description="Technical analysis question")
    tech_assessment: str = OutputField(description="Technical feasibility assessment")
    architecture_notes: list[str] = OutputField(
        description="Architecture considerations"
    )
    scalability: str = OutputField(description="Scalability assessment")


financial_agent = SimpleQAAgent(model=model)
legal_agent = SimpleQAAgent(model=model)
technical_agent = SimpleQAAgent(model=model)

print(f"\n=== Specialist Agents ===")
print(f"  1. FinancialAgent: revenue, risk, profitability")
print(f"  2. LegalAgent: compliance, regulation, legal risk")
print(f"  3. TechnicalAgent: architecture, scalability, feasibility")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Create router agent via Pipeline.router()
# ══════════════════════════════════════════════════════════════════════

# Pipeline.router() uses LLM reasoning to route queries to the best specialist
# — NOT keyword matching. The LLM reads each agent's description (capability card)
# and reasons about which specialist is best suited for each query.
router = Pipeline.router(
    agents=[financial_agent, legal_agent, technical_agent],
)

print(f"\n=== Router Agent ===")
print(f"Pipeline.router() uses LLM reasoning to select the best specialist.")
print(f"Each agent's description serves as a capability card.")
print(f"The router examines the query and agent descriptions, then delegates.")
print(f"This is fundamentally different from keyword matching or dispatch tables.")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Supervisor pattern with delegation
# ══════════════════════════════════════════════════════════════════════


class SupervisorSignature(Signature):
    """Coordinate specialist analyses into a unified assessment."""

    document: str = InputField(description="Original business document")
    financial_analysis: str = InputField(description="Financial specialist's analysis")
    legal_analysis: str = InputField(description="Legal specialist's analysis")
    technical_analysis: str = InputField(description="Technical specialist's analysis")
    executive_summary: str = OutputField(description="Unified executive summary")
    overall_risk: str = OutputField(description="Overall risk rating: low/medium/high")
    action_items: list[str] = OutputField(description="Prioritized action items")


supervisor = SimpleQAAgent(model=model)

print(f"\n=== Supervisor Pattern ===")
print(f"1. Router dispatches query to appropriate specialist")
print(f"2. Each specialist analyses from their domain perspective")
print(f"3. Supervisor synthesizes all analyses into unified assessment")
print(f"This is fan-out (parallel specialists) → fan-in (supervisor synthesis)")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Run multi-agent analysis
# ══════════════════════════════════════════════════════════════════════


async def multi_agent_analysis():
    doc = reports["text"][0]
    question = "Should we invest in this company's expansion plans?"

    print(f"\n=== Multi-Agent Analysis ===")
    print(f"Question: {question}")

    # Step 1: Run specialists in parallel
    doc_excerpt = doc[:2000]
    financial_result = financial_agent.run(
        f"Analyse the financial health and revenue potential of this document:\n\n{doc_excerpt}"
    )
    legal_result = legal_agent.run(
        f"Identify compliance risks and regulatory requirements in this document:\n\n{doc_excerpt}"
    )
    technical_result = technical_agent.run(
        f"Assess technical feasibility and scalability from this document:\n\n{doc_excerpt}"
    )

    fin_text = str(financial_result)[:300]
    leg_text = str(legal_result)[:300]
    tech_text = str(technical_result)[:300]

    print(f"\n--- Financial Analysis ---")
    print(f"Result: {fin_text}...")

    print(f"\n--- Legal Analysis ---")
    print(f"Result: {leg_text}...")

    print(f"\n--- Technical Analysis ---")
    print(f"Result: {tech_text}...")

    # Step 2: Supervisor synthesizes
    supervisor_result = supervisor.run(
        f"Synthesize these specialist analyses into an executive summary with risk rating:\n\n"
        f"Financial: {fin_text}\n\nLegal: {leg_text}\n\nTechnical: {tech_text}"
    )

    print(f"\n--- Supervisor Summary ---")
    print(f"Result: {str(supervisor_result)[:400]}...")

    return supervisor_result


multi_result = asyncio.run(multi_agent_analysis())


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Compare single-agent vs multi-agent
# ══════════════════════════════════════════════════════════════════════


async def single_agent_comparison():
    doc = reports["text"][0]
    delegate = Delegate(model=model, budget_usd=3.0)

    print(f"\n=== Single-Agent Comparison ===")
    response = ""
    async for event in delegate.run(
        f"Analyse this business document from financial, legal, and technical perspectives. "
        f"Provide an executive summary with risk assessment.\n\n{doc[:2000]}"
    ):
        if hasattr(event, "text"):
            response += event.text

    print(f"Single-agent response: {response[:400]}...")
    return response


single_result = asyncio.run(single_agent_comparison())

print(f"\n=== Comparison ===")
print(f"Single-agent: one LLM call, broad but shallow analysis")
print(f"Multi-agent:  3 specialists + supervisor, deep domain-specific insights")
print(f"\nWhen to use multi-agent:")
print(f"  - Task requires multiple domain expertise areas")
print(f"  - Deep analysis needed per domain (not just surface-level)")
print(f"  - Quality matters more than latency")
print(f"  - Audit trail needed (which specialist said what)")

print("\n✓ Exercise 6 complete — multi-agent orchestration with Pipeline.router()")
