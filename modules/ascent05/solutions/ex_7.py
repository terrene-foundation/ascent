# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT05 — Exercise 7: Multi-Agent Orchestration
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Implement formal multi-agent coordination patterns using
#   Delegate agents. Cover four production patterns:
#   supervisor-worker, sequential pipeline, parallel fan-out, and handoff.
#   Each pattern is demonstrated with a practical ML scenario.
#
# TASKS:
#   1. Supervisor-Worker: one coordinator dispatches to specialised workers
#   2. Sequential Pipeline: strict ordering, each agent builds on prior output
#   3. Parallel Fan-out: independent agents run concurrently, results merged
#   4. Handoff Pattern: agent transfers ownership based on conditions
#   5. Choosing the right pattern for ML production scenarios
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os
import time

import polars as pl

from kaizen import Signature, InputField, OutputField
from kaizen_agents import Delegate

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

data_context = (
    f"Singapore Credit Scoring: {credit.height:,} rows, "
    f"{len(credit.columns)} columns, target=default "
    f"({credit['default'].mean():.1%} default rate)"
)

print(f"=== Multi-Agent Orchestration ===")
print(f"Dataset: {data_context}")
print(f"Patterns: supervisor-worker, sequential, parallel, handoff")


# ── Helper: collect Delegate text output ─────────────────────────────


async def run_delegate(delegate: Delegate, prompt: str) -> str:
    """Run a Delegate and collect the full text response."""
    text = ""
    async for event in delegate.run(prompt):
        if hasattr(event, "text"):
            text += event.text
    return text


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Supervisor-Worker Pattern
# ══════════════════════════════════════════════════════════════════════
#
# Pattern: One supervisor decomposes a goal into subtasks, dispatches to
# specialised workers, then synthesizes results. The supervisor has
# global context; workers are domain experts.
#
# ML use case: Model audit — supervisor assigns data quality, bias,
# and performance audits to specialist workers.
#
#   Supervisor
#       ├── Worker A: data quality audit
#       ├── Worker B: bias and fairness audit
#       └── Worker C: model performance audit
#           ↓
#      Supervisor synthesizes
#
# In production, use SupervisorWorkerPattern from kaizen_agents.patterns:
#   from kaizen_agents.patterns.patterns import SupervisorWorkerPattern
#   pattern = SupervisorWorkerPattern(supervisor=supervisor_agent, workers=[...])
#   result = await pattern.execute_task("Audit the model")
#
# Here we implement the pattern manually with Delegate to show the mechanics.


async def pattern_supervisor_worker():
    print(f"\n{'─' * 60}")
    print(f"PATTERN 1: Supervisor-Worker")
    print(f"{'─' * 60}")
    print(f"Scenario: Autonomous model audit before production deployment")

    # --- Define worker agents ---

    data_quality_worker = Delegate(model=model, budget_usd=0.5)
    bias_worker = Delegate(model=model, budget_usd=0.5)
    performance_worker = Delegate(model=model, budget_usd=0.5)

    # --- Dispatch tasks to workers ---
    worker_prompts = {
        "data_quality": (
            f"You are a data quality auditor. Examine this credit dataset and report:\n"
            f"{data_context}\n"
            f"Assess: null rates, outliers, class imbalance, data leakage risk."
        ),
        "bias_fairness": (
            f"You are a fairness auditor. Examine this credit dataset:\n"
            f"{data_context}\n"
            f"Assess: potential protected attributes, disparate impact, demographic parity."
        ),
        "model_performance": (
            f"You are a model performance reviewer. Assess readiness of:\n"
            f"{data_context}\n"
            f"Assess: appropriate metrics (AUC-PR for imbalanced), validation strategy, "
            f"production readiness criteria."
        ),
    }

    workers = {
        "data_quality": data_quality_worker,
        "bias_fairness": bias_worker,
        "model_performance": performance_worker,
    }

    # Run all workers concurrently
    worker_results = {}
    tasks = {
        name: run_delegate(agent, worker_prompts[name])
        for name, agent in workers.items()
    }
    results = await asyncio.gather(*tasks.values())
    worker_results = dict(zip(tasks.keys(), results))

    print(f"\nWorker results collected: {list(worker_results.keys())}")
    for name, output in worker_results.items():
        print(f"\n  [{name}]: {output[:200]}...")

    # --- Supervisor synthesizes ---
    supervisor = Delegate(model=model, budget_usd=0.8)
    synthesis_prompt = (
        f"You are a senior ML engineer supervising a pre-deployment audit.\n"
        f"Goal: Comprehensive pre-deployment audit of Singapore credit scoring model.\n\n"
        + "\n\n".join(
            f"--- {name} audit ---\n{output}" for name, output in worker_results.items()
        )
        + "\n\nSynthesize these audits into: final recommendation, task assignments for next steps, and identified gaps."
    )

    synthesis = await run_delegate(supervisor, synthesis_prompt)
    print(f"\nSupervisor synthesis:")
    print(f"  {synthesis[:400]}...")

    return worker_results, synthesis


supervisor_result = asyncio.run(pattern_supervisor_worker())


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Sequential Pipeline Pattern
# ══════════════════════════════════════════════════════════════════════
#
# Pattern: Strict ordering. Each agent receives the previous agent's
# handoff context and builds on it. No parallelism — order matters.
#
# ML use case: Feature engineering pipeline where each stage depends
# on the prior stage's decisions.
#
#   DataProfiler → FeatureSelector → TransformDesigner → ValidationPlan
#       handoff       →   handoff      →   handoff       → final output
#
# In production, use SequentialPipelinePattern from kaizen_agents.patterns:
#   from kaizen_agents.patterns.patterns import SequentialPipelinePattern
#   pipeline = create_sequential_pipeline(stage_agents)
#   result = await pipeline.execute("Build features for credit scoring")


async def pattern_sequential():
    print(f"\n{'─' * 60}")
    print(f"PATTERN 2: Sequential Pipeline")
    print(f"{'─' * 60}")
    print(f"Scenario: Feature engineering — each stage depends on prior output")

    stage_configs = [
        (
            "data_profiler",
            "Stage 1 — Data Profiler: Profile this credit dataset and identify the "
            "most important characteristics for feature engineering:\n"
            f"{data_context}\n"
            "Output a concise handoff context for the feature selector.",
        ),
        (
            "feature_selector",
            "Stage 2 — Feature Selector: Based on the profiler's findings, select "
            "the most informative features for predicting default. Explain your "
            "selection rationale. Output a handoff context for the transformer.",
        ),
        (
            "transform_designer",
            "Stage 3 — Transform Designer: Design the transformation pipeline for "
            "the selected features. Specify: encoding, scaling, imputation strategies. "
            "Output a handoff context for the validation planner.",
        ),
        (
            "validation_planner",
            "Stage 4 — Validation Planner: Design a validation strategy for this "
            "feature engineering pipeline. Specify: CV strategy, held-out test, "
            "metrics to track. Output the final validation plan.",
        ),
    ]

    # Sequential execution — each stage receives the prior output
    stage_outputs = {}
    context = data_context

    for stage_name, base_prompt in stage_configs:
        agent = Delegate(model=model, budget_usd=0.5)
        prompt = f"Previous stage context:\n{context}\n\n{base_prompt}"
        output = await run_delegate(agent, prompt)
        stage_outputs[stage_name] = output
        context = output  # Handoff to next stage
        print(f"\n  Stage ({stage_name}): {output[:150]}...")

    print(f"\nPipeline stages completed: {len(stage_outputs)}")
    print(f"\nFinal stage output:")
    print(f"  {list(stage_outputs.values())[-1][:300]}...")

    return stage_outputs


sequential_result = asyncio.run(pattern_sequential())


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Parallel Fan-out Pattern
# ══════════════════════════════════════════════════════════════════════
#
# Pattern: Independent tasks dispatched concurrently; results merged.
# Each agent is unaware of the others. Fastest pattern — wall-clock
# time is max(individual times), not sum(individual times).
#
# ML use case: Model comparison — evaluate 3 model architectures in
# parallel and merge results for final selection.
#
#   ┌─ LightGBM analyst ──┐
#   ├─ XGBoost analyst ───┤ → Merge → Final recommendation
#   └─ Neural Net analyst ┘
#
# Parallel fan-out uses asyncio.gather for concurrent execution.
# There is no dedicated ParallelPattern class — concurrency is built
# into Python's asyncio. The pattern is in the structure, not a class.


async def pattern_parallel():
    print(f"\n{'─' * 60}")
    print(f"PATTERN 3: Parallel Fan-out")
    print(f"{'─' * 60}")
    print(f"Scenario: Model architecture comparison — 3 agents run concurrently")

    # Independent agents — same model, different tasks
    parallel_tasks = {
        "lightgbm": (
            Delegate(model=model, budget_usd=0.5),
            (
                f"You are a LightGBM specialist. Assess whether LightGBM is the right "
                f"model for this task:\n{data_context}\n"
                f"Evaluate: strengths, weaknesses, expected AUC-PR, training time, "
                f"interpretability. Give a priority rating (HIGH/MEDIUM/LOW)."
            ),
        ),
        "xgboost": (
            Delegate(model=model, budget_usd=0.5),
            (
                f"You are an XGBoost specialist. Assess whether XGBoost is the right "
                f"model for this task:\n{data_context}\n"
                f"Evaluate: strengths, weaknesses, expected AUC-PR, training time, "
                f"interpretability. Give a priority rating (HIGH/MEDIUM/LOW)."
            ),
        ),
        "neural_net": (
            Delegate(model=model, budget_usd=0.5),
            (
                f"You are a neural network specialist. Assess whether a TabNet/MLP is "
                f"right for this task:\n{data_context}\n"
                f"Evaluate: strengths, weaknesses, expected AUC-PR, training time, "
                f"interpretability. Give a priority rating (HIGH/MEDIUM/LOW)."
            ),
        ),
    }

    # Launch all agents concurrently with asyncio.gather
    t_start = time.perf_counter()
    coros = [run_delegate(agent, prompt) for agent, prompt in parallel_tasks.values()]
    results = await asyncio.gather(*coros)
    t_end = time.perf_counter()
    wall_clock = t_end - t_start

    agent_results = dict(zip(parallel_tasks.keys(), results))

    print(f"\nParallel execution complete:")
    print(f"  Wall-clock time: {wall_clock:.2f}s")
    print(f"  Agents run: {len(agent_results)}")

    print(f"\nAgent results:")
    for agent_name, agent_output in agent_results.items():
        print(f"\n  [{agent_name}]: {agent_output[:200]}...")

    # Merge phase: a synthesis agent picks the winner
    merger = Delegate(model=model, budget_usd=0.5)
    merge_prompt = (
        f"Three ML specialists have assessed model architectures for credit default prediction.\n\n"
        + "\n\n".join(
            f"--- {name} ---\n{output}" for name, output in agent_results.items()
        )
        + "\n\nSynthesize these assessments. Which architecture do you recommend and why?"
    )
    merged_text = await run_delegate(merger, merge_prompt)

    print(f"\nMerged recommendation:")
    print(f"  {merged_text[:300]}...")

    return agent_results, merged_text


parallel_result, merged_recommendation = asyncio.run(pattern_parallel())


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Handoff Pattern
# ══════════════════════════════════════════════════════════════════════
#
# Pattern: An agent evaluates a task and either handles it or transfers
# ownership to a more appropriate specialist. Creates a routing layer
# without a central coordinator.
#
# ML use case: Intelligent request routing — an incoming model query
# is routed to the right specialist based on query type.
#
#   Query → Router Agent
#                ├── HANDLE (within capability)
#                └── HANDOFF → Specialist Agent
#                                  └── Result
#
# In production, use HandoffPattern from kaizen_agents.patterns:
#   from kaizen_agents.patterns.patterns import HandoffPattern
#   pattern = create_handoff_pattern(handoff_agents)
#   result = await pattern.execute("Route this query")


async def pattern_handoff():
    print(f"\n{'─' * 60}")
    print(f"PATTERN 4: Handoff")
    print(f"{'─' * 60}")
    print(f"Scenario: Query routing — route ML questions to the right specialist")

    # Define specialist agents
    specialists = {
        "data_quality": Delegate(model=model, budget_usd=0.5),
        "model_performance": Delegate(model=model, budget_usd=0.5),
        "deployment": Delegate(model=model, budget_usd=0.5),
        "governance": Delegate(model=model, budget_usd=0.5),
    }

    routing_criteria = {
        "data_quality": "data issues, null values, distributions, profiling",
        "model_performance": "AUC, precision, recall, metrics, evaluation",
        "deployment": "serving, latency, InferenceServer, production, scaling",
        "governance": "fairness, bias, compliance, audit, PACT, regulations",
    }

    # Router agent decides which specialist should handle each query
    router = Delegate(model=model, budget_usd=0.3)

    queries = [
        "What is the null rate in the annual_income column?",
        "Why is AUC-PR better than AUC-ROC for this credit dataset?",
        "How do I reduce inference latency below 20ms?",
        "Does the credit model satisfy MAS FEAT requirements?",
    ]

    criteria_text = "\n".join(
        f"  {name}: {desc}" for name, desc in routing_criteria.items()
    )

    print(f"\nRouting {len(queries)} queries:")
    for q in queries:
        # Step 1: Router decides who handles the query
        routing_prompt = (
            f"You are a query router. Route this query to the best specialist.\n\n"
            f"Available specialists:\n{criteria_text}\n\n"
            f"Query: {q}\n\n"
            f"Respond with ONLY the specialist name (one of: {list(specialists.keys())})"
        )
        route_decision = await run_delegate(router, routing_prompt)

        # Find the closest matching specialist
        routed_to = "data_quality"  # default
        for name in specialists:
            if name in route_decision.lower():
                routed_to = name
                break

        # Step 2: Specialist handles the query
        specialist = specialists[routed_to]
        response = await run_delegate(
            specialist,
            f"You are a {routed_to} specialist. Answer this question about "
            f"the Singapore credit scoring dataset:\n{q}\n\nDataset context: {data_context}",
        )

        print(f"\n  Query: {q}")
        print(f"  Routed to: {routed_to}")
        print(f"  Response: {response[:150]}...")

    return specialists


handoff_pattern = asyncio.run(pattern_handoff())


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Pattern selection guide
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'═' * 60}")
print(f"   PATTERN SELECTION GUIDE")
print(f"{'═' * 60}")
print(
    """
┌──────────────────────┬──────────────────────────────────────────┐
│ Pattern              │ When to use                              │
├──────────────────────┼──────────────────────────────────────────┤
│ Supervisor-Worker    │ - Complex goal needing decomposition     │
│                      │ - Workers have different specialisations │
│                      │ - Need global synthesis across workers   │
│                      │ - Example: model audit, research report  │
├──────────────────────┼──────────────────────────────────────────┤
│ Sequential Pipeline  │ - Strict ordering required               │
│                      │ - Each stage needs prior output          │
│                      │ - State must accumulate progressively    │
│                      │ - Example: ETL, feature engineering      │
├──────────────────────┼──────────────────────────────────────────┤
│ Parallel Fan-out     │ - Independent subtasks                   │
│                      │ - Latency matters more than cost         │
│                      │ - Results need merging at end            │
│                      │ - Example: model comparison, search      │
├──────────────────────┼──────────────────────────────────────────┤
│ Handoff              │ - Heterogeneous query types              │
│                      │ - Single entry point, multiple handlers  │
│                      │ - Agents know their own limits           │
│                      │ - Example: support routing, triage       │
└──────────────────────┴──────────────────────────────────────────┘

Production pattern classes (kaizen_agents.patterns.patterns):
  SupervisorWorkerPattern — managed supervisor + worker lifecycle
  SequentialPipelinePattern — stage-by-stage with handoff context
  HandoffPattern — agent-to-agent ownership transfer

Cost comparison (approx for this exercise):
  Supervisor-Worker: N_workers x worker_cost + supervisor_cost
  Sequential:        Sum of all stage costs (sequential billing)
  Parallel:          Max of individual costs (wall-clock savings)
  Handoff:           router_cost + 1 x specialist_cost

In Module 6, all patterns are wrapped with PACT governance:
  -> Each agent gets a frozen GovernanceContext
  -> Supervisor cannot grant workers more than its own permissions
  -> Every handoff is logged in the AuditChain
"""
)

print("✓ Exercise 7 complete — four multi-agent orchestration patterns")
