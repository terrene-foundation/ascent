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
    # TODO: Implement run_delegate — iterate async events from delegate.run(prompt)
    #   and accumulate any event.text into a string; return the full text
    text = ""
    ____  # Hint: async for event in delegate.run(prompt): if hasattr(event, "text"): text += event.text
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

    # TODO: Implement the Supervisor-Worker pattern:
    #   1. Create three Delegate workers with budget_usd=0.5 each:
    #      data_quality_worker, bias_worker, performance_worker
    #   2. Build worker_prompts dict (keys: data_quality, bias_fairness, model_performance)
    #      Each prompt asks a different specialist to audit the credit model using data_context
    #   3. Build workers dict mapping names to agents
    #   4. Run all workers concurrently via asyncio.gather(*[run_delegate(agent, prompt) ...])
    #      zip results back to names → worker_results dict
    #   5. Print worker results (first 200 chars each)
    #   6. Create supervisor Delegate with budget_usd=0.8; build synthesis_prompt from worker outputs
    #   7. Run supervisor via run_delegate; print synthesis[:400]; return (worker_results, synthesis)
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
    ____
    ____
    ____
    ____


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

    # TODO: Implement the Sequential Pipeline pattern:
    #   1. Build stage_configs list of (stage_name, base_prompt) for 4 stages:
    #      data_profiler, feature_selector, transform_designer, validation_planner
    #      Each prompt instructs one specialist stage and asks for a handoff context
    #   2. Start with context = data_context
    #   3. For each stage: create Delegate(budget_usd=0.5), build prompt with previous context,
    #      run via run_delegate, store output, update context = output (handoff)
    #   4. Print stage count; print each stage output[:150]; print final stage output[:300]
    #   5. Return stage_outputs dict
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
    ____


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

    # TODO: Implement Parallel Fan-out:
    #   1. Build parallel_tasks dict with keys "lightgbm", "xgboost", "neural_net"
    #      Each value is (Delegate(budget_usd=0.5), specialist_prompt) — prompt asks
    #      each specialist to evaluate their model for the credit task and give HIGH/MEDIUM/LOW
    #   2. Time the wall-clock: record t_start = time.perf_counter()
    #   3. Build coros = [run_delegate(agent, prompt) for agent, prompt in parallel_tasks.values()]
    #   4. results = await asyncio.gather(*coros); t_end = time.perf_counter()
    #   5. agent_results = dict(zip(parallel_tasks.keys(), results))
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

    print(f"\nParallel execution complete:")
    print(f"  Wall-clock time: {wall_clock:.2f}s")
    print(f"  Agents run: {len(agent_results)}")

    print(f"\nAgent results:")
    for agent_name, agent_output in agent_results.items():
        print(f"\n  [{agent_name}]: {agent_output[:200]}...")

    # TODO: Create a merger Delegate with budget_usd=0.5 and collect its synthesis
    merger = ____  # Hint: Delegate(model=model, budget_usd=0.5)
    merge_prompt = (
        f"Three ML specialists have assessed model architectures for credit default prediction.\n\n"
        + "\n\n".join(
            f"--- {name} ---\n{output}" for name, output in agent_results.items()
        )
        + "\n\nSynthesize these assessments. Which architecture do you recommend and why?"
    )
    # TODO: Run the merger agent using run_delegate
    merged_text = ____  # Hint: await run_delegate(merger, merge_prompt)

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

    # TODO: Implement the Handoff pattern:
    #   1. Create specialists dict with four Delegate agents (budget_usd=0.5 each):
    #      keys: data_quality, model_performance, deployment, governance
    #   2. Build routing_criteria dict mapping each specialist to keywords describing its domain
    #   3. Create a router Delegate with budget_usd=0.3
    #   4. For each of four queries (null rate, AUC-PR vs AUC-ROC, latency, MAS FEAT):
    #      a. Build routing_prompt asking router to respond with ONLY the specialist name
    #      b. Await run_delegate(router, routing_prompt) → route_decision
    #      c. Match route_decision to a specialist key (default: data_quality)
    #      d. Await run_delegate(specialist, domain-specific prompt)
    #      e. Print query, routed_to, and response[:150]
    #   5. Return specialists dict
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
    ____
    ____
    ____
    ____


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
