# Chapter 10: Multi-Agent Orchestration Patterns

## Overview

Kailash Kaizen provides five formal multi-agent coordination patterns, each implementing a distinct collaboration model. All patterns inherit from `BaseMultiAgentPattern` and provide a uniform `execute()` interface. This chapter teaches you how to choose between supervisor-worker delegation, democratic consensus, adversarial debate, sequential handoff, and staged pipeline -- and when each pattern is the right tool.

## Prerequisites

- Python 3.10+ installed
- Kailash Kaizen installed (`pip install kailash-kaizen`)
- Completion of Chapters 1-9 (all individual agent types, GovernedSupervisor, Pipeline, ToT)
- Understanding of multi-agent composition from Chapter 8 (Pipeline)

## Concepts

### Concept 1: BaseMultiAgentPattern

All orchestration patterns inherit from `BaseMultiAgentPattern`, which provides agent registration, execution lifecycle management, and result aggregation. This base class ensures that every pattern -- regardless of its coordination model -- has the same interface: register agents, call `execute()`, get results.

- **What**: An abstract base class that defines the contract for all multi-agent patterns
- **Why**: A uniform interface makes patterns interchangeable -- you can swap supervisor-worker for consensus without changing the calling code
- **How**: Subclass `BaseMultiAgentPattern`, implement the coordination logic, call `execute(task)` to run
- **When**: When building a new coordination pattern that is not covered by the five built-in ones

### Concept 2: Supervisor-Worker Pattern

A supervisor agent delegates tasks to specialized worker agents, then aggregates their results. The supervisor decides what to delegate (using LLM reasoning, not code conditionals), and workers execute independently. A coordinator handles task distribution.

- **What**: One supervisor + N workers, where the supervisor plans and workers execute
- **Why**: Complex analysis benefits from specialist decomposition -- one agent cannot be expert at everything
- **How**: `create_supervisor_worker_pattern(supervisor, workers)` creates the pattern; `execute(task)` runs it
- **When**: When a task naturally decomposes into specialist sub-tasks (e.g., data analysis split across data analyst, model builder, and evaluator)

### Concept 3: Consensus Pattern

Multiple agents propose solutions, vote on each other's proposals, and an aggregator synthesizes the final decision. This is democratic decision-making for high-stakes scenarios where no single agent's judgment should be definitive.

- **What**: Proposers generate solutions, voters rank them, an aggregator produces the final verdict
- **Why**: High-stakes decisions (model selection, risk assessment) benefit from diverse perspectives and democratic validation
- **How**: `create_consensus_pattern(proposers, voters, aggregator)` sets up the pattern
- **When**: When the cost of a wrong decision is high and multiple independent perspectives reduce risk

### Concept 4: Debate Pattern

An adversarial pattern with three fixed roles: a proponent argues FOR a position, an opponent argues AGAINST, and a judge evaluates both arguments to decide. Multiple rounds of debate are supported. This forces robust reasoning through challenge.

- **What**: Proponent + Opponent + Judge in structured argumentation rounds
- **Why**: Adversarial challenge reveals weaknesses that collaborative agents miss -- if an argument survives opposition, it is more robust
- **How**: `create_debate_pattern(proponent, opponent, judge, rounds=3)` creates the debate
- **When**: Risk assessment, policy review, production readiness evaluation -- any decision where devil's advocacy adds value

### Concept 5: Handoff and Sequential Pipeline Patterns

Two sequential patterns for ordered execution. **Handoff** passes work between specialists in a relay-race fashion, where each agent handles one phase. **Sequential Pipeline** is similar but uses stage-level processing signatures with explicit input/output transformation at each stage.

- **What**: Ordered agent chains where work flows from one specialist to the next
- **Why**: Many real-world workflows are inherently sequential -- ML pipelines (data prep, feature engineering, training, evaluation), document processing, approval chains
- **How**: `create_handoff_pattern(agents)` or `create_sequential_pipeline(stages)` creates the chain
- **When**: When each step requires a different specialization and order matters

### Key API

| Class / Method                       | Parameters                                 | Returns                     | Description                        |
| ------------------------------------ | ------------------------------------------ | --------------------------- | ---------------------------------- |
| `BaseMultiAgentPattern`              | --                                         | (abstract)                  | Base class for all patterns        |
| `SupervisorWorkerPattern`            | --                                         | `SupervisorWorkerPattern`   | Delegation-based coordination      |
| `create_supervisor_worker_pattern()` | `supervisor`, `workers: list`              | `SupervisorWorkerPattern`   | Factory for supervisor-worker      |
| `ConsensusPattern`                   | --                                         | `ConsensusPattern`          | Democratic voting coordination     |
| `create_consensus_pattern()`         | `proposers`, `voters`, `aggregator`        | `ConsensusPattern`          | Factory for consensus              |
| `DebatePattern`                      | --                                         | `DebatePattern`             | Adversarial reasoning coordination |
| `create_debate_pattern()`            | `proponent`, `opponent`, `judge`, `rounds` | `DebatePattern`             | Factory for debate                 |
| `HandoffPattern`                     | --                                         | `HandoffPattern`            | Sequential specialist relay        |
| `create_handoff_pattern()`           | `agents: list`                             | `HandoffPattern`            | Factory for handoff                |
| `SequentialPipelinePattern`          | --                                         | `SequentialPipelinePattern` | Stage-based sequential processing  |
| `create_sequential_pipeline()`       | `stages: list`                             | `SequentialPipelinePattern` | Factory for sequential pipeline    |

### Agent Roles

Each pattern uses specialized agent classes:

| Pattern             | Agent Classes                                        |
| ------------------- | ---------------------------------------------------- |
| Supervisor-Worker   | `SupervisorAgent`, `WorkerAgent`, `CoordinatorAgent` |
| Consensus           | `ProposerAgent`, `VoterAgent`, `AggregatorAgent`     |
| Debate              | `ProponentAgent`, `OpponentAgent`, `JudgeAgent`      |
| Handoff             | `HandoffAgent`                                       |
| Sequential Pipeline | `PipelineStageAgent`                                 |

## Code Walkthrough

### Imports and Setup

```python
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

from kaizen_agents.patterns.patterns import (
    # Supervisor-Worker
    SupervisorWorkerPattern,
    SupervisorAgent,
    WorkerAgent,
    CoordinatorAgent,
    create_supervisor_worker_pattern,
    TaskDelegationSignature,
    # Consensus
    ConsensusPattern,
    ProposerAgent,
    VoterAgent,
    AggregatorAgent,
    create_consensus_pattern,
    # Debate
    DebatePattern,
    ProponentAgent,
    OpponentAgent,
    JudgeAgent,
    create_debate_pattern,
    # Handoff
    HandoffPattern,
    HandoffAgent,
    create_handoff_pattern,
    # Sequential Pipeline
    SequentialPipelinePattern,
    PipelineStageAgent,
    create_sequential_pipeline,
    # Base
    BaseMultiAgentPattern,
)

model = os.environ.get("DEFAULT_LLM_MODEL", "claude-sonnet-4-20250514")
```

All patterns and their agent classes are imported from `kaizen_agents.patterns.patterns`. Each pattern has a class (e.g., `SupervisorWorkerPattern`), role-specific agents (e.g., `SupervisorAgent`, `WorkerAgent`), and a factory function (e.g., `create_supervisor_worker_pattern`).

### Pattern Hierarchy

```python
assert issubclass(SupervisorWorkerPattern, BaseMultiAgentPattern)
assert issubclass(ConsensusPattern, BaseMultiAgentPattern)
assert issubclass(DebatePattern, BaseMultiAgentPattern)
assert issubclass(HandoffPattern, BaseMultiAgentPattern)
assert issubclass(SequentialPipelinePattern, BaseMultiAgentPattern)
```

All five patterns inherit from `BaseMultiAgentPattern`. This guarantees a uniform `execute()` interface across all coordination models.

### Supervisor-Worker Pattern

```python
# A supervisor delegates tasks to workers and aggregates results.
# The supervisor decides WHAT to delegate (LLM reasoning, not code).
#
# Use case: Complex analysis split across specialist agents.
#
#   supervisor = SupervisorAgent(model=model, ...)
#   workers = [
#       WorkerAgent(model=model, role="data_analyst", ...),
#       WorkerAgent(model=model, role="model_builder", ...),
#   ]
#   pattern = create_supervisor_worker_pattern(supervisor, workers)
#   result = await pattern.execute("Analyze and model the dataset")

assert SupervisorAgent is not None
assert WorkerAgent is not None
assert CoordinatorAgent is not None
assert TaskDelegationSignature is not None
```

The supervisor uses LLM reasoning to decompose the task and assign sub-tasks to workers. The `CoordinatorAgent` handles distribution logistics. `TaskDelegationSignature` defines the structured delegation I/O.

### Consensus Pattern

```python
# Multiple agents propose solutions, vote, then aggregate.
# Democratic decision-making for high-stakes decisions.
#
# Use case: Model selection -- 3 agents each recommend a model,
# then vote on which recommendation to adopt.
#
#   proposers = [ProposerAgent(...) for _ in range(3)]
#   voters = [VoterAgent(...) for _ in range(3)]
#   aggregator = AggregatorAgent(...)
#   pattern = create_consensus_pattern(proposers, voters, aggregator)
#   result = await pattern.execute("Select the best model for credit scoring")

assert ProposerAgent is not None
assert VoterAgent is not None
assert AggregatorAgent is not None
```

Consensus requires 2N+1 agents: N proposers, N voters, and 1 aggregator. Proposers generate diverse solutions, voters rank them, and the aggregator synthesizes the final decision.

### Debate Pattern

```python
# Adversarial: proponent argues FOR, opponent argues AGAINST,
# judge decides. Forces robust reasoning through challenge.
#
# Use case: Risk assessment -- proponent argues the model is safe,
# opponent challenges with failure scenarios, judge decides.
#
#   proponent = ProponentAgent(model=model, position="Model is production-ready")
#   opponent = OpponentAgent(model=model, position="Model has risks")
#   judge = JudgeAgent(model=model)
#   pattern = create_debate_pattern(proponent, opponent, judge, rounds=3)
#   result = await pattern.execute("Should we deploy the credit scoring model?")

assert ProponentAgent is not None
assert OpponentAgent is not None
assert JudgeAgent is not None
```

Debate is a fixed three-agent pattern: proponent, opponent, judge. The `rounds` parameter controls how many back-and-forth exchanges occur before the judge renders a verdict.

### Handoff Pattern

```python
# Sequential specialists -- each agent handles one phase, then
# hands off to the next. Like a relay race.
#
# Use case: ML pipeline stages -- data prep -> feature eng -> training -> eval
#
#   agents = [
#       HandoffAgent(model=model, role="data_prep"),
#       HandoffAgent(model=model, role="feature_eng"),
#       HandoffAgent(model=model, role="trainer"),
#       HandoffAgent(model=model, role="evaluator"),
#   ]
#   pattern = create_handoff_pattern(agents)
#   result = await pattern.execute("Build an end-to-end model")

assert HandoffAgent is not None
```

Handoff is the simplest sequential pattern -- each agent receives the previous agent's output and passes its own output to the next.

### Sequential Pipeline Pattern

```python
# Similar to handoff but with stage-level processing signatures.
# Each stage processes and transforms the output for the next stage.
#
#   stages = [
#       PipelineStageAgent(model=model, stage_name="research"),
#       PipelineStageAgent(model=model, stage_name="analyze"),
#       PipelineStageAgent(model=model, stage_name="recommend"),
#   ]
#   pipeline = create_sequential_pipeline(stages)
#   result = await pipeline.execute("Evaluate market opportunity")

assert PipelineStageAgent is not None
```

Sequential Pipeline adds stage-level processing signatures to the handoff model. Each `PipelineStageAgent` has a `stage_name` and explicit input/output transformation.

### Choosing the Right Pattern

| Pattern    | Best For                       | Agent Count | Key Characteristic         |
| ---------- | ------------------------------ | ----------- | -------------------------- |
| Supervisor | Complex delegated tasks        | 1 + N       | Top-down decomposition     |
| Consensus  | High-stakes decisions          | 2N + 1      | Democratic validation      |
| Debate     | Risk assessment, policy review | 3 (fixed)   | Adversarial challenge      |
| Handoff    | Sequential specialist pipeline | N           | Relay-race execution       |
| Pipeline   | Data transformation chains     | N           | Stage-level transformation |

All patterns use LLM reasoning for decisions (not code conditionals). The supervisor does not use `if-else` to delegate; it reasons about the task and decides.

## Common Mistakes

| Mistake                                   | Correct Pattern                                        | Why                                                                                            |
| ----------------------------------------- | ------------------------------------------------------ | ---------------------------------------------------------------------------------------------- |
| Using debate for collaborative tasks      | Use consensus or ensemble instead                      | Debate is adversarial by design -- it challenges rather than collaborates                      |
| Using consensus for simple decisions      | Use a single agent or supervisor-worker                | Consensus overhead (2N+1 agents) is wasted on decisions that do not need multiple perspectives |
| Hardcoding delegation logic in supervisor | Let the supervisor LLM reason about task decomposition | Code-based delegation violates the LLM-First rule and breaks on novel tasks                    |
| Using handoff when stages are independent | Use `Pipeline.parallel()` for concurrent execution     | Sequential execution of independent tasks wastes time                                          |

## Exercises

1. For each of the five patterns, describe a real-world use case in ML engineering where that pattern would be the best choice. Explain why the other four patterns would be worse fits for each case.
2. Draw the agent communication diagram for a consensus pattern with 3 proposers, 3 voters, and 1 aggregator. How many total LLM calls are needed if each agent makes one call?
3. Design a debate pattern for evaluating whether a trained model should be deployed to production. Define the proponent's position, the opponent's position, and what criteria the judge should use. How many rounds would you configure, and why?

## Key Takeaways

- Five formal patterns cover the full spectrum of multi-agent coordination: supervisor-worker, consensus, debate, handoff, and sequential pipeline
- All patterns inherit from `BaseMultiAgentPattern` with a uniform `execute()` interface
- Factory functions (`create_*_pattern()`) are the recommended entry point for each pattern
- Pattern choice depends on the task: delegation for complex decomposition, consensus for high-stakes decisions, debate for adversarial validation, handoff/pipeline for sequential processing
- All patterns use LLM reasoning for decisions -- deterministic routing is not permitted in agent decision paths
- Agent roles (SupervisorAgent, ProposerAgent, JudgeAgent, etc.) encode the responsibilities specific to each pattern
