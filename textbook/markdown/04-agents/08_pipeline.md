# Chapter 8: Pipeline Composition

## Overview

The **Pipeline** abstraction composes multiple agents into coordinated multi-agent workflows. Instead of building monolithic agents, you wire smaller specialized agents together using composition patterns: sequential chains, intelligent routers, parallel execution, ensemble synthesis, and more. This chapter teaches you the Pipeline base class, the `SequentialPipeline`, the `to_agent()` bridge, and the nine factory methods that create specific coordination patterns.

## Prerequisites

- Python 3.10+ installed
- Kailash Kaizen installed (`pip install kailash-kaizen`)
- Completion of Chapters 1-7 (all individual agent types and GovernedSupervisor)
- Understanding of `BaseAgent` and the agent lifecycle

## Concepts

### Concept 1: Pipeline as Composable Abstraction

`Pipeline` is the abstract base class for all multi-agent coordination patterns. It provides two methods: `.run(**inputs)` to execute the pipeline, and `.to_agent()` to convert the pipeline into a `BaseAgent` for nested composition. This means pipelines can contain other pipelines.

- **What**: An abstract base class with `run()` for execution and `to_agent()` for composition
- **Why**: A single abstraction for all coordination patterns enables uniform composition -- any pipeline can be used anywhere an agent is expected
- **How**: Subclass `Pipeline`, implement `run()`, use `to_agent()` to nest inside larger orchestrations
- **When**: Whenever a task requires multiple agents working together in a defined coordination pattern

### Concept 2: SequentialPipeline

`SequentialPipeline` executes agents in order, passing each agent's output as input to the next. It is the simplest multi-agent pattern: `Input -> [Agent 1] -> [Agent 2] -> [Agent 3] -> Output`. Created via `Pipeline.sequential(agents=[...])`.

- **What**: A linear chain where each agent processes the previous agent's output
- **Why**: Many workflows are naturally sequential -- ETL (extract, transform, load), analysis pipelines, document processing
- **How**: `Pipeline.sequential(agents=[a, b, c])` creates a chain; `run(data="input")` executes it
- **When**: When each step depends on the previous step's output and order matters

### Concept 3: Pipeline-to-Agent Bridge

`pipeline.to_agent(name, description)` converts any pipeline into a `BaseAgent`. This enables recursive composition: a sequential pipeline can be wrapped as an agent and used inside a parallel pipeline, which itself can be wrapped and used in a supervisor pattern.

- **What**: A method that wraps a pipeline's `run()` in a `BaseAgent` interface
- **Why**: Pipelines and agents should be interchangeable -- this bridge makes all coordination patterns composable at any level
- **How**: `agent = pipeline.to_agent(name="etl", description="ETL pipeline")`; the agent delegates `run()` to the pipeline
- **When**: When you need to use a pipeline as a component in a larger orchestration

### Concept 4: Factory Methods for Coordination Patterns

`Pipeline` provides nine static factory methods, each creating a different coordination pattern. These patterns cover the full spectrum of multi-agent coordination, from simple chains to adversarial reasoning.

- **What**: Static methods like `Pipeline.sequential()`, `Pipeline.router()`, `Pipeline.parallel()`, etc.
- **Why**: Factory methods encode proven coordination patterns -- you select the pattern, not the implementation
- **How**: Call the factory with a list of agents and pattern-specific parameters
- **When**: Choose the pattern that matches your coordination need (see the pattern summary table below)

### Key API

| Class / Method                 | Parameters                                       | Returns              | Description                               |
| ------------------------------ | ------------------------------------------------ | -------------------- | ----------------------------------------- |
| `Pipeline` (abstract)          | --                                               | `Pipeline`           | Base class for all patterns               |
| `pipeline.run()`               | `**inputs`                                       | `dict`               | Execute the pipeline                      |
| `pipeline.to_agent()`          | `name: str`, `description: str`                  | `BaseAgent`          | Convert pipeline to agent for composition |
| `Pipeline.sequential()`        | `agents: list[BaseAgent]`                        | `SequentialPipeline` | Linear agent chain                        |
| `Pipeline.router()`            | `agents: list`, `routing_strategy: str`          | `Pipeline`           | Intelligent request routing               |
| `Pipeline.parallel()`          | `agents: list`, `aggregator`, `max_workers`, ... | `Pipeline`           | Concurrent execution                      |
| `Pipeline.ensemble()`          | `agents: list`, `synthesizer`, `top_k`, ...      | `Pipeline`           | Multi-perspective synthesis               |
| `Pipeline.supervisor_worker()` | `supervisor`, `workers`, `selection_mode`        | `Pipeline`           | Delegated task execution                  |
| `Pipeline.blackboard()`        | `specialists`, `controller`, `max_iterations`    | `Pipeline`           | Iterative specialist collaboration        |
| `Pipeline.consensus()`         | `agents: list`                                   | `Pipeline`           | Democratic voting                         |
| `Pipeline.debate()`            | `agents: list`                                   | `Pipeline`           | Adversarial reasoning                     |
| `Pipeline.handoff()`           | `agents: list`                                   | `Pipeline`           | Tier escalation                           |
| `SequentialPipeline`           | --                                               | `SequentialPipeline` | Concrete linear chain implementation      |

## Code Walkthrough

### Imports and Setup

```python
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

from kaizen_agents.patterns.pipeline import Pipeline, SequentialPipeline
from kaizen.core.base_agent import BaseAgent, BaseAgentConfig
from kaizen.signatures import InputField, OutputField, Signature

model = os.environ.get("DEFAULT_LLM_MODEL", "claude-sonnet-4-20250514")
```

The Pipeline module provides the base class and `SequentialPipeline`. `BaseAgent`, `BaseAgentConfig`, and `Signature` are needed to create agents that participate in pipelines.

### Custom Pipeline Subclass

```python
class DataProcessingPipeline(Pipeline):
    """Example: a three-step data processing pipeline."""

    def run(self, **inputs):
        data = inputs.get("data", "")
        # Step 1: Clean
        cleaned = data.strip().lower()
        # Step 2: Transform
        transformed = cleaned.replace(" ", "_")
        # Step 3: Validate
        is_valid = len(transformed) > 0
        return {
            "original": data,
            "cleaned": cleaned,
            "transformed": transformed,
            "valid": is_valid,
        }


pipeline = DataProcessingPipeline()
result = pipeline.run(data="  Hello World  ")

assert result["original"] == "  Hello World  "
assert result["cleaned"] == "hello world"
assert result["transformed"] == "hello_world"
assert result["valid"] is True
```

Subclassing `Pipeline` and implementing `run()` is the simplest way to create a multi-step pipeline. The `run()` method receives keyword arguments and returns a dict. In this example, three processing steps are encoded directly -- in practice, these would delegate to agents.

### Pipeline-to-Agent Bridge

```python
agent = pipeline.to_agent(
    name="data_processor",
    description="Cleans and transforms data",
)

assert isinstance(agent, BaseAgent)
assert agent.agent_id == "data_processor"

# The agent delegates to the pipeline's run()
agent_result = agent.run(data="  Test Input  ")
assert agent_result["transformed"] == "test_input"
```

`to_agent()` wraps the pipeline in a `BaseAgent` shell. The resulting agent has an `agent_id` and delegates its `run()` to the pipeline. This agent can now be used in any orchestration that accepts `BaseAgent` instances.

### SequentialPipeline -- Linear Agent Chain

```python
class StepAgent(BaseAgent):
    """Minimal agent that adds a step marker to input."""

    def __init__(self, step_name: str):
        config = BaseAgentConfig(llm_provider="mock", model="mock")

        class StepSig(Signature):
            """Process a pipeline step."""

            data: str = InputField(description="Input data")
            result: str = OutputField(description="Step output")

        super().__init__(config=config, signature=StepSig())
        self.step_name = step_name

    def run(self, **inputs):
        data = inputs.get("data", inputs.get("result", "start"))
        return {
            "result": f"{data} -> {self.step_name}",
            "data": f"{data} -> {self.step_name}",
        }


agent_a = StepAgent("extract")
agent_b = StepAgent("transform")
agent_c = StepAgent("load")

# Create sequential pipeline via factory method
seq_pipeline = Pipeline.sequential(agents=[agent_a, agent_b, agent_c])

assert isinstance(seq_pipeline, SequentialPipeline)
assert len(seq_pipeline.agents) == 3

# Execute the sequential pipeline
seq_result = seq_pipeline.run(data="raw_data")

assert "final_output" in seq_result
assert "intermediate_results" in seq_result
assert len(seq_result["intermediate_results"]) == 3

# Each step is recorded
steps = seq_result["intermediate_results"]
assert steps[0]["step"] == 1
assert steps[0]["agent"] == "StepAgent"
assert steps[1]["step"] == 2
assert steps[2]["step"] == 3
```

`Pipeline.sequential()` creates a `SequentialPipeline` from a list of agents. Each agent's output flows to the next. The result includes `final_output` (the last agent's output) and `intermediate_results` (a list of step records with step number, agent name, and output).

### SequentialPipeline as Agent

```python
seq_agent = seq_pipeline.to_agent(name="etl_pipeline")
assert isinstance(seq_agent, BaseAgent)
```

Sequential pipelines can also be converted to agents for nested composition.

### Pipeline Factory Methods (Documentation Patterns)

The remaining factory methods create advanced coordination patterns:

**Pipeline.router()** -- Routes requests to the best agent based on A2A capability matching. Strategies include `"semantic"` (A2A-based, recommended), `"round-robin"`, and `"random"`.

**Pipeline.parallel()** -- Executes all agents concurrently with optional custom aggregator, max workers, and timeout.

**Pipeline.ensemble()** -- Selects top-k agents with best capability matches, executes them, then synthesizes perspectives into a unified result via a synthesizer agent.

**Pipeline.supervisor_worker()** -- Supervisor delegates tasks to workers with A2A semantic matching. Workers execute independently and results are aggregated.

**Pipeline.blackboard()** -- Maintains shared state, iteratively selects specialists based on evolving needs, and uses a controller to determine completion.

**Pipeline.consensus()** -- Democratic voting where multiple agents propose and vote.

**Pipeline.debate()** -- Adversarial reasoning with proponent, opponent, and judge.

**Pipeline.handoff()** -- Tier-based escalation where each agent handles one phase then hands off.

### Pattern Summary

| Pattern             | Use Case                        | Agent Count |
| ------------------- | ------------------------------- | ----------- |
| `sequential`        | Linear processing chain         | N           |
| `router`            | Best-agent selection            | N           |
| `parallel`          | Concurrent independent tasks    | N           |
| `ensemble`          | Multi-perspective synthesis     | N + 1       |
| `supervisor_worker` | Delegated task management       | 1 + N       |
| `blackboard`        | Iterative specialist refinement | N + 1       |
| `consensus`         | Democratic voting               | N           |
| `debate`            | Adversarial argument evaluation | 2+          |
| `handoff`           | Tier-based escalation           | N tiers     |

## Common Mistakes

| Mistake                                         | Correct Pattern                                         | Why                                                                                     |
| ----------------------------------------------- | ------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| Using `sequential` when agents are independent  | Use `Pipeline.parallel()` for concurrent tasks          | Sequential execution wastes time when agents do not depend on each other                |
| Not using `to_agent()` for nested composition   | Wrap pipelines with `to_agent()` before nesting         | Pipelines and agents have different interfaces -- the bridge makes them interchangeable |
| Hardcoding agent order in sequential pipelines  | Consider `router` for dynamic selection                 | Fixed order breaks when the task does not follow the expected pattern                   |
| Building monolithic agents instead of composing | Split into specialized agents and compose with Pipeline | Smaller agents are easier to test, reuse, and replace                                   |

## Exercises

1. Create three `StepAgent` instances (research, analyze, recommend) and compose them into a `SequentialPipeline`. Execute with `run(data="market opportunity")` and verify that `intermediate_results` has three entries with the correct step numbers.
2. Create a custom `Pipeline` subclass that takes two agents, runs both, and returns whichever result has a longer `"result"` string. Test it with two `StepAgent` instances that produce different-length outputs.
3. Create a `SequentialPipeline`, convert it to an agent with `to_agent()`, then create a second `SequentialPipeline` that includes the converted agent as one of its steps. Execute the nested pipeline and verify the output traces through all steps.

## Key Takeaways

- `Pipeline` is the composable abstraction for all multi-agent coordination patterns
- `to_agent()` converts any pipeline into a `BaseAgent` for recursive composition
- `Pipeline.sequential()` creates a linear chain where each agent's output feeds the next
- Nine factory methods cover the full spectrum: sequential, router, parallel, ensemble, supervisor-worker, blackboard, consensus, debate, handoff
- `intermediate_results` in `SequentialPipeline` provides full step-by-step traceability
- Pipelines within pipelines enable arbitrarily complex orchestrations

## Next Chapter

[Chapter 9: Tree-of-Thoughts Agent](09_tree_of_thoughts.md) -- Explore multiple reasoning paths in parallel, evaluate them independently, and select the best one for complex decision-making.
