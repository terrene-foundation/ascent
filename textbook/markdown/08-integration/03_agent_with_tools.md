# Chapter 3: Agent with ML Tools

## Overview

AI agents become powerful when they can use ML engines as tools. Instead of hardcoding an analysis pipeline, you give a **Kaizen agent** access to DataExplorer, ModelVisualizer, and TrainingPipeline as callable tools. The agent decides **when** to call each tool based on its reasoning, not your code logic. This chapter teaches the tool wrapping pattern, the ReAct agent loop, the double opt-in safety protocol, and budget-governed agent deployment.

## Prerequisites

- [Chapter 2: ML to Nexus Deployment](02_ml_to_nexus.md)
- Kailash Kaizen installed (`pip install kailash-kaizen`)
- Understanding of AI agent concepts (ReAct, tool use)

## Concepts

### Concept 1: Signatures Define Agent Behavior

A `Signature` declares what the agent receives (InputFields) and what it must produce (OutputFields). The docstring becomes the system prompt, `__intent__` describes the agent's purpose, and `__guidelines__` provide behavioral constraints. The LLM reasons about these fields -- it is never hardcoded.

- **What**: A declarative specification of agent inputs, outputs, intent, and guidelines
- **Why**: Signatures separate the agent's behavioral contract from its implementation, making agents configurable without code changes
- **How**: Subclass `Signature` with typed fields decorated with `InputField` and `OutputField`
- **When**: Every agent starts with a signature definition

### Concept 2: Tool Wrapping Pattern

ML engines become agent tools by wrapping them in async functions. The function docstring tells the agent what the tool does. The function body calls the engine. The agent decides when to call each tool based on its reasoning.

- **What**: Async functions that wrap ML engine calls for agent consumption
- **Why**: Tools are dumb data endpoints -- they fetch, compute, or store. The LLM does all the reasoning about when and why to call them
- **How**: Write an async function with a clear docstring, call the engine inside, return raw results
- **When**: For every ML engine you want the agent to be able to use

### Concept 3: AgentInfusionProtocol -- Double Opt-In

When ML agents interact with ML engines, both sides must consent. The engine declares "I accept agent guidance" (`MLToolProtocol`). The agent declares "I respect engine constraints" (`AgentInfusionProtocol`). This prevents agents from bypassing engine safety checks.

- **What**: A bidirectional consent protocol between agents and ML engines
- **Why**: Without double opt-in, agents could call engines in unsafe ways (e.g., skipping validation)
- **How**: Engines implement `MLToolProtocol`, agents implement `AgentInfusionProtocol`
- **When**: Automatically enforced when agents call ML engines through the Kaizen tool protocol

### Concept 4: Budget-Governed Agents

In production, agents must have a cost budget. The `Delegate` class accepts `budget_usd` to cap LLM API costs. Without a budget, a runaway agent could consume unlimited API credits.

### Key API

| Class / Method          | Parameters                                      | Returns    | Description                          |
| ----------------------- | ----------------------------------------------- | ---------- | ------------------------------------ |
| `Signature`             | typed fields with `InputField` / `OutputField`  | --         | Agent behavioral contract            |
| `InputField()`          | `description: str`                              | field      | Declare an input to the agent        |
| `OutputField()`         | `description: str`                              | field      | Declare an expected output           |
| `Delegate()`            | `model`, `tools`, `budget_usd`, `system_prompt` | `Delegate` | Budget-governed autonomous agent     |
| `AgentInfusionProtocol` | --                                              | protocol   | Agent-side consent for engine access |
| `MLToolProtocol`        | --                                              | protocol   | Engine-side consent for agent access |

## Code Walkthrough

```python
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

from kaizen import Signature, InputField, OutputField
```

### Defining an Agent Signature

```python
class DataAnalystSignature(Signature):
    """You are a data analyst with access to ML tools.
    Use the available tools to explore data and build insights."""

    __intent__ = "Analyze datasets and provide actionable insights"
    __guidelines__ = [
        "Always profile data before analysis",
        "Use DataExplorer for initial profiling",
        "Use ModelVisualizer for charts",
        "Stay within budget",
    ]

    question: str = InputField(description="Analysis question")
    dataset_name: str = InputField(description="Name of the dataset to analyze")
    analysis: str = OutputField(description="Detailed analysis with findings")
    recommendations: str = OutputField(description="Actionable recommendations")


assert "question" in DataAnalystSignature._signature_inputs
assert "analysis" in DataAnalystSignature._signature_outputs
```

The signature tells the LLM exactly what role it plays, what it receives, and what it must produce. The `__guidelines__` list constrains behavior -- the agent will profile data before analysis because the guideline says so, not because code forces it.

### Tool Wrapping Pattern

```python
# ML engines become agent tools by wrapping them in async functions.
# The agent decides WHEN to call each tool (LLM reasoning, not code logic).
#
# Pattern:
#   async def profile_data(dataset: str) -> dict:
#       """Profile a dataset using DataExplorer."""
#       df = loader.load(dataset)
#       explorer = DataExplorer()
#       profile = await explorer.profile(df)
#       return profile.to_dict()
#
#   async def create_chart(data: str, chart_type: str) -> str:
#       """Create a visualization using ModelVisualizer."""
#       viz = ModelVisualizer()
#       fig = viz.create(json.loads(data), chart_type=chart_type)
#       return fig.to_html()
```

Each tool is a simple async function with a docstring. The docstring is critical -- the LLM reads it to understand what the tool does. The function body calls the ML engine and returns raw results. No decision logic in the tool.

### ReActAgent with ML Tools

```python
# ReActAgent uses Reason -> Act -> Observe loops:
#   1. Reason about what to do next
#   2. Call a tool (Act)
#   3. Observe the result
#   4. Decide if done or need more info
#
# from kaizen_agents.agents.specialized.react import ReActAgent
#
# agent = ReActAgent(
#     config=agent_config,
#     tools=[profile_data, create_chart, run_query],
# )
# result = await agent.solve("What are the key trends in HDB prices?")
```

The ReAct loop is the core agent execution pattern. The agent reasons about each step, calls tools as needed, observes results, and decides when it has enough information to answer. All reasoning happens in the LLM, not in code.

### Double Opt-In Safety

```python
from kailash_ml.types import AgentInfusionProtocol, MLToolProtocol

assert AgentInfusionProtocol is not None
assert MLToolProtocol is not None
```

Both protocols exist in the type system. When an agent calls an ML engine through the Kaizen tool protocol, both sides verify consent before the call proceeds.

### Production Pattern: Budget-Governed Analyst

```python
# from kaizen_agents import Delegate
#
# delegate = Delegate(
#     model=os.environ["DEFAULT_LLM_MODEL"],
#     tools=[profile_data, create_chart, run_query],
#     budget_usd=2.0,  # MANDATORY: cap LLM costs
#     system_prompt="You are a data analyst. Use tools to answer questions.",
# )
#
# async for event in delegate.run("Analyze the credit scoring dataset"):
#     handle_event(event)

model = os.environ.get("DEFAULT_LLM_MODEL", "claude-sonnet-4-20250514")
assert len(model) > 0
```

In production, always use `Delegate` with `budget_usd` set. The Delegate handles the ReAct loop, tool dispatch, and budget tracking automatically. Without a budget, a complex analysis could run up costs indefinitely.

## Common Mistakes

| Mistake                        | Correct Pattern                        | Why                                                        |
| ------------------------------ | -------------------------------------- | ---------------------------------------------------------- |
| Decision logic in tools        | Tools fetch/compute only; LLM decides  | Logic in tools is invisible to the LLM's reasoning trace   |
| Missing `budget_usd`           | Always set `budget_usd` on Delegate    | Runaway agents can consume unlimited API credits           |
| Hardcoded tool selection       | Let the LLM choose which tools to call | Hardcoded selection defeats the purpose of agent reasoning |
| No docstring on tool functions | Every tool needs a clear docstring     | The LLM reads docstrings to understand tool capabilities   |

## Exercises

1. Define a Signature for a "Model Diagnostician" agent that takes a model name and a dataset, and outputs a diagnostic report and improvement recommendations. What `__guidelines__` would you include?
2. Write three tool wrapper functions: one for DataExplorer profiling, one for TrainingPipeline training, and one for ModelVisualizer charting. Each should be an async function with a clear docstring.
3. Why is `budget_usd` mandatory in production? Calculate the worst-case cost of a ReAct agent with 20 reasoning steps using a model that costs $0.015 per 1K output tokens, assuming 500 tokens per step.

## Key Takeaways

- Agent signatures declare inputs, outputs, intent, and guidelines -- the LLM does the reasoning
- ML engines become agent tools via async wrapper functions with clear docstrings
- Tools are dumb data endpoints: they fetch, compute, and return. They never decide.
- AgentInfusionProtocol enforces double opt-in safety between agents and ML engines
- Always set `budget_usd` on production agents to cap LLM API costs
- ReAct (Reason, Act, Observe) is the core agent execution loop

## Next Chapter

[Chapter 4: PACT-Governed Pipeline](04_governed_pipeline.md) -- Combine PACT governance with Kaizen agents and ML engines for production-safe AI systems.
