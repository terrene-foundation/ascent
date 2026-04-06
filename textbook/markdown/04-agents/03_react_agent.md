# Chapter 3: ReAct Agent

## Overview

The **ReActAgent** implements the Reason-Act-Observe pattern for autonomous multi-cycle execution. Unlike the single-shot SimpleQA, ReAct agents iteratively reason about a task, take actions (tool calls), observe results, and decide whether to continue or stop. This chapter teaches you the ReAct loop, convergence detection, and the `MultiCycleStrategy` that powers autonomous agents.

## Prerequisites

- Python 3.10+ installed
- Kailash Kaizen installed (`pip install kailash-kaizen`)
- Completion of Chapters 1-2 (Delegate, SimpleQA)
- Understanding of `BaseAgent` and `AsyncSingleShotStrategy` (Chapter 2)

## Concepts

### Concept 1: The ReAct Pattern

ReAct agents operate in iterative cycles: Reason (think about the current state), Act (execute a tool call, ask for clarification, or finish), Observe (examine the result), and Repeat until converged. This is fundamentally different from single-shot agents -- ReAct is autonomous and decides its own execution path.

- **What**: An iterative reasoning loop where the agent alternates between thinking and acting
- **Why**: Many tasks require multiple steps with intermediate observations -- searching, reading files, refining analysis
- **How**: The `MultiCycleStrategy` drives the loop, calling the LLM repeatedly until convergence criteria are met
- **When**: Use ReAct when tasks require real-time data, multi-step workflows, or open-ended exploration

### Concept 2: ActionType Vocabulary

The ReAct agent uses a fixed vocabulary of action types: `TOOL_USE` (invoke a tool), `FINISH` (task complete), and `CLARIFY` (ask the user for more information). This structured vocabulary replaces free-form action strings with type-safe enum values.

- **What**: An enum of permitted actions the agent can take at each cycle
- **Why**: A fixed vocabulary prevents the model from inventing arbitrary actions and makes convergence detection reliable
- **How**: The LLM's `action` output field must be one of the `ActionType` values
- **When**: Every ReAct cycle produces an action -- it is part of the signature contract

### Concept 3: MultiCycleStrategy

`MultiCycleStrategy` is the execution strategy for autonomous agents. It calls the LLM in a loop, checking convergence after each cycle. This is the critical difference from interactive agents: instead of one call, the strategy manages many calls with a configurable cycle limit.

- **What**: An execution strategy that loops LLM calls until convergence or cycle exhaustion
- **Why**: Autonomous reasoning requires iteration -- the agent cannot know in advance how many steps it will need
- **How**: The strategy calls `_check_convergence()` after each cycle; if not converged and cycles remain, it continues
- **When**: Used automatically by ReAct, RAG, and other autonomous agent types

### Concept 4: Convergence Detection

Convergence determines when the agent should stop iterating. Kaizen uses a two-tier system: **objective convergence** (ADR-013) checks the `tool_calls` field -- if empty, the agent has no more actions to take. **Subjective fallback** checks `action` and `confidence` when `tool_calls` is absent.

- **What**: A decision function that returns True when the agent should stop and False when it should continue
- **Why**: Without convergence detection, autonomous agents loop forever (or until budget exhaustion)
- **How**: Non-empty `tool_calls` means continue; empty `tool_calls` means stop; fallback checks action/confidence
- **When**: Called automatically by `MultiCycleStrategy` after every cycle

### Key API

| Class / Method               | Parameters                                                    | Returns       | Description                                        |
| ---------------------------- | ------------------------------------------------------------- | ------------- | -------------------------------------------------- |
| `ReActAgent()`               | `llm_provider`, `model`, `max_cycles`, `confidence_threshold` | `ReActAgent`  | Create a ReAct agent                               |
| `agent.run()`                | `task: str`, `context: str = ""`                              | `dict`        | Execute the ReAct loop                             |
| `agent._check_convergence()` | `result: dict`                                                | `bool`        | Test if the agent should stop                      |
| `ReActConfig()`              | `max_cycles`, `confidence_threshold`, `enable_parallel_tools` | `ReActConfig` | Configuration for ReAct behavior                   |
| `ReActSignature`             | --                                                            | `Signature`   | Multi-field I/O for the ReAct cycle                |
| `ActionType`                 | --                                                            | `Enum`        | `TOOL_USE`, `FINISH`, `CLARIFY`                    |
| `MultiCycleStrategy`         | --                                                            | `Strategy`    | Iterative execution strategy for autonomous agents |

## Code Walkthrough

### Imports and Setup

```python
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

from kaizen_agents.agents.specialized.react import (
    ReActAgent,
    ReActConfig,
    ReActSignature,
    ActionType,
)
from kaizen.core.base_agent import BaseAgent
from kaizen.strategies.multi_cycle import MultiCycleStrategy

model = os.environ.get("DEFAULT_LLM_MODEL", "claude-sonnet-4-20250514")
```

The ReAct module provides four key exports: the agent class, its config, the signature, and the action type enum. `MultiCycleStrategy` is imported from the core strategies module to verify the agent uses it.

### ActionType -- Structured Action Vocabulary

```python
assert ActionType.TOOL_USE.value == "tool_use"
assert ActionType.FINISH.value == "finish"
assert ActionType.CLARIFY.value == "clarify"
```

Three actions cover all agent decisions: use a tool, finish the task, or ask for clarification. The model selects one of these at each cycle.

### ReActSignature -- Multi-Field I/O

```python
assert "task" in ReActSignature._signature_inputs
assert "context" in ReActSignature._signature_inputs
assert "available_tools" in ReActSignature._signature_inputs
assert "previous_actions" in ReActSignature._signature_inputs

assert "thought" in ReActSignature._signature_outputs
assert "action" in ReActSignature._signature_outputs
assert "action_input" in ReActSignature._signature_outputs
assert "confidence" in ReActSignature._signature_outputs
assert "need_tool" in ReActSignature._signature_outputs
assert (
    "tool_calls" in ReActSignature._signature_outputs
), "ADR-013: tool_calls enables objective convergence detection"
```

The ReAct signature is richer than QASignature. Inputs include the task, context, available tools, and the history of previous actions. Outputs include the agent's thought process, chosen action, confidence level, and a `tool_calls` field that enables objective convergence detection (ADR-013).

### ReActConfig

```python
config = ReActConfig()

assert config.max_cycles == 10, "Default: up to 10 reasoning cycles"
assert config.confidence_threshold == 0.7, "Converge at 0.7 confidence"
assert config.mcp_discovery_enabled is True, "Autonomous agents discover tools"
assert config.enable_parallel_tools is False, "Sequential tool execution by default"

# Custom configuration
custom_config = ReActConfig(
    max_cycles=15,
    confidence_threshold=0.8,
    enable_parallel_tools=True,
)
assert custom_config.max_cycles == 15
```

ReActConfig adds autonomous-agent-specific settings: `max_cycles` caps how many iterations the agent can run, `confidence_threshold` controls when subjective convergence triggers, and `mcp_discovery_enabled` allows the agent to discover tools via MCP. Parallel tool execution is opt-in.

### ReActAgent Instantiation

```python
agent = ReActAgent(
    llm_provider="mock",
    model=model,
    max_cycles=10,
    confidence_threshold=0.7,
)

assert isinstance(agent, ReActAgent)
assert isinstance(agent, BaseAgent)
assert agent.react_config.max_cycles == 10

# Verify MultiCycleStrategy is used
assert isinstance(
    agent.strategy, MultiCycleStrategy
), "ReAct MUST use MultiCycleStrategy for iterative execution"
```

The critical difference from SimpleQA: ReActAgent uses `MultiCycleStrategy`, not `AsyncSingleShotStrategy`. This strategy drives the iterative Reason-Act-Observe loop. The assertion confirms the correct strategy is in place.

### Objective Convergence Detection (ADR-013)

```python
# Simulate: tool calls present -> continue
result_with_tools = {
    "tool_calls": [{"name": "search", "params": {"query": "flights"}}],
    "action": "tool_use",
    "confidence": 0.5,
}
assert (
    agent._check_convergence(result_with_tools) is False
), "Non-empty tool_calls means NOT converged"

# Simulate: empty tool calls -> converged
result_empty_tools = {
    "tool_calls": [],
    "action": "finish",
    "confidence": 0.9,
}
assert (
    agent._check_convergence(result_empty_tools) is True
), "Empty tool_calls means CONVERGED"
```

Objective convergence follows the `while(tool_call_exists)` pattern: if `tool_calls` is present and non-empty, the agent has more work to do. If `tool_calls` is present but empty, the agent is done. This is the primary convergence signal.

### Subjective Convergence Fallback

```python
# action == "finish" -> converged
result_finish = {"action": "finish", "confidence": 0.6}
assert agent._check_convergence(result_finish) is True

# confidence >= threshold -> converged
result_high_conf = {"action": "tool_use", "confidence": 0.9}
assert agent._check_convergence(result_high_conf) is True

# action == "tool_use" with low confidence -> NOT converged
result_continue = {"action": "tool_use", "confidence": 0.3}
assert agent._check_convergence(result_continue) is False

# No signals -> default safe fallback (converged)
assert agent._check_convergence({}) is True
```

When `tool_calls` is absent (e.g., an older model that does not populate it), the agent falls back to subjective checks: `action == "finish"` means done, `confidence >= threshold` means done, and an empty result defaults to converged (safe fallback to prevent infinite loops).

### Input Validation

```python
empty_result = agent.run(task="")
assert empty_result["error"] == "INVALID_INPUT"
assert empty_result["action"] == ActionType.FINISH.value
assert empty_result["confidence"] == 0.0
assert empty_result["cycles_used"] == 0

whitespace_result = agent.run(task="   ")
assert whitespace_result["error"] == "INVALID_INPUT"
```

Like SimpleQA, empty or whitespace tasks are caught before any LLM call. The error response includes `cycles_used: 0` to confirm no execution occurred.

### Return Structure

```python
# ReActAgent.run() returns:
# {
#     "thought": "I need to search for flights...",
#     "action": "tool_use",
#     "action_input": {"tool": "search", "query": "flights to Paris"},
#     "confidence": 0.85,
#     "need_tool": True,
#     "cycles_used": 3,
#     "total_cycles": 10,
# }
```

The return dict includes the final thought, action, confidence, and cycle usage statistics (`cycles_used` vs `total_cycles`).

## Common Mistakes

| Mistake                                 | Correct Pattern                      | Why                                                                                       |
| --------------------------------------- | ------------------------------------ | ----------------------------------------------------------------------------------------- |
| Setting `max_cycles` too low            | Start with `max_cycles=10` (default) | Complex tasks may need many cycles -- too few causes premature termination                |
| Setting `confidence_threshold` too high | Start with `0.7` (default)           | Very high thresholds cause the agent to loop excessively seeking certainty                |
| Using ReAct for simple QA               | Use `SimpleQAAgent` instead          | ReAct's multi-cycle overhead is wasted on single-answer questions                         |
| Ignoring `cycles_used` in the result    | Log and monitor cycle counts         | High cycle counts relative to `total_cycles` indicate the agent is struggling to converge |

## Exercises

1. Create a `ReActAgent` with `max_cycles=5` and `confidence_threshold=0.9`. Construct three result dictionaries and call `_check_convergence()` on each: one that should continue, one that converges by empty `tool_calls`, and one that converges by high confidence.
2. Compare `ReActSignature._signature_outputs` with `QASignature._signature_outputs` from Chapter 2. List the fields that are unique to ReAct and explain why each is needed for iterative reasoning.
3. Create a `ReActConfig` with `enable_parallel_tools=True` and `max_cycles=20`. What is the total maximum number of tool calls this agent could make in a single run, assuming each cycle makes one tool call?

## Key Takeaways

- ReAct agents iterate through Reason-Act-Observe cycles, unlike single-shot agents
- `MultiCycleStrategy` drives the iterative loop with configurable cycle limits
- `ActionType` provides a fixed vocabulary: `TOOL_USE`, `FINISH`, `CLARIFY`
- Objective convergence (ADR-013) uses `tool_calls` -- empty means done, non-empty means continue
- Subjective fallback uses `action` and `confidence` when `tool_calls` is unavailable
- Input validation catches empty tasks before any LLM call, preserving budget

## Next Chapter

[Chapter 4: Chain-of-Thought Agent](04_chain_of_thought.md) -- Use structured step-by-step reasoning for auditable problem-solving without iterative tool use.
