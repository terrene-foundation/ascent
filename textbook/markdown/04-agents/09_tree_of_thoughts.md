# Chapter 9: Tree-of-Thoughts Agent

## Overview

The **ToTAgent** (Tree-of-Thoughts) explores multiple reasoning paths in parallel, evaluates each independently, and selects the best one. Unlike Chain-of-Thought (one linear path) or ReAct (one iterative path), ToT generates N diverse approaches simultaneously, scores them, and picks the winner. This chapter teaches you the four-phase ToT pattern, path evaluation criteria, best-path selection, and configuration for controlling exploration breadth.

## Prerequisites

- Python 3.10+ installed
- Kailash Kaizen installed (`pip install kailash-kaizen`)
- Completion of Chapters 3-4 (ReAct and Chain-of-Thought)
- Understanding of the difference between single-path and multi-path reasoning

## Concepts

### Concept 1: Multi-Path Exploration

Tree-of-Thoughts explores N reasoning paths in parallel rather than committing to a single approach. Each path represents a different strategy for solving the same problem. After generation, paths are evaluated and the best one is selected. This is analogous to brainstorming multiple solutions before choosing.

- **What**: Parallel generation of N diverse reasoning paths for the same problem
- **Why**: Complex problems often have multiple valid approaches -- exploring several increases the chance of finding the best one
- **How**: The agent generates `num_paths` independent paths, each with its own reasoning and steps
- **When**: Use for complex decisions, creative problem-solving, or tasks where the optimal approach is not obvious

### Concept 2: Four-Phase Execution

ToT operates in four phases: Generate (create N paths), Evaluate (score each path), Select (pick the highest score), Execute (extract the final result). Unlike ReAct's iterative loop, ToT's phases are sequential and deterministic.

- **What**: A fixed four-phase pipeline: generate, evaluate, select, execute
- **Why**: Separating generation from evaluation prevents the model from anchoring on its first idea
- **How**: Phase 1 calls the LLM N times for diverse paths; Phase 2 scores each; Phase 3 picks the max; Phase 4 extracts the result
- **When**: Always -- this is the internal execution model of every ToT run

### Concept 3: Path Evaluation Criteria

Each path is evaluated on four weighted criteria: Completeness (reasoning length > 20 chars, weight 0.3), No errors (no "error" key, weight 0.3), Structured steps (non-empty steps list, weight 0.2), and Reasoning quality (reasoning > 100 chars, weight 0.2). A perfect path scores 1.0.

- **What**: A weighted scoring function that assesses path quality on four dimensions
- **Why**: Automated scoring removes subjective bias and enables consistent best-path selection
- **How**: Each criterion contributes its weight if met; scores sum to a maximum of 1.0
- **When**: Called automatically during the Evaluate phase for every generated path

### Concept 4: Configuration Guards

`ToTAgentConfig` includes safety guards: `num_paths` cannot exceed `max_paths` (default 20). High temperature (0.9) encourages diverse path generation. The `evaluation_criteria` parameter selects the scoring strategy (default: "quality").

- **What**: Validation rules that prevent misconfiguration and ensure sensible exploration bounds
- **Why**: Unbounded path generation wastes tokens and time -- `max_paths` caps the cost
- **How**: The agent validates `num_paths <= max_paths` at construction time; violation raises `ValueError`
- **When**: Always validated at instantiation -- invalid configs are caught before any LLM call

### Key API

| Class / Method              | Parameters                                                     | Returns          | Description                            |
| --------------------------- | -------------------------------------------------------------- | ---------------- | -------------------------------------- |
| `ToTAgent()`                | `llm_provider`, `model`, `num_paths`, `evaluation_criteria`    | `ToTAgent`       | Create a Tree-of-Thoughts agent        |
| `agent.run()`               | `task: str`                                                    | `dict`           | Execute the four-phase ToT pipeline    |
| `agent._evaluate_path()`    | `path: dict`, `task: str`                                      | `ToTEvaluation`  | Score a single path                    |
| `agent._select_best_path()` | `evaluations: list`                                            | `ToTEvaluation`  | Pick the highest-scoring evaluation    |
| `ToTAgentConfig()`          | `num_paths`, `max_paths`, `evaluation_criteria`, `temperature` | `ToTAgentConfig` | Configuration for exploration behavior |
| `ToTSignature`              | --                                                             | `Signature`      | Multi-path I/O definition              |
| `ToTPath`                   | --                                                             | `TypedDict`      | Structure for a single reasoning path  |
| `ToTEvaluation`             | --                                                             | `TypedDict`      | Structure for a path evaluation        |

## Code Walkthrough

### Imports and Setup

```python
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

from kaizen_agents.agents.specialized.tree_of_thoughts import (
    ToTAgent,
    ToTAgentConfig,
    ToTSignature,
    ToTPath,
    ToTEvaluation,
)
from kaizen.core.base_agent import BaseAgent

model = os.environ.get("DEFAULT_LLM_MODEL", "claude-sonnet-4-20250514")
```

The module exports the agent, config, signature, and two TypedDicts (`ToTPath` and `ToTEvaluation`) that provide type safety for paths and their evaluations.

### ToTPath and ToTEvaluation -- Typed Structures

```python
# ToTPath has required and optional fields
sample_path: ToTPath = {
    "path_id": 1,
    "reasoning": "Approach via cost-benefit analysis",
}
assert sample_path["path_id"] == 1
assert sample_path["reasoning"] == "Approach via cost-benefit analysis"

# ToTEvaluation scores a path
sample_eval: ToTEvaluation = {
    "path": {"path_id": 1, "reasoning": "Cost-benefit analysis"},
    "score": 0.85,
    "reasoning": "Well-structured approach with clear trade-offs",
}
assert sample_eval["score"] == 0.85
```

`ToTPath` holds a path's ID, reasoning, and optional steps. `ToTEvaluation` wraps a path with a score (0.0-1.0) and evaluation reasoning. These TypedDicts provide structure without the overhead of full dataclasses.

### ToTSignature -- Multi-Path I/O

```python
assert "task" in ToTSignature._signature_inputs

assert "paths" in ToTSignature._signature_outputs
assert "evaluations" in ToTSignature._signature_outputs
assert "best_path" in ToTSignature._signature_outputs
assert "final_result" in ToTSignature._signature_outputs
```

The signature takes a `task` input and produces four outputs: all generated `paths`, their `evaluations`, the `best_path` selection, and the extracted `final_result`.

### ToTAgentConfig

```python
config = ToTAgentConfig()

assert config.num_paths == 5, "Generate 5 reasoning paths by default"
assert config.max_paths == 20, "Safety limit: max 20 paths"
assert config.evaluation_criteria == "quality", "Default: quality-based evaluation"
assert config.parallel_execution is True, "Parallel generation by default"
assert config.temperature == 0.9, "High temperature for diverse paths"

# Custom configuration
custom_config = ToTAgentConfig(
    num_paths=10,
    max_paths=20,
    evaluation_criteria="creativity",
    parallel_execution=True,
)
assert custom_config.num_paths == 10
assert custom_config.evaluation_criteria == "creativity"
```

Notable defaults: `temperature=0.9` is much higher than other agents (0.1-0.7) because diverse paths require creative variation. `parallel_execution=True` generates all paths concurrently for speed. `num_paths=5` balances exploration breadth against token cost.

### Agent Instantiation and Validation

```python
agent = ToTAgent(
    llm_provider="mock",
    model=model,
    num_paths=5,
    evaluation_criteria="quality",
)

assert isinstance(agent, ToTAgent)
assert isinstance(agent, BaseAgent)
assert agent.tot_config.num_paths == 5

# num_paths > max_paths raises ValueError
try:
    ToTAgent(
        llm_provider="mock",
        model=model,
        num_paths=25,
        max_paths=20,
    )
    assert False, "num_paths > max_paths should raise ValueError"
except ValueError as e:
    assert "exceeds" in str(e).lower()
```

ToT uses `AsyncSingleShotStrategy` (the default) because the multi-path generation is handled internally -- each path is a separate `BaseAgent.run()` call. The `num_paths > max_paths` guard catches misconfiguration at construction time.

### Path Evaluation Logic

```python
# Good path: meets all criteria
good_path = {
    "path_id": 1,
    "reasoning": "A" * 150,  # Long reasoning
    "steps": ["step1", "step2", "step3"],
}
good_eval = agent._evaluate_path(good_path, "test task")
assert good_eval["score"] == 1.0, "Perfect path gets 1.0"

# Path with error: loses 0.3
error_path = {
    "path_id": 2,
    "reasoning": "A" * 150,
    "steps": ["step1"],
    "error": "generation failed",
}
error_eval = agent._evaluate_path(error_path, "test task")
assert error_eval["score"] == 0.7, "Error costs 0.3"

# Empty path: scores 0.0
empty_path = {"path_id": 3, "reasoning": ""}
empty_eval = agent._evaluate_path(empty_path, "test task")
assert empty_eval["score"] == 0.0, "Empty path scores 0.0"

# Short path: partial score
short_path = {"path_id": 4, "reasoning": "Brief reasoning about the problem"}
short_eval = agent._evaluate_path(short_path, "test task")
assert 0.0 < short_eval["score"] < 1.0, "Short path gets partial score"
```

The four evaluation criteria and their weights:

| Criterion         | Weight | Condition                      |
| ----------------- | ------ | ------------------------------ |
| Completeness      | 0.3    | Reasoning length > 20 chars    |
| No errors         | 0.3    | No `"error"` key in path dict  |
| Structured steps  | 0.2    | Non-empty `steps` list present |
| Reasoning quality | 0.2    | Reasoning length > 100 chars   |

A path with long reasoning, steps, and no errors scores 1.0. An empty reasoning string scores 0.0 (fails all criteria). An error costs 0.3 even if everything else is perfect.

### Best Path Selection

```python
evaluations = [
    {"path": {"path_id": 1}, "score": 0.6, "reasoning": "OK"},
    {"path": {"path_id": 2}, "score": 0.9, "reasoning": "Excellent"},
    {"path": {"path_id": 3}, "score": 0.3, "reasoning": "Weak"},
]

best = agent._select_best_path(evaluations)
assert best["score"] == 0.9
assert best["path"]["path_id"] == 2

# Empty evaluations: graceful fallback
empty_best = agent._select_best_path([])
assert empty_best["score"] == 0.0
```

Selection simply picks the evaluation with the highest score. Empty evaluations return a fallback with score 0.0 rather than raising an error.

### Input Validation

```python
empty_result = agent.run(task="")
assert empty_result["error"] == "INVALID_INPUT"
assert empty_result["paths"] == []
assert empty_result["evaluations"] == []
assert empty_result["final_result"] == ""
```

Empty tasks return immediately with all output fields populated as empty defaults.

### Return Structure

```python
# ToTAgent.run() returns:
# {
#     "paths": [
#         {"path_id": 0, "reasoning": "...", "steps": [...]},
#         {"path_id": 1, "reasoning": "...", "steps": [...]},
#         ...
#     ],
#     "evaluations": [
#         {"path": {...}, "score": 0.85, "reasoning": "..."},
#         ...
#     ],
#     "best_path": {"path": {...}, "score": 0.92, "reasoning": "..."},
#     "final_result": "The recommended approach is..."
# }
#
# Error codes: INVALID_INPUT, PATH_GENERATION_FAILED,
#              EVALUATION_FAILED, SELECTION_FAILED, EXECUTION_FAILED
```

The full result includes all paths (for transparency), all evaluations (for audit), the winning path, and the extracted final result. Five error codes cover different failure modes across the four phases.

## Common Mistakes

| Mistake                         | Correct Pattern                               | Why                                                                                            |
| ------------------------------- | --------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Setting `num_paths` too high    | Start with 3-5 paths                          | Each path requires an LLM call -- 20 paths means 20x the token cost of a single call           |
| Using low temperature for ToT   | Keep `temperature=0.9` (default)              | Low temperature produces near-identical paths, defeating the purpose of multi-path exploration |
| Using ToT for simple questions  | Use SimpleQA or CoT instead                   | ToT's overhead (N path generations + evaluations) is wasted on straightforward tasks           |
| Ignoring the `evaluations` list | Inspect all evaluations, not just `best_path` | The evaluation spread reveals how close alternatives were -- useful for decision confidence    |

## Exercises

1. Create a `ToTAgent` with `num_paths=3`. Construct three path dicts with different reasoning lengths (empty, short, long with steps) and call `_evaluate_path()` on each. Verify the scores match the criteria weights.
2. Create a list of five `ToTEvaluation` dicts with scores ranging from 0.1 to 0.9. Pass them to `_select_best_path()` and verify it returns the 0.9-scored path. Then pass an empty list and verify the fallback.
3. Try creating a `ToTAgent` with `num_paths=25` and `max_paths=20`. Catch the `ValueError` and verify the error message mentions "exceeds".

## Key Takeaways

- ToT explores N diverse reasoning paths in parallel, then evaluates and selects the best
- Four phases: Generate, Evaluate, Select, Execute -- each phase has distinct responsibilities
- Path evaluation uses four weighted criteria: completeness, no errors, structured steps, reasoning quality
- High temperature (0.9) is essential for generating diverse paths
- `num_paths` is guarded by `max_paths` to prevent runaway token costs
- The full result includes all paths and evaluations for transparency and audit

## Next Chapter

[Chapter 10: Multi-Agent Orchestration Patterns](10_orchestration_patterns.md) -- Explore the formal multi-agent coordination patterns: supervisor-worker, consensus, debate, handoff, and sequential pipeline.
