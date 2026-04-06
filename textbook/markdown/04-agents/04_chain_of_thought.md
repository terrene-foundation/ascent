# Chapter 4: Chain-of-Thought Agent

## Overview

The **ChainOfThoughtAgent** decomposes complex problems into five explicit reasoning steps, producing an auditable chain of logic in a single LLM call. Unlike ReAct (iterative with tools) or SimpleQA (single answer), CoT fills in a structured five-step template that makes every intermediate thought visible. This chapter teaches you the CoT pattern, verification, and when to choose CoT over other reasoning strategies.

## Prerequisites

- Python 3.10+ installed
- Kailash Kaizen installed (`pip install kailash-kaizen`)
- Completion of Chapters 1-3 (Delegate, SimpleQA, ReAct)
- Understanding of the difference between single-shot and multi-cycle strategies

## Concepts

### Concept 1: Five-Step Reasoning Structure

CoT uses a fixed five-step structure: (1) Problem understanding, (2) Data identification and organization, (3) Systematic calculation or analysis, (4) Solution verification, (5) Final answer formulation. This forces the LLM to show its work rather than jump to a conclusion.

- **What**: A structured template with five named reasoning steps plus a final answer and confidence score
- **Why**: Explicit steps make reasoning auditable -- every intermediate thought is recorded and reviewable
- **How**: The `ChainOfThoughtSignature` defines `step1` through `step5`, `final_answer`, and `confidence` as outputs
- **When**: Best for math problems, logical deduction, analysis tasks, and anywhere an audit trail is required

### Concept 2: Single-Shot Structured Reasoning

CoT is a single-shot agent (like SimpleQA) that uses `AsyncSingleShotStrategy`. All five steps are produced in one LLM call. This is different from ReAct, which iterates over multiple calls. The entire reasoning chain is generated at once.

- **What**: One LLM call produces the complete five-step reasoning chain
- **Why**: Step-by-step analysis that does not need external data can be completed in a single pass
- **How**: The LLM receives the problem and fills in all five step fields plus the final answer simultaneously
- **When**: When the problem can be solved from the prompt alone without fetching external information

### Concept 3: Verification

When `enable_verification=True`, the agent adds a `verified` boolean to the result. If confidence meets or exceeds the threshold, the answer is marked as verified. Below threshold, a warning is added. This provides a programmatic quality gate on CoT outputs.

- **What**: A post-processing step that compares confidence against the configured threshold
- **Why**: Automated verification flags uncertain answers for human review without blocking the pipeline
- **How**: `verified = confidence >= confidence_threshold`; if False, a `warning` field is added
- **When**: Enable for production workloads where answer quality must be programmatically gated

### Concept 4: Text Response Extraction (Fallback Parser)

When the LLM returns plain text instead of structured JSON (e.g., with a mock provider), CoT extracts structure from the text using pattern matching. This is a safety net for graceful degradation, not the primary path.

- **What**: A regex-based fallback that parses "Step N:" and "Final answer:" markers from freeform text
- **Why**: Not all LLM providers reliably return structured JSON -- the fallback prevents total failure
- **How**: `_extract_from_text_response()` scans for step markers, extracts content, and assigns default confidence (0.5)
- **When**: Automatically invoked when the LLM response is not valid JSON

### Key API

| Class / Method                        | Parameters                                                             | Returns                | Description                              |
| ------------------------------------- | ---------------------------------------------------------------------- | ---------------------- | ---------------------------------------- |
| `ChainOfThoughtAgent()`               | `llm_provider`, `model`, `confidence_threshold`, `enable_verification` | `ChainOfThoughtAgent`  | Create a CoT agent                       |
| `agent.run()`                         | `problem: str`, `context: str = ""`                                    | `dict`                 | Solve a problem with five-step reasoning |
| `agent._extract_from_text_response()` | `text: str`                                                            | `dict`                 | Fallback parser for non-JSON responses   |
| `ChainOfThoughtConfig()`              | `temperature`, `reasoning_steps`, `enable_verification`, ...           | `ChainOfThoughtConfig` | Configuration for CoT behavior           |
| `ChainOfThoughtSignature`             | --                                                                     | `Signature`            | Five-step I/O definition                 |

## Code Walkthrough

### Imports and Setup

```python
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

from kaizen_agents.agents.specialized.chain_of_thought import (
    ChainOfThoughtAgent,
    ChainOfThoughtConfig,
    ChainOfThoughtSignature,
)
from kaizen.core.base_agent import BaseAgent

model = os.environ.get("DEFAULT_LLM_MODEL", "claude-sonnet-4-20250514")
```

### ChainOfThoughtSignature -- Five-Step Structure

```python
assert "problem" in ChainOfThoughtSignature._signature_inputs
assert "context" in ChainOfThoughtSignature._signature_inputs

# Five reasoning steps + final answer + confidence
assert "step1" in ChainOfThoughtSignature._signature_outputs
assert "step2" in ChainOfThoughtSignature._signature_outputs
assert "step3" in ChainOfThoughtSignature._signature_outputs
assert "step4" in ChainOfThoughtSignature._signature_outputs
assert "step5" in ChainOfThoughtSignature._signature_outputs
assert "final_answer" in ChainOfThoughtSignature._signature_outputs
assert "confidence" in ChainOfThoughtSignature._signature_outputs

# context is optional (has default)
context_field = ChainOfThoughtSignature._signature_inputs["context"]
assert context_field.required is False
```

Two inputs (`problem` required, `context` optional) and seven outputs (five steps, final answer, confidence). The five steps follow a fixed reasoning progression: understand, identify data, analyze, verify, formulate.

### ChainOfThoughtConfig

```python
config = ChainOfThoughtConfig()

assert config.temperature == 0.1, "Low temperature for precise reasoning"
assert config.max_tokens == 1500, "Larger context for step-by-step output"
assert config.timeout == 45, "Longer timeout for complex reasoning"
assert config.reasoning_steps == 5, "Five reasoning steps"
assert config.confidence_threshold == 0.7
assert config.enable_verification is True, "Verification enabled by default"

# Custom configuration for stricter reasoning
strict_config = ChainOfThoughtConfig(
    confidence_threshold=0.9,
    reasoning_steps=5,
    enable_verification=True,
)
assert strict_config.confidence_threshold == 0.9
```

CoT defaults differ from SimpleQA: higher `max_tokens` (1500 vs 300) to accommodate the five-step output, longer timeout (45s vs 30s) for complex reasoning, and verification enabled by default. Temperature remains low (0.1) for precision.

### Agent Instantiation

```python
agent = ChainOfThoughtAgent(
    llm_provider="mock",
    model=model,
    confidence_threshold=0.7,
    enable_verification=True,
)

assert isinstance(agent, ChainOfThoughtAgent)
assert isinstance(agent, BaseAgent)
assert agent.cot_config.enable_verification is True
```

CoT uses `AsyncSingleShotStrategy` (the default). Unlike ReAct, it does not iterate -- all five steps are produced in one LLM call.

### Input Validation

```python
empty_result = agent.run(problem="")
assert empty_result["error"] == "INVALID_INPUT"
assert empty_result["confidence"] == 0.0
assert empty_result["final_answer"] == "Please provide a clear problem to solve."

# All five steps present even on error
for i in range(1, 6):
    assert f"step{i}" in empty_result
```

Even on validation failure, all five step keys are present in the result. This ensures consumers can always access the expected fields without key-checking.

### Text Response Extraction (Fallback)

```python
text_response = """Step 1: Understand the problem - we need to multiply 15 by 23.
Step 2: Break down the numbers - 15 = 10 + 5, 23 = 20 + 3.
Step 3: Apply distributive property - (10+5)(20+3) = 200+30+100+15.
Step 4: Sum the parts - 200+30+100+15 = 345.
Step 5: Verify by estimation - 15*20=300, 15*3=45, total=345. Correct.
Final answer: 345"""

extracted = agent._extract_from_text_response(text_response)

assert "step1" in extracted
assert "final_answer" in extracted
assert extracted["final_answer"] == "345"
assert extracted["confidence"] == 0.5, "Default confidence for text-parsed responses"

# Missing steps get empty strings
sparse_text = "Final answer: 42"
sparse_result = agent._extract_from_text_response(sparse_text)
assert sparse_result["final_answer"] == "42"
for i in range(1, 6):
    assert f"step{i}" in sparse_result  # All step keys present
```

The fallback parser handles freeform text by scanning for "Step N:" markers and "Final answer:" prefix. Missing steps get empty strings. Confidence defaults to 0.5 for text-parsed responses since the parser cannot assess the model's certainty.

### Verification Flag

```python
# When enable_verification=True, the result includes:
# {
#     "step1": "Understanding the problem...",
#     "step2": "Identifying the numbers...",
#     "step3": "Systematic calculation...",
#     "step4": "Verifying the result...",
#     "step5": "Formulating final answer...",
#     "final_answer": "345",
#     "confidence": 0.95,
#     "verified": True,   # confidence >= threshold
# }
#
# When confidence < threshold:
# {
#     "final_answer": "...",
#     "confidence": 0.4,
#     "verified": False,
#     "warning": "Low confidence (0.40 < 0.7)",
# }

no_verify_agent = ChainOfThoughtAgent(
    llm_provider="mock",
    model=model,
    enable_verification=False,
)
assert no_verify_agent.cot_config.enable_verification is False
```

Verification adds a programmatic quality gate. When disabled, the `verified` and `warning` fields are absent from the result.

### CoT vs ReAct: When to Use Which

| Criterion    | CoT (This Chapter)         | ReAct (Chapter 3)                               |
| ------------ | -------------------------- | ----------------------------------------------- |
| Execution    | Single-shot (one LLM call) | Multi-cycle (many LLM calls)                    |
| Tools        | No external tools          | Uses tools for data retrieval                   |
| Best for     | Math, logic, audit trails  | Research, multi-step tasks, dynamic exploration |
| Transparency | All five steps visible     | Thought-action log per cycle                    |
| Strategy     | `AsyncSingleShotStrategy`  | `MultiCycleStrategy`                            |

## Common Mistakes

| Mistake                                   | Correct Pattern                             | Why                                                                        |
| ----------------------------------------- | ------------------------------------------- | -------------------------------------------------------------------------- |
| Using CoT for tasks needing external data | Use ReActAgent with tools instead           | CoT cannot fetch data -- it reasons only from the prompt context           |
| Ignoring the fallback parser              | Handle both JSON and text responses         | Some providers return freeform text -- the fallback prevents total failure |
| Trusting `confidence` from text parsing   | Treat 0.5 default confidence as "uncertain" | Text-parsed confidence is a placeholder, not a model assessment            |
| Disabling verification in production      | Keep `enable_verification=True`             | Without verification, low-confidence answers pass through unchecked        |

## Exercises

1. Create a `ChainOfThoughtAgent` with `confidence_threshold=0.8` and `enable_verification=True`. Construct a mock result dict with `confidence=0.6` and determine whether the `verified` field would be True or False, and whether a `warning` would be present.
2. Write a text response string that includes all five "Step N:" markers and a "Final answer:" line. Pass it to `_extract_from_text_response()` and verify all fields are populated. Then write a response with only steps 1 and 5 -- what happens to steps 2-4?
3. Compare the `ChainOfThoughtConfig` defaults with `SimpleQAConfig` defaults. Why does CoT use higher `max_tokens` and longer `timeout`?

## Key Takeaways

- CoT decomposes problems into five explicit reasoning steps in a single LLM call
- `ChainOfThoughtSignature` defines `step1` through `step5`, `final_answer`, and `confidence`
- Verification adds a `verified` boolean and optional `warning` for programmatic quality gating
- The text fallback parser provides graceful degradation when structured JSON is unavailable
- CoT is single-shot -- for tasks needing external data or iterative refinement, use ReAct instead
- All step keys are always present in the result, even on error or sparse parsing

## Next Chapter

[Chapter 5: RAG Research Agent](05_rag_research.md) -- Build a retrieval-augmented generation agent that combines vector search with LLM synthesis for research tasks.
