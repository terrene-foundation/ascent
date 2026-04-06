# Chapter 2: SimpleQA Agent

## Overview

The **SimpleQAAgent** is a zero-config question-answering agent that works out of the box with sensible defaults. This chapter teaches you how to use Kaizen's signature system for structured I/O, configure the agent progressively (from zero-config to fully customized), and understand the single-shot execution strategy that underpins interactive agents.

## Prerequisites

- Python 3.10+ installed
- Kailash Kaizen installed (`pip install kailash-kaizen`)
- Completion of Chapter 1 (Delegate with Tools & Events)
- Understanding of the BaseAgent class hierarchy

## Concepts

### Concept 1: Signatures for Structured I/O

Kaizen agents use **Signatures** to define typed inputs and outputs. The LLM receives the signature fields as structured prompts and returns structured data that maps back to the output fields. This replaces unstructured prompt engineering with a typed contract.

- **What**: A declarative schema of `InputField` and `OutputField` definitions that the agent uses for I/O
- **Why**: Structured I/O ensures consistent return formats, enables validation, and makes agent behavior predictable
- **How**: `QASignature` defines `question` and `context` as inputs, `answer`, `confidence`, and `reasoning` as outputs
- **When**: Every Kaizen agent uses a Signature -- it is the fundamental I/O contract

### Concept 2: Progressive Configuration

SimpleQAAgent supports three configuration levels: zero-config (just works), constructor overrides (targeted changes), and full config object (complete control). This progressive approach means beginners start immediately while experts can tune every parameter.

- **What**: A layered configuration system where each level adds more control
- **Why**: Reduces the barrier to entry -- you can use the agent with zero parameters and add configuration as needs grow
- **How**: (1) `SimpleQAAgent()`, (2) `SimpleQAAgent(temperature=0.2)`, (3) `SimpleQAAgent(config=SimpleQAConfig(...))`
- **When**: Start with zero-config, add parameters only when you need to change specific behaviors

### Concept 3: AsyncSingleShotStrategy

SimpleQA is an **interactive** agent -- it makes one LLM call per `run()` invocation. This is the `AsyncSingleShotStrategy`, which is the default when no `strategy=` is passed to BaseAgent. Autonomous agents (ReAct, RAG) use `MultiCycleStrategy` instead.

- **What**: An execution strategy that makes exactly one LLM call per `run()` invocation
- **Why**: Interactive agents answer questions directly -- they do not need iterative reasoning or tool loops
- **How**: The strategy is selected automatically based on the agent type; SimpleQA inherits the default
- **When**: For question answering, classification, summarization -- any task that needs one LLM call

### Key API

| Class / Method              | Parameters                                                    | Returns          | Description                                |
| --------------------------- | ------------------------------------------------------------- | ---------------- | ------------------------------------------ |
| `SimpleQAAgent()`           | `llm_provider: str`, `model: str`, `temperature: float`, ...  | `SimpleQAAgent`  | Create a QA agent (all params optional)    |
| `SimpleQAAgent(config=...)` | `config: SimpleQAConfig`                                      | `SimpleQAAgent`  | Create from a full config object           |
| `agent.run()`               | `question: str`, `context: str = ""`                          | `dict`           | Ask a question and get a structured answer |
| `SimpleQAConfig()`          | `temperature`, `max_tokens`, `timeout`, `retry_attempts`, ... | `SimpleQAConfig` | Configuration dataclass with defaults      |
| `QASignature`               | --                                                            | `Signature`      | Structured I/O definition for QA           |
| `SimpleQAAgent.metadata`    | --                                                            | `NodeMetadata`   | Auto-discovery metadata for Kailash Studio |

## Code Walkthrough

### Imports and Setup

```python
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

from kaizen_agents.agents.specialized.simple_qa import (
    SimpleQAAgent,
    SimpleQAConfig,
    QASignature,
)
from kaizen.core.base_agent import BaseAgent
from kaizen.signatures import InputField, OutputField, Signature

model = os.environ.get("DEFAULT_LLM_MODEL", "claude-sonnet-4-20250514")
```

SimpleQAAgent lives in `kaizen_agents.agents.specialized.simple_qa`. It extends `BaseAgent` from the core Kaizen framework. The `QASignature` defines the structured I/O contract.

### QASignature -- Structured I/O

```python
sig = QASignature()

assert "question" in QASignature._signature_inputs
assert "context" in QASignature._signature_inputs
assert "answer" in QASignature._signature_outputs
assert "confidence" in QASignature._signature_outputs
assert "reasoning" in QASignature._signature_outputs

# context has a default (optional), question does not (required)
context_field = QASignature._signature_inputs["context"]
assert context_field.required is False, "Context is optional"

question_field = QASignature._signature_inputs["question"]
assert question_field.required is True, "Question is required"
```

`QASignature` defines two inputs (`question` is required, `context` is optional) and three outputs (`answer`, `confidence`, `reasoning`). The LLM fills in the output fields based on the input fields. The `required` flag on each field controls whether the agent validates its presence before calling the LLM.

### SimpleQAConfig -- Sensible Defaults

```python
config = SimpleQAConfig()

assert config.temperature == 0.1, "Low temperature for factual answers"
assert config.max_tokens == 300, "Short answers by default"
assert config.timeout == 30
assert config.retry_attempts == 3
assert config.min_confidence_threshold == 0.5
assert config.max_turns is None, "Memory disabled by default (opt-in)"

# Override specific parameters
custom_config = SimpleQAConfig(
    temperature=0.3,
    max_tokens=500,
    min_confidence_threshold=0.7,
)
assert custom_config.temperature == 0.3
assert custom_config.min_confidence_threshold == 0.7
```

The config dataclass provides production-ready defaults: low temperature (0.1) for factual accuracy, short max tokens (300) for concise answers, 3 retries for resilience, and a 0.5 confidence threshold for quality gating. Memory is disabled by default (opt-in via `max_turns`).

### Zero-Config Instantiation

```python
agent = SimpleQAAgent(llm_provider="mock")

assert isinstance(agent, SimpleQAAgent)
assert isinstance(agent, BaseAgent), "SimpleQAAgent extends BaseAgent"
assert agent.qa_config is not None
```

With just a provider, SimpleQAAgent works immediately. It reads `KAIZEN_*` environment variables or falls back to defaults. This is the zero-config pattern -- no boilerplate required to get started.

### Progressive Configuration via Constructor

```python
configured_agent = SimpleQAAgent(
    llm_provider="mock",
    model=model,
    temperature=0.2,
    max_tokens=400,
    min_confidence_threshold=0.8,
)

assert configured_agent.qa_config.temperature == 0.2
assert configured_agent.qa_config.max_tokens == 400
assert configured_agent.qa_config.min_confidence_threshold == 0.8
```

Override individual parameters without creating a config object. The constructor passes them through to `SimpleQAConfig` internally. This is the sweet spot for most use cases -- change what you need, keep defaults for everything else.

### Configuration via Config Object

```python
full_config = SimpleQAConfig(
    llm_provider="mock",
    model=model,
    temperature=0.1,
    max_tokens=300,
    timeout=60,
    retry_attempts=5,
    min_confidence_threshold=0.6,
)

config_agent = SimpleQAAgent(config=full_config)
assert config_agent.qa_config.timeout == 60
assert config_agent.qa_config.retry_attempts == 5
```

For full control, create a `SimpleQAConfig` and pass it directly. This is useful when you need to share configuration across multiple agents or persist it.

### Memory Opt-In

```python
memory_agent = SimpleQAAgent(
    llm_provider="mock",
    max_turns=10,
)

assert memory_agent.qa_config.max_turns == 10
```

Memory is disabled by default. Set `max_turns` to enable `BufferMemory` for multi-turn conversation continuity. The buffer keeps the last N turns in context.

### Input Validation

```python
empty_result = agent.run(question="")
assert empty_result["error"] == "INVALID_INPUT"
assert empty_result["confidence"] == 0.0

whitespace_result = agent.run(question="   ")
assert whitespace_result["error"] == "INVALID_INPUT"
```

SimpleQA validates input before sending to the LLM. Empty or whitespace-only questions return an error immediately -- no LLM call is made, no tokens are consumed. This is input validation (a permitted deterministic guard), not agent reasoning.

### Return Structure

The `run()` method returns a dictionary with structured fields:

```python
# Successful response:
# {
#     "answer": "Paris",
#     "confidence": 0.95,
#     "reasoning": "France's capital is a well-known fact",
# }

# Low confidence response (below threshold):
# {
#     "answer": "...",
#     "confidence": 0.3,
#     "reasoning": "...",
#     "warning": "Low confidence (0.30 < 0.5)"
# }
```

When confidence falls below `min_confidence_threshold`, a `warning` field is added to alert the consumer.

### Node Metadata

```python
assert SimpleQAAgent.metadata.name == "SimpleQAAgent"
assert "qa" in SimpleQAAgent.metadata.tags
assert "question-answering" in SimpleQAAgent.metadata.tags
```

`NodeMetadata` enables auto-discovery in Kailash Studio. Tags like `"qa"` and `"question-answering"` allow the studio to categorize and present agents to users.

## Common Mistakes

| Mistake                                     | Correct Pattern                                     | Why                                                                                                |
| ------------------------------------------- | --------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| Passing `max_turns` without understanding   | Start with `max_turns=None` (default)               | Memory adds context to every call, increasing token costs -- enable only when multi-turn is needed |
| Setting temperature too high for factual QA | `temperature=0.1` (default)                         | High temperature produces creative but less accurate answers for factual questions                 |
| Ignoring the `warning` field in responses   | Check `if "warning" in result:` after every `run()` | Low-confidence answers may be wrong -- the warning signals the need for human review               |
| Using SimpleQA for multi-step reasoning     | Use ReActAgent or ChainOfThoughtAgent instead       | SimpleQA makes one LLM call -- it cannot iteratively refine or use tools                           |

## Exercises

1. Create a `SimpleQAAgent` with a custom config that sets `min_confidence_threshold=0.9`. Test it with `run(question="")` and verify the error response contains `"INVALID_INPUT"` with zero confidence.
2. Compare the `QASignature` fields with those of a custom `Signature` you define. Create a new signature with `InputField(description="...")` and `OutputField(description="...")` and verify that `_signature_inputs` and `_signature_outputs` are populated correctly.
3. Create two agents -- one with `max_turns=5` (memory enabled) and one with default settings (memory disabled). What configuration differences do you observe in their `qa_config`?

## Key Takeaways

- `SimpleQAAgent` is a zero-config question-answering agent that works out of the box
- `QASignature` defines structured I/O: required `question`, optional `context`, and three outputs
- Progressive configuration: zero-config -> constructor overrides -> full config object
- `AsyncSingleShotStrategy` means one LLM call per `run()` -- ideal for interactive agents
- Input validation catches empty questions before any LLM call
- Memory is opt-in via `max_turns` -- disabled by default to control token costs

## Next Chapter

[Chapter 3: ReAct Agent](03_react_agent.md) -- Build an autonomous agent that reasons and acts iteratively with tools, using the multi-cycle execution strategy.
