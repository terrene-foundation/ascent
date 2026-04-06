# Chapter 1: Delegate with Tools & Events

## Overview

The **Delegate** is the primary entry point for building autonomous AI agents in Kailash Kaizen. This chapter teaches you how to register tools that the Delegate can call during its Think-Act-Observe-Decide (TAOD) loop, and how to consume the typed event stream it produces. You will learn the ToolRegistry, ToolDef, and the complete event hierarchy.

## Prerequisites

- Python 3.10+ installed
- Kailash Kaizen installed (`pip install kailash-kaizen`)
- A `.env` file with `DEFAULT_LLM_MODEL` configured (see `rules/env-models.md`)
- Familiarity with async/await in Python

## Concepts

### Concept 1: ToolRegistry

ToolRegistry holds the tools a Delegate can call during execution. Each tool has a unique name, a human-readable description (for the model), a JSON Schema defining its parameters, and an async executor function that performs the action.

- **What**: A mutable container that collects tool definitions and converts them to the OpenAI function-calling wire format
- **Why**: The Delegate needs a structured way to discover and invoke tools -- the registry provides this with type safety and format conversion
- **How**: Call `registry.register(name, description, parameters, executor)` for each tool, then pass the registry to the Delegate constructor
- **When**: Use ToolRegistry whenever your Delegate needs to interact with external systems (file I/O, search, APIs, databases)

### Concept 2: ToolDef

ToolDef is the dataclass that holds a single tool's metadata (name, description, parameters). It can convert itself to the OpenAI function-calling format via `.to_openai_format()`. While ToolRegistry manages collections of tools, ToolDef represents one individual tool definition.

- **What**: A frozen dataclass representing a single tool's schema
- **Why**: Separates tool metadata from execution, allowing introspection and serialization
- **How**: Create with `ToolDef(name=..., description=..., parameters=...)` and call `.to_openai_format()`
- **When**: Use ToolDef when you need to inspect or serialize individual tool definitions outside of a registry

### Concept 3: Delegate Event Stream

When the Delegate runs, it yields a stream of typed events. Consumers match on event type using Python's structural pattern matching (`match`/`case`). This provides a clean, type-safe way to handle streaming text, tool calls, completions, errors, and budget exhaustion.

- **What**: An async iterator of `DelegateEvent` subclasses yielded by `delegate.run()`
- **Why**: Typed events replace fragile string parsing -- each event type has specific fields, and the compiler can check exhaustiveness
- **How**: `async for event in delegate.run("prompt"): match event: case TextDelta(text=t): ...`
- **When**: Always -- this is the standard pattern for consuming Delegate output

### Concept 4: Budget Tracking

The Delegate tracks LLM token costs against a configurable USD budget. Each LLM turn (not each tool call) incurs token cost. When the budget is exhausted, a `BudgetExhausted` event is emitted and execution stops gracefully.

- **What**: Per-session financial constraint on LLM usage
- **Why**: Prevents runaway costs from autonomous agents that make many LLM calls
- **How**: Set `budget_usd=5.0` in the Delegate constructor; check `consumed_usd` and `budget_remaining` at any time
- **When**: Always set a budget for autonomous agents -- it is the primary safety guard against infinite loops

### Key API

| Class / Method                | Parameters                                                                     | Returns                        | Description                               |
| ----------------------------- | ------------------------------------------------------------------------------ | ------------------------------ | ----------------------------------------- |
| `ToolRegistry()`              | --                                                                             | `ToolRegistry`                 | Create an empty tool registry             |
| `registry.register()`         | `name: str`, `description: str`, `parameters: dict`, `executor: Callable`      | `None`                         | Register a tool with its executor         |
| `registry.has_tool()`         | `name: str`                                                                    | `bool`                         | Check if a tool is registered             |
| `registry.tool_names`         | --                                                                             | `list[str]`                    | List of registered tool names             |
| `registry.get_openai_tools()` | --                                                                             | `list[dict]`                   | Convert all tools to OpenAI wire format   |
| `ToolDef()`                   | `name: str`, `description: str`, `parameters: dict`                            | `ToolDef`                      | Create a single tool definition           |
| `ToolDef.to_openai_format()`  | --                                                                             | `dict`                         | Convert to OpenAI function-calling format |
| `Delegate()`                  | `model: str`, `tools: ToolRegistry`, `system_prompt: str`, `budget_usd: float` | `Delegate`                     | Create a Delegate with tools and budget   |
| `delegate.run()`              | `prompt: str`                                                                  | `AsyncIterator[DelegateEvent]` | Execute and stream events                 |

## Code Walkthrough

### Setting Up the Environment

```python
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

from kaizen_agents import Delegate
from kaizen_agents.delegate.loop import ToolRegistry, ToolDef
from kaizen_agents.delegate.events import (
    DelegateEvent,
    TextDelta,
    ToolCallStart,
    ToolCallEnd,
    TurnComplete,
    BudgetExhausted,
    ErrorEvent,
)

model = os.environ.get("DEFAULT_LLM_MODEL", "claude-sonnet-4-20250514")
```

The imports pull in the Delegate itself, the tool registration system, and the complete event hierarchy. The model is read from environment variables -- never hardcoded.

### Creating and Registering Tools

```python
registry = ToolRegistry()

assert isinstance(registry, ToolRegistry)
assert len(registry.tool_names) == 0, "Fresh registry is empty"


# Define an async executor function (tools are always async)
async def read_file_executor(path: str) -> str:
    """Read a file and return its contents."""
    return f"Contents of {path}"


# Register the tool with name, description, schema, and executor
registry.register(
    name="read_file",
    description="Read a file from the filesystem",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file"},
        },
        "required": ["path"],
    },
    executor=read_file_executor,
)

assert registry.has_tool("read_file"), "Tool was registered"
assert not registry.has_tool("nonexistent"), "Unknown tool not found"
assert registry.tool_names == ["read_file"]
```

A fresh `ToolRegistry` starts empty. Each tool registration requires four things: a unique name, a human-readable description (the LLM reads this to decide when to call the tool), a JSON Schema for the parameters, and an async callable that executes the action. Tool executors are always async because they typically involve I/O (file reads, API calls, database queries).

### Multiple Tool Registration

```python
async def search_executor(query: str, max_results: int = 5) -> str:
    """Search for information."""
    return f"Results for '{query}' (max {max_results})"


registry.register(
    name="search",
    description="Search for information in the codebase",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "max_results": {
                "type": "integer",
                "description": "Max results",
                "default": 5,
            },
        },
        "required": ["query"],
    },
    executor=search_executor,
)

assert len(registry.tool_names) == 2
assert "search" in registry.tool_names
```

Real agents typically have many tools. Each follows the same registration pattern. The JSON Schema `"required"` array distinguishes mandatory from optional parameters.

### OpenAI Function-Calling Format

```python
openai_tools = registry.get_openai_tools()
assert len(openai_tools) == 2

first_tool = openai_tools[0]
assert first_tool["type"] == "function"
assert "function" in first_tool
assert first_tool["function"]["name"] == "read_file"
assert "parameters" in first_tool["function"]
```

`get_openai_tools()` converts all registered tools to the OpenAI function-calling wire format that LLM APIs expect. This is the format the model sees when deciding which tool to call.

### ToolDef -- Individual Tool Definitions

```python
tool_def = ToolDef(
    name="write_file",
    description="Write content to a file",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    },
)

assert tool_def.name == "write_file"
openai_format = tool_def.to_openai_format()
assert openai_format["type"] == "function"
assert openai_format["function"]["name"] == "write_file"
```

`ToolDef` is the standalone dataclass for a single tool's metadata. It can convert itself to OpenAI format independently of a registry -- useful for introspection and serialization.

### Delegate with Tool Registry

```python
delegate_with_tools = Delegate(
    model=model,
    tools=registry,
    system_prompt="You are a code reviewer. Use the available tools.",
    budget_usd=5.0,
)

assert delegate_with_tools.tool_registry is registry
assert delegate_with_tools.tool_registry.has_tool("read_file")
assert delegate_with_tools.tool_registry.has_tool("search")
```

Pass a pre-built `ToolRegistry` to the Delegate constructor. The Delegate's AgentLoop uses these tools during its TAOD (Think-Act-Observe-Decide) loop: the model decides which tool to call, the loop executes it, and the result is fed back to the model.

### Event Type System

```python
td = TextDelta(text="Hello")
assert td.event_type == "text_delta"
assert td.text == "Hello"
assert td.timestamp > 0, "Monotonic timestamp is always positive"

tcs = ToolCallStart(call_id="call_001", name="read_file")
assert tcs.event_type == "tool_call_start"
assert tcs.call_id == "call_001"
assert tcs.name == "read_file"

tce = ToolCallEnd(call_id="call_001", name="read_file", result="file contents")
assert tce.event_type == "tool_call_end"
assert tce.result == "file contents"
assert tce.error == "", "No error by default"

tc = TurnComplete(
    text="Analysis complete", usage={"prompt_tokens": 100, "completion_tokens": 50}
)
assert tc.event_type == "turn_complete"
assert tc.usage["prompt_tokens"] == 100

be = BudgetExhausted(budget_usd=5.0, consumed_usd=5.01)
assert be.event_type == "budget_exhausted"
assert be.consumed_usd > be.budget_usd

err = ErrorEvent(error="Connection timeout", details={"exception_type": "TimeoutError"})
assert err.event_type == "error"
assert err.details["exception_type"] == "TimeoutError"
```

The complete event hierarchy consists of six types, all inheriting from `DelegateEvent`:

| Event             | Purpose                         | Key Fields                           |
| ----------------- | ------------------------------- | ------------------------------------ |
| `TextDelta`       | Streaming text from the model   | `text`                               |
| `ToolCallStart`   | A tool invocation has begun     | `call_id`, `name`                    |
| `ToolCallEnd`     | A tool invocation has completed | `call_id`, `name`, `result`, `error` |
| `TurnComplete`    | The model finished responding   | `text`, `usage`                      |
| `BudgetExhausted` | Budget cap exceeded             | `budget_usd`, `consumed_usd`         |
| `ErrorEvent`      | An error occurred               | `error`, `details`                   |

All events carry an `event_type` discriminator and a monotonic `timestamp`.

### Event Stream Consumption Pattern

```python
# The standard pattern for consuming Delegate events:
#
#   async for event in delegate.run("prompt"):
#       match event:
#           case TextDelta(text=t):
#               print(t, end="")
#           case ToolCallStart(name=n):
#               show_spinner(n)
#           case ToolCallEnd(name=n, result=r):
#               hide_spinner(n)
#           case TurnComplete(text=t, usage=u):
#               print(f"\nTokens: {u}")
#           case BudgetExhausted():
#               print("Budget exceeded")
#           case ErrorEvent(error=e):
#               print(f"Error: {e}")
```

This is the canonical pattern for consuming Delegate output. Python's structural pattern matching provides exhaustive, type-safe event handling.

### Tool Execution Error Handling

```python
tce_err = ToolCallEnd(
    call_id="call_002",
    name="read_file",
    result="",
    error="FileNotFoundError: /nonexistent.py",
)
assert tce_err.error != "", "Error field populated on failure"
assert tce_err.result == "", "No result on failure"
```

When a tool call fails, `ToolCallEnd` carries the error message in its `error` field. The `result` field is empty on failure. The model receives this error and can decide to retry, try a different approach, or report the failure.

### Budget Tracking

```python
assert delegate_with_tools.budget_usd == 5.0
assert delegate_with_tools.consumed_usd == 0.0
assert delegate_with_tools.budget_remaining == 5.0
```

Budget tracking applies to the full session including tool calls. Each LLM turn (not each tool call) incurs token cost. Before any execution, `consumed_usd` is zero and `budget_remaining` equals the full budget.

## Common Mistakes

| Mistake                              | Correct Pattern                                 | Why                                                                                       |
| ------------------------------------ | ----------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Synchronous tool executor            | `async def executor(...):`                      | All tool executors must be async because they typically involve I/O operations            |
| Missing JSON Schema `required` array | `"required": ["path"]` in parameters            | Without `required`, the model may omit mandatory parameters, causing runtime errors       |
| Ignoring `ToolCallEnd.error`         | Check `if tce.error:` before using `tce.result` | Failed tool calls have empty results -- using them silently produces wrong answers        |
| No budget set on autonomous agents   | `Delegate(model=model, budget_usd=5.0)`         | Without a budget, a reasoning loop can make unlimited LLM calls and incur unbounded costs |

## Exercises

1. Create a `ToolRegistry` with three tools: `list_files` (lists directory contents), `read_file` (reads a file), and `write_file` (writes content to a file). Convert the registry to OpenAI format and verify each tool has the correct parameter schema.
2. Write an event consumer function that counts how many tool calls a Delegate makes during a run. Use pattern matching to count `ToolCallStart` events and print a summary when `TurnComplete` arrives.
3. Create two `ToolDef` instances with different parameter schemas (one with required-only params, one with required + optional). Convert both to OpenAI format and compare the resulting JSON structures.

## Key Takeaways

- `ToolRegistry` is the container for all tools a Delegate can call during its TAOD loop
- Each tool requires a name, description, JSON Schema parameters, and an async executor
- `get_openai_tools()` converts registered tools to the LLM wire format
- `Delegate.run()` yields typed events -- use pattern matching for clean consumption
- Six event types cover the full lifecycle: text streaming, tool calls, completion, budget, and errors
- Always set `budget_usd` on autonomous agents to prevent runaway costs

## Next Chapter

[Chapter 2: SimpleQA Agent](02_simple_qa.md) -- Build a zero-config question-answering agent with structured inputs and outputs using Kaizen's signature system.
