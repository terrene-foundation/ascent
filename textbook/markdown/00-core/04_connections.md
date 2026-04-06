# Chapter 4: Connections and Wiring

## Overview

Connections define how data flows between nodes in a workflow. Every connection maps an **output port** on one node to an **input port** on another. This chapter covers the three connection styles (default ports, mapping-based, and named ports), the low-level `add_connection()` API, fan-out patterns, and edge cases like self-connections and duplicates.

## Prerequisites

- Python 3.10+ installed
- Kailash SDK installed (`pip install kailash`)
- Completed [Chapter 1: WorkflowBuilder Basics](01_workflow_builder.md)
- Familiarity with `add_node()` and `build()`

## Concepts

### Concept 1: Default Port Routing

When you call `connect("source", "target")` with no extra arguments, Kailash uses `"data"` as both the output port name and the input port name. This is the simplest wiring pattern and works when the source node produces a single output and the target expects a single input.

- **What**: Implicit port routing using the `"data"` convention
- **Why**: Reduces boilerplate for the common case where each node has one input and one output
- **How**: Internally, `connect()` sets `from_output="data"` and `to_input="data"` when no mapping or port names are provided
- **When**: Use default ports for linear pipelines where each node consumes and produces a single data stream

### Concept 2: Mapping-Based Connections

The `mapping` parameter accepts a dictionary of `{from_port: to_port}` pairs. A single `connect()` call with a multi-key mapping creates multiple connections -- one per key. This is the preferred pattern when a node has multiple named outputs.

- **What**: A dictionary that explicitly maps output port names to input port names
- **Why**: Allows one call to wire multiple ports, and makes port routing self-documenting
- **How**: `connect("src", "dst", mapping={"output_a": "input_x", "output_b": "input_y"})` creates two connections
- **When**: Use mapping when a node exposes multiple outputs, or when port names differ between source and target

### Concept 3: Named Port Connections

For single-port wiring with explicit names, use `from_output` and `to_input` keyword arguments instead of `mapping`. This is equivalent to a single-entry mapping but reads more naturally for one-to-one connections.

- **What**: Keyword arguments that name the source output port and target input port
- **Why**: More readable than a one-key mapping dictionary for single-port connections
- **How**: `connect("src", "dst", from_output="result", to_input="value")`
- **When**: Use named ports when connecting exactly one output to one input with non-default names

### Concept 4: The add_connection() API

`add_connection()` is the low-level 4-argument method: `add_connection(from_node, from_output, to_node, to_input)`. `connect()` is syntactic sugar on top of it. Use `add_connection()` when you need maximum control or are building connections programmatically.

### Concept 5: Fan-Out

A single node can connect to multiple downstream nodes. Each `connect()` call creates an independent connection. Fan-out is how you broadcast one node's output to several consumers running in parallel.

## Key API

| Method             | Parameters                                                                                              | Returns | Description                              |
| ------------------ | ------------------------------------------------------------------------------------------------------- | ------- | ---------------------------------------- |
| `connect()`        | `source: str`, `target: str`, `mapping: dict = None`, `from_output: str = None`, `to_input: str = None` | `None`  | Wire an output port to an input port     |
| `add_connection()` | `from_node: str`, `from_output: str`, `to_node: str`, `to_input: str`                                   | `None`  | Low-level 4-argument connection creation |
| `connections`      | --                                                                                                      | `list`  | Current list of connection descriptors   |

## Code Walkthrough

```python
from __future__ import annotations

from kailash import WorkflowBuilder
from kailash.sdk_exceptions import ConnectionError, WorkflowValidationError

# ── 1. Default connection: "data" → "data" ─────────────────────────
# When no mapping is specified, connect() uses "data" as both the
# output port and input port.

builder = WorkflowBuilder()
builder.add_node("PythonCodeNode", "a", {"code": "output = 42", "output_type": "int"})
builder.add_node(
    "PythonCodeNode",
    "b",
    {"code": "output = data * 2", "inputs": {"data": "int"}, "output_type": "int"},
)
builder.connect("a", "b")  # defaults to data → data

assert len(builder.connections) == 1
conn = builder.connections[0]
assert conn["from_output"] == "data"
assert conn["to_input"] == "data"

# ── 2. Mapping-based connection ─────────────────────────────────────
# mapping={"from_port": "to_port"} gives explicit control over which
# output feeds which input. Multiple ports can be mapped at once.

builder2 = WorkflowBuilder()
builder2.add_node(
    "PythonCodeNode",
    "splitter",
    {
        "code": "first = text[:5]\nrest = text[5:]",
        "inputs": {"text": "str"},
        "output_type": "dict",
    },
)
builder2.add_node(
    "PythonCodeNode",
    "joiner",
    {
        "code": "output = f'{a} | {b}'",
        "inputs": {"a": "str", "b": "str"},
        "output_type": "str",
    },
)

# Map two outputs to two inputs in one call
builder2.connect("splitter", "joiner", mapping={"first": "a", "rest": "b"})
assert len(builder2.connections) == 2, "mapping with 2 keys creates 2 connections"

# ── 3. Named port connection (from_output / to_input) ──────────────
# Alternative to mapping for single-port connections.

builder3 = WorkflowBuilder()
builder3.add_node(
    "PythonCodeNode", "gen", {"code": "result = 99", "output_type": "int"}
)
builder3.add_node(
    "PythonCodeNode",
    "recv",
    {"code": "output = value + 1", "inputs": {"value": "int"}, "output_type": "int"},
)

builder3.connect("gen", "recv", from_output="result", to_input="value")
assert builder3.connections[0]["from_output"] == "result"
assert builder3.connections[0]["to_input"] == "value"

# ── 4. add_connection() — low-level 4-argument form ────────────────
# connect() is syntactic sugar. add_connection() is the underlying API:
#   add_connection(from_node, from_output, to_node, to_input)

builder4 = WorkflowBuilder()
builder4.add_node("PythonCodeNode", "x", {"code": "output = 1", "output_type": "int"})
builder4.add_node(
    "PythonCodeNode",
    "y",
    {"code": "output = n + 1", "inputs": {"n": "int"}, "output_type": "int"},
)

builder4.add_connection("x", "output", "y", "n")
assert len(builder4.connections) == 1
assert builder4.connections[0]["from_node"] == "x"
assert builder4.connections[0]["to_node"] == "y"

# ── 5. Fan-out: one node feeds multiple downstream nodes ───────────

builder5 = WorkflowBuilder()
builder5.add_node(
    "PythonCodeNode", "source", {"code": "output = 10", "output_type": "int"}
)
builder5.add_node(
    "PythonCodeNode",
    "double",
    {"code": "output = data * 2", "inputs": {"data": "int"}, "output_type": "int"},
)
builder5.add_node(
    "PythonCodeNode",
    "triple",
    {"code": "output = data * 3", "inputs": {"data": "int"}, "output_type": "int"},
)

builder5.connect("source", "double")
builder5.connect("source", "triple")
assert len(builder5.connections) == 2, "Fan-out: one source, two targets"

wf5 = builder5.build(name="fan-out")
assert len(wf5.node_instances) == 3

# ── 6. Edge case: self-connection ───────────────────────────────────

try:
    builder.connect("a", "a")
    assert False, "Self-connection should raise ConnectionError"
except ConnectionError:
    pass  # Expected

# ── 7. Edge case: duplicate connection ──────────────────────────────

builder6 = WorkflowBuilder()
builder6.add_node("PythonCodeNode", "p", {"code": "output = 1", "output_type": "int"})
builder6.add_node(
    "PythonCodeNode",
    "q",
    {"code": "output = data", "inputs": {"data": "int"}, "output_type": "int"},
)
builder6.connect("p", "q")

try:
    builder6.connect("p", "q")  # Same connection again
    assert False, "Duplicate connection should raise ConnectionError"
except ConnectionError:
    pass  # Expected

print("PASS: 00-core/04_connections")
```

### Step-by-Step Explanation

1. **Default ports**: `connect("a", "b")` with no extra arguments uses `"data"` for both `from_output` and `to_input`. Inspect `builder.connections[0]` to confirm. This is the zero-config path for simple linear pipelines.

2. **Mapping with multiple ports**: `connect("splitter", "joiner", mapping={"first": "a", "rest": "b"})` creates **two** connections from a single call. Each key in the mapping dict becomes a separate connection. This is how you wire a node with multiple named outputs.

3. **Named ports**: `connect("gen", "recv", from_output="result", to_input="value")` is the keyword-argument alternative to a single-key mapping. It reads more naturally when wiring exactly one port.

4. **Low-level add_connection()**: `add_connection("x", "output", "y", "n")` takes four positional arguments (from_node, from_output, to_node, to_input). This is the underlying API that `connect()` delegates to.

5. **Fan-out**: Two separate `connect()` calls from `"source"` to `"double"` and `"triple"` create a fan-out pattern. Both downstream nodes receive the same output from `"source"` and execute independently.

6. **Self-connection guard**: Connecting a node to itself raises `ConnectionError`. This prevents infinite loops in the execution graph.

7. **Duplicate connection guard**: Adding the same connection twice also raises `ConnectionError`. This prevents accidental double-wiring that would cause data duplication.

## Common Mistakes

| Mistake                                        | Correct Pattern                                       | Why                                                                                           |
| ---------------------------------------------- | ----------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Using `connect("a", "b")` when ports are named | `connect("a", "b", mapping={"output": "text"})`       | Default ports use `"data"` -- if your node's output is `"output"`, you must map it explicitly |
| Mapping with wrong direction                   | `mapping={"source_port": "target_port"}`              | Keys are source output names, values are target input names, not the reverse                  |
| Calling `add_connection` with 2 args           | `add_connection("src", "out_port", "dst", "in_port")` | `add_connection` always requires all four arguments; use `connect()` for the shorthand        |
| Expecting fan-in to merge results              | Use a MergeNode to combine multiple inputs            | Connecting multiple sources to the same input port overwrites; use MergeNode for aggregation  |

## Exercises

1. Create a workflow where one `PythonCodeNode` produces a dictionary with `"name"` and `"score"` fields, and two downstream nodes consume `"name"` and `"score"` respectively using mapping connections. Verify both downstream nodes receive the correct field.

2. What happens if you call `connect("a", "b", mapping={"x": "y"}, from_output="x")` -- providing both a mapping and named ports? Try it and observe whether the SDK raises an error or prefers one over the other.

3. Build a diamond-shaped workflow: one source fans out to two intermediate nodes, which both fan in to one final node. Use `add_connection()` for all wiring. What happens at the final node if both intermediates write to the same input port?

## Key Takeaways

- `connect()` supports three wiring styles: default ports, mapping-based, and named ports
- A multi-key `mapping` creates one connection per key in a single call
- `add_connection()` is the low-level 4-argument API underneath `connect()`
- Fan-out is built-in: connect one source to multiple targets with separate `connect()` calls
- Self-connections and duplicate connections are caught immediately with `ConnectionError`
- Default port routing uses `"data"` -- always use explicit mapping when your ports have custom names

## Next Chapter

[Chapter 5: LocalRuntime Execution](05_local_runtime.md) -- Execute workflows with LocalRuntime, inspect results, and use parameter overrides.
