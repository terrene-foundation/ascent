# Chapter 5: LocalRuntime Execution

## Overview

Once you have a built `Workflow`, you need a **runtime** to execute it. `LocalRuntime` is the synchronous, single-machine runtime that runs all nodes in topological order and returns results as a dictionary. This chapter covers basic execution, the context manager pattern, parameter overrides, multiple executions on a single runtime, and the result structure.

## Prerequisites

- Python 3.10+ installed
- Kailash SDK installed (`pip install kailash`)
- Completed [Chapter 1: WorkflowBuilder Basics](01_workflow_builder.md) and [Chapter 4: Connections](04_connections.md)
- Understanding of `WorkflowBuilder.build()` producing an immutable `Workflow`

## Concepts

### Concept 1: The Runtime Abstraction

A runtime takes an immutable `Workflow` and executes it. Kailash provides two runtimes: `LocalRuntime` (synchronous, for scripts and CLI tools) and `AsyncLocalRuntime` (asynchronous, for servers and Docker). `LocalRuntime` wraps the async internals with its own event loop, so you never need `await`.

- **What**: A synchronous executor that runs workflows on the local machine
- **Why**: Provides a simple `execute(workflow)` call without requiring async boilerplate
- **How**: Internally creates an event loop, executes nodes in topological order, and returns a `(results_dict, run_id)` tuple
- **When**: Use `LocalRuntime` for scripts, CLI tools, notebooks, and tests. Use `AsyncLocalRuntime` for web servers and production deployments

### Concept 2: The Result Tuple

`execute()` returns a tuple of `(results: dict, run_id: str)`. The results dictionary is keyed by node ID, and each value contains that node's output. The `run_id` is a unique identifier for this execution -- useful for logging and debugging.

- **What**: A `(dict, str)` pair containing all node outputs and a unique execution identifier
- **Why**: The dict provides access to every node's output, not just the final one, enabling inspection and debugging
- **How**: Results are collected as each node completes; the dict is built incrementally during execution
- **When**: Always -- every call to `execute()` returns this tuple

### Concept 3: Context Manager Pattern

`LocalRuntime` manages an internal event loop and connection resources. Using `with LocalRuntime() as rt:` ensures proper cleanup. Without the context manager, you must call `runtime.close()` manually.

- **What**: Python's `with` statement applied to runtime lifecycle management
- **Why**: Guarantees resources are released even if an exception occurs during execution
- **How**: `__enter__` initializes the runtime; `__exit__` calls `close()` to tear down the event loop
- **When**: Always prefer the context manager unless you need the runtime to persist across multiple function boundaries

### Concept 4: Parameter Overrides

You can pass `parameters={node_id: {param: value}}` to `execute()` to inject values into nodes at runtime without rebuilding the workflow. This is how you make workflows reusable with different inputs.

- **What**: A dictionary of node-level parameter overrides passed at execution time
- **Why**: Separates the workflow definition (what to do) from the workflow input (what data to process)
- **How**: Overrides are merged into each node's input namespace before the node's code runs
- **When**: Use for any workflow that processes external input -- user queries, file paths, configuration values

## Key API

| Method / Property | Parameters                                      | Returns        | Description                                |
| ----------------- | ----------------------------------------------- | -------------- | ------------------------------------------ |
| `LocalRuntime()`  | --                                              | `LocalRuntime` | Create a synchronous runtime               |
| `execute()`       | `workflow: Workflow`, `parameters: dict = None` | `(dict, str)`  | Run the workflow, return (results, run_id) |
| `close()`         | --                                              | `None`         | Release all runtime resources              |
| Context manager   | `with LocalRuntime() as rt:`                    | `LocalRuntime` | Auto-cleanup runtime lifecycle             |

## Code Walkthrough

```python
from __future__ import annotations

from kailash import WorkflowBuilder, LocalRuntime


# ── Helper: build a simple workflow ─────────────────────────────────

def build_math_workflow() -> "Workflow":
    """Two-node workflow: generate a number, then double it."""
    builder = WorkflowBuilder()
    builder.add_node(
        "PythonCodeNode",
        "generate",
        {"code": "output = 21", "output_type": "int"},
    )
    builder.add_node(
        "PythonCodeNode",
        "double",
        {
            "code": "output = value * 2",
            "inputs": {"value": "int"},
            "output_type": "int",
        },
    )
    builder.connect("generate", "double", mapping={"output": "value"})
    return builder.build(name="math-demo")


# ── 1. Basic execution ──────────────────────────────────────────────
# LocalRuntime.execute() returns (results_dict, run_id).
# Results are keyed by node_id.

workflow = build_math_workflow()

runtime = LocalRuntime()
results, run_id = runtime.execute(workflow)
runtime.close()

assert isinstance(results, dict), "Results is a dict"
assert run_id is not None, "run_id is always returned"
assert "generate" in results, "Each node produces output"
assert "double" in results

# ── 2. Context manager pattern (recommended) ────────────────────────
# Ensures proper cleanup of the runtime's event loop and resources.

with LocalRuntime() as rt:
    results2, run_id2 = rt.execute(workflow)

assert results2 is not None
assert run_id2 != run_id, "Each execution gets a unique run_id"

# ── 3. Parameter overrides ──────────────────────────────────────────
# Pass parameters={node_id: {param: value}} to override node inputs
# at execution time without rebuilding the workflow.

builder = WorkflowBuilder()
builder.add_node(
    "PythonCodeNode",
    "greet",
    {
        "code": "output = f'Hello, {name}!'",
        "inputs": {"name": "str"},
        "output_type": "str",
    },
)
wf_greet = builder.build(name="param-demo")

with LocalRuntime() as rt:
    results3, _ = rt.execute(wf_greet, parameters={"greet": {"name": "Kailash"}})

greet_output = results3.get("greet", {})

# ── 4. Multiple executions on same runtime ──────────────────────────
# A single runtime instance can execute multiple workflows.
# Connection pools and async resources are reused.

with LocalRuntime() as rt:
    r1, id1 = rt.execute(workflow)
    r2, id2 = rt.execute(workflow)

assert id1 != id2, "Each execution gets a unique run_id"

# ── 5. Return structure ─────────────────────────────────────────────
# execute() returns a (dict, str) tuple. The dict is keyed by node_id.

assert isinstance(results, dict)
assert isinstance(run_id, (str, type(None)))

# ── 6. Edge case: empty workflow ────────────────────────────────────

empty_wf = WorkflowBuilder().build(name="empty")
with LocalRuntime() as rt:
    empty_results, empty_id = rt.execute(empty_wf)

assert isinstance(empty_results, dict)
assert len(empty_results) == 0, "Empty workflow produces no results"

print("PASS: 00-core/05_local_runtime")
```

### Step-by-Step Explanation

1. **Helper function**: `build_math_workflow()` creates a reusable two-node pipeline that generates 21 and doubles it to 42. Separating workflow construction from execution is a standard pattern.

2. **Basic execution**: Create a `LocalRuntime`, call `execute(workflow)`, then `close()`. The result tuple gives you every node's output plus a unique run ID for this execution.

3. **Context manager**: `with LocalRuntime() as rt:` is the recommended pattern. It handles `close()` automatically, even if your code raises an exception.

4. **Parameter overrides**: `parameters={"greet": {"name": "Kailash"}}` injects `name="Kailash"` into the `"greet"` node at execution time. The workflow definition stays unchanged -- you can reuse it with different parameters.

5. **Multiple executions**: A single runtime instance can execute the same or different workflows multiple times. Each call returns a new, unique `run_id`. Internal resources like connection pools are reused.

6. **Empty workflow**: Executing a workflow with zero nodes is valid. It returns an empty dict and a run ID.

## Common Mistakes

| Mistake                                            | Correct Pattern                      | Why                                                                                      |
| -------------------------------------------------- | ------------------------------------ | ---------------------------------------------------------------------------------------- |
| `runtime.execute(builder)` (no `.build()`)         | `runtime.execute(builder.build())`   | The runtime expects a `Workflow`, not a `WorkflowBuilder`. Missing `.build()` is a crash |
| Forgetting to call `close()`                       | Use `with LocalRuntime() as rt:`     | Leaked runtimes hold event loops and connection pools open indefinitely                  |
| Unpacking only results: `results = rt.execute(wf)` | `results, run_id = rt.execute(wf)`   | `execute()` returns a tuple; assigning to one variable gives you the tuple, not the dict |
| Using `LocalRuntime` in async code                 | Use `AsyncLocalRuntime` with `await` | `LocalRuntime` creates its own event loop, which conflicts with an existing one          |

## Exercises

1. Build a three-node pipeline where the first node generates a list of names, the second filters names starting with "A", and the third counts them. Execute it and inspect all three nodes' results from the returned dict.

2. Create a workflow with a `"greet"` node that uses a `name` input parameter. Execute it three times with different names using parameter overrides, all on the same `LocalRuntime` instance. Verify each run gets a unique `run_id`.

3. What happens if you pass a parameter override for a node ID that does not exist in the workflow? Try `parameters={"nonexistent": {"x": 1}}` and observe the behavior.

## Key Takeaways

- `LocalRuntime` is the synchronous runtime for scripts, CLI tools, and notebooks
- `execute()` returns `(results_dict, run_id)` -- results are keyed by node ID
- Always use the context manager pattern (`with LocalRuntime() as rt:`) for automatic cleanup
- Parameter overrides let you reuse workflows with different inputs without rebuilding
- A single runtime can execute multiple workflows; each gets a unique run ID
- Empty workflows execute successfully and return an empty results dict

## Next Chapter

[Chapter 6: PythonCodeNode Deep Dive](06_python_code_node.md) -- Master PythonCodeNode for custom logic: code strings, inputs, multiple outputs, `from_function()`, and `from_class()`.
