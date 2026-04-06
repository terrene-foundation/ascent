# Chapter 6: PythonCodeNode Deep Dive

## Overview

`PythonCodeNode` is the most flexible node in Kailash -- it executes arbitrary Python code strings inside a sandboxed namespace. This chapter covers the full range of PythonCodeNode patterns: simple code strings, typed inputs, multiple outputs, stdlib imports, chaining, direct instantiation, `from_function()` for wrapping existing functions, and `from_class()` for stateful processing.

## Prerequisites

- Python 3.10+ installed
- Kailash SDK installed (`pip install kailash`)
- Completed [Chapter 5: LocalRuntime Execution](05_local_runtime.md)
- Familiarity with `WorkflowBuilder`, `connect()`, and `LocalRuntime`

## Concepts

### Concept 1: Code Strings and the Output Variable

The simplest PythonCodeNode pattern takes a `"code"` string that sets a variable named `output`. The runtime captures this variable as the node's result. Config keys are `"code"` (the Python source), `"output_type"` (the expected type), and optionally `"inputs"` (declared input variables).

- **What**: A Python code string executed in a sandboxed namespace, producing a result via the `output` variable
- **Why**: Allows custom logic without writing a full custom node class
- **How**: The runtime compiles and executes the code string, then extracts `output` from the resulting namespace
- **When**: Use for any custom transformation, computation, or logic that built-in nodes do not cover

### Concept 2: Input Variables

The `"inputs"` config key declares named variables that the code expects to receive from upstream connections. These variables are injected directly into the execution namespace before the code runs.

- **What**: A dictionary mapping variable names to type strings, declaring what the code expects
- **Why**: Makes the node's data contract explicit, enabling validation and documentation
- **How**: The runtime injects each declared input as a local variable in the code's namespace
- **When**: Use whenever the code needs data from upstream nodes

### Concept 3: Multiple Outputs

When the code sets multiple variables (not just `output`), all local variables become available as named outputs for downstream connections. Set `"output_type": "dict"` when producing multiple named values.

- **What**: Exporting several variables from a single code execution
- **Why**: A single node can split or decompose data without needing separate nodes
- **How**: All local variables in the code's namespace after execution are captured as outputs
- **When**: Use when a node needs to produce multiple distinct values consumed by different downstream nodes

### Concept 4: from_function() and from_class()

Beyond code strings, PythonCodeNode can wrap existing Python functions (`from_function()`) and stateful classes (`from_class()`). `from_function()` introspects the function's type annotations to build input and output types automatically. `from_class()` wraps a class whose `process()` method is called on each invocation, maintaining state across calls within one workflow run.

- **What**: Factory methods that create PythonCodeNode instances from functions or classes
- **Why**: Enables reuse of existing Python code without rewriting it as a code string
- **How**: `from_function()` reads type hints; `from_class()` instantiates the class and delegates to its `process()` method
- **When**: Use `from_function()` for pure transformations; use `from_class()` when you need accumulated state

## Key API

| Method / Config Key | Parameters                                     | Returns          | Description                     |
| ------------------- | ---------------------------------------------- | ---------------- | ------------------------------- |
| `"code"`            | Python source string                           | --               | The code to execute             |
| `"inputs"`          | `{name: type_string}` dict                     | --               | Declared input variables        |
| `"output_type"`     | Type string (`"int"`, `"str"`, `"dict"`, etc.) | --               | Expected output type            |
| `PythonCodeNode()`  | `name`, `code`, `input_types`, `output_type`   | `PythonCodeNode` | Direct constructor              |
| `from_function()`   | `func: Callable`                               | `PythonCodeNode` | Wrap a typed function as a node |
| `from_class()`      | `cls: type`                                    | `PythonCodeNode` | Wrap a stateful class as a node |

## Code Walkthrough

```python
from __future__ import annotations

from kailash import LocalRuntime, WorkflowBuilder
from kailash.nodes.code.python import PythonCodeNode

# ── 1. Simple code string — no inputs ──────────────────────────────
# The most basic pattern: a code string that sets the "output" variable.

builder = WorkflowBuilder()
builder.add_node(
    "PythonCodeNode",
    "constant",
    {
        "code": "output = 42",
        "output_type": "int",
    },
)

workflow = builder.build(name="simple-code")

with LocalRuntime() as rt:
    results, run_id = rt.execute(workflow)

constant_out = results.get("constant", {})

# ── 2. Code with inputs — the "inputs" config key ──────────────────
# Use "inputs" to declare named input variables the code expects.

builder2 = WorkflowBuilder()
builder2.add_node(
    "PythonCodeNode",
    "source",
    {
        "code": "output = 'kailash sdk'",
        "output_type": "str",
    },
)
builder2.add_node(
    "PythonCodeNode",
    "shout",
    {
        "code": "output = text.upper() + '!'",
        "inputs": {"text": "str"},
        "output_type": "str",
    },
)
builder2.connect("source", "shout", mapping={"output": "text"})

workflow2 = builder2.build(name="code-with-inputs")

with LocalRuntime() as rt:
    results2, _ = rt.execute(workflow2)

# ── 3. Multiple outputs — exporting several variables ──────────────
# When you set multiple variables (not just "output"), all become
# available as named outputs for downstream connections.

builder3 = WorkflowBuilder()
builder3.add_node(
    "PythonCodeNode",
    "splitter",
    {
        "code": "first_half = text[:len(text)//2]\nsecond_half = text[len(text)//2:]",
        "inputs": {"text": "str"},
        "output_type": "dict",
    },
)
builder3.add_node(
    "PythonCodeNode",
    "source3",
    {
        "code": "output = 'HelloWorld'",
        "output_type": "str",
    },
)
builder3.connect("source3", "splitter", mapping={"output": "text"})

workflow3 = builder3.build(name="multi-output")

with LocalRuntime() as rt:
    results3, _ = rt.execute(workflow3)

# ── 4. Code importing stdlib modules ───────────────────────────────
# PythonCodeNode allows importing from an allowlist of safe modules.

builder4 = WorkflowBuilder()
builder4.add_node(
    "PythonCodeNode",
    "math_node",
    {
        "code": "import math\noutput = math.sqrt(144) + math.pi",
        "output_type": "float",
    },
)

workflow4 = builder4.build(name="stdlib-import")

with LocalRuntime() as rt:
    results4, _ = rt.execute(workflow4)

# ── 5. Chaining multiple PythonCodeNodes ───────────────────────────
# A three-node pipeline: generate list -> filter -> sum.

builder5 = WorkflowBuilder()
builder5.add_node(
    "PythonCodeNode",
    "gen_list",
    {
        "code": "output = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]",
        "output_type": "list",
    },
)
builder5.add_node(
    "PythonCodeNode",
    "filter_evens",
    {
        "code": "output = [x for x in numbers if x % 2 == 0]",
        "inputs": {"numbers": "list"},
        "output_type": "list",
    },
)
builder5.add_node(
    "PythonCodeNode",
    "sum_them",
    {
        "code": "output = sum(values)",
        "inputs": {"values": "list"},
        "output_type": "int",
    },
)
builder5.connect("gen_list", "filter_evens", mapping={"output": "numbers"})
builder5.connect("filter_evens", "sum_them", mapping={"output": "values"})

workflow5 = builder5.build(name="chain-demo")

with LocalRuntime() as rt:
    results5, _ = rt.execute(workflow5)

# ── 6. Direct instantiation — PythonCodeNode constructor ───────────
# You can instantiate PythonCodeNode directly with typed parameters.

direct_node = PythonCodeNode(
    name="direct_greet",
    code="result = f'Hello, {name}!'",
    input_types={"name": str},
    output_type=str,
)

assert direct_node.code == "result = f'Hello, {name}!'"
assert direct_node.output_type is str
assert "name" in direct_node.input_types

builder6 = WorkflowBuilder()
builder6.add_node(direct_node, "greeter")

workflow6 = builder6.build(name="direct-instance")
assert "greeter" in workflow6.node_instances

# ── 7. from_function — wrap a Python function as a node ────────────
# PythonCodeNode.from_function() introspects the function signature
# to build input_types and output_type automatically.

def fahrenheit_to_celsius(temp_f: float) -> float:
    """Convert Fahrenheit to Celsius."""
    return (temp_f - 32) * 5 / 9

fn_node = PythonCodeNode.from_function(fahrenheit_to_celsius)

assert fn_node.function is fahrenheit_to_celsius
assert "temp_f" in fn_node.input_types

builder7 = WorkflowBuilder()
builder7.add_node(fn_node, "converter")

workflow7 = builder7.build(name="from-function-demo")
assert "converter" in workflow7.node_instances

# ── 8. from_class — wrap a stateful class as a node ────────────────
# Classes maintain state across invocations within the same workflow run.

class RunningTotal:
    """Accumulates values across calls."""

    def __init__(self):
        self.total = 0

    def process(self, value: float) -> dict:
        self.total += value
        return {"total": self.total, "last_value": value}

cls_node = PythonCodeNode.from_class(RunningTotal)

assert cls_node.class_type is RunningTotal
assert cls_node.instance is not None, "Class is pre-instantiated"

# ── 9. Edge case: code with no output variable ─────────────────────
# If the code does not set "output" or "result", the runtime captures
# all local variables as outputs.

builder9 = WorkflowBuilder()
builder9.add_node(
    "PythonCodeNode",
    "no_output_var",
    {
        "code": "x = 10\ny = 20\ntotal = x + y",
        "output_type": "dict",
    },
)

workflow9 = builder9.build(name="no-output-var")

with LocalRuntime() as rt:
    results9, _ = rt.execute(workflow9)

print("PASS: 00-core/06_python_code_node")
```

### Step-by-Step Explanation

1. **Simple code string**: The most basic pattern -- `"code": "output = 42"` sets the magic `output` variable that becomes the node's result. No inputs, no connections needed.

2. **Code with inputs**: Declaring `"inputs": {"text": "str"}` tells the node to expect a `text` variable in its namespace. The upstream connection via `mapping={"output": "text"}` delivers the value.

3. **Multiple outputs**: When the code sets variables like `first_half` and `second_half` instead of just `output`, all become available as named output ports for downstream mapping.

4. **Stdlib imports**: PythonCodeNode allows importing from a curated allowlist of safe standard library modules (`math`, `json`, `datetime`, `collections`, `itertools`, etc.).

5. **Chaining**: Three PythonCodeNodes wired in sequence form a pipeline: generate a list, filter for even numbers, sum them. Each node's output feeds the next via mapping connections.

6. **Direct instantiation**: `PythonCodeNode(name=..., code=..., input_types=..., output_type=...)` creates a typed node instance. Pass the instance directly to `add_node(direct_node, "node_id")`.

7. **from_function()**: Wraps an existing Python function. Type annotations on the function are used to auto-populate `input_types` and `output_type`. The function itself becomes the node's execution logic.

8. **from_class()**: Wraps a class with a `process()` method. The class is instantiated once and persists across invocations within the same workflow run, enabling stateful accumulation.

9. **No output variable**: When the code does not set `output` or `result`, the runtime captures all local variables as a dictionary of outputs. This is a fallback behavior, not a recommended pattern.

## Common Mistakes

| Mistake                                  | Correct Pattern                                 | Why                                                                                      |
| ---------------------------------------- | ----------------------------------------------- | ---------------------------------------------------------------------------------------- |
| Forgetting `"output_type"` in config     | `{"code": "output = 42", "output_type": "int"}` | The output type declaration is required for type checking and downstream connections     |
| Using undeclared input variables         | Add `"inputs": {"var_name": "type"}` to config  | Undeclared inputs are not injected into the namespace, causing `NameError`               |
| Importing blocked modules                | Use modules from the stdlib allowlist           | PythonCodeNode restricts imports to safe modules; attempting `import os` raises an error |
| Using `from_function` without type hints | Add `-> float` and parameter annotations        | `from_function()` reads type annotations; untyped functions produce untyped nodes        |

## Exercises

1. Create a PythonCodeNode that takes two string inputs (`first_name` and `last_name`) and produces a formatted greeting. Wire it to two source nodes and execute it.

2. Write a Python function with type annotations that computes the area of a circle given a radius. Use `from_function()` to wrap it as a node, add it to a workflow, and execute it with a parameter override.

3. Create a class with a `process()` method that counts how many times it has been called. Use `from_class()` and explain how the state would behave in a cyclic workflow.

## Key Takeaways

- PythonCodeNode executes code strings in a sandboxed namespace with the `output` variable as the result
- The `"inputs"` config key declares named input variables injected from upstream connections
- Multiple local variables become named output ports when `"output_type"` is `"dict"`
- Stdlib imports are allowed from a curated allowlist (math, json, datetime, etc.)
- `from_function()` wraps typed Python functions; `from_class()` wraps stateful classes
- Direct instantiation via `PythonCodeNode(...)` is an alternative to the config-dict pattern

## Next Chapter

[Chapter 7: Conditional Branching with SwitchNode](07_conditional_node.md) -- Implement workflow branching with SwitchNode for conditional routing based on data values.
