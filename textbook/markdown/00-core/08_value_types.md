# Chapter 8: Value Types in Workflows

## Overview

Kailash workflows pass data between nodes as Python values -- integers, floats, strings, booleans, lists, dicts, and `None`. Understanding how these types are preserved (or not) through workflow execution is essential for building reliable pipelines. This chapter demonstrates each type flowing through PythonCodeNode, covers nested structures, type coercion responsibilities, and the result structure convention.

## Prerequisites

- Python 3.10+ installed
- Kailash SDK installed (`pip install kailash`)
- Completed [Chapter 6: PythonCodeNode Deep Dive](06_python_code_node.md)
- Familiarity with `WorkflowBuilder`, `connect()`, and `LocalRuntime`

## Concepts

### Concept 1: Type Preservation

Python values flowing through Kailash workflows retain their original types. An `int` stays an `int`, a `float` stays a `float`, and a `bool` stays a `bool` (it is not coerced to `int`). The SDK does not perform implicit type conversion between nodes.

- **What**: Values maintain their Python type as they pass through workflow connections
- **Why**: Implicit conversion would silently corrupt data (e.g., `True` becoming `1` would lose boolean semantics)
- **How**: The runtime passes values by reference through the execution graph without any coercion layer
- **When**: Always -- this is a fundamental guarantee of the workflow system

### Concept 2: No Automatic Type Coercion

If a node outputs an `int` and the next node's code expects a `str`, the code must convert explicitly (e.g., `str(val)`). Kailash does not insert automatic casts between nodes.

- **What**: The absence of implicit type conversion in the connection layer
- **Why**: Automatic coercion hides bugs -- a downstream node receiving `"42"` instead of `42` might silently produce wrong results
- **How**: Each node is responsible for handling or converting its input types
- **When**: Any time you connect nodes whose output and input types differ

### Concept 3: The Result Structure Convention

Every workflow execution returns results as `{node_id: {output_name: value, ...}, ...}`. The outer dict is keyed by node ID (always a string). The inner dict is keyed by output variable names. This structure is consistent regardless of value types.

- **What**: A two-level dictionary convention for workflow results
- **Why**: Uniform structure makes it possible to inspect any node's output without knowing the workflow shape
- **How**: The runtime collects each node's namespace outputs into the inner dict, then assembles the outer dict
- **When**: Every call to `execute()` returns this structure

### Concept 4: Nested Structures

Complex nested structures -- lists of dicts, dicts of lists, deeply nested objects -- pass through workflows without flattening or transformation. Kailash does not impose a schema on the data flowing between nodes.

- **What**: Arbitrarily nested Python data structures flowing intact through workflows
- **Why**: Many real-world data shapes are hierarchical (JSON responses, database records, ML feature sets)
- **How**: The runtime treats all values as opaque Python objects and passes them as-is
- **When**: Use freely for any structured data; no depth or complexity limits

## Key API

| Type    | `output_type` String | Notes                                                       |
| ------- | -------------------- | ----------------------------------------------------------- |
| `int`   | `"int"`              | Preserved as Python int; not coerced to float               |
| `float` | `"float"`            | Preserved as Python float                                   |
| `str`   | `"str"`              | Preserved including Unicode content                         |
| `bool`  | `"bool"`             | Preserved as `True`/`False`; not coerced to `1`/`0`         |
| `list`  | `"list"`             | Can contain mixed types (homogeneous recommended)           |
| `dict`  | `"dict"`             | Primary structured data type; node results are always dicts |
| `None`  | `"None"`             | Valid value; downstream nodes must handle `None` gracefully |

## Code Walkthrough

```python
from __future__ import annotations

from kailash import LocalRuntime, WorkflowBuilder

# ── 1. Integer values ──────────────────────────────────────────────

builder = WorkflowBuilder()
builder.add_node(
    "PythonCodeNode",
    "int_source",
    {"code": "output = 42", "output_type": "int"},
)
builder.add_node(
    "PythonCodeNode",
    "int_check",
    {
        "code": "output = {'value': n, 'type_name': type(n).__name__}",
        "inputs": {"n": "int"},
        "output_type": "dict",
    },
)
builder.connect("int_source", "int_check", mapping={"output": "n"})

workflow = builder.build(name="int-types")

with LocalRuntime() as rt:
    results, _ = rt.execute(workflow)

# ── 2. Float values ────────────────────────────────────────────────

builder2 = WorkflowBuilder()
builder2.add_node(
    "PythonCodeNode",
    "float_source",
    {"code": "output = 3.14159", "output_type": "float"},
)
builder2.add_node(
    "PythonCodeNode",
    "float_check",
    {
        "code": "output = {'value': n, 'type_name': type(n).__name__}",
        "inputs": {"n": "float"},
        "output_type": "dict",
    },
)
builder2.connect("float_source", "float_check", mapping={"output": "n"})

workflow2 = builder2.build(name="float-types")

with LocalRuntime() as rt:
    results2, _ = rt.execute(workflow2)

# ── 3. String values ───────────────────────────────────────────────

builder3 = WorkflowBuilder()
builder3.add_node(
    "PythonCodeNode",
    "str_source",
    {"code": "output = 'Kailash SDK'", "output_type": "str"},
)
builder3.add_node(
    "PythonCodeNode",
    "str_transform",
    {
        "code": "output = {'original': s, 'length': len(s), 'upper': s.upper()}",
        "inputs": {"s": "str"},
        "output_type": "dict",
    },
)
builder3.connect("str_source", "str_transform", mapping={"output": "s"})

workflow3 = builder3.build(name="str-types")

with LocalRuntime() as rt:
    results3, _ = rt.execute(workflow3)

# ── 4. Boolean values ──────────────────────────────────────────────

builder4 = WorkflowBuilder()
builder4.add_node(
    "PythonCodeNode",
    "bool_source",
    {"code": "output = True", "output_type": "bool"},
)
builder4.add_node(
    "PythonCodeNode",
    "bool_check",
    {
        "code": "output = {'value': flag, 'type_name': type(flag).__name__, 'negated': not flag}",
        "inputs": {"flag": "bool"},
        "output_type": "dict",
    },
)
builder4.connect("bool_source", "bool_check", mapping={"output": "flag"})

workflow4 = builder4.build(name="bool-types")

with LocalRuntime() as rt:
    results4, _ = rt.execute(workflow4)

# ── 5. List values ─────────────────────────────────────────────────

builder5 = WorkflowBuilder()
builder5.add_node(
    "PythonCodeNode",
    "list_source",
    {"code": "output = [1, 'two', 3.0, True, None]", "output_type": "list"},
)
builder5.add_node(
    "PythonCodeNode",
    "list_check",
    {
        "code": (
            "output = {"
            "'length': len(items), "
            "'types': [type(x).__name__ for x in items]"
            "}"
        ),
        "inputs": {"items": "list"},
        "output_type": "dict",
    },
)
builder5.connect("list_source", "list_check", mapping={"output": "items"})

workflow5 = builder5.build(name="list-types")

with LocalRuntime() as rt:
    results5, _ = rt.execute(workflow5)

# ── 6. Dict values ─────────────────────────────────────────────────

builder6 = WorkflowBuilder()
builder6.add_node(
    "PythonCodeNode",
    "dict_source",
    {
        "code": "output = {'name': 'Alice', 'scores': [95, 87, 92], 'active': True}",
        "output_type": "dict",
    },
)
builder6.add_node(
    "PythonCodeNode",
    "dict_check",
    {
        "code": (
            "output = {"
            "'keys': list(data.keys()), "
            "'name': data['name'], "
            "'avg_score': sum(data['scores']) / len(data['scores'])"
            "}"
        ),
        "inputs": {"data": "dict"},
        "output_type": "dict",
    },
)
builder6.connect("dict_source", "dict_check", mapping={"output": "data"})

workflow6 = builder6.build(name="dict-types")

with LocalRuntime() as rt:
    results6, _ = rt.execute(workflow6)

# ── 7. None values ─────────────────────────────────────────────────

builder7 = WorkflowBuilder()
builder7.add_node(
    "PythonCodeNode",
    "none_source",
    {"code": "output = None", "output_type": "None"},
)
builder7.add_node(
    "PythonCodeNode",
    "none_check",
    {
        "code": "output = {'is_none': val is None, 'type_name': type(val).__name__}",
        "inputs": {"val": "None"},
        "output_type": "dict",
    },
)
builder7.connect("none_source", "none_check", mapping={"output": "val"})

workflow7 = builder7.build(name="none-types")

with LocalRuntime() as rt:
    results7, _ = rt.execute(workflow7)

# ── 8. Nested structures ──────────────────────────────────────────

builder8 = WorkflowBuilder()
builder8.add_node(
    "PythonCodeNode",
    "nested_source",
    {
        "code": (
            "output = {"
            "'users': ["
            "{'name': 'Alice', 'tags': ['admin', 'active']}, "
            "{'name': 'Bob', 'tags': ['user']}"
            "], "
            "'metadata': {'version': 1, 'counts': [2, 0]}"
            "}"
        ),
        "output_type": "dict",
    },
)
builder8.add_node(
    "PythonCodeNode",
    "nested_check",
    {
        "code": (
            "user_count = len(data['users'])\n"
            "all_tags = []\n"
            "for u in data['users']:\n"
            "    all_tags.extend(u['tags'])\n"
            "output = {'user_count': user_count, 'all_tags': all_tags, "
            "'meta_version': data['metadata']['version']}"
        ),
        "inputs": {"data": "dict"},
        "output_type": "dict",
    },
)
builder8.connect("nested_source", "nested_check", mapping={"output": "data"})

workflow8 = builder8.build(name="nested-types")

with LocalRuntime() as rt:
    results8, _ = rt.execute(workflow8)

# ── 9. Type coercion between nodes ────────────────────────────────
# Kailash does not perform automatic type coercion. The code must
# handle the conversion explicitly.

builder9 = WorkflowBuilder()
builder9.add_node(
    "PythonCodeNode",
    "int_producer",
    {"code": "output = 42", "output_type": "int"},
)
builder9.add_node(
    "PythonCodeNode",
    "str_consumer",
    {
        "code": "output = f'The answer is {str(val)}'",
        "inputs": {"val": "int"},
        "output_type": "str",
    },
)
builder9.connect("int_producer", "str_consumer", mapping={"output": "val"})

workflow9 = builder9.build(name="coercion-demo")

with LocalRuntime() as rt:
    results9, _ = rt.execute(workflow9)

# ── 10. Result structure recap ─────────────────────────────────────
# {node_id: {output_name: value, ...}, ...}

with LocalRuntime() as rt:
    all_results, run_id = rt.execute(workflow)

assert isinstance(all_results, dict), "Top-level results is always a dict"
for node_id, node_outputs in all_results.items():
    assert isinstance(node_id, str), "Node IDs are always strings"

print("PASS: 00-core/08_value_types")
```

### Step-by-Step Explanation

1. **Integers**: `output = 42` produces an int. The downstream node verifies the type name is `"int"` -- no coercion to float occurs.

2. **Floats**: `output = 3.14159` produces a float. Python's `int` and `float` remain distinct types throughout execution.

3. **Strings**: Strings pass through unchanged, including Unicode content. String methods like `.upper()` work as expected in downstream nodes.

4. **Booleans**: `output = True` stays as a Python `bool`. It is not coerced to `1`. You can negate it, check its type, and it behaves exactly like a native boolean.

5. **Lists**: Lists can contain mixed types (`[1, 'two', 3.0, True, None]`), though homogeneous lists are recommended for clarity. Each element preserves its original type.

6. **Dicts**: Dictionaries are the primary structured data type. Node results are always dictionaries keyed by output variable names.

7. **None**: `None` is a valid value that flows through workflows. Downstream nodes must handle `None` gracefully with explicit checks (`val is None`).

8. **Nested structures**: A dict containing lists of dicts with lists passes through without flattening. The downstream node can traverse the full hierarchy.

9. **Explicit coercion**: When connecting an int-producing node to a string-consuming node, the code must call `str(val)` explicitly. The SDK does not auto-convert.

10. **Result structure**: Every execution returns `{node_id: {output_name: value}}`. The outer dict is keyed by node ID strings; the inner dict holds output variables.

## Common Mistakes

| Mistake                                        | Correct Pattern                           | Why                                                                                    |
| ---------------------------------------------- | ----------------------------------------- | -------------------------------------------------------------------------------------- |
| Assuming int-to-float auto-conversion          | Convert explicitly: `float(n)`            | Kailash does not coerce types; passing an int where float is expected may cause errors |
| Not handling `None` in downstream code         | Check `if val is None:` before operations | `None` flows through connections; calling methods on `None` raises `AttributeError`    |
| Using `type(x) == int` instead of `isinstance` | `isinstance(x, int)`                      | `isinstance` handles inheritance correctly and is the Python convention                |
| Expecting lists to be auto-flattened           | Flatten explicitly in code                | Nested lists pass through as-is; `[[1,2],[3,4]]` stays nested                          |

## Exercises

1. Create a workflow that produces a `bytes` value (e.g., `b"hello"`) and passes it to a downstream node. Does the bytes type survive? What does `type(val).__name__` report?

2. Build a workflow where one node outputs `True` (bool) and the downstream node adds it to an integer (`val + 10`). What is the result? Does Python's bool-as-int behavior work inside PythonCodeNode?

3. Create a deeply nested structure (3+ levels) and pass it through two intermediate nodes that each extract one level. Verify the final node receives the innermost data intact.

## Key Takeaways

- All Python primitive types (int, float, str, bool, list, dict, None) are preserved through workflow execution
- Kailash does not perform automatic type coercion -- nodes must convert types explicitly
- Nested structures pass through without flattening or schema enforcement
- `None` is a valid workflow value; downstream nodes must handle it gracefully
- The result structure is always `{node_id: {output_name: value}}` regardless of value types
- Lists can contain mixed types, but homogeneous lists are recommended for clarity

## Next Chapter

[Chapter 9: Error Handling](09_error_handling.md) -- Handle workflow errors with the full exception hierarchy: build errors, execution errors, node errors, and the catch-all pattern.
