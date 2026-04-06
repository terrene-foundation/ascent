# Chapter 7: Conditional Branching with SwitchNode

## Overview

Real-world workflows need branching -- routing data down different paths based on conditions. Kailash provides `SwitchNode` for conditional routing (boolean true/false or multi-case) and `MergeNode` for rejoining branches. This chapter covers boolean conditions, all supported operators, multi-case switching, merging strategies, and using PythonCodeNode for custom conditional logic.

## Prerequisites

- Python 3.10+ installed
- Kailash SDK installed (`pip install kailash`)
- Completed [Chapter 6: PythonCodeNode Deep Dive](06_python_code_node.md)
- Understanding of `connect()` with mapping and `LocalRuntime`

## Concepts

### Concept 1: Boolean Mode

SwitchNode's simplest mode evaluates a condition and routes data to either `"true_output"` or `"false_output"`. You specify `condition_field` (which field in the input dict to check), `operator` (the comparison), and `value` (the comparison target).

- **What**: A binary routing decision based on a single condition
- **Why**: Enables if/else branching in a declarative workflow graph
- **How**: SwitchNode reads the field, applies the operator, and populates either `true_output` or `false_output` (the other is `None`)
- **When**: Use for binary decisions: pass/fail thresholds, feature flags, null checks

### Concept 2: Supported Operators

SwitchNode supports ten operators: `==`, `!=`, `>`, `<`, `>=`, `<=`, `in`, `contains`, `is_null`, and `is_not_null`. The `in` operator checks membership in a list. Unknown operators default to `False` (route to `false_output`).

- **What**: A set of comparison operators for condition evaluation
- **Why**: Covers equality, ordering, membership, and null checks without custom code
- **How**: The operator string is matched internally and applied to the field value and comparison value
- **When**: Choose the operator that matches your condition: `>=` for thresholds, `in` for allowlists, `is_null` for missing data

### Concept 3: Multi-Case Mode

Instead of binary true/false, SwitchNode can route to multiple named outputs using the `cases` parameter. Each case creates an output field named `"case_<value>"`. The matching case gets the data; non-matching cases are `None`. A `"default"` output always receives the data.

- **What**: N-way routing based on the value of a field
- **Why**: Implements switch/case logic without chaining multiple boolean SwitchNodes
- **How**: Each value in the `cases` list creates a named output port; the matched case is populated
- **When**: Use for categorical routing: priority levels, status codes, region-based processing

### Concept 4: MergeNode

After branching, `MergeNode` rejoins paths. It supports three merge types: `"concat"` (concatenate lists), `"zip"` (pair elements), and `"merge_dict"` (merge dictionaries, with later values winning on conflict).

- **What**: A node that combines data from multiple branches back into a single stream
- **Why**: Without merging, branched workflows produce fragmented results
- **How**: `MergeNode.run(data1=..., data2=..., merge_type="concat")` combines the inputs
- **When**: Use after any branching pattern where downstream processing needs the combined result

## Key API

| Class / Method        | Parameters                                                                   | Returns      | Description                                 |
| --------------------- | ---------------------------------------------------------------------------- | ------------ | ------------------------------------------- |
| `SwitchNode()`        | --                                                                           | `SwitchNode` | Create a conditional routing node           |
| `switch.run()`        | `input_data`, `condition_field`, `operator`, `value`, `cases`, `case_prefix` | `dict`       | Evaluate condition and route data           |
| `get_output_schema()` | --                                                                           | `dict`       | Inspect available output fields             |
| `get_parameters()`    | --                                                                           | `dict`       | Inspect all input parameters                |
| `MergeNode()`         | --                                                                           | `MergeNode`  | Create a merge node                         |
| `merge.run()`         | `data1`, `data2`, `merge_type`                                               | `dict`       | Combine inputs using the specified strategy |

## Code Walkthrough

```python
from __future__ import annotations

from kailash import WorkflowBuilder, LocalRuntime
from kailash.nodes.logic.operations import MergeNode, SwitchNode

# ── 1. SwitchNode basics — boolean mode ─────────────────────────────
# SwitchNode routes data to "true_output" or "false_output" based on
# evaluating a condition.

switch = SwitchNode()

# Boolean mode: check if score >= 80
result = switch.run(
    input_data={"score": 85, "name": "Alice"},
    condition_field="score",
    operator=">=",
    value=80,
)

assert result["true_output"] is not None, "Score 85 >= 80, should route to true"
assert result["false_output"] is None, "false_output should be None"
assert result["true_output"]["score"] == 85

# Condition that evaluates to false
result_fail = switch.run(
    input_data={"score": 50, "name": "Bob"},
    condition_field="score",
    operator=">=",
    value=80,
)

assert result_fail["true_output"] is None
assert result_fail["false_output"] is not None

# ── 2. Supported operators ──────────────────────────────────────────

# Equality
eq_result = switch.run(
    input_data={"status": "active"},
    condition_field="status",
    operator="==",
    value="active",
)
assert eq_result["true_output"] is not None

# Inequality
neq_result = switch.run(
    input_data={"status": "inactive"},
    condition_field="status",
    operator="!=",
    value="active",
)
assert neq_result["true_output"] is not None

# Membership (in)
in_result = switch.run(
    input_data={"role": "admin"},
    condition_field="role",
    operator="in",
    value=["admin", "superadmin"],
)
assert in_result["true_output"] is not None

# Null check
null_result = switch.run(
    input_data={"email": None},
    condition_field="email",
    operator="is_null",
    value=None,
)
assert null_result["true_output"] is not None

# ── 3. Multi-case switching ─────────────────────────────────────────

multi_result = switch.run(
    input_data={"priority": "high", "task": "deploy"},
    condition_field="priority",
    cases=["high", "medium", "low"],
)

assert multi_result["case_high"] is not None, "Input matches 'high' case"
assert multi_result["case_medium"] is None
assert multi_result["case_low"] is None
assert multi_result["default"] is not None, "Default always has the data"
assert multi_result["condition_result"] == "high"

# ── 4. MergeNode — rejoining branches ───────────────────────────────

merge = MergeNode()

# Concatenation merge
concat_result = merge.run(
    data1=[1, 2, 3],
    data2=[4, 5, 6],
    merge_type="concat",
)
assert concat_result["merged_data"] == [1, 2, 3, 4, 5, 6]

# Dictionary merge
dict_result = merge.run(
    data1={"name": "Alice", "age": 30},
    data2={"age": 31, "role": "engineer"},
    merge_type="merge_dict",
)
assert dict_result["merged_data"]["age"] == 31, "Later dict wins on conflict"
assert dict_result["merged_data"]["name"] == "Alice"

# ── 5. Conditional branching with PythonCodeNode ────────────────────
# For custom conditional logic beyond SwitchNode's operators.

builder = WorkflowBuilder()

builder.add_node(
    "PythonCodeNode",
    "categorize",
    {
        "code": (
            "if score >= 90:\n"
            "    output = 'excellent'\n"
            "elif score >= 70:\n"
            "    output = 'good'\n"
            "else:\n"
            "    output = 'needs_improvement'"
        ),
        "inputs": {"score": "int"},
        "output_type": "str",
    },
)

workflow = builder.build(name="python-conditional")

with LocalRuntime() as rt:
    results, _ = rt.execute(workflow, parameters={"categorize": {"score": 85}})

# ── 6. SwitchNode output schema ─────────────────────────────────────

output_schema = switch.get_output_schema()
assert "true_output" in output_schema
assert "false_output" in output_schema
assert "default" in output_schema
assert "condition_result" in output_schema

# ── 7. Edge case: missing condition field ───────────────────────────
# When condition_field is not present in the data, SwitchNode uses
# the input data directly as the value to check.

direct_result = switch.run(
    input_data=42,
    operator=">=",
    value=40,
)
assert direct_result["true_output"] == 42

# ── 8. Edge case: unknown operator ──────────────────────────────────
# An unknown operator returns False (routes to false_output).

unknown_op_result = switch.run(
    input_data={"val": 10},
    condition_field="val",
    operator="~=",
    value=10,
)
assert unknown_op_result["false_output"] is not None

print("PASS: 00-core/07_conditional_node")
```

### Step-by-Step Explanation

1. **Boolean mode**: `switch.run(input_data=..., condition_field="score", operator=">=", value=80)` evaluates the condition and populates either `true_output` or `false_output`. The other side is `None`. The full input data is passed through to the matching output.

2. **Operators**: Ten operators cover the common comparison needs. `==` and `!=` for equality, `>`, `<`, `>=`, `<=` for ordering, `in` for list membership, `contains` for substring checks, and `is_null`/`is_not_null` for null detection.

3. **Multi-case switching**: The `cases` parameter turns SwitchNode into a multi-way router. Each case value creates a `"case_<value>"` output field. The matched case gets the data; others are `None`. The `"default"` output always receives the data regardless of which case matched.

4. **MergeNode**: After branching, use MergeNode to combine results. `"concat"` joins lists, `"merge_dict"` merges dictionaries (later values win on key conflicts), and `"zip"` pairs elements.

5. **PythonCodeNode conditionals**: For complex multi-branch logic that does not fit SwitchNode's operators, use PythonCodeNode with if/elif/else in the code string. This gives full Python expressiveness at the cost of less declarative structure.

6. **Output schema**: `get_output_schema()` reveals all output fields a SwitchNode can produce, useful for programmatic workflow construction.

7. **Direct value comparison**: When `condition_field` is omitted or not found in the data, SwitchNode compares `input_data` directly against `value`.

8. **Unknown operator**: An unrecognized operator string defaults to `False` rather than raising an error, routing data to `false_output`.

## Common Mistakes

| Mistake                                    | Correct Pattern                       | Why                                                                                 |
| ------------------------------------------ | ------------------------------------- | ----------------------------------------------------------------------------------- |
| Using `"="` instead of `"=="`              | `operator="=="`                       | Single `=` is assignment; SwitchNode requires the comparison operator string `"=="` |
| Expecting `case_X` without passing `cases` | Add `cases=["high", "medium", "low"]` | Multi-case outputs are only created when the `cases` parameter is provided          |
| Forgetting MergeNode after branching       | Wire both branches into a MergeNode   | Without merging, only one branch's result reaches downstream nodes                  |
| Using SwitchNode for complex logic         | Use PythonCodeNode with if/elif/else  | SwitchNode handles single-condition routing; multi-condition logic needs code       |

## Exercises

1. Build a workflow that classifies temperatures: below 0 is "freezing", 0-20 is "cold", 20-35 is "warm", above 35 is "hot". Use either chained SwitchNodes or a PythonCodeNode with if/elif.

2. Create a workflow with a SwitchNode that routes orders by priority ("high", "medium", "low") using multi-case mode. Wire each case to a different PythonCodeNode that adds a processing tag. Merge the results at the end.

3. What happens when you use the `"in"` operator with a non-list value (e.g., `value="abc"`)? Test whether SwitchNode checks string membership or raises an error.

## Key Takeaways

- SwitchNode provides declarative conditional routing in boolean mode (true/false) or multi-case mode
- Ten operators cover equality, ordering, membership, and null checks
- Multi-case mode creates `"case_<value>"` output fields plus a `"default"` that always receives data
- MergeNode rejoins branches with `"concat"`, `"zip"`, or `"merge_dict"` strategies
- PythonCodeNode with if/elif/else handles complex conditional logic that SwitchNode cannot express
- Unknown operators default to `False` rather than raising errors

## Next Chapter

[Chapter 8: Value Types in Workflows](08_value_types.md) -- Understand how Python values map to Kailash's internal type system and how types are preserved through workflow execution.
