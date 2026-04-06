# Chapter 9: Error Handling

## Overview

Kailash provides a structured exception hierarchy rooted at `KailashException`. Every SDK error falls into a clear category -- build errors, connection errors, node errors, execution errors -- enabling both granular handling and catch-all safety nets. This chapter maps the full hierarchy, demonstrates each error type with triggering code, and shows the recommended error handling patterns for production use.

## Prerequisites

- Python 3.10+ installed
- Kailash SDK installed (`pip install kailash`)
- Completed [Chapter 5: LocalRuntime Execution](05_local_runtime.md)
- Basic understanding of Python exception handling (`try`/`except`)

## Concepts

### Concept 1: The Exception Hierarchy

All Kailash exceptions inherit from `KailashException`. Below it, two major branches exist: `WorkflowException` (build-time and connection errors) and `NodeException` (node-level configuration, execution, and safety errors). `RuntimeExecutionError` sits directly under `KailashException` for runtime-level failures.

- **What**: A tree of exception classes with `KailashException` at the root
- **Why**: Enables catching all SDK errors with one handler while still allowing specific catches for known conditions
- **How**: Python's exception inheritance means `except KailashException` catches every subclass
- **When**: Use specific types for expected conditions; use `KailashException` as a catch-all at the top level

### Concept 2: Build-Time vs Execution-Time Errors

`WorkflowValidationError` fires at build time (during `add_node()`, `connect()`, or `build()`). `NodeExecutionError` and `WorkflowExecutionError` fire during `execute()`. This separation means you can catch graph construction problems before any node runs.

- **What**: Two distinct phases where errors can occur
- **Why**: Build-time validation catches structural problems (duplicate IDs, missing nodes) cheaply, before any execution resources are allocated
- **How**: The builder validates incrementally: `add_node()` checks for duplicates, `connect()` checks for existence, `build()` validates the full graph
- **When**: Always handle both phases; build errors are programming mistakes, execution errors are runtime conditions

### Concept 3: The Granular Handling Pattern

For detailed diagnostics, catch specific exceptions first (most specific to least specific), then fall back to `KailashException`. Python's exception matching is order-dependent -- put narrow catches before broad ones.

- **What**: A try/except chain ordered from most specific to most general
- **Why**: Specific catches enable targeted recovery (retry on execution error, abort on validation error)
- **How**: `except WorkflowValidationError` before `except WorkflowException` before `except KailashException`
- **When**: Use in production code where different error types require different responses

### Concept 4: RetryExhaustedException

Some exceptions carry extra context beyond a message string. `RetryExhaustedException` includes the operation name, attempt count, last error, and total wait time. These attributes enable structured logging and alerting.

- **What**: An exception with typed attributes for retry-specific diagnostics
- **Why**: Structured attributes are machine-parseable, unlike message strings
- **How**: Created by retry logic when all attempts fail; carries the full retry history
- **When**: Encountered when database connections, API calls, or other retryable operations exhaust their retry budget

## Key API

| Exception Class           | Parent              | When Raised                                    |
| ------------------------- | ------------------- | ---------------------------------------------- |
| `KailashException`        | `Exception`         | Root of all SDK exceptions                     |
| `WorkflowException`       | `KailashException`  | Workflow-level errors (build + connect)        |
| `WorkflowValidationError` | `WorkflowException` | Invalid graph: duplicate IDs, missing nodes    |
| `WorkflowExecutionError`  | `WorkflowException` | Failure during workflow execution              |
| `ConnectionError`         | `WorkflowException` | Self-connection, duplicate connection          |
| `NodeException`           | `KailashException`  | Node-level errors                              |
| `NodeExecutionError`      | `NodeException`     | Node code fails during runtime execution       |
| `NodeValidationError`     | `NodeException`     | Node parameter validation fails                |
| `NodeConfigurationError`  | `NodeException`     | Invalid node construction (missing code, etc.) |
| `CodeExecutionError`      | `NodeException`     | PythonCodeNode code string fails               |
| `SafetyViolationError`    | `NodeException`     | Blocked operation in sandboxed code            |
| `RuntimeExecutionError`   | `KailashException`  | Runtime-level failure                          |
| `RetryExhaustedException` | `KailashException`  | All retry attempts failed                      |

## Code Walkthrough

```python
from __future__ import annotations

from kailash import LocalRuntime, WorkflowBuilder
from kailash.sdk_exceptions import (
    CodeExecutionError,
    ConnectionError,
    KailashException,
    NodeConfigurationError,
    NodeException,
    NodeExecutionError,
    NodeValidationError,
    RuntimeExecutionError,
    SafetyViolationError,
    WorkflowException,
    WorkflowExecutionError,
    WorkflowValidationError,
)

# ── 1. Exception hierarchy ─────────────────────────────────────────

assert issubclass(WorkflowValidationError, WorkflowException)
assert issubclass(WorkflowExecutionError, WorkflowException)
assert issubclass(WorkflowException, KailashException)

assert issubclass(NodeExecutionError, NodeException)
assert issubclass(NodeValidationError, NodeException)
assert issubclass(NodeConfigurationError, NodeException)
assert issubclass(SafetyViolationError, NodeException)
assert issubclass(CodeExecutionError, NodeException)
assert issubclass(NodeException, KailashException)

assert issubclass(ConnectionError, WorkflowException)
assert issubclass(RuntimeExecutionError, KailashException)

# ── 2. Build errors — WorkflowValidationError ──────────────────────

# Duplicate node ID
builder = WorkflowBuilder()
builder.add_node(
    "PythonCodeNode", "my_node", {"code": "output = 1", "output_type": "int"}
)

try:
    builder.add_node(
        "PythonCodeNode", "my_node", {"code": "output = 2", "output_type": "int"}
    )
except WorkflowValidationError as e:
    assert "my_node" in str(e)

# Connecting non-existent source node
builder2 = WorkflowBuilder()
builder2.add_node(
    "PythonCodeNode", "real_node", {"code": "output = 1", "output_type": "int"}
)

try:
    builder2.connect("ghost_node", "real_node")
except WorkflowValidationError:
    pass  # Expected: source node not found

# Connecting non-existent target node
try:
    builder2.connect("real_node", "ghost_target")
except WorkflowValidationError:
    pass  # Expected: target node not found

# ── 3. Connection errors — ConnectionError ─────────────────────────

builder3 = WorkflowBuilder()
builder3.add_node("PythonCodeNode", "a", {"code": "output = 1", "output_type": "int"})
builder3.add_node(
    "PythonCodeNode",
    "b",
    {"code": "output = data", "inputs": {"data": "int"}, "output_type": "int"},
)

# Self-connection
try:
    builder3.connect("a", "a")
except ConnectionError:
    pass  # Expected

# Duplicate connection
builder3.connect("a", "b")
try:
    builder3.connect("a", "b")
except ConnectionError:
    pass  # Expected

# ── 4. Node configuration errors — NodeConfigurationError ──────────

from kailash.nodes.code.python import PythonCodeNode

# No code, function, or class provided
try:
    node = PythonCodeNode(name="broken")
except NodeConfigurationError:
    pass  # Expected

# Both code and function provided (ambiguous)
try:
    node = PythonCodeNode(
        name="ambiguous",
        code="output = 1",
        function=lambda x: x,
    )
except NodeConfigurationError:
    pass  # Expected

# ── 5. Catch-all pattern with KailashException ─────────────────────

errors_caught = []

for scenario in ["valid", "duplicate_node", "bad_config"]:
    try:
        b = WorkflowBuilder()
        if scenario == "valid":
            b.add_node(
                "PythonCodeNode",
                "ok",
                {"code": "output = 'works'", "output_type": "str"},
            )
            wf = b.build(name="ok")
            with LocalRuntime() as rt:
                rt.execute(wf)
        elif scenario == "duplicate_node":
            b.add_node("PythonCodeNode", "dup", {"code": "output = 1"})
            b.add_node("PythonCodeNode", "dup", {"code": "output = 2"})
        elif scenario == "bad_config":
            PythonCodeNode(name="no_code_at_all")
    except KailashException as e:
        errors_caught.append((scenario, type(e).__name__, str(e)))

assert len(errors_caught) == 2

# ── 6. Granular error handling pattern ─────────────────────────────

def safe_build_and_run(builder: WorkflowBuilder, name: str) -> dict | str:
    """Build and run a workflow with comprehensive error handling."""
    try:
        workflow = builder.build(name=name)
        with LocalRuntime() as rt:
            results, run_id = rt.execute(workflow)
        return results
    except WorkflowValidationError as e:
        return f"BUILD_ERROR: {e}"
    except NodeExecutionError as e:
        return f"NODE_ERROR: {e}"
    except WorkflowExecutionError as e:
        return f"EXEC_ERROR: {e}"
    except KailashException as e:
        return f"SDK_ERROR ({type(e).__name__}): {e}"

# Test with a valid workflow
valid_builder = WorkflowBuilder()
valid_builder.add_node(
    "PythonCodeNode",
    "hello",
    {"code": "output = 'hello'", "output_type": "str"},
)
result = safe_build_and_run(valid_builder, "safe-test")
assert isinstance(result, dict)

# ── 7. Legacy compatibility names ──────────────────────────────────

from kailash.sdk_exceptions import KailashRuntimeError, KailashValidationError

assert KailashRuntimeError is RuntimeExecutionError
assert KailashValidationError is NodeValidationError

# ── 8. Specialized exception attributes ────────────────────────────

from kailash.sdk_exceptions import RetryExhaustedException

original_error = TimeoutError("Connection timed out")
retry_error = RetryExhaustedException(
    operation="database_connect",
    attempts=3,
    last_error=original_error,
    total_wait_time=4.5,
)

assert retry_error.operation == "database_connect"
assert retry_error.attempts == 3
assert retry_error.last_error is original_error
assert retry_error.total_wait_time == 4.5
assert "3 retry attempts" in str(retry_error)

print("PASS: 00-core/09_error_handling")
```

### Step-by-Step Explanation

1. **Hierarchy verification**: The `issubclass` checks confirm the inheritance tree. `WorkflowValidationError` descends from `WorkflowException` which descends from `KailashException`. This means `except KailashException` catches everything.

2. **Duplicate node ID**: Adding a node with an ID that already exists raises `WorkflowValidationError` immediately at `add_node()` time, not at `build()` time. The error message includes the duplicate ID.

3. **Non-existent connections**: Connecting to or from a node that does not exist raises `WorkflowValidationError` at `connect()` time. Both source and target are validated.

4. **Connection errors**: `ConnectionError` is distinct from `WorkflowValidationError`. It covers structural connection problems: self-connections (a node wired to itself) and duplicate connections (the same pair wired twice).

5. **Node configuration errors**: `NodeConfigurationError` fires when constructing a node with invalid parameters -- no code/function/class provided, or multiple conflicting sources.

6. **Catch-all pattern**: Wrapping everything in `except KailashException` ensures no SDK error escapes unhandled. The valid scenario runs without error; the other two are caught.

7. **Granular handling**: The `safe_build_and_run` function demonstrates the recommended production pattern: specific catches for build, node, and execution errors, with `KailashException` as the final fallback.

8. **Legacy aliases**: `KailashRuntimeError` and `KailashValidationError` are re-exports pointing to the current class names, ensuring backward compatibility.

9. **RetryExhaustedException attributes**: This exception carries typed attributes (`operation`, `attempts`, `last_error`, `total_wait_time`) for structured logging. The string representation includes human-readable details.

## Common Mistakes

| Mistake                                          | Correct Pattern                                         | Why                                                                                     |
| ------------------------------------------------ | ------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| Catching bare `Exception`                        | Catch `KailashException` for SDK errors                 | Bare `Exception` hides non-SDK bugs (like `TypeError` in your own code)                 |
| Putting `KailashException` before specific types | Order specific catches first, general last              | Python matches the first applicable `except`; a general catch shadows specific ones     |
| Ignoring error messages                          | Log `str(e)` -- messages contain actionable suggestions | Kailash error messages often suggest the fix (e.g., "did you mean node X?")             |
| Assuming all errors fire at `build()` time       | Handle both build-time and execution-time errors        | `WorkflowValidationError` fires at build; `NodeExecutionError` fires during `execute()` |

## Exercises

1. Create a PythonCodeNode whose code raises a `ZeroDivisionError`. Execute it and catch the resulting exception. Which Kailash exception type wraps it? Can you access the original error?

2. Write a `safe_execute` function that retries execution up to 3 times on `WorkflowExecutionError` but immediately aborts on `WorkflowValidationError`. Test it with both error types.

3. Construct a `RetryExhaustedException` with your own values. Verify that `str(retry_error)` includes both the attempt count and total wait time in a human-readable format.

## Key Takeaways

- All Kailash exceptions inherit from `KailashException`, enabling catch-all handling
- `WorkflowValidationError` fires at build time; `NodeExecutionError` fires during execution
- `ConnectionError` covers self-connections and duplicate connections
- `NodeConfigurationError` catches invalid node construction (missing code, ambiguous config)
- The recommended pattern: specific catches first, `KailashException` catch-all last
- Legacy aliases (`KailashRuntimeError`, `KailashValidationError`) maintain backward compatibility
- `RetryExhaustedException` carries typed attributes for structured diagnostics

## Next Chapter

[Chapter 10: ConnectionManager](10_connection_manager.md) -- Use ConnectionManager for database-backed workflow infrastructure: the foundation for stateful engines.
