# Chapter 7: Governed Supervisor

## Overview

The **GovernedSupervisor** orchestrates multi-agent execution under PACT governance. It enforces budget limits, data clearance levels, tool allowlists, and audit trails -- providing a safe operating envelope for autonomous agent teams. This chapter teaches you the three-layer progressive API, constraint envelopes, governance subsystems, and the `SupervisorResult` dataclass.

## Prerequisites

- Python 3.10+ installed
- Kailash Kaizen installed (`pip install kailash-kaizen`)
- Completion of Chapters 1-6 (all individual agent types)
- Familiarity with async/await and `asyncio.run()`

## Concepts

### Concept 1: Progressive Disclosure (Three Layers)

GovernedSupervisor hides its full 20-concept surface area behind three progressive layers. Layer 1 requires only a model and budget. Layer 2 adds tools, clearance, and thresholds. Layer 3 exposes the full governance subsystems. This means beginners can start with two parameters while experts access everything.

- **What**: A three-tier API where each layer reveals more configuration options
- **Why**: Multi-agent governance is inherently complex -- progressive disclosure prevents beginners from drowning in options
- **How**: Layer 1: `GovernedSupervisor(model, budget_usd)`. Layer 2: add `tools`, `data_clearance`, `warning_threshold`. Layer 3: access `.audit`, `.budget`, `.accountability` directly
- **When**: Start at Layer 1, escalate to Layer 2 when you need tool restrictions or clearance, Layer 3 for programmatic governance inspection

### Concept 2: PACT Default-Deny

PACT governance follows a default-deny model: an empty tools list means no tools are allowed, `PUBLIC` clearance means no sensitive data access, and a $1 budget means minimal spending. Every capability must be explicitly granted. This is the opposite of an "allow all, restrict later" approach.

- **What**: A governance model where all capabilities start at zero and must be explicitly enabled
- **Why**: Default-deny prevents accidental access to tools, data, or budgets that were not intentionally granted
- **How**: Empty `tools=[]`, `ConfidentialityLevel.PUBLIC`, minimal `budget_usd` are the defaults
- **When**: Always -- this is the foundation of PACT governance, not an option

### Concept 3: Constraint Envelope

The constraint envelope encapsulates all operational limits: financial (budget), operational (max children, max depth), temporal (timeout), and data access (clearance level). It is returned as a deep copy each time, ensuring consumers cannot mutate the supervisor's internal state.

- **What**: An immutable snapshot of all constraints governing agent execution
- **Why**: Centralizing constraints in one object makes it possible to inspect, serialize, and audit the full operating envelope
- **How**: `supervisor.envelope` returns a deep copy with `.financial`, `.operational`, `.temporal`, and `.data_access` sections
- **When**: Use for inspection, logging, or passing constraints to child agents

### Concept 4: Governance Subsystems

Layer 3 exposes seven governance subsystems as read-only proxies: audit trail, budget tracker, accountability tracker, cascade manager, clearance enforcer, dereliction detector, bypass manager, and vacancy manager. Each proxy allows query methods but blocks mutation.

- **What**: Read-only views into the internal governance machinery
- **Why**: Observability without risk -- you can inspect any governance state without accidentally modifying it
- **How**: `supervisor.audit`, `supervisor.budget`, `supervisor.accountability`, etc.
- **When**: For monitoring dashboards, compliance reports, post-run analysis

### Concept 5: SupervisorResult

`SupervisorResult` is a frozen (immutable) dataclass returned by `supervisor.run()`. It contains the execution outcome, per-task results, budget consumption, audit trail, events, and plan. Immutability guarantees that results cannot be tampered with after execution.

- **What**: A frozen dataclass capturing the complete outcome of a supervised run
- **Why**: Immutability ensures audit integrity -- results cannot be modified after the fact
- **How**: Access fields like `result.success`, `result.budget_consumed`, `result.audit_trail`
- **When**: Always -- every `supervisor.run()` returns a `SupervisorResult`

### Key API

| Class / Method                 | Parameters                                                       | Returns              | Description                                 |
| ------------------------------ | ---------------------------------------------------------------- | -------------------- | ------------------------------------------- |
| `GovernedSupervisor()`         | `model`, `budget_usd`, `tools`, `data_clearance`, ...            | `GovernedSupervisor` | Create a governed supervisor                |
| `supervisor.run()`             | `task: str`, `execute_node: Callable = None`                     | `SupervisorResult`   | Execute a task under governance (async)     |
| `supervisor.envelope`          | --                                                               | `Envelope` (copy)    | Get the constraint envelope (deep copy)     |
| `supervisor.record_tool_use()` | `tool: str`, `params: dict`, `blocked: bool`, `reason: str`      | `None`               | Record a tool invocation in the audit trail |
| `supervisor.record_cost()`     | `amount: float`, `source: str`                                   | `None`               | Record a cost against the session budget    |
| `SupervisorResult()`           | `success`, `results`, `budget_consumed`, `budget_allocated`, ... | `SupervisorResult`   | Frozen execution outcome                    |
| `supervisor.audit`             | --                                                               | Read-only proxy      | Audit trail subsystem                       |
| `supervisor.budget`            | --                                                               | Read-only proxy      | Budget tracker subsystem                    |

## Code Walkthrough

### Imports and Setup

```python
from __future__ import annotations

import asyncio
import math
import os

from dotenv import load_dotenv

load_dotenv()

from kaizen_agents import GovernedSupervisor, SupervisorResult
from kaizen_agents.supervisor import HoldRecord
from kailash.trust import ConfidentialityLevel

model = os.environ.get("DEFAULT_LLM_MODEL", "claude-sonnet-4-20250514")
```

`GovernedSupervisor` and `SupervisorResult` are top-level exports from `kaizen_agents`. `ConfidentialityLevel` comes from the Kailash trust module and defines PACT data clearance levels.

### Layer 1: Minimal Instantiation

```python
supervisor = GovernedSupervisor(
    model=model,
    budget_usd=10.0,
)

assert isinstance(supervisor, GovernedSupervisor)
assert supervisor.model == model
assert supervisor.tools == [], "Default-deny: no tools allowed"
assert supervisor.clearance_level == ConfidentialityLevel.PUBLIC
```

Two parameters. Everything else gets PACT defaults: no tools (default-deny), PUBLIC clearance, no special thresholds. This is the entry point for beginners.

### Layer 2: Configured Supervisor

```python
configured = GovernedSupervisor(
    model=model,
    budget_usd=25.0,
    tools=["read_file", "grep", "write_report"],
    data_clearance="restricted",
    warning_threshold=0.70,
    timeout_seconds=600.0,
    max_children=15,
    max_depth=3,
    policy_source="jack.hong@example.com",
)

assert configured.tools == ["read_file", "grep", "write_report"]
assert configured.clearance_level == ConfidentialityLevel.RESTRICTED
assert len(configured.tools) == 3
```

Layer 2 adds operational configuration: which tools are allowed, what data the supervisor can access, when to warn about budget consumption, execution timeout, and organizational accountability via `policy_source`.

### Budget Validation

```python
try:
    GovernedSupervisor(model=model, budget_usd=-1.0)
    assert False, "Negative budget should raise ValueError"
except ValueError:
    pass

try:
    GovernedSupervisor(model=model, budget_usd=float("inf"))
    assert False, "Infinite budget should raise ValueError"
except ValueError:
    pass

try:
    GovernedSupervisor(model=model, budget_usd=float("nan"))
    assert False, "NaN budget should raise ValueError"
except ValueError:
    pass
```

PACT governance strictly validates financial constraints. Negative, infinite, and NaN budgets are all rejected with `ValueError`. This prevents accidental creation of unconstrained supervisors.

### Timeout Validation

```python
try:
    GovernedSupervisor(model=model, timeout_seconds=0)
    assert False, "Zero timeout should raise ValueError"
except ValueError:
    pass

try:
    GovernedSupervisor(model=model, timeout_seconds=-10)
    assert False, "Negative timeout should raise ValueError"
except ValueError:
    pass
```

Timeouts must be positive. Zero and negative values are rejected -- a supervisor with no timeout would run indefinitely.

### Data Clearance Levels

```python
clearance_mapping = {
    "public": ConfidentialityLevel.PUBLIC,
    "internal": ConfidentialityLevel.RESTRICTED,
    "restricted": ConfidentialityLevel.RESTRICTED,
    "confidential": ConfidentialityLevel.CONFIDENTIAL,
    "secret": ConfidentialityLevel.SECRET,
    "top_secret": ConfidentialityLevel.TOP_SECRET,
}

for name, expected_level in clearance_mapping.items():
    s = GovernedSupervisor(model=model, data_clearance=name)
    assert s.clearance_level == expected_level, f"{name} -> {expected_level}"

# Invalid clearance raises ValueError
try:
    GovernedSupervisor(model=model, data_clearance="invalid")
    assert False, "Invalid clearance should raise ValueError"
except ValueError:
    pass
```

The supervisor maps user-friendly string names to PACT `ConfidentialityLevel` enum values. Invalid clearance names are rejected. Note that `"internal"` and `"restricted"` both map to `RESTRICTED`.

### Constraint Envelope

```python
envelope = supervisor.envelope
assert envelope is not None
assert envelope.financial is not None
assert envelope.financial.max_spend_usd == 10.0

# Deep copy: mutations do not affect the supervisor
envelope_copy = supervisor.envelope
assert envelope_copy is not envelope, "Deep copy each time"
```

The envelope is a comprehensive snapshot of all constraints. Each access returns a fresh deep copy -- you can safely modify or serialize it without affecting the supervisor's internal state.

### Layer 3: Governance Subsystems

```python
# Audit trail
audit = supervisor.audit
assert audit is not None

# Budget tracker
budget = supervisor.budget
snapshot = budget.get_snapshot("root")
assert snapshot is not None, "Root budget is always allocated"

# Accountability tracker
accountability = supervisor.accountability
assert accountability is not None

# Cascade manager
cascade = supervisor.cascade
assert cascade is not None

# Clearance enforcer
clearance = supervisor.clearance
assert clearance is not None

# Dereliction detector
dereliction = supervisor.dereliction
assert dereliction is not None

# Bypass manager
bypass = supervisor.bypass_manager
assert bypass is not None

# Vacancy manager
vacancy = supervisor.vacancy
assert vacancy is not None
```

Seven governance subsystems, each exposed as a read-only proxy. Query methods work; mutation methods are blocked. This enables deep inspection without risk.

### SupervisorResult

```python
result = SupervisorResult(
    success=True,
    results={"task-0": "Analysis complete"},
    budget_consumed=2.50,
    budget_allocated=10.0,
)

assert result.success is True
assert result.results["task-0"] == "Analysis complete"
assert result.budget_consumed == 2.50
assert result.budget_allocated == 10.0
assert result.audit_trail == [], "Empty by default"
assert result.events == [], "Empty by default"
assert result.modifications == [], "Empty by default"

# SupervisorResult is frozen (immutable after construction)
try:
    result.success = False  # type: ignore[misc]
    assert False, "Frozen dataclass should not allow mutation"
except AttributeError:
    pass
```

`SupervisorResult` is a frozen dataclass -- after construction, no field can be modified. This guarantees audit integrity. The result includes per-task outputs, budget tracking, audit trail, events, and modifications.

### Dry Run with Default Executor

```python
async def _dry_run():
    s = GovernedSupervisor(model=model, budget_usd=5.0)
    result = await s.run("Analyze the codebase for security issues")
    assert isinstance(result, SupervisorResult)
    assert result.success is True, "Dry run always succeeds"
    assert result.budget_consumed == 0.0, "No-op executor has zero cost"
    assert len(result.audit_trail) > 0, "Audit trail recorded"
    assert result.plan is not None, "Plan was built"
    return result


dry_result = asyncio.run(_dry_run())
assert dry_result.success is True
```

When no `execute_node` callback is provided, the supervisor uses a default no-op executor. This is useful for plan validation -- you can see what the supervisor would do without actually executing anything. The audit trail is still recorded.

### Tool Use and Cost Recording

```python
supervisor.record_tool_use("read_file", {"path": "/src/main.py"})
supervisor.record_tool_use(
    "grep", {"pattern": "TODO"}, blocked=True, reason="not in allowed tools"
)

audit_entries = supervisor.audit.to_list()
assert len(audit_entries) > 0, "Tool use recorded in audit trail"

supervisor.record_cost(1.50, source="llm_tokens")
budget_snap = supervisor.budget.get_snapshot("root")
assert budget_snap.consumed >= 1.50

# Invalid costs are silently ignored (safety guard)
supervisor.record_cost(-1.0, source="invalid")
supervisor.record_cost(float("inf"), source="invalid")
supervisor.record_cost(float("nan"), source="invalid")
```

External code (CLI, entrypoints) can record tool invocations and costs in the supervisor's audit trail and budget tracker. Blocked tool uses are recorded with a reason. Invalid cost amounts (negative, infinite, NaN) are silently ignored as a safety guard.

## Common Mistakes

| Mistake                                           | Correct Pattern                                         | Why                                                                                           |
| ------------------------------------------------- | ------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Not setting `budget_usd` (or setting it too high) | Start with a conservative budget and increase as needed | Unconstrained budgets defeat the purpose of governance                                        |
| Forgetting that `tools=[]` is default-deny        | Explicitly list all tools the supervisor needs          | No tools are available by default -- the supervisor cannot call anything not in the list      |
| Mutating the envelope object                      | Treat `supervisor.envelope` as read-only                | Each access returns a deep copy, so mutations are lost and give a false sense of state change |
| Using `run()` synchronously                       | `asyncio.run(supervisor.run(task))`                     | `run()` is async -- call it with `await` or `asyncio.run()`                                   |

## Exercises

1. Create a `GovernedSupervisor` at Layer 1 (model + budget only). Verify that `tools` is empty and `clearance_level` is `PUBLIC`. Then create a Layer 2 supervisor with three tools and `"confidential"` clearance -- verify the tools list and clearance level.
2. Write a test that attempts to create supervisors with invalid budgets (-1, infinity, NaN) and invalid timeouts (0, -10). Verify that all six raise `ValueError`.
3. Run a dry run with `asyncio.run()` and inspect the returned `SupervisorResult`. What fields are present? Is the audit trail empty or populated?

## Key Takeaways

- `GovernedSupervisor` orchestrates multi-agent execution under PACT governance
- Three progressive layers: minimal (model + budget), configured (tools + clearance), advanced (subsystems)
- PACT default-deny: empty tools, PUBLIC clearance, minimal budget -- everything must be explicitly granted
- Budget, timeout, and clearance are strictly validated -- no negative, infinite, or invalid values
- `SupervisorResult` is frozen (immutable) for audit integrity
- Seven governance subsystems are exposed as read-only proxies for deep inspection

## Next Chapter

[Chapter 8: Pipeline Composition](08_pipeline.md) -- Compose agents into multi-agent pipelines using sequential, router, parallel, and ensemble patterns.
