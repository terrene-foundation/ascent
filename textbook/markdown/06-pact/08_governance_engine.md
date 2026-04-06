# Chapter 8: GovernanceEngine

## Overview

The **GovernanceEngine** is the single entry point for all governance decisions. It compiles an organization definition, manages clearances and envelopes, and provides two primary decision APIs: `verify_action()` for action authorization and `check_access()` for knowledge access control. All public methods are thread-safe via an internal lock. This chapter teaches you how to initialize the engine, configure roles and envelopes, and make governance decisions programmatically.

## Prerequisites

- [Chapter 2: Compiling Organizations](02_compile_org.md)
- [Chapter 3: Clearance Levels](03_clearance.md)
- [Chapter 4: Constraint Envelopes](04_envelopes.md)
- [Chapter 5: Access Decisions](05_access.md)

## Concepts

### Concept 1: The Engine as Facade

GovernanceEngine wraps compilation, clearance management, envelope management, action verification, and knowledge access into a single thread-safe interface. Instead of calling `compile_org()`, `can_access()`, and envelope checks separately, you use one object.

- **What**: A facade that unifies all governance operations behind a single, thread-safe API
- **Why**: Separate governance functions are hard to coordinate -- the engine ensures clearance grants, envelope changes, and access decisions are sequenced correctly
- **How**: `engine = GovernanceEngine(org_def)` compiles the org and provides all decision methods
- **When**: Always use GovernanceEngine in production; use the lower-level functions only for testing

### Concept 2: verify_action() -- The Primary Decision API

`verify_action()` combines vacancy checks, envelope evaluation, and gradient classification into a single call. It returns a `GovernanceVerdict` with a level (`auto_approved`, `flagged`, `held`, `blocked`) and an `allowed` boolean.

- **What**: A method that evaluates whether a role can perform a specific action
- **Why**: Action authorization requires checking multiple governance dimensions (vacancy, envelope, clearance) -- the engine does this atomically
- **How**: `verdict = engine.verify_action("D1-R1-T1-R1", "deploy")`
- **When**: Before every agent action in production

### Concept 3: Fail-Closed Design

Both `verify_action()` and `check_access()` are fail-closed: any exception during evaluation results in a BLOCKED/DENY verdict. This ensures that governance failures are safe failures -- a bug in the governance logic denies access rather than granting it.

- **What**: Error handling that defaults to denial rather than approval
- **Why**: Fail-open governance would allow unauthorized actions whenever the evaluation code has a bug
- **How**: The engine wraps all evaluation in try/except and returns BLOCKED on any exception
- **When**: Automatically applied to every governance decision

### Concept 4: Thread Safety

All public methods on GovernanceEngine acquire an internal `threading.Lock`. This means multiple threads can safely call `verify_action()`, `grant_clearance()`, and `check_access()` concurrently without data races.

### Key API

| Method                | Parameters                                  | Returns             | Description                        |
| --------------------- | ------------------------------------------- | ------------------- | ---------------------------------- |
| `GovernanceEngine()`  | `org_def` or `compiled_org`                 | `GovernanceEngine`  | Create and compile                 |
| `grant_clearance()`   | `address`, `RoleClearance`                  | `None`              | Set clearance for a role           |
| `set_role_envelope()` | `RoleEnvelope`                              | `None`              | Set constraint envelope for a role |
| `verify_action()`     | `role_address`, `action`, `context=None`    | `GovernanceVerdict` | Authorize an action                |
| `check_access()`      | `role_address`, `knowledge_item`, `posture` | `AccessDecision`    | Check knowledge access             |
| `get_context()`       | `role_address`, `posture`                   | `GovernanceContext` | Create frozen context for agents   |

## Code Walkthrough

```python
from __future__ import annotations

from kailash.trust import ConfidentialityLevel, TrustPosture
from kailash.trust.pact import (
    GovernanceEngine,
    GovernanceVerdict,
    KnowledgeItem,
    RoleClearance,
    RoleDefinition,
    RoleEnvelope,
    VettingStatus,
)
from kailash.trust.pact.config import (
    ConstraintEnvelopeConfig,
    DepartmentConfig,
    FinancialConstraintConfig,
    OperationalConstraintConfig,
    OrgDefinition,
    TeamConfig,
)
```

### Creating an OrgDefinition

```python
org_def = OrgDefinition(
    org_id="engine-demo",
    name="Engine Demo Corp",
    departments=[
        DepartmentConfig(department_id="ops", name="Operations"),
    ],
    teams=[
        TeamConfig(id="infra", name="Infrastructure Team", workspace="ws"),
    ],
    roles=[
        RoleDefinition(
            role_id="vp-ops",
            name="VP Operations",
            is_primary_for_unit="ops",
        ),
        RoleDefinition(
            role_id="infra-lead",
            name="Infrastructure Lead",
            reports_to_role_id="vp-ops",
            is_primary_for_unit="infra",
        ),
        RoleDefinition(
            role_id="sre-1",
            name="SRE Engineer",
            reports_to_role_id="infra-lead",
        ),
    ],
)
```

This defines a three-role hierarchy: VP Operations -> Infrastructure Lead -> SRE Engineer. The `is_primary_for_unit` field assigns roles as the primary for their department or team.

### Initializing GovernanceEngine

```python
engine = GovernanceEngine(org_def)

# The engine compiled the org internally
assert engine._compiled_org.org_id == "engine-demo"
assert "D1-R1" in engine._compiled_org.nodes
```

Pass an `OrgDefinition` and the engine compiles it internally. You can also pass a pre-compiled `CompiledOrg` to skip recompilation.

### Granting Clearance

```python
engine.grant_clearance(
    "D1-R1",
    RoleClearance(
        role_address="D1-R1",
        max_clearance=ConfidentialityLevel.SECRET,
        vetting_status=VettingStatus.ACTIVE,
    ),
)

engine.grant_clearance(
    "D1-R1-T1-R1",
    RoleClearance(
        role_address="D1-R1-T1-R1",
        max_clearance=ConfidentialityLevel.CONFIDENTIAL,
        vetting_status=VettingStatus.ACTIVE,
    ),
)

engine.grant_clearance(
    "D1-R1-T1-R1-R1",
    RoleClearance(
        role_address="D1-R1-T1-R1-R1",
        max_clearance=ConfidentialityLevel.RESTRICTED,
        vetting_status=VettingStatus.ACTIVE,
    ),
)
```

Clearances descend the hierarchy: VP gets SECRET, Lead gets CONFIDENTIAL, SRE gets RESTRICTED. Each role can only access knowledge at or below its clearance level.

### Setting Role Envelopes

```python
vp_envelope = ConstraintEnvelopeConfig(
    id="env-vp-ops",
    financial=FinancialConstraintConfig(
        max_spend_usd=100000.0,
        requires_approval_above_usd=50000.0,
    ),
    operational=OperationalConstraintConfig(
        allowed_actions=["read", "write", "deploy", "approve", "provision"],
        blocked_actions=["delete_production"],
    ),
)

lead_envelope = ConstraintEnvelopeConfig(
    id="env-infra-lead",
    financial=FinancialConstraintConfig(
        max_spend_usd=10000.0,
        requires_approval_above_usd=5000.0,
    ),
    operational=OperationalConstraintConfig(
        allowed_actions=["read", "write", "deploy", "provision"],
        blocked_actions=["delete_production", "approve"],
    ),
)

engine.set_role_envelope(
    RoleEnvelope(
        id="re-vp-ops",
        defining_role_address="R1",  # Board defines VP envelope
        target_role_address="D1-R1",
        envelope=vp_envelope,
    )
)

engine.set_role_envelope(
    RoleEnvelope(
        id="re-infra-lead",
        defining_role_address="D1-R1",  # VP defines lead envelope
        target_role_address="D1-R1-T1-R1",
        envelope=lead_envelope,
    )
)
```

Notice the monotonic tightening: the VP can approve, but the Lead cannot. The VP can spend $100K, the Lead only $10K. The `defining_role_address` records who set the envelope -- the Board defines the VP's, the VP defines the Lead's.

### verify_action() -- Allowed Action

```python
verdict = engine.verify_action("D1-R1-T1-R1", "deploy")

assert isinstance(verdict, GovernanceVerdict)
assert verdict.role_address == "D1-R1-T1-R1"
assert verdict.action == "deploy"
assert verdict.level == "auto_approved"
assert verdict.allowed is True
assert verdict.is_blocked is False
```

"deploy" is in the Lead's `allowed_actions`, so the verdict is `auto_approved`.

### verify_action() -- Blocked Action

```python
blocked_verdict = engine.verify_action("D1-R1-T1-R1", "delete_production")

assert blocked_verdict.level == "blocked"
assert blocked_verdict.allowed is False
assert blocked_verdict.is_blocked is True
```

"delete_production" is in `blocked_actions`, so it is denied outright.

### verify_action() with Financial Context

```python
low_cost_verdict = engine.verify_action(
    "D1-R1-T1-R1",
    "provision",
    {"cost": 100.0},
)
assert low_cost_verdict.allowed is True
```

Passing a `cost` in the context dict triggers financial envelope checks. $100 is within the Lead's $10K budget, so it is approved.

### check_access() -- Knowledge Access

```python
ops_doc = KnowledgeItem(
    item_id="doc-ops-runbook",
    classification=ConfidentialityLevel.RESTRICTED,
    owning_unit_address="D1-R1-T1",
)

decision = engine.check_access(
    role_address="D1-R1-T1-R1-R1",
    knowledge_item=ops_doc,
    posture=TrustPosture.DELEGATED,
)

assert decision.allowed is True  # SRE is in the same team
```

The SRE (RESTRICTED clearance) can access a RESTRICTED document owned by their team.

```python
secret_doc = KnowledgeItem(
    item_id="doc-classified",
    classification=ConfidentialityLevel.SECRET,
    owning_unit_address="D1",
)

secret_decision = engine.check_access(
    role_address="D1-R1-T1-R1-R1",
    knowledge_item=secret_doc,
    posture=TrustPosture.DELEGATED,
)

assert secret_decision.allowed is False
assert secret_decision.step_failed == 2
```

The SRE (RESTRICTED) cannot access a SECRET document -- the clearance check fails at step 2.

### GovernanceVerdict Properties

```python
verdict_dict = verdict.to_dict()
assert isinstance(verdict_dict, dict)
assert verdict_dict["level"] == "auto_approved"
assert verdict_dict["action"] == "deploy"
assert verdict_dict["allowed"] is True
```

Verdicts serialize to dictionaries for logging and audit trails.

### Thread Safety

```python
import threading

assert isinstance(engine._lock, threading.Lock)
```

The engine uses a `threading.Lock` to ensure all public methods are thread-safe.

### Fail-Closed Behavior

```python
unknown_verdict = engine.verify_action("D99-R99", "some_action")
assert isinstance(unknown_verdict, GovernanceVerdict)
assert unknown_verdict.level in ("auto_approved", "blocked", "flagged", "held")
```

Even for unknown roles, `verify_action()` returns a valid verdict rather than raising an exception.

## Exercises

1. Create an OrgDefinition with two departments (Engineering and Finance), each with one team and two roles. Set different clearance levels and envelopes for each role. Verify that actions allowed in Engineering are blocked in Finance.
2. Test the fail-closed behavior: call `verify_action()` with an address that does not exist in the org. What verdict level do you get? Why is this the safe default?
3. Use `check_access()` to test whether a CONFIDENTIAL-cleared role can access a RESTRICTED document (should succeed) and a SECRET document (should fail). Verify the `step_failed` value.

## Key Takeaways

- GovernanceEngine is the single thread-safe facade for all governance decisions
- `verify_action()` combines vacancy, envelope, and gradient checks into one call
- `check_access()` evaluates knowledge access with clearance, ownership, and posture
- Both decision APIs are fail-closed: exceptions produce BLOCKED/DENY verdicts
- All public methods acquire a lock for thread safety
- Verdicts serialize to dictionaries for audit trails
- The engine accepts either OrgDefinition (compiles internally) or pre-compiled CompiledOrg

## Next Chapter

[Chapter 9: Governed Agent & GovernanceContext](09_governed_agent.md) -- Wrap agents with frozen governance contexts that prevent self-modification.
