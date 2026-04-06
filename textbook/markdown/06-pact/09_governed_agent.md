# Chapter 9: Governed Agent & GovernanceContext

## Overview

**GovernanceContext** is the frozen, read-only snapshot that agents receive instead of the mutable GovernanceEngine. This is the anti-self-modification defense: agents can check their own permissions but cannot escalate them. This chapter teaches you how to obtain a GovernanceContext, inspect its contents, and implement the governance-check-before-action pattern that production agents use.

## Prerequisites

- [Chapter 8: GovernanceEngine](08_governance_engine.md)
- [Chapter 3: Clearance Levels](03_clearance.md)
- [Chapter 4: Constraint Envelopes](04_envelopes.md)

## Concepts

### Concept 1: GovernanceContext — The Frozen Snapshot

GovernanceContext is an immutable (frozen) dataclass that captures a role's governance state at a point in time. It contains the role address, trust posture, effective envelope, clearance, allowed actions, and compartments. Because it is frozen, agents cannot modify their own permissions.

- **What**: A frozen snapshot of a role's governance permissions at context-creation time
- **Why**: Giving agents direct access to GovernanceEngine would allow self-modification — escalating clearance, widening envelopes, or changing org structure
- **How**: `ctx = engine.get_context(role_address="...", posture=TrustPosture.SUPERVISED)`
- **When**: Every agent receives a GovernanceContext at initialization; it is the only governance object agents should hold

### Concept 2: Trust Posture Capping

The effective clearance level is the minimum of the role's granted clearance and the posture ceiling. Higher trust postures (DELEGATED) have higher ceilings; lower postures (PSEUDO_AGENT) cap clearance to PUBLIC regardless of what was granted.

- **What**: A mechanism that limits effective clearance based on the operational trust posture
- **Why**: Even a highly-cleared agent running in a low-trust mode should not access sensitive data
- **How**: `effective = min(role_clearance.max_clearance, posture_ceiling[posture])`
- **When**: Automatically applied when `engine.get_context()` creates the snapshot

### Concept 3: The Check-Before-Action Pattern

Agents follow a two-step pattern: (1) quick check against the context's `allowed_actions` set (no engine call), then (2) full verification through `engine.verify_action()` for the audit trail. If either step blocks, the agent raises `GovernanceBlockedError`.

- **What**: A defense-in-depth pattern where agents verify governance before every action
- **Why**: Quick local checks are fast but may miss dynamic state; full engine verification is authoritative but slower — combining both gives speed and correctness
- **How**: Check `action in ctx.allowed_actions` first, then call `engine.verify_action()`
- **When**: Before every action that modifies state, accesses data, or spends budget

### Concept 4: Security Invariants

GovernanceContext enforces three security invariants: (1) frozen — no attribute mutation after creation, (2) unpicklable — cannot be serialized and forged, (3) no engine reference — the context has no access to mutable governance state.

- **What**: Invariants that prevent privilege escalation, context forging, and self-modification
- **Why**: Without these, a compromised agent could pickle a context, modify it, unpickle it, and gain elevated permissions
- **How**: `@dataclass(frozen=True)`, custom `__reduce__` that raises TypeError, no engine attribute
- **When**: Always enforced — these are not configurable

### Key API

| Method / Attribute | Type | Description |
|---|---|---|
| `engine.get_context()` | `role_address, posture` → `GovernanceContext` | Create a frozen governance snapshot |
| `ctx.role_address` | `str` | The D/T/R address of the governed role |
| `ctx.posture` | `TrustPosture` | Trust posture level |
| `ctx.effective_envelope` | `ConstraintEnvelopeConfig` | Effective constraint envelope |
| `ctx.allowed_actions` | `frozenset[str]` | Permitted actions |
| `ctx.clearance` | `RoleClearance | None` | Clearance grant |
| `ctx.effective_clearance_level` | `ConfidentialityLevel | None` | Posture-capped clearance |
| `ctx.compartments` | `frozenset[str]` | Accessible compartments |
| `ctx.to_dict()` | → `dict` | Safe serialization for logging |
| `GovernanceContext.from_dict()` | `dict` → `GovernanceContext` | Reconstruct (emits warning) |
| `GovernanceBlockedError` | Exception | Raised on blocked actions |

## Code Walkthrough

### Setting up the organization and engine

```python
from kailash.trust.pact import (
    GovernanceBlockedError, GovernanceContext, GovernanceEngine,
    GovernanceVerdict, RoleClearance, RoleDefinition, RoleEnvelope, VettingStatus,
)
from kailash.trust.pact.config import (
    ConstraintEnvelopeConfig, DepartmentConfig, FinancialConstraintConfig,
    OperationalConstraintConfig, OrgDefinition, TeamConfig,
)
from kailash.trust import ConfidentialityLevel, TrustPosture

org_def = OrgDefinition(
    org_id="agent-demo",
    name="Agent Demo Corp",
    departments=[DepartmentConfig(department_id="ops", name="Operations")],
    teams=[TeamConfig(id="deploy", name="Deployment Team", workspace="ws")],
    roles=[
        RoleDefinition(role_id="vp-ops", name="VP Operations", is_primary_for_unit="ops"),
        RoleDefinition(role_id="deploy-lead", name="Deployment Lead",
                       reports_to_role_id="vp-ops", is_primary_for_unit="deploy"),
        RoleDefinition(role_id="deploy-agent", name="Deploy Agent",
                       reports_to_role_id="deploy-lead", agent_id="agent-deploy-01"),
    ],
)
engine = GovernanceEngine(org_def)
```

We define a simple three-role hierarchy: VP Operations → Deployment Lead → Deploy Agent. The agent role has an `agent_id`, marking it as an AI agent.

### Granting clearance and setting envelopes

```python
engine.grant_clearance("D1-R1-T1-R1-R1", RoleClearance(
    role_address="D1-R1-T1-R1-R1",
    max_clearance=ConfidentialityLevel.RESTRICTED,
    compartments=frozenset({"deployment-configs"}),
    vetting_status=VettingStatus.ACTIVE,
))

engine.set_role_envelope(RoleEnvelope(
    id="re-deploy-agent",
    defining_role_address="D1-R1-T1-R1",
    target_role_address="D1-R1-T1-R1-R1",
    envelope=ConstraintEnvelopeConfig(
        id="env-deploy-agent",
        confidentiality_clearance=ConfidentialityLevel.RESTRICTED,
        financial=FinancialConstraintConfig(max_spend_usd=100.0),
        operational=OperationalConstraintConfig(
            allowed_actions=["read", "deploy", "rollback"],
            blocked_actions=["delete_production", "approve"],
        ),
    ),
))
```

The deploy lead defines the agent's envelope: read, deploy, rollback are allowed; delete_production and approve are blocked.

### Getting a GovernanceContext

```python
ctx = engine.get_context(role_address="D1-R1-T1-R1-R1", posture=TrustPosture.SUPERVISED)
assert isinstance(ctx, GovernanceContext)
assert ctx.role_address == "D1-R1-T1-R1-R1"
```

### Verifying immutability

```python
try:
    ctx.posture = TrustPosture.DELEGATED
except AttributeError:
    pass  # Expected: frozen=True prevents mutation
```

### Checking allowed actions

```python
assert "deploy" in ctx.allowed_actions
assert "delete_production" not in ctx.allowed_actions
```

### The check-before-action pattern

```python
def agent_perform_action(context, engine_ref, action):
    if action not in context.allowed_actions:
        raise GovernanceBlockedError(GovernanceVerdict(
            level="blocked", reason=f"Action '{action}' not in allowed_actions",
            role_address=context.role_address, action=action,
        ))
    verdict = engine_ref.verify_action(context.role_address, action)
    if verdict.is_blocked:
        raise GovernanceBlockedError(verdict)
    return f"Action '{action}' executed successfully"

result = agent_perform_action(ctx, engine, "deploy")
assert "successfully" in result
```

### Posture-capped clearance

```python
# SUPERVISED ceiling = RESTRICTED → effective = RESTRICTED
assert ctx.effective_clearance_level == ConfidentialityLevel.RESTRICTED

# PSEUDO_AGENT ceiling = PUBLIC → effective = PUBLIC
pseudo_ctx = engine.get_context(role_address="D1-R1-T1-R1-R1", posture=TrustPosture.PSEUDO_AGENT)
assert pseudo_ctx.effective_clearance_level == ConfidentialityLevel.PUBLIC
```

### Security: unpicklable context

```python
import pickle
try:
    pickle.dumps(ctx)
except TypeError as e:
    assert "cannot be pickled" in str(e).lower()
```

## Exercises

1. **Add a second agent role** — Define a "monitor-agent" under the deploy team with read-only actions. Create its context and verify it cannot deploy.

2. **Budget checking** — Extend `agent_perform_action` to check `ctx.effective_envelope.financial.max_spend_usd` before allowing actions that cost money. Simulate a $150 action and verify it raises GovernanceBlockedError.

3. **Posture comparison** — Get contexts for the same role at all three posture levels (DELEGATED, SUPERVISED, PSEUDO_AGENT). Create a table showing effective clearance level and allowed compartments at each posture.
