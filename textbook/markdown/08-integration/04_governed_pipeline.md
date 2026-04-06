# Chapter 4: PACT-Governed Pipeline

## Overview

In production, AI agents must operate within governance boundaries. **PACT** (the governance framework) wraps Kaizen agents with clearance checks, budget limits, and action restrictions. The pattern is: compile an organization structure, create a frozen GovernanceContext, and hand it to a GovernedSupervisor that enforces constraints on every agent action. This chapter teaches the governance + agent integration pattern and how all eight Kailash packages combine in a governed ML platform.

## Prerequisites

- [Chapter 3: Agent with ML Tools](03_agent_with_tools.md)
- Familiarity with PACT concepts from the PACT section (D/T/R addressing, clearance, envelopes)

## Concepts

### Concept 1: The Governance + Agent Pattern

The production pattern chains three components:

```
GovernanceEngine.compile_org(org_yaml)
    -> GovernanceContext (frozen, immutable)
        -> PactGovernedAgent(agent, context)
            -> Agent can only act within its envelope
```

- **What**: A three-layer pattern that wraps agent autonomy with organizational constraints
- **Why**: Without governance, agents can access data above their clearance, exceed budgets, and modify their own constraints
- **How**: Compile an org, create a frozen context, pass it to the governed agent or supervisor
- **When**: Every production agent deployment

### Concept 2: GovernanceContext Is Frozen

Agents receive a `GovernanceContext`, **not** a `GovernanceEngine`. The context is a frozen snapshot of the agent's envelope, clearance, and allowed actions. Because it is frozen, agents cannot escalate their own privileges, widen their envelope, or modify organizational structure.

- **What**: An immutable snapshot of governance constraints for a specific role
- **Why**: If agents could access the engine, they could call `grant_clearance()` or `set_role_envelope()` on themselves
- **How**: `engine.get_context(role_address="D1-R1-T1-R1-R1", posture=TrustPosture.SUPERVISED)`
- **When**: Before handing any governance information to an agent

### Concept 3: GovernedSupervisor

GovernedSupervisor wraps a multi-agent pipeline with governance. Before each agent action, it checks clearance. It tracks cost against budget. It enforces monotonic tightening on child agent envelopes (children can never have more authority than their parent). And it creates an audit trail of all decisions.

### Concept 4: D/T/R Addressing

PACT uses Department/Team/Role addressing to locate entities in the organizational hierarchy. Addresses like `D1-R1-T1-R1` identify specific positions. The `Address` class parses and validates these strings.

### Key API

| Class / Method         | Parameters                                  | Returns              | Description                              |
| ---------------------- | ------------------------------------------- | -------------------- | ---------------------------------------- |
| `Address.parse()`      | `address_str: str`                          | `Address`            | Parse a D/T/R address string             |
| `GovernanceEngine()`   | `org_def` or `compiled_org`                 | `GovernanceEngine`   | Create the governance decision engine    |
| `engine.get_context()` | `role_address`, `posture`                   | `GovernanceContext`  | Create a frozen context for an agent     |
| `GovernedSupervisor()` | `model`, `budget_usd`, `governance_context` | `GovernedSupervisor` | Create a governed multi-agent supervisor |

## Code Walkthrough

```python
from __future__ import annotations

from kailash.trust.pact import Address

addr = Address.parse("D1-R1-T1-R1")
assert addr.depth() == 4
```

The address `D1-R1-T1-R1` has depth 4: department 1, role 1, team 1, role 1. This identifies a specific role within a specific team within a specific department.

### Organization Structure

```yaml
# PACT uses D/T/R (Department/Team/Role) addressing:
#
# org:
#   departments:
#     - name: "Data Science"
#       teams:
#         - name: "ML Engineering"
#           roles:
#             - name: "Senior ML Engineer"
#               clearance: SECRET
#               envelope:
#                 max_cost_usd: 50.0
#                 allowed_tools: [DataExplorer, TrainingPipeline, ModelRegistry]
#
# Address: D1-T1-R1 (first dept, first team, first role)
```

Each role has a clearance level and a constraint envelope that limits what it can do and spend.

### Frozen GovernanceContext

```python
# GovernanceContext is frozen at creation -- agents cannot modify it.
# This is the anti-self-modification defense.
#
# from kailash.trust.pact import GovernanceContext, GovernanceEngine
#
# engine = GovernanceEngine()
# compiled = engine.compile_org(org_definition)
# context = GovernanceContext(compiled_org=compiled, agent_address=addr)
#
# context.envelope is immutable
# context.clearance is immutable
# Attempting to modify raises GovernanceBlockedError
```

The frozen context prevents:

- Agents accessing data above their clearance level
- Agents exceeding their budget or time constraints
- Agents modifying their own governance context

### GovernedSupervisor Pattern

```python
# GovernedSupervisor wraps a multi-agent pipeline with governance:
#
# from kaizen_agents import GovernedSupervisor
#
# supervisor = GovernedSupervisor(
#     model=os.environ["DEFAULT_LLM_MODEL"],
#     budget_usd=10.0,
#     governance_context=context,
# )
# result = await supervisor.run("Analyze and retrain the credit model")
```

The supervisor performs four governance functions at runtime:

1. **Clearance check** before each agent action
2. **Cost tracking** against the budget
3. **Monotonic tightening** on child agent envelopes (children never get more authority)
4. **Audit trail** of all decisions

### The Full Platform: All Eight Packages

```python
# All 8 packages working together:
#
#   kailash (core)      -> Workflow orchestration
#   kailash-ml          -> ML engines (training, inference, drift)
#   kailash-dataflow    -> Database persistence
#   kailash-nexus       -> Multi-channel deployment
#   kailash-kaizen      -> Agent framework (signatures, LLM)
#   kaizen-agents       -> Specialized agents (ReAct, RAG, etc.)
#   kailash-pact        -> Governance (D/T/R, clearance, envelopes)
#   kailash-align       -> LLM fine-tuning
```

This is the capstone integration: a governed ML platform where agents train models, serve predictions, and operate within organizational constraints -- all through Kailash frameworks.

## Exercises

1. Create an Address for "Department 2, Team 3, Role 1" and verify its depth. How would you use this address to create a GovernanceContext?
2. Why do agents receive GovernanceContext instead of GovernanceEngine? List three things an agent could do if it had engine access that the frozen context prevents.
3. Design a governed ML pipeline where: (a) a data analyst agent profiles data, (b) a training agent builds a model, and (c) a deployment agent pushes to production. Each agent should have different clearance levels and budget limits. Describe the org structure and envelopes.

## Key Takeaways

- PACT governance wraps agents with clearance, budget, and action constraints
- Agents receive frozen GovernanceContext, never GovernanceEngine (anti-self-modification)
- GovernedSupervisor enforces governance on every agent action with audit trails
- D/T/R addressing identifies roles in the organizational hierarchy
- Monotonic tightening ensures child agents never exceed parent authority
- All eight Kailash packages combine into a governed ML platform

## Next Chapter

[Chapter 5: Full Platform](05_full_platform.md) -- Import and validate all eight Kailash packages working together.
