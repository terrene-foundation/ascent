# Chapter 15: ML Agents and Guardrails

## Overview

Kailash ML includes six specialized AI agents that augment ML engines with LLM-powered reasoning: data scientist, feature engineer, model selector, experiment interpreter, drift analyst, and retraining decision agent. These agents follow a **double opt-in** pattern (install the extra + enable at the call site) and are governed by five mandatory guardrails: confidence thresholds, cost budgets, human approval gates, baseline comparison, and audit trails. This chapter covers agent discovery, the infusion protocol, guardrail configuration, and each guardrail mechanism.

## Prerequisites

- Python 3.10+ installed
- Kailash ML installed (`pip install kailash-ml`)
- Optional: `pip install kailash-ml[agents]` for live agent execution (requires Kaizen)
- Completed [Chapter 14: OnnxBridge](14_onnx_bridge.md)
- Understanding of Kaizen Delegate from [Kaizen Chapter 2](../03-kaizen/02_delegate.md)

## Concepts

### Concept 1: The Six ML Agents

Kailash ML provides six agents, each specialized for a phase of the ML lifecycle: `DataScientistAgent` (data analysis), `FeatureEngineerAgent` (feature suggestions), `ModelSelectorAgent` (algorithm selection), `ExperimentInterpreterAgent` (results interpretation), `DriftAnalystAgent` (drift analysis), and `RetrainingDecisionAgent` (retrain/no-retrain decisions). All are lazy-loaded via `kailash_ml.agents`.

- **What**: Six LLM-powered agents specialized for ML lifecycle decisions
- **Why**: ML decisions (which model? which features? retrain now?) benefit from LLM reasoning combined with quantitative analysis
- **How**: `from kailash_ml.agents import ModelSelectorAgent` -- lazy-loaded, requires `kailash-ml[agents]`
- **When**: When you want AI-assisted recommendations at any ML lifecycle decision point

### Concept 2: AgentInfusionProtocol (Double Opt-In)

Agent features are not active by default. The double opt-in pattern requires: (1) installing the agents extra (`pip install kailash-ml[agents]`), and (2) passing `agent=True` at the call site. Without both, engines use pure algorithmic paths. This prevents accidental LLM costs and ensures agents are a deliberate choice.

- **What**: A two-step activation pattern for agent-augmented ML operations
- **Why**: Prevents accidental LLM API calls and costs; keeps the default path deterministic and free
- **How**: `AgentInfusionProtocol` is a runtime-checkable Protocol with methods like `suggest_model()`, `suggest_features()`, `interpret_results()`, and `interpret_drift()`
- **When**: Opt in when you want LLM reasoning to augment engine decisions; leave off for pure algorithmic mode

### Concept 3: The Five Mandatory Guardrails

Every ML agent operation must pass through five guardrails, implemented via `AgentGuardrailMixin`:

1. **Confidence threshold** -- Agent recommendations below `min_confidence` are rejected (engine falls back to algorithmic mode)
2. **Cost budget** -- Cumulative LLM cost is tracked; exceeding `max_llm_cost_usd` raises `GuardrailBudgetExceededError`
3. **Human approval gate** -- Non-auto-approved recommendations create `ApprovalRequest` objects that require explicit human approval or rejection
4. **Baseline comparison** -- When `require_baseline=True`, the engine runs a pure algorithmic baseline alongside the agent for comparison
5. **Audit trail** -- Every agent operation is logged as an `AuditEntry` with agent name, engine name, input/output summaries, confidence, cost, and approval status

- **What**: Five mandatory checks that govern every agent recommendation
- **Why**: LLM agents are non-deterministic; guardrails ensure safety, accountability, and cost control
- **How**: `AgentGuardrailMixin` provides `_check_confidence()`, `_record_cost()`, `_request_approval()`, and `_log_audit()`
- **When**: Automatically on every agent-augmented operation; configured via `GuardrailConfig`

### Concept 4: GuardrailConfig

`GuardrailConfig` centralizes guardrail settings: `max_llm_cost_usd` (default 1.0), `auto_approve` (default False), `require_baseline` (default True), `audit_trail` (default True), and `min_confidence` (default 0.5). All numeric values are validated against NaN, Inf, and negative numbers.

- **What**: A configuration object for all five guardrails
- **Why**: Centralizes safety parameters in one place, making them reviewable and auditable
- **How**: Pass to any engine that uses `AgentGuardrailMixin`: `MockEngine(GuardrailConfig(max_llm_cost_usd=0.05))`
- **When**: At engine initialization, before any agent operations

## Key API

| Class / Method           | Parameters                                                                              | Returns           | Description                       |
| ------------------------ | --------------------------------------------------------------------------------------- | ----------------- | --------------------------------- |
| `GuardrailConfig()`      | `max_llm_cost_usd`, `auto_approve`, `require_baseline`, `audit_trail`, `min_confidence` | `GuardrailConfig` | Configure all five guardrails     |
| `CostTracker()`          | `max_budget_usd: float`                                                                 | `CostTracker`     | Track cumulative LLM costs        |
| `tracker.record()`       | `model`, `input_tokens`, `output_tokens`                                                | `float`           | Record a call and return its cost |
| `_check_confidence()`    | `confidence: float`, `agent_name: str`                                                  | `bool`            | Guardrail 1: confidence gate      |
| `_record_cost()`         | `model`, `input_tokens`, `output_tokens`                                                | `None`            | Guardrail 2: cost tracking        |
| `_request_approval()`    | `agent_name`, `recommendation_summary`, `confidence`, ...                               | `ApprovalRequest` | Guardrail 3: human approval gate  |
| `approve()` / `reject()` | `request_id`, `approved_by`, `reason=None`                                              | `ApprovalResult`  | Resolve an approval request       |
| `_log_audit()`           | `agent_name`, `engine_name`, `input_summary`, `output_summary`, ...                     | `AuditEntry`      | Guardrail 5: audit trail          |

## Code Walkthrough

```python
from __future__ import annotations

from kailash_ml.engines._guardrails import (
    AgentGuardrailMixin,
    ApprovalRequest,
    ApprovalResult,
    AuditEntry,
    CostTracker,
    GuardrailBudgetExceededError,
    GuardrailConfig,
)
from kailash_ml.types import AgentInfusionProtocol

# ── 1. The 6 ML agents are lazy-loaded via __getattr__ ─────────────

from kailash_ml import agents

expected_agents = {
    "DataScientistAgent",
    "FeatureEngineerAgent",
    "ModelSelectorAgent",
    "ExperimentInterpreterAgent",
    "DriftAnalystAgent",
    "RetrainingDecisionAgent",
}

assert set(agents.__all__) == expected_agents

# Non-existent agent raises AttributeError
try:
    agents.__getattr__("NonExistentAgent")
except AttributeError:
    pass  # Expected

# ── 2. AgentInfusionProtocol (double opt-in) ───────────────────────

assert hasattr(AgentInfusionProtocol, "suggest_model")
assert hasattr(AgentInfusionProtocol, "suggest_features")
assert hasattr(AgentInfusionProtocol, "interpret_results")
assert hasattr(AgentInfusionProtocol, "interpret_drift")

# ── 3. GuardrailConfig — defaults ──────────────────────────────────

config = GuardrailConfig()
assert config.max_llm_cost_usd == 1.0
assert config.auto_approve is False
assert config.require_baseline is True
assert config.audit_trail is True
assert config.min_confidence == 0.5

# ── 4. GuardrailConfig — NaN/Inf rejection ─────────────────────────

try:
    GuardrailConfig(max_llm_cost_usd=float("nan"))
except ValueError:
    pass  # Expected

try:
    GuardrailConfig(max_llm_cost_usd=-5.0)
except ValueError:
    pass  # Expected

# ── 5. CostTracker — tracks cumulative LLM costs ───────────────────

tracker = CostTracker(max_budget_usd=0.10)
assert tracker.total_spent == 0.0
assert tracker.remaining == 0.10

cost = tracker.record("test-model", input_tokens=100, output_tokens=50)
assert cost > 0
assert tracker.total_spent == cost
assert len(tracker.calls) == 1

# Exceed budget
try:
    tracker.record("test-model", input_tokens=100000, output_tokens=100000)
except GuardrailBudgetExceededError:
    pass  # Expected

tracker.reset()
assert tracker.total_spent == 0.0

# ── 6. AgentGuardrailMixin — the 5 guardrails ──────────────────────

class MockEngine(AgentGuardrailMixin):
    def __init__(self, config=None):
        self._init_guardrails(config)

# Guardrail 1: Confidence
engine = MockEngine(GuardrailConfig(min_confidence=0.6))
assert engine._check_confidence(0.8, "TestAgent") is True
assert engine._check_confidence(0.3, "TestAgent") is False
assert engine._check_confidence(0.6, "TestAgent") is True

# Guardrail 2: Cost budget
budget_engine = MockEngine(GuardrailConfig(max_llm_cost_usd=0.05))
budget_engine._record_cost("model", 100, 50)
assert budget_engine._budget_remaining < 0.05

try:
    budget_engine._record_cost("model", 100000, 100000)
except GuardrailBudgetExceededError:
    pass  # Expected

# Guardrail 3: Human approval gate
approval_engine = MockEngine(GuardrailConfig(auto_approve=False))
request = approval_engine._request_approval(
    agent_name="ModelSelector",
    recommendation_summary="Use GradientBoosting (confidence 0.85)",
    confidence=0.85,
    baseline_comparison="RF baseline: 0.88 accuracy",
)

assert isinstance(request, ApprovalRequest)
result = approval_engine.approve(request.id, approved_by="human_reviewer")
assert isinstance(result, ApprovalResult)
assert result.approved is True

# auto_approve=True skips approval
auto_engine = MockEngine(GuardrailConfig(auto_approve=True))
no_request = auto_engine._request_approval("Agent", "Summary", 0.9)
assert no_request is None

# Guardrail 4: Baseline comparison (flag checked in engine logic)
assert engine._guardrail_config.require_baseline is True

# Guardrail 5: Audit trail
audit_engine = MockEngine(GuardrailConfig(audit_trail=True))
entry = audit_engine._log_audit(
    agent_name="DataScientist",
    engine_name="AutoMLEngine",
    input_summary="200 rows, 5 features, classification",
    output_summary="Recommended GradientBoosting, confidence 0.85",
    confidence=0.85,
    llm_cost_usd=0.003,
    approved_by="human",
    baseline_result="RF accuracy: 0.88",
)

assert isinstance(entry, AuditEntry)
assert entry.agent_name == "DataScientist"
assert len(audit_engine.audit_entries) == 1

# AuditEntry serialization
ae_dict = entry.to_dict()
ae_restored = AuditEntry.from_dict(ae_dict)
assert ae_restored.id == entry.id

# Edge case: approve non-existent request
try:
    approval_engine.approve("nonexistent-id", "reviewer")
except ValueError:
    pass  # Expected

print("PASS: 05-ml/15_ml_agents")
```

### Step-by-Step Explanation

1. **Agent discovery**: The `kailash_ml.agents` module exposes six agents via `__all__`. They are lazy-loaded -- the actual classes require `kailash-ml[agents]` (which pulls in Kaizen). Accessing a non-existent agent raises `AttributeError`.

2. **AgentInfusionProtocol**: A runtime-checkable Protocol that defines the agent interface: `suggest_model()`, `suggest_features()`, `interpret_results()`, `interpret_drift()`. Engines check for this protocol to determine whether agent features are available.

3. **GuardrailConfig defaults**: The default configuration provides conservative safety: $1.00 budget, no auto-approval, baseline required, audit trail enabled, 0.5 minimum confidence.

4. **Config validation**: Non-finite and negative values are rejected at construction, preventing misconfigured guardrails from silently disabling safety checks.

5. **CostTracker**: Tracks cumulative LLM costs across calls. `record()` adds a call and returns its estimated cost. Exceeding the budget raises `GuardrailBudgetExceededError`. `reset()` clears the tracking for reuse.

6. **The five guardrails in action**:
   - **Confidence**: `_check_confidence(0.3, "Agent")` returns `False` when below the 0.6 threshold, signaling the engine to fall back to algorithmic mode.
   - **Cost budget**: `_record_cost()` tracks spending via the mixin; exceeding the budget raises an error.
   - **Approval gate**: `_request_approval()` creates an `ApprovalRequest` that must be explicitly approved or rejected. With `auto_approve=True`, it returns `None` (no approval needed).
   - **Baseline comparison**: The `require_baseline` flag tells engines to run a pure algorithmic baseline alongside the agent. This is checked in engine logic, not the mixin.
   - **Audit trail**: `_log_audit()` creates an `AuditEntry` with full context: agent name, engine, I/O summaries, confidence, cost, approval, and baseline. Entries are buffered in `audit_entries`.

## Common Mistakes

| Mistake                                   | Correct Pattern                                  | Why                                                                          |
| ----------------------------------------- | ------------------------------------------------ | ---------------------------------------------------------------------------- |
| Using agents without guardrails           | Always configure `GuardrailConfig`               | Unguarded agents can produce costly, unreviewed recommendations              |
| Setting `auto_approve=True` in production | Keep `auto_approve=False` for production         | Auto-approval bypasses human review of LLM recommendations                   |
| Ignoring confidence scores                | Check `_check_confidence()` and fall back if low | Low-confidence recommendations are unreliable; algorithmic fallback is safer |
| Not checking audit entries                | Review `audit_entries` periodically              | Audit trail is useless if never reviewed                                     |

## Exercises

1. Create a `MockEngine` with a $0.05 budget. Record three small LLM calls and verify the remaining budget decreases. Then trigger `GuardrailBudgetExceededError` with a large call.

2. Implement an approval workflow: create an approval request, inspect its fields, approve it, and verify the result. Then create another request and reject it with a reason.

3. Design a `GuardrailConfig` for a high-stakes production system (medical ML). What values would you choose for each parameter? Justify each choice.

## Key Takeaways

- Six ML agents cover the full ML lifecycle: data analysis, feature engineering, model selection, experiment interpretation, drift analysis, and retraining decisions
- Double opt-in (install extra + `agent=True`) prevents accidental LLM costs
- Five mandatory guardrails: confidence, cost budget, approval gate, baseline comparison, audit trail
- `GuardrailConfig` centralizes all safety settings with validated defaults
- `CostTracker` enforces per-operation LLM spending limits
- Approval requests require explicit human approve/reject when `auto_approve=False`
- `AuditEntry` captures full context for every agent operation

## Next Chapter

[Chapter 16: RLTrainer](16_rl_trainer.md) -- Train reinforcement learning agents using RLTrainer with EnvironmentRegistry and PolicyRegistry.
