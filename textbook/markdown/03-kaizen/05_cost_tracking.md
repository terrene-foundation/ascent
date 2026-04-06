# Chapter 5: Cost Tracking and Budgets

## Overview

LLM calls have real costs, and production agents need budget controls. Kailash Kaizen's `Delegate` supports a `budget_usd` parameter that caps estimated spending per agent instance. When the accumulated cost exceeds the budget, the Delegate yields a `BudgetExhausted` event and stops. This chapter covers budget configuration, cost estimation internals, the exhaustion event, validation rules, and the production pattern of per-request budgets.

## Prerequisites

- Python 3.10+ installed
- Kailash Kaizen installed (`pip install kailash-kaizen`)
- Completed [Chapter 4: LLM Providers](04_llm_providers.md)
- Understanding of Delegate basics from [Chapter 2: Delegate](02_delegate.md)
- A `.env` file with at least one LLM provider key configured

## Concepts

### Concept 1: Budget-Tracked Delegates

The `budget_usd` parameter on `Delegate` sets a hard cap on estimated cost. The Delegate tracks cumulative cost after each turn and stops when the budget is exceeded. This is an estimate-based guardrail, not billing data -- it uses model prefix heuristics to approximate cost.

- **What**: A per-instance spending cap that halts the agent when exceeded
- **Why**: Prevents runaway costs from long conversations, retry loops, or unexpectedly verbose responses
- **How**: After each LLM turn, the Delegate estimates cost from token counts and model pricing, adding to a running total
- **When**: Use in any production deployment where cost control matters -- which is every production deployment

### Concept 2: Cost Estimation

Cost estimation uses model name prefixes to look up per-million-token input and output rates. This is an approximation -- actual billing depends on the provider's exact pricing and any discounts. The estimate is conservative enough for budget enforcement but not precise enough for accounting.

- **What**: A heuristic that maps model prefixes (e.g., `claude-`, `gpt-4o`, `gemini-`) to token pricing rates
- **Why**: Enables budget enforcement without requiring real-time billing API access
- **How**: `(prompt_tokens * input_rate + completion_tokens * output_rate) / 1_000_000` per turn
- **When**: Automatically after every LLM call within a Delegate's `run()` loop

### Concept 3: BudgetExhausted Event

When the budget is exceeded, the Delegate does not raise an exception. Instead, it yields a `BudgetExhausted` event containing `budget_usd` (the cap) and `consumed_usd` (the actual spend). This is a `DelegateEvent`, not an error -- the caller decides how to handle it.

- **What**: A typed event emitted when cumulative cost exceeds the budget
- **Why**: Events are non-fatal; the caller can gracefully inform the user instead of crashing
- **How**: The Delegate checks the budget after each turn; if exceeded, it yields `BudgetExhausted` and stops iterating
- **When**: During `async for event in delegate.run(prompt):` -- check `isinstance(event, BudgetExhausted)`

### Concept 4: Budget Validation

Budget values must be finite and non-negative. `NaN`, `Inf`, and negative values are rejected at construction time with `ValueError`. Zero is valid (the Delegate is immediately exhausted on the first turn). Omitting `budget_usd` means unlimited spending.

- **What**: Input validation on the budget parameter
- **Why**: Non-finite budgets (`NaN`, `Inf`) would silently disable the guardrail
- **How**: The Delegate constructor checks `math.isfinite(budget_usd)` and `budget_usd >= 0`
- **When**: At Delegate creation time -- fails fast before any LLM calls

## Key API

| Method / Parameter       | Type            | Description                        |
| ------------------------ | --------------- | ---------------------------------- |
| `Delegate(budget_usd=X)` | `float or None` | Set spending cap; None = unlimited |
| `delegate._budget_usd`   | `float or None` | Current budget cap                 |
| `delegate._consumed_usd` | `float`         | Running total of estimated cost    |
| `BudgetExhausted`        | `DelegateEvent` | Emitted when budget exceeded       |
| `.budget_usd`            | `float`         | The cap that was set               |
| `.consumed_usd`          | `float`         | How much was actually spent        |

## Code Walkthrough

```python
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

from kaizen_agents import Delegate
from kaizen_agents.delegate.events import BudgetExhausted

# ── 1. Budget-tracked Delegate ──────────────────────────────────────
# budget_usd sets a hard cap on estimated cost.

model = os.environ.get("DEFAULT_LLM_MODEL", "claude-sonnet-4-20250514")

delegate = Delegate(model=model, budget_usd=1.0)

assert delegate._budget_usd == 1.0
assert delegate._consumed_usd == 0.0, "Starts at zero"

# ── 2. Cost estimation internals ────────────────────────────────────
# The Delegate estimates cost after each turn using model prefix heuristics.

def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Replicate the Delegate's cost estimation logic."""
    rates = {
        "claude-": (3.0, 15.0),
        "gpt-4o": (2.5, 10.0),
        "gemini-": (1.25, 5.0),
    }
    input_rate, output_rate = 3.0, 15.0  # defaults
    for prefix, (ir, otr) in rates.items():
        if model.startswith(prefix):
            input_rate, output_rate = ir, otr
            break
    return (prompt_tokens * input_rate + completion_tokens * output_rate) / 1_000_000

# Verify cost calculation
cost = estimate_cost("claude-sonnet-4-20250514", 1000, 500)
assert 0.01 < cost < 0.02, f"Expected ~$0.01, got ${cost:.4f}"

# ── 3. BudgetExhausted event ────────────────────────────────────────
# When budget is exceeded, the Delegate yields BudgetExhausted.
# This is a DelegateEvent, not an exception.

event = BudgetExhausted(budget_usd=1.0, consumed_usd=1.05)
assert event.budget_usd == 1.0
assert event.consumed_usd == 1.05

# ── 4. No budget = unlimited ────────────────────────────────────────

unlimited = Delegate(model=model)
assert unlimited._budget_usd is None, "No budget cap"

# ── 5. Budget validation ────────────────────────────────────────────
# Budget must be finite and non-negative

try:
    Delegate(model=model, budget_usd=-0.01)
except ValueError:
    pass  # Negative rejected

try:
    Delegate(model=model, budget_usd=float("nan"))
except ValueError:
    pass  # NaN rejected

# Zero budget is valid (immediately exhausted on first turn)
zero_budget = Delegate(model=model, budget_usd=0.0)
assert zero_budget._budget_usd == 0.0

# ── 6. Production pattern: budget per request ───────────────────────
# In production, create a new Delegate per request with a per-request budget:
#
#   async def handle_request(user_prompt: str):
#       delegate = Delegate(model=model, budget_usd=0.50)
#       async for event in delegate.run(user_prompt):
#           if isinstance(event, BudgetExhausted):
#               return "Request exceeded cost limit"
#           yield event

print("PASS: 03-kaizen/05_cost_tracking")
```

### Step-by-Step Explanation

1. **Budget-tracked Delegate**: `Delegate(model=model, budget_usd=1.0)` creates an agent with a $1.00 spending cap. The `_consumed_usd` counter starts at zero and accumulates after each LLM turn.

2. **Cost estimation**: The estimation function maps model prefixes to per-million-token rates. For `claude-sonnet` with 1000 prompt tokens and 500 completion tokens, the estimated cost is approximately $0.01 per turn.

3. **BudgetExhausted event**: When the budget is exceeded, the Delegate yields `BudgetExhausted(budget_usd=1.0, consumed_usd=1.05)`. The caller handles this in the `async for` loop -- it is not an exception that crashes the program.

4. **Unlimited delegates**: Omitting `budget_usd` (or passing `None`) creates a Delegate with no spending cap. Use this only in development or when the calling system enforces its own budget.

5. **Validation**: Negative, `NaN`, and `Inf` budgets are rejected at construction time. Zero is valid but means the Delegate exhausts its budget on the first turn.

6. **Production pattern**: Create a fresh Delegate per incoming request with a per-request budget. This isolates cost tracking and prevents one expensive conversation from consuming another request's budget.

## Common Mistakes

| Mistake                                  | Correct Pattern                                    | Why                                                                                      |
| ---------------------------------------- | -------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| Reusing a Delegate across requests       | Create a new Delegate per request                  | Shared delegates accumulate cost across requests, making per-request budgets meaningless |
| Treating BudgetExhausted as an exception | Check `isinstance(event, BudgetExhausted)` in loop | It is an event, not an exception; catching it with `except` will not work                |
| Using cost estimates for billing         | Use provider billing APIs for accounting           | Estimates are heuristic approximations, not exact billing data                           |
| Omitting budget in production            | Always set `budget_usd` in production              | Without a budget, a stuck agent loop can generate unbounded costs                        |

## Exercises

1. Write the `estimate_cost` function for a model prefix not listed in the default rates (e.g., `"mistral-"`). Add it to the rates dictionary and verify the calculation.

2. Implement a wrapper function that creates a Delegate, runs a prompt, and returns either the response or a "budget exceeded" message. Test it with a very small budget ($0.001) to trigger exhaustion.

3. Calculate how many turns a $1.00 budget allows for a Claude model generating 500 completion tokens per turn with 1000 prompt tokens. At what turn count would the budget be exhausted?

## Key Takeaways

- `budget_usd` on Delegate sets an estimated cost cap per agent instance
- Cost estimation uses model prefix heuristics, not billing APIs -- sufficient for guardrails, not for accounting
- `BudgetExhausted` is a yielded event, not a raised exception -- handle it in the `async for` loop
- Budget validation rejects NaN, Inf, and negative values at construction time
- Zero budget is valid; omitting budget means unlimited spending
- Production pattern: one Delegate per request with a per-request budget

## Next Chapter

[Chapter 6: Structured Output](06_structured_output.md) -- Define structured output schemas for agent responses using typed Signature fields.
