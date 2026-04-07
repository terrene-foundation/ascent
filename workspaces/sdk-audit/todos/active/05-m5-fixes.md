# Milestone 5: Fix M5 (8 failing exercises)

All M5 exercises fail — kaizen/nexus/MCP API mismatches.

## TODO 5.1: Fix M5 ex_1 — `Delegate(max_llm_cost_usd=)` → `budget_usd=`

File: `modules/ascent05/solutions/ex_1.py`
Error: `TypeError: Delegate.__init__() got an unexpected keyword argument 'max_llm_cost_usd'`
Fix: Replace `max_llm_cost_usd=X` with `budget_usd=X`. Also verify other Delegate params match actual signature: `Delegate(model="", *, tools=None, system_prompt=None, max_turns=50, budget_usd=None, adapter=None, config=None)`.

## TODO 5.2: Fix M5 ex_2 — BaseAgent subclass signature conflict

File: `modules/ascent05/solutions/ex_2.py`
Error: `TypeError: BaseAgent.__init__() got multiple values for keyword argument 'signature'`
Fix: The subclass must NOT accept `signature` as its own `__init__` param AND pass it to `super().__init__()`. Correct pattern:

```python
class MyAgent(BaseAgent):
    def __init__(self, config, **kwargs):
        super().__init__(
            config=config,
            signature=MySignature(),  # pass here only
            **kwargs,
        )
```

Import path: `from kaizen.core.base_agent import BaseAgent`

## TODO 5.3: Fix M5 ex_3 — same BaseAgent signature conflict

File: `modules/ascent05/solutions/ex_3.py`
Same fix as TODO 5.2.

## TODO 5.4: Fix M5 ex_4 — same BaseAgent signature conflict

File: `modules/ascent05/solutions/ex_4.py`
Same fix as TODO 5.2.

## TODO 5.5: Fix M5 ex_5 — `MCPTool` → `StructuredTool` or `structured_tool` decorator

File: `modules/ascent05/solutions/ex_5.py`
Error: `ImportError: cannot import name 'MCPTool' from 'kailash.mcp_server'`
Fix: `MCPTool` doesn't exist. The actual API uses `StructuredTool` class or `@structured_tool` decorator from `kailash.mcp_server`. Read the solution to understand intent, then rewrite to use the correct class.

## TODO 5.6: Fix M5 ex_6 — `max_llm_cost_usd` on custom agent

File: `modules/ascent05/solutions/ex_6.py`
Error: `TypeError: DataScientistAgent.__init__() got an unexpected keyword argument 'max_llm_cost_usd'`
Fix: Same as TODO 5.1 — replace with `budget_usd`. Also check the custom agent subclass pattern (same as TODO 5.2).

## TODO 5.7: Fix M5 ex_7 — `SupervisorWorkerPattern` import path

File: `modules/ascent05/solutions/ex_7.py`
Error: `ImportError: cannot import name 'SupervisorWorkerPattern' from 'kaizen_agents'`
Fix: Change to `from kaizen_agents.patterns.patterns import SupervisorWorkerPattern`.

## TODO 5.8: Fix M5 ex_8 — `JWTAuth` import from nexus.auth

File: `modules/ascent05/solutions/ex_8.py`
Error: `ImportError: cannot import name 'JWTAuth' from 'nexus.auth'`
Fix: `JWTAuth` exists at `kailash.mcp_server` (not `nexus.auth`). The nexus auth equivalent is `JWTMiddleware` + `JWTConfig` from `nexus.auth`. Read the solution to determine which is appropriate — if it's about Nexus API auth, use `from nexus.auth import JWTMiddleware, JWTConfig`. If it's about MCP auth, use `from kailash.mcp_server import JWTAuth`.
