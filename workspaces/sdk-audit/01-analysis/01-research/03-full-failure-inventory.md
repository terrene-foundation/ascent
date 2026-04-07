# Full Exercise Failure Inventory

Date: 2026-04-07. After M1 fixes + dataset generation.

## Scorecard

| Module    | Pass   | Fail   | Total  |
| --------- | ------ | ------ | ------ |
| M1        | 8      | 0      | 8      |
| M2        | 6      | 2      | 8      |
| M3        | 0      | 8      | 8      |
| M4        | 6      | 2      | 8      |
| M5        | 0      | 8      | 8      |
| M6        | 0      | 8      | 8      |
| M7        | 6      | 2      | 8      |
| M8        | 7      | 1      | 8      |
| M9        | 5      | 3      | 8      |
| M10       | 5      | 3      | 8      |
| **Total** | **43** | **37** | **80** |

## Failures by Root Cause

### WRONG_CLASS (12 exercises)

| Exercise    | Wrong                                                  | Correct                                                                                         |
| ----------- | ------------------------------------------------------ | ----------------------------------------------------------------------------------------------- |
| M5 ex_5     | `from kailash.mcp_server import MCPTool`               | Find actual class name                                                                          |
| M5 ex_7     | `from kaizen_agents import SupervisorWorkerPattern`    | `from kaizen_agents.patterns.patterns import SupervisorWorkerPattern`                           |
| M5 ex_8     | `from nexus.auth import JWTAuth`                       | `from nexus.auth import JWTMiddleware, JWTConfig`                                               |
| M6 ex_4     | `from kailash_align.merge import TIESConfig`           | Does not exist — redesign exercise                                                              |
| M6 ex_7     | `from pact.governance import ClearanceLevel`           | `from pact.governance import ClearanceSpec` or `from kailash.trust import ConfidentialityLevel` |
| M7 ex_8     | `from kailash.infrastructure import ConnectionManager` | Find actual import path                                                                         |
| M8 ex_7     | `from kailash.infrastructure import ConnectionManager` | Same as above                                                                                   |
| M9 ex_5,6,8 | `from kaizen.core import BaseAgent`                    | `from kaizen.core import CoreAgent`                                                             |
| M10 ex_3,4  | `from kailash_ml import RLTrainer`                     | Does not exist in kailash-ml — check actual RL API                                              |
| M10 ex_8    | `from kaizen.core import BaseAgent`                    | `from kaizen.core import CoreAgent`                                                             |

### WRONG_PARAM (9 exercises)

| Exercise    | Wrong                                             | Correct                                 |
| ----------- | ------------------------------------------------- | --------------------------------------- |
| M3 ex_7     | `ParamDistribution(distribution="int_uniform")`   | `ParamDistribution(type="int_uniform")` |
| M5 ex_1     | `Delegate(max_llm_cost_usd=10.0)`                 | `Delegate(budget_usd=10.0)`             |
| M5 ex_2,3,4 | BaseAgent subclass double-passing `signature`     | Pass only in `super().__init__()`       |
| M5 ex_6     | Agent `max_llm_cost_usd`                          | `budget_usd`                            |
| M6 ex_1,2   | `AlignmentConfig(base_model="...")`               | `AlignmentConfig(base_model_id="...")`  |
| M10 ex_6    | `GovernanceEngine()` missing `org` positional arg | Pass org object                         |

### API_MISMATCH (7 exercises)

| Exercise | Error                                        | Fix                       |
| -------- | -------------------------------------------- | ------------------------- |
| M2 ex_8  | `ExperimentTracker.initialize()`             | Remove — auto-initializes |
| M4 ex_3  | Walrus operator in slice syntax error        | Fix syntax                |
| M4 ex_8  | `ModelRegistry.initialize()`                 | Remove — auto-initializes |
| M6 ex_5  | PACT YAML missing org_id                     | Add org_id to YAML        |
| M6 ex_6  | `dict.departments` — expects object not dict | Fix to use LoadedOrg      |
| M6 ex_8  | Pickle truncation in InferenceServer         | Fix model path/loading    |
| M7 ex_3  | `ModelVisualizer.plot_training_curves()`     | Use `training_history()`  |

### MISSING_DEP (4 exercises)

| Exercise | Missing                           | Fix                                  |
| -------- | --------------------------------- | ------------------------------------ |
| M3 ex_2  | xgboost                           | Add to pyproject.toml optional deps  |
| M3 ex_4  | shap                              | Add to pyproject.toml optional deps  |
| M3 ex_5  | `kailash_dataflow` (wrong import) | Change to `from dataflow import ...` |
| M6 ex_3  | gymnasium                         | Add to pyproject.toml optional deps  |

### SYSTEM_DEP (3 exercises)

| Exercise    | Issue                                | Fix                                      |
| ----------- | ------------------------------------ | ---------------------------------------- |
| M3 ex_3,6,8 | libomp missing for LightGBM on macOS | Document `brew install libomp` in README |

### COLUMN_MISMATCH (2 exercises)

| Exercise | Wrong           | Correct            |
| -------- | --------------- | ------------------ |
| M2 ex_4  | `group`         | `experiment_group` |
| M3 ex_1  | `annual_income` | `income_sgd`       |
