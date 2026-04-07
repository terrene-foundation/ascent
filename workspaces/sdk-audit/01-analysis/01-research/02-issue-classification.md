# Issue Classification: ASCENT Bugs vs SDK Improvements

## Category A: ASCENT Solution Bugs (fix in ASCENT, not SDK)

These are cases where solutions were written against imagined APIs instead of reading the actual SDK.

### A1. Wrong parameter names

| Exercise      | Wrong                                           | Correct                                 | Package       |
| ------------- | ----------------------------------------------- | --------------------------------------- | ------------- |
| M3 ex_7       | `ParamDistribution(distribution="int_uniform")` | `ParamDistribution(type="int_uniform")` | kailash-ml    |
| M5 ex_1, ex_6 | `Delegate(max_llm_cost_usd=10.0)`               | `Delegate(budget_usd=10.0)`             | kaizen-agents |
| M6 ex_1, ex_2 | `AlignmentConfig(base_model="...")`             | `AlignmentConfig(base_model_id="...")`  | kailash-align |

### A2. Wrong class names

| Exercise | Wrong            | Correct                                   | Package |
| -------- | ---------------- | ----------------------------------------- | ------- |
| M5 ex_8  | `JWTAuth`        | `JWTMiddleware` + `JWTConfig`             | nexus   |
| M6 ex_7  | `ClearanceLevel` | `ConfidentialityLevel` or `ClearanceSpec` | pact    |

### A3. Wrong import paths

| Exercise | Wrong                                               | Correct                                                               |
| -------- | --------------------------------------------------- | --------------------------------------------------------------------- |
| M3 ex_5  | `from kailash_dataflow import DataFlow`             | `from dataflow import DataFlow`                                       |
| M5 ex_7  | `from kaizen_agents import SupervisorWorkerPattern` | `from kaizen_agents.patterns.patterns import SupervisorWorkerPattern` |

### A4. Non-existent methods called

| Exercise      | Wrong                                   | What to do instead                                             |
| ------------- | --------------------------------------- | -------------------------------------------------------------- |
| M1 ex_6, ex_8 | `viz.feature_distribution()`            | Use `plotly.express.histogram()` or `DataExplorer.visualize()` |
| M1 ex_6       | `viz.scatter_plot()`                    | Use `plotly.express.scatter()`                                 |
| M2 ex_8       | `tracker.initialize()`                  | Remove — tables auto-initialize on first use                   |
| M6 ex_4       | `TIESConfig` (from kailash_align.merge) | Does not exist. Remove or redesign exercise                    |

### A5. Non-existent API features used

| Exercise  | Wrong                                         | Reality                                        |
| --------- | --------------------------------------------- | ---------------------------------------------- |
| M5 ex_2-4 | BaseAgent subclass double-passing `signature` | Pass `signature=` only in `super().__init__()` |
| M6 ex_5   | PACT YAML without `org_id`                    | `org_id` is a required top-level field         |

### A6. Column name mismatches (solutions vs generated data)

| Exercise | Wrong column         | Correct column                                    | Dataset                   |
| -------- | -------------------- | ------------------------------------------------- | ------------------------- |
| M1 ex_1  | `mean_temperature_c` | computed from `temperature_max`/`temperature_min` | sg_weather.csv            |
| M1 ex_8  | `fare`               | `fare_sgd`                                        | sg_taxi_trips.parquet     |
| M2 ex_4  | `group`              | `experiment_group`                                | experiment_data.parquet   |
| M3 ex_1  | `annual_income`      | `income_sgd`                                      | sg_credit_scoring.parquet |

### A7. Missing datasets (24 datasets never generated)

These are ASCENT infrastructure gaps, not SDK issues. Solutions reference datasets that were never added to `generate_datasets.py`.

## Category B: Legitimate SDK Improvements (file on kailash-py)

These are genuine gaps in the SDK that would benefit all users, not just ASCENT.

### B1. PreprocessingPipeline: No cardinality protection for one-hot encoding (kailash-ml)

**Problem**: `PreprocessingPipeline.setup(categorical_encoding="onehot")` creates one column per unique value with no upper bound. A 50k-unique column produces 50k new columns. No warning, no guard.

**Evidence**: kailash-rs's `OneHotEncoder` already has `max_categories: Option<usize>` and `min_frequency: Option<f64>`. The Python version lacks this.

**Impact**: Silent OOM in production. Any dataset with an ID-like categorical column will explode. The 25+ enterprise deployments are either avoiding one-hot or pre-filtering columns manually.

**Recommendation**: Add `max_cardinality: int = 50` to `setup()`. Columns above threshold auto-switch to ordinal encoding with a logged warning.

### B2. ModelVisualizer: Missing EDA chart methods (kailash-ml)

**Problem**: ModelVisualizer has 9 methods, all ML-evaluation-specific. No general-purpose histogram, scatter, or box plot. For EDA (the most common data science activity), users must drop to raw Plotly.

**Context**: `DataExplorer.visualize()` generates some EDA charts, but they're embedded in a profiling report — not callable individually. There's no way to say "give me a histogram of this column" through any Kailash engine.

**Recommendation**: Add to ModelVisualizer (or a new `DataVisualizer`):

- `distribution(values, feature_name, bins=50)` — histogram + KDE
- `scatter(x, y, color=None, labels=None)` — scatter plot
- `box_plot(values, groups=None, feature_name=None)` — box/violin

This closes the gap with kailash-rs's `DataExplorer` which generates distribution plots in its HTML reports.

### B3. training_history: Missing y_label parameter (kailash-ml)

**Problem**: `training_history(metrics, x_label="Epoch")` has `x_label` but no `y_label`. Asymmetric API.

**Recommendation**: Add `y_label: str = "Value"` parameter.

### B4. kailash-rs parity: Missing high-level engines (kailash-rs)

**Problem**: kailash-rs implements algorithms at the estimator level (`Fit`/`Predict` traits) but lacks orchestration engines that kailash-py provides:

- No `ModelVisualizer` (even for ML evaluation)
- No `FeatureEngineer`
- No `FeatureStore`
- No `DriftMonitor`
- No `InferenceServer`
- No `OnnxBridge`

**Context**: These engines compose the primitives into opinionated workflows. Without them, Rust users must manually wire preprocessing, training, evaluation, and registry — the exact boilerplate these engines eliminate.

**Recommendation**: Priority order for Rust engine ports:

1. `ModelVisualizer` (HTML report generation already exists in `kailash-ml-explorer`)
2. `InferenceServer` (ONNX runtime is a Rust strength)
3. `FeatureStore` + `FeatureEngineer`
4. `DriftMonitor`

### B5. SupervisorWorkerPattern not in kaizen_agents top-level exports (kaizen-agents)

**Problem**: `SupervisorWorkerPattern` exists at `kaizen_agents.patterns.patterns` but is not exported from `kaizen_agents.__init__`. Users must know the deep import path.

**Context**: `Delegate`, `Agent`, `ReActAgent`, `Pipeline` are all top-level exports. `SupervisorWorkerPattern` is the conspicuous absence.

**Recommendation**: Add to `kaizen_agents.__init__.__all__`.

## Category C: Documentation Gaps (file on kailash-py)

### C1. ExperimentTracker requires ConnectionManager but no guidance on creating one

The constructor takes `conn: ConnectionManager` but there's no obvious import path or factory method for creating a ConnectionManager for simple local use. Enterprise users have this wired up, but standalone usage (teaching, prototyping) needs a `create_local_tracker()` convenience.

### C2. ParamDistribution `type` field name

Using `type` as a field name shadows Python's `type()` builtin. While functional, it confuses beginners and triggers linter warnings. Not a bug, but worth noting in docs: "The `type` field specifies the distribution type (e.g., `'int_uniform'`, `'log_uniform'`). It shadows the Python builtin but this is intentional for API consistency."
