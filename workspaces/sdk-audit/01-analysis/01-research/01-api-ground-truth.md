# SDK API Ground Truth

Research date: 2026-04-06. Source: kailash-py monorepo at `~/repos/loom/kailash-py`.

## Package Import Paths

Every kailash package has its own Python module name. These are the **correct** import paths:

| pip package        | Python import          | ASCENT used (wrong?)              |
| ------------------ | ---------------------- | --------------------------------- |
| `kailash`          | `import kailash`       | correct                           |
| `kailash-ml`       | `import kailash_ml`    | correct                           |
| `kailash-dataflow` | `import dataflow`      | `import kailash_dataflow` (WRONG) |
| `kailash-nexus`    | `import nexus`         | correct                           |
| `kailash-kaizen`   | `import kaizen`        | correct                           |
| `kaizen-agents`    | `import kaizen_agents` | correct                           |
| `kailash-pact`     | `import pact`          | correct                           |
| `kailash-align`    | `import kailash_align` | correct                           |

## kailash-ml 0.4.0

### ModelVisualizer (P2 Experimental)

- Import: `from kailash_ml import ModelVisualizer`
- Constructor: `ModelVisualizer()` (no params)
- **9 methods** (all return `plotly.graph_objects.Figure`):
  - `confusion_matrix(y_true, y_pred, labels=None)`
  - `roc_curve(y_true, y_scores, pos_label=1)`
  - `precision_recall_curve(y_true, y_scores, pos_label=1)`
  - `feature_importance(model, feature_names, top_n=20, *, X=None, y=None)`
  - `learning_curve(model, X, y, cv=5, train_sizes=None)`
  - `residuals(y_true, y_pred)`
  - `calibration_curve(y_true, y_proba, n_bins=10)`
  - `metric_comparison(results: dict[str, dict[str, float]])`
  - `training_history(metrics: dict[str, list[float]], x_label="Epoch")`
- **Does NOT have**: `feature_distribution()`, `scatter_plot()`, `histogram()`, `box_plot()`, `heatmap()`
- **Design intent**: ML model evaluation, not general EDA

### PreprocessingPipeline (P1 Production with Caveats)

- Import: `from kailash_ml import PreprocessingPipeline`
- Constructor: `PreprocessingPipeline()` (no params)
- `setup(data, target, *, train_size=0.8, seed=42, normalize=True, categorical_encoding="onehot", imputation_strategy="mean", remove_outliers=False, outlier_threshold=0.05) -> SetupResult`
- `transform(data) -> pl.DataFrame`
- `inverse_transform(data) -> pl.DataFrame`
- **No cardinality protection**: one-hot encodes ALL unique values. No `max_cardinality`, no `exclude_columns`.

### DataExplorer (P1 Production)

- Import: `from kailash_ml import DataExplorer`
- Constructor: `DataExplorer(*, output_format="dict", alert_config=None)`
- Methods (all async): `profile()`, `visualize()`, `to_html()`, `compare()`
- `AlertConfig` fields: `high_correlation_threshold=0.9`, `high_null_pct_threshold=0.05`, `constant_threshold=1`, `high_cardinality_ratio=0.9`, `skewness_threshold=2.0`, `zero_pct_threshold=0.5`, `imbalance_ratio_threshold=0.1`, `duplicate_pct_threshold=0.0`

### ParamDistribution

- Import: `from kailash_ml.engines.hyperparameter_search import ParamDistribution`
- Constructor: `ParamDistribution(name, type, low=None, high=None, choices=None)`
- `type` options: `"uniform"`, `"log_uniform"`, `"int_uniform"`, `"categorical"`
- **Field is `type`, not `distribution`**

### ExperimentTracker

- Import: `from kailash_ml import ExperimentTracker`
- Constructor: `ExperimentTracker(conn, artifact_root="./mlartifacts")`
- **No `initialize()` method** — tables auto-created lazily
- Usage pattern: `async with tracker.run(experiment_name, run_name) as ctx:`

### TrainingPipeline

- Import: `from kailash_ml import TrainingPipeline`
- Constructor: `TrainingPipeline(feature_store, registry)`
- `train(data, schema, model_spec, eval_spec, experiment_name, *, agent=None) -> TrainingResult`
- `ModelSpec(model_class, hyperparameters={}, framework="sklearn")`
- `EvalSpec(metrics=["accuracy"], split_strategy="holdout", n_splits=5, test_size=0.2, min_threshold={})`

### FeatureEngineer (P2 Experimental)

- Import: `from kailash_ml import FeatureEngineer`
- Constructor: `FeatureEngineer(feature_store=None, *, max_features=50)`
- `generate(data, schema, *, strategies=None) -> GeneratedFeatures` (SYNC, not async)
- `select(data, candidates, target, *, method="importance", top_k=None) -> SelectedFeatures` (SYNC)

### HyperparameterSearch

- Import: `from kailash_ml import HyperparameterSearch`
- Constructor: `HyperparameterSearch(pipeline: TrainingPipeline)`
- `search(data, schema, base_model_spec, search_space, config, eval_spec, experiment_name) -> SearchResult`
- `SearchConfig(strategy="bayesian", n_trials=50, timeout_seconds=None, metric_to_optimize="accuracy", direction="maximize", early_stopping_patience=None, n_jobs=1, register_best=True)`

## kaizen / kaizen-agents

### Delegate

- Import: `from kaizen_agents import Delegate`
- Constructor: `Delegate(model="", *, tools=None, system_prompt=None, max_turns=50, budget_usd=None, adapter=None, config=None)`
- **Budget param is `budget_usd`**, not `max_llm_cost_usd`

### BaseAgent

- Import: `from kaizen.core import BaseAgent`
- Constructor: `BaseAgent(config, signature=None, strategy=None, memory=None, shared_memory=None, agent_id=None, control_protocol=None, mcp_servers=None, hook_manager=None, **kwargs)`
- **Correct subclassing**: Pass `signature=` explicitly in `super().__init__()`, or override `_default_signature()`. Do NOT also accept `signature` as your own `__init__` param — this causes "multiple values" conflict.

### SupervisorWorkerPattern

- **Exists** but NOT top-level exported
- Import: `from kaizen_agents.patterns.patterns import SupervisorWorkerPattern`

### kaizen_agents top-level exports

- `Delegate`, `GovernedSupervisor`, `SupervisorResult`, `Agent`, `ReActAgent`, `Pipeline`

## nexus

### Auth

- **No `JWTAuth` class**
- Actual: `JWTMiddleware`, `JWTConfig`
- Import: `from nexus.auth import JWTMiddleware, JWTConfig`

## pact

### Clearance

- **No `ClearanceLevel` class**
- Actual: `ConfidentialityLevel` (enum: PUBLIC, RESTRICTED, CONFIDENTIAL, SECRET, TOP_SECRET) or `ClearanceSpec` (YAML spec dataclass)
- YAML requires `org_id` as top-level field (validated)

## kailash-align

### AlignmentConfig

- Parameter is `base_model_id`, not `base_model`
- `AlignmentConfig(method="sft_then_dpo", base_model_id="", ...)`

### Merge API

- **No `TIESConfig`** — only `AdapterMerger` for LoRA-to-full-model merge
- Import: `from kailash_align import AdapterMerger`

## kailash-rs Parity Notes

- **No ModelVisualizer** in Rust — closest is `kailash-ml-explorer` (data profiling only)
- **OneHotEncoder in Rust HAS `max_categories`** — Python version does not
- **ndarray-native** (not polars) — fundamentally different data paradigm
- **Missing high-level engines**: FeatureEngineer, FeatureStore, DriftMonitor, InferenceServer, OnnxBridge
- **Ahead in RL**: Full environments (CartPole, GridWorld, MountainCar, etc.)
