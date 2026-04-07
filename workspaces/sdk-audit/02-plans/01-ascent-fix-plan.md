# ASCENT Fix Plan

## Issues Filed on SDK Repos

| #   | Repo                              | Issue                                   | Type           |
| --- | --------------------------------- | --------------------------------------- | -------------- |
| 1   | terrene-foundation/kailash-py#313 | PreprocessingPipeline cardinality guard | enhancement P1 |
| 2   | terrene-foundation/kailash-py#314 | ModelVisualizer EDA chart methods       | enhancement P1 |
| 3   | terrene-foundation/kailash-py#315 | training_history y_label parameter      | enhancement    |
| 4   | terrene-foundation/kailash-py#316 | Export SupervisorWorkerPattern          | enhancement    |
| 5   | terrene-foundation/kailash-py#317 | ExperimentTracker standalone usage docs | documentation  |
| 6   | terrene-foundation/kailash-py#318 | ParamDistribution type field docs       | documentation  |
| 7   | esperie-enterprise/kailash-rs#223 | Missing high-level engine crates        | enhancement    |

## What to Fix in ASCENT Now (Not Blocked on SDK)

### Phase 1: Solution API fixes (all 80 exercises)

Every wrong API call, import, parameter name, and class name must match the actual SDK. This is the bulk of the work.

**By module:**

| Module         | Fixes needed                                                                                                        | Blocked by SDK?    |
| -------------- | ------------------------------------------------------------------------------------------------------------------- | ------------------ |
| M1 (8 ex)      | DONE this session                                                                                                   | No                 |
| M2 (8 ex)      | Column names (experiment_group), ExperimentTracker usage, missing datasets                                          | Datasets only      |
| M3 (8 ex)      | Column name (income_sgd), `ParamDistribution(type=)`, `from dataflow import`, xgboost/shap deps, libomp             | No                 |
| M4 (8 ex)      | Missing datasets, check for API mismatches                                                                          | Datasets only      |
| M5 (8 ex)      | `Delegate(budget_usd=)`, BaseAgent subclassing, `SupervisorWorkerPattern` import, `MCPTool` import, `JWTMiddleware` | No                 |
| M6 (8 ex)      | `AlignmentConfig(base_model_id=)`, remove TIESConfig, `ConfidentialityLevel`, PACT YAML org_id, gymnasium dep       | No                 |
| M7-M10 (32 ex) | Missing datasets + likely API mismatches (not yet diagnosed)                                                        | Datasets + unknown |

**Fix for ModelVisualizer gap (not blocked on SDK):**

- Use `plotly.express` for EDA charts (histogram, scatter, box)
- Use `ModelVisualizer` for ML evaluation (confusion_matrix, roc_curve, etc.)
- Use `DataExplorer.visualize()` for automated profiling charts
- Document this pattern in exercise headers: "Plotly for EDA, ModelVisualizer for ML evaluation"

**Fix for PreprocessingPipeline gap (not blocked on SDK):**

- Always exclude ID columns from feature set before calling `setup()`
- Add `# IMPORTANT: Exclude high-cardinality ID columns before one-hot encoding` comments
- Use `categorical_encoding="ordinal"` for columns with >50 unique values

### Phase 2: Missing datasets (24 datasets)

Add generators to `scripts/generate_datasets.py` for all 24 missing datasets. Schema must match what the (fixed) solutions expect.

### Phase 3: Missing Python dependencies

Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
ml = [
    "xgboost>=2.0",
    "shap>=0.44",
]
rl = [
    "gymnasium>=1.0",
]
```

Document `brew install libomp` for macOS LightGBM in README.

### Phase 4: Regenerate notebooks

After all solution fixes, re-run `py_to_notebook.py` to regenerate all 160 notebooks (80 Jupyter + 80 Colab).

## Execution Order

1. Fix M2-M6 solutions (API mismatches) — can start immediately
2. Generate M2-M10 missing datasets — can start immediately in parallel
3. Add missing deps to pyproject.toml — quick fix
4. Diagnose and fix M7-M10 solutions — after datasets exist
5. Regenerate all notebooks — after all solutions fixed
6. Re-run all 80 solutions end-to-end — final verification
