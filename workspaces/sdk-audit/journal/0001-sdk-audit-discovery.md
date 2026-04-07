---
type: DISCOVERY
date: 2026-04-06
---

# SDK Audit: ASCENT solutions were written against imagined APIs

## Finding

All 69 failing exercises (of 80 total) fail because solutions were authored speculatively — someone wrote code against hypothetical API signatures without importing and testing against the actual installed packages. The SDK itself is correct and production-proven across 25+ enterprise deployments.

## Root cause categories

1. **Wrong parameter names** (6 exercises): `max_llm_cost_usd` → `budget_usd`, `distribution` → `type`, `base_model` → `base_model_id`
2. **Wrong class names** (2 exercises): `JWTAuth` → `JWTMiddleware`, `ClearanceLevel` → `ConfidentialityLevel`
3. **Wrong import paths** (2 exercises): `kailash_dataflow` → `dataflow`, deep import for `SupervisorWorkerPattern`
4. **Non-existent methods** (4 exercises): `feature_distribution()`, `scatter_plot()`, `tracker.initialize()`, `TIESConfig`
5. **Column name mismatches** (4 exercises): solutions reference columns that don't match generated data schemas
6. **Missing datasets** (24 datasets): solutions reference datasets never added to the generator
7. **Missing Python deps** (3 packages): xgboost, shap, gymnasium not in pyproject.toml

## Legitimate SDK improvements filed

7 issues filed (5 kailash-py, 1 kailash-rs, 1 docs). The two P1 items:

- PreprocessingPipeline needs cardinality guard (kailash-rs already has `max_categories`)
- ModelVisualizer needs EDA chart methods (histogram, scatter, box)

## Key insight

The import path confusion (`kailash_dataflow` vs `dataflow`, `kailash_nexus` vs `nexus`) is NOT an SDK bug. The packages follow standard Python conventions — the pip package name and import name are intentionally different (like `Pillow` → `import PIL`, `scikit-learn` → `import sklearn`). The ASCENT solutions just guessed wrong.
