# Milestone 8: Verification and Regeneration

Depends on: All milestones 1-7 complete.

## TODO 8.1: Run all 80 exercises end-to-end

Run every solution across M1-M10 and verify 80/80 pass:

```bash
for mod in ascent01 ascent02 ascent03 ascent04 ascent05 ascent06 ascent07 ascent08 ascent09 ascent10; do
  for f in modules/$mod/solutions/ex_*.py; do
    timeout 120 uv run python "$f" > /dev/null 2>&1 && echo "OK: $f" || echo "FAIL: $f"
  done
done
```

Any remaining failures must be diagnosed and fixed before proceeding.

## TODO 8.2: Regenerate all 160 notebooks

After all solutions are fixed, regenerate Jupyter + Colab notebooks:

```bash
uv run python scripts/py_to_notebook.py --all
```

Verify the converter runs without error and produces 160 notebooks (80 Jupyter + 80 Colab).

## TODO 8.3: Clean up generated HTML files

Ensure no `ex*.html` files are left in the repo root from exercise runs. Add `*.html` to `.gitignore` if not already there (or be more specific: `ex[0-9]*.html`).

## TODO 8.4: Commit all fixes

Single commit with conventional format:

```
fix(solutions): align all 80 exercises with actual SDK APIs

- Fix wrong parameter names (budget_usd, base_model_id, type)
- Fix wrong class names (JWTMiddleware, ConfidentialityLevel, StructuredTool)
- Fix wrong import paths (dataflow, kaizen.core.base_agent, kailash_ml.rl.trainer)
- Remove non-existent method calls (initialize, plot_training_curves)
- Fix column name mismatches (income_sgd, experiment_group)
- Add missing deps (xgboost, shap, gymnasium)
- Document macOS libomp requirement
- Regenerate all 160 notebooks
```
