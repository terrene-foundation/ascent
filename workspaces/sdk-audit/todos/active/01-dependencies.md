# Milestone 1: Dependencies and Infrastructure

These unblock everything else. No code changes to solutions.

## TODO 1.1: Add missing Python deps to pyproject.toml

Add to `[project.optional-dependencies]`:

```toml
ml = ["xgboost>=2.0", "shap>=0.44"]
rl = ["gymnasium>=1.0", "kailash-ml[rl]"]
```

Unblocks: M3 ex_2 (xgboost), M3 ex_4 (shap), M6 ex_3 (gymnasium)

## TODO 1.2: Document macOS LightGBM requirement in README

Add to Quick Start section:

```bash
# macOS only — LightGBM needs OpenMP
brew install libomp
```

Unblocks: M3 ex_3, ex_6, ex_8 (libomp)

## TODO 1.3: Install deps and verify libomp

Run `brew install libomp` locally, then `uv sync --extra ml --extra rl` to verify all deps install.
