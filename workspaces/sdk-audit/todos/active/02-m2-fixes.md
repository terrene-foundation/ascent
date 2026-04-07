# Milestone 2: Fix M2 (2 failing exercises)

## TODO 2.1: Fix M2 ex_4 — column name `group` → `experiment_group`

File: `modules/ascent02/solutions/ex_4.py`
Error: `ColumnNotFoundError: unable to find column "group"`
Fix: Replace all references to column `"group"` with `"experiment_group"` in the experiment_data.parquet dataset.

## TODO 2.2: Fix M2 ex_8 — remove `ExperimentTracker.initialize()` call

File: `modules/ascent02/solutions/ex_8.py`
Error: `AttributeError: 'ExperimentTracker' object has no attribute 'initialize'`
Fix: Remove the `tracker.initialize()` call. ExperimentTracker auto-initializes tables lazily on first use. Also verify the `ConnectionManager` import path — use `from kailash.db import ConnectionManager` if needed, or restructure to use `ExperimentTracker` with the correct constructor pattern.
