# Milestone 4: Fix M4 (2 failing exercises)

## TODO 4.1: Fix M4 ex_3 — syntax error (walrus operator in slice)

File: `modules/ascent04/solutions/ex_3.py`
Error: `SyntaxError: invalid syntax` — walrus operator used inside array slice `Vt[:n_components_to_inspect := min(5, n_features), :]`
Fix: Extract the walrus operator to a separate line:

```python
n_components_to_inspect = min(5, n_features)
Vt[:n_components_to_inspect, :]
```

## TODO 4.2: Fix M4 ex_8 — remove `ModelRegistry.initialize()` call

File: `modules/ascent04/solutions/ex_8.py`
Error: `AttributeError: 'ModelRegistry' object has no attribute 'initialize'`
Fix: Remove the `.initialize()` call. ModelRegistry (like ExperimentTracker) auto-initializes. Also verify the constructor call matches the actual API.
