# Milestone 7: Fix M7-M10 (8 failing exercises)

## TODO 7.1: Fix M7 ex_3 — `ModelVisualizer.plot_training_curves()` → `training_history()`

File: `modules/ascent07/solutions/ex_3.py`
Error: `AttributeError: 'ModelVisualizer' object has no attribute 'plot_training_curves'`
Fix: Replace `viz.plot_training_curves(...)` with `viz.training_history(metrics=..., x_label=...)`. The `training_history` method takes `metrics: dict[str, list[float]]` and `x_label: str`.

## TODO 7.2: Fix M7 ex_8 — `ConnectionManager` import path

File: `modules/ascent07/solutions/ex_8.py`
Error: `ImportError: cannot import name 'ConnectionManager' from 'kailash.infrastructure'`
Fix: `ConnectionManager` does not exist at `kailash.infrastructure`. It's at `kailash.db`. Change import to `from kailash.db import ConnectionManager`. Verify the constructor signature matches what the solution passes.

## TODO 7.3: Fix M8 ex_7 — same `ConnectionManager` import

File: `modules/ascent08/solutions/ex_7.py`
Same fix as TODO 7.2 — `from kailash.db import ConnectionManager`.

## TODO 7.4: Fix M9 ex_5 — `BaseAgent` import path

File: `modules/ascent09/solutions/ex_5.py`
Error: `ImportError: cannot import name 'BaseAgent' from 'kaizen.core'`
Fix: `BaseAgent` is not exported from `kaizen.core.__init__`. Use explicit submodule import: `from kaizen.core.base_agent import BaseAgent`. Also verify the subclass pattern doesn't double-pass `signature` (same issue as M5 TODO 5.2).

## TODO 7.5: Fix M9 ex_6 — same `BaseAgent` import

File: `modules/ascent09/solutions/ex_6.py`
Same fix as TODO 7.4.

## TODO 7.6: Fix M9 ex_8 — same `BaseAgent` import

File: `modules/ascent09/solutions/ex_8.py`
Same fix as TODO 7.4.

## TODO 7.7: Fix M10 ex_3 — `RLTrainer` import path

File: `modules/ascent10/solutions/ex_3.py`
Error: `ImportError: cannot import name 'RLTrainer' from 'kailash_ml'`
Fix: `RLTrainer` is not a top-level export. Use: `from kailash_ml.rl.trainer import RLTrainer, RLTrainingConfig, RLTrainingResult`.

## TODO 7.8: Fix M10 ex_4 — same `RLTrainer` import

File: `modules/ascent10/solutions/ex_4.py`
Same fix as TODO 7.7.

## TODO 7.9: Fix M10 ex_6 — `GovernanceEngine` missing `org` arg

File: `modules/ascent10/solutions/ex_6.py`
Error: `TypeError: GovernanceEngine.__init__() missing 1 required positional argument: 'org'`
Fix: Read the solution to find how GovernanceEngine is constructed. The constructor requires an `org` positional argument (a loaded org object from PACT). Add the org argument to the constructor call.

## TODO 7.10: Fix M10 ex_8 — same `BaseAgent` import

File: `modules/ascent10/solutions/ex_8.py`
Same fix as TODO 7.4.
