# Milestone 6: Fix M6 (8 failing exercises)

All M6 exercises fail — align/pact/RL API mismatches.

## TODO 6.1: Fix M6 ex_1 — `AlignmentConfig(base_model=)` → `base_model_id=`

File: `modules/ascent06/solutions/ex_1.py`
Error: `TypeError: AlignmentConfig.__init__() got an unexpected keyword argument 'base_model'. Did you mean 'base_model_id'?`
Fix: Replace `base_model="..."` with `base_model_id="..."` throughout. Verify other AlignmentConfig params match actual signature.

## TODO 6.2: Fix M6 ex_2 — same `base_model` → `base_model_id`

File: `modules/ascent06/solutions/ex_2.py`
Same fix as TODO 6.1.

## TODO 6.3: Fix M6 ex_3 — verify after gymnasium install

File: `modules/ascent06/solutions/ex_3.py`
Error: `ModuleNotFoundError: No module named 'gymnasium'`
Fix: Should pass after TODO 1.1 (add gymnasium to deps). Re-run to verify. If other errors surface, fix them.

## TODO 6.4: Fix M6 ex_4 — remove TIESConfig (doesn't exist)

File: `modules/ascent06/solutions/ex_4.py`
Error: `ImportError: cannot import name 'TIESConfig' from 'kailash_align.merge'`
Fix: `TIESConfig` does not exist. kailash-align's merge API only has `AdapterMerger` for LoRA-to-full-model merge. Read the exercise to understand intent:

- If it's about merging LoRA adapters into base model: rewrite to use `from kailash_align import AdapterMerger` / `from kailash_align.merge import merge_adapter`
- If it's about multi-model merging (TIES, DARE, model soup): this capability doesn't exist in kailash-align. Redesign the exercise to use adapter merging instead.

## TODO 6.5: Fix M6 ex_5 — PACT YAML missing `org_id`

File: `modules/ascent06/solutions/ex_5.py`
Error: `ConfigurationError: Required field 'org_id' is missing from YAML org definition`
Fix: The PACT YAML loader requires `org_id` as a top-level field. Read the solution, find the YAML construction (likely a dict or tempfile), and add `org_id` to the YAML structure. Required schema: `org_id` (string), `name` (string), optional: `departments`, `teams`, `roles`, `clearances`, `envelopes`, `bridges`, `ksps`.

## TODO 6.6: Fix M6 ex_6 — `dict.departments` → use LoadedOrg

File: `modules/ascent06/solutions/ex_6.py`
Error: `AttributeError: 'dict' object has no attribute 'departments'`
Fix: The code calls `.departments` on a dict — it expects a `LoadedOrg` object (returned by `load_org_yaml()`). Read the solution and fix to properly use `from pact import load_org_yaml, LoadedOrg` and access attributes on the returned object.

## TODO 6.7: Fix M6 ex_7 — `ClearanceLevel` → correct class

File: `modules/ascent06/solutions/ex_7.py`
Error: `ImportError: cannot import name 'ClearanceLevel' from 'pact.governance'. Did you mean: 'ClearanceSpec'?`
Fix: Determine intent:

- If the exercise needs the enum of classification levels: use `ConfidentialityLevel` from `kailash.trust` (values: PUBLIC, RESTRICTED, CONFIDENTIAL, SECRET, TOP_SECRET)
- If it needs the YAML clearance spec: use `ClearanceSpec` from `pact.governance`
  Read the solution context to determine which is correct.

## TODO 6.8: Fix M6 ex_8 — pickle truncation in InferenceServer

File: `modules/ascent06/solutions/ex_8.py`
Error: `_pickle.UnpicklingError: pickle data was truncated`
Fix: The exercise likely creates a fake/minimal model file that InferenceServer tries to load. Read the solution — either:

- Fix the model creation to produce a valid pickle
- Use a properly trained model artifact from an earlier exercise
- Mock the model loading path correctly
