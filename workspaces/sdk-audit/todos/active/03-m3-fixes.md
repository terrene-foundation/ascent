# Milestone 3: Fix M3 (8 failing exercises)

Depends on: Milestone 1 (deps installed)

## TODO 3.1: Fix M3 ex_1 — column name `annual_income` → `income_sgd`

File: `modules/ascent03/solutions/ex_1.py`
Error: `ValueError: Target column 'annual_income' not found in data.`
Fix: Replace all references to `"annual_income"` with `"income_sgd"`. Check for any other column name mismatches against the actual `sg_credit_scoring.parquet` schema: `customer_id, age, gender, race, nationality, region, income_sgd, employment_years, months_employed, credit_utilization, avg_balance_utilization, num_credit_lines, credit_age_years, num_hard_inquiries, payment_history_score, num_late_payments, revolving_balance, installment_balance, loan_amount_sgd, loan_purpose, monthly_installment, marital_status, education, housing_type, num_dependents, debt_to_income, savings_balance, checking_balance, previous_defaults, property_value_sgd, loan_to_value, coe_vehicle_owner, cpf_monthly_contribution, application_channel, future_default_indicator, default`.

## TODO 3.2: Fix M3 ex_2 — verify after xgboost install

File: `modules/ascent03/solutions/ex_2.py`
Error: `ModuleNotFoundError: No module named 'xgboost'`
Fix: Should pass after TODO 1.1 (add xgboost to deps). Re-run to verify. If other errors surface, fix them.

## TODO 3.3: Fix M3 ex_3, ex_6, ex_8 — verify after libomp install

Files: `modules/ascent03/solutions/ex_{3,6,8}.py`
Error: `OSError: Library not loaded: @rpath/libomp.dylib`
Fix: Should pass after TODO 1.3 (brew install libomp). Re-run to verify. If other errors surface, fix them.

## TODO 3.4: Fix M3 ex_4 — verify after shap install

File: `modules/ascent03/solutions/ex_4.py`
Error: `ModuleNotFoundError: No module named 'shap'`
Fix: Should pass after TODO 1.1 (add shap to deps). Re-run to verify.

## TODO 3.5: Fix M3 ex_5 — import path `kailash_dataflow` → `dataflow`

File: `modules/ascent03/solutions/ex_5.py`
Error: `ModuleNotFoundError: No module named 'kailash_dataflow'`
Fix: Change `from kailash_dataflow import DataFlow, field` to `from dataflow import DataFlow`. Check what `field` should import from — may be `from dataflow import DataFlow` plus `from dataclasses import field` or similar.

## TODO 3.6: Fix M3 ex_7 — `ParamDistribution(distribution=)` → `type=`

File: `modules/ascent03/solutions/ex_7.py`
Error: `TypeError: ParamDistribution.__init__() got an unexpected keyword argument 'distribution'`
Fix: Replace all `ParamDistribution(... distribution="int_uniform" ...)` with `ParamDistribution(... type="int_uniform" ...)` throughout the file.
