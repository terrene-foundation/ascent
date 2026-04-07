# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT05 — Exercise 6: ML Agent Pipeline
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Chain all 6 kailash-ml agents in a full ML pipeline:
#   DataScientist → FeatureEngineer → ModelSelector → ExperimentInterpreter
#   → DriftAnalyst → RetrainingDecision. Demonstrate the double opt-in
#   pattern (AgentInfusionProtocol).
#
# TASKS:
#   1. Set up the 6 ML agents with cost budgets
#   2. DataScientistAgent: initial data analysis
#   3. FeatureEngineerAgent: suggest features
#   4. ModelSelectorAgent: recommend model architecture
#   5. ExperimentInterpreterAgent: interpret results
#   6. DriftAnalyst + RetrainingDecision: production monitoring
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os

import polars as pl
from dotenv import load_dotenv

# Patch kaizen.core to re-export BaseAgent for kailash-ml compatibility.
# kailash-ml agents expect `from kaizen.core import BaseAgent` but the installed
# kaizen version only exports it at kaizen.core.base_agent.  This shim bridges
# the gap until the SDK packages are aligned.
import kaizen.core as _kaizen_core
from kaizen.core.base_agent import BaseAgent as _BaseAgent

if not hasattr(_kaizen_core, "BaseAgent"):
    _kaizen_core.BaseAgent = _BaseAgent
from kaizen import Signature, InputField, OutputField

if not hasattr(_kaizen_core, "Signature"):
    _kaizen_core.Signature = Signature
if not hasattr(_kaizen_core, "InputField"):
    _kaizen_core.InputField = InputField
if not hasattr(_kaizen_core, "OutputField"):
    _kaizen_core.OutputField = OutputField

from kailash_ml.agents.data_scientist import DataScientistAgent
from kailash_ml.agents.model_selector import ModelSelectorAgent
from kailash_ml.agents.feature_engineer import FeatureEngineerAgent
from kailash_ml.agents.experiment_interpreter import ExperimentInterpreterAgent
from kailash_ml.agents.drift_analyst import DriftAnalystAgent
from kailash_ml.agents.retraining_decision import RetrainingDecisionAgent

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()

model = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))
if not model or not os.environ.get("OPENAI_API_KEY"):
    print("Set OPENAI_API_KEY and DEFAULT_LLM_MODEL in .env to run this exercise")
    raise SystemExit(0)


# ── Data Loading ──────────────────────────────────────────────────────

loader = ASCENTDataLoader()
credit = loader.load("ascent03", "sg_credit_scoring.parquet")

data_context = {
    "dataset": "Singapore Credit Scoring",
    "rows": credit.height,
    "columns": credit.columns,
    "target": "default",
    "default_rate": float(credit["default"].mean()),
    "features": len(credit.columns) - 1,
}

print(f"=== ML Agent Pipeline ===")
print(f"Dataset: {data_context['dataset']}")
print(f"Shape: {data_context['rows']:,} x {data_context['features']} features")
print(f"Default rate: {data_context['default_rate']:.2%}")

# Check kailash-ml + kaizen version compatibility.  If the installed versions
# are mismatched (BaseAgent constructor changed between kaizen releases) the
# agent.analyze() call will fail at runtime.  In that case we fall back to
# a demonstration mode that shows the correct API pattern without LLM calls.
_ML_AGENTS_AVAILABLE = True
try:
    _test_agent = DataScientistAgent(model=model)
    # Try building the internal _Agent to verify compatibility
    _test_agent._ensure_agent()
except Exception as _compat_err:
    _ML_AGENTS_AVAILABLE = False
    print(
        f"\nNote: kailash-ml agent runtime unavailable ({type(_compat_err).__name__})."
    )
    print(f"Running in demonstration mode — correct API patterns shown below.")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: DataScientistAgent — initial analysis
# ══════════════════════════════════════════════════════════════════════


async def step_1_data_scientist():
    # TODO: Implement step_1_data_scientist():
    #   1. Create DataScientistAgent(model=model)
    #   2. Build data_profile string from credit shape, columns, target rate, numeric features
    #   3. Call agent.analyze(data_profile=..., business_context="Singapore credit scoring for a bank. 12% default rate. Need production model.")
    #   4. Print each result key:value[:200]; return result
    ____
    ____
    ____
    ____
    ____
    ____
    ____


if _ML_AGENTS_AVAILABLE:
    ds_result = asyncio.run(step_1_data_scientist())
else:
    print("\n=== Step 1: DataScientistAgent (demo) ===")
    print(
        "  API: DataScientistAgent(model=model).analyze(data_profile=..., business_context=...)"
    )
    ds_result = {}


# ══════════════════════════════════════════════════════════════════════
# TASK 2: FeatureEngineerAgent — suggest features
# ══════════════════════════════════════════════════════════════════════


async def step_2_feature_engineer():
    # TODO: Implement step_2_feature_engineer():
    #   1. Create FeatureEngineerAgent(model=model)
    #   2. Build data_profile string describing credit dataset rows, columns, target, key features
    #   3. Call agent.suggest(data_profile=..., target_description="Binary default prediction (0/1) for credit scoring", existing_features=", ".join(credit.columns))
    #   4. Print each result key:value[:200]; return result
    ____
    ____
    ____
    ____
    ____
    ____
    ____


if _ML_AGENTS_AVAILABLE:
    fe_result = asyncio.run(step_2_feature_engineer())
else:
    print("\n=== Step 2: FeatureEngineerAgent (demo) ===")
    print(
        "  API: FeatureEngineerAgent(model=model).suggest(data_profile=..., target_description=...)"
    )
    fe_result = {}


# ══════════════════════════════════════════════════════════════════════
# TASK 3: ModelSelectorAgent — recommend model
# ══════════════════════════════════════════════════════════════════════


async def step_3_model_selector():
    # TODO: Implement step_3_model_selector():
    #   1. Create ModelSelectorAgent(model=model)
    #   2. Build data_characteristics string: binary classification, row count, features, 12% imbalance
    #   3. Call agent.select(data_characteristics=..., task_type="binary_classification",
    #      constraints="High interpretability required. Inference latency < 50ms. Target metric: AUC-PR.")
    #   4. Print each result key: value[:200]; return result
    ____
    ____
    ____
    ____
    ____
    ____
    ____


if _ML_AGENTS_AVAILABLE:
    ms_result = asyncio.run(step_3_model_selector())
else:
    print("\n=== Step 3: ModelSelectorAgent (demo) ===")
    print(
        "  API: ModelSelectorAgent(model=model).select(data_characteristics=..., task_type=...)"
    )
    ms_result = {}


# ══════════════════════════════════════════════════════════════════════
# TASK 4: ExperimentInterpreterAgent — interpret M3 results
# ══════════════════════════════════════════════════════════════════════

# Simulated experiment results from Module 3
experiment_results = {
    "LightGBM": {"auc_roc": 0.89, "auc_pr": 0.62, "brier": 0.08, "log_loss": 0.31},
    "XGBoost": {"auc_roc": 0.88, "auc_pr": 0.60, "brier": 0.09, "log_loss": 0.33},
    "CatBoost": {"auc_roc": 0.87, "auc_pr": 0.58, "brier": 0.10, "log_loss": 0.35},
}


async def step_4_experiment_interpreter():
    # TODO: Implement step_4_experiment_interpreter():
    #   1. Create ExperimentInterpreterAgent(model=model)
    #   2. Format experiment_results dict as a string (model: AUC-ROC, AUC-PR, Brier, LogLoss per line)
    #   3. Call agent.interpret(experiment_results=f"Model comparison results:\n{exp_str}", experiment_goal="Select best model for Singapore credit scoring (target metric: AUC-PR)", data_context="100k samples, 12% default rate, imbalanced binary classification")
    #   4. Print result; return
    ____
    ____
    ____
    ____
    ____


if _ML_AGENTS_AVAILABLE:
    ei_result = asyncio.run(step_4_experiment_interpreter())
else:
    print("\n=== Step 4: ExperimentInterpreterAgent (demo) ===")
    print(
        "  API: ExperimentInterpreterAgent(model=model).interpret(experiment_results=..., experiment_goal=...)"
    )
    ei_result = {}


# ══════════════════════════════════════════════════════════════════════
# TASK 5: DriftAnalyst + RetrainingDecision
# ══════════════════════════════════════════════════════════════════════

# Simulated drift report from Module 4
drift_report = {
    "model": "credit_default_lgbm",
    "overall_severity": "moderate",
    "features_drifted": 3,
    "max_psi": 0.18,
    "drifted_features": ["annual_income", "total_debt", "credit_utilisation"],
    "current_auc_pr": 0.55,
    "baseline_auc_pr": 0.62,
}


async def step_5_drift_and_retrain():
    # TODO: Implement step_5_drift_and_retrain():
    #   1. Create DriftAnalystAgent(model=model); format drift_report as string
    #   2. Call drift_agent.analyze(drift_report=f"Drift report:\n{drift_str}", domain_context="Credit model in production for 6 months, economic downturn in Singapore")
    #   3. Print drift_analysis; create RetrainingDecisionAgent(model=model)
    #   4. Compute perf_degradation = baseline_auc_pr - current_auc_pr
    #   5. Call retrain_agent.decide(drift_assessment=str(drift_analysis), current_performance=f"AUC-PR degraded by {perf_degradation:.2f}...", training_cost="Estimated retraining cost: $500. Model age: 180 days.", business_impact="Credit scoring model serves Singapore bank loan decisions.")
    #   6. Print retrain_decision; return (drift_analysis, retrain_decision)
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____


if _ML_AGENTS_AVAILABLE:
    drift_analysis, retrain_decision = asyncio.run(step_5_drift_and_retrain())
else:
    print("\n=== Step 5: DriftAnalystAgent + RetrainingDecisionAgent (demo) ===")
    print(
        "  API: DriftAnalystAgent(model=model).analyze(drift_report=..., domain_context=...)"
    )
    print(
        "  API: RetrainingDecisionAgent(model=model).decide(drift_assessment=..., current_performance=...)"
    )
    drift_analysis, retrain_decision = {}, {}


# ══════════════════════════════════════════════════════════════════════
# Summary: Full ML Agent Pipeline
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Full Pipeline Summary ===")
print(
    """
DataScientistAgent → FeatureEngineerAgent → ModelSelectorAgent
       ↓                    ↓                      ↓
  Data quality         Feature ideas          Model choice
       ↓                    ↓                      ↓
       └──────────→ Train + Evaluate ←─────────────┘
                         ↓
              ExperimentInterpreterAgent
                         ↓
                   Deploy to production
                         ↓
                 DriftAnalystAgent (monitor)
                         ↓
              RetrainingDecisionAgent (trigger)

Each agent has budget_usd — governance at every step.
The double opt-in pattern: agent=True in config + kailash-ml[agents] installed.
"""
)

print("✓ Exercise 6 complete — full ML agent pipeline with 6 agents")
