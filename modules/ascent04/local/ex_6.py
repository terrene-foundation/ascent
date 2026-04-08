# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT04 — Exercise 6: Drift Monitoring
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Deploy M3's credit model, simulate drift, detect with PSI
#   and KS tests. Frame as governance: you must prove your model still
#   performs in production.
#
# TASKS:
#   1. Establish reference distribution from training data
#   2. Configure DriftMonitor with DriftSpec thresholds
#   3. Simulate realistic data drift (feature shift, concept drift)
#   4. Detect drift with PSI and KS statistics
#   5. Analyse drift severity and affected features
#   6. Governance discussion: monitoring obligations
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

import numpy as np
import polars as pl
from kailash.db.connection import ConnectionManager
from kailash_ml import PreprocessingPipeline, ModelVisualizer
from kailash_ml.engines.drift_monitor import DriftMonitor, DriftSpec
from kailash_ml.interop import to_sklearn_input

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()


# ── Data Loading ──────────────────────────────────────────────────────

loader = ASCENTDataLoader()
credit = loader.load("ascent03", "sg_credit_scoring.parquet")

pipeline = PreprocessingPipeline()
result = pipeline.setup(
    credit, target="default", seed=42, normalize=False, categorical_encoding="ordinal"
)

feature_cols = [c for c in result.train_data.columns if c != "default"]
print(f"=== Credit Scoring Data ===")
print(f"Features: {len(feature_cols)}")
print(f"Train: {result.train_data.shape}, Test: {result.test_data.shape}")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Establish reference distribution
# ══════════════════════════════════════════════════════════════════════

# TODO: Reference = training data restricted to feature columns
# Hint: result.train_data.select(feature_cols)
reference_data = ____
print(f"\nReference distribution: {reference_data.shape}")
print("This represents the data the model was trained on.")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Configure DriftMonitor
# ══════════════════════════════════════════════════════════════════════


async def setup_monitoring():
    conn = ConnectionManager("sqlite:///ascent04_drift.db")
    await conn.initialize()

    # TODO: Build DriftMonitor with psi_threshold=0.1, ks_threshold=0.05,
    #       performance_threshold=0.1
    monitor = ____

    # TODO: Set the reference distribution for "credit_default_lgbm"
    # Hint: await monitor.set_reference(model_name=..., reference_data=...,
    #         feature_columns=feature_cols)
    ____

    print(f"\n=== DriftMonitor Configured ===")
    print(f"PSI threshold: 0.1 (>0.2 = severe)")
    print(f"KS threshold: 0.05")
    print(f"Model: credit_default_lgbm")

    return conn, monitor


conn, monitor = asyncio.run(setup_monitoring())


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Simulate realistic data drift
# ══════════════════════════════════════════════════════════════════════
# Real drift scenarios:
# a) Feature drift: income distribution shifts (economic downturn)
# b) Concept drift: same features, different default relationship
# c) Gradual drift: slow shift over months

rng = np.random.default_rng(42)

# Scenario A: No drift (control)
# TODO: Just take feature columns from the test set
no_drift = ____

# Scenario B: Moderate drift (income drops 20%, debt rises 15%)
# TODO: Use with_columns to multiply annual_income by 0.8 and total_debt by 1.15
# Hint: result.test_data.select(feature_cols).with_columns(
#         (pl.col("annual_income") * 0.8).alias("annual_income")
#             if "annual_income" in feature_cols else pl.lit(0).alias("_fallback_income"),
#         (pl.col("total_debt") * 1.15).alias("total_debt")
#             if "total_debt" in feature_cols else pl.lit(0).alias("_fallback_debt"),
#     )
moderate_drift = ____

# Scenario C: Severe drift (multiple features shifted)
severe_drift_cols = []
for col in feature_cols:
    dtype = result.test_data[col].dtype
    if dtype in (pl.Float64, pl.Float32):
        # TODO: Random shift in [-0.3, 0.3] and random scale in [0.8, 1.5]
        shift = ____
        scale = ____
        # TODO: Build expression: pl.col(col) * scale + pl.col(col).mean() * shift
        severe_drift_cols.append(____)
    else:
        severe_drift_cols.append(pl.col(col))

# TODO: Apply the column expressions to test data to create the severe drift df
severe_drift = ____

print(f"\n=== Simulated Drift Scenarios ===")
print(f"A) No drift: test data from same distribution")
print(f"B) Moderate: income -20%, debt +15%")
print(f"C) Severe: multiple features shifted (crisis simulation)")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Detect drift
# ══════════════════════════════════════════════════════════════════════


async def check_all_scenarios():
    scenarios = [
        ("No Drift", no_drift),
        ("Moderate Drift", moderate_drift),
        ("Severe Drift", severe_drift),
    ]

    all_reports = {}
    for name, data in scenarios:
        # TODO: Call monitor.check_drift for credit_default_lgbm with current_data=data
        report = ____

        all_reports[name] = report

        print(f"\n=== {name} ===")
        print(f"Overall drift: {report.overall_drift_detected}")
        print(f"Severity: {report.overall_severity}")

        # TODO: Filter feature_results for r.drift_detected == True
        drifted = ____
        print(f"Features with drift: {len(drifted)} / {len(report.feature_results)}")

        if drifted:
            print(
                f"\n  {'Feature':<25} {'PSI':>8} {'KS stat':>8} {'KS p':>10} {'Type':>10}"
            )
            print("  " + "─" * 65)
            for r in sorted(drifted, key=lambda x: x.psi, reverse=True)[:10]:
                print(
                    f"  {r.feature_name:<25} {r.psi:>8.4f} {r.ks_statistic:>8.4f} "
                    f"{r.ks_pvalue:>10.6f} {r.drift_type:>10}"
                )

    return all_reports


all_reports = asyncio.run(check_all_scenarios())


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Analyse drift severity
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Drift Severity Analysis ===")
print(f"\nPSI Interpretation:")
print(f"  < 0.1  : No significant drift")
print(f"  0.1-0.2: Moderate drift — investigate")
print(f"  > 0.2  : Severe drift — retrain model")

for name, report in all_reports.items():
    # TODO: Collect r.psi for every entry in report.feature_results
    psi_values = ____
    print(f"\n{name}:")
    print(f"  Mean PSI: {np.mean(psi_values):.4f}")
    print(f"  Max PSI:  {max(psi_values):.4f}")
    print(f"  Features above 0.1: {sum(1 for p in psi_values if p > 0.1)}")
    print(f"  Features above 0.2: {sum(1 for p in psi_values if p > 0.2)}")


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Governance — monitoring as obligation
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Governance: Model Monitoring Obligations ===")
print(
    """
In production, you MUST prove your model still performs.
DriftMonitor is how you meet this obligation.

1. REGULATORY: MAS AI guidelines require ongoing model validation
2. OPERATIONAL: Drift detection → retrain trigger → ModelRegistry update
3. AUDIT TRAIL: Every drift check is logged (get_drift_history)
4. ALERTING: DriftSpec.on_drift_detected for automated notifications
5. MODULE 6: PACT GovernanceEngine formalises these obligations

Pipeline: DriftMonitor → alert → retrain (M3 pipeline) → promote (ModelRegistry)
"""
)


async def show_history():
    # TODO: Get last 10 drift checks via monitor.get_drift_history
    history = ____
    print(f"Drift check history: {len(history)} entries")
    for h in history[:5]:
        print(f"  {h.get('checked_at', '?')}: severity={h.get('severity', '?')}")
    await conn.close()


asyncio.run(show_history())

viz = ModelVisualizer()
# TODO: Build a comparison dict mapping each scenario to its Mean_PSI, Max_PSI,
# and number of drifted features
# Hint: {name: {"Mean_PSI": np.mean([r.psi for r in report.feature_results]),
#               "Max_PSI": max(r.psi for r in report.feature_results),
#               "Drifted_Features": sum(1 for r in report.feature_results if r.drift_detected)}
#        for name, report in all_reports.items()}
comparison = ____
fig = viz.metric_comparison(comparison)
fig.update_layout(title="Drift Detection: Scenario Comparison")
fig.write_html("ex4_drift_comparison.html")
print("\nSaved: ex4_drift_comparison.html")

print("\n✓ Exercise 6 complete — DriftMonitor for production model monitoring")
