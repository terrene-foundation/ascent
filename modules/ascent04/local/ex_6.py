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
# TODO: Run pipeline.setup() with target="default", seed=42, normalize=False,
# categorical_encoding="ordinal"
result = ____  # Hint: pipeline.setup(credit, target="default", seed=42, normalize=False, categorical_encoding="ordinal")

# TODO: Extract feature columns (all columns except "default")
feature_cols = ____  # Hint: [c for c in result.train_data.columns if c != "default"]
print(f"=== Credit Scoring Data ===")
print(f"Features: {len(feature_cols)}")
print(f"Train: {result.train_data.shape}, Test: {result.test_data.shape}")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Establish reference distribution
# ══════════════════════════════════════════════════════════════════════

# TODO: Extract the reference feature distribution from the training data
reference_data = ____  # Hint: result.train_data.select(feature_cols)
print(f"\nReference distribution: {reference_data.shape}")
print("This represents the data the model was trained on.")
print("Any significant deviation from this distribution = potential drift.")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Configure DriftMonitor
# ══════════════════════════════════════════════════════════════════════


async def setup_monitoring():
    # TODO: Create ConnectionManager for SQLite and initialize it
    conn = ____  # Hint: ConnectionManager("sqlite:///ascent04_drift.db")
    ____  # Hint: await conn.initialize()

    # TODO: Create DriftMonitor with psi_threshold=0.1, ks_threshold=0.05,
    # performance_threshold=0.1
    monitor = ____  # Hint: DriftMonitor(conn, psi_threshold=0.1, ks_threshold=0.05, performance_threshold=0.1)

    # TODO: Set the reference distribution for the model using monitor.set_reference()
    # model_name="credit_default_lgbm", reference_data=reference_data, feature_columns=feature_cols
    await ____  # Hint: monitor.set_reference(model_name="credit_default_lgbm", reference_data=reference_data, feature_columns=feature_cols)

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

# Scenario A: No drift (control — test data from same distribution)
# TODO: Select only feature_cols from the test data for the no-drift baseline
no_drift = ____  # Hint: result.test_data.select(feature_cols)

# TODO: Scenario B — moderate feature drift: income drops 20%, debt rises 15%
# Use with_columns to scale annual_income * 0.8 and total_debt * 1.15
moderate_drift = result.test_data.select(feature_cols).with_columns(
    (
        ____  # Hint: (pl.col("annual_income") * 0.8).alias("annual_income") if "annual_income" in feature_cols else pl.lit(0).alias("_placeholder_b")
    ),
    (
        ____  # Hint: (pl.col("total_debt") * 1.15).alias("total_debt") if "total_debt" in feature_cols else pl.lit(0).alias("_placeholder_b2")
    ),
)

# TODO: Scenario C — severe drift: apply random scale/shift to each float column
severe_drift_cols = []
for col in feature_cols:
    dtype = result.test_data[col].dtype
    if dtype in (pl.Float64, pl.Float32):
        shift = rng.uniform(-0.3, 0.3)
        scale = rng.uniform(0.8, 1.5)
        severe_drift_cols.append(
            ____  # Hint: (pl.col(col) * scale + pl.col(col).mean() * shift).alias(col)
        )
    else:
        severe_drift_cols.append(pl.col(col))

# TODO: Apply the severe_drift_cols transformations using with_columns
severe_drift = (
    ____  # Hint: result.test_data.select(feature_cols).with_columns(severe_drift_cols)
)

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
        # TODO: Call monitor.check_drift() with the model name and current data
        report = (
            await ____
        )  # Hint: monitor.check_drift(model_name="credit_default_lgbm", current_data=data)

        all_reports[name] = report

        print(f"\n=== {name} ===")
        print(f"Overall drift: {report.overall_drift_detected}")
        print(f"Severity: {report.overall_severity}")

        drifted = [r for r in report.feature_results if r.drift_detected]
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
    # TODO: Extract PSI values from all feature_results, compute mean and max
    psi_values = ____  # Hint: [r.psi for r in report.feature_results]
    print(f"\n{name}:")
    print(f"  Mean PSI: {np.mean(psi_values):.4f}")
    print(f"  Max PSI:  {max(psi_values):.4f}")
    # TODO: Count features where PSI > 0.1 and PSI > 0.2
    print(
        f"  Features above 0.1: {____}"
    )  # Hint: sum(1 for p in psi_values if p > 0.1)
    print(
        f"  Features above 0.2: {____}"
    )  # Hint: sum(1 for p in psi_values if p > 0.2)


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


# TODO: Retrieve and display drift check history for the credit model
async def show_history():
    # Hint: monitor.get_drift_history("credit_default_lgbm", limit=10)
    history = (
        await ____
    )  # Hint: monitor.get_drift_history("credit_default_lgbm", limit=10)
    print(f"Drift check history: {len(history)} entries")
    for h in history[:5]:
        print(f"  {h.get('checked_at', '?')}: severity={h.get('severity', '?')}")
    await conn.close()


asyncio.run(show_history())

viz = ModelVisualizer()
# TODO: Build comparison chart of mean PSI, max PSI, and drifted features per scenario
comparison = {
    name: {
        "Mean_PSI": ____,  # Hint: np.mean([r.psi for r in report.feature_results])
        "Max_PSI": ____,  # Hint: max(r.psi for r in report.feature_results)
        "Drifted_Features": ____,  # Hint: sum(1 for r in report.feature_results if r.drift_detected)
    }
    for name, report in all_reports.items()
}
fig = ____  # Hint: viz.metric_comparison(comparison)
fig.update_layout(title="Drift Detection: Scenario Comparison")
fig.write_html("ex6_drift_comparison.html")
print("\nSaved: ex6_drift_comparison.html")

print("\n✓ Exercise 6 complete — DriftMonitor for production model monitoring")
