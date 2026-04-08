# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT10 — Exercise 4: Model Cards, Datasheets, and Compliance Artefacts
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Generate structured compliance documentation -- model cards,
#   dataset datasheets, bias audits, and system cards -- using Kaizen
#   Signatures and DriftMonitor for post-deployment monitoring.
#
# TASKS:
#   1. Implement ModelCard generator using Signature
#   2. Run bias audit on fraud model — demographic parity, equalised odds
#   3. Generate DatasetDatasheet for sg_company_reports
#   4. Implement SystemCard for governed agent pipeline (EU AI Act mapping)
#   5. DriftMonitor post-deployment monitoring report
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import json
import os
import pickle
from datetime import datetime

import numpy as np
import polars as pl
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from kaizen import Signature, InputField, OutputField
from kaizen.core import Agent as BaseAgent
from kailash.db.connection import ConnectionManager
from kailash_ml.engines.drift_monitor import DriftMonitor

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Implement ModelCard generator using Signature
# ══════════════════════════════════════════════════════════════════════

loader = ASCENTDataLoader()
fraud_data = loader.load("ascent04", "credit_card_fraud.parquet")

print("=== Credit Card Fraud Dataset ===")
print(f"Shape: {fraud_data.shape}")
print(f"Columns: {fraud_data.columns}")
print(f"Fraud rate: {fraud_data['is_fraud'].mean():.4%}")

# TODO: Build the design matrix using V-features + amount and the is_fraud target.
feature_cols = ____
X = ____
y = ____

# TODO: Stratified 80/20 train/test split with random_state=42.
X_train, X_test, y_train, y_test = ____

# TODO: Train a GradientBoostingClassifier (n_estimators=30, max_depth=4).
model = ____
____

# TODO: Score the test set: predict + predict_proba[:, 1].
y_pred = ____
y_proba = ____

# TODO: Compute overall metrics dict (accuracy, precision, recall, f1, auc_roc).
overall_metrics = ____

print(f"\n=== Fraud Model Trained ===")
for metric, value in overall_metrics.items():
    print(f"  {metric}: {value:.4f}")


class ModelCardSignature(Signature):
    """Structured model card following Model Cards for Model Reporting (Mitchell et al., 2019)."""

    # TODO: Declare InputFields for model_name, model_version, model_type, training_date.
    # TODO: Declare OutputFields for intended_use, out_of_scope_uses, performance_summary,
    # TODO: per_group_metrics, limitations, ethical_considerations, regulatory_references.
    ____


# TODO: Build a model_card dict populating every field with text relevant to
# TODO: a Singapore credit fraud screening model. Cite PDPA, MAS TRM, MAS Notice 655,
# TODO: and EU AI Act in regulatory_references.
model_card = ____

print(f"\n=== Model Card ===")
for key, value in model_card.items():
    if isinstance(value, dict):
        print(f"\n{key}:")
        for k, v in value.items():
            print(f"  {k}: {v}")
    else:
        print(f"\n{key}:")
        print(f"  {value}")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Bias audit — demographic parity, equalised odds
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print(f"=== Bias Audit ===")
print(f"{'=' * 70}")

# TODO: Build segment_labels by bucketing test-set transaction amount into
# TODO: low_value (<33rd pct), mid_value (<66th), high_value (otherwise).
amount_test = ____
segment_labels = ____

# TODO: For each segment compute n, fraud_rate, accuracy, precision,
# TODO: recall, f1, positive_rate (use zero_division=0). Skip groups with
# TODO: fewer than 2 classes (note='Single class').
segments = ["low_value", "mid_value", "high_value"]
bias_report = {}

____

print(f"\nPer-segment metrics:")
print(
    f"{'Segment':<15} {'N':>6} {'Fraud%':>8} {'Acc':>8} {'Prec':>8} "
    f"{'Recall':>8} {'F1':>8} {'Pos Rate':>10}"
)
print("-" * 80)

# TODO: Print one row per segment from bias_report (handle the N/A case).
____

# TODO: Compute demographic parity gap = max(positive_rate) - min(positive_rate).
# TODO: Compute equalised odds gap on recall similarly. PASS at <= 0.05.
valid_segments = ____
if len(valid_segments) >= 2:
    pos_rates = ____
    max_gap = ____
    print(f"\nDemographic parity gap: {max_gap:.4f}")
    print(f"  {'PASS' if max_gap <= 0.05 else 'FAIL'} (threshold: 0.05)")

    recall_values = ____
    recall_gap = ____
    print(f"\nEqualised odds (recall gap): {recall_gap:.4f}")
    print(f"  {'PASS' if recall_gap <= 0.05 else 'FAIL'} (threshold: 0.05)")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Generate DatasetDatasheet for sg_company_reports
# ══════════════════════════════════════════════════════════════════════

reports = loader.load("ascent10", "sg_company_reports.parquet")

print(f"\n{'=' * 70}")
print(f"=== Dataset Datasheet (Gebru et al., 2021) ===")
print(f"{'=' * 70}")

# TODO: Build a datasheet dict following Gebru et al. (2021) structure:
# TODO:   - dataset_name, version, created_date
# TODO:   - motivation (purpose, creators, funding)
# TODO:   - composition (instances, columns, dtypes, missing, categories)
# TODO:   - collection_process, preprocessing, uses (intended/not_intended)
# TODO:   - distribution (license=Apache-2.0), maintenance
datasheet = ____

print(f"\nDataset: {datasheet['dataset_name']} v{datasheet['version']}")
print(f"\nMotivation:")
for k, v in datasheet["motivation"].items():
    print(f"  {k}: {v}")
print(f"\nComposition:")
print(f"  Instances: {datasheet['composition']['instances']}")
print(f"  Columns: {datasheet['composition']['columns']}")
print(f"  Categories: {datasheet['composition']['categories']}")
print(f"\nIntended uses:")
for use in datasheet["uses"]["intended"]:
    print(f"  - {use}")
print(f"\nNot intended for:")
for use in datasheet["uses"]["not_intended"]:
    print(f"  - {use}")
print(f"\nLicense: {datasheet['distribution']['license']}")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: SystemCard for governed agent pipeline (EU AI Act mapping)
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print(f"=== System Card (EU AI Act Risk Mapping) ===")
print(f"{'=' * 70}")

# TODO: Build a system_card dict listing components (Fraud Detection Model,
# TODO: PACT GovernanceEngine, Compliance Monitoring Agent) with EU AI Act
# TODO: risk categories (HIGH/N/A/LIMITED), justifications, mitigations.
# TODO: Include regulatory_mapping for EU AI Act, MAS TRM, PDPA, SG AI Verify.
system_card = ____

print(f"\nSystem: {system_card['system_name']} v{system_card['version']}")
print(f"\nComponents and EU AI Act risk categories:")
for comp in system_card["components"]:
    print(f"\n  {comp['name']} [{comp['risk_category']}]")
    print(f"    Type: {comp['type']}")
    print(f"    Justification: {comp['justification']}")
    print(f"    Mitigations:")
    for m in comp["mitigations"]:
        print(f"      - {m}")

print(f"\nRegulatory mapping:")
for regulation, articles in system_card["regulatory_mapping"].items():
    print(f"\n  {regulation}:")
    for article, description in articles.items():
        print(f"    {article}: {description}")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: DriftMonitor post-deployment monitoring report
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print(f"=== Post-Deployment Monitoring (DriftMonitor) ===")
print(f"{'=' * 70}")

feature_names = feature_cols + ["amount"]


async def _setup_monitor():
    # TODO: Build a ConnectionManager(":memory:"), initialise it,
    # TODO: instantiate DriftMonitor(conn, psi_threshold=0.2),
    # TODO: convert X_train[:2000] to a polars DataFrame and call
    # TODO: monitor.set_reference(model_name, reference_data, feature_columns).
    ____


conn, monitor = asyncio.run(_setup_monitor())

# TODO: Simulate 3 weeks of production data: stable (X_test[:500]),
# TODO: mild drift (amount * 1.3), significant drift (amount * 2.0 + V1 shift).
rng = np.random.default_rng(42)

week1_data = ____
week2_data = ____
____
week3_data = ____
____
____

weeks = [
    ("Week 1 (stable)", week1_data),
    ("Week 2 (mild drift)", week2_data),
    ("Week 3 (significant drift)", week3_data),
]

# TODO: For each week, call monitor.check_drift, print has_drift,
# TODO: and list any feature with PSI > 0.1 with [WARNING] / [ALERT] tags.
____

print(f"\nMonitoring thresholds:")
print(f"  PSI < 0.1:  Stable (no action)")
print(f"  0.1 <= PSI < 0.2:  Warning (investigate)")
print(f"  PSI >= 0.2:  Alert (retrain or review)")
print(f"\nRecommended actions for Week 3 drift:")
print(f"  1. Investigate root cause (transaction amount distribution shift)")
print(f"  2. Compare model performance on drifted vs reference data")
print(f"  3. Retrain if performance degradation exceeds 5% on key metrics")
print(f"  4. Update model card with new evaluation results")
print(f"  5. File incident report in governance audit trail")

print(
    "\n--- Exercise 4 complete: model cards, bias audit, datasheets, system cards, drift ---"
)
