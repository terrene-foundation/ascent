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

feature_cols = [c for c in fraud_data.columns if c.startswith("v")]
X = fraud_data.select(feature_cols + ["amount"]).to_numpy()
y = fraud_data["is_fraud"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

overall_metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
    "auc_roc": roc_auc_score(y_test, y_proba),
}

print(f"\n=== Fraud Model Trained ===")
for metric, value in overall_metrics.items():
    print(f"  {metric}: {value:.4f}")


# TODO: Define ModelCardSignature using Signature with InputFields and OutputFields
# Hint: class ModelCardSignature(Signature): with InputField for model_name, version, type, date
#        and OutputField for intended_use, out_of_scope_uses, performance_summary,
#        per_group_metrics, limitations, ethical_considerations, regulatory_references
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
____
____
____
____
____
____
____


# TODO: Build model_card dict with all required fields
# Hint: include intended_use (fraud screening, not autonomous blocking),
#        regulatory_references (PDPA, MAS TRM, MAS Notice 655, EU AI Act),
#        limitations (class imbalance, PCA features), ethical_considerations
model_card = {
    "model_name": "credit_fraud_detector_v1",
    "model_version": "1.0.0",
    "model_type": "GradientBoostingClassifier",
    "training_date": datetime.now().strftime("%Y-%m-%d"),
    "intended_use": ____,
    "out_of_scope_uses": ____,
    "performance_summary": {k: f"{v:.4f}" for k, v in overall_metrics.items()},
    "limitations": ____,
    "ethical_considerations": ____,
    "regulatory_references": ____,
}

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

amount_test = fraud_data.select("amount").to_numpy().flatten()[: len(y_test)]
segment_labels = np.where(
    amount_test < np.percentile(amount_test, 33),
    "low_value",
    np.where(
        amount_test < np.percentile(amount_test, 66),
        "mid_value",
        "high_value",
    ),
)

segments = ["low_value", "mid_value", "high_value"]
bias_report = {}

# TODO: Compute per-segment metrics (accuracy, precision, recall, f1, positive_rate)
# Hint: for each segment, mask = segment_labels == segment
#        compute metrics only if len(np.unique(seg_y)) >= 2
#        include "positive_rate": float(seg_pred.mean()) for demographic parity check
for segment in segments:
    mask = segment_labels == segment
    if mask.sum() == 0:
        continue

    seg_y = y_test[mask]
    seg_pred = y_pred[mask]
    seg_proba = y_proba[mask]

    if len(np.unique(seg_y)) < 2:
        bias_report[segment] = {
            "n_samples": int(mask.sum()),
            "fraud_rate": float(seg_y.mean()),
            "note": "Single class - metrics undefined",
        }
        continue

    seg_metrics = {
        "n_samples": ____,
        "fraud_rate": ____,
        "accuracy": ____,
        "precision": ____,
        "recall": ____,
        "f1": ____,
        "positive_rate": ____,
    }
    bias_report[segment] = seg_metrics

print(f"\nPer-segment metrics:")
print(
    f"{'Segment':<15} {'N':>6} {'Fraud%':>8} {'Acc':>8} {'Prec':>8} "
    f"{'Recall':>8} {'F1':>8} {'Pos Rate':>10}"
)
print("-" * 80)

for segment, metrics in bias_report.items():
    if "note" in metrics:
        print(
            f"{segment:<15} {metrics['n_samples']:>6} {metrics['fraud_rate']:>8.4f} "
            f"{'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>10}"
        )
    else:
        print(
            f"{segment:<15} {metrics['n_samples']:>6} {metrics['fraud_rate']:>8.4f} "
            f"{metrics['accuracy']:>8.4f} {metrics['precision']:>8.4f} "
            f"{metrics['recall']:>8.4f} {metrics['f1']:>8.4f} "
            f"{metrics['positive_rate']:>10.4f}"
        )

valid_segments = {k: v for k, v in bias_report.items() if "positive_rate" in v}
if len(valid_segments) >= 2:
    pos_rates = [v["positive_rate"] for v in valid_segments.values()]
    max_gap = max(pos_rates) - min(pos_rates)
    print(f"\nDemographic parity gap: {max_gap:.4f}")
    print(f"  {'PASS' if max_gap <= 0.05 else 'FAIL'} (threshold: 0.05)")

    recall_values = [v["recall"] for v in valid_segments.values()]
    recall_gap = max(recall_values) - min(recall_values)
    print(f"\nEqualised odds (recall gap): {recall_gap:.4f}")
    print(f"  {'PASS' if recall_gap <= 0.05 else 'FAIL'} (threshold: 0.05)")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Generate DatasetDatasheet for sg_company_reports
# ══════════════════════════════════════════════════════════════════════

reports = loader.load("ascent10", "sg_company_reports.parquet")

print(f"\n{'=' * 70}")
print(f"=== Dataset Datasheet (Gebru et al., 2021) ===")
print(f"{'=' * 70}")

# TODO: Build datasheet dict with: motivation, composition, collection_process,
#        preprocessing, uses (intended + not_intended), distribution, maintenance
# Hint: composition includes instances, columns, data_types, missing_values, categories
datasheet = {
    "dataset_name": "sg_company_reports",
    "version": "1.0.0",
    "created_date": "2026-01-15",
    "motivation": ____,
    "composition": ____,
    "collection_process": ____,
    "preprocessing": ____,
    "uses": ____,
    "distribution": ____,
    "maintenance": ____,
}

print(f"\nDataset: {datasheet['dataset_name']} v{datasheet['version']}")
print(f"\nMotivation:")
for k, v in datasheet["motivation"].items():
    print(f"  {k}: {v}")
print(f"\nComposition:")
print(f"  Instances: {datasheet['composition']['instances']}")
print(f"  Columns: {datasheet['composition']['columns']}")
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

# TODO: Build system_card dict with:
#   - components: list of {name, type, risk_category, justification, mitigations}
#     Include: Fraud Detection Model (HIGH), PACT GovernanceEngine (N/A), Compliance Agent (LIMITED)
#   - regulatory_mapping: EU AI Act (Art 6-14), MAS TRM (7.2, 7.5, 8.1), PDPA, SG AI Verify
# Hint: EU AI Act HIGH applies to financial services AI per Annex III
system_card = {
    "system_name": "ASCENT Governed Fraud Detection System",
    "version": "1.0.0",
    "components": ____,
    "regulatory_mapping": ____,
}

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


# TODO: Set up DriftMonitor — needs ConnectionManager, then await monitor.set_reference()
# Hint: DriftMonitor(conn, psi_threshold=0.2); then set_reference(model_name=, reference_data=, feature_columns=)
async def _setup_monitor():
    conn = ConnectionManager("sqlite:///:memory:")
    await conn.initialize()
    mon = ____
    ref_df = pl.DataFrame(X_train[:2000], schema=feature_names)
    await mon.set_reference(
        model_name="fraud_detector_v1",
        reference_data=____,
        feature_columns=____,
    )
    return conn, mon


conn, monitor = asyncio.run(_setup_monitor())

rng = np.random.default_rng(42)

week1_data = X_test[:500]
week2_data = X_test[500:1000].copy()
week2_data[:, -1] *= 1.3
week3_data = X_test[1000:1500].copy()
week3_data[:, -1] *= 2.0
week3_data[:, 0] += rng.normal(0.5, 0.2, 500)

weeks = [
    ("Week 1 (stable)", week1_data),
    ("Week 2 (mild drift)", week2_data),
    ("Week 3 (significant drift)", week3_data),
]

# TODO: For each week, call monitor.check_drift(data) and print drift status + PSI scores
# Hint: report = monitor.check_drift(data); report.has_drift, report.feature_scores
for week_name, data in weeks:
    report = ____
    print(f"\n{week_name}:")
    print(f"  Drift detected: {report.has_drift}")

    if report.feature_scores:
        drifted = {k: v for k, v in report.feature_scores.items() if v > 0.1}
        if drifted:
            print(f"  Drifted features (PSI > 0.1):")
            for feat, psi in sorted(drifted.items(), key=lambda x: -x[1]):
                flag = " [ALERT]" if psi > 0.2 else " [WARNING]"
                print(f"    {feat}: PSI={psi:.4f}{flag}")
        else:
            print(f"  All features stable (PSI < 0.1)")

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
