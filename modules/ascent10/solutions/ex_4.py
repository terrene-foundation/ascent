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

# Train a fraud detection model for evaluation
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


# Define ModelCard Signature
class ModelCardSignature(Signature):
    """Structured model card following Model Cards for Model Reporting (Mitchell et al., 2019)."""

    # Inputs
    model_name: str = InputField(desc="Name of the model")
    model_version: str = InputField(desc="Version identifier")
    model_type: str = InputField(desc="Type of model (e.g., classifier, regressor)")
    training_date: str = InputField(desc="Date model was trained")

    # Outputs
    intended_use: str = OutputField(desc="Primary intended use cases")
    out_of_scope_uses: str = OutputField(desc="Uses the model should NOT be used for")
    performance_summary: str = OutputField(desc="Overall performance metrics summary")
    per_group_metrics: str = OutputField(
        desc="Performance broken down by demographic group"
    )
    limitations: str = OutputField(desc="Known limitations and failure modes")
    ethical_considerations: str = OutputField(desc="Ethical risks and mitigations")
    regulatory_references: str = OutputField(
        desc="Applicable regulations (PDPA, MAS TRM, EU AI Act)"
    )


# Generate model card programmatically (no LLM needed for structured data)
model_card = {
    "model_name": "credit_fraud_detector_v1",
    "model_version": "1.0.0",
    "model_type": "GradientBoostingClassifier",
    "training_date": datetime.now().strftime("%Y-%m-%d"),
    "intended_use": (
        "Fraud detection for credit card transactions in Singapore banking. "
        "Designed as a first-pass screening tool to flag suspicious transactions "
        "for human review. Not intended for autonomous decision-making."
    ),
    "out_of_scope_uses": (
        "- Autonomous transaction blocking without human review\n"
        "- Credit scoring or loan approval decisions\n"
        "- Customer profiling for marketing purposes\n"
        "- Use outside Singapore regulatory jurisdiction"
    ),
    "performance_summary": {k: f"{v:.4f}" for k, v in overall_metrics.items()},
    "limitations": (
        "- Trained on synthetic/anonymised data; real-world distribution may differ\n"
        "- V1-V28 features are PCA-transformed; original feature semantics lost\n"
        "- Temporal patterns (time_seconds) may not generalise across seasons\n"
        "- Class imbalance ({:.2%} fraud) may cause false positive fatigue"
    ).format(y.mean()),
    "ethical_considerations": (
        "- False positives may disproportionately affect certain demographics\n"
        "- Model should not be the sole basis for blocking transactions\n"
        "- Explainability limited due to PCA-transformed features\n"
        "- Regular bias audits required per MAS Notice 655"
    ),
    "regulatory_references": (
        "- PDPA (Personal Data Protection Act): data minimisation applies\n"
        "- MAS TRM (Technology Risk Management): model risk management\n"
        "- MAS Notice 655: AI/ML in financial services\n"
        "- EU AI Act: high-risk AI system (credit/financial domain)"
    ),
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

# Create demographic segments based on transaction amount quantiles
# (proxy for customer segments since we lack demographic columns)
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

for segment in segments:
    mask = segment_labels == segment
    if mask.sum() == 0:
        continue

    seg_y = y_test[mask]
    seg_pred = y_pred[mask]
    seg_proba = y_proba[mask]

    # Skip if only one class present
    if len(np.unique(seg_y)) < 2:
        bias_report[segment] = {
            "n_samples": int(mask.sum()),
            "fraud_rate": float(seg_y.mean()),
            "note": "Single class - metrics undefined",
        }
        continue

    seg_metrics = {
        "n_samples": int(mask.sum()),
        "fraud_rate": float(seg_y.mean()),
        "accuracy": accuracy_score(seg_y, seg_pred),
        "precision": precision_score(seg_y, seg_pred, zero_division=0),
        "recall": recall_score(seg_y, seg_pred, zero_division=0),
        "f1": f1_score(seg_y, seg_pred, zero_division=0),
        "positive_rate": float(seg_pred.mean()),  # For demographic parity
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

# Demographic parity check: positive prediction rate should be similar across groups
valid_segments = {k: v for k, v in bias_report.items() if "positive_rate" in v}
if len(valid_segments) >= 2:
    pos_rates = [v["positive_rate"] for v in valid_segments.values()]
    max_gap = max(pos_rates) - min(pos_rates)
    print(f"\nDemographic parity gap: {max_gap:.4f}")
    print(f"  {'PASS' if max_gap <= 0.05 else 'FAIL'} (threshold: 0.05)")

    # Equalised odds: TPR and FPR should be similar across groups
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

datasheet = {
    "dataset_name": "sg_company_reports",
    "version": "1.0.0",
    "created_date": "2026-01-15",
    "motivation": {
        "purpose": "NLP training and evaluation for Singapore financial document analysis",
        "creators": "Terrene Foundation (open-source educational use)",
        "funding": "Terrene Foundation (Singapore CLG, non-profit)",
    },
    "composition": {
        "instances": reports.height,
        "columns": reports.columns,
        "data_types": {col: str(reports[col].dtype) for col in reports.columns},
        "missing_values": {col: reports[col].null_count() for col in reports.columns},
        "categories": (
            reports["report_type"].unique().to_list()
            if "report_type" in reports.columns
            else []
        ),
    },
    "collection_process": {
        "method": "Synthetic generation based on real Singapore corporate report structures",
        "consent": "N/A (synthetic data, no real individuals)",
        "time_period": "Simulated FY2020-2025",
    },
    "preprocessing": {
        "steps": [
            "Text normalisation (unicode, whitespace)",
            "Company name anonymisation",
            "Report type classification",
        ],
    },
    "uses": {
        "intended": [
            "NLP model training (text classification, NER, summarisation)",
            "Educational exercises (ASCENT M8-M10)",
            "Benchmarking Singapore-domain NLP models",
        ],
        "not_intended": [
            "Financial advice or investment decisions",
            "Real company analysis (data is synthetic)",
            "Training production models without validation on real data",
        ],
    },
    "distribution": {
        "license": "Apache-2.0",
        "access": "Open (via ASCENTDataLoader)",
    },
    "maintenance": {
        "owner": "Terrene Foundation",
        "update_frequency": "Per ASCENT curriculum release",
        "contact": "https://github.com/terrene-foundation",
    },
}

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

system_card = {
    "system_name": "ASCENT Governed Fraud Detection System",
    "version": "1.0.0",
    "components": [
        {
            "name": "Fraud Detection Model",
            "type": "ML Model (GradientBoostingClassifier)",
            "risk_category": "HIGH",
            "justification": "Financial services AI per EU AI Act Annex III",
            "mitigations": [
                "Human-in-the-loop review for all flagged transactions",
                "Model card with per-group metrics",
                "Regular bias audits (quarterly)",
                "DriftMonitor for distribution shift detection",
            ],
        },
        {
            "name": "PACT GovernanceEngine",
            "type": "Governance Framework",
            "risk_category": "N/A (infrastructure)",
            "justification": "Governance layer, not a decision-making AI",
            "mitigations": [
                "Fail-closed design (deny by default)",
                "Immutable audit trail",
                "Frozen GovernanceContext (no self-modification)",
            ],
        },
        {
            "name": "Compliance Monitoring Agent",
            "type": "AI Agent (PactGovernedAgent)",
            "risk_category": "LIMITED",
            "justification": "Monitoring/advisory role, no autonomous decisions",
            "mitigations": [
                "Operating envelope restricts to read-only actions",
                "Clearance-based data access",
                "All actions audited via GovernanceEngine",
            ],
        },
    ],
    "regulatory_mapping": {
        "EU AI Act": {
            "Art 6-7": "High-risk classification applied to fraud model",
            "Art 9": "Risk management via PACT envelopes + DriftMonitor",
            "Art 10": "Data governance via DatasetDatasheet + bias audit",
            "Art 11": "Technical documentation via ModelCard + SystemCard",
            "Art 12": "Record-keeping via GovernanceEngine audit trail",
            "Art 13": "Transparency via model card + system card",
            "Art 14": "Human oversight via human-in-the-loop design",
        },
        "MAS TRM": {
            "7.2": "Technology risk management framework (PACT governance)",
            "7.5": "Audit trail requirements (GovernanceEngine logging)",
            "8.1": "System reliability (DriftMonitor + automated alerts)",
        },
        "PDPA": {
            "Part IV": "Data minimisation (PCA-transformed features)",
            "Part V": "Access limitation (PACT clearance levels)",
        },
        "SG AI Verify": {
            "Accountability": "D/T/R chains trace to human delegator",
            "Fairness": "Bias audit with demographic parity checks",
            "Transparency": "Model card + system card published",
        },
    },
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

# Set up DriftMonitor with training data as reference
feature_names = feature_cols + ["amount"]


async def _setup_monitor():
    conn = ConnectionManager("sqlite:///:memory:")
    await conn.initialize()
    mon = DriftMonitor(conn, psi_threshold=0.2)
    ref_df = pl.DataFrame(X_train[:2000], schema=feature_names)
    await mon.set_reference(
        model_name="fraud_detector_v1",
        reference_data=ref_df,
        feature_columns=feature_names,
    )
    return conn, mon


conn, monitor = asyncio.run(_setup_monitor())

# Simulate production traffic with gradual drift
rng = np.random.default_rng(42)

# Week 1: no drift
week1_data = X_test[:500]
# Week 2: slight drift in amount feature
week2_data = X_test[500:1000].copy()
week2_data[:, -1] *= 1.3  # 30% increase in transaction amounts
# Week 3: significant drift
week3_data = X_test[1000:1500].copy()
week3_data[:, -1] *= 2.0  # 100% increase in amounts
week3_data[:, 0] += rng.normal(0.5, 0.2, 500)  # Shift V1

weeks = [
    ("Week 1 (stable)", week1_data),
    ("Week 2 (mild drift)", week2_data),
    ("Week 3 (significant drift)", week3_data),
]

for week_name, data in weeks:
    report = monitor.check_drift(data)
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

# Monitoring summary
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
