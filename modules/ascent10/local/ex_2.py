# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT10 — Exercise 2: Federated Learning with Privacy Guarantees
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Implement federated averaging (FedAvg) from scratch with
#   differential privacy, integrate with PACT governance for client
#   isolation, and deploy via InferenceServer with DriftMonitor.
#
# TASKS:
#   1. Implement FedAvg with 3 virtual hospital clients
#   2. Add Gaussian noise DP mechanism with moments accountant
#   3. Privacy-utility trade-off at varying epsilon values
#   4. Integrate with PACT: each client as PactGovernedAgent
#   5. Deploy federated model via InferenceServer + DriftMonitor
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os
import pickle
import tempfile
import uuid
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from kailash.trust import ConfidentialityLevel, TrustPosture
from kailash.db.connection import ConnectionManager
from kailash_ml.engines.model_registry import ModelRegistry
from kailash_ml.engines.inference_server import InferenceServer
from kailash_ml.engines.drift_monitor import DriftMonitor
from kailash_ml.types import FeatureSchema, FeatureField, ModelSignature, MetricSpec
from pact import GovernanceEngine, PactGovernedAgent, compile_org, load_org_yaml
from pact.governance import RoleClearance, RoleEnvelope, KnowledgeItem
from kailash.trust.pact.config import (
    ConstraintEnvelopeConfig,
    OperationalConstraintConfig,
    TemporalConstraintConfig,
    DataAccessConstraintConfig,
    CommunicationConstraintConfig,
)

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Implement FedAvg with 3 virtual hospital clients
# ══════════════════════════════════════════════════════════════════════

loader = ASCENTDataLoader()
fraud_raw = loader.load("ascent04", "credit_card_fraud.parquet")

# Sample 2000 rows for speed; keep class balance
fraud_fraud = fraud_raw.filter(pl.col("is_fraud") == 1).head(200)
fraud_legit = fraud_raw.filter(pl.col("is_fraud") == 0).head(1800)
patients = pl.concat([fraud_fraud, fraud_legit]).sample(fraction=1.0, seed=42, shuffle=True)

print("=== Credit Card Fraud Dataset (sampled for federated learning) ===")
print(f"Shape: {patients.shape}")
print(f"Fraud rate: {patients['is_fraud'].mean():.4%}")

feature_names = [c for c in patients.columns if c.startswith("v")] + ["amount"]
X = patients.select(feature_names).to_numpy()
y = patients["is_fraud"].to_numpy()
n = len(y)

# Partition into 3 virtual clients (~1000 each, non-IID by amount)
rng = np.random.default_rng(42)
indices = np.argsort(patients["amount"].to_numpy())
n_per_client = n // 3

client_indices = {
    "hospital_a": indices[:n_per_client],
    "hospital_b": indices[n_per_client : 2 * n_per_client],
    "hospital_c": indices[2 * n_per_client :],
}

client_data = {}
for name, idx in client_indices.items():
    client_data[name] = (X[idx], y[idx])
    print(f"  {name}: {len(idx)} samples, target_rate={y[idx].mean():.4f}")


def train_local_model(X_local: np.ndarray, y_local: np.ndarray) -> np.ndarray:
    """Train a local logistic regression and return weight vector."""
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_local, y_local)
    return np.concatenate([model.coef_.flatten(), model.intercept_])


# TODO: Implement fedavg — weighted mean of client model parameters
# Hint: total_samples = sum(n for _, n in client_weights); then sum weights*(n/total)
def fedavg(client_weights: list[tuple[np.ndarray, int]]) -> np.ndarray:
    """Federated averaging: weighted mean of client model parameters."""
    ____
    ____
    ____
    ____


def set_model_weights(model: LogisticRegression, weights: np.ndarray) -> None:
    """Set logistic regression weights from a flat vector."""
    n_features = len(weights) - 1
    model.coef_ = weights[:n_features].reshape(1, -1)
    model.intercept_ = weights[n_features:]
    model.classes_ = np.array([0, 1])


n_rounds = 5
print(f"\n=== FedAvg Training ({n_rounds} rounds) ===")

global_weights = None
for round_num in range(1, n_rounds + 1):
    round_client_weights = []
    for name, (X_c, y_c) in client_data.items():
        if global_weights is not None:
            local_model = LogisticRegression(max_iter=200, random_state=42)
            local_model.fit(X_c, y_c)
            set_model_weights(local_model, global_weights)
            local_model.fit(X_c, y_c)
            local_w = np.concatenate(
                [local_model.coef_.flatten(), local_model.intercept_]
            )
        else:
            local_w = train_local_model(X_c, y_c)
        round_client_weights.append((local_w, len(y_c)))

    global_weights = fedavg(round_client_weights)

    global_model = LogisticRegression()
    set_model_weights(global_model, global_weights)
    y_pred = global_model.predict(X)
    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, global_model.predict_proba(X)[:, 1])
    print(f"  Round {round_num}: accuracy={acc:.4f}, AUC={auc:.4f}")

centralized = LogisticRegression(max_iter=200, random_state=42)
centralized.fit(X, y)
cent_acc = accuracy_score(y, centralized.predict(X))
cent_auc = roc_auc_score(y, centralized.predict_proba(X)[:, 1])
print(f"\n  Centralized baseline: accuracy={cent_acc:.4f}, AUC={cent_auc:.4f}")
print(
    f"  FedAvg gap: accuracy={abs(acc - cent_acc):.4f}, AUC={abs(auc - cent_auc):.4f}"
)


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Gaussian noise DP mechanism with moments accountant
# ══════════════════════════════════════════════════════════════════════


def clip_gradients(weights: np.ndarray, max_norm: float) -> np.ndarray:
    """Clip weight update to sensitivity bound S (L2 norm)."""
    norm = np.linalg.norm(weights)
    if norm > max_norm:
        return weights * (max_norm / norm)
    return weights


# TODO: Implement add_gaussian_noise
# Hint: noise = np.random.normal(0, sigma * sensitivity, size=weights.shape); return weights + noise
def add_gaussian_noise(
    weights: np.ndarray, sensitivity: float, sigma: float
) -> np.ndarray:
    """Add calibrated Gaussian noise for (epsilon, delta)-DP."""
    ____
    ____


# TODO: Implement moments_accountant
# Hint: q = batch_size / n_samples
#       eps = sqrt(2 * n_rounds * log(1/delta)) * (q / sigma) + n_rounds * q^2 / (2 * sigma^2)
def moments_accountant(
    n_rounds: int,
    n_samples: int,
    batch_size: int,
    sigma: float,
    delta: float = 1e-5,
) -> float:
    """Simple moments accountant for tracking privacy budget epsilon."""
    ____
    ____
    ____


# TODO: Implement fedavg_dp — FedAvg with gradient clipping + Gaussian noise aggregation
# Hint: for each round: train local, clip update if global_weights exists, fedavg, add_gaussian_noise
# Then compute epsilon via moments_accountant; return (global_weights, epsilon)
def fedavg_dp(
    client_data: dict,
    n_rounds: int,
    clip_norm: float,
    sigma: float,
) -> tuple[np.ndarray, float]:
    """FedAvg with differential privacy (gradient clipping + Gaussian noise)."""
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


print(f"\n=== Differential Privacy Mechanism ===")
print(f"Components:")
print(f"  1. Gradient clipping: L2 norm bound = sensitivity S")
print(f"  2. Gaussian noise: N(0, sigma^2 * S^2) added to aggregated weights")
print(f"  3. Moments accountant: tracks cumulative epsilon across rounds")

dp_weights, eps = fedavg_dp(client_data, n_rounds=5, clip_norm=1.0, sigma=1.0)
dp_model = LogisticRegression()
set_model_weights(dp_model, dp_weights)
dp_acc = accuracy_score(y, dp_model.predict(X))
dp_auc = roc_auc_score(y, dp_model.predict_proba(X)[:, 1])
print(
    f"\nDP-FedAvg (sigma=1.0): accuracy={dp_acc:.4f}, AUC={dp_auc:.4f}, epsilon={eps:.4f}"
)


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Privacy-utility trade-off at varying epsilon
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Privacy-Utility Trade-off ===")
print(f"{'Sigma':<10} {'Epsilon':<12} {'Accuracy':<12} {'AUC':<12}")
print("-" * 46)

sigma_values = [0.1, 0.5, 1.0, 2.0, 5.0]
results = []
for sigma in sigma_values:
    w, eps = fedavg_dp(client_data, n_rounds=5, clip_norm=1.0, sigma=sigma)
    m = LogisticRegression()
    set_model_weights(m, w)
    a = accuracy_score(y, m.predict(X))
    au = roc_auc_score(y, m.predict_proba(X)[:, 1])
    results.append({"sigma": sigma, "epsilon": eps, "accuracy": a, "auc": au})
    print(f"{sigma:<10.1f} {eps:<12.4f} {a:<12.4f} {au:<12.4f}")

print(f"\nInterpretation:")
print(f"  Low sigma (0.1) -> high epsilon (weak privacy, high utility)")
print(f"  High sigma (5.0) -> low epsilon (strong privacy, lower utility)")
print(f"  Target: epsilon <= 1.0 for meaningful differential privacy")
print(f"  Centralized baseline: accuracy={cent_acc:.4f}, AUC={cent_auc:.4f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: PACT governance — each client as PactGovernedAgent
# ══════════════════════════════════════════════════════════════════════

fed_org_yaml = """
org_id: federated_health
name: "Federated Health Consortium"

departments:
  - id: aggregation
    name: "Aggregation Server"
  - id: hospital_a
    name: "Hospital A"
  - id: hospital_b
    name: "Hospital B"
  - id: hospital_c
    name: "Hospital C"

roles:
  - id: aggregator
    name: "Federated Aggregator"
    is_primary_for_unit: aggregation

  - id: client_a
    name: "Hospital A Training Agent"
    is_primary_for_unit: hospital_a
    agent: true

  - id: client_b
    name: "Hospital B Training Agent"
    is_primary_for_unit: hospital_b
    agent: true

  - id: client_c
    name: "Hospital C Training Agent"
    is_primary_for_unit: hospital_c
    agent: true

clearances:
  - role: aggregator
    level: secret
    compartments: [aggregated_weights, global_model, drift_metrics]
  - role: client_a
    level: confidential
    compartments: [local_data_a, local_weights_a]
  - role: client_b
    level: confidential
    compartments: [local_data_b, local_weights_b]
  - role: client_c
    level: confidential
    compartments: [local_data_c, local_weights_c]
"""

fed_file = Path(tempfile.mktemp(suffix=".yaml"))
fed_file.write_text(fed_org_yaml)
fed_loaded = load_org_yaml(str(fed_file))
fed_compiled = compile_org(fed_loaded.org_definition)
fed_engine = GovernanceEngine(fed_loaded.org_definition)

fed_roles = {}
for addr, node in fed_compiled.nodes.items():
    if node.role_definition is not None and not node.is_vacant:
        fed_roles[node.node_id] = addr

level_map = {
    "confidential": ConfidentialityLevel.CONFIDENTIAL,
    "secret": ConfidentialityLevel.SECRET,
}

for cs in fed_loaded.clearances:
    if cs.role_id in fed_roles:
        clearance = RoleClearance(
            role_address=fed_roles[cs.role_id],
            max_clearance=level_map[cs.level],
            compartments=frozenset(cs.compartments),
            granted_by_role_address="system_init",
        )
        fed_engine.grant_clearance(fed_roles[cs.role_id], clearance)

# TODO: Set envelopes for each client — allowed_actions=["train_local","submit_weights"]
# Hint: for client_id in ["client_a","client_b","client_c"]: create RoleEnvelope with
#   defining_role_address=fed_roles["aggregator"], then call fed_engine.set_role_envelope(env)
for client_id in ["client_a", "client_b", "client_c"]:
    env = RoleEnvelope(
        id=f"env-{uuid.uuid4().hex[:8]}",
        defining_role_address=____,
        target_role_address=____,
        envelope=ConstraintEnvelopeConfig(
            id=f"{client_id}-envelope",
            operational=OperationalConstraintConfig(
                allowed_actions=____,
            ),
            temporal=TemporalConstraintConfig(),
            data_access=DataAccessConstraintConfig(),
            communication=CommunicationConstraintConfig(),
        ),
    )
    fed_engine.set_role_envelope(env)

print(f"\n=== PACT Federated Governance ===")
print(f"Each hospital is a PactGovernedAgent with isolated compartments.\n")

governed_clients = {}
for client_id in ["client_a", "client_b", "client_c"]:
    agent = PactGovernedAgent(
        engine=fed_engine,
        role_address=fed_roles[client_id],
    )
    governed_clients[client_id] = agent
    ctx = agent.context
    print(
        f"  {client_id}: clearance={ctx.effective_clearance_level}, "
        f"compartments={sorted(ctx.compartments)}"
    )

print(f"\nCross-client inspection tests:")
client_data_items = {
    "client_a": KnowledgeItem(
        item_id="hospital_a_patient_data",
        classification=ConfidentialityLevel.CONFIDENTIAL,
        owning_unit_address=fed_roles["client_a"],
        compartments=frozenset(["local_data_a"]),
    ),
    "client_b": KnowledgeItem(
        item_id="hospital_b_patient_data",
        classification=ConfidentialityLevel.CONFIDENTIAL,
        owning_unit_address=fed_roles["client_b"],
        compartments=frozenset(["local_data_b"]),
    ),
}

# TODO: Check client_b cannot access client_a data
# Hint: fed_engine.check_access(fed_roles["client_b"], client_data_items["client_a"], TrustPosture.SUPERVISED)
result = ____
print(
    f"  Client B -> Client A data: {'ALLOW' if result.allowed else 'DENY'} ({result.reason})"
)

agg_item = KnowledgeItem(
    item_id="aggregated_model_weights",
    classification=ConfidentialityLevel.SECRET,
    owning_unit_address=fed_roles["aggregator"],
    compartments=frozenset(["aggregated_weights"]),
)
result = fed_engine.check_access(
    fed_roles["aggregator"], agg_item, TrustPosture.SUPERVISED
)
print(f"  Aggregator -> aggregated weights: {'ALLOW' if result.allowed else 'DENY'}")

fed_file.unlink()


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Deploy federated model via InferenceServer + DriftMonitor
# ══════════════════════════════════════════════════════════════════════


# TODO: Implement deploy_federated()
# Hint: 1) ConnectionManager("sqlite:///..."), await conn.initialize()
#        2) ModelRegistry(conn), pick best DP result (epsilon closest to 1.0)
#        3) Serialize LogisticRegression with pickle.dumps
#        4) registry.register_model("federated_fraud_dp", artifact, metrics=[MetricSpec(...)], signature=...)
#        5) InferenceServer(registry, cache_size=5), await server.warm_cache(["federated_fraud_dp"])
#        6) test_features = {f: float(X[0, i]) for i, f in enumerate(feature_names)}
#           test_features["transaction_id"] = "TXN-TEST-001"
#           await server.predict(model_name="federated_fraud_dp", features=test_features)
#        7) DriftMonitor(reference_data=X[:200], feature_names=feature_names, psi_threshold=0.2)
#        8) monitor.check_drift(shifted_data) where shifted_data shifts amount column
async def deploy_federated():
    """Register and deploy the DP-federated model."""
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
    ____
    ____
    ____


asyncio.run(deploy_federated())

print("\n--- Exercise 2 complete: federated learning with DP + PACT governance ---")
