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

# TODO: Sample ~2000 rows preserving class balance: 200 fraud + 1800 legit,
# TODO: then concat and shuffle with sample(fraction=1.0, seed=42, shuffle=True).
fraud_fraud = ____
fraud_legit = ____
patients = ____

print("=== Credit Card Fraud Dataset (sampled for federated learning) ===")
print(f"Shape: {patients.shape}")
print(f"Fraud rate: {patients['is_fraud'].mean():.4%}")

# TODO: Pull V-feature columns + amount as the design matrix.
feature_names = ____
X = ____
y = ____
n = len(y)

# TODO: Partition into 3 non-IID clients by sorting on amount and slicing thirds.
rng = np.random.default_rng(42)
indices = ____
n_per_client = ____

client_indices = ____

client_data = {}
# TODO: Build client_data[name] = (X[idx], y[idx]) and print sizes.
____


def train_local_model(X_local: np.ndarray, y_local: np.ndarray) -> np.ndarray:
    """Train a local logistic regression and return weight vector."""
    # TODO: Fit a LogisticRegression(max_iter=50, random_state=42).
    # TODO: Return concatenated [coef_.flatten(), intercept_].
    ____


def fedavg(client_weights: list[tuple[np.ndarray, int]]) -> np.ndarray:
    """Federated averaging: weighted mean of client model parameters."""
    # TODO: Compute total_samples across all clients.
    # TODO: Initialise an aggregated zero vector matching weight shape.
    # TODO: Sum weights * (n_samples / total_samples) per client.
    ____


def set_model_weights(model: LogisticRegression, weights: np.ndarray) -> None:
    """Set logistic regression weights from a flat vector."""
    # TODO: Split weights into coef_ and intercept_; assign classes_ = [0,1].
    ____


# TODO: Run FedAvg for 3 rounds. Each round: train local on each client
# TODO: starting from previous global weights (if any), call fedavg(),
# TODO: evaluate accuracy + AUC of the resulting global model on (X, y).
n_rounds = 3
print(f"\n=== FedAvg Training ({n_rounds} rounds) ===")

global_weights = None
____

# TODO: Train a centralised baseline LogisticRegression on (X, y) and print
# TODO: its accuracy + AUC for comparison.
centralized = ____
____
cent_acc = ____
cent_auc = ____
print(f"\n  Centralized baseline: accuracy={cent_acc:.4f}, AUC={cent_auc:.4f}")
print(
    f"  FedAvg gap: accuracy={abs(acc - cent_acc):.4f}, AUC={abs(auc - cent_auc):.4f}"
)


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Gaussian noise DP mechanism with moments accountant
# ══════════════════════════════════════════════════════════════════════


def clip_gradients(weights: np.ndarray, max_norm: float) -> np.ndarray:
    """Clip weight update to sensitivity bound S (L2 norm)."""
    # TODO: If L2 norm exceeds max_norm, scale to max_norm.
    ____


def add_gaussian_noise(
    weights: np.ndarray, sensitivity: float, sigma: float
) -> np.ndarray:
    """Add calibrated Gaussian noise for (epsilon, delta)-DP."""
    # TODO: Draw noise ~ N(0, sigma * sensitivity) and add to weights.
    ____


def moments_accountant(
    n_rounds: int,
    n_samples: int,
    batch_size: int,
    sigma: float,
    delta: float = 1e-5,
) -> float:
    """Simple moments accountant for tracking privacy budget epsilon."""
    # TODO: Compute sampling probability q = batch_size / n_samples.
    # TODO: Use advanced composition: sqrt(2*n_rounds*ln(1/delta)) * (q/sigma)
    # TODO: + n_rounds * q^2 / (2 * sigma^2).
    ____


def fedavg_dp(
    client_data: dict,
    n_rounds: int,
    clip_norm: float,
    sigma: float,
) -> tuple[np.ndarray, float]:
    """FedAvg with differential privacy (gradient clipping + Gaussian noise)."""
    # TODO: For each round, train each client locally, clip the update
    # TODO: relative to global, fedavg the updates, then add Gaussian noise.
    # TODO: After all rounds, compute epsilon via moments_accountant.
    ____


print(f"\n=== Differential Privacy Mechanism ===")
print(f"Components:")
print(f"  1. Gradient clipping: L2 norm bound = sensitivity S")
print(f"  2. Gaussian noise: N(0, sigma^2 * S^2) added to aggregated weights")
print(f"  3. Moments accountant: tracks cumulative epsilon across rounds")

# TODO: Run fedavg_dp at sigma=1.0, clip_norm=1.0, n_rounds=5; report metrics.
dp_weights, eps = ____
dp_model = ____
____
dp_acc = ____
dp_auc = ____
print(
    f"\nDP-FedAvg (sigma=1.0): accuracy={dp_acc:.4f}, AUC={dp_auc:.4f}, epsilon={eps:.4f}"
)


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Privacy-utility trade-off at varying epsilon
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Privacy-Utility Trade-off ===")
print(f"{'Sigma':<10} {'Epsilon':<12} {'Accuracy':<12} {'AUC':<12}")
print("-" * 46)

# TODO: Sweep sigma in [0.5, 1.0, 2.0]; record (sigma, eps, accuracy, auc).
sigma_values = [0.5, 1.0, 2.0]
results = []
____

print(f"\nInterpretation:")
print(f"  Low sigma (0.1) -> high epsilon (weak privacy, high utility)")
print(f"  High sigma (5.0) -> low epsilon (strong privacy, lower utility)")
print(f"  Target: epsilon <= 1.0 for meaningful differential privacy")
print(f"  Centralized baseline: accuracy={cent_acc:.4f}, AUC={cent_auc:.4f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: PACT governance — each client as PactGovernedAgent
# ══════════════════════════════════════════════════════════════════════

# TODO: Define a YAML org with one aggregator role and 3 client_X agents,
# TODO: each with isolated compartments (local_data_X, local_weights_X).
fed_org_yaml = ____

# TODO: Write to temp file, load_org_yaml, compile_org, build GovernanceEngine.
fed_file = ____
____
fed_loaded = ____
fed_compiled = ____
fed_engine = ____

# TODO: Build fed_roles dict (role_id -> address) and grant clearances.
fed_roles = ____
____

level_map = ____

____

# TODO: For each client role, set a RoleEnvelope allowing only train_local
# TODO: and submit_weights. Defining role is the aggregator.
____

print(f"\n=== PACT Federated Governance ===")
print(f"Each hospital is a PactGovernedAgent with isolated compartments.\n")

# TODO: Wrap each client_X in PactGovernedAgent and print clearance + compartments.
governed_clients = {}
____

# TODO: Build KnowledgeItems for hospital_a / hospital_b patient data and
# TODO: verify Client B is denied access to Client A data via fed_engine.
print(f"\nCross-client inspection tests:")
client_data_items = ____

# Client B tries to access Client A data
result = ____
print(
    f"  Client B -> Client A data: {'ALLOW' if result.allowed else 'DENY'} ({result.reason})"
)

# TODO: Build the aggregated_weights KnowledgeItem and confirm aggregator can read.
agg_item = ____
result = ____
print(f"  Aggregator -> aggregated weights: {'ALLOW' if result.allowed else 'DENY'}")

# TODO: Clean up temp YAML file.
____


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Deploy federated model via InferenceServer + DriftMonitor
# ══════════════════════════════════════════════════════════════════════


async def deploy_federated():
    """Register and deploy the DP-federated model."""
    # TODO: ConnectionManager("sqlite:///ascent10_federated.db"); initialise.
    # TODO: Create ModelRegistry, pick best DP result (epsilon ~ 1.0),
    # TODO: re-train fedavg_dp at that sigma, set weights into a fresh
    # TODO: LogisticRegression, pickle it, build a ModelSignature, register
    # TODO: with metrics. Then create an InferenceServer, warm cache, run
    # TODO: a single .predict, and create a DriftMonitor checking shifted
    # TODO: production data. Close the connection at the end.
    ____


asyncio.run(deploy_federated())

print("\n--- Exercise 2 complete: federated learning with DP + PACT governance ---")
