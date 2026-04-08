# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT07 — Exercise 4: Loss Functions and Weight Initialization
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Compare MSE vs CrossEntropy loss on classification, and
#   Xavier vs He initialization on deep networks.
#
# TASKS:
#   1. Implement CrossEntropy from scratch
#   2. Compare MSE vs CE on same classification task
#   3. Implement Xavier and He initialization
#   4. Train 10-layer network with each init strategy
#   5. Visualize gradient flow per layer
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import math
import random

import polars as pl

from kailash_ml import DataExplorer, ModelVisualizer

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()

random.seed(42)


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Implement CrossEntropy from scratch
# ══════════════════════════════════════════════════════════════════════


def softmax(z: list[float]) -> list[float]:
    """Softmax: exp(z_i) / sum(exp(z_j))."""
    # TODO: Subtract max(z) for stability, exponentiate, normalize.
    return ____


def cross_entropy_loss(y_true: list[float], y_pred: list[float]) -> float:
    """Cross-entropy: -sum(y_true * log(y_pred))."""
    # TODO: Sum -yt * log(yp + eps) with eps=1e-8.
    return ____


def mse_loss(y_true: list[float], y_pred: list[float]) -> float:
    """MSE: (1/n) * sum((y_true - y_pred)^2)."""
    # TODO: Return mean squared difference.
    return ____


y_true = [0.0, 0.0, 1.0, 0.0, 0.0]
y_confident = [0.01, 0.01, 0.95, 0.01, 0.02]
y_uncertain = [0.15, 0.20, 0.30, 0.20, 0.15]
y_wrong = [0.60, 0.20, 0.05, 0.10, 0.05]

print("=== Loss Function Comparison ===")
for name, pred in [
    ("confident", y_confident),
    ("uncertain", y_uncertain),
    ("wrong", y_wrong),
]:
    print(
        f"{name:>12} | CE={cross_entropy_loss(y_true, pred):.4f} | MSE={mse_loss(y_true, pred):.4f}"
    )


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Compare MSE vs CE on same classification task
# ══════════════════════════════════════════════════════════════════════

loader = ASCENTDataLoader()
df = loader.load("ascent07", "mnist_sample.parquet")

explorer = DataExplorer()
summary = asyncio.run(explorer.profile(df))

feature_cols = [c for c in df.columns if c != "label"]
X = [
    [pixel / 255.0 for pixel in row]
    for row in df.select(feature_cols).to_numpy().tolist()
]
y_labels = df["label"].to_list()
n_classes = 10
Y = [[1.0 if j == label else 0.0 for j in range(n_classes)] for label in y_labels]

n_features = len(X[0])
train_size = min(200, len(X))

print(f"\nMNIST: features={n_features}, train={train_size}, classes={n_classes}")


def train_with_loss(loss_type: str, epochs: int = 15) -> list[float]:
    """Train a simple network with specified loss function."""
    hidden = 32
    # TODO: Xavier init for W1 (n_features × hidden) and W2 (hidden × n_classes).
    scale = ____
    W1 = ____
    b1 = [0.0] * hidden
    scale2 = ____
    W2 = ____
    b2 = [0.0] * n_classes

    lr = 0.01
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for idx in range(train_size):
            x = X[idx]
            y = Y[idx]

            # TODO: Hidden layer forward with ReLU.
            h = ____

            # TODO: Output layer forward with softmax.
            z_out = ____
            out = softmax(z_out)

            # TODO: Compute loss based on loss_type ("ce" or "mse").
            if loss_type == "ce":
                loss = ____
            else:
                loss = ____
            epoch_loss += loss

            # TODO: Output gradient: CE simplifies to (out-y); MSE is 2*(out-y)/n_classes.
            if loss_type == "ce":
                d_out = ____
            else:
                d_out = ____

            # TODO: Update W2 and b2.
            ____
            ____

            # TODO: Hidden pre-activations and deltas d_h.
            z_hidden = ____
            d_h = ____

            # TODO: Update W1 and b1.
            ____
            ____

        losses.append(epoch_loss / train_size)

    return losses


print(f"\nTraining with MSE loss...")
mse_losses = train_with_loss("mse")
print(f"Training with CrossEntropy loss...")
ce_losses = train_with_loss("ce")

viz = ModelVisualizer()
fig = viz.training_history({"mse_loss": mse_losses, "ce_loss": ce_losses})

print(f"  MSE final loss:  {mse_losses[-1]:.4f}")
print(f"  CE final loss:   {ce_losses[-1]:.4f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Implement Xavier and He initialization
# ══════════════════════════════════════════════════════════════════════


def xavier_init(fan_in: int, fan_out: int) -> list[list[float]]:
    """Xavier/Glorot: N(0, sqrt(2 / (fan_in + fan_out))). Best for sigmoid/tanh/GELU."""
    # TODO: Compute std and return fan_in × fan_out gaussian matrix.
    std = ____
    return ____


def he_init(fan_in: int, fan_out: int) -> list[list[float]]:
    """He/Kaiming: N(0, sqrt(2 / fan_in)). Best for ReLU."""
    # TODO: Compute std and return fan_in × fan_out gaussian matrix.
    std = ____
    return ____


def zero_init(fan_in: int, fan_out: int) -> list[list[float]]:
    """Zero initialization (bad — for demonstration only)."""
    return [[0.0 for _ in range(fan_out)] for _ in range(fan_in)]


print(f"\n=== Initialization Comparison ===")
for name, init_fn in [("Xavier", xavier_init), ("He", he_init), ("Zero", zero_init)]:
    W = init_fn(784, 128)
    flat = [w for row in W for w in row]
    mean_w = sum(flat) / len(flat)
    var_w = sum((w - mean_w) ** 2 for w in flat) / len(flat)
    print(f"  {name:>6}: mean={mean_w:.6f}, var={var_w:.6f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Train 10-layer network with each init strategy
# ══════════════════════════════════════════════════════════════════════


def build_deep_network(n_layers: int, init_fn) -> list[dict]:
    """Build a deep network with specified initialization."""
    layers = []
    dims = [n_features] + [64] * n_layers + [n_classes]
    # TODO: For each layer pair, create {"W": init_fn(...), "b": zeros, "in": dims[i], "out": dims[i+1]}.
    ____
    return layers


def forward_deep(x: list[float], layers: list[dict]) -> list[list[float]]:
    """Forward pass through deep network, returning activations per layer."""
    activations = [x]
    current = x

    for i, layer in enumerate(layers):
        # TODO: Compute z = Wx + b for this layer.
        z = ____

        # TODO: ReLU for hidden layers, softmax for output layer.
        if i < len(layers) - 1:
            current = ____
        else:
            current = ____

        activations.append(current)

    return activations


print(f"\n=== Deep Network (10 layers) ===")
n_deep_layers = 10

for name, init_fn in [("Xavier", xavier_init), ("He", he_init)]:
    layers = build_deep_network(n_deep_layers, init_fn)
    activations = forward_deep(X[0], layers)
    print(f"\n  {name} init — activation magnitudes:")
    for i, act in enumerate(activations[1:], 1):
        mean_act = sum(abs(a) for a in act) / len(act)
        zero_frac = sum(1 for a in act if abs(a) < 1e-6) / len(act)
        print(f"    Layer {i:2d}: mean|act|={mean_act:.6f}, dead={zero_frac:.1%}")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Visualize gradient flow per layer
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Gradient Flow Analysis ===")
print(f"Sigmoid/Tanh -> Xavier; ReLU/variants -> He.")
print(f"Correct init keeps activation variance ~1.0 across all layers.")

print("\n✓ Exercise 4 complete — loss functions and initialization strategies compared")
