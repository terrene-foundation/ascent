# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT07 — Exercise 2: Hidden Layers and the XOR Problem
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Show why hidden layers matter by solving XOR, then build a
#   multi-layer perceptron demonstrating automatic feature interaction
#   discovery.
#
# TASKS:
#   1. Show XOR is not linearly separable
#   2. Build 2-layer network with sigmoid
#   3. Train on XOR via gradient descent
#   4. Visualize decision boundary with ModelVisualizer
#   5. Extend to multi-class on synthetic spirals data
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
# TASK 1: Show XOR is not linearly separable
# ══════════════════════════════════════════════════════════════════════

# XOR truth table
xor_inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
xor_targets = [0.0, 1.0, 1.0, 0.0]

print("=== XOR Truth Table ===")
for inp, tgt in zip(xor_inputs, xor_targets):
    print(f"  {inp} -> {tgt}")

print("\nNo single line w1*x1 + w2*x2 + b = 0 separates diagonals from anti-diagonals.")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Build 2-layer network with sigmoid
# ══════════════════════════════════════════════════════════════════════


def sigmoid(z: float) -> float:
    """Sigmoid activation: 1 / (1 + e^(-z))."""
    # TODO: Clip z to [-500, 500] then return 1 / (1 + exp(-z)).
    z = ____
    return ____


def sigmoid_derivative(a: float) -> float:
    """Derivative of sigmoid in terms of the activation: a * (1 - a)."""
    # TODO: Return a * (1 - a).
    return ____


def init_weight(fan_in: int, fan_out: int) -> list[list[float]]:
    """Xavier/Glorot: uniform(-sqrt(6/(in+out)), sqrt(6/(in+out)))."""
    # TODO: Compute limit = sqrt(6 / (fan_in + fan_out)).
    limit = ____
    # TODO: Return fan_in × fan_out matrix of random.uniform(-limit, limit).
    return ____


# TODO: Initialize W1 (2→2 via init_weight), b1 = [0.0, 0.0].
W1 = ____
b1 = ____
# TODO: Initialize W2 (2→1 via init_weight), b2 = [0.0].
W2 = ____
b2 = ____

print(f"\n=== Network: 2 → 2 (sigmoid) → 1 (sigmoid), 9 parameters ===")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Train on XOR via gradient descent
# ══════════════════════════════════════════════════════════════════════


def forward_pass(x: list[float]) -> tuple:
    """Forward pass through 2-layer network."""
    # TODO: Hidden layer: for j in range(2), z = sum(x[i]*W1[i][j]) + b1[j]; apply sigmoid.
    h = ____
    # TODO: Output: z_out = sum(h[j]*W2[j][0]) + b2[0]; apply sigmoid to get y_hat.
    y_hat = ____
    return h, y_hat


learning_rate = 1.0
epochs = 5000
history: dict[str, list] = {"epoch": [], "loss": []}

for epoch in range(epochs):
    total_loss = 0.0

    for x, y in zip(xor_inputs, xor_targets):
        h, y_hat = forward_pass(x)

        # TODO: Binary cross-entropy loss with eps=1e-8 for log stability.
        eps = 1e-8
        loss = ____
        total_loss += loss

        # TODO: Output delta for BCE + sigmoid simplifies to y_hat - y.
        d_out = ____

        # TODO: Gradients for W2 (list of h[j]*d_out) and b2 (d_out).
        dW2 = ____
        db2 = ____

        # TODO: Hidden layer deltas: d_out * W2[j][0] * sigmoid_derivative(h[j]).
        d_hidden = ____

        # TODO: Update W1[i][j] using x[i] and d_hidden[j] with learning_rate.
        ____
        # TODO: Update b1[j] using d_hidden[j].
        ____

        # TODO: Update W2[j][0] using dW2[j].
        ____
        # TODO: Update b2[0] using db2.
        ____

    avg_loss = total_loss / 4.0
    if epoch % 1000 == 0:
        history["epoch"].append(epoch)
        history["loss"].append(avg_loss)
        print(f"Epoch {epoch:5d}: loss={avg_loss:.6f}")

print(f"\n=== Trained XOR Predictions ===")
for x, y in zip(xor_inputs, xor_targets):
    _, y_hat = forward_pass(x)
    print(f"  {x} -> {y_hat:.4f} (target: {y})")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Visualize decision boundary with ModelVisualizer
# ══════════════════════════════════════════════════════════════════════

grid_points = []
for gx in range(0, 101, 5):
    for gy in range(0, 101, 5):
        # TODO: Run forward_pass on [gx/100, gy/100], append dict with x1/x2/prediction.
        ____

boundary_df = pl.DataFrame(grid_points)
viz = ModelVisualizer()

xor_df = pl.DataFrame(
    {
        "x1": [x[0] for x in xor_inputs],
        "x2": [x[1] for x in xor_inputs],
        "label": xor_targets,
    }
)

print(f"\nGrid points: {boundary_df.height}")
for x, y in zip(xor_inputs, xor_targets):
    h, y_hat = forward_pass(x)
    print(f"  {x} -> hidden=({h[0]:.3f}, {h[1]:.3f}) -> out={y_hat:.3f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Extend to multi-class on synthetic spirals data
# ══════════════════════════════════════════════════════════════════════

loader = ASCENTDataLoader()
spirals_df = loader.load("ascent07", "synthetic_spirals.parquet")

explorer = DataExplorer()
spiral_summary = asyncio.run(explorer.profile(spirals_df))

print(
    f"\nSpirals shape: {spirals_df.shape}, classes: {spirals_df['label'].unique().sort().to_list()}"
)

# TODO: Build class_counts via group_by("label").len().sort("label") and print.
class_counts = ____
print(class_counts)

print("\n✓ Exercise 2 complete — hidden layers solve XOR and non-linear classification")
