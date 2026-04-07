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

xor_inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
xor_targets = [0.0, 1.0, 1.0, 0.0]

print("=== XOR Truth Table ===")
for inp, tgt in zip(xor_inputs, xor_targets):
    print(f"  {inp} -> {tgt}")

print(
    "No line w1*x1+w2*x2+b=0 separates (0,0),(1,1) from (0,1),(1,0) — XOR is non-linear."
)


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Build 2-layer network with sigmoid
# ══════════════════════════════════════════════════════════════════════


def sigmoid(z: float) -> float:
    """Sigmoid activation: 1 / (1 + e^(-z))."""
    z = max(-500, min(500, z))
    return 1.0 / (1.0 + math.exp(-z))


def sigmoid_derivative(a: float) -> float:
    """Derivative of sigmoid: a * (1 - a) where a = sigmoid(z)."""
    return a * (1.0 - a)


def init_weight(fan_in: int, fan_out: int) -> list[list[float]]:
    """Xavier/Glorot initialization: uniform(-sqrt(6/(in+out)), sqrt(6/(in+out)))."""
    # TODO: Compute Xavier limit = sqrt(6/(fan_in+fan_out)).
    # Return (fan_in × fan_out) matrix with entries uniform in [-limit, limit].
    ____
    ____


W1 = init_weight(2, 2)
b1 = [0.0, 0.0]
W2 = init_weight(2, 1)
b2 = [0.0]

print(f"Architecture: 2→2(sigmoid)→1(sigmoid), {2*2+2+2*1+1} parameters")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Train on XOR via gradient descent
# ══════════════════════════════════════════════════════════════════════


def forward_pass(x: list[float]) -> tuple:
    """Forward pass: 2-layer sigmoid network. Returns (h, y_hat)."""
    # TODO: Hidden layer: for j in range(2), z = sum(x[i]*W1[i][j])+b1[j], h.append(sigmoid(z)).
    # Output: z_out = sum(h[j]*W2[j][0])+b2[0]; y_hat = sigmoid(z_out). Return (h, y_hat).
    ____
    ____
    ____
    ____
    ____


learning_rate = 1.0
epochs = 5000
history = {"epoch": [], "loss": []}

for epoch in range(epochs):
    total_loss = 0.0

    for x, y in zip(xor_inputs, xor_targets):
        h, y_hat = forward_pass(x)

        # TODO: BCE loss = -(y*log(y_hat+eps)+(1-y)*log(1-y_hat+eps)); accumulate to total_loss.
        # d_out = y_hat - y; dW2[j] = h[j]*d_out; db2 = d_out.
        # d_hidden[j] = d_out*W2[j][0]*sigmoid_derivative(h[j]).
        # Update W1[i][j] -= lr*x[i]*d_hidden[j]; b1[j] -= lr*d_hidden[j].
        # Update W2[j][0] -= lr*dW2[j]; b2[0] -= lr*db2.
        ____
        ____
        ____
        ____
        ____
        ____
        ____
        ____
        ____
        total_loss += loss

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

# TODO: Generate 21×21 grid over [0,1]×[0,1]; call forward_pass for each point.
# Build boundary_df = pl.DataFrame(grid_points); create xor_df with x1, x2, label.
____
____
____
____
____
____
____

print(f"\n=== Decision Boundary ===")
print(f"Grid points: {boundary_df.height}")
print(f"Boundary shows how hidden layer creates non-linear separation")

print(f"\nHidden representations:")
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

print(f"\n=== Synthetic Spirals Dataset ===")
print(f"Shape: {spirals_df.shape}")
print(f"Columns: {spirals_df.columns}")
print(f"Classes: {spirals_df['label'].unique().sort().to_list()}")

class_counts = spirals_df.group_by("label").len().sort("label")
print(f"Samples per class:")
print(class_counts)

print(f"\nKey insight: XOR is the simplest non-linearly-separable problem.")
print(f"  Spirals are a harder version — they require more hidden neurons")
print(f"  to carve out the curved decision boundaries.")
print(f"  Each hidden neuron adds one linear boundary in input space;")
print(f"  combining them creates complex, non-linear regions.")

print("\n✓ Exercise 2 complete — hidden layers solve XOR and non-linear classification")
