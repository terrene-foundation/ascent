# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT07 — Exercise 3: Activation Functions and Layer Design
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Compare activation functions (sigmoid, ReLU, GELU) on
#   identical architectures and understand how activation choice affects
#   gradient flow and training.
#
# TASKS:
#   1. Implement sigmoid, ReLU, GELU from formulas
#   2. Plot activation functions and their derivatives
#   3. Build identical networks with different activations
#   4. Train on same dataset, compare convergence with ModelVisualizer
#   5. Analyze gradient magnitudes per layer
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
# TASK 1: Implement sigmoid, ReLU, GELU from formulas
# ══════════════════════════════════════════════════════════════════════


def sigmoid(z: float) -> float:
    """Sigmoid: sigma(z) = 1 / (1 + e^(-z))."""
    z = max(-500, min(500, z))
    return 1.0 / (1.0 + math.exp(-z))


def sigmoid_deriv(z: float) -> float:
    """Sigmoid derivative: sigma'(z) = sigma(z) * (1 - sigma(z))."""
    s = sigmoid(z)
    return s * (1.0 - s)


def relu(z: float) -> float:
    """ReLU: max(0, z)."""
    # TODO: Return max(0.0, z).
    ____


def relu_deriv(z: float) -> float:
    """ReLU derivative: 1 if z > 0, else 0."""
    # TODO: Return 1.0 if z > 0 else 0.0.
    ____


def gelu(z: float) -> float:
    """GELU approximation: 0.5 * z * (1 + tanh(sqrt(2/pi) * (z + 0.044715*z^3)))."""
    # TODO: Implement using math.tanh and math.sqrt(2.0 / math.pi).
    ____


def gelu_deriv(z: float) -> float:
    """GELU derivative (numerical approximation)."""
    h = 1e-5
    return (gelu(z + h) - gelu(z - h)) / (2 * h)


print("=== Activation Functions ===")
test_values = [-3.0, -1.0, 0.0, 1.0, 3.0]
print(f"{'z':>6} | {'sigmoid':>8} | {'ReLU':>8} | {'GELU':>8}")
print("-" * 42)
for z in test_values:
    print(f"{z:6.1f} | {sigmoid(z):8.4f} | {relu(z):8.4f} | {gelu(z):8.4f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Plot activation functions and their derivatives
# ══════════════════════════════════════════════════════════════════════

z_range = [i * 0.1 for i in range(-50, 51)]

plot_data = pl.DataFrame(
    {
        "z": z_range,
        "sigmoid": [sigmoid(z) for z in z_range],
        "sigmoid_deriv": [sigmoid_deriv(z) for z in z_range],
        "relu": [relu(z) for z in z_range],
        "relu_deriv": [relu_deriv(z) for z in z_range],
        "gelu": [gelu(z) for z in z_range],
        "gelu_deriv": [gelu_deriv(z) for z in z_range],
    }
)

viz = ModelVisualizer()
fig = viz.training_history(
    metrics={
        "sigmoid": plot_data["sigmoid"].to_list(),
        "relu": plot_data["relu"].to_list(),
        "gelu": plot_data["gelu"].to_list(),
    },
    x_label="z",
)

print(f"\n=== Derivatives at z=0 ===")
print(f"sigmoid'(0) = {sigmoid_deriv(0.0):.4f} (max value = 0.25)")
print(f"ReLU'(0)    = {relu_deriv(0.0):.4f} (discontinuous at 0)")
print(f"GELU'(0)    = {gelu_deriv(0.0):.4f} (smooth, ~0.5)")
print(f"\nSigmoid saturates: sigma'(-5) = {sigmoid_deriv(-5.0):.6f} (nearly zero!)")
print(f"ReLU is dead for z<0: relu'(-5) = {relu_deriv(-5.0):.1f}")
print(f"GELU is smooth everywhere: gelu'(-1) = {gelu_deriv(-1.0):.4f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Build identical networks with different activations
# ══════════════════════════════════════════════════════════════════════

loader = ASCENTDataLoader()
df = loader.load("ascent07", "mnist_sample.parquet")

explorer = DataExplorer()
summary = asyncio.run(explorer.profile(df))

print(f"\n=== MNIST Sample ===")
print(f"Shape: {df.shape}")

feature_cols = [c for c in df.columns if c != "label"]
X = df.select(feature_cols).to_numpy().tolist()
y_labels = df["label"].to_list()
X = [[pixel / 255.0 for pixel in row] for row in X]
n_classes = 10
Y = [[1.0 if j == label else 0.0 for j in range(n_classes)] for label in y_labels]
n_features = len(X[0])
n_samples = len(X)
print(f"Features: {n_features}, Samples: {n_samples}, Classes: {n_classes}")


class SimpleNetwork:
    """3-layer network with configurable activation."""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, activation: str
    ):
        self.activation = activation
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # TODO: He init (scale=sqrt(2/fan_in)) for "relu"; Xavier (scale=sqrt(2/(fan_in+fan_out))) otherwise.
        # Initialise W1 (input_dim×hidden_dim), b1, W2 (hidden_dim×output_dim), b2, grad_magnitudes=[].
        ____
        ____
        ____
        ____
        ____
        ____

    def activate(self, z: float) -> float:
        if self.activation == "sigmoid":
            return sigmoid(z)
        elif self.activation == "relu":
            return relu(z)
        else:
            return gelu(z)

    def activate_deriv(self, z: float) -> float:
        if self.activation == "sigmoid":
            return sigmoid_deriv(z)
        elif self.activation == "relu":
            return relu_deriv(z)
        else:
            return gelu_deriv(z)

    def forward(self, x: list[float]) -> tuple:
        # TODO: Hidden: z1[j] = sum(x[i]*W1[i][j])+b1[j]; h1 = [activate(z) for z in z1].
        # Output: z2[k] = sum(h1[j]*W2[j][k])+b2[k]; stable softmax. Return (z1, h1, z2, out).
        ____
        ____
        ____
        ____
        ____
        ____


print(f"\n=== Networks Created ===")
for act_name in ["sigmoid", "relu", "gelu"]:
    net = SimpleNetwork(n_features, 32, n_classes, act_name)
    print(f"  {act_name}: {n_features}->32->{n_classes}")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Train on same dataset, compare convergence
# ══════════════════════════════════════════════════════════════════════

results = {}
train_size = min(200, n_samples)

for act_name in ["sigmoid", "relu", "gelu"]:
    net = SimpleNetwork(n_features, 32, n_classes, act_name)
    lr = 0.01
    losses = []

    for epoch in range(20):
        epoch_loss = 0.0
        for idx in range(train_size):
            x = X[idx]
            y = Y[idx]
            z1, h1, z2, out = net.forward(x)

            # TODO: CE loss; d_out = out[k]-y[k]; update W2/b2 with lr*d_out[k]*h1[j].
            # d_h1[j] = sum(d_out[k]*W2[j][k])*activate_deriv(z1[j]); update W1/b1.
            ____
            ____
            ____
            ____
            ____
            ____
            ____
            ____

        avg_loss = epoch_loss / train_size
        losses.append(avg_loss)
        if epoch % 5 == 0:
            print(f"  [{act_name:>7}] Epoch {epoch:3d}: loss={avg_loss:.4f}")

    results[act_name] = losses

fig = viz.training_history(
    metrics={
        "sigmoid_loss": results["sigmoid"],
        "relu_loss": results["relu"],
        "gelu_loss": results["gelu"],
    },
    x_label="Epoch",
)
print(f"\nConvergence comparison plotted.")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Analyze gradient magnitudes per layer
# ══════════════════════════════════════════════════════════════════════

print(
    f"Sigmoid 0.25^10 = {0.25**10:.2e}; ReLU: no vanishing, dead neurons; GELU: smooth."
)

# TODO: For each activation, sample 1000 z~N(0,1), compute mean gradient and zero fraction.
for act_name in ["sigmoid", "relu", "gelu"]:
    ____
    ____
    ____
    ____

print("\n✓ Exercise 3 complete — activation functions compared across architectures")
