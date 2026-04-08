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
    # TODO: Clip z to [-500, 500], return 1 / (1 + exp(-z)).
    z = ____
    return ____


def sigmoid_deriv(z: float) -> float:
    """Sigmoid derivative: sigma(z) * (1 - sigma(z))."""
    # TODO: Compute s = sigmoid(z), return s * (1 - s).
    ____
    return ____


def relu(z: float) -> float:
    """ReLU: max(0, z)."""
    # TODO: Return max(0, z).
    return ____


def relu_deriv(z: float) -> float:
    """ReLU derivative: 1 if z > 0, else 0."""
    # TODO: Return 1.0 if z > 0 else 0.0.
    return ____


def gelu(z: float) -> float:
    """GELU approximation: 0.5 * z * (1 + tanh(sqrt(2/pi) * (z + 0.044715 * z^3)))."""
    # TODO: Implement the tanh-based GELU approximation.
    return ____


def gelu_deriv(z: float) -> float:
    """GELU derivative (numerical approximation)."""
    # TODO: Return central difference: (gelu(z+h) - gelu(z-h)) / (2*h) with h=1e-5.
    h = 1e-5
    return ____


print("=== Activation Functions ===")
test_values = [-3.0, -1.0, 0.0, 1.0, 3.0]
for z in test_values:
    print(
        f"{z:6.1f} | sigmoid={sigmoid(z):.4f} | ReLU={relu(z):.4f} | GELU={gelu(z):.4f}"
    )


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Plot activation functions and their derivatives
# ══════════════════════════════════════════════════════════════════════

z_range = [i * 0.1 for i in range(-50, 51)]

# TODO: Build plot_data DataFrame with columns z, sigmoid, sigmoid_deriv, relu, relu_deriv, gelu, gelu_deriv.
plot_data = ____

viz = ModelVisualizer()
# TODO: Call viz.training_history with metrics dict for sigmoid/relu/gelu values.
fig = ____

print(
    f"\nsigmoid'(0)={sigmoid_deriv(0.0):.4f}, ReLU'(0)={relu_deriv(0.0):.4f}, GELU'(0)={gelu_deriv(0.0):.4f}"
)
print(
    f"sigma'(-5)={sigmoid_deriv(-5.0):.6f} (saturated), ReLU'(-5)={relu_deriv(-5.0):.1f} (dead)"
)


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Build identical networks with different activations
# ══════════════════════════════════════════════════════════════════════

loader = ASCENTDataLoader()
df = loader.load("ascent07", "mnist_sample.parquet")

explorer = DataExplorer()
summary = asyncio.run(explorer.profile(df))

feature_cols = [c for c in df.columns if c != "label"]
X = df.select(feature_cols).to_numpy().tolist()
y_labels = df["label"].to_list()
X = [[pixel / 255.0 for pixel in row] for row in X]

n_classes = 10
Y = [[1.0 if j == label else 0.0 for j in range(n_classes)] for label in y_labels]

n_features = len(X[0])
n_samples = len(X)
print(f"\nMNIST: features={n_features}, samples={n_samples}, classes={n_classes}")


class SimpleNetwork:
    """3-layer network with configurable activation."""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, activation: str
    ):
        self.activation = activation
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # TODO: He init for relu (scale = sqrt(2/fan_in)), Xavier otherwise (sqrt(2/(in+out))).
        if activation == "relu":
            scale1 = ____
            scale2 = ____
        else:
            scale1 = ____
            scale2 = ____

        # TODO: Initialize W1 as input_dim × hidden_dim with random.gauss(0, scale1).
        self.W1 = ____
        self.b1 = [0.0] * hidden_dim
        # TODO: Initialize W2 as hidden_dim × output_dim with random.gauss(0, scale2).
        self.W2 = ____
        self.b2 = [0.0] * output_dim
        self.grad_magnitudes = []

    def activate(self, z: float) -> float:
        # TODO: Dispatch on self.activation to sigmoid / relu / gelu.
        ____

    def activate_deriv(self, z: float) -> float:
        # TODO: Dispatch on self.activation to sigmoid_deriv / relu_deriv / gelu_deriv.
        ____

    def forward(self, x: list[float]) -> tuple:
        # TODO: Hidden pre-activations z1[j] = sum(x[i]*W1[i][j]) + b1[j].
        z1 = ____
        # TODO: Hidden activations h1 = activate each z1.
        h1 = ____
        # TODO: Output pre-activations z2[k] = sum(h1[j]*W2[j][k]) + b2[k].
        z2 = ____
        # TODO: Apply softmax to z2 (subtract max for numerical stability).
        out = ____
        return z1, h1, z2, out


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

            # TODO: Cross-entropy loss with eps=1e-8.
            eps = 1e-8
            loss = ____
            epoch_loss += loss

            # TODO: Output gradient d_out[k] = out[k] - y[k] (softmax + CE).
            d_out = ____

            # TODO: Update W2 via net.W2[j][k] -= lr * d_out[k] * h1[j].
            ____
            # TODO: Update b2 via net.b2[k] -= lr * d_out[k].
            ____

            # TODO: Hidden deltas d_h1[j] = sum(d_out[k]*W2[j][k]) * activate_deriv(z1[j]).
            d_h1 = ____

            # TODO: Update W1 via net.W1[i][j] -= lr * d_h1[j] * x[i].
            ____
            # TODO: Update b1 via net.b1[j] -= lr * d_h1[j].
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

print(f"\n=== Gradient Analysis ===")
print(f"Sigmoid max grad=0.25; 10 layers -> 0.25^10 = {0.25**10:.2e} (vanishing)")

for act_name in ["sigmoid", "relu", "gelu"]:
    # TODO: Sample 1000 random gauss(0,1) values; compute activation derivative; report mean and zero-fraction.
    grads = ____
    mean_grad = ____
    zero_grads = ____
    print(f"  {act_name}: mean|grad|={mean_grad:.4f}, zero_fraction={zero_grads:.1%}")

print("\n✓ Exercise 3 complete — activation functions compared across architectures")
