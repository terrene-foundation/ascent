# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT07 — Exercise 5: Backpropagation from Scratch
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Implement the backpropagation algorithm step by step —
#   chain rule, gradient computation, gradient checking — then diagnose
#   vanishing gradients.
#
# TASKS:
#   1. Implement forward pass for 3-layer network
#   2. Derive and implement backward pass (chain rule)
#   3. Verify with numerical gradient checking
#   4. Demonstrate vanishing gradients with sigmoid
#   5. Fix with ReLU + proper initialization
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import math
import random

import polars as pl

from kailash_ml import DataExplorer, ModelVisualizer

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()

random.seed(42)


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Implement forward pass for 3-layer network
# ══════════════════════════════════════════════════════════════════════

loader = ASCENTDataLoader()
df = loader.load("ascent07", "mnist_sample.parquet")

feature_cols = [c for c in df.columns if c != "label"]
X = [
    [pixel / 255.0 for pixel in row]
    for row in df.select(feature_cols).to_numpy().tolist()
]
y_labels = df["label"].to_list()
n_classes = 10
Y = [[1.0 if j == label else 0.0 for j in range(n_classes)] for label in y_labels]

n_features = len(X[0])
print(f"MNIST: features={n_features}, samples={len(X)}, classes={n_classes}")

dims = [n_features, 64, 32, n_classes]


def sigmoid(z: float) -> float:
    z = max(-500, min(500, z))
    return 1.0 / (1.0 + math.exp(-z))


def softmax(z: list[float]) -> list[float]:
    max_z = max(z)
    exp_z = [math.exp(zi - max_z) for zi in z]
    total = sum(exp_z)
    return [e / total for e in exp_z]


def init_params(dims: list[int]) -> tuple[list, list]:
    weights = []
    biases = []
    for i in range(len(dims) - 1):
        # TODO: Xavier std = sqrt(2 / (dims[i] + dims[i+1])).
        std = ____
        # TODO: Build W as dims[i] × dims[i+1] gauss(0, std), b as zeros.
        W = ____
        b = ____
        weights.append(W)
        biases.append(b)
    return weights, biases


weights, biases = init_params(dims)

print(f"\nArchitecture: {dims}")


def forward(x: list[float], weights: list, biases: list) -> dict:
    """Full forward pass, caching all intermediates for backprop."""
    cache = {"activations": [x], "pre_activations": []}
    current = x

    for layer_idx in range(len(weights)):
        W = weights[layer_idx]
        b = biases[layer_idx]
        d_in = len(current)
        d_out = len(b)

        # TODO: Linear z = Wx + b.
        z = ____
        cache["pre_activations"].append(z)

        # TODO: Sigmoid for hidden layers, softmax for last layer.
        if layer_idx < len(weights) - 1:
            current = ____
        else:
            current = ____

        cache["activations"].append(current)

    return cache


cache = forward(X[0], weights, biases)
output = cache["activations"][-1]
print(
    f"Forward pass: predicted class={output.index(max(output))}, max prob={max(output):.4f}"
)


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Derive and implement backward pass (chain rule)
# ══════════════════════════════════════════════════════════════════════


def backward(cache: dict, y_true: list[float], weights: list) -> tuple[list, list]:
    """Backpropagation via chain rule.

    For each layer l (output → input):
      dL/dW_l = a_{l-1}^T @ delta_l
      dL/db_l = delta_l
      delta_{l-1} = (W_l^T @ delta_l) * sigma'(z_{l-1})
    """
    activations = cache["activations"]
    pre_activations = cache["pre_activations"]
    n_layers = len(weights)

    dW_list = []
    db_list = []

    # TODO: Initial delta = softmax output - y_true (simplification for softmax+CE).
    delta = ____

    for layer_idx in range(n_layers - 1, -1, -1):
        a_prev = activations[layer_idx]
        d_in = len(a_prev)
        d_out = len(delta)

        # TODO: Gradient dW[j][k] = a_prev[j] * delta[k]; db = delta.
        dW = ____
        db = ____

        dW_list.insert(0, dW)
        db_list.insert(0, db)

        # TODO: If not first layer, propagate delta backward through sigmoid.
        if layer_idx > 0:
            W = weights[layer_idx]
            z_prev = pre_activations[layer_idx - 1]
            # TODO: delta_new[j] = sum(W[j][k]*delta[k]) * sigmoid(z_prev[j]) * (1-sigmoid(z_prev[j])).
            delta_new = ____
            delta = delta_new

    return dW_list, db_list


cache = forward(X[0], weights, biases)
dW_list, db_list = backward(cache, Y[0], weights)

print(f"\n=== Backward Pass ===")
for i, (dW, db) in enumerate(zip(dW_list, db_list)):
    flat_dW = [abs(g) for row in dW for g in row]
    mean_grad = sum(flat_dW) / len(flat_dW)
    print(f"  Layer {i+1}: |dW| mean={mean_grad:.8f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Verify with numerical gradient checking
# ══════════════════════════════════════════════════════════════════════


def compute_loss(x: list[float], y: list[float], weights: list, biases: list) -> float:
    """Compute cross-entropy loss for a single sample."""
    cache = forward(x, weights, biases)
    output = cache["activations"][-1]
    eps = 1e-8
    return -sum(y[k] * math.log(output[k] + eps) for k in range(len(y)))


def numerical_gradient_check(
    x: list[float], y: list[float], weights: list, biases: list, epsilon: float = 1e-5
) -> float:
    """Compare analytical vs numerical gradients. Returns max relative error."""
    cache = forward(x, weights, biases)
    dW_analytical, db_analytical = backward(cache, y, weights)

    max_rel_error = 0.0

    for layer_idx in range(len(weights)):
        for i in range(min(3, len(weights[layer_idx]))):
            for j in range(min(3, len(weights[layer_idx][0]))):
                original = weights[layer_idx][i][j]

                # TODO: Perturb weight by +epsilon and compute loss_plus.
                weights[layer_idx][i][j] = ____
                loss_plus = compute_loss(x, y, weights, biases)

                # TODO: Perturb weight by -epsilon and compute loss_minus.
                weights[layer_idx][i][j] = ____
                loss_minus = compute_loss(x, y, weights, biases)

                weights[layer_idx][i][j] = original

                # TODO: Numerical gradient = (loss_plus - loss_minus) / (2*epsilon).
                numerical = ____
                analytical = dW_analytical[layer_idx][i][j]

                denom = max(abs(numerical) + abs(analytical), 1e-8)
                rel_error = abs(numerical - analytical) / denom
                max_rel_error = max(max_rel_error, rel_error)

    return max_rel_error


rel_error = numerical_gradient_check(X[0], Y[0], weights, biases)
print(
    f"\nGradient check: max rel error = {rel_error:.2e} "
    f"({'PASS' if rel_error < 1e-5 else 'WARN'})"
)


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Demonstrate vanishing gradients with sigmoid
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Vanishing Gradients with Sigmoid ===")

for depth in [3, 5, 8, 12]:
    deep_dims = [n_features] + [64] * depth + [n_classes]
    # TODO: Init params, forward pass, backward pass, compute grad magnitudes per layer.
    deep_w, deep_b = ____
    deep_cache = ____
    deep_dW, deep_db = ____

    grad_mags = []
    for i, dW in enumerate(deep_dW):
        flat = [abs(g) for row in dW for g in row]
        grad_mags.append(sum(flat) / len(flat))

    ratio = grad_mags[0] / max(grad_mags[-1], 1e-15)
    print(f"  Depth={depth}: first/last ratio={ratio:.0e}")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Fix with ReLU + proper initialization
# ══════════════════════════════════════════════════════════════════════


def relu(z: float) -> float:
    return max(0.0, z)


def forward_relu(x: list[float], weights: list, biases: list) -> dict:
    """Forward pass with ReLU activations (except output)."""
    cache = {"activations": [x], "pre_activations": []}
    current = x

    for layer_idx in range(len(weights)):
        W = weights[layer_idx]
        b = biases[layer_idx]
        d_in = len(current)
        d_out = len(b)

        # TODO: Linear z = Wx + b; then ReLU for hidden, softmax for output.
        z = ____
        cache["pre_activations"].append(z)

        if layer_idx < len(weights) - 1:
            current = ____
        else:
            current = ____

        cache["activations"].append(current)

    return cache


def backward_relu(cache: dict, y_true: list[float], weights: list) -> tuple[list, list]:
    """Backprop with ReLU."""
    activations = cache["activations"]
    pre_activations = cache["pre_activations"]
    n_layers = len(weights)

    dW_list = []
    db_list = []
    # TODO: delta = output - y_true.
    delta = ____

    for layer_idx in range(n_layers - 1, -1, -1):
        a_prev = activations[layer_idx]
        d_in = len(a_prev)
        d_out = len(delta)

        # TODO: dW = outer(a_prev, delta); db = delta.
        dW = ____
        db = ____
        dW_list.insert(0, dW)
        db_list.insert(0, db)

        if layer_idx > 0:
            W = weights[layer_idx]
            z_prev = pre_activations[layer_idx - 1]
            # TODO: delta_new[j] = sum(W[j][k]*delta[k]) * (1 if z_prev[j]>0 else 0).
            delta_new = ____
            delta = delta_new

    return dW_list, db_list


def he_init_params(dims: list[int]) -> tuple[list, list]:
    """He initialization for ReLU networks."""
    weights = []
    biases = []
    for i in range(len(dims) - 1):
        # TODO: He std = sqrt(2 / dims[i]).
        std = ____
        W = ____
        b = [0.0] * dims[i + 1]
        weights.append(W)
        biases.append(b)
    return weights, biases


print(f"\n=== ReLU + He Init: Gradient Flow ===")

for depth in [3, 5, 8, 12]:
    deep_dims = [n_features] + [64] * depth + [n_classes]
    deep_w, deep_b = he_init_params(deep_dims)
    deep_cache = forward_relu(X[0], deep_w, deep_b)
    deep_dW, deep_db = backward_relu(deep_cache, Y[0], deep_w)

    grad_mags = []
    for i, dW in enumerate(deep_dW):
        flat = [abs(g) for row in dW for g in row]
        grad_mags.append(sum(flat) / len(flat))

    ratio = grad_mags[0] / max(grad_mags[-1], 1e-15)
    print(f"  Depth={depth}: first/last ratio={ratio:.1f}x (much better!)")

viz = ModelVisualizer()
print(
    "\n✓ Exercise 5 complete — backpropagation implemented and gradient flow analyzed"
)
