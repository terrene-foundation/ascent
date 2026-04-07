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
print(f"=== MNIST Data Loaded ===")
print(f"Features: {n_features}, Samples: {len(X)}, Classes: {n_classes}")

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
        std = math.sqrt(2.0 / (dims[i] + dims[i + 1]))
        W = [[random.gauss(0, std) for _ in range(dims[i + 1])] for _ in range(dims[i])]
        b = [0.0] * dims[i + 1]
        weights.append(W)
        biases.append(b)
    return weights, biases


weights, biases = init_params(dims)

total_params = sum(dims[i] * dims[i + 1] + dims[i + 1] for i in range(len(dims) - 1))
print(f"Architecture: {dims}, total parameters: {total_params}")


def forward(x: list[float], weights: list, biases: list) -> dict:
    """Full forward pass, caching all intermediates for backprop."""
    # TODO: Build cache = {"activations": [x], "pre_activations": []}.
    # Each layer: z = linear transform (sum over inputs + bias); store in pre_activations.
    # Apply sigmoid for hidden layers, softmax for output. Append to activations.
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____


cache = forward(X[0], weights, biases)
output = cache["activations"][-1]
predicted_class = output.index(max(output))
print(f"\nForward pass test:")
print(f"  Output shape: {len(output)}")
print(f"  Predicted class: {predicted_class}")
print(f"  Max probability: {max(output):.4f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Derive and implement backward pass (chain rule)
# ══════════════════════════════════════════════════════════════════════


def backward(cache: dict, y_true: list[float], weights: list) -> tuple[list, list]:
    """Backpropagation via chain rule.

    For each layer l (from output to input):
      dL/dW_l = a_{l-1}^T @ delta_l
      dL/db_l = delta_l
      delta_{l-1} = (W_l^T @ delta_l) * sigma'(z_{l-1})
    """
    # TODO: Implement full backprop.
    # Output delta = activations[-1][k] - y_true[k] (softmax+CE simplification).
    # Iterate layers in reverse: dW = outer product a_prev ⊗ delta; db = delta.
    # Propagate delta: for j, sum W[j][k]*delta[k] over k, multiply by sigmoid'(z).
    # Insert dW, db at front (layer 0 first). Return (dW_list, db_list).
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


cache = forward(X[0], weights, biases)
dW_list, db_list = backward(cache, Y[0], weights)

print(f"\n=== Backward Pass ===")
for i, (dW, db) in enumerate(zip(dW_list, db_list)):
    flat_dW = [abs(g) for row in dW for g in row]
    mean_grad = sum(flat_dW) / len(flat_dW)
    print(
        f"  Layer {i+1}: |dW| mean={mean_grad:.8f}, |db| mean={sum(abs(g) for g in db)/len(db):.8f}"
    )


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
                weights[layer_idx][i][j] = original + epsilon
                loss_plus = compute_loss(x, y, weights, biases)
                weights[layer_idx][i][j] = original - epsilon
                loss_minus = compute_loss(x, y, weights, biases)
                weights[layer_idx][i][j] = original

                # TODO: numerical = central finite difference; rel_error = |num-ana|/max(|num|+|ana|,1e-8)
                numerical = ____
                analytical = dW_analytical[layer_idx][i][j]
                denom = ____
                rel_error = ____
                max_rel_error = max(max_rel_error, rel_error)

    return max_rel_error


rel_error = numerical_gradient_check(X[0], Y[0], weights, biases)
print(f"\n=== Gradient Check ===")
print(f"Max relative error: {rel_error:.2e}")
print(f"Status: {'PASS' if rel_error < 1e-5 else 'WARN (>1e-5)'}")
print(f"  (< 1e-5 means backprop is correctly implemented)")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Demonstrate vanishing gradients with sigmoid
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Vanishing Gradients with Sigmoid ===")

for depth in [3, 5, 8, 12]:
    deep_dims = [n_features] + [64] * depth + [n_classes]
    deep_w, deep_b = init_params(deep_dims)
    deep_cache = forward(X[0], deep_w, deep_b)
    deep_dW, deep_db = backward(deep_cache, Y[0], deep_w)

    grad_mags = []
    for i, dW in enumerate(deep_dW):
        flat = [abs(g) for row in dW for g in row]
        grad_mags.append(sum(flat) / len(flat))

    print(f"\n  Depth={depth}: gradient magnitudes per layer")
    for i, mag in enumerate(grad_mags):
        bar = "#" * max(1, int(mag * 1e6))
        print(f"    Layer {i+1:2d}: {mag:.2e} {bar[:40]}")

    ratio = grad_mags[0] / max(grad_mags[-1], 1e-15)
    print(f"    Ratio first/last: {ratio:.0e}")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Fix with ReLU + proper initialization
# ══════════════════════════════════════════════════════════════════════


def relu(z: float) -> float:
    return max(0.0, z)


def forward_relu(x: list[float], weights: list, biases: list) -> dict:
    """Forward pass with ReLU (hidden) + softmax (output), caching intermediates."""
    # TODO: Same structure as forward() but use relu for hidden layers.
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____


def backward_relu(cache: dict, y_true: list[float], weights: list) -> tuple[list, list]:
    """Backprop with ReLU derivative (1 if z>0 else 0)."""
    # TODO: Same structure as backward() but delta propagation uses relu_deriv = 1 if z>0 else 0.
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


def he_init_params(dims: list[int]) -> tuple[list, list]:
    """He initialization: std = sqrt(2/fan_in) per layer."""
    # TODO: For each layer pair, std = sqrt(2/dims[i]); generate W (gauss) and zero b.
    ____
    ____
    ____
    ____
    ____
    ____


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

    print(f"\n  Depth={depth}: gradient magnitudes per layer")
    for i, mag in enumerate(grad_mags):
        bar = "#" * max(1, int(mag * 1e4))
        print(f"    Layer {i+1:2d}: {mag:.2e} {bar[:40]}")

    ratio = grad_mags[0] / max(grad_mags[-1], 1e-15)
    print(f"    Ratio first/last: {ratio:.1f}x (much better!)")

viz = ModelVisualizer()
print(f"\nKey takeaway: ReLU + He init keeps gradients flowing through deep networks.")
print(f"  Sigmoid max gradient = 0.25 -> 0.25^12 = {0.25**12:.2e} (vanished)")
print(f"  ReLU gradient = 1.0 for active neurons -> gradients preserved")

print(
    "\n✓ Exercise 5 complete — backpropagation implemented and gradient flow analyzed"
)
