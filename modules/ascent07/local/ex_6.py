# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT07 — Exercise 6: Optimizers and Learning Rate Scheduling
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Compare SGD, SGD+momentum, Adam optimizers and implement
#   learning rate warmup + cosine annealing.
#
# TASKS:
#   1. Implement SGD with mini-batches
#   2. Add momentum to SGD
#   3. Implement Adam optimizer
#   4. Compare convergence curves with ModelVisualizer
#   5. Add learning rate warmup + cosine decay schedule
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import copy
import math
import random

import polars as pl

from kailash_ml import ModelVisualizer

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Implement SGD with mini-batches
# ══════════════════════════════════════════════════════════════════════

loader = ASCENTDataLoader()
data = loader.load("ascent07", "mnist_sample.parquet")

X = data.select([c for c in data.columns if c != "label"]).to_numpy() / 255.0
y_raw = data["label"].to_numpy()

n_classes = 10
y = [[1.0 if j == int(label) else 0.0 for j in range(n_classes)] for label in y_raw]

n_samples, n_features = X.shape
print(
    f"=== Dataset: {n_samples} samples, {n_features} features, {n_classes} classes ==="
)

random.seed(42)
hidden_size = 128


def init_weights(rows: int, cols: int) -> list[list[float]]:
    """He initialization: std = sqrt(2 / fan_in)."""
    # TODO: Return rows × cols matrix of random.gauss(0, sqrt(2/rows)).
    s = ____
    return ____


W1 = init_weights(n_features, hidden_size)
b1 = [0.0] * hidden_size
W2 = init_weights(hidden_size, n_classes)
b2 = [0.0] * n_classes


def relu(x: list[float]) -> list[float]:
    return [max(0.0, v) for v in x]


def softmax(x: list[float]) -> list[float]:
    max_x = max(x)
    exps = [math.exp(v - max_x) for v in x]
    s = sum(exps)
    return [e / s for e in exps]


def forward(x_row, w1, b1_, w2, b2_):
    """Forward pass: input → ReLU → softmax."""
    # TODO: Compute hidden = Wx + b1, h_act = relu(hidden).
    hidden = ____
    h_act = ____
    # TODO: logits = W2*h_act + b2, probs = softmax(logits).
    logits = ____
    probs = ____
    return hidden, h_act, logits, probs


def cross_entropy_loss(probs: list[float], target: list[float]) -> float:
    return -sum(t * math.log(max(p, 1e-10)) for p, t in zip(probs, target))


# Train with vanilla SGD
batch_size = 32
lr = 0.01
epochs = 5
sgd_losses = []

W1_sgd, b1_sgd = copy.deepcopy(W1), copy.deepcopy(b1)
W2_sgd, b2_sgd = copy.deepcopy(W2), copy.deepcopy(b2)

for epoch in range(epochs):
    epoch_loss = 0.0
    n_batches = 0
    indices = list(range(min(500, n_samples)))
    random.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        batch_loss = 0.0

        for idx in batch_idx:
            x_row = X[idx].tolist()
            target = y[idx]
            hidden, h_act, logits, probs = forward(
                x_row, W1_sgd, b1_sgd, W2_sgd, b2_sgd
            )
            batch_loss += cross_entropy_loss(probs, target)

            # TODO: d_logits = probs - target.
            d_logits = ____

            # TODO: SGD update for W2 and b2 (divide by len(batch_idx)).
            ____
            ____

            # TODO: d_hidden[j] = sum(d_logits[k]*W2[j][k]) * (1 if hidden[j]>0 else 0).
            d_hidden = ____

            # TODO: SGD update for W1 and b1.
            ____
            ____

        epoch_loss += batch_loss / len(batch_idx)
        n_batches += 1

    avg_loss = epoch_loss / n_batches
    sgd_losses.append(avg_loss)
    print(f"SGD Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: SGD with momentum
# ══════════════════════════════════════════════════════════════════════

W1_mom, b1_mom = copy.deepcopy(W1), copy.deepcopy(b1)
W2_mom, b2_mom = copy.deepcopy(W2), copy.deepcopy(b2)
momentum = 0.9

# TODO: Velocity buffers same shape as W1, b1, W2, b2 initialized to 0.
vW1 = ____
vb1 = ____
vW2 = ____
vb2 = ____

mom_losses = []
for epoch in range(epochs):
    epoch_loss = 0.0
    n_batches = 0
    indices = list(range(min(500, n_samples)))
    random.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        batch_loss = 0.0

        gW2 = [[0.0] * n_classes for _ in range(hidden_size)]
        gb2 = [0.0] * n_classes
        gW1 = [[0.0] * hidden_size for _ in range(n_features)]
        gb1_ = [0.0] * hidden_size

        for idx in batch_idx:
            x_row = X[idx].tolist()
            target = y[idx]
            hidden, h_act, logits, probs = forward(
                x_row, W1_mom, b1_mom, W2_mom, b2_mom
            )
            batch_loss += cross_entropy_loss(probs, target)

            # TODO: Accumulate gradients gW2, gb2, gW1, gb1_ over batch.
            d_logits = ____
            ____  # accumulate gW2
            ____  # accumulate gb2

            d_hidden = ____
            ____  # accumulate gW1
            ____  # accumulate gb1_

        # TODO: Momentum update: v = momentum*v + grad; w -= lr*v for all parameters.
        ____

        epoch_loss += batch_loss / len(batch_idx)
        n_batches += 1

    avg_loss = epoch_loss / n_batches
    mom_losses.append(avg_loss)
    print(f"SGD+Momentum Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Implement Adam optimizer
# ══════════════════════════════════════════════════════════════════════

print("\n=== Adam Optimizer ===")

# Adam update rule:
#   m = beta1 * m + (1 - beta1) * grad
#   v = beta2 * v + (1 - beta2) * grad^2
#   m_hat = m / (1 - beta1^t); v_hat = v / (1 - beta2^t)
#   param -= lr * m_hat / (sqrt(v_hat) + eps)

adam_lr = 0.001
beta1, beta2, eps = 0.9, 0.999, 1e-8

W1_adam, b1_adam = copy.deepcopy(W1), copy.deepcopy(b1)
W2_adam, b2_adam = copy.deepcopy(W2), copy.deepcopy(b2)

# TODO: Initialize first moments (mW1, mb1, mW2, mb2) and second moments (vW1_adam, vb1_adam, vW2_adam, vb2_adam) as zeros.
mW1 = ____
vW1_adam = ____
mb1 = ____
vb1_adam = ____
mW2 = ____
vW2_adam = ____
mb2 = ____
vb2_adam = ____

adam_losses = []
t_step = 0

for epoch in range(epochs):
    epoch_loss = 0.0
    n_batches = 0
    indices = list(range(min(500, n_samples)))
    random.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        batch_loss = 0.0
        t_step += 1

        gW2 = [[0.0] * n_classes for _ in range(hidden_size)]
        gb2 = [0.0] * n_classes
        gW1_batch = [[0.0] * hidden_size for _ in range(n_features)]
        gb1_batch = [0.0] * hidden_size

        for idx in batch_idx:
            x_row = X[idx].tolist()
            target = y[idx]
            hidden, h_act, logits, probs = forward(
                x_row, W1_adam, b1_adam, W2_adam, b2_adam
            )
            batch_loss += cross_entropy_loss(probs, target)

            # TODO: Accumulate gW1_batch, gb1_batch, gW2, gb2.
            d_logits = ____
            ____
            ____
            d_hidden = ____
            ____
            ____

        # TODO: Compute bias corrections bc1 = 1 - beta1^t_step, bc2 = 1 - beta2^t_step.
        bc1 = ____
        bc2 = ____

        # TODO: Adam update for all four parameter groups using m, v, m_hat, v_hat.
        ____

        epoch_loss += batch_loss / len(batch_idx)
        n_batches += 1

    avg_loss = epoch_loss / n_batches
    adam_losses.append(avg_loss)
    print(f"Adam Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Compare convergence curves
# ══════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()
fig = viz.training_history(
    {"SGD": sgd_losses, "SGD+Momentum": mom_losses, "Adam": adam_losses}
)
fig.write_html("optimizer_comparison.html")

print(f"\nSGD final: {sgd_losses[-1]:.4f}")
print(f"SGD+Momentum final: {mom_losses[-1]:.4f}")
print(f"Adam final: {adam_losses[-1]:.4f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Learning rate warmup + cosine decay
# ══════════════════════════════════════════════════════════════════════


def cosine_schedule(
    step: int, total_steps: int, warmup_steps: int, max_lr: float, min_lr: float = 1e-6
) -> float:
    """Cosine annealing with linear warmup."""
    # TODO: Linear warmup: if step < warmup_steps return max_lr * step / warmup_steps.
    if step < warmup_steps:
        return ____
    # TODO: Cosine decay: progress in [0,1], return min_lr + 0.5*(max_lr-min_lr)*(1+cos(pi*progress)).
    progress = ____
    return ____


total_steps = 1000
warmup_steps = 100
max_lr = 0.001

schedule = [
    cosine_schedule(s, total_steps, warmup_steps, max_lr) for s in range(total_steps)
]

print(f"\n=== Cosine Schedule with Warmup ===")
print(
    f"LR at step 0: {schedule[0]:.6f}, step 100 (peak): {schedule[100]:.6f}, step 999: {schedule[999]:.6f}"
)

print("\n✓ Exercise 6 complete — SGD vs Momentum vs Adam + cosine scheduling")
