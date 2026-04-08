# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT07 — Exercise 1: Linear Regression as a Neural Network
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Build linear regression as a single-neuron network — forward
#   pass, MSE loss, gradient descent — to show that DL starts from
#   familiar ground.
#
# TASKS:
#   1. Load Singapore HDB resale data
#   2. Implement forward pass (y = wx + b)
#   3. Compute MSE loss
#   4. Implement gradient descent manually
#   5. Compare with polars-native OLS solution
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import math

import polars as pl

from kailash_ml import DataExplorer, ModelVisualizer

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Load Singapore HDB resale data
# ══════════════════════════════════════════════════════════════════════

loader = ASCENTDataLoader()
df = loader.load("ascent07", "hdb_resale_sample.parquet")

explorer = DataExplorer()
summary = asyncio.run(explorer.profile(df))

print(f"Shape: {df.shape}")
print(df.head(5))

x_raw = df["floor_area_sqm"].to_list()
y_raw = df["resale_price"].to_list()

# TODO: Compute x_mean as sum(x_raw)/len(x_raw).
x_mean = ____
# TODO: Compute x_std as sqrt(mean squared deviation from x_mean).
x_std = ____
# TODO: Compute y_mean and y_std the same way.
y_mean = ____
y_std = ____
# TODO: Z-normalise using list comprehensions: (xi - mean) / std.
x_norm = ____
y_norm = ____
n = len(x_norm)

print(f"Feature: floor_area_sqm (mean={x_mean:.1f}, std={x_std:.1f}), Samples: {n}")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Implement forward pass (y = wx + b)
# ══════════════════════════════════════════════════════════════════════

w = 0.0
b = 0.0


def forward(x_i: float, w: float, b: float) -> float:
    """Single neuron forward pass: y = wx + b."""
    # TODO: Return w * x_i + b.
    return ____


y_hat_test = forward(x_norm[0], w, b)
print(f"Prediction: {y_hat_test:.4f} (expected ~0 with zero weights)")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Compute MSE loss
# ══════════════════════════════════════════════════════════════════════


def mse_loss(y_true: list[float], y_pred: list[float]) -> float:
    """Mean Squared Error: L = (1/n) * sum((y - y_hat)^2)."""
    # TODO: Return sum of squared differences divided by len(y_true).
    return ____


predictions = [forward(xi, w, b) for xi in x_norm]
initial_loss = mse_loss(y_norm, predictions)
print(f"Initial loss (w=0, b=0): {initial_loss:.4f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Implement gradient descent manually
# ══════════════════════════════════════════════════════════════════════

# dL/dw = (2/n)*sum((y_hat-y)*x);  dL/db = (2/n)*sum(y_hat-y)

learning_rate = 0.1
epochs = 50
history: dict[str, list] = {"epoch": [], "loss": [], "w": [], "b": []}

for epoch in range(epochs):
    # TODO: Forward pass over x_norm using forward(xi, w, b).
    y_pred = ____
    # TODO: Compute loss via mse_loss.
    loss = ____
    # TODO: Compute gradient dw = (2/n)*sum((yp-yt)*xi).
    dw = ____
    # TODO: Compute gradient db = (2/n)*sum(yp-yt).
    db = ____
    # TODO: Update parameters w and b with learning_rate.
    w = ____
    b = ____
    # TODO: Append epoch, loss, w, b to history lists.
    ____

print(f"Final: w={w:.4f}, b={b:.4f}, loss={history['loss'][-1]:.6f}")
ModelVisualizer().training_history({"loss": history["loss"]})


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Compare with polars-native OLS solution
# ══════════════════════════════════════════════════════════════════════

# OLS closed-form: w = cov(x,y)/var(x),  b = mean(y) - w*mean(x)
norm_df = pl.DataFrame({"x": x_norm, "y": y_norm})

# TODO: Compute cov_xy = mean((x - mean(x)) * (y - mean(y))) via polars select/.item().
cov_xy = ____
# TODO: Compute var_x = pl.col("x").var() via polars select/.item().
var_x = ____
# TODO: w_ols = cov_xy / var_x.
w_ols = ____
# TODO: b_ols = mean(y_norm) - w_ols * mean(x_norm).
b_ols = ____
# TODO: ols_loss via mse_loss on predictions w_ols*xi + b_ols.
ols_loss = ____

print(f"GD:  w={w:.4f}, b={b:.4f}, loss={history['loss'][-1]:.6f}")
print(f"OLS: w={w_ols:.4f}, b={b_ols:.4f}, loss={ols_loss:.6f}")
print(f"Difference in w: {abs(w - w_ols):.6f}")
print(f"Difference in b: {abs(b - b_ols):.6f}")

print("\n✓ Exercise 1 complete — linear regression as a single-neuron network")
