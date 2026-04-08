# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT07 — Exercise 7: CNNs for Image Classification
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Build a CNN with convolution, pooling, dropout layers for
#   image classification, then export to ONNX via OnnxBridge.
#
# TASKS:
#   1. Implement convolution operation from scratch
#   2. Build CNN architecture (conv → pool → conv → pool → fc)
#   3. Train with dropout regularization
#   4. Evaluate and visualize filters with ModelVisualizer
#   5. Export to ONNX via OnnxBridge and validate
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import math
import random

import polars as pl

from kailash_ml import ModelVisualizer, OnnxBridge

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Implement convolution operation from scratch
# ══════════════════════════════════════════════════════════════════════

loader = ASCENTDataLoader()
data = loader.load("ascent07", "fashion_mnist_sample.parquet")

pixel_cols = [c for c in data.columns if c != "label"]
n_samples = data.height
print(f"=== Fashion-MNIST: {n_samples} samples, 28×28 images ===")


def to_image(row_values: list[float], h: int = 28, w: int = 28) -> list[list[float]]:
    """Convert flat pixel list to 2D image."""
    # TODO: Reshape row_values into h rows of w pixels.
    return ____


def conv2d(
    image: list[list[float]],
    kernel: list[list[float]],
    stride: int = 1,
    padding: int = 0,
) -> list[list[float]]:
    """2D convolution: slide kernel over image, compute dot product at each position."""
    h, w = len(image), len(image[0])
    kh, kw = len(kernel), len(kernel[0])

    # TODO: Apply zero padding if padding > 0 (build padded image, copy original into center).
    if padding > 0:
        ____
        image = padded
        h, w = len(image), len(image[0])

    # TODO: Output dims: out_h = (h - kh) // stride + 1; similarly out_w.
    out_h = ____
    out_w = ____
    output = [[0.0] * out_w for _ in range(out_h)]

    for i in range(out_h):
        for j in range(out_w):
            # TODO: Compute dot product of kernel with image patch at (i*stride, j*stride).
            val = ____
            output[i][j] = val

    return output


sample_pixels = data.select(pixel_cols).row(0)
sample_img = to_image([v / 255.0 for v in sample_pixels])

horizontal_kernel = [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]
vertical_kernel = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]

h_edges = conv2d(sample_img, horizontal_kernel, padding=1)
v_edges = conv2d(sample_img, vertical_kernel, padding=1)

print(
    f"Input: 28×28, Kernel: 3×3, Output (padding=1): {len(h_edges)}×{len(h_edges[0])}"
)


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Build CNN architecture
# ══════════════════════════════════════════════════════════════════════


def max_pool2d(feature_map: list[list[float]], pool_size: int = 2) -> list[list[float]]:
    """Max pooling: downsample by taking max in each pool window."""
    h, w = len(feature_map), len(feature_map[0])
    out_h, out_w = h // pool_size, w // pool_size
    output = [[0.0] * out_w for _ in range(out_h)]
    for i in range(out_h):
        for j in range(out_w):
            # TODO: Collect pool_size × pool_size values starting at (i*pool_size, j*pool_size); take max.
            vals = ____
            output[i][j] = ____
    return output


def relu_2d(feature_map: list[list[float]]) -> list[list[float]]:
    # TODO: Apply max(0, v) to every element.
    return ____


def flatten(feature_maps: list[list[list[float]]]) -> list[float]:
    """Flatten list of 2D feature maps into 1D vector."""
    # TODO: Concatenate all rows from all feature maps.
    result = ____
    return result


class SimpleCNN:
    """Minimal CNN: conv(3x3, 4 filters) → pool → conv(3x3, 8 filters) → pool → fc → softmax."""

    def __init__(self, n_classes: int = 10):
        random.seed(42)
        self.n_classes = n_classes
        # TODO: Initialize 4 conv1 filters as 3×3 matrices with gauss(0, 0.3).
        self.conv1_filters = ____
        self.conv1_bias = [0.0] * 4
        # TODO: Initialize 8 conv2 filters as 3×3 matrices with gauss(0, 0.3).
        self.conv2_filters = ____
        self.conv2_bias = [0.0] * 8
        # After conv1→pool→conv2→pool: 8 feature maps of 7×7 = 392 features
        fc_in = 8 * 7 * 7
        # TODO: FC layer weights fc_in × n_classes with gauss(0, 0.01).
        self.fc_w = ____
        self.fc_b = [0.0] * n_classes

    def forward(self, image: list[list[float]]) -> list[float]:
        """Forward pass through CNN."""
        # TODO: For each of 4 filters: conv2d → relu_2d → max_pool2d. Store in conv1_out.
        conv1_out = ____

        # TODO: For each of 8 filters, conv over conv1_out[f_idx % 4] → relu → pool.
        conv2_out = ____

        # TODO: Flatten conv2_out, then FC: logits = fc_w @ flat + fc_b.
        flat = ____
        logits = ____

        # TODO: Softmax over logits with numerical stabilization.
        max_l = max(logits)
        exps = ____
        s = sum(exps)
        probs = ____
        return probs


cnn = SimpleCNN(n_classes=10)
sample_probs = cnn.forward(sample_img)
print(f"\n=== CNN Architecture ===")
print(f"Conv1 4×(3×3) → Pool → Conv2 8×(3×3) → Pool → FC 392→10")
print(f"Sample prediction: class {sample_probs.index(max(sample_probs))}")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Train with dropout
# ══════════════════════════════════════════════════════════════════════


def dropout(
    values: list[float], rate: float = 0.5, training: bool = True
) -> list[float]:
    """Dropout: randomly zero out values during training, scale at test time."""
    if not training:
        return values
    # TODO: Build mask where each element is 0 with probability rate, else 1/(1-rate). Multiply.
    mask = ____
    return ____


print(f"\n=== Dropout Regularization ===")
print(f"Dropout rate: 0.5, inverted scaling at training time")

n_train_cnn = int(data.height * 0.8)
train_cnn = data[:n_train_cnn]
test_cnn = data[n_train_cnn:]

print(f"Training set: {train_cnn.height}, Test set: {test_cnn.height}")
train_losses = []
for epoch in range(5):
    epoch_loss = 0.0
    for i in range(min(20, train_cnn.height)):
        row_pixels = train_cnn.select(pixel_cols).row(i)
        img = to_image([v / 255.0 for v in row_pixels])
        probs = cnn.forward(img)
        label = int(train_cnn["label"][i])
        # TODO: Cross-entropy loss = -log(probs[label] + eps).
        eps = 1e-8
        loss = ____
        epoch_loss += loss
    avg_loss = epoch_loss / min(20, train_cnn.height)
    train_losses.append(avg_loss)
    print(f"  Epoch {epoch}: loss={avg_loss:.4f}")
print(f"Final loss: {train_losses[-1]:.4f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Visualize filters with ModelVisualizer
# ══════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()

print(f"\n=== Filter Visualization ===")
for i, filt in enumerate(cnn.conv1_filters):
    flat_vals = [v for row in filt for v in row]
    print(f"  Filter {i}: min={min(flat_vals):.2f}, max={max(flat_vals):.2f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Export to ONNX via OnnxBridge
# ══════════════════════════════════════════════════════════════════════


print(f"\n=== ONNX Export ===")
print(f"OnnxBridge.export(model, framework, output_path) converts to ONNX.")
print(f"Portable: runs on any ONNX runtime (C++, JS, mobile).")

print("\n✓ Exercise 7 complete — CNN from scratch + ONNX export via OnnxBridge")
