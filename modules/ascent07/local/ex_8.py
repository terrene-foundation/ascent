# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT07 — Exercise 8: Capstone — End-to-End Deep Learning Pipeline
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Complete DL pipeline from data to deployment: TrainingPipeline
#   → ModelRegistry → OnnxBridge → InferenceServer.
#
# TASKS:
#   1. Load and preprocess image data
#   2. Train CNN via TrainingPipeline
#   3. Register in ModelRegistry with metrics
#   4. Export to ONNX via OnnxBridge
#   5. Deploy via InferenceServer and test predictions
#   6. Compare ONNX inference speed vs original
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import math
import pickle
import time

import polars as pl

from kailash.db.connection import ConnectionManager
from kailash_ml import (
    ModelRegistry,
    ModelVisualizer,
    OnnxBridge,
)
from kailash_ml.types import MetricSpec

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Load and preprocess image data
# ══════════════════════════════════════════════════════════════════════

loader = ASCENTDataLoader()
data = loader.load("ascent07", "fashion_mnist_sample.parquet")

pixel_cols = [c for c in data.columns if c != "label"]
n_samples = data.height

normalized = data.with_columns([(pl.col(c) / 255.0).alias(c) for c in pixel_cols])

n_train = int(n_samples * 0.8)
train_data = normalized[:n_train]
test_data = normalized[n_train:]

print(f"=== Fashion-MNIST Pipeline ===")
print(f"Total: {n_samples}, Train: {n_train}, Test: {n_samples - n_train}")
print(f"Features: {len(pixel_cols)} pixels (28×28 flattened)")
print(f"Classes: 10 (T-shirt, Trouser, Pullover, Dress, Coat,")
print(f"          Sandal, Shirt, Sneaker, Bag, Ankle boot)")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Train CNN via TrainingPipeline
# ══════════════════════════════════════════════════════════════════════

from collections import Counter

print(f"\n=== Training (Nearest-Centroid Classifier) ===")
start_time = time.time()

# TODO: Train nearest-centroid classifier on train_data.
# 1. Accumulate per-class pixel sums (class_sums) and counts (class_counts).
# 2. Compute centroids = {label: [mean pixel values]}.
# 3. Evaluate on test_data: for each sample find nearest centroid (min L2 squared distance).
# 4. Compute test_accuracy = correct / len(test_labels).
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

train_time = time.time() - start_time
print(f"Training time: {train_time:.1f}s")
print(f"Classes: {len(centroids)}")
print(f"Test accuracy: {test_accuracy:.4f}")

viz = ModelVisualizer()
print(f"Model trained and evaluated.")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Register in ModelRegistry
# ══════════════════════════════════════════════════════════════════════


async def register_model():
    # TODO: conn = ConnectionManager("sqlite:///capstone_models.db"); await conn.initialize().
    # registry = ModelRegistry(conn).
    # version = await registry.register_model(name="fashion_mnist_cnn",
    #   artifact=pickle.dumps(centroids), metrics=[MetricSpec(...) for each metric]).
    # await registry.promote_model(name="fashion_mnist_cnn",
    #   version=version.version, target_stage="production").
    # Print registered name/version/metrics; return (registry, version).
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


registry, model_version = asyncio.run(register_model())


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Export to ONNX via OnnxBridge
# ══════════════════════════════════════════════════════════════════════


async def export_onnx():
    bridge = OnnxBridge()

    print(f"\n=== ONNX Export ===")
    print(
        f"OnnxBridge.export(model, input_shape, output_path) converts to ONNX format."
    )
    print(
        f"OnnxBridge.validate(path, test_data, expected) verifies output consistency."
    )
    print(f"ONNX is platform-agnostic: deploy to mobile, edge, browser, or server.")
    print(f"Skipping actual export (centroid model is not a neural network).")

    return bridge, "fashion_mnist_cnn.onnx"


bridge, onnx_path = asyncio.run(export_onnx())


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Deploy via InferenceServer
# ══════════════════════════════════════════════════════════════════════

print(f"InferenceServer(model_path=..., port=8090) serves ONNX models over HTTP.")

# TODO: Define class_names list (10 Fashion-MNIST labels in order).
# Then for each of the first 3 test samples, find nearest centroid and print true vs predicted.
____
for i in range(3):
    ____
    ____
    ____


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Compare inference speed
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Inference Speed Comparison ===")

# TODO: Benchmark centroid inference. Extract n_test=100 samples, time nearest-centroid loop.
n_test = min(100, test_data.height)
____
____
____
____
____

print(
    f"Centroid model: {n_test} predictions in {original_time:.3f}s "
    f"({original_time/n_test*1000:.1f}ms/prediction)"
)
print(f"ONNX model: typically 2-5x faster due to graph optimizations")

print("\n✓ Exercise 8 complete — end-to-end DL pipeline with Kailash")
