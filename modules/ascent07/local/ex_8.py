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

# TODO: Normalize pixels to [0, 1] using with_columns and pl.col(c)/255.0 for each pixel column.
normalized = ____

# TODO: Train/test split 80/20 via slicing.
n_train = ____
train_data = ____
test_data = ____

print(f"=== Fashion-MNIST Pipeline ===")
print(f"Total: {n_samples}, Train: {n_train}, Test: {n_samples - n_train}")
print(f"Features: {len(pixel_cols)} pixels (28×28 flattened)")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Train CNN via TrainingPipeline
# ══════════════════════════════════════════════════════════════════════

# Note: In production you would use TrainingPipeline(feature_store, registry).
# Here we demonstrate the pipeline concept with a nearest-centroid classifier.
from collections import Counter
import math

print(f"\n=== Training (Nearest-Centroid Classifier) ===")
start_time = time.time()

class_sums: dict[int, list[float]] = {}
class_counts: dict[int, int] = {}
# TODO: Iterate train_data, accumulating sums of pixel vectors per class and counts per class.
for i in range(train_data.height):
    ____

# TODO: Compute centroids[label] = class_sums[label] / class_counts[label].
centroids = ____

train_time = time.time() - start_time
print(f"Training time: {train_time:.1f}s, classes: {len(centroids)}")

# TODO: Predict on test set via nearest centroid (minimum squared distance).
test_labels = test_data["label"].to_list()
pred_labels = []
for i in range(test_data.height):
    row = list(test_data.select(pixel_cols).row(i))
    best_label = ____
    pred_labels.append(best_label)

# TODO: Compute test_accuracy.
correct = ____
test_accuracy = ____
print(f"Test accuracy: {test_accuracy:.4f}")

viz = ModelVisualizer()


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Register in ModelRegistry
# ══════════════════════════════════════════════════════════════════════


async def register_model():
    # TODO: Create ConnectionManager("sqlite:///capstone_models.db") and initialize.
    conn = ____
    ____

    # TODO: Build ModelRegistry and call register_model with name, pickled centroids, metrics list.
    registry = ____

    version = ____

    # TODO: Promote to production stage via registry.promote_model.
    ____

    print(f"\n=== ModelRegistry ===")
    print(f"Registered: fashion_mnist_cnn v{version.version} (production)")
    print(f"Metrics: accuracy={test_accuracy:.4f}, train_time={train_time:.1f}s")

    return registry, version


registry, model_version = asyncio.run(register_model())


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Export to ONNX via OnnxBridge
# ══════════════════════════════════════════════════════════════════════


async def export_onnx():
    # TODO: Instantiate OnnxBridge.
    bridge = ____

    print(f"\n=== ONNX Export ===")
    print(f"OnnxBridge.export(model, input_shape, output_path) → .onnx file.")
    print(
        f"OnnxBridge.validate(path, test_data, expected) verifies output consistency."
    )

    return bridge, "fashion_mnist_cnn.onnx"


bridge, onnx_path = asyncio.run(export_onnx())


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Deploy via InferenceServer
# ══════════════════════════════════════════════════════════════════════


print(f"\n=== InferenceServer Deployment ===")
print(f"InferenceServer(model_path=..., port=8090) serves ONNX over HTTP.")

class_names = [
    "T-shirt",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

for i in range(3):
    sample = list(test_data.select(pixel_cols).row(i))
    # TODO: Predict best_label via nearest-centroid over sample.
    best_label = ____
    true_label = int(test_data["label"][i])
    print(
        f"  Sample {i+1}: true={class_names[true_label]}, pred={class_names[best_label]}"
    )


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Compare inference speed
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Inference Speed Comparison ===")

n_test = min(100, test_data.height)
test_samples = [list(test_data.select(pixel_cols).row(i)) for i in range(n_test)]

# TODO: Time n_test nearest-centroid predictions; compute original_time.
start = time.time()
for sample in test_samples:
    ____
original_time = time.time() - start

print(
    f"Centroid model: {n_test} preds in {original_time:.3f}s ({original_time/n_test*1000:.1f}ms/pred)"
)
print(f"ONNX typically 2-5x faster due to graph optimizations.")

print(f"\n=== Full Pipeline Summary ===")
print(f"Data → Training → ModelRegistry → OnnxBridge → InferenceServer")

print("\n✓ Exercise 8 complete — end-to-end DL pipeline with Kailash")
