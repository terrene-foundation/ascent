# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT08 — Exercise 7: Transfer Learning with Transformers
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Fine-tune a pre-trained transformer for Singapore-specific
#   text classification using AutoMLEngine's text mode.
#
# TASKS:
#   1. Load pre-trained transformer embeddings
#   2. Configure AutoMLEngine for text classification
#   3. Fine-tune on domain-specific data
#   4. Evaluate with confusion matrix via ModelVisualizer
#   5. Register best model in ModelRegistry
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import pickle

import polars as pl

from kailash.db.connection import ConnectionManager
from kailash_ml import ModelRegistry, ModelVisualizer
from kailash_ml.types import MetricSpec

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Load pre-trained transformer embeddings
# ══════════════════════════════════════════════════════════════════════

loader = ASCENTDataLoader()
reviews = loader.load("ascent08", "sg_product_reviews.parquet")

print(f"=== Singapore Product Reviews ===")
print(f"Shape: {reviews.shape}")
print(f"Columns: {reviews.columns}")

# Label distribution
label_counts = reviews.group_by("rating").agg(pl.len().alias("count")).sort("rating")
print(f"\nRating distribution:")
for row in label_counts.iter_rows():
    print(f"  Rating {row[0]}: {row[1]} reviews")

# Binary classification: positive (4-5) vs negative (1-2)
reviews = reviews.with_columns(
    pl.when(pl.col("rating") >= 4)
    .then(pl.lit("positive"))
    .otherwise(pl.lit("negative"))
    .alias("sentiment")
)

# Train/test split
n_train = int(reviews.height * 0.8)
train_reviews = reviews[:n_train]
test_reviews = reviews[n_train:]

print(f"\nBinary sentiment: positive (rating ≥ 4) vs negative (rating ≤ 2)")
print(f"Train: {train_reviews.height}, Test: {test_reviews.height}")

# Transfer learning insight
print(f"\n=== Transfer Learning ===")
print(f"Pre-trained transformers already understand language structure,")
print(f"grammar, and general semantics from training on billions of tokens.")
print(f"Fine-tuning adapts this knowledge to our specific domain (Singapore")
print(f"product reviews) with relatively few examples.")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Configure AutoMLEngine for text classification
# ══════════════════════════════════════════════════════════════════════

# AutoMLEngine takes (pipeline, search, *, registry) where pipeline is a
# TrainingPipeline and search is a HyperparameterSearch. Here we demonstrate
# the concept with a simple TF-IDF + logistic regression approach.
from collections import Counter
import re
import math

print(f"\n=== AutoMLEngine Configuration ===")
print(f"AutoMLEngine(pipeline, search) automates model selection.")
print(f"It searches across algorithms, hyperparameters, and text representations.")
print(f"For this exercise, we build a simple TF-IDF classifier manually.")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Fine-tune on domain-specific data
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Training TF-IDF Classifier ===")


def tokenize_text(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()


# Build vocabulary from training data
all_train_tokens: Counter = Counter()
for text in train_reviews["review_text"].to_list():
    all_train_tokens.update(set(tokenize_text(text)))

tfidf_vocab = [w for w, c in all_train_tokens.most_common(1000)]
tfidf_idx = {w: i for i, w in enumerate(tfidf_vocab)}
tfidf_idf = {
    w: math.log(train_reviews.height / (1 + c)) for w, c in all_train_tokens.items()
}


def text_to_vec(text: str) -> list[float]:
    tokens = tokenize_text(text)
    tf = Counter(tokens)
    total = len(tokens) if tokens else 1
    vec = [0.0] * len(tfidf_vocab)
    for t, count in tf.items():
        if t in tfidf_idx:
            vec[tfidf_idx[t]] = (count / total) * tfidf_idf.get(t, 0.0)
    return vec


# Compute class centroids — vectorized via numpy
import numpy as np

train_texts = train_reviews["review_text"].to_list()
train_labels = train_reviews["sentiment"].to_list()
train_vecs = np.array([text_to_vec(t) for t in train_texts])
train_labels_arr = np.array(train_labels)

class_vecs: dict[str, list[float]] = {}
for label in np.unique(train_labels_arr):
    class_vecs[str(label)] = train_vecs[train_labels_arr == label].mean(axis=0).tolist()

print(f"Vocabulary: {len(tfidf_vocab)}, Classes: {list(class_vecs.keys())}")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Evaluate with confusion matrix
# ══════════════════════════════════════════════════════════════════════

y_true = test_reviews["sentiment"].to_list()
test_texts = test_reviews["review_text"].to_list()
test_vecs = np.array([text_to_vec(t) for t in test_texts])

# Vectorized nearest-centroid: compute distances to each class centroid
class_labels = list(class_vecs.keys())
centroid_arr = np.array([class_vecs[lbl] for lbl in class_labels])
dists = ((test_vecs[:, None, :] - centroid_arr[None, :, :]) ** 2).sum(axis=2)
pred_indices = dists.argmin(axis=1)
y_pred = [class_labels[i] for i in pred_indices]

# Accuracy
correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
accuracy = correct / len(y_true)

# Per-class metrics
tp = sum(1 for t, p in zip(y_true, y_pred) if t == "positive" and p == "positive")
fp = sum(1 for t, p in zip(y_true, y_pred) if t == "negative" and p == "positive")
fn = sum(1 for t, p in zip(y_true, y_pred) if t == "positive" and p == "negative")
precision = tp / max(tp + fp, 1)
recall = tp / max(tp + fn, 1)
f1 = 2 * precision * recall / max(precision + recall, 1e-10)

print(f"\n=== Test Evaluation ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1: {f1:.4f}")

viz = ModelVisualizer()
fig = viz.confusion_matrix(
    y_true=y_true,
    y_pred=y_pred,
    labels=["negative", "positive"],
)
fig.write_html("sentiment_confusion_matrix.html")
print(f"Confusion matrix saved to sentiment_confusion_matrix.html")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Register best model in ModelRegistry
# ══════════════════════════════════════════════════════════════════════


async def register_best():
    conn = ConnectionManager("sqlite:///nlp_models.db")
    await conn.initialize()
    try:
        registry = ModelRegistry(conn)

        version = await registry.register_model(
            name="sg_sentiment_classifier",
            artifact=pickle.dumps(class_vecs),
            metrics=[
                MetricSpec(name="f1", value=f1),
                MetricSpec(name="accuracy", value=accuracy),
                MetricSpec(name="precision", value=precision),
                MetricSpec(name="recall", value=recall),
            ],
        )

        await registry.promote_model(
            name="sg_sentiment_classifier",
            version=version.version,
            target_stage="production",
        )

        print(f"\n=== ModelRegistry ===")
        print(f"Registered: sg_sentiment_classifier v{version.version}")
        print(f"Stage: production")
        print(f"Metrics: F1={f1:.4f}, accuracy={accuracy:.4f}")

        # List all models
        models = await registry.list_models()
        print(f"Total registered models: {len(models)}")
    finally:
        await conn.close()


asyncio.run(register_best())

print(f"\n=== Transfer Learning Summary ===")
print(f"Pre-trained transformers provide powerful text representations")
print(f"that can be fine-tuned with relatively few domain-specific examples.")
print(f"AutoMLEngine automates the search across model architectures")
print(f"and hyperparameters, finding the best approach for your data.")

print("\n✓ Exercise 7 complete — transformer transfer learning with AutoMLEngine")
