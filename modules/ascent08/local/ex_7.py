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
import math
import pickle
import re
from collections import Counter

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

label_counts = reviews.group_by("rating").agg(pl.len().alias("count")).sort("rating")
print(f"\nRating distribution:")
for row in label_counts.iter_rows():
    print(f"  Rating {row[0]}: {row[1]} reviews")

# TODO: add a "sentiment" column: positive if rating >= 4, negative if <= 2
#       Hint: pl.when(pl.col("rating") >= 4).then(pl.lit("positive")).otherwise(pl.lit("negative"))
reviews = ____

# TODO: 80/20 train/test split on rows
n_train = ____
train_reviews = ____
test_reviews = ____

print(f"\nBinary sentiment: positive (rating ≥ 4) vs negative (rating ≤ 2)")
print(f"Train: {train_reviews.height}, Test: {test_reviews.height}")

print(f"\n=== Transfer Learning ===")
print(f"Pre-trained transformers already understand language structure,")
print(f"grammar, and general semantics from training on billions of tokens.")
print(f"Fine-tuning adapts this knowledge to our specific domain (Singapore")
print(f"product reviews) with relatively few examples.")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Configure AutoMLEngine for text classification
# ══════════════════════════════════════════════════════════════════════

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


# TODO: build a Counter of unique tokens per document across the training set
all_train_tokens: Counter = Counter()
for text in train_reviews["review_text"].to_list():
    ____

# TODO: keep top 1000 tokens as vocabulary
tfidf_vocab = ____
tfidf_idx = {w: i for i, w in enumerate(tfidf_vocab)}
# TODO: IDF per vocab token: log(n_train / (1 + doc_freq))
tfidf_idf = ____


def text_to_vec(text: str) -> list[float]:
    """Convert text to a TF-IDF vector using tfidf_vocab / tfidf_idf."""
    tokens = tokenize_text(text)
    tf = Counter(tokens)
    total = len(tokens) if tokens else 1
    vec = [0.0] * len(tfidf_vocab)
    # TODO: for each token in tf, if in vocab, vec[idx] = (count/total) * idf
    for t, count in tf.items():
        ____
    return vec


# TODO: compute class centroids — average vector per sentiment class
class_vecs: dict[str, list[float]] = {}
class_cts: dict[str, int] = {}
for i in range(train_reviews.height):
    label = train_reviews["sentiment"][i]
    vec = text_to_vec(train_reviews["review_text"][i])
    # TODO: initialise class_vecs[label] and class_cts[label] if not present
    ____
    ____
    # TODO: accumulate vec into class_vecs[label]; increment class_cts[label]
    ____
    ____

# TODO: divide each class centroid by its count to get the mean
for label in class_vecs:
    class_vecs[label] = ____

print(f"Vocabulary: {len(tfidf_vocab)}, Classes: {list(class_vecs.keys())}")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Evaluate with confusion matrix
# ══════════════════════════════════════════════════════════════════════

y_true = test_reviews["sentiment"].to_list()
y_pred = []
for text in test_reviews["review_text"].to_list():
    vec = text_to_vec(text)
    # TODO: predict label = class whose centroid is closest (min squared distance)
    best_label = ____
    y_pred.append(best_label)

# TODO: accuracy = correct / total
correct = ____
accuracy = ____

# TODO: compute TP/FP/FN for "positive" class, then precision, recall, F1
tp = ____
fp = ____
fn = ____
precision = ____
recall = ____
f1 = ____

print(f"\n=== Test Evaluation ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1: {f1:.4f}")

# TODO: instantiate ModelVisualizer
viz = ____
# TODO: fig = viz.confusion_matrix(y_true=..., y_pred=..., labels=["negative","positive"])
fig = ____
fig.write_html("sentiment_confusion_matrix.html")
print(f"Confusion matrix saved to sentiment_confusion_matrix.html")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Register best model in ModelRegistry
# ══════════════════════════════════════════════════════════════════════


async def register_best():
    conn = ConnectionManager("sqlite:///nlp_models.db")
    await conn.initialize()

    registry = ModelRegistry(conn)

    # TODO: registry.register_model(name, artifact=pickle.dumps(class_vecs), metrics=[...])
    version = ____

    # TODO: promote to "production" stage
    ____

    print(f"\n=== ModelRegistry ===")
    print(f"Registered: sg_sentiment_classifier v{version.version}")
    print(f"Stage: production")
    print(f"Metrics: F1={f1:.4f}, accuracy={accuracy:.4f}")

    models = await registry.list_models()
    print(f"Total registered models: {len(models)}")

    return registry


registry = asyncio.run(register_best())

print(f"\n=== Transfer Learning Summary ===")
print(f"Pre-trained transformers provide powerful text representations")
print(f"that can be fine-tuned with relatively few domain-specific examples.")
print(f"AutoMLEngine automates the search across model architectures")
print(f"and hyperparameters, finding the best approach for your data.")

print("\n✓ Exercise 7 complete — transformer transfer learning with AutoMLEngine")
