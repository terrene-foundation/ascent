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

print(f"=== Singapore Product Reviews: {reviews.shape} ===")
print(f"Columns: {reviews.columns}")

label_counts = reviews.group_by("rating").agg(pl.len().alias("count")).sort("rating")
print(f"\nRating distribution:")
for row in label_counts.iter_rows():
    print(f"  Rating {row[0]}: {row[1]} reviews")

# TODO: Add "sentiment" column — "positive" when rating >= 4, else "negative"
reviews = reviews.with_columns(
    ____  # Hint: pl.when(pl.col("rating") >= 4).then(pl.lit("positive")).otherwise(pl.lit("negative")).alias("sentiment")
)

# TODO: Split 80% train / 20% test
n_train = ____  # Hint: int(reviews.height * 0.8)
train_reviews = ____  # Hint: reviews[:n_train]
test_reviews = ____  # Hint: reviews[n_train:]

print(f"\nTrain: {train_reviews.height}, Test: {test_reviews.height}")
print(f"\n=== Transfer Learning ===")
print(f"Pre-trained transformers understand language from billions of tokens.")
print(f"Fine-tuning adapts to Singapore product reviews with fewer examples.")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Configure AutoMLEngine for text classification
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== AutoMLEngine Configuration ===")
print(f"AutoMLEngine(pipeline, search) automates model selection across")
print(f"algorithms, hyperparameters, and text representations.")
print(f"Building TF-IDF + nearest-centroid classifier below.")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Fine-tune on domain-specific data
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Training TF-IDF Classifier ===")


def tokenize_text(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()


# TODO: Count document-level token frequencies across training set
all_train_tokens: Counter = Counter()
for text in train_reviews["review_text"].to_list():
    ____  # Hint: all_train_tokens.update(set(tokenize_text(text)))

# TODO: Build vocabulary (top 1000) and IDF lookup
tfidf_vocab = ____  # Hint: [w for w, c in all_train_tokens.most_common(1000)]
tfidf_idx = {w: i for i, w in enumerate(tfidf_vocab)}
tfidf_idf = ____  # Hint: {w: math.log(train_reviews.height / (1 + c)) for w, c in all_train_tokens.items()}


def text_to_vec(text: str) -> list[float]:
    tokens = tokenize_text(text)
    tf = Counter(tokens)
    total = len(tokens) if tokens else 1
    vec = [0.0] * len(tfidf_vocab)
    for t, count in tf.items():
        if t in tfidf_idx:
            vec[tfidf_idx[t]] = (count / total) * tfidf_idf.get(t, 0.0)
    return vec


# TODO: Compute per-class centroid vectors from training data
class_vecs: dict[str, list[float]] = {}
class_cts: dict[str, int] = {}
for i in range(train_reviews.height):
    label = train_reviews["sentiment"][i]
    vec = text_to_vec(train_reviews["review_text"][i])
    if label not in class_vecs:
        class_vecs[label] = [0.0] * len(tfidf_vocab)
        class_cts[label] = 0
    for j in range(len(vec)):
        ____  # Hint: class_vecs[label][j] += vec[j]
    ____  # Hint: class_cts[label] += 1

# TODO: Normalize each centroid
for label in class_vecs:
    class_vecs[label] = ____  # Hint: [v / class_cts[label] for v in class_vecs[label]]

print(f"Vocab: {len(tfidf_vocab)}, Classes: {list(class_vecs.keys())}")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Evaluate with confusion matrix
# ══════════════════════════════════════════════════════════════════════

y_true = test_reviews["sentiment"].to_list()
y_pred = []
for text in test_reviews["review_text"].to_list():
    vec = text_to_vec(text)
    # TODO: Predict class with minimum L2 distance to class centroid
    best_label = ____  # Hint: min(class_vecs.keys(), key=lambda c: sum((a-b)**2 for a,b in zip(vec, class_vecs[c])))
    y_pred.append(best_label)

correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
accuracy = correct / len(y_true)
tp = sum(1 for t, p in zip(y_true, y_pred) if t == "positive" and p == "positive")
fp = sum(1 for t, p in zip(y_true, y_pred) if t == "negative" and p == "positive")
fn = sum(1 for t, p in zip(y_true, y_pred) if t == "positive" and p == "negative")
precision = tp / max(tp + fp, 1)
recall = tp / max(tp + fn, 1)
f1 = 2 * precision * recall / max(precision + recall, 1e-10)

print(f"\n=== Test Evaluation ===")
print(
    f"Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}"
)

# TODO: Instantiate ModelVisualizer and plot confusion matrix; save to HTML
viz = ____  # Hint: ModelVisualizer()
fig = viz.confusion_matrix(
    y_true=y_true, y_pred=y_pred, labels=["negative", "positive"]
)
fig.write_html("sentiment_confusion_matrix.html")
print(f"Confusion matrix saved to sentiment_confusion_matrix.html")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Register best model in ModelRegistry
# ══════════════════════════════════════════════════════════════════════


async def register_best():
    # TODO: Initialize ConnectionManager and ModelRegistry
    conn = ____  # Hint: ConnectionManager("sqlite:///nlp_models.db")
    await conn.initialize()
    registry = ____  # Hint: ModelRegistry(conn)

    # TODO: Register model with all four metrics
    version = await registry.register_model(
        name="sg_sentiment_classifier",
        artifact=pickle.dumps(class_vecs),
        metrics=[
            ____,  # Hint: MetricSpec(name="f1", value=f1)
            ____,  # Hint: MetricSpec(name="accuracy", value=accuracy)
            ____,  # Hint: MetricSpec(name="precision", value=precision)
            ____,  # Hint: MetricSpec(name="recall", value=recall)
        ],
    )

    # TODO: Promote the registered version to "production" stage
    await registry.promote_model(
        name="sg_sentiment_classifier",
        version=____,  # Hint: version.version
        target_stage="production",
    )

    print(f"\n=== ModelRegistry ===")
    print(f"Registered sg_sentiment_classifier v{version.version} → production")
    print(f"F1={f1:.4f}, accuracy={accuracy:.4f}")
    models = await registry.list_models()
    print(f"Total registered: {len(models)}")
    return registry


registry = asyncio.run(register_best())

print(f"\n=== Transfer Learning Summary ===")
print(f"Pre-trained representations + domain fine-tuning = strong baselines")
print(f"AutoMLEngine automates architecture and hyperparameter search.")

print("\n✓ Exercise 7 complete — transformer transfer learning with AutoMLEngine")
