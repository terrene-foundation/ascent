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
# TODO: print rating distribution by iterating label_counts rows
____

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
print(
    f"AutoMLEngine(pipeline, search) automates model selection and hyperparameter search."
)


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
    """TF-IDF vector: for each token, weight = (count/total) * idf[token]."""
    # TODO: tokenize; compute tf; build zero vec; fill TF-IDF weights for in-vocab tokens
    ____
    ____
    ____
    return vec


# TODO: Accumulate class_vecs and class_cts; normalize each centroid
class_vecs: dict[str, list[float]] = {}
class_cts: dict[str, int] = {}
____
____
____
____
____

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

# TODO: Compute accuracy, tp/fp/fn, precision, recall, f1 from y_true and y_pred
accuracy = ____
tp, fp, fn = ____, ____, ____
precision = ____
recall = ____
f1 = ____
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
    # TODO: (1) ConnectionManager("sqlite:///nlp_models.db") → conn.initialize()
    # TODO: (2) ModelRegistry(conn) → register_model with f1/accuracy/precision/recall MetricSpec
    # TODO: (3) promote_model to "production"; list_models; return registry
    conn = ____  # Hint: ConnectionManager("sqlite:///nlp_models.db")
    await conn.initialize()
    registry = ____  # Hint: ModelRegistry(conn)
    version = await registry.register_model(
        name="sg_sentiment_classifier",
        artifact=pickle.dumps(class_vecs),
        metrics=[____, ____, ____, ____],  # Hint: MetricSpec(name=..., value=...) x4
    )
    await registry.promote_model(
        name="sg_sentiment_classifier", version=____, target_stage="production"
    )
    print(
        f"Registered sg_sentiment_classifier v{version.version} → production | F1={f1:.4f}"
    )
    return registry


registry = asyncio.run(register_best())

print(f"\n=== Transfer Learning Summary ===")
print(f"Pre-trained representations + domain fine-tuning = strong baselines")
print(f"AutoMLEngine automates architecture and hyperparameter search.")

print("\n✓ Exercise 7 complete — transformer transfer learning with AutoMLEngine")
