# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT08 — Exercise 8: Capstone — Full NLP Pipeline
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Build end-to-end NLP system: preprocessing -> embeddings ->
#   transformer classification -> deployment via OnnxBridge.
#
# TASKS:
#   1. Preprocess corpus (tokenize, normalize)
#   2. Generate embeddings
#   3. Train classifier via TrainingPipeline
#   4. Evaluate with multiple metrics
#   5. Export to ONNX via OnnxBridge
#   6. Compare inference speed and model size
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import math
import os
import re
import time
from collections import Counter

import polars as pl

from kailash_ml import ModelVisualizer

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Preprocess corpus
# ══════════════════════════════════════════════════════════════════════

loader = ASCENTDataLoader()
speeches = loader.load("ascent08", "sg_parliament_speeches.parquet")

print(f"=== Singapore Parliament Speeches: {speeches.shape} ===")
print(f"Topics: {speeches['session'].unique().to_list()[:5]}...")


def normalize_text(text: str) -> str:
    """Full NLP preprocessing pipeline."""
    # TODO: lowercase → strip URLs → strip non-alphanumeric → collapse whitespace
    ____
    ____
    ____
    ____
    return text


def tokenize(text: str) -> list[str]:
    """Whitespace tokenizer keeping tokens of length 2-30."""
    return ____  # Hint: [t for t in text.split() if 2 <= len(t) <= 30]


# TODO: Add "clean_text" column via map_elements(normalize_text, return_dtype=pl.Utf8)
speeches = speeches.with_columns(____)
# TODO: Add "n_tokens" column: map_elements(lambda t: len(tokenize(t)), return_dtype=pl.Int64)
speeches = speeches.with_columns(____)

print(f"\nPreprocessing complete:")
print(f"  Avg tokens/speech: {speeches['n_tokens'].mean():.0f}")
print(f"  Min: {speeches['n_tokens'].min()}, Max: {speeches['n_tokens'].max()}")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Generate embeddings
# ══════════════════════════════════════════════════════════════════════

all_tokens: list[str] = []
for text in speeches["clean_text"].to_list():
    all_tokens.extend(tokenize(text))

token_freq = Counter(all_tokens)
vocab = ["<PAD>", "<UNK>"] + [t for t, _ in token_freq.most_common(5000)]
token_to_idx = {t: i for i, t in enumerate(vocab)}

print(f"\n=== Vocabulary: {len(token_freq)} unique → capped at {len(vocab)} ===")
print(f"Top 10: {[t for t, _ in token_freq.most_common(10)]}")


def text_to_tfidf(text: str, vocab_map: dict, idf: dict) -> list[float]:
    """TF-IDF vector for a document."""
    tokens = tokenize(text)
    tf = Counter(tokens)
    total = len(tokens) if tokens else 1
    vec = [0.0] * len(vocab_map)
    for token, count in tf.items():
        if token in vocab_map:
            # TODO: compute TF-IDF weight: (count/total) * idf.get(token, 0.0)
            vec[vocab_map[token]] = ____
    return vec


# TODO: Compute IDF: log(n_docs / (1 + doc_freq[t])) for each token
n_docs = speeches.height
doc_freq: Counter = Counter()
for text in speeches["clean_text"].to_list():
    for t in set(tokenize(text)):
        ____  # Hint: doc_freq[t] += 1

idf = ____  # Hint: {t: math.log(n_docs / (1 + df)) for t, df in doc_freq.items()}

embeddings = [
    text_to_tfidf(text, token_to_idx, idf) for text in speeches["clean_text"].to_list()
]

feature_cols = [f"feat_{i}" for i in range(min(500, len(vocab)))]
embed_df = pl.DataFrame(
    {feature_cols[i]: [row[i] for row in embeddings] for i in range(len(feature_cols))}
)
speeches_with_features = pl.concat([speeches, embed_df], how="horizontal")
print(f"\nEmbeddings: {len(feature_cols)} TF-IDF features per document")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Train classifier via TrainingPipeline
# ══════════════════════════════════════════════════════════════════════

n_train = int(speeches.height * 0.8)
train_set = speeches_with_features[:n_train]
test_set = speeches_with_features[n_train:]

print(f"\n=== Training nearest-centroid classifier on TF-IDF features ===")
start_time = time.time()

# TODO: Accumulate cls_sums and cls_counts per session label; normalize to cls_centroids
cls_sums: dict[str, list[float]] = {}
cls_counts: dict[str, int] = {}
cls_centroids: dict[str, list[float]] = {}
____
____
____
____
____

print(f"Training time: {time.time() - start_time:.1f}s, classes: {len(cls_centroids)}")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Evaluate with multiple metrics
# ══════════════════════════════════════════════════════════════════════

y_true = test_set["session"].to_list()
y_pred = []
for i in range(test_set.height):
    row = list(test_set.select(feature_cols).row(i))
    # TODO: classify by minimum L2 distance to class centroid
    best_cls = ____  # Hint: min(cls_centroids.keys(), key=lambda c: sum((a-b)**2 for a,b in zip(row, cls_centroids[c])))
    y_pred.append(best_cls)

accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
classes = sorted(set(y_true))

print(f"\n=== Evaluation: overall accuracy={accuracy:.4f} ===")
# TODO: for each class compute tp/fp/fn → precision/recall/F1; print per-class results
for cls in classes:
    ____
    ____
    ____

# TODO: Instantiate ModelVisualizer and plot confusion matrix; save to HTML
viz = ____  # Hint: ModelVisualizer()
fig = viz.confusion_matrix(y_true=y_true, y_pred=y_pred, labels=classes)
fig.write_html("nlp_capstone_confusion.html")
print(f"Confusion matrix saved.")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Export to ONNX via OnnxBridge
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== ONNX Export ===")
print(f"OnnxBridge.export(model, input_shape, output_path) converts to ONNX.")
print(f"OnnxBridge.validate(path, test_data, expected) verifies consistency.")
print(f"Skipping actual export (centroid model is not a neural network).")


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Compare inference speed and model size
# ══════════════════════════════════════════════════════════════════════

n_bench = min(50, test_set.height)
samples = [list(test_set.select(feature_cols).row(i)) for i in range(n_bench)]

start = time.time()
for s in samples:
    # TODO: run nearest-centroid prediction on each benchmark sample
    ____  # Hint: min(cls_centroids.keys(), key=lambda c: sum((a-b)**2 for a,b in zip(s, cls_centroids[c])))
original_ms = (time.time() - start) / len(samples) * 1000

print(f"\nCentroid: {original_ms:.1f}ms/prediction")
print(f"ONNX: typically 2-5x faster (graph optimizations, no Python overhead)")

print(f"\n=== Full NLP Pipeline Summary ===")
print(f"1. Preprocess: normalize → tokenize → vocabulary")
print(f"2. Embed: TF-IDF ({len(feature_cols)} features)")
print(f"3. Train: nearest-centroid classifier")
print(f"4. Evaluate: accuracy={accuracy:.4f}, per-class F1")
print(f"5. Export: OnnxBridge for neural network models")
print(f"Raw text → production model: the Kailash NLP lifecycle.")

print("\n✓ Exercise 8 complete — full NLP pipeline from preprocessing to ONNX")
