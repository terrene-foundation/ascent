# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT08 — Exercise 8: Capstone — Full NLP Pipeline
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Build end-to-end NLP system: preprocessing → embeddings →
#   transformer classification → deployment via OnnxBridge.
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

print(f"=== Singapore Parliament Speeches ===")
print(f"Shape: {speeches.shape}")
print(f"Columns: {speeches.columns}")
print(f"Topics: {speeches['session'].unique().to_list()[:5]}...")


def normalize_text(text: str) -> str:
    """Full NLP preprocessing: lowercase, strip URLs, non-alphanum, collapse whitespace."""
    # TODO: lowercase the text
    text = ____
    # TODO: remove URLs via regex r"http\S+|www\.\S+"
    text = ____
    # TODO: replace non-alphanumeric characters with a space
    text = ____
    # TODO: collapse whitespace runs to a single space and strip
    text = ____
    return text


def tokenize(text: str) -> list[str]:
    """Whitespace tokenizer with 2-30 character length filter."""
    # TODO: split on whitespace, keep tokens with 2 <= len <= 30
    return ____


# TODO: add "clean_text" column by applying normalize_text
speeches = ____
# TODO: add "n_tokens" column by mapping len(tokenize(t)) over clean_text
speeches = ____

print(f"\nPreprocessing complete:")
print(f"  Avg tokens per speech: {speeches['n_tokens'].mean():.0f}")
print(f"  Min: {speeches['n_tokens'].min()}, Max: {speeches['n_tokens'].max()}")
print(f"\nSample (first 100 chars):")
print(f"  Original: {speeches['text'][0][:100]}...")
print(f"  Cleaned:  {speeches['clean_text'][0][:100]}...")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Generate embeddings
# ══════════════════════════════════════════════════════════════════════

# TODO: collect all tokens across all clean_text entries
all_tokens = []
for text in speeches["clean_text"].to_list():
    ____

# TODO: build vocab: ["<PAD>", "<UNK>"] + top 5000 most common tokens
token_freq = Counter(all_tokens)
vocab = ____
token_to_idx = {t: i for i, t in enumerate(vocab)}

print(f"\n=== Vocabulary ===")
print(f"Total unique tokens: {len(token_freq)}")
print(f"Vocabulary size (capped): {len(vocab)}")
print(f"Top 10: {[t for t, _ in token_freq.most_common(10)]}")


def text_to_tfidf(text: str, vocab_map: dict, idf: dict) -> list[float]:
    """Convert text to a TF-IDF vector."""
    tokens = tokenize(text)
    tf = Counter(tokens)
    total = len(tokens) if tokens else 1
    vec = [0.0] * len(vocab_map)
    # TODO: for each token in tf, if in vocab_map, vec[idx] = (count/total) * idf.get(token, 0)
    for token, count in tf.items():
        ____
    return vec


# TODO: compute doc_freq[token] = count of docs containing token
n_docs = speeches.height
doc_freq = Counter()
for text in speeches["clean_text"].to_list():
    ____

# TODO: idf[t] = log(n_docs / (1 + df))
idf = ____

# TODO: compute embeddings list: text_to_tfidf for every clean_text row
embeddings = ____

feature_cols = [f"feat_{i}" for i in range(len(vocab))]
embed_df = pl.DataFrame(
    {
        feature_cols[i]: [row[i] for row in embeddings]
        for i in range(min(500, len(vocab)))
    }
)
feature_cols = feature_cols[:500]

speeches_with_features = pl.concat([speeches, embed_df], how="horizontal")
print(f"\nEmbeddings: {len(feature_cols)} TF-IDF features per document")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Train classifier via TrainingPipeline
# ══════════════════════════════════════════════════════════════════════

n_train = int(speeches.height * 0.8)
train_set = speeches_with_features[:n_train]
test_set = speeches_with_features[n_train:]

print(f"\n=== Training (Nearest-Centroid on TF-IDF) ===")
start_time = time.time()

# TODO: compute per-class centroid by accumulating feature rows and dividing by count
cls_sums: dict[str, list[float]] = {}
cls_counts: dict[str, int] = {}
for i in range(train_set.height):
    label = train_set["session"][i]
    row = list(train_set.select(feature_cols).row(i))
    # TODO: initialise cls_sums[label] and cls_counts[label] if absent
    ____
    ____
    # TODO: add row into cls_sums[label] elementwise; cls_counts[label] += 1
    ____
    ____

cls_centroids = {}
# TODO: for each label, divide cls_sums[label] by cls_counts[label]
for label in cls_sums:
    cls_centroids[label] = ____

train_time = time.time() - start_time
print(f"Training time: {train_time:.1f}s")
print(f"Classes: {len(cls_centroids)}")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Evaluate with multiple metrics
# ══════════════════════════════════════════════════════════════════════

y_true = test_set["session"].to_list()
y_pred = []
for i in range(test_set.height):
    row = list(test_set.select(feature_cols).row(i))
    # TODO: predict label = closest centroid by squared Euclidean distance
    best_cls = ____
    y_pred.append(best_cls)

# TODO: overall accuracy
correct = ____
accuracy = ____

classes = list(set(y_true))
print(f"\n=== Evaluation ===")
print(f"Overall accuracy: {accuracy:.4f}")
print(f"\nPer-class performance:")
for cls in sorted(classes):
    # TODO: per-class TP/FP/FN → precision, recall, F1
    tp = ____
    fp = ____
    fn = ____
    prec = ____
    rec = ____
    f1 = ____
    print(f"  {cls}: precision={prec:.3f}, recall={rec:.3f}, F1={f1:.3f}")

# TODO: instantiate ModelVisualizer and plot confusion matrix
viz = ____
fig = ____
fig.write_html("nlp_capstone_confusion.html")
print(f"Confusion matrix saved.")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Export to ONNX via OnnxBridge
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== ONNX Export ===")
print(f"OnnxBridge.export(model, input_shape, output_path) converts to ONNX.")
print(f"OnnxBridge.validate(path, test_data, expected) verifies consistency.")
print(f"Skipping actual export (centroid model is not a neural network).")
onnx_path = "nlp_classifier.onnx"


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Compare inference speed and model size
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Model Comparison ===")

n_bench = min(50, test_set.height)
samples = [list(test_set.select(feature_cols).row(i)) for i in range(n_bench)]

# TODO: time how long it takes to predict each sample (min over centroids)
start = time.time()
for s in samples:
    ____
original_ms = (time.time() - start) / len(samples) * 1000

print(f"Centroid model: {original_ms:.1f}ms per prediction")
print(f"ONNX: typically 2-5x faster (graph optimizations, no Python overhead)")

print(f"\n=== Full NLP Pipeline Summary ===")
print(f"1. Preprocessing: normalize → tokenize → vocabulary")
print(f"2. Embeddings: TF-IDF (500 features)")
print(f"3. Training: Nearest-centroid classifier (demo)")
print(f"4. Evaluation: accuracy={accuracy:.4f}, per-class F1")
print(f"5. Export: OnnxBridge (for neural network models)")
print(f"From raw text to production-ready model — the Kailash NLP lifecycle.")

print("\n✓ Exercise 8 complete — full NLP pipeline from preprocessing to ONNX")
