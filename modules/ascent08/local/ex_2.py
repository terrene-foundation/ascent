# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT08 — Exercise 2: Bag of Words and TF-IDF
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Implement BoW, TF-IDF, and BM25 from scratch, then use them
#   for document classification on Singapore Parliament speeches via
#   TrainingPipeline.
#
# TASKS:
#   1. Build Bag-of-Words representation
#   2. Implement TF-IDF from formula
#   3. Implement BM25 scoring
#   4. Compare retrieval quality across methods
#   5. Use TF-IDF features for text classification via TrainingPipeline
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import math
from collections import Counter

import polars as pl

from kailash_ml import DataExplorer

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()


# ── Data Loading ──────────────────────────────────────────────────────

loader = ASCENTDataLoader()
df = loader.load("ascent08", "sg_parliament_speeches.parquet")

explorer = DataExplorer()
summary = asyncio.run(explorer.profile(df))
print(f"=== Dataset: {df.height} speeches, columns: {df.columns} ===")
print(summary)


# ── Helpers ───────────────────────────────────────────────────────────


def tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer with lowercasing."""
    import re

    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return text.split()


corpus = df.select("text").to_series().to_list()
tokenized_corpus = [tokenize(doc) for doc in corpus]

print(f"Corpus: {len(corpus)} documents")
print(
    f"Avg tokens/doc: {sum(len(d) for d in tokenized_corpus) / len(tokenized_corpus):.0f}"
)


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Bag-of-Words representation
# ══════════════════════════════════════════════════════════════════════


def build_vocabulary(
    tokenized_docs: list[list[str]], max_vocab: int = 2000
) -> list[str]:
    """Build vocabulary from most common tokens across corpus."""
    # TODO: Count tokens across all docs; return top max_vocab words
    #   - Counter() to accumulate; .update(doc) per doc; .most_common(max_vocab)
    ____
    ____
    return ____  # Hint: [word for word, _ in word_counts.most_common(max_vocab)]


def bow_vectorize(tokens: list[str], vocab: list[str]) -> list[int]:
    """Convert token list to BoW vector using vocabulary."""
    # TODO: Build word-to-index dict; init zero vector; increment for each token in vocab
    ____  # Hint: word_to_idx = {w: i for i, w in enumerate(vocab)}
    ____  # Hint: vector = [0] * len(vocab)
    for token in tokens:
        ____  # Hint: if token in word_to_idx: vector[word_to_idx[token]] += 1
    return vector  # Hint: return vector


vocab = build_vocabulary(tokenized_corpus, max_vocab=2000)
print(f"\nVocabulary size: {len(vocab)}, top 10: {vocab[:10]}")

bow_vectors = [bow_vectorize(doc, vocab) for doc in tokenized_corpus]
print(f"BoW matrix shape: ({len(bow_vectors)}, {len(bow_vectors[0])})")

total_entries = len(bow_vectors) * len(bow_vectors[0])
nonzero = sum(1 for row in bow_vectors for v in row if v > 0)
print(f"Sparsity: {1 - nonzero / total_entries:.2%} zeros")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: TF-IDF from formula
# ══════════════════════════════════════════════════════════════════════


def compute_tf(tokens: list[str], vocab: list[str]) -> list[float]:
    """Term frequency: count(t, d) / len(d)."""
    # TODO: Build word_to_idx; init tf list; for each token add 1/doc_len to its index
    ____  # Hint: word_to_idx = {w: i for i, w in enumerate(vocab)}
    ____  # Hint: tf = [0.0] * len(vocab); doc_len = len(tokens) if tokens else 1
    for token in tokens:
        ____  # Hint: if token in word_to_idx: tf[word_to_idx[token]] += 1.0 / doc_len
    return tf


def compute_idf(tokenized_docs: list[list[str]], vocab: list[str]) -> list[float]:
    """Inverse document frequency: log(N / df(t))."""
    # TODO: Count document frequency per vocab word; compute smoothed IDF list
    #   smoothed IDF = log((n_docs + 1) / (df + 1)) + 1
    n_docs = len(tokenized_docs)
    doc_freq = Counter()
    for doc in tokenized_docs:
        for token in set(doc):
            ____  # Hint: if token in set(vocab): doc_freq[token] += 1
    idf = []
    for word in vocab:
        df = doc_freq.get(word, 0)
        ____  # Hint: idf.append(math.log((n_docs + 1) / (df + 1)) + 1)
    return idf


def tfidf_vectorize(
    tokens: list[str], vocab: list[str], idf: list[float]
) -> list[float]:
    """TF-IDF = TF * IDF."""
    tf = compute_tf(tokens, vocab)
    return ____  # Hint: [t * i for t, i in zip(tf, idf)]


idf_values = compute_idf(tokenized_corpus, vocab)
tfidf_vectors = [tfidf_vectorize(doc, vocab, idf_values) for doc in tokenized_corpus]

print(f"\nTF-IDF matrix: ({len(tfidf_vectors)}, {len(tfidf_vectors[0])})")
print(f"IDF range: [{min(idf_values):.2f}, {max(idf_values):.2f}]")

first_tfidf = tfidf_vectors[0]
top_indices = sorted(
    range(len(first_tfidf)), key=lambda i: first_tfidf[i], reverse=True
)[:10]
print(
    f"Top TF-IDF terms (doc 0): {[(vocab[i], f'{first_tfidf[i]:.4f}') for i in top_indices]}"
)


# ══════════════════════════════════════════════════════════════════════
# TASK 3: BM25 scoring
# ══════════════════════════════════════════════════════════════════════


def bm25_score(
    query_tokens: list[str],
    doc_tokens: list[str],
    tokenized_docs: list[list[str]],
    vocab: list[str],
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """BM25 score for a query against a single document."""
    # TODO: Compute n_docs, avg_dl, dl, doc_freq Counter
    #   For each query token: compute BM25 IDF and TF normalization, accumulate score
    #   IDF  = log((n_docs - df + 0.5) / (df + 0.5) + 1)
    #   TF_n = (tf * (k1+1)) / (tf + k1*(1 - b + b*dl/avg_dl))
    n_docs = len(tokenized_docs)
    avg_dl = sum(len(d) for d in tokenized_docs) / n_docs
    dl = len(doc_tokens)
    doc_freq = Counter()
    for doc in tokenized_docs:
        for token in set(doc):
            doc_freq[token] += 1
    score = 0.0
    doc_counts = Counter(doc_tokens)
    for qt in query_tokens:
        if qt not in set(vocab):
            continue
        df = doc_freq.get(qt, 0)
        idf = ____  # Hint: math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
        tf = doc_counts.get(qt, 0)
        tf_norm = ____  # Hint: (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
        score += idf * tf_norm
    return score


query = tokenize("economic policy budget Singapore")
print(f"\n--- BM25 Retrieval: '{' '.join(query)}' ---")

scores = [
    (i, bm25_score(query, doc_tokens, tokenized_corpus, vocab))
    for i, doc_tokens in enumerate(tokenized_corpus)
]
scores.sort(key=lambda x: x[1], reverse=True)
for rank, (idx, score) in enumerate(scores[:5]):
    snippet = corpus[idx][:80].replace("\n", " ")
    print(f"  Rank {rank+1}: doc[{idx}] score={score:.3f} — {snippet}...")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Compare retrieval quality
# ══════════════════════════════════════════════════════════════════════


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    # TODO: dot product / (||a|| * ||b||); return 0 if either norm is zero
    dot = ____  # Hint: sum(x * y for x, y in zip(a, b))
    norm_a = ____  # Hint: math.sqrt(sum(x * x for x in a))
    norm_b = ____  # Hint: math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


query_bow = bow_vectorize(query, vocab)
query_tfidf = tfidf_vectorize(query, vocab, idf_values)

bow_scores = sorted(
    [
        (
            i,
            cosine_similarity(
                [float(v) for v in query_bow], [float(v) for v in bow_vectors[i]]
            ),
        )
        for i in range(len(corpus))
    ],
    key=lambda x: x[1],
    reverse=True,
)
tfidf_scores = sorted(
    [(i, cosine_similarity(query_tfidf, tfidf_vectors[i])) for i in range(len(corpus))],
    key=lambda x: x[1],
    reverse=True,
)
bm25_top = scores[:5]

print(f"\n--- Retrieval Comparison ---")
print(f"{'Rank':<6} {'BoW cos':<12} {'TF-IDF cos':<12} {'BM25':<10}")
print("-" * 40)
for rank in range(min(5, len(corpus))):
    b_idx, b_s = bow_scores[rank]
    t_idx, t_s = tfidf_scores[rank]
    bm_idx, bm_s = bm25_top[rank] if rank < len(bm25_top) else (-1, 0)
    print(
        f"  {rank+1:<6} doc[{b_idx}]={b_s:.3f}  doc[{t_idx}]={t_s:.3f}  doc[{bm_idx}]={bm_s:.3f}"
    )

print("\nBoW: raw frequency, no term importance weighting")
print("TF-IDF: down-weights common terms, highlights distinctive terms")
print("BM25: TF saturation + length normalization, best for retrieval")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: TF-IDF features for classification via TrainingPipeline
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== TF-IDF Classification ===")
print(f"The TF-IDF vectors computed above can be used as features for any classifier.")
print(
    f"In production, TrainingPipeline(feature_store, registry) manages the full lifecycle:"
)
print(f"  - Feature versioning via FeatureStore")
print(f"  - Model versioning via ModelRegistry")
print(f"  - Automated evaluation and comparison")

print(
    "\n✓ Exercise 2 complete — BoW, TF-IDF, BM25 from scratch + TrainingPipeline classification"
)
