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

loader = ASCENTDataLoader()
df = loader.load("ascent08", "sg_parliament_speeches.parquet")

explorer = DataExplorer()
summary = asyncio.run(explorer.profile(df))
print(f"=== Dataset: {df.height} speeches, columns: {df.columns} ===")
print(summary)


def tokenize(text: str) -> list[str]:
    import re

    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()


corpus = df.select("text").to_series().to_list()
tokenized_corpus = [tokenize(doc) for doc in corpus]
avg_len = sum(len(d) for d in tokenized_corpus) / len(tokenized_corpus)
print(f"Corpus: {len(corpus)} docs, avg {avg_len:.0f} tokens/doc")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Bag-of-Words representation
# ══════════════════════════════════════════════════════════════════════


def build_vocabulary(
    tokenized_docs: list[list[str]], max_vocab: int = 2000
) -> list[str]:
    """Build vocabulary from most common tokens across corpus."""
    # TODO: Count tokens across all docs; return top max_vocab words
    ____
    ____
    return ____


def bow_vectorize(tokens: list[str], vocab: list[str]) -> list[int]:
    """Convert token list to BoW vector using vocabulary."""
    # TODO: build word_to_idx; zero vector; count each in-vocab token
    ____
    ____
    for token in tokens:
        ____
    return vector


vocab = build_vocabulary(tokenized_corpus, max_vocab=2000)
print(f"\nVocabulary: {len(vocab)}, top 10: {vocab[:10]}")
bow_vectors = [bow_vectorize(doc, vocab) for doc in tokenized_corpus]
print(f"BoW shape: ({len(bow_vectors)}, {len(bow_vectors[0])})")
nonzero = sum(1 for row in bow_vectors for v in row if v > 0)
____  # TODO: print sparsity: 1 - nonzero/(rows*cols) as percentage


# ══════════════════════════════════════════════════════════════════════
# TASK 2: TF-IDF from formula
# ══════════════════════════════════════════════════════════════════════


def compute_tf(tokens: list[str], vocab: list[str]) -> list[float]:
    """Term frequency: count(t, d) / len(d)."""
    # TODO: word_to_idx; tf list of floats; add 1/doc_len per in-vocab token
    ____
    ____
    for token in tokens:
        ____
    return tf


def compute_idf(tokenized_docs: list[list[str]], vocab: list[str]) -> list[float]:
    """Inverse document frequency: smoothed log((N+1)/(df+1))+1."""
    # TODO: count doc_freq per vocab token; build idf list with smoothed formula
    ____
    ____
    ____
    ____
    return idf


def tfidf_vectorize(
    tokens: list[str], vocab: list[str], idf: list[float]
) -> list[float]:
    """TF-IDF = TF * IDF element-wise."""
    tf = compute_tf(tokens, vocab)
    return ____  # Hint: [t*i for t,i in zip(tf, idf)]


idf_values = compute_idf(tokenized_corpus, vocab)
tfidf_vectors = [tfidf_vectorize(doc, vocab, idf_values) for doc in tokenized_corpus]
print(f"\nTF-IDF shape: ({len(tfidf_vectors)}, {len(tfidf_vectors[0])})")
print(f"IDF range: [{min(idf_values):.2f}, {max(idf_values):.2f}]")
first_tfidf = tfidf_vectors[0]
top_idx = sorted(range(len(first_tfidf)), key=lambda i: first_tfidf[i], reverse=True)[
    :10
]
print(
    f"Top TF-IDF terms (doc 0): {[(vocab[i], f'{first_tfidf[i]:.3f}') for i in top_idx]}"
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
    """BM25: idf=(log((N-df+0.5)/(df+0.5)+1)), tf_norm=(tf*(k1+1))/(tf+k1*(1-b+b*dl/avg_dl))."""
    # TODO: compute n_docs, avg_dl, dl, doc_freq; loop query tokens; sum idf*tf_norm
    ____
    ____
    ____
    ____
    for qt in query_tokens:
        if qt not in set(vocab):
            continue
        df = doc_freq.get(qt, 0)
        idf = ____  # Hint: math.log((n_docs-df+0.5)/(df+0.5)+1)
        tf = doc_counts.get(qt, 0)
        tf_norm = ____  # Hint: (tf*(k1+1))/(tf+k1*(1-b+b*dl/avg_dl))
        score += idf * tf_norm
    return score


query = tokenize("economic policy budget Singapore")
print(f"\n--- BM25 for '{' '.join(query)}' ---")
bm25_scores = sorted(
    [
        (i, bm25_score(query, doc, tokenized_corpus, vocab))
        for i, doc in enumerate(tokenized_corpus)
    ],
    key=lambda x: x[1],
    reverse=True,
)
for rank, (idx, score) in enumerate(bm25_scores[:5]):
    print(f"  Rank {rank+1}: doc[{idx}] score={score:.3f} — {corpus[idx][:60]}...")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Compare retrieval quality
# ══════════════════════════════════════════════════════════════════════


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity: dot(a,b) / (||a|| * ||b||)."""
    # TODO: dot product, norms, return dot/(na*nb); guard zero norms
    dot = ____  # Hint: sum(x*y for x,y in zip(a,b))
    na = ____  # Hint: math.sqrt(sum(x*x for x in a))
    nb = ____  # Hint: math.sqrt(sum(y*y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


query_bow = bow_vectorize(query, vocab)
query_tfidf = tfidf_vectorize(query, vocab, idf_values)

# TODO: rank corpus by cosine_similarity for both BoW and TF-IDF queries
bow_ranked = ____
tfidf_ranked = ____

print(f"\n--- Retrieval Comparison ---")
print(f"{'Rank':<6} {'BoW':<14} {'TF-IDF':<14} {'BM25'}")
for rank in range(min(5, len(corpus))):
    b_idx, b_s = bow_ranked[rank]
    t_idx, t_s = tfidf_ranked[rank]
    bm_idx, bm_s = bm25_scores[rank]
    print(
        f"  {rank+1}: [{b_idx}]={b_s:.3f}  [{t_idx}]={t_s:.3f}  [{bm_idx}]={bm_s:.3f}"
    )

print("\nBoW: raw freq | TF-IDF: reweights rare terms | BM25: length-normalised")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: TF-IDF features for classification via TrainingPipeline
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== TF-IDF Classification ===")
print(f"TF-IDF vectors → features for TrainingPipeline(feature_store, registry)")
print(f"  Feature versioning: FeatureStore")
print(f"  Model versioning:   ModelRegistry")
print(f"  Auto evaluation:    TrainingPipeline.fit()")

print("\n✓ Exercise 2 complete — BoW, TF-IDF, BM25 + TrainingPipeline classification")
