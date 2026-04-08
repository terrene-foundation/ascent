# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT08 — Exercise 3: Word Embeddings
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Train Word2Vec skip-gram from scratch, explore word analogies,
#   and visualize embedding space with ModelVisualizer (t-SNE).
#
# TASKS:
#   1. Build skip-gram training pairs from corpus
#   2. Implement Word2Vec training loop
#   3. Test word similarity and analogies
#   4. Visualize embeddings with ModelVisualizer (t-SNE)
#   5. Compare with pre-trained GloVe vectors
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import math
import random
import re
from collections import Counter

import polars as pl

from kailash_ml import DataExplorer, ModelVisualizer

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()


# ── Data Loading ──────────────────────────────────────────────────────

loader = ASCENTDataLoader()
df = loader.load("ascent08", "sg_news_articles.parquet")

explorer = DataExplorer()
summary = asyncio.run(explorer.profile(df))
print(f"=== Dataset: {df.height} articles ===")
print(summary)


def tokenize(text: str) -> list[str]:
    """Lowercase, strip non-alphanumeric, split."""
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()


corpus_texts = df.select("text").to_series().to_list()
tokenized_corpus = [tokenize(t) for t in corpus_texts]
all_tokens = [tok for doc in tokenized_corpus for tok in doc]

word_counts = Counter(all_tokens)
vocab = [w for w, c in word_counts.most_common(3000) if c >= 2]
word_to_idx = {w: i for i, w in enumerate(vocab)}
vocab_size = len(vocab)

print(f"Vocabulary: {vocab_size} words, total tokens: {len(all_tokens):,}")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Build skip-gram training pairs
# ══════════════════════════════════════════════════════════════════════


def build_skipgram_pairs(
    tokens: list[str], word_to_idx: dict[str, int], window: int = 2
) -> list[tuple[int, int]]:
    """Generate (center, context) index pairs within a sliding window."""
    # TODO: for each in-vocab token, emit (center_idx, context_idx) for all
    #       context positions j in [max(0, i-window), min(len, i+window+1)) where j != i
    pairs = []
    ____
    ____
    ____
    ____
    return pairs


sample_tokens = [tok for doc in tokenized_corpus[:50] for tok in doc]
pairs = build_skipgram_pairs(sample_tokens, word_to_idx, window=2)
print(f"\nSkip-gram pairs: {len(pairs):,}")
print(f"Sample: {[(vocab[c], vocab[t]) for c, t in pairs[:5]]}")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Word2Vec training loop
# ══════════════════════════════════════════════════════════════════════

embedding_dim = 50

# TODO: Initialize W_center: vocab_size x embedding_dim, each entry random.gauss(0, 0.1)
W_center = ____
# TODO: Initialize W_context: vocab_size x embedding_dim, each entry random.gauss(0, 0.1)
W_context = ____


def sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def dot_product(a: list[float], b: list[float]) -> float:
    """Dot product of two vectors."""
    return sum(x * y for x, y in zip(a, b))


def train_skipgram(
    pairs: list[tuple[int, int]],
    W_center: list[list[float]],
    W_context: list[list[float]],
    epochs: int = 3,
    lr: float = 0.01,
    n_negative: int = 5,
) -> list[float]:
    """Train skip-gram with negative sampling."""
    # TODO: build unigram^0.75 distribution for negative sampling (neg_probs)
    token_freq = [0] * vocab_size
    ____
    ____
    neg_probs = ____

    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        random.shuffle(pairs)
        for center, context in pairs[:5000]:
            # TODO: positive pair — compute score, prob=sigmoid, loss, grad=(prob-1)*lr
            # TODO: update W_center[center] and W_context[context] using cached center_orig
            ____
            ____
            ____
            ____
            ____

            # TODO: for each of n_negative samples drawn from neg_probs:
            #   skip if neg == context
            #   compute score, prob; loss += -log(1-prob+1e-10); grad = prob*lr
            #   update W_center[center] and W_context[neg]
            ____
            ____
            ____
            ____

            epoch_loss += loss
        avg_loss = epoch_loss / min(len(pairs), 5000)
        losses.append(avg_loss)
        print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")
    return losses


print(f"\nTraining skip-gram dim={embedding_dim}...")
losses = train_skipgram(pairs, W_center, W_context, epochs=3, lr=0.01)


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Word similarity and analogies
# ══════════════════════════════════════════════════════════════════════


def cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    # TODO: dot / (||a||*||b||); return 0.0 if either norm is zero
    ____
    ____
    ____
    ____


def most_similar(word: str, top_k: int = 5) -> list[tuple[str, float]]:
    """Find the most similar words by cosine similarity."""
    if word not in word_to_idx:
        return []
    # TODO: get embedding at idx; compute (vocab[i], cosine_sim) for all i != idx;
    #       sort descending by similarity; return top_k
    ____
    ____
    ____


def analogy(a: str, b: str, c: str, top_k: int = 3) -> list[tuple[str, float]]:
    """Solve a:b :: c:? via vector arithmetic (b - a + c)."""
    if any(w not in word_to_idx for w in [a, b, c]):
        return []
    # TODO: vec = emb(b) - emb(a) + emb(c);
    #       rank all vocab words not in {a,b,c} by cosine_sim(vec, W_center[i]);
    #       return top_k
    ____
    ____
    ____


for word in ["singapore", "economy", "government", "policy"]:
    sim = most_similar(word, top_k=5)
    if sim:
        print(f"\nSimilar to '{word}': {sim}")

print(f"\n--- Word Analogies ---")
for a, b, c in [("man", "woman", "king"), ("singapore", "asia", "london")]:
    result = analogy(a, b, c)
    if result:
        print(f"  {a}:{b} :: {c}:? -> {result}")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Visualize embeddings with ModelVisualizer (t-SNE)
# ══════════════════════════════════════════════════════════════════════

top_words = vocab[:100]
top_embeddings = [W_center[word_to_idx[w]] for w in top_words]

# TODO: instantiate ModelVisualizer
viz = ____
print(f"\n=== t-SNE embedding plot: {len(top_words)} words ===")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Compare with pre-trained GloVe vectors
# ══════════════════════════════════════════════════════════════════════

print(f"\n--- Trained vs Pre-trained ---")
print(f"Ours: {vocab_size} words, {embedding_dim}D, {len(pairs):,} pairs")
print(f"GloVe-6B: 400K words, 50-300D, 6B tokens")
print(f"Pre-trained: better general semantics")
print(f"Custom-trained: captures domain-specific terms (e.g. Singapore policy)")

print("\n✓ Exercise 3 complete — Word2Vec skip-gram, analogies, t-SNE visualization")
