# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT08 — Exercise 5: Attention Mechanisms
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Implement scaled dot-product attention and multi-head
#   attention from scratch, then visualize attention patterns with
#   ModelVisualizer.
#
# TASKS:
#   1. Implement scaled dot-product attention
#   2. Build multi-head attention module
#   3. Visualize attention weights on sample sequences
#   4. Demonstrate how attention solves the context bottleneck
#   5. Compare attention vs LSTM on long sequences
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
df = loader.load("ascent08", "sg_product_reviews.parquet")

explorer = DataExplorer()
summary = asyncio.run(explorer.profile(df))
print(f"=== Dataset: {df.height} reviews ===")
print(summary)


def tokenize(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()


corpus = df.select("review_text").to_series().to_list()
word_counts = Counter(tok for t in corpus for tok in tokenize(t))
vocab = ["<pad>", "<unk>"] + [w for w, c in word_counts.most_common(2000) if c >= 2]
word_to_idx = {w: i for i, w in enumerate(vocab)}
vocab_size = len(vocab)
embed_dim = 32

# Random embeddings for demonstration
embeddings = [
    [random.gauss(0, 0.1) for _ in range(embed_dim)] for _ in range(vocab_size)
]

print(f"Vocabulary: {vocab_size}, embedding dim: {embed_dim}")


def text_to_embeddings(text: str, max_len: int = 20) -> list[list[float]]:
    """Convert text to a sequence of embedding vectors."""
    tokens = tokenize(text)[:max_len]
    indices = [word_to_idx.get(t, 1) for t in tokens]
    return [embeddings[idx] for idx in indices]


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Scaled dot-product attention
# ══════════════════════════════════════════════════════════════════════


def softmax(scores: list[float]) -> list[float]:
    """Numerically stable softmax."""
    # TODO: subtract max for numerical stability
    max_s = ____
    # TODO: exponentiate shifted scores
    exps = ____
    # TODO: normalise by sum
    total = ____
    return ____


def scaled_dot_product_attention(
    Q: list[list[float]],
    K: list[list[float]],
    V: list[list[float]],
) -> tuple[list[list[float]], list[list[float]]]:
    """Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V.

    Returns (output, attention_weights).
    """
    d_k = len(Q[0])
    # TODO: scale = sqrt(d_k)
    scale = ____

    # TODO: scores[i][j] = dot(Q[i], K[j]) / scale — build the seq_len_q x seq_len_k score matrix
    scores = ____

    # TODO: row-wise softmax over keys → attention weights
    weights = ____

    # TODO: output[i] = sum_j weights[i][j] * V[j] — weighted sum of values
    output = ____

    return output, weights


sample_text = corpus[0] if corpus else "this product is great for singapore weather"
sample_emb = text_to_embeddings(sample_text, max_len=10)
sample_tokens = tokenize(sample_text)[:10]

output, attn_weights = scaled_dot_product_attention(sample_emb, sample_emb, sample_emb)

print(f"\n--- Scaled Dot-Product Attention ---")
print(f"Input sequence: {sample_tokens}")
print(f"Q/K/V dim: {len(sample_emb)}x{embed_dim}")
print(f"Output shape: {len(output)}x{len(output[0])}")
print(f"\nAttention weights (first token attends to):")
for j, w in enumerate(attn_weights[0]):
    token = sample_tokens[j] if j < len(sample_tokens) else "?"
    bar = "#" * int(w * 40)
    print(f"  {token:<15} {w:.3f} {bar}")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Multi-head attention
# ══════════════════════════════════════════════════════════════════════


def linear_projection(X: list[list[float]], W: list[list[float]]) -> list[list[float]]:
    """Project X (seq_len x d_in) with W (d_in x d_out)."""
    # TODO: for each row x in X, compute [sum_k x[k]*W[k][j] for j in range(d_out)]
    return ____


class MultiHeadAttention:
    """Multi-head attention: parallel attention heads, then concatenate + project."""

    def __init__(self, d_model: int, n_heads: int):
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_model = d_model

        scale = 1.0 / math.sqrt(self.d_k)
        # TODO: per-head W_q, W_k, W_v — each a list of (d_model x d_k) Gaussian matrices
        self.W_q = ____
        self.W_k = ____
        self.W_v = ____
        # TODO: output projection W_o of shape (d_model x d_model)
        self.W_o = ____

    def forward(
        self, X: list[list[float]]
    ) -> tuple[list[list[float]], list[list[list[float]]]]:
        """Multi-head self-attention. Returns (output, all_head_weights)."""
        all_head_outputs = []
        all_head_weights = []

        # TODO: for each head h: project into Q_h, K_h, V_h via linear_projection
        # TODO: run scaled_dot_product_attention and append results
        for h in range(self.n_heads):
            ____
            ____
            ____
            ____

        # TODO: concatenate heads at each position — concat[i] = [head0[i] ++ head1[i] ++ ...]
        concat = ____

        # TODO: apply output projection W_o via linear_projection
        output = ____
        return output, all_head_weights


n_heads = 4
mha = MultiHeadAttention(d_model=embed_dim, n_heads=n_heads)
mha_output, head_weights = mha.forward(sample_emb)

print(f"\n--- Multi-Head Attention ({n_heads} heads) ---")
print(f"Input: {len(sample_emb)}x{embed_dim}")
print(f"Per-head dim: {embed_dim // n_heads}")
print(f"Output: {len(mha_output)}x{len(mha_output[0])}")

for h in range(n_heads):
    top_attn = sorted(
        range(len(head_weights[h][0])),
        key=lambda j: head_weights[h][0][j],
        reverse=True,
    )[:3]
    top_tokens = [sample_tokens[j] if j < len(sample_tokens) else "?" for j in top_attn]
    print(f"  Head {h}: token[0] attends most to {top_tokens}")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Visualize attention weights
# ══════════════════════════════════════════════════════════════════════

# TODO: instantiate ModelVisualizer
viz = ____

print(f"\n=== Attention visualization generated ===")
print(f"Each head learns different linguistic relationships:")
print(f"  Head 0: may capture syntactic dependencies")
print(f"  Head 1: may capture semantic similarity")
print(f"  Head 2: may capture positional patterns")
print(f"  Head 3: may capture entity co-reference")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Attention solves the context bottleneck
# ══════════════════════════════════════════════════════════════════════

print(f"\n--- Context Bottleneck Problem ---")
print(f"RNN/LSTM: entire sequence compressed into fixed-size vector h_T")
print(f"  Long sequences → early tokens are 'forgotten'")
print(f"  Bottleneck: all information must pass through hidden state")
print(f"\nAttention: each output position can directly attend to ALL inputs")
print(f"  No bottleneck — O(1) path from any input to any output")
print(f"  Trade-off: O(n^2) memory for attention matrix")

if len(sample_tokens) >= 5:
    first_to_last = attn_weights[0][-1]
    last_to_first = attn_weights[-1][0]
    print(
        f"\n  Attention from '{sample_tokens[0]}' to '{sample_tokens[-1]}': {first_to_last:.4f}"
    )
    print(
        f"  Attention from '{sample_tokens[-1]}' to '{sample_tokens[0]}': {last_to_first:.4f}"
    )
    print(f"  → Direct connection regardless of distance!")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Compare attention vs LSTM on long sequences
# ══════════════════════════════════════════════════════════════════════

print(f"\n--- Attention vs LSTM: Long Sequence Scaling ---")
print(
    f"{'Seq Length':<12} {'LSTM path':<15} {'Attention path':<15} {'Attention memory'}"
)
print("-" * 60)
for seq_len in [10, 50, 100, 500, 1000]:
    # TODO: LSTM path length = seq_len (sequential)
    lstm_path = ____
    # TODO: attention path = 1 (direct)
    attn_path = ____
    # TODO: attention memory ~ seq_len**2
    attn_memory = ____
    print(f"  {seq_len:<12} {lstm_path:<15} {attn_path:<15} {attn_memory:,}")

print(f"\nLSTM: O(n) path length, O(n) memory — gradients vanish over long paths")
print(f"Attention: O(1) path length, O(n^2) memory — constant gradient path")
print(f"Transformers combine attention's O(1) paths with parallelism (no recurrence)")

print(
    "\n✓ Exercise 5 complete — scaled dot-product + multi-head attention, visualization"
)
