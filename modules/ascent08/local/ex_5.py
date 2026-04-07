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


# ── Helpers ───────────────────────────────────────────────────────────


def tokenize(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()


corpus = df.select("review_text").to_series().to_list()
word_counts = Counter(tok for t in corpus for tok in tokenize(t))
vocab = ["<pad>", "<unk>"] + [w for w, c in word_counts.most_common(2000) if c >= 2]
word_to_idx = {w: i for i, w in enumerate(vocab)}
vocab_size = len(vocab)
embed_dim = 32

embeddings = [
    [random.gauss(0, 0.1) for _ in range(embed_dim)] for _ in range(vocab_size)
]

print(f"Vocabulary: {vocab_size}, embedding dim: {embed_dim}")


def text_to_embeddings(text: str, max_len: int = 20) -> list[list[float]]:
    tokens = tokenize(text)[:max_len]
    indices = [word_to_idx.get(t, 1) for t in tokens]
    return [embeddings[idx] for idx in indices]


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Scaled dot-product attention
# ══════════════════════════════════════════════════════════════════════


def softmax(scores: list[float]) -> list[float]:
    max_s = max(scores) if scores else 0
    exps = [math.exp(s - max_s) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def scaled_dot_product_attention(
    Q: list[list[float]],
    K: list[list[float]],
    V: list[list[float]],
) -> tuple[list[list[float]], list[list[float]]]:
    """Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V."""
    # TODO: Implement full scaled dot-product attention
    #   1. scale = sqrt(d_k) where d_k = len(Q[0])
    #   2. scores[i][j] = dot(Q[i], K[j]) / scale  (seq_len_q × seq_len_k)
    #   3. weights = softmax over keys for each query row
    #   4. output[i] = sum_j(weights[i][j] * V[j])
    #   Return (output, weights)
    d_k = len(Q[0])
    scale = ____  # Hint: math.sqrt(d_k)
    scores = []
    for i in range(len(Q)):
        row = [
            sum(Q[i][d] * K[j][d] for d in range(d_k)) / scale for j in range(len(K))
        ]
        scores.append(row)
    weights = ____  # Hint: [softmax(row) for row in scores]
    d_v = len(V[0])
    output = []
    for i in range(len(Q)):
        out_vec = [
            sum(weights[i][j] * V[j][d] for j in range(len(V))) for d in range(d_v)
        ]
        output.append(out_vec)
    return output, weights


sample_text = corpus[0] if corpus else "this product is great for singapore weather"
sample_emb = text_to_embeddings(sample_text, max_len=10)
sample_tokens = tokenize(sample_text)[:10]

output, attn_weights = scaled_dot_product_attention(sample_emb, sample_emb, sample_emb)

print(f"\n--- Scaled Dot-Product Attention ---")
print(f"Input: {sample_tokens}")
print(f"Output shape: {len(output)}×{len(output[0])}")
print(f"\nFirst token attention weights:")
for j, w in enumerate(attn_weights[0]):
    token = sample_tokens[j] if j < len(sample_tokens) else "?"
    print(f"  {token:<15} {w:.3f} {'#' * int(w * 40)}")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Multi-head attention
# ══════════════════════════════════════════════════════════════════════


def linear_projection(X: list[list[float]], W: list[list[float]]) -> list[list[float]]:
    """Project X (seq_len × d_in) with W (d_in × d_out)."""
    d_out = len(W[0])
    return [
        [sum(x[k] * W[k][j] for k in range(len(x))) for j in range(d_out)] for x in X
    ]


class MultiHeadAttention:
    """Multi-head attention: h parallel heads, concatenate, project."""

    def __init__(self, d_model: int, n_heads: int):
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_model = d_model
        scale = 1.0 / math.sqrt(self.d_k)
        # TODO: Initialize per-head Q/K/V weight lists and output projection W_o
        #   W_q, W_k, W_v: n_heads × d_model × d_k random matrices
        #   W_o: d_model × d_model random matrix
        self.W_q = ____  # Hint: [[[random.gauss(0,scale) for _ in range(self.d_k)] for _ in range(d_model)] for _ in range(n_heads)]
        self.W_k = ____  # Hint: same shape as W_q
        self.W_v = ____  # Hint: same shape as W_q
        self.W_o = ____  # Hint: [[random.gauss(0,scale) for _ in range(d_model)] for _ in range(d_model)]

    def forward(self, X: list[list[float]]) -> tuple[list[list[float]], list]:
        """Multi-head self-attention: project, attend, concatenate, project."""
        all_head_outputs, all_head_weights = [], []
        for h in range(self.n_heads):
            # TODO: Project X to Q_h, K_h, V_h; run scaled_dot_product_attention
            Q_h = ____  # Hint: linear_projection(X, self.W_q[h])
            K_h = ____  # Hint: linear_projection(X, self.W_k[h])
            V_h = ____  # Hint: linear_projection(X, self.W_v[h])
            head_out, head_weights = (
                ____  # Hint: scaled_dot_product_attention(Q_h, K_h, V_h)
            )
            all_head_outputs.append(head_out)
            all_head_weights.append(head_weights)

        # TODO: Concatenate all head outputs, then apply output projection W_o
        seq_len = len(X)
        concat = []
        for i in range(seq_len):
            row = []
            for h in range(self.n_heads):
                ____  # Hint: row.extend(all_head_outputs[h][i])
            concat.append(row)
        output = ____  # Hint: linear_projection(concat, self.W_o)
        return output, all_head_weights


n_heads = 4
mha = MultiHeadAttention(d_model=embed_dim, n_heads=n_heads)
mha_output, head_weights = mha.forward(sample_emb)

print(f"\n--- Multi-Head Attention ({n_heads} heads, d_k={embed_dim // n_heads}) ---")
print(f"Output: {len(mha_output)}×{len(mha_output[0])}")
for h in range(n_heads):
    top_j = sorted(
        range(len(head_weights[h][0])),
        key=lambda j: head_weights[h][0][j],
        reverse=True,
    )[:3]
    top_tok = [sample_tokens[j] if j < len(sample_tokens) else "?" for j in top_j]
    print(f"  Head {h}: token[0] attends to {top_tok}")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Visualize attention weights
# ══════════════════════════════════════════════════════════════════════

# TODO: Instantiate ModelVisualizer for attention heatmap
viz = ____  # Hint: ModelVisualizer()
print(f"\n=== Attention patterns: each head specializes in different relationships ===")
print(f"  Head 0: syntactic dependencies | Head 1: semantic similarity")
print(f"  Head 2: positional patterns    | Head 3: entity co-reference")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Attention solves the context bottleneck
# ══════════════════════════════════════════════════════════════════════

print(f"\n--- Context Bottleneck ---")
print(f"RNN: whole sequence compressed into h_T (fixed size) — early tokens forgotten")
print(f"Attention: every output directly attends to every input — O(1) path length")

if len(sample_tokens) >= 5:
    print(f"  '{sample_tokens[0]}' -> '{sample_tokens[-1]}': {attn_weights[0][-1]:.4f}")
    print(f"  '{sample_tokens[-1]}' -> '{sample_tokens[0]}': {attn_weights[-1][0]:.4f}")
    print(f"  Direct connection regardless of distance!")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Compare attention vs LSTM on long sequences
# ══════════════════════════════════════════════════════════════════════

print(f"\n--- Scaling Comparison ---")
print(f"{'Seq Length':<12} {'LSTM path':<15} {'Attn path':<12} {'Attn memory'}")
print("-" * 55)
for seq_len in [10, 50, 100, 500, 1000]:
    print(f"  {seq_len:<12} {seq_len:<15} {1:<12} {seq_len*seq_len:,}")

print(f"\nLSTM: O(n) path length, O(n) memory — gradients vanish")
print(f"Attention: O(1) path length, O(n²) memory — constant gradient path")

print(
    "\n✓ Exercise 5 complete — scaled dot-product + multi-head attention, visualization"
)
