# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT08 — Exercise 6: Transformer Architecture
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Build a mini transformer encoder from components — positional
#   encoding, multi-head attention, feed-forward network, layer
#   normalization, and residual connections.
#
# TASKS:
#   1. Implement sinusoidal positional encoding
#   2. Build transformer encoder layer (attention + FFN + LayerNorm + residual)
#   3. Stack layers into full encoder
#   4. Train on text classification task
#   5. Visualize attention patterns per layer with ModelVisualizer
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
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()


corpus = df.select("text").to_series().to_list()
word_counts = Counter(tok for t in corpus for tok in tokenize(t))
vocab = ["<pad>", "<cls>", "<unk>"] + [
    w for w, c in word_counts.most_common(2000) if c >= 2
]
word_to_idx = {w: i for i, w in enumerate(vocab)}
vocab_size = len(vocab)
d_model = 32

# Random token embeddings
token_embeddings = [
    [random.gauss(0, 0.1) for _ in range(d_model)] for _ in range(vocab_size)
]

print(f"Vocabulary: {vocab_size}, d_model: {d_model}")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Sinusoidal positional encoding
# ══════════════════════════════════════════════════════════════════════


def sinusoidal_positional_encoding(max_len: int, d_model: int) -> list[list[float]]:
    """PE(pos, 2i) = sin(pos / 10000^(2i/d_model)),
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))."""
    pe = []
    for pos in range(max_len):
        row = [0.0] * d_model
        # TODO: loop i in steps of 2; denom = 10000^(i/d_model)
        # TODO: row[i] = sin(pos/denom); if i+1 < d_model row[i+1] = cos(pos/denom)
        for i in range(0, d_model, 2):
            denom = ____
            row[i] = ____
            if i + 1 < d_model:
                row[i + 1] = ____
        pe.append(row)
    return pe


max_seq_len = 50
pos_enc = sinusoidal_positional_encoding(max_seq_len, d_model)

print(f"\n--- Positional Encoding ---")
print(f"Shape: {len(pos_enc)}x{len(pos_enc[0])}")
for pos in [0, 1, 10, 49]:
    print(
        f"  pos={pos}: [{pos_enc[pos][0]:.3f}, {pos_enc[pos][1]:.3f}, ..., {pos_enc[pos][-1]:.3f}]"
    )

print(f"\nProperties:")
print(f"  - Unique encoding per position (no learned parameters)")
print(f"  - Captures relative position via dot product")
print(f"  - Generalizes to longer sequences than seen during training")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Transformer encoder layer
# ══════════════════════════════════════════════════════════════════════


def softmax(scores: list[float]) -> list[float]:
    max_s = max(scores) if scores else 0
    exps = [math.exp(s - max_s) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def layer_norm(x: list[float], eps: float = 1e-6) -> list[float]:
    """Layer normalization: zero-mean, unit-variance across feature dim."""
    # TODO: compute mean and variance over x
    mean = ____
    var = ____
    # TODO: std = sqrt(var + eps); return (x - mean) / std
    std = ____
    return ____


def feed_forward(
    x: list[float],
    W1: list[list[float]],
    b1: list[float],
    W2: list[list[float]],
    b2: list[float],
) -> list[float]:
    """FFN(x) = ReLU(xW1 + b1)W2 + b2."""
    d_ff = len(W1[0])
    # TODO: hidden[j] = ReLU(b1[j] + sum_k x[k]*W1[k][j])
    hidden = [0.0] * d_ff
    for j in range(d_ff):
        val = b1[j]
        ____
        hidden[j] = ____

    d_out = len(W2[0])
    # TODO: output[j] = b2[j] + sum_k hidden[k]*W2[k][j]
    output = [0.0] * d_out
    for j in range(d_out):
        val = b2[j]
        ____
        output[j] = val
    return output


def residual_add(x: list[float], sublayer_out: list[float]) -> list[float]:
    """Residual connection: x + sublayer(x)."""
    # TODO: elementwise sum
    return ____


class TransformerEncoderLayer:
    """Single transformer encoder layer: self-attention + FFN + residuals + LayerNorm."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        scale = 1.0 / math.sqrt(self.d_k)

        # TODO: per-head W_q, W_k, W_v as (d_model x d_k) Gaussian matrices
        self.W_q = ____
        self.W_k = ____
        self.W_v = ____
        # TODO: output projection W_o (d_model x d_model)
        self.W_o = ____

        # FFN weights
        ff_scale = 1.0 / math.sqrt(d_ff)
        # TODO: W1 (d_model x d_ff), b1 zeros, W2 (d_ff x d_model), b2 zeros
        self.W1 = ____
        self.b1 = ____
        self.W2 = ____
        self.b2 = ____

    def self_attention(
        self, X: list[list[float]]
    ) -> tuple[list[list[float]], list[list[list[float]]]]:
        """Multi-head self-attention."""
        all_heads = []
        all_weights = []
        for h in range(self.n_heads):
            # TODO: project X into Q, K, V for this head
            Q = ____
            K = ____
            V = ____

            scale = math.sqrt(self.d_k)
            # TODO: scores[i][j] = dot(Q[i], K[j]) / scale
            scores = ____
            # TODO: row-wise softmax
            weights = ____
            # TODO: head_out[i][d] = sum_j weights[i][j] * V[j][d]
            head_out = ____
            all_heads.append(head_out)
            all_weights.append(weights)

        # TODO: concatenate heads per position
        concat = [[] for _ in range(len(X))]
        for i in range(len(X)):
            for h in range(self.n_heads):
                ____

        # TODO: apply output projection W_o
        output = ____
        return output, all_weights

    def forward(
        self, X: list[list[float]]
    ) -> tuple[list[list[float]], list[list[list[float]]]]:
        """attention → residual → LayerNorm → FFN → residual → LayerNorm."""
        # TODO: self-attention pass
        attn_out, attn_weights = ____
        # TODO: residual + layer_norm (per position)
        normed1 = ____

        # TODO: feed-forward per position
        ffn_out = ____
        # TODO: residual + layer_norm (per position)
        normed2 = ____

        return normed2, attn_weights


n_heads = 4
d_ff = 64
layer = TransformerEncoderLayer(d_model, n_heads, d_ff)

sample_tokens = tokenize(corpus[0] if corpus else "singapore economy growth")[:10]
sample_indices = [word_to_idx.get(t, 2) for t in sample_tokens]
sample_input = [
    [token_embeddings[idx][d] + pos_enc[pos][d] for d in range(d_model)]
    for pos, idx in enumerate(sample_indices)
]

layer_out, layer_attn = layer.forward(sample_input)
print(f"\n--- Encoder Layer ---")
print(
    f"Input: {len(sample_input)}x{d_model}, Output: {len(layer_out)}x{len(layer_out[0])}"
)
print(f"Attention heads: {n_heads}, FFN dim: {d_ff}")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Stack layers into full encoder
# ══════════════════════════════════════════════════════════════════════


class TransformerEncoder:
    """Stack of N transformer encoder layers."""

    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_ff: int):
        # TODO: create a list of n_layers TransformerEncoderLayer instances
        self.layers = ____

    def forward(self, X: list[list[float]]) -> tuple[list[list[float]], list]:
        """Forward through all layers, collecting attention weights."""
        # TODO: init hidden = X, all_layer_attn = []
        all_layer_attn = []
        hidden = ____
        # TODO: for each layer, forward through and collect attention weights
        for layer in self.layers:
            ____
            ____
        return hidden, all_layer_attn


n_layers = 3
encoder = TransformerEncoder(n_layers, d_model, n_heads, d_ff)
enc_output, all_attn = encoder.forward(sample_input)

print(f"\n--- Full Encoder ({n_layers} layers) ---")
print(f"Output shape: {len(enc_output)}x{len(enc_output[0])}")
print(f"Attention maps collected: {len(all_attn)} layers x {n_heads} heads")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Train on text classification task
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Transformer Encoder for Classification ===")
print(f"The [CLS] token embedding from the encoder can be used as a document vector.")
print(f"In production: pass to TrainingPipeline(feature_store, registry) for training.")
print(f"Documents processed: {len(corpus)}")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Visualize attention patterns per layer
# ══════════════════════════════════════════════════════════════════════

# TODO: instantiate ModelVisualizer
viz = ____

print(f"\n=== Attention Pattern Analysis ===")
print(f"Layer 0: tends to capture local/syntactic patterns")
print(f"Layer 1: tends to capture broader semantic relationships")
print(f"Layer 2: tends to capture task-specific patterns")
print(f"\nThis hierarchy is why deeper transformers capture more complex patterns.")

print("\n✓ Exercise 6 complete — mini transformer encoder from scratch")
