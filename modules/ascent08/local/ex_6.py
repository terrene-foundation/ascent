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


# ── Helpers ───────────────────────────────────────────────────────────


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

token_embeddings = [
    [random.gauss(0, 0.1) for _ in range(d_model)] for _ in range(vocab_size)
]

print(f"Vocabulary: {vocab_size}, d_model: {d_model}")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Sinusoidal positional encoding
# ══════════════════════════════════════════════════════════════════════


def sinusoidal_positional_encoding(max_len: int, d_model: int) -> list[list[float]]:
    """PE(pos, 2i) = sin(pos/10000^(2i/d_model)), PE(pos, 2i+1) = cos(...)"""
    # TODO: Build max_len × d_model table; for each pos and even index i:
    #   denom = 10000^(i/d_model); row[i] = sin(pos/denom); row[i+1] = cos(pos/denom)
    pe = []
    for pos in range(max_len):
        row = [0.0] * d_model
        for i in range(0, d_model, 2):
            denom = ____  # Hint: 10000.0 ** (i / d_model)
            row[i] = ____  # Hint: math.sin(pos / denom)
            if i + 1 < d_model:
                row[i + 1] = ____  # Hint: math.cos(pos / denom)
        pe.append(row)
    return pe


max_seq_len = 50
pos_enc = sinusoidal_positional_encoding(max_seq_len, d_model)

print(f"\n--- Positional Encoding: {len(pos_enc)}×{len(pos_enc[0])} ---")
for pos in [0, 1, 10, 49]:
    print(
        f"  pos={pos}: [{pos_enc[pos][0]:.3f}, {pos_enc[pos][1]:.3f}, ..., {pos_enc[pos][-1]:.3f}]"
    )
print(f"\nUnique per position, no learned params, generalizes to longer sequences.")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Transformer encoder layer
# ══════════════════════════════════════════════════════════════════════


def softmax(scores: list[float]) -> list[float]:
    max_s = max(scores) if scores else 0
    exps = [math.exp(s - max_s) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def layer_norm(x: list[float], eps: float = 1e-6) -> list[float]:
    """Normalize across feature dimension."""
    # TODO: Compute mean and variance; return (x - mean) / sqrt(var + eps)
    mean = ____  # Hint: sum(x) / len(x)
    var = ____  # Hint: sum((v - mean) ** 2 for v in x) / len(x)
    std = math.sqrt(var + eps)
    return ____  # Hint: [(v - mean) / std for v in x]


def feed_forward(
    x: list[float],
    W1: list[list[float]],
    b1: list[float],
    W2: list[list[float]],
    b2: list[float],
) -> list[float]:
    """FFN(x) = ReLU(xW1 + b1)W2 + b2."""
    # TODO: First linear layer with ReLU activation
    d_ff = len(W1[0])
    hidden = [
        max(0.0, b1[j] + sum(x[k] * W1[k][j] for k in range(len(x))))
        for j in range(d_ff)
    ]
    # TODO: Second linear layer
    d_out = len(W2[0])
    return [
        b2[j] + sum(hidden[k] * W2[k][j] for k in range(d_ff)) for j in range(d_out)
    ]


def residual_add(x: list[float], sublayer_out: list[float]) -> list[float]:
    return ____  # Hint: [a + b for a, b in zip(x, sublayer_out)]


class TransformerEncoderLayer:
    """Self-attention + FFN + residual connections + LayerNorm."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        scale = 1.0 / math.sqrt(self.d_k)
        # Attention weights
        self.W_q = [
            [[random.gauss(0, scale) for _ in range(self.d_k)] for _ in range(d_model)]
            for _ in range(n_heads)
        ]
        self.W_k = [
            [[random.gauss(0, scale) for _ in range(self.d_k)] for _ in range(d_model)]
            for _ in range(n_heads)
        ]
        self.W_v = [
            [[random.gauss(0, scale) for _ in range(self.d_k)] for _ in range(d_model)]
            for _ in range(n_heads)
        ]
        self.W_o = [
            [random.gauss(0, scale) for _ in range(d_model)] for _ in range(d_model)
        ]
        # FFN weights
        ff_scale = 1.0 / math.sqrt(d_ff)
        self.W1 = [
            [random.gauss(0, ff_scale) for _ in range(d_ff)] for _ in range(d_model)
        ]
        self.b1 = [0.0] * d_ff
        self.W2 = [
            [random.gauss(0, scale) for _ in range(d_model)] for _ in range(d_ff)
        ]
        self.b2 = [0.0] * d_model

    def self_attention(self, X: list[list[float]]) -> tuple[list[list[float]], list]:
        """Multi-head self-attention with QKV projection."""
        # TODO: For each head h: project X -> Q_h, K_h, V_h; compute scaled attention;
        #   concatenate heads; apply W_o projection
        all_heads, all_weights = [], []
        for h in range(self.n_heads):
            Q = ____  # Hint: [[sum(X[i][k]*self.W_q[h][k][j] for k in range(self.d_model)) for j in range(self.d_k)] for i in range(len(X))]
            K = ____  # Hint: same pattern with W_k[h]
            V = ____  # Hint: same pattern with W_v[h]
            scale = math.sqrt(self.d_k)
            scores = [
                [
                    sum(Q[i][d] * K[j][d] for d in range(self.d_k)) / scale
                    for j in range(len(K))
                ]
                for i in range(len(Q))
            ]
            weights = [softmax(row) for row in scores]
            head_out = [
                [
                    sum(weights[i][j] * V[j][d] for j in range(len(V)))
                    for d in range(self.d_k)
                ]
                for i in range(len(Q))
            ]
            all_heads.append(head_out)
            all_weights.append(weights)
        concat = [[] for _ in range(len(X))]
        for i in range(len(X)):
            for h in range(self.n_heads):
                concat[i].extend(all_heads[h][i])
        output = [
            [
                sum(concat[i][k] * self.W_o[k][j] for k in range(self.d_model))
                for j in range(self.d_model)
            ]
            for i in range(len(X))
        ]
        return output, all_weights

    def forward(self, X: list[list[float]]) -> tuple[list[list[float]], list]:
        """Encoder layer: attn → Add&Norm → FFN → Add&Norm."""
        # TODO: Sub-layer 1: self-attention + residual + LayerNorm
        attn_out, attn_weights = self.self_attention(X)
        normed1 = ____  # Hint: [layer_norm(residual_add(X[i], attn_out[i])) for i in range(len(X))]
        # TODO: Sub-layer 2: FFN + residual + LayerNorm
        ffn_out = ____  # Hint: [feed_forward(normed1[i], self.W1, self.b1, self.W2, self.b2) for i in range(len(normed1))]
        normed2 = ____  # Hint: [layer_norm(residual_add(normed1[i], ffn_out[i])) for i in range(len(normed1))]
        return normed2, attn_weights


n_heads, d_ff = 4, 64
layer = TransformerEncoderLayer(d_model, n_heads, d_ff)

sample_tokens = tokenize(corpus[0] if corpus else "singapore economy growth")[:10]
sample_indices = [word_to_idx.get(t, 2) for t in sample_tokens]
# TODO: Build input embeddings by adding token embeddings and positional encodings
sample_input = ____  # Hint: [[token_embeddings[idx][d] + pos_enc[pos][d] for d in range(d_model)] for pos, idx in enumerate(sample_indices)]

layer_out, layer_attn = layer.forward(sample_input)
print(f"\n--- Encoder Layer ---")
print(
    f"Input {len(sample_input)}×{d_model} → Output {len(layer_out)}×{len(layer_out[0])}, heads={n_heads}, d_ff={d_ff}"
)


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Stack layers into full encoder
# ══════════════════════════════════════════════════════════════════════


class TransformerEncoder:
    """Stack of N encoder layers."""

    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_ff: int):
        # TODO: Create n_layers TransformerEncoderLayer instances
        self.layers = ____  # Hint: [TransformerEncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)]

    def forward(self, X: list[list[float]]) -> tuple[list[list[float]], list]:
        """Pass through all layers, collecting per-layer attention."""
        all_layer_attn = []
        hidden = X
        for lyr in self.layers:
            # TODO: Forward through each layer; collect attention weights
            hidden, attn_weights = ____  # Hint: lyr.forward(hidden)
            all_layer_attn.append(attn_weights)
        return hidden, all_layer_attn


n_layers = 3
encoder = TransformerEncoder(n_layers, d_model, n_heads, d_ff)
enc_output, all_attn = encoder.forward(sample_input)

print(f"\n--- Full Encoder ({n_layers} layers) ---")
print(f"Output: {len(enc_output)}×{len(enc_output[0])}")
print(f"Attention maps: {len(all_attn)} layers × {n_heads} heads")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Train on text classification task
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Transformer Encoder for Classification ===")
print(f"[CLS] token embedding → document vector → classifier")
print(f"In production: pass to TrainingPipeline(feature_store, registry)")
print(f"Documents available: {len(corpus)}")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Visualize attention patterns per layer
# ══════════════════════════════════════════════════════════════════════

# TODO: Instantiate ModelVisualizer for per-layer attention heatmaps
viz = ____  # Hint: ModelVisualizer()
print(f"\n=== Attention Pattern Hierarchy ===")
print(f"Layer 0: local/syntactic | Layer 1: semantic | Layer 2: task-specific")
print(f"Deeper transformers capture increasingly abstract patterns.")

print("\n✓ Exercise 6 complete — mini transformer encoder from scratch")
