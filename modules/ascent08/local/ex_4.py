# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT08 — Exercise 4: Sequence Models — RNNs and LSTMs
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Understand sequence modeling with RNNs and LSTMs — vanishing
#   gradients, gating mechanisms — for text classification.
#
# TASKS:
#   1. Implement vanilla RNN cell forward pass
#   2. Demonstrate vanishing gradient problem
#   3. Implement LSTM cell with gates (forget, input, output)
#   4. Build bidirectional LSTM for sentiment analysis
#   5. Compare RNN vs LSTM convergence with ModelVisualizer
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
print(f"=== Dataset: {df.height} reviews, columns: {df.columns} ===")
print(summary)


# ── Helpers ───────────────────────────────────────────────────────────


def tokenize(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()


corpus = df.select("review_text").to_series().to_list()
word_counts = Counter(tok for t in corpus for tok in tokenize(t))
vocab = ["<pad>", "<unk>"] + [w for w, c in word_counts.most_common(2000) if c >= 2]
word_to_idx = {w: i for i, w in enumerate(vocab)}
vocab_size = len(vocab)

print(f"Vocabulary: {vocab_size} words")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Vanilla RNN cell forward pass
# ══════════════════════════════════════════════════════════════════════


def tanh(x: float) -> float:
    return math.tanh(x)


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


class RNNCell:
    """Vanilla RNN: h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b)."""

    def __init__(self, input_dim: int, hidden_dim: int):
        self.hidden_dim = hidden_dim
        scale = 1.0 / math.sqrt(hidden_dim)
        self.W_xh = [
            [random.gauss(0, scale) for _ in range(hidden_dim)]
            for _ in range(input_dim)
        ]
        self.W_hh = [
            [random.gauss(0, scale) for _ in range(hidden_dim)]
            for _ in range(hidden_dim)
        ]
        self.b_h = [0.0] * hidden_dim

    def forward(self, x: list[float], h_prev: list[float]) -> list[float]:
        """Single RNN step: h_t = tanh(W_xh * x + W_hh * h_prev + b)."""
        # TODO: For each hidden unit j, accumulate b_h[j] + W_xh contribution + W_hh contribution, apply tanh
        h_new = [0.0] * self.hidden_dim
        for j in range(self.hidden_dim):
            val = self.b_h[j]
            for k in range(len(x)):
                ____  # Hint: val += x[k] * self.W_xh[k][j]
            for k in range(self.hidden_dim):
                ____  # Hint: val += h_prev[k] * self.W_hh[k][j]
            h_new[j] = ____  # Hint: tanh(val)
        return h_new

    def forward_sequence(self, sequence: list[list[float]]) -> list[list[float]]:
        """Process full sequence, collect hidden states."""
        h = [0.0] * self.hidden_dim
        hidden_states = []
        for x_t in sequence:
            h = self.forward(x_t, h)
            hidden_states.append(h[:])
        return hidden_states


input_dim, hidden_dim = 10, 8
rnn = RNNCell(input_dim, hidden_dim)
demo_seq = [[random.gauss(0, 1) for _ in range(input_dim)] for _ in range(5)]
states = rnn.forward_sequence(demo_seq)

print(f"\nRNN Cell: input_dim={input_dim}, hidden_dim={hidden_dim}")
for t, h in enumerate(states):
    norm = math.sqrt(sum(v * v for v in h))
    print(f"  t={t}: ||h||={norm:.4f}, h[:3]={[f'{v:.3f}' for v in h[:3]]}")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Demonstrate vanishing gradient problem
# ══════════════════════════════════════════════════════════════════════


def measure_gradient_flow(cell: RNNCell, seq_len: int) -> list[float]:
    """Approximate gradient magnitude at each time step via perturbation."""
    epsilon = 1e-5
    sequence = [
        [random.gauss(0, 0.5) for _ in range(input_dim)] for _ in range(seq_len)
    ]
    states = cell.forward_sequence(sequence)
    final_output = sum(states[-1])
    sensitivities = []
    for t in range(seq_len):
        perturbed = [row[:] for row in sequence]
        perturbed[t][0] += epsilon
        perturbed_output = sum(cell.forward_sequence(perturbed)[-1])
        sensitivities.append(abs(perturbed_output - final_output) / epsilon)
    return sensitivities


print(f"\n--- Vanishing Gradient Demonstration ---")
for seq_len in [10, 25, 50]:
    grads = measure_gradient_flow(rnn, seq_len)
    print(
        f"\n  seq_len={seq_len}: t=0 grad={grads[0]:.6f}, t={seq_len-1} grad={grads[-1]:.6f}, ratio={grads[0]/(grads[-1]+1e-10):.2f}"
    )

print(
    f"\ntanh derivatives < 1 compound multiplicatively — early inputs lose influence."
)


# ══════════════════════════════════════════════════════════════════════
# TASK 3: LSTM cell with gates
# ══════════════════════════════════════════════════════════════════════


class LSTMCell:
    """LSTM with forget, input, and output gates."""

    def __init__(self, input_dim: int, hidden_dim: int):
        self.hidden_dim = hidden_dim
        scale = 1.0 / math.sqrt(hidden_dim)
        combined = input_dim + hidden_dim
        self.W_f = [
            [random.gauss(0, scale) for _ in range(hidden_dim)] for _ in range(combined)
        ]
        self.W_i = [
            [random.gauss(0, scale) for _ in range(hidden_dim)] for _ in range(combined)
        ]
        self.W_c = [
            [random.gauss(0, scale) for _ in range(hidden_dim)] for _ in range(combined)
        ]
        self.W_o = [
            [random.gauss(0, scale) for _ in range(hidden_dim)] for _ in range(combined)
        ]
        self.b_f = [1.0] * hidden_dim  # forget bias = 1 → remember by default
        self.b_i = [0.0] * hidden_dim
        self.b_c = [0.0] * hidden_dim
        self.b_o = [0.0] * hidden_dim

    def _gate(
        self, combined: list[float], W: list[list[float]], b: list[float], activation
    ) -> list[float]:
        result = [0.0] * self.hidden_dim
        for j in range(self.hidden_dim):
            val = b[j] + sum(combined[k] * W[k][j] for k in range(len(combined)))
            result[j] = activation(val)
        return result

    def forward(
        self, x: list[float], h_prev: list[float], c_prev: list[float]
    ) -> tuple[list[float], list[float]]:
        """Single LSTM step: returns (h_t, c_t)."""
        combined = x + h_prev
        # TODO: Compute forget, input, cell-candidate, output gates using _gate
        f_t = ____  # Hint: self._gate(combined, self.W_f, self.b_f, sigmoid)
        i_t = ____  # Hint: self._gate(combined, self.W_i, self.b_i, sigmoid)
        c_hat = ____  # Hint: self._gate(combined, self.W_c, self.b_c, tanh)
        o_t = ____  # Hint: self._gate(combined, self.W_o, self.b_o, sigmoid)
        # TODO: Cell state update: c_t = f_t ⊙ c_prev + i_t ⊙ c_hat
        c_t = ____  # Hint: [f_t[j] * c_prev[j] + i_t[j] * c_hat[j] for j in range(self.hidden_dim)]
        # TODO: Hidden state: h_t = o_t ⊙ tanh(c_t)
        h_t = ____  # Hint: [o_t[j] * tanh(c_t[j]) for j in range(self.hidden_dim)]
        return h_t, c_t

    def forward_sequence(self, sequence: list[list[float]]) -> list[list[float]]:
        h, c = [0.0] * self.hidden_dim, [0.0] * self.hidden_dim
        hidden_states = []
        for x_t in sequence:
            h, c = self.forward(x_t, h, c)
            hidden_states.append(h[:])
        return hidden_states


lstm = LSTMCell(input_dim, hidden_dim)
lstm_states = lstm.forward_sequence(demo_seq)

print(f"\nLSTM Cell: {input_dim}→{hidden_dim}")
for t, h in enumerate(lstm_states):
    print(f"  t={t}: ||h||={math.sqrt(sum(v*v for v in h)):.4f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Bidirectional LSTM for sentiment
# ══════════════════════════════════════════════════════════════════════


class BiLSTM:
    """Bidirectional LSTM: forward + backward passes, concatenated output."""

    def __init__(self, input_dim: int, hidden_dim: int):
        self.forward_lstm = LSTMCell(input_dim, hidden_dim)
        self.backward_lstm = LSTMCell(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward_sequence(self, sequence: list[list[float]]) -> list[list[float]]:
        """Concatenate [forward; backward] hidden states at each position."""
        # TODO: Forward pass on sequence; backward pass on reversed sequence (then re-reverse)
        fwd_states = ____  # Hint: self.forward_lstm.forward_sequence(sequence)
        bwd_states = (
            ____  # Hint: self.backward_lstm.forward_sequence(sequence[::-1])[::-1]
        )
        # TODO: Concatenate forward and backward at each time step
        return ____  # Hint: [f + b for f, b in zip(fwd_states, bwd_states)]


bilstm = BiLSTM(input_dim, hidden_dim)
bi_states = bilstm.forward_sequence(demo_seq)

print(f"\nBiLSTM output dim: {len(bi_states[0])} (2 × {hidden_dim})")
print(f"Forward captures left context; backward captures right context.")
print(f"In production: TrainingPipeline(feature_store, registry).fit(data)")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Compare RNN vs LSTM convergence
# ══════════════════════════════════════════════════════════════════════

rnn_grads = measure_gradient_flow(rnn, 50)
lstm_grads = measure_gradient_flow(lstm, 50)

print(f"\n--- RNN vs LSTM Gradient Flow (seq_len=50) ---")
print(f"{'Step':<8} {'RNN grad':<15} {'LSTM grad'}")
print("-" * 38)
for t in [0, 10, 25, 40, 49]:
    print(f"  t={t:<4} {rnn_grads[t]:<15.6f} {lstm_grads[t]:.6f}")

# TODO: Instantiate ModelVisualizer and plot training history
viz = ____  # Hint: ModelVisualizer()
fig = viz.training_history(
    metrics={"rnn_gradient": rnn_grads, "lstm_gradient": lstm_grads}
)
print(f"\nLSTM cell-state highway lets gradients pass with forget-gate ≈ 1.")

print("\n✓ Exercise 4 complete — RNN/LSTM cells, vanishing gradients, BiLSTM sentiment")
