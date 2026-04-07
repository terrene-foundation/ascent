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

loader = ASCENTDataLoader()
df = loader.load("ascent08", "sg_product_reviews.parquet")

explorer = DataExplorer()
summary = asyncio.run(explorer.profile(df))
print(f"=== Dataset: {df.height} reviews, columns: {df.columns} ===")
print(summary)


def tokenize(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()


corpus = df.select("review_text").to_series().to_list()
word_counts = Counter(tok for t in corpus for tok in tokenize(t))
vocab = ["<pad>", "<unk>"] + [w for w, c in word_counts.most_common(2000) if c >= 2]
word_to_idx = {w: i for i, w in enumerate(vocab)}
vocab_size = len(vocab)
print(f"Vocabulary: {vocab_size} words")


def tanh(x: float) -> float:
    return math.tanh(x)


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Vanilla RNN cell forward pass
# ══════════════════════════════════════════════════════════════════════


class RNNCell:
    """Vanilla RNN: h_t = tanh(W_xh*x + W_hh*h_{t-1} + b)."""

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
        """h_t = tanh(W_xh*x + W_hh*h_prev + b) for each hidden unit."""
        # TODO: for each unit j: val = b_h[j] + sum x[k]*W_xh[k][j] + sum h_prev[k]*W_hh[k][j]; h_new[j]=tanh(val)
        h_new = [0.0] * self.hidden_dim
        for j in range(self.hidden_dim):
            val = self.b_h[j]
            for k in range(len(x)):
                ____  # Hint: val += x[k]*self.W_xh[k][j]
            for k in range(self.hidden_dim):
                ____  # Hint: val += h_prev[k]*self.W_hh[k][j]
            h_new[j] = ____  # Hint: tanh(val)
        return h_new

    def forward_sequence(self, sequence: list[list[float]]) -> list[list[float]]:
        # TODO: init h=zeros; loop x_t; accumulate forward() results; return all states
        ____
        ____
        ____
        return hidden_states


input_dim, hidden_dim = 10, 8
rnn = RNNCell(input_dim, hidden_dim)
demo_seq = [[random.gauss(0, 1) for _ in range(input_dim)] for _ in range(5)]
states = rnn.forward_sequence(demo_seq)

print(f"\nRNN: input={input_dim}, hidden={hidden_dim}")
for t, h in enumerate(states):
    norm = math.sqrt(sum(v * v for v in h))
    print(f"  t={t}: ||h||={norm:.4f}, h[:3]={[f'{v:.3f}' for v in h[:3]]}")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Demonstrate vanishing gradient problem
# ══════════════════════════════════════════════════════════════════════


def measure_gradient_flow(cell: RNNCell, seq_len: int) -> list[float]:
    """Sensitivity of final output to each input time step via perturbation."""
    # TODO: for each t, perturb input[t][0] by epsilon; measure change in final hidden sum
    epsilon = 1e-5
    ____
    ____
    sensitivities = []
    for t in range(seq_len):
        ____
        ____
    return sensitivities


print(f"\n--- Vanishing Gradient ---")
for seq_len in [10, 25, 50]:
    g = measure_gradient_flow(rnn, seq_len)
    print(
        f"  len={seq_len}: grad[0]={g[0]:.6f}, grad[-1]={g[-1]:.6f}, ratio={g[0]/(g[-1]+1e-10):.2f}"
    )
print(f"\ntanh derivatives < 1 multiply across steps — early inputs lose influence.")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: LSTM cell with gates
# ══════════════════════════════════════════════════════════════════════


class LSTMCell:
    """LSTM with forget, input, cell-candidate, and output gates."""

    def __init__(self, input_dim: int, hidden_dim: int):
        # TODO: init hidden_dim, scale, combined; W_f/W_i/W_c/W_o as (combined x hidden) Gaussian;
        #   b_f=[1.0]*hidden_dim; b_i/b_c/b_o=[0.0]*hidden_dim
        self.hidden_dim = hidden_dim
        ____
        ____
        ____
        ____
        ____
        ____
        ____

    def _gate(
        self, combined: list[float], W: list[list[float]], b: list[float], activation
    ) -> list[float]:
        return [
            activation(b[j] + sum(combined[k] * W[k][j] for k in range(len(combined))))
            for j in range(self.hidden_dim)
        ]

    def forward(
        self, x: list[float], h_prev: list[float], c_prev: list[float]
    ) -> tuple[list[float], list[float]]:
        """Single LSTM step: forget→input→cell→output gates → (h_t, c_t)."""
        combined = x + h_prev
        # TODO: compute four gates using _gate; then c_t = f*c_prev + i*c_hat; h_t = o*tanh(c_t)
        f_t = ____  # Hint: self._gate(combined, self.W_f, self.b_f, sigmoid)
        i_t = ____  # Hint: self._gate(combined, self.W_i, self.b_i, sigmoid)
        c_hat = ____  # Hint: self._gate(combined, self.W_c, self.b_c, tanh)
        o_t = ____  # Hint: self._gate(combined, self.W_o, self.b_o, sigmoid)
        c_t = ____  # Hint: [f_t[j]*c_prev[j]+i_t[j]*c_hat[j] for j in range(self.hidden_dim)]
        h_t = ____  # Hint: [o_t[j]*tanh(c_t[j]) for j in range(self.hidden_dim)]
        return h_t, c_t

    def forward_sequence(self, sequence: list[list[float]]) -> list[list[float]]:
        h, c = [0.0] * self.hidden_dim, [0.0] * self.hidden_dim
        states = []
        for x_t in sequence:
            h, c = self.forward(x_t, h, c)
            states.append(h[:])
        return states


lstm = LSTMCell(input_dim, hidden_dim)
lstm_states = lstm.forward_sequence(demo_seq)
print(f"\nLSTM: {input_dim}→{hidden_dim}")
for t, h in enumerate(lstm_states):
    print(f"  t={t}: ||h||={math.sqrt(sum(v*v for v in h)):.4f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Bidirectional LSTM for sentiment
# ══════════════════════════════════════════════════════════════════════


class BiLSTM:
    """Forward + backward LSTM, concatenated at each position."""

    def __init__(self, input_dim: int, hidden_dim: int):
        self.forward_lstm = LSTMCell(input_dim, hidden_dim)
        self.backward_lstm = LSTMCell(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward_sequence(self, sequence: list[list[float]]) -> list[list[float]]:
        """Concatenate [forward; backward] hidden states."""
        # TODO: forward pass; backward pass on reversed sequence (re-reverse); concatenate
        fwd_states = ____  # Hint: self.forward_lstm.forward_sequence(sequence)
        bwd_states = (
            ____  # Hint: self.backward_lstm.forward_sequence(sequence[::-1])[::-1]
        )
        return ____  # Hint: [f+b for f,b in zip(fwd_states, bwd_states)]


bilstm = BiLSTM(input_dim, hidden_dim)
bi_states = bilstm.forward_sequence(demo_seq)
print(f"\nBiLSTM output dim: {len(bi_states[0])} (2×{hidden_dim})")
print(f"Forward: left context | Backward: right context")
print(f"Production: TrainingPipeline(feature_store, registry).fit(data)")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Compare RNN vs LSTM convergence
# ══════════════════════════════════════════════════════════════════════

rnn_grads = measure_gradient_flow(rnn, 50)
lstm_grads = measure_gradient_flow(lstm, 50)

print(f"\n--- Gradient Flow (seq_len=50) ---")
print(f"{'t':<6} {'RNN':<14} {'LSTM'}")
for t in [0, 10, 25, 40, 49]:
    print(f"  {t:<6} {rnn_grads[t]:<14.6f} {lstm_grads[t]:.6f}")

# TODO: Instantiate ModelVisualizer and plot gradient flow comparison
viz = ____  # Hint: ModelVisualizer()
fig = viz.training_history(
    metrics={"rnn_gradient": rnn_grads, "lstm_gradient": lstm_grads}
)
print(f"\nLSTM cell-state highway: forget-gate≈1 lets gradients pass unattenuated.")

print("\n✓ Exercise 4 complete — RNN/LSTM cells, vanishing gradients, BiLSTM")
