# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT09 — Exercise 1: LoRA Internals and Rank Selection
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Derive LoRA from first principles — implement low-rank
#   decomposition, LoRA forward pass, and rank selection heuristics.
#   Validate from-scratch implementations against AlignmentPipeline.
#
# TASKS:
#   1. Implement low-rank matrix decomposition (SVD-based)
#   2. Measure reconstruction error vs rank vs parameter count
#   3. Implement LoRA forward pass from scratch
#   4. Use AlignmentPipeline with LoRAConfig at multiple ranks
#   5. Implement rank selector heuristic (stable rank)
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os

import numpy as np
import polars as pl

from kailash_align import AlignmentConfig, AlignmentPipeline, AdapterRegistry
from kailash_align.config import SFTConfig, LoRAConfig
from kailash_align.registry import AdapterSignature

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()

base_model = os.environ.get("SFT_BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# ── Data Loading ─────────────────────────────────────────────────────

loader = ASCENTDataLoader()
reports = loader.load("ascent09", "sg_company_reports.parquet")

print(f"=== SG Company Reports Dataset ===")
print(f"Shape: {reports.shape}")
print(f"Columns: {reports.columns}")
print(reports.head(3))


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Implement low-rank matrix decomposition from scratch (SVD)
# ══════════════════════════════════════════════════════════════════════
# Full fine-tuning updates W directly: W' = W + delta_W
# LoRA insight: delta_W is low-rank during fine-tuning.
# Factor delta_W = B @ A where B in R^{d x r}, A in R^{r x k}, r << min(d, k)
# This reduces parameters from d*k to r*(d+k).

rng = np.random.default_rng(42)

# Simulate a pretrained weight matrix (e.g., a query projection in attention)
d_model = 512  # output dimension
d_input = 512  # input dimension
W_pretrained = rng.standard_normal((d_model, d_input)).astype(np.float32)

# Simulate a fine-tuning update (this would come from full fine-tuning)
# In practice, delta_W is approximately low-rank
delta_W = rng.standard_normal((d_model, d_input)).astype(np.float32) * 0.01


def low_rank_decomposition(
    matrix: np.ndarray, rank: int
) -> tuple[np.ndarray, np.ndarray]:
    """Decompose matrix into B @ A using truncated SVD.

    Args:
        matrix: (d, k) matrix to decompose
        rank: target rank r

    Returns:
        B: (d, r) matrix
        A: (r, k) matrix
    Such that matrix ≈ B @ A
    """
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    # Truncate to rank r
    U_r = U[:, :rank]  # (d, r)
    S_r = S[:rank]  # (r,)
    Vt_r = Vt[:rank, :]  # (r, k)

    # Split singular values between B and A
    # B = U_r * sqrt(S_r), A = sqrt(S_r) * Vt_r
    sqrt_S = np.sqrt(S_r)
    B = U_r * sqrt_S[np.newaxis, :]  # (d, r)
    A = sqrt_S[:, np.newaxis] * Vt_r  # (r, k)

    return B, A


# Decompose delta_W at various ranks
for rank in [1, 4, 8, 16, 32, 64]:
    B, A = low_rank_decomposition(delta_W, rank)
    reconstructed = B @ A
    error = np.linalg.norm(delta_W - reconstructed, "fro")
    original_norm = np.linalg.norm(delta_W, "fro")
    relative_error = error / original_norm

    print(
        f"  Rank {rank:>3}: B{B.shape} @ A{A.shape} | "
        f"Relative error: {relative_error:.6f} | "
        f"Params: {rank * (d_model + d_input):>7,} vs {d_model * d_input:>7,} full"
    )

# Verify reconstruction at full rank
B_full, A_full = low_rank_decomposition(delta_W, min(d_model, d_input))
assert np.allclose(
    delta_W, B_full @ A_full, atol=1e-4
), "Full rank should reconstruct exactly"
print("\nFull-rank reconstruction verified: delta_W == B @ A (within float tolerance)")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Measure reconstruction error as function of rank
# ══════════════════════════════════════════════════════════════════════
# Plot the trade-off between approximation quality and parameter count.

ranks = list(range(1, 65))
errors = []
param_counts = []
param_savings = []

full_params = d_model * d_input

for r in ranks:
    B, A = low_rank_decomposition(delta_W, r)
    recon_error = np.linalg.norm(delta_W - (B @ A), "fro") / np.linalg.norm(
        delta_W, "fro"
    )
    lora_params = r * (d_model + d_input)
    errors.append(recon_error)
    param_counts.append(lora_params)
    param_savings.append(1.0 - lora_params / full_params)

# SVD spectrum analysis — where does the energy concentrate?
_, S, _ = np.linalg.svd(delta_W, full_matrices=False)
cumulative_energy = np.cumsum(S**2) / np.sum(S**2)

print(f"\n=== Singular Value Energy Distribution ===")
for pct in [0.90, 0.95, 0.99, 0.999]:
    rank_needed = int(np.searchsorted(cumulative_energy, pct)) + 1
    savings = 1.0 - rank_needed * (d_model + d_input) / full_params
    print(
        f"  {pct*100:.1f}% energy captured at rank {rank_needed:>3} "
        f"({savings*100:.1f}% param savings)"
    )

# Summary table
results_df = pl.DataFrame(
    {
        "rank": [4, 8, 16, 32, 64],
        "relative_error": [errors[3], errors[7], errors[15], errors[31], errors[63]],
        "param_count": [
            param_counts[3],
            param_counts[7],
            param_counts[15],
            param_counts[31],
            param_counts[63],
        ],
        "param_savings_pct": [
            param_savings[3] * 100,
            param_savings[7] * 100,
            param_savings[15] * 100,
            param_savings[31] * 100,
            param_savings[63] * 100,
        ],
    }
)
print(f"\n=== Rank vs Error vs Parameters ===")
print(results_df)


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Implement LoRA forward pass from scratch
# ══════════════════════════════════════════════════════════════════════
# LoRA modifies the forward pass of a linear layer:
#   h = W_0 @ x + (B @ A) @ x * (alpha / r)
# where alpha is a scaling hyperparameter and r is the rank.
# During training, only B and A are updated; W_0 is frozen.


class LoRALinear:
    """LoRA-augmented linear layer (from-scratch implementation).

    Original: h = W @ x
    LoRA:     h = W @ x + (B @ A) @ x * (alpha / r)

    Parameters:
        W (frozen): (d_out, d_in) pretrained weights
        B (trainable): (d_out, r) initialized to zero
        A (trainable): (r, d_in) initialized from N(0, sigma^2)
        alpha: scaling factor
        r: rank
    """

    def __init__(self, W: np.ndarray, rank: int, alpha: float = 1.0):
        self.W = W.copy()  # Frozen pretrained weights
        self.rank = rank
        self.alpha = alpha
        d_out, d_in = W.shape

        # LoRA initialization: B=0, A=Kaiming uniform
        # This ensures delta_W = B @ A = 0 at init (no disruption)
        self.B = np.zeros((d_out, rank), dtype=np.float32)
        self.A = rng.standard_normal((rank, d_in)).astype(np.float32) * np.sqrt(
            2.0 / d_in
        )

        self.scaling = alpha / rank

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: h = W @ x + B @ A @ x * (alpha / r)."""
        h_base = self.W @ x  # Pretrained contribution
        h_lora = (self.B @ self.A) @ x  # LoRA contribution
        return h_base + h_lora * self.scaling  # Combined

    @property
    def trainable_params(self) -> int:
        return self.B.size + self.A.size

    @property
    def total_params(self) -> int:
        return self.W.size + self.trainable_params

    def merge(self) -> np.ndarray:
        """Merge LoRA into base weights: W' = W + B @ A * (alpha / r)."""
        return self.W + (self.B @ self.A) * self.scaling


# Test the LoRA layer
x = rng.standard_normal((d_input, 1)).astype(np.float32)

lora_layer = LoRALinear(W_pretrained, rank=16, alpha=32.0)

# At initialization, LoRA output should match base (B=0 -> delta=0)
h_base = W_pretrained @ x
h_lora = lora_layer.forward(x)
assert np.allclose(h_base, h_lora, atol=1e-6), "At init, LoRA should not change output"

print(f"\n=== LoRA Layer (rank=16, alpha=32) ===")
print(f"  Weight matrix W: {W_pretrained.shape}")
print(f"  LoRA B: {lora_layer.B.shape}, A: {lora_layer.A.shape}")
print(
    f"  Scaling: alpha/r = {lora_layer.alpha}/{lora_layer.rank} = {lora_layer.scaling}"
)
print(
    f"  Trainable params: {lora_layer.trainable_params:,} / {lora_layer.total_params:,} total"
)
print(
    f"  Trainable fraction: {lora_layer.trainable_params / lora_layer.total_params:.4%}"
)
print(f"  At init: max|h_base - h_lora| = {np.max(np.abs(h_base - h_lora)):.2e}")

# Simulate training by setting B to non-zero
lora_layer.B = rng.standard_normal(lora_layer.B.shape).astype(np.float32) * 0.01
h_after = lora_layer.forward(x)
delta = np.max(np.abs(h_base - h_after))
print(f"  After B update: max|h_base - h_trained| = {delta:.4f}")

# Verify merge produces same output
W_merged = lora_layer.merge()
h_merged = W_merged @ x
assert np.allclose(
    h_after, h_merged, atol=1e-5
), "Merged weights should match LoRA forward"
print(f"  Merge verified: h_lora == W_merged @ x (within tolerance)")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Use AlignmentPipeline with LoRAConfig at ranks 4, 16, 64
# ══════════════════════════════════════════════════════════════════════
# Compare how different ranks affect the AlignmentConfig.

rank_configs = {}

for rank in [4, 16, 64]:
    alpha = rank * 2  # Common heuristic: alpha = 2 * rank

    config = AlignmentConfig(
        method="sft",
        base_model_id=base_model,
        lora=LoRAConfig(
            rank=rank,
            alpha=alpha,
            dropout=0.05,
            target_modules=("q_proj", "v_proj"),
        ),
        sft=SFTConfig(
            num_train_epochs=3,
            per_device_train_batch_size=4,
            learning_rate=2e-4,
            max_seq_length=512,
        ),
        experiment_dir=f"./sft_rank_{rank}",
    )

    rank_configs[rank] = config

    # Estimate trainable parameters for TinyLlama
    # TinyLlama: 22 layers, d_model=2048, q_proj and v_proj
    n_layers = 22
    d_model_llama = 2048
    n_target_modules = len(config.lora.target_modules)
    # Each LoRA adapter adds: rank * d_model (for A) + d_model * rank (for B)
    lora_params = n_layers * n_target_modules * 2 * d_model_llama * rank
    total_model_params = 1_100_000_000  # ~1.1B for TinyLlama

    print(f"\n  Rank {rank:>2} (alpha={alpha:>3}):")
    print(
        f"    LoRA params: {lora_params:>10,} ({lora_params / total_model_params:.4%} of model)"
    )
    print(
        f"    Config: lr={config.sft.learning_rate}, epochs={config.sft.num_train_epochs}"
    )

# Demonstrate pipeline creation (API demo — GPU training not required)
print(f"\n=== AlignmentPipeline API Demo ===")
registry = AdapterRegistry()
for rank, config in rank_configs.items():
    pipeline = AlignmentPipeline(config, adapter_registry=registry)
    print(f"  Pipeline created: method={config.method}, rank={config.lora.rank}")

print(f"\nProduction usage (requires GPU):")
print(f"  result = await pipeline.train(dataset=hf_dataset, adapter_name='sg_sft_r16')")
print(f"  # Returns: AlignmentResult with training metrics and adapter path")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Implement rank selector heuristic (stable rank)
# ══════════════════════════════════════════════════════════════════════
# Stable rank: r_stable(W) = ||W||_F^2 / ||W||_2^2
# where ||W||_F is the Frobenius norm and ||W||_2 is the spectral norm.
# Intuition: stable rank measures the "effective dimensionality" of W.
# A matrix with one dominant singular value has stable rank near 1.
# A matrix with uniformly distributed singular values has stable rank near min(d, k).


def stable_rank(W: np.ndarray) -> float:
    """Compute the stable rank of a matrix.

    stable_rank(W) = ||W||_F^2 / ||W||_2^2
    = sum(sigma_i^2) / sigma_max^2

    This gives the effective rank — how many dimensions carry significant information.
    """
    S = np.linalg.svd(W, compute_uv=False)
    fro_sq = np.sum(S**2)
    spectral_sq = S[0] ** 2
    return fro_sq / max(spectral_sq, 1e-10)


def recommend_lora_rank(W: np.ndarray, budget_fraction: float = 0.01) -> dict:
    """Recommend LoRA rank based on weight matrix analysis.

    Strategy:
    1. Compute stable rank as upper bound on useful rank
    2. Apply budget constraint (fraction of full parameters)
    3. Return recommendation with reasoning

    Args:
        W: Weight matrix to analyze
        budget_fraction: Max fraction of original params for LoRA
    """
    d, k = W.shape
    sr = stable_rank(W)

    # Budget constraint: rank * (d + k) <= budget_fraction * d * k
    max_rank_budget = int(budget_fraction * d * k / (d + k))

    # Stable rank gives an estimate of intrinsic dimensionality
    # Use fraction of stable rank as LoRA target
    rank_from_stable = max(1, int(sr * 0.5))

    # Final recommendation: minimum of budget and stable rank
    recommended = min(rank_from_stable, max_rank_budget)
    # Clip to powers of 2 for hardware efficiency
    recommended = max(1, 2 ** int(np.log2(max(1, recommended))))

    return {
        "stable_rank": sr,
        "rank_from_stable": rank_from_stable,
        "max_rank_budget": max_rank_budget,
        "recommended_rank": recommended,
        "dimensions": (d, k),
    }


# Test on different matrix types
print(f"\n=== Rank Selection Heuristic ===")

# Case 1: Random matrix (high stable rank)
W_random = rng.standard_normal((512, 512)).astype(np.float32)
rec_random = recommend_lora_rank(W_random)
print(f"\n  Random matrix {rec_random['dimensions']}:")
print(f"    Stable rank: {rec_random['stable_rank']:.1f}")
print(f"    Recommended LoRA rank: {rec_random['recommended_rank']}")

# Case 2: Low-rank matrix (low stable rank)
U = rng.standard_normal((512, 8)).astype(np.float32)
V = rng.standard_normal((8, 512)).astype(np.float32)
W_lowrank = U @ V
rec_lowrank = recommend_lora_rank(W_lowrank)
print(f"\n  Low-rank matrix (intrinsic rank=8):")
print(f"    Stable rank: {rec_lowrank['stable_rank']:.1f}")
print(f"    Recommended LoRA rank: {rec_lowrank['recommended_rank']}")

# Case 3: Identity-like matrix (stable rank = dimension)
W_identity = (
    np.eye(512, dtype=np.float32)
    + rng.standard_normal((512, 512)).astype(np.float32) * 0.01
)
rec_identity = recommend_lora_rank(W_identity)
print(f"\n  Near-identity matrix:")
print(f"    Stable rank: {rec_identity['stable_rank']:.1f}")
print(f"    Recommended LoRA rank: {rec_identity['recommended_rank']}")

# Compare against hand-tuned ranks
print(f"\n=== Comparison: Heuristic vs Common Choices ===")
print(f"{'Matrix Type':<20} {'Stable Rank':>12} {'Heuristic':>10} {'Common':>10}")
print("-" * 55)
print(
    f"{'Random':<20} {rec_random['stable_rank']:>12.1f} {rec_random['recommended_rank']:>10} {'16':>10}"
)
print(
    f"{'Low-rank (r=8)':<20} {rec_lowrank['stable_rank']:>12.1f} {rec_lowrank['recommended_rank']:>10} {'8':>10}"
)
print(
    f"{'Near-identity':<20} {rec_identity['stable_rank']:>12.1f} {rec_identity['recommended_rank']:>10} {'16':>10}"
)

print(f"\n  Key insight: stable rank tells you the intrinsic dimensionality of the")
print(f"  weight matrix. If stable rank is low, a small LoRA rank suffices.")
print(f"  If stable rank is high, you need more capacity — but budget constrains you.")
print(f"  In practice, rank=16 is a good default for most 7B-class models.")

print("\n=== Exercise 1 complete — LoRA internals and rank selection ===")
