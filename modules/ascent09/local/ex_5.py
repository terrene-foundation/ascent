# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT09 — Exercise 5: Inference Optimization and Quantization
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Implement key inference optimizations from scratch —
#   quantization (INT8 absmax and zero-point), KV cache management,
#   speculative decoding — and validate against OnnxBridge/InferenceServer.
#
# TASKS:
#   1. Implement INT8 quantization (absmax and zero-point)
#   2. Use OnnxBridge.export() at FP32, FP16, INT8
#   3. Implement KV cache manager with sliding window eviction
#   4. Implement speculative decoding (draft + verify)
#   5. Use InferenceServer with batch benchmarking
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os
import time

import numpy as np
import polars as pl

from kailash_ml import ModelVisualizer, OnnxBridge
from kailash_ml.engines.inference_server import InferenceServer

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()

# ── Data Loading ─────────────────────────────────────────────────────

loader = ASCENTDataLoader()
reports = loader.load("ascent09", "sg_company_reports.parquet")

print(f"=== SG Company Reports Dataset ===")
print(f"Shape: {reports.shape}")
print(f"Columns: {reports.columns}")
sample_texts = reports.select("text").head(10).to_series().to_list()
print(f"Sample text length: {len(sample_texts[0])} chars")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Implement INT8 quantization from scratch
# ══════════════════════════════════════════════════════════════════════
# Quantization cuts memory 2-4x and enables integer arithmetic acceleration.
# Absmax: scale = max(|W|) / 127,  W_int8 = round(W / scale)
# Zero-point: scale = (max - min) / 255,  zero_point = round(-min / scale)

rng = np.random.default_rng(42)


def quantize_absmax(weights: np.ndarray) -> dict:
    """Absmax (symmetric) quantization to INT8.

    Maps [-absmax, absmax] -> [-127, 127]
    scale = absmax / 127
    """
    # TODO: Compute scale, quantize to int8, dequantize back to float32
    # Hint: absmax = np.max(np.abs(weights))
    #   scale = absmax / 127.0
    #   quantized = np.clip(np.round(weights / scale), -127, 127).astype(np.int8)
    #   dequantized = quantized.astype(np.float32) * scale
    absmax = ____
    scale = ____
    quantized = ____
    dequantized = ____

    return {
        "quantized": quantized,
        "dequantized": dequantized,
        "scale": scale,
        "absmax": absmax,
    }


def quantize_zero_point(weights: np.ndarray) -> dict:
    """Zero-point (asymmetric) quantization to INT8.

    Maps [min, max] -> [0, 255]
    scale = (max - min) / 255,  zero_point = round(-min / scale)
    """
    # TODO: Compute scale and zero_point, quantize to uint8, dequantize
    # Hint: w_min, w_max = weights.min(), weights.max()
    #   scale = (w_max - w_min) / 255.0
    #   zero_point = np.clip(int(np.round(-w_min / scale)), 0, 255)
    #   quantized = np.clip(np.round(weights / scale) + zero_point, 0, 255).astype(np.uint8)
    #   dequantized = (quantized.astype(np.float32) - zero_point) * scale
    w_min, w_max = ____
    scale = ____
    zero_point = ____
    quantized = ____
    dequantized = ____

    return {
        "quantized": quantized,
        "dequantized": dequantized,
        "scale": scale,
        "zero_point": zero_point,
    }


def compute_quantization_error(original: np.ndarray, dequantized: np.ndarray) -> dict:
    """Compute quantization error metrics."""
    error = original - dequantized
    mse = float(np.mean(error**2))
    max_err = float(np.max(np.abs(error)))
    signal_power = float(np.mean(original**2))
    sqnr_db = 10 * np.log10(signal_power / max(mse, 1e-20))

    return {
        "mse": mse,
        "max_error": max_err,
        "sqnr_db": sqnr_db,
        "relative_error": float(np.sqrt(mse) / max(np.sqrt(signal_power), 1e-10)),
    }


W_normal = rng.standard_normal((1024, 1024)).astype(np.float32) * 0.02
W_relu = np.maximum(0, rng.standard_normal((1024, 1024)).astype(np.float32) * 0.02)

print(f"\n=== INT8 Quantization ===")
for name, W in [("Normal weights", W_normal), ("ReLU-activated", W_relu)]:
    abs_result = quantize_absmax(W)
    zp_result = quantize_zero_point(W)

    abs_err = compute_quantization_error(W, abs_result["dequantized"])
    zp_err = compute_quantization_error(W, zp_result["dequantized"])

    print(f"\n  {name} {W.shape}:")
    print(f"    Original: {W.nbytes / 1024:.0f} KB (FP32)")
    print(f"    INT8:     {W.size / 1024:.0f} KB (4x reduction)")
    print(f"")
    print(f"    {'Metric':<20} {'Absmax':>12} {'Zero-Point':>12}")
    print(f"    {'-' * 44}")
    print(f"    {'MSE':<20} {abs_err['mse']:>12.2e} {zp_err['mse']:>12.2e}")
    print(
        f"    {'Max error':<20} {abs_err['max_error']:>12.2e} {zp_err['max_error']:>12.2e}"
    )
    print(
        f"    {'SQNR (dB)':<20} {abs_err['sqnr_db']:>12.1f} {zp_err['sqnr_db']:>12.1f}"
    )

model_params = 1_100_000_000
print(f"\n  Memory Savings by Precision:")
for dtype, bytes_per, label in [
    ("FP32", 4, "Full precision"),
    ("FP16/BF16", 2, "Half precision"),
    ("INT8", 1, "8-bit quantized"),
    ("INT4", 0.5, "4-bit quantized (NF4/GPTQ)"),
]:
    size_gb = model_params * bytes_per / (1024**3)
    print(f"    {dtype:<10} {size_gb:>6.2f} GB  ({label})")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Use OnnxBridge.export() at FP32, FP16, INT8
# ══════════════════════════════════════════════════════════════════════

# TODO: Create an OnnxBridge instance
# Hint: bridge = OnnxBridge()
bridge = ____

print(f"\n=== OnnxBridge Export ===")
print(f"  OnnxBridge converts models to ONNX format for portable deployment.")
print(f"")
print(f"  Export API:")
print(f"    bridge = OnnxBridge()")
print(f"    await bridge.export(model, input_shape=(1, 512), output_path='model.onnx')")
print(
    f"    await bridge.export(model, input_shape=(1, 512), output_path='model_fp16.onnx', dtype='fp16')"
)
print(
    f"    await bridge.export(model, input_shape=(1, 512), output_path='model_int8.onnx', dtype='int8')"
)
print(f"")
print(f"  Expected file sizes for a 1.1B parameter model:")
for dtype, factor, label in [
    ("FP32", 4.0, "model.onnx"),
    ("FP16", 2.0, "model_fp16.onnx"),
    ("INT8", 1.0, "model_int8.onnx"),
]:
    size_gb = model_params * factor / (1024**3)
    print(f"    {label:<25} ~{size_gb:.1f} GB")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Implement KV cache manager with sliding window eviction
# ══════════════════════════════════════════════════════════════════════
# KV cache grows linearly with sequence length.
# Sliding window limits it to W most recent tokens: O(n) -> O(W).


class KVCacheManager:
    """KV cache with sliding window eviction."""

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        d_head: int,
        window_size: int,
        dtype: np.dtype = np.float16,
    ):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_head = d_head
        self.window_size = window_size
        self.dtype = dtype

        # TODO: Pre-allocate circular cache: shape (n_layers, 2, n_heads, window_size, d_head)
        # Hint: np.zeros((n_layers, 2, n_heads, window_size, d_head), dtype=dtype)
        self.cache = ____
        self.position = 0
        self.length = 0

    def add_token(self, layer: int, key: np.ndarray, value: np.ndarray):
        """Add a new KV pair for one token at the specified layer."""
        # TODO: Write key/value at circular position self.position % self.window_size
        # Hint: write_pos = self.position % self.window_size
        #   self.cache[layer, 0, :, write_pos, :] = key
        write_pos = ____
        ____
        ____

    def step(self):
        """Advance the position counter after all layers have written."""
        self.position += 1
        self.length = min(self.length + 1, self.window_size)

    def get_kv(self, layer: int) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve current K, V cache for a layer."""
        if self.length < self.window_size:
            keys = self.cache[layer, 0, :, : self.length, :]
            values = self.cache[layer, 1, :, : self.length, :]
        else:
            # TODO: Reorder circular buffer from oldest to newest
            # Hint: start = self.position % self.window_size
            #   indices = [(start + i) % self.window_size for i in range(self.window_size)]
            start = ____
            indices = ____
            keys = self.cache[layer, 0, :, indices, :]
            values = self.cache[layer, 1, :, indices, :]
        return keys, values

    @property
    def memory_bytes(self) -> int:
        return self.cache.nbytes

    @property
    def memory_mb(self) -> float:
        return self.memory_bytes / (1024**2)

    def utilization(self) -> float:
        return self.length / self.window_size


n_layers = 32
n_heads = 32
d_head = 128
window_size = 4096

cache = KVCacheManager(n_layers, n_heads, d_head, window_size, dtype=np.float16)

print(f"\n=== KV Cache Manager ===")
print(f"  Config: {n_layers} layers, {n_heads} heads, d_head={d_head}")
print(f"  Window size: {window_size} tokens")
print(f"  Total cache memory: {cache.memory_mb:.1f} MB")
print(f"  Bytes per token: {2 * n_layers * n_heads * d_head * 2:,}")

for step in range(100):
    for layer in range(n_layers):
        k = rng.standard_normal((n_heads, d_head)).astype(np.float16)
        v = rng.standard_normal((n_heads, d_head)).astype(np.float16)
        cache.add_token(layer, k, v)
    cache.step()

print(
    f"  After 100 tokens: length={cache.length}, utilization={cache.utilization():.2%}"
)

keys, values = cache.get_kv(0)
print(f"  Layer 0 KV shape: keys={keys.shape}, values={values.shape}")

print(f"\n  Memory Comparison (32-layer, 32-head, d_head=128, FP16):")
print(f"    {'Seq Length':>12} {'Full Cache':>12} {'Window=4096':>12} {'Savings':>10}")
print(f"    {'-' * 48}")
for seq_len in [1024, 4096, 8192, 16384, 32768, 131072]:
    full_mb = 2 * n_layers * n_heads * d_head * 2 * seq_len / (1024**2)
    window_mb = cache.memory_mb
    savings = max(0, 1 - window_mb / full_mb) * 100
    print(
        f"    {seq_len:>12} {full_mb:>10.1f} MB {window_mb:>10.1f} MB {savings:>9.0f}%"
    )


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Implement speculative decoding (draft + verify)
# ══════════════════════════════════════════════════════════════════════
# Draft model proposes K tokens; target model verifies all K in one pass.
# Accept tokens left-to-right until first rejection — always get at least 1.


def simulate_speculative_decoding(
    n_tokens: int,
    draft_acceptance_rate: float = 0.7,
    draft_lookahead: int = 5,
    draft_latency_ms: float = 2.0,
    target_latency_ms: float = 30.0,
) -> dict:
    """Simulate speculative decoding throughput."""
    tokens_generated = 0
    total_time_ms = 0.0
    n_rounds = 0
    accepted_counts = []

    while tokens_generated < n_tokens:
        n_rounds += 1

        # TODO: Compute time for draft (K sequential) + target (1 verify pass)
        # Hint: draft_time = draft_lookahead * draft_latency_ms
        #   verify_time = target_latency_ms
        draft_time = ____
        verify_time = ____

        # Accept tokens until first rejection (always get at least 1 corrected token)
        accepted = 0
        for _ in range(draft_lookahead):
            if rng.random() < draft_acceptance_rate:
                accepted += 1
            else:
                break
        tokens_this_round = accepted + 1
        accepted_counts.append(accepted)

        tokens_generated += tokens_this_round
        total_time_ms += draft_time + verify_time

    baseline_time_ms = n_tokens * target_latency_ms

    return {
        "tokens": min(tokens_generated, n_tokens),
        "rounds": n_rounds,
        "total_time_ms": total_time_ms,
        "baseline_time_ms": baseline_time_ms,
        "speedup": baseline_time_ms / max(total_time_ms, 1),
        "avg_accepted": float(np.mean(accepted_counts)),
        "tokens_per_second": n_tokens / (total_time_ms / 1000),
        "baseline_tps": n_tokens / (baseline_time_ms / 1000),
    }


print(f"\n=== Speculative Decoding ===")
print(f"  Draft model proposes K tokens, target verifies in one pass.")

print(f"\n  {'Accept Rate':>12} {'Lookahead':>10} {'Speedup':>10} {'Avg Accepted':>14}")
print(f"  {'-' * 48}")
for accept_rate in [0.5, 0.6, 0.7, 0.8, 0.9]:
    for lookahead in [3, 5, 8]:
        result = simulate_speculative_decoding(
            n_tokens=200,
            draft_acceptance_rate=accept_rate,
            draft_lookahead=lookahead,
        )
        if lookahead == 5:
            print(
                f"  {accept_rate:>12.0%} {lookahead:>10} "
                f"{result['speedup']:>9.2f}x {result['avg_accepted']:>14.1f}"
            )

typical = simulate_speculative_decoding(
    n_tokens=500,
    draft_acceptance_rate=0.75,
    draft_lookahead=5,
    draft_latency_ms=2.0,
    target_latency_ms=30.0,
)
print(f"\n  Typical scenario (75% acceptance, K=5):")
print(f"    Tokens: {typical['tokens']}")
print(f"    Rounds: {typical['rounds']} (vs {typical['tokens']} autoregressive steps)")
print(
    f"    Time: {typical['total_time_ms']:.0f} ms (vs {typical['baseline_time_ms']:.0f} ms baseline)"
)
print(f"    Speedup: {typical['speedup']:.2f}x")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Use InferenceServer with batch benchmarking
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== InferenceServer Benchmarking ===")
print(f"  InferenceServer provides production model serving with:")
print(f"    - Model caching (warm_cache for low first-request latency)")
print(f"    - Batch inference (amortize overhead across requests)")
print(f"    - Performance monitoring (latency percentiles)")
print(f"")
print(f"  API pattern:")
print(f"    server = InferenceServer(model_registry, cache_size=5)")
print(f"    await server.warm_cache(['model_name'])")
print(f"    result = await server.predict(model_name='...', features={{...}})")

print(f"\n  Throughput vs Latency (Pareto Frontier):")
print(
    f"    {'Batch Size':>12} {'Latency (ms)':>14} {'Throughput':>14} {'Efficiency':>12}"
)
print(f"    {'-' * 55}")

for batch_size in [1, 2, 4, 8, 16, 32, 64]:
    base_latency = 30.0
    overhead_per_sample = 2.0
    batch_latency = base_latency + overhead_per_sample * np.log2(max(1, batch_size))
    throughput = batch_size / (batch_latency / 1000)
    efficiency = throughput / batch_size

    print(
        f"    {batch_size:>12} {batch_latency:>12.1f} ms "
        f"{throughput:>12.0f} req/s {efficiency:>11.1f}%"
    )

print(f"\n=== Optimization Stack Summary ===")
print(f"  {'Technique':<30} {'Memory Saving':>15} {'Speedup':>10}")
print(f"  {'-' * 57}")
print(f"  {'INT8 quantization':<30} {'4x':>15} {'1.5-2x':>10}")
print(f"  {'INT4 quantization (NF4)':<30} {'8x':>15} {'2-3x':>10}")
print(f"  {'KV cache windowing':<30} {'O(W) vs O(n)':>15} {'1x':>10}")
print(f"  {'Speculative decoding':<30} {'1x':>15} {'2-3x':>10}")
print(f"  {'Batch inference':<30} {'1x':>15} {'4-8x':>10}")
print(f"  {'ONNX Runtime':<30} {'1x':>15} {'1.5-2x':>10}")
print(f"  {'Combined':<30} {'4-8x':>15} {'5-15x':>10}")

print("\n=== Exercise 5 complete -- inference optimization and quantization ===")
