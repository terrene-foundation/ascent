# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT09 — Exercise 1: LLM Architecture and Tokenization
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Understand decoder-only transformer internals — tokenization,
#   KV cache, parameter counting — and make your first LLM call via Delegate.
#
# TASKS:
#   1. Implement BPE tokenizer from scratch
#   2. Calculate model parameter count from architecture spec
#   3. Estimate KV cache memory requirements
#   4. Make first Delegate call with cost budget
#   5. Compare tokenizer output vs model's built-in tokenizer
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os
import re
from collections import Counter

import polars as pl

from kaizen_agents import Delegate

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()

if not os.environ.get("OPENAI_API_KEY"):
    print("\u26a0 OPENAI_API_KEY not set \u2014 skipping LLM exercises.")
    print("  Set it in .env to run this exercise with real LLM calls.")
    import sys

    sys.exit(0)

model = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))
print(f"LLM Model: {model}")

loader = ASCENTDataLoader()
reports = loader.load("ascent09", "sg_company_reports.parquet")
sample_texts = reports.select("text").head(5).to_series().to_list()
sample_text = (
    sample_texts[0] if sample_texts else "Singapore is a global financial hub."
)
print(f"Loaded {reports.height:,} documents")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Implement BPE tokenizer from scratch
# ══════════════════════════════════════════════════════════════════════


def get_word_freqs(text: str) -> dict[str, int]:
    # TODO: Use re.findall(r"\\w+|[^\\w\\s]", text.lower()) and return Counter.
    ____


def get_pair_freqs(vocab: dict[str, int]) -> Counter:
    # TODO: For each (word, freq), split word into chars, count adjacent pairs.
    ____


def train_bpe(
    text: str, num_merges: int = 50
) -> tuple[list[tuple[str, str]], dict[str, int]]:
    """BPE training: word-freq → char-level vocab → iterative pair-merge loop.
    Initial vocab: each word split into space-separated chars ("hi" → "h i").
    Each step: find most-frequent pair, record it, collapse it everywhere in vocab.
    """
    # TODO: Implement full BPE training loop — word freqs, char-level vocab init,
    # num_merges iterations of: get_pair_freqs → most_common(1) → rebuild vocab.
    ____
    ____
    ____


def tokenize_bpe(text: str, merges: list[tuple[str, str]]) -> list[str]:
    # TODO: Split into words; for each word start with list(word); apply each
    # merge rule left-to-right in-place; extend tokens list.
    ____
    ____


merges, vocab = train_bpe(sample_text, num_merges=50)
bpe_tokens = tokenize_bpe(sample_text[:500], merges)
print(f"\n=== BPE Tokenizer ===")
print(f"Learned {len(merges)} merge rules | Vocab size: {len(vocab)}")
print(f"Sample ({len(bpe_tokens)} tokens): {bpe_tokens[:20]}...")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Calculate model parameter count from architecture spec
# ══════════════════════════════════════════════════════════════════════


def count_parameters(
    vocab_size: int, d_model: int, n_layers: int, n_heads: int, d_ff: int
) -> dict[str, int]:
    """Return breakdown dict for a decoder-only transformer (seq_len=4096).
    Components: token_embedding, position_embedding, attention_per_layer
    (4×d_model²), ffn_per_layer (2×d_model×d_ff), layernorm_per_layer,
    total_transformer_layers, final_layernorm, lm_head, total_parameters.
    """
    # TODO: Compute all nine components and return them in the dict below.
    ____
    ____
    return {
        "token_embedding": token_emb,
        "position_embedding": pos_emb,
        "attention_per_layer": attn_per_layer,
        "ffn_per_layer": ffn_per_layer,
        "layernorm_per_layer": ln_per_layer,
        "total_transformer_layers": all_layers,
        "final_layernorm": final_ln,
        "lm_head": lm_head,
        "total_parameters": total,
    }


params = count_parameters(
    vocab_size=50_257, d_model=768, n_layers=12, n_heads=12, d_ff=3072
)
print(f"\n=== Parameter Count (GPT-2 Small) ===")
for k, v in params.items():
    print(f"  {k}: {v:>15,}")
print(f"  Total: {params['total_parameters'] / 1e6:.1f}M parameters")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Estimate KV cache memory requirements
# ══════════════════════════════════════════════════════════════════════


def estimate_kv_cache(
    batch_size: int,
    seq_len: int,
    n_layers: int,
    n_heads: int,
    d_head: int,
    dtype_bytes: int = 2,
) -> dict[str, float]:
    """per_token = 2 × n_layers × n_heads × d_head × dtype_bytes; total = batch × seq × per_token."""
    # TODO: Compute per_token, total_bytes, total_gb; return the dict below.
    ____
    return {
        "bytes_per_token": per_token,
        "total_bytes": total_bytes,
        "total_mb": total_bytes / (1024**2),
        "total_gb": total_gb,
        "batch_size": batch_size,
        "seq_len": seq_len,
    }


kv = estimate_kv_cache(1, 4096, 32, 32, 128, 2)
print(f"\n=== KV Cache (7B, seq_len=4096): {kv['total_mb']:.1f} MB ===")
for seq in [512, 2048, 4096, 16384, 32768]:
    k = estimate_kv_cache(1, seq, 32, 32, 128, 2)
    print(f"  seq_len={seq:>6}: {k['total_mb']:>8.1f} MB")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Make first Delegate call with cost budget
# ══════════════════════════════════════════════════════════════════════


async def first_delegate_call():
    # TODO: Create Delegate(model=model, budget_usd=1.0). Build a prompt asking
    # for financial metrics, sector, and sentiment from sample_text[:1000].
    # Stream response accumulating event.text. Budget cap is MANDATORY.
    ____
    ____
    ____


delegate_response = asyncio.run(first_delegate_call())


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Compare tokenizer output vs model's built-in tokenizer
# ══════════════════════════════════════════════════════════════════════


async def compare_tokenizers():
    # TODO: Tokenize test_sentence with our BPE. Create Delegate(budget_usd=0.5),
    # ask it how a modern LLM would tokenize the same sentence, stream response.
    # Print both tokenizations and note the key difference (training data scale).
    test_sentence = "Singapore's financial sector grew 8.2% in Q3."
    ____
    ____
    ____


asyncio.run(compare_tokenizers())

print(
    "\n✓ Exercise 1 complete — LLM architecture, tokenization, and first Delegate call"
)
