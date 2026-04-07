# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT08 — Exercise 1: Text Preprocessing Pipeline
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Build a complete NLP preprocessing pipeline — tokenization
#   (word/BPE), stemming, lemmatization, n-grams, and text normalization
#   on Singapore news articles using kailash-ml DataExplorer.
#
# TASKS:
#   1. Implement word-level tokenization with normalization
#   2. Implement BPE tokenization step-by-step
#   3. Compare stemming vs lemmatization
#   4. Extract n-grams and analyze frequencies
#   5. Build full preprocessing pipeline function
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import re
from collections import Counter

import polars as pl

from kailash_ml import DataExplorer, PreprocessingPipeline

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()

loader = ASCENTDataLoader()
df = loader.load("ascent08", "sg_news_articles.parquet")

explorer = DataExplorer()
summary = asyncio.run(explorer.profile(df))
print(f"=== Dataset: {df.height} articles, columns: {df.columns} ===")
print(summary)


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Word-level tokenization with normalization
# ══════════════════════════════════════════════════════════════════════


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    # TODO: three steps — lowercase, strip non-alphanum chars, collapse whitespace
    ____
    ____
    ____
    return text


def word_tokenize(text: str) -> list[str]:
    """Split normalized text into word tokens."""
    # TODO: normalize then split on whitespace
    return ____


sample_texts = df.select("text").head(5).to_series().to_list()
for i, text in enumerate(sample_texts):
    tokens = word_tokenize(text)
    print(f"\nArticle {i+1}: {len(tokens)} tokens — first 10: {tokens[:10]}")

print("\n--- Word tokenization complete ---")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: BPE tokenization step-by-step
# ══════════════════════════════════════════════════════════════════════


def get_pair_freqs(vocab: dict[str, int]) -> Counter:
    """Count adjacent symbol pairs across the vocabulary."""
    # TODO: for each word split into symbols; accumulate (sym[j], sym[j+1]) counts * freq
    ____
    ____
    ____
    ____


def merge_pair(pair: tuple[str, str], vocab: dict[str, int]) -> dict[str, int]:
    """Merge the most frequent pair everywhere in the vocabulary."""
    # TODO: bigram=" ".join(pair); replacement="".join(pair); return rebuilt vocab
    ____


corpus_words: Counter = Counter()
# TODO: populate corpus_words: for each word in sample_texts, add " ".join(list(w)) + " </w>"
____
____

print(f"\nInitial BPE vocab: {len(corpus_words)} word forms")

num_merges, merges_log, bpe_vocab = 10, [], dict(corpus_words)
# TODO: loop num_merges steps: get best pair, merge vocab, log, print
for step in range(num_merges):
    pairs = get_pair_freqs(bpe_vocab)
    if not pairs:
        break
    best_pair = ____  # Hint: pairs.most_common(1)[0]
    bpe_vocab = ____  # Hint: merge_pair(best_pair[0], bpe_vocab)
    ____  # Hint: merges_log.append((best_pair[0], best_pair[1]))
    ____  # Hint: print(f"  Merge {step+1}: {best_pair[0]} (freq={best_pair[1]})")

print(f"\nAfter {len(merges_log)} merges: {len(bpe_vocab)} word forms")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Compare stemming vs lemmatization
# ══════════════════════════════════════════════════════════════════════


def simple_stem(word: str) -> str:
    """Porter-like suffix stripping (simplified)."""
    # TODO: define suffixes list (ing, tion, ment, ness, able, ible, ed, ly, er, es, s)
    # TODO: for each suffix: if word.endswith(suffix) and len(word)-len(suffix) >= 3, strip it
    ____
    ____
    ____
    return word


test_words = [
    "running",
    "universities",
    "better",
    "happiness",
    "studies",
    "economically",
]
print(f"\n{'Word':<18} {'Stemmed':<15}")
for word in test_words:
    print(f"{word:<18} {simple_stem(word):<15}")
print("\nStemming: fast, rule-based | Lemmatization: dictionary-based, valid words")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Extract n-grams and analyze frequencies
# ══════════════════════════════════════════════════════════════════════


def extract_ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """Extract n-grams from a list of tokens."""
    # TODO: sliding window of size n over tokens list
    return ____


# TODO: build all_tokens by extending with word_tokenize(text) for each article
all_tokens: list[str] = []
____

for n in [1, 2, 3]:
    # TODO: extract n-grams, get top 10, print label and counts
    ____
    ____
    ____


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Full preprocessing pipeline function
# ══════════════════════════════════════════════════════════════════════


def preprocess_corpus(
    df: pl.DataFrame,
    text_col: str = "text",
    max_vocab: int = 5000,
) -> pl.DataFrame:
    """End-to-end: normalize → tokenize → token_count → vocabulary."""
    # TODO: build normalized, tokenized, result DataFrames; build vocab Counter
    ____
    ____
    ____
    all_words: list[str] = []
    ____
    vocab = ____
    print(
        f"\nPipeline: {result.height} docs, avg {result['token_count'].mean():.1f} tokens, vocab={len(vocab)}"
    )
    return result


processed = preprocess_corpus(df, text_col="text", max_vocab=5000)

explorer_post = DataExplorer()
post_summary = asyncio.run(explorer_post.profile(processed.select("token_count")))
print(f"\nToken count distribution:\n{post_summary}")
print(
    "\n✓ Exercise 1 complete — NLP preprocessing: tokenization, BPE, stemming, n-grams"
)
