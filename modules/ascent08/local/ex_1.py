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


# ── Data Loading ──────────────────────────────────────────────────────

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
    # TODO: (1) lowercase, (2) replace non-alphanumeric with space, (3) collapse whitespace
    text = ____  # Hint: text.lower()
    text = ____  # Hint: re.sub(r"[^a-z0-9\s]", " ", text)
    text = ____  # Hint: re.sub(r"\s+", " ", text).strip()
    return text


def word_tokenize(text: str) -> list[str]:
    """Split normalized text into word tokens."""
    return ____  # Hint: normalize_text(text).split()


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
    # TODO: Iterate over (word, freq) pairs; split word into symbols;
    #   accumulate freq for each adjacent (symbols[j], symbols[j+1]) pair
    pairs = Counter()
    ____
    ____
    ____
    return pairs  # Hint: pairs[(symbols[j], symbols[j+1])] += freq


def merge_pair(pair: tuple[str, str], vocab: dict[str, int]) -> dict[str, int]:
    """Merge the most frequent pair in the vocabulary."""
    # TODO: Build bigram=" ".join(pair), replacement="".join(pair);
    #   return {word.replace(bigram, replacement): freq for word, freq in vocab.items()}
    ____
    ____
    return ____  # Hint: {word.replace(bigram, replacement): freq for word, freq in vocab.items()}


# Build initial character-level vocabulary from corpus
corpus_words = Counter()
for text in sample_texts:
    for w in word_tokenize(text):
        corpus_words[____] += 1  # Hint: " ".join(list(w)) + " </w>"

print(f"\nInitial BPE vocab size: {len(corpus_words)} word forms")

# Run 10 BPE merge iterations
num_merges = 10
merges_log = []
bpe_vocab = dict(corpus_words)

for step in range(num_merges):
    pairs = get_pair_freqs(bpe_vocab)
    if not pairs:
        break
    best_pair = ____  # Hint: pairs.most_common(1)[0]
    bpe_vocab = ____  # Hint: merge_pair(best_pair[0], bpe_vocab)
    ____  # Hint: merges_log.append((best_pair[0], best_pair[1]))
    print(f"  Merge {step+1}: {best_pair[0]} (freq={best_pair[1]})")

print(f"\nAfter {len(merges_log)} merges: {len(bpe_vocab)} word forms")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Compare stemming vs lemmatization
# ══════════════════════════════════════════════════════════════════════


def simple_stem(word: str) -> str:
    """Porter-like suffix stripping (simplified)."""
    suffixes = [
        "ing",
        "tion",
        "ment",
        "ness",
        "able",
        "ible",
        "ed",
        "ly",
        "er",
        "es",
        "s",
    ]
    for suffix in suffixes:
        # TODO: Strip suffix when remaining stem has at least 3 chars
        if ____:  # Hint: word.endswith(suffix) and len(word) - len(suffix) >= 3
            return ____  # Hint: word[: -len(suffix)]
    return word


test_words = [
    "running",
    "universities",
    "better",
    "happiness",
    "studies",
    "economically",
]
print(f"\n{'Word':<18} {'Stemmed':<15} {'Note'}")
print("-" * 50)
for word in test_words:
    stemmed = simple_stem(word)
    print(
        f"{word:<18} {stemmed:<15} {'aggressive' if stemmed != word[:3] else 'minimal'}"
    )

print("\nStemming: fast, rule-based, lossy (may produce non-words)")
print("Lemmatization: dictionary-based, produces valid words, context-aware")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Extract n-grams and analyze frequencies
# ══════════════════════════════════════════════════════════════════════


def extract_ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """Extract n-grams from a list of tokens."""
    return ____  # Hint: [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


# TODO: Collect all tokens across the full corpus into all_tokens
all_tokens = []
for text in df.select("text").to_series().to_list():
    ____  # Hint: all_tokens.extend(word_tokenize(text))

for n in [1, 2, 3]:
    ngrams = extract_ngrams(all_tokens, n)
    freq = Counter(ngrams).most_common(10)
    label = {1: "Unigrams", 2: "Bigrams", 3: "Trigrams"}[n]
    print(f"\n--- Top 10 {label} ---")
    for gram, count in freq:
        print(f"  {' '.join(gram):<30} {count:>6}")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Full preprocessing pipeline function
# ══════════════════════════════════════════════════════════════════════


def preprocess_corpus(
    df: pl.DataFrame,
    text_col: str = "text",
    max_vocab: int = 5000,
) -> pl.DataFrame:
    """End-to-end preprocessing: normalize, tokenize, count vocab."""
    # TODO: Step 1 — add "normalized" column by applying normalize_text via map_elements
    normalized = ____  # Hint: df.with_columns(pl.col(text_col).map_elements(normalize_text, return_dtype=pl.Utf8).alias("normalized"))

    # TODO: Step 2 — add "tokens_str" column: space-join of word_tokenize results
    tokenized = normalized.with_columns(
        ____
    )  # Hint: pl.col("normalized").map_elements(lambda t: " ".join(word_tokenize(t)), return_dtype=pl.Utf8).alias("tokens_str")

    # TODO: Step 3 — add "token_count" column: split "tokens_str" and take list length
    result = tokenized.with_columns(
        ____
    )  # Hint: pl.col("tokens_str").str.split(" ").list.len().alias("token_count")

    # TODO: Step 4 — build vocabulary: extend all_words from each row, then Counter.most_common
    all_words: list[str] = []
    for row in result.select("tokens_str").to_series().to_list():
        ____  # Hint: all_words.extend(row.split())
    vocab = ____  # Hint: Counter(all_words).most_common(max_vocab)

    print(f"\nPipeline summary:")
    print(f"  Documents processed: {result.height}")
    print(f"  Avg tokens/doc: {result['token_count'].mean():.1f}")
    print(f"  Vocabulary size (capped): {len(vocab)}")
    print(f"  Top 5 terms: {[w for w, _ in vocab[:5]]}")
    return result


processed = preprocess_corpus(df, text_col="text", max_vocab=5000)

explorer_post = DataExplorer()
post_summary = asyncio.run(explorer_post.profile(processed.select("token_count")))
print(f"\nToken count distribution:\n{post_summary}")

print(
    "\n✓ Exercise 1 complete — NLP preprocessing pipeline with tokenization, BPE, stemming, n-grams"
)
