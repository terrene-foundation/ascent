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
    # TODO: three cleaning steps — lowercase, strip non-alphanum, collapse spaces
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
    # TODO: split each word into symbols; count (sym[j], sym[j+1]) pairs by freq
    pairs = Counter()
    ____
    ____
    ____
    return pairs


def merge_pair(pair: tuple[str, str], vocab: dict[str, int]) -> dict[str, int]:
    """Merge the most frequent pair in the vocabulary."""
    # TODO: bigram=" ".join(pair); replacement="".join(pair)
    #   return new vocab with bigram replaced everywhere
    ____
    ____
    return ____


corpus_words = Counter()
for text in sample_texts:
    for w in word_tokenize(text):
        corpus_words[____] += 1  # Hint: " ".join(list(w)) + " </w>"

print(f"\nInitial BPE vocab: {len(corpus_words)} word forms")

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
        if ____:  # Hint: word.endswith(suffix) and len(word)-len(suffix) >= 3
            return ____  # Hint: word[:-len(suffix)]
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

print("\nStemming: fast, rule-based, lossy")
print("Lemmatization: dictionary-based, produces valid words")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Extract n-grams and analyze frequencies
# ══════════════════════════════════════════════════════════════════════


def extract_ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """Extract n-grams from a list of tokens."""
    return ____  # Hint: [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]


all_tokens: list[str] = []
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
    """End-to-end: normalize → tokenize → token_count → vocabulary."""
    # TODO: Step 1 — add "normalized" column via map_elements(normalize_text)
    normalized = ____

    # TODO: Step 2 — add "tokens_str" (space-joined tokens) via map_elements
    tokenized = normalized.with_columns(____)

    # TODO: Step 3 — add "token_count" via str.split(" ").list.len()
    result = tokenized.with_columns(____)

    # TODO: Step 4 — collect all words; build Counter vocab capped at max_vocab
    all_words: list[str] = []
    for row in result.select("tokens_str").to_series().to_list():
        ____
    vocab = ____

    print(
        f"\nPipeline: {result.height} docs, avg {result['token_count'].mean():.1f} tokens, vocab={len(vocab)}"
    )
    print(f"Top 5: {[w for w, _ in vocab[:5]]}")
    return result


processed = preprocess_corpus(df, text_col="text", max_vocab=5000)

explorer_post = DataExplorer()
post_summary = asyncio.run(explorer_post.profile(processed.select("token_count")))
print(f"\nToken count distribution:\n{post_summary}")

print(
    "\n✓ Exercise 1 complete — NLP preprocessing: tokenization, BPE, stemming, n-grams"
)
