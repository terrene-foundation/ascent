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
summary = explorer.analyze(df)
print(f"=== Dataset: {df.height} articles, columns: {df.columns} ===")
print(summary)


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Word-level tokenization with normalization
# ══════════════════════════════════════════════════════════════════════


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    # TODO: Lowercase the text.
    # Hint: str.lower()
    text = ____
    # TODO: Replace non-alphanumeric characters with a space.
    # Hint: re.sub with a character class excluding letters, digits, whitespace
    text = ____
    # TODO: Collapse runs of whitespace to a single space and strip edges.
    # Hint: re.sub(r"\s+", ...) followed by .strip()
    text = ____
    return text


def word_tokenize(text: str) -> list[str]:
    """Split normalized text into word tokens."""
    # TODO: Normalize then split on whitespace.
    # Hint: normalize_text(text).split()
    return ____


sample_texts = df.select("text").head(5).to_series().to_list()
for i, text in enumerate(sample_texts):
    tokens = word_tokenize(text)
    print(f"\nArticle {i+1}: {len(tokens)} tokens")
    print(f"  First 10: {tokens[:10]}")

print("\n--- Word tokenization complete ---")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: BPE tokenization step-by-step
# ══════════════════════════════════════════════════════════════════════


def get_pair_freqs(vocab: dict[str, int]) -> Counter:
    """Count adjacent symbol pairs across the vocabulary."""
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for j in range(len(symbols) - 1):
            # TODO: Increment the count of the adjacent symbol pair by freq.
            # Hint: pairs[(symbols[j], symbols[j + 1])] += freq
            ____
    return pairs


def merge_pair(pair: tuple[str, str], vocab: dict[str, int]) -> dict[str, int]:
    """Merge the most frequent pair in the vocabulary."""
    new_vocab = {}
    # TODO: Build the bigram string and the merged replacement string.
    # Hint: " ".join(pair) and "".join(pair)
    bigram = ____
    replacement = ____
    for word, freq in vocab.items():
        # TODO: Replace every occurrence of the bigram with the merged token.
        # Hint: word.replace(bigram, replacement)
        new_word = ____
        new_vocab[new_word] = freq
    return new_vocab


# Build initial character-level vocabulary from corpus
corpus_words = Counter()
for text in sample_texts:
    for w in word_tokenize(text):
        # TODO: Add a space-separated character sequence with "</w>" end marker.
        # Hint: " ".join(list(w)) + " </w>"
        corpus_words[____] += 1

print(f"\nInitial BPE vocab size: {len(corpus_words)} word forms")

# Run 10 BPE merge iterations
num_merges = 10
merges_log = []
bpe_vocab = dict(corpus_words)

for step in range(num_merges):
    pairs = get_pair_freqs(bpe_vocab)
    if not pairs:
        break
    # TODO: Select the single most frequent pair from the Counter.
    # Hint: pairs.most_common(1)[0]
    best_pair = ____
    # TODO: Apply the merge to produce an updated vocabulary.
    # Hint: merge_pair(best_pair[0], bpe_vocab)
    bpe_vocab = ____
    # TODO: Record the merged pair and its frequency in merges_log.
    # Hint: merges_log.append((best_pair[0], best_pair[1]))
    ____
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
        # TODO: If word ends with the suffix and the stem is at least 3 chars, return the stem.
        # Hint: word.endswith(suffix) and len(word) - len(suffix) >= 3
        if ____:
            return ____
    return word


# TODO: Configure PreprocessingPipeline to lemmatize the "text" column.
# Hint: PreprocessingPipeline(text_columns=[...], steps=["lemmatize"])
pipeline = ____
lemmatized_df = pipeline.transform(df.head(5))

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
    # TODO: Return a list of n-gram tuples using a sliding window over tokens.
    # Hint: [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return ____


# Collect n-grams across all articles
all_tokens = []
for text in df.select("text").to_series().to_list():
    # TODO: Tokenize the text and extend all_tokens with the result.
    # Hint: word_tokenize(text) returns a list of tokens
    ____

for n in [1, 2, 3]:
    ngrams = extract_ngrams(all_tokens, n)
    freq = Counter(ngrams).most_common(10)
    label = {1: "Unigrams", 2: "Bigrams", 3: "Trigrams"}[n]
    print(f"\n--- Top 10 {label} ---")
    for gram, count in freq:
        display = " ".join(gram)
        print(f"  {display:<30} {count:>6}")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Full preprocessing pipeline function
# ══════════════════════════════════════════════════════════════════════


def preprocess_corpus(
    df: pl.DataFrame,
    text_col: str = "text",
    max_vocab: int = 5000,
) -> pl.DataFrame:
    """End-to-end preprocessing: normalize, tokenize, count vocab."""
    # Step 1: Normalize — add a "normalized" column
    # TODO: Use with_columns + map_elements to apply normalize_text to text_col.
    # Hint: pl.col(text_col).map_elements(normalize_text, return_dtype=pl.Utf8).alias("normalized")
    normalized = ____

    # Step 2: Tokenize — add a "tokens_str" column (space-joined tokens)
    # TODO: Apply word_tokenize to "normalized" and join results into a string.
    # Hint: map_elements(lambda t: " ".join(word_tokenize(t)), return_dtype=pl.Utf8).alias("tokens_str")
    tokenized = normalized.with_columns(____)

    # Step 3: Token count per document — add "token_count" column
    # TODO: Split "tokens_str" on spaces and take the list length.
    # Hint: pl.col("tokens_str").str.split(" ").list.len().alias("token_count")
    result = tokenized.with_columns(____)

    # Step 4: Build vocabulary from all tokens
    all_words = []
    for row in result.select("tokens_str").to_series().to_list():
        # TODO: Split the space-joined token string and extend all_words.
        # Hint: row.split()
        ____
    # TODO: Build a frequency-ordered vocabulary capped at max_vocab entries.
    # Hint: Counter(all_words).most_common(max_vocab)
    vocab = ____

    print(f"\nPipeline summary:")
    print(f"  Documents processed: {result.height}")
    print(f"  Avg tokens/doc: {result['token_count'].mean():.1f}")
    print(f"  Vocabulary size (capped): {len(vocab)}")
    print(f"  Top 5 terms: {[w for w, _ in vocab[:5]]}")

    return result


processed = preprocess_corpus(df, text_col="text", max_vocab=5000)

# Validate with DataExplorer
explorer_post = DataExplorer()
post_summary = explorer_post.analyze(processed.select("token_count"))
print(f"\nToken count distribution:\n{post_summary}")

print(
    "\n✓ Exercise 1 complete — NLP preprocessing pipeline with tokenization, BPE, stemming, n-grams"
)
