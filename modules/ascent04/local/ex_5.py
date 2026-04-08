# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT04 — Exercise 5: NLP: Text to Topics
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Full NLP pipeline from TF-IDF fundamentals to BERTopic —
#   bag-of-words warmup, TF-IDF, NMF, then neural topic modelling.
#
# TASKS:
#   1. TF-IDF warmup: bag-of-words, term frequency, inverse document frequency
#   2. NMF topic extraction from TF-IDF matrix
#   3. Load and preprocess Singapore news corpus
#   4. Build BERTopic model (UMAP + HDBSCAN + c-TF-IDF)
#   5. Evaluate topic coherence (NPMI) and visualise distributions
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import polars as pl
from collections import Counter

from kailash_ml import ModelVisualizer

from shared import ASCENTDataLoader

try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
except ImportError:
    BERTopic = None
    SentenceTransformer = None


# ══════════════════════════════════════════════════════════════════════
# TASK 1: TF-IDF Warmup — from bag-of-words to term weighting
# ══════════════════════════════════════════════════════════════════════
# TF-IDF = Term Frequency × Inverse Document Frequency
#   TF(t, d)  = count(t in d) / count(all words in d)
#   IDF(t)    = log(N / df(t))     where df(t) = number of docs containing t

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

toy_corpus = [
    "Singapore economy grew strongly in 2024",
    "Singapore property market shows resilience",
    "MAS tightens monetary policy amid global uncertainty",
    "Property developers report strong demand",
    "Singapore government announces new housing measures",
]

# TODO: Build CountVectorizer(stop_words="english") and fit_transform on toy_corpus
bow_vectorizer = ____
X_bow = ____
bow_vocab = bow_vectorizer.get_feature_names_out()

print(f"=== Bag-of-Words Warmup ===")
print(f"Vocabulary size: {len(bow_vocab)}")
print(f"Matrix shape: {X_bow.shape} (docs × vocab)")
print(f"\nDocument 0 non-zero terms:")
doc0 = X_bow[0].toarray()[0]
for term, count in zip(bow_vocab, doc0):
    if count > 0:
        print(f"  '{term}': {int(count)}")

# TODO: Build TfidfVectorizer(stop_words="english", norm="l2") and fit_transform
tfidf_vectorizer = ____
X_tfidf = ____
tfidf_vocab = tfidf_vectorizer.get_feature_names_out()

print(f"\n=== TF-IDF Weights (Document 0 vs Document 1) ===")
print(f"{'Term':<20} {'Doc0 TF-IDF':>14} {'Doc1 TF-IDF':>14} {'IDF':>10}")
print("─" * 62)
idf_values = tfidf_vectorizer.idf_
doc0_tfidf = X_tfidf[0].toarray()[0]
doc1_tfidf = X_tfidf[1].toarray()[0]
for term, idf, t0, t1 in sorted(
    zip(tfidf_vocab, idf_values, doc0_tfidf, doc1_tfidf),
    key=lambda x: -abs(x[2] + x[3]),
)[:12]:
    print(f"  {term:<20} {t0:>14.4f} {t1:>14.4f} {idf:>10.4f}")

print("\nKey insight:")
print("  'singapore' appears in 3/5 docs → lower IDF → penalised")
print("  'monetary' appears in 1/5 docs  → higher IDF → rewarded")

# Step 3: NMF on TF-IDF
from sklearn.decomposition import NMF

n_nmf_topics = 2
# TODO: NMF(n_components=n_nmf_topics, random_state=42) and fit_transform on X_tfidf
nmf_toy = ____
W_toy = ____  # Doc-topic matrix
# TODO: Topic-word matrix (the .components_ attribute)
H_toy = ____

print(f"\n=== NMF Topics from TF-IDF (toy corpus) ===")
for t in range(n_nmf_topics):
    top_words = [tfidf_vocab[i] for i in H_toy[t].argsort()[-5:][::-1]]
    print(f"  Topic {t}: {', '.join(top_words)}")

print("\nNMF factorises X ≈ W × H where:")
print("  W[doc, topic] = document's weight for each topic")
print("  H[topic, word] = topic's weight for each word")


# ── Data Loading ──────────────────────────────────────────────────────

loader = ASCENTDataLoader()
news = loader.load("ascent04", "sg_news_corpus.parquet")

print(f"\n=== Singapore News Corpus ===")
print(f"Shape: {news.shape}")
print(f"Columns: {news.columns}")
print(f"Date range: {news['published_date'].min()} to {news['published_date'].max()}")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Text preprocessing with Polars
# ══════════════════════════════════════════════════════════════════════

news_clean = (
    news.with_columns(
        (pl.col("title") + ". " + pl.col("body")).alias("text"),
        pl.col("published_date").str.to_date("%Y-%m-%d").alias("date"),
    )
    .filter(pl.col("body").str.len_chars() > 100)
    .with_columns(
        pl.col("date").dt.strftime("%Y-%m").alias("year_month"),
    )
)

documents = news_clean["text"].to_list()
dates = news_clean["date"].to_list()
print(f"\nCleaned corpus: {len(documents):,} articles")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: BERTopic model
# ══════════════════════════════════════════════════════════════════════

if BERTopic is not None:
    # BERTopic pipeline:
    # 1. Sentence embeddings (SBERT)  2. UMAP  3. HDBSCAN  4. c-TF-IDF
    # TODO: Build BERTopic with embedding_model="all-MiniLM-L6-v2",
    #       umap_model=None, hdbscan_model=None, min_topic_size=20,
    #       nr_topics="auto", verbose=True
    topic_model = ____

    # TODO: Fit-transform on documents → returns (topics, probs)
    topics, probs = ____

    topic_info = topic_model.get_topic_info()
    n_topics = len(topic_info) - 1  # Exclude outlier topic -1
    print(f"\n=== BERTopic Results ===")
    print(f"Topics found: {n_topics}")
    print(f"Outlier documents: {(np.array(topics) == -1).sum():,}")

    print(f"\nTop 10 Topics:")
    for _, row in topic_info.head(11).iterrows():
        if row["Topic"] == -1:
            continue
        print(f"  Topic {row['Topic']}: {row['Name'][:60]} (n={row['Count']})")

else:
    # Fallback: TF-IDF + NMF
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import NMF

    print("\nBERTopic not installed, using TF-IDF + NMF fallback")

    # TODO: TfidfVectorizer(max_features=5000, stop_words="english", max_df=0.95, min_df=5)
    vectorizer = ____
    # TODO: Fit-transform documents
    tfidf_matrix = ____
    feature_names = vectorizer.get_feature_names_out()

    n_topics = 15
    # TODO: NMF(n_components=n_topics, random_state=42) and fit_transform
    nmf = ____
    W = ____  # Document-topic matrix
    H = nmf.components_  # Topic-word matrix

    topics = W.argmax(axis=1).tolist()
    probs = W / (W.sum(axis=1, keepdims=True) + 1e-10)

    print(f"\nNMF Topics ({n_topics}):")
    for topic_idx in range(n_topics):
        top_words = [feature_names[i] for i in H[topic_idx].argsort()[-8:][::-1]]
        count = sum(1 for t in topics if t == topic_idx)
        print(f"  Topic {topic_idx}: {', '.join(top_words)} (n={count})")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Topic coherence evaluation (NPMI)
# ══════════════════════════════════════════════════════════════════════
# NPMI(w_i, w_j) = (log P(w_i,w_j)/(P(w_i)P(w_j))) / (-log P(w_i,w_j))
# Range: [-1, 1]. Higher = more coherent topic.


def compute_npmi(
    documents: list[str], topic_words: list[list[str]], window_size: int = 10
) -> list[float]:
    """Compute NPMI coherence for each topic."""
    word_doc_count = Counter()
    pair_doc_count = Counter()
    n_docs = len(documents)

    for doc in documents:
        words = set(doc.lower().split())
        # TODO: Update word_doc_count for every word in this document
        for w in words:
            word_doc_count[____] += 1
        word_list = list(words)
        for i in range(len(word_list)):
            for j in range(i + 1, len(word_list)):
                pair = tuple(sorted([word_list[i], word_list[j]]))
                pair_doc_count[pair] += 1

    coherences = []
    for topic in topic_words:
        npmi_sum = 0
        n_pairs = 0
        for i in range(len(topic)):
            for j in range(i + 1, len(topic)):
                w_i, w_j = topic[i].lower(), topic[j].lower()
                pair = tuple(sorted([w_i, w_j]))
                # TODO: Marginal probabilities for w_i and w_j; joint for the pair
                p_i = ____  # Hint: word_doc_count.get(w_i, 0) / n_docs
                p_j = ____  # Hint: word_doc_count.get(w_j, 0) / n_docs
                p_ij = ____  # Hint: pair_doc_count.get(pair, 0) / n_docs

                if p_ij > 0 and p_i > 0 and p_j > 0:
                    # TODO: PMI = log(p_ij / (p_i * p_j))
                    pmi = ____
                    # TODO: Normalise PMI to NPMI by dividing by -log(p_ij)
                    npmi = ____
                    npmi_sum += npmi
                    n_pairs += 1

        coherences.append(npmi_sum / max(n_pairs, 1))
    return coherences


# Get topic words
if BERTopic is not None:
    topic_words = []
    for topic_id in range(n_topics):
        words = [w for w, _ in topic_model.get_topic(topic_id)[:10]]
        topic_words.append(words)
else:
    topic_words = []
    for topic_idx in range(n_topics):
        words = [feature_names[i] for i in H[topic_idx].argsort()[-10:][::-1]]
        topic_words.append(words)

coherences = compute_npmi(documents[:5000], topic_words)

print(f"\n=== Topic Coherence (NPMI) ===")
print(f"Mean NPMI: {np.mean(coherences):.4f}")
print(f"Best topic: {np.argmax(coherences)} (NPMI={max(coherences):.4f})")
print(f"Worst topic: {np.argmin(coherences)} (NPMI={min(coherences):.4f})")
for i, c in enumerate(coherences[:10]):
    bar = "█" * max(0, int((c + 0.5) * 20))
    print(f"  Topic {i}: {c:+.4f} {bar}")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Temporal topic evolution
# ══════════════════════════════════════════════════════════════════════

news_with_topics = news_clean.with_columns(
    pl.Series("topic", topics[: news_clean.height])
)

# TODO: Group by year_month + topic, count, sort
# Hint: news_with_topics.filter(pl.col("topic") >= 0)
#         .group_by("year_month", "topic")
#         .agg(pl.col("topic").count().alias("count"))
#         .sort("year_month", "topic")
temporal = ____

monthly_totals = temporal.group_by("year_month").agg(
    pl.col("count").sum().alias("total")
)
temporal = temporal.join(monthly_totals, on="year_month").with_columns(
    (pl.col("count") / pl.col("total")).alias("proportion")
)

print(f"\n=== Temporal Evolution ===")
print(f"Months covered: {temporal['year_month'].n_unique()}")

months = sorted(temporal["year_month"].unique().to_list())
if len(months) >= 6:
    early = months[:3]
    late = months[-3:]

    print("\nTrending topics (last 3 months vs first 3 months):")
    for topic_id in range(min(n_topics, 10)):
        early_prop = temporal.filter(
            (pl.col("year_month").is_in(early)) & (pl.col("topic") == topic_id)
        )["proportion"].mean()
        late_prop = temporal.filter(
            (pl.col("year_month").is_in(late)) & (pl.col("topic") == topic_id)
        )["proportion"].mean()

        if early_prop is not None and late_prop is not None:
            change = (late_prop - early_prop) * 100
            arrow = "↑" if change > 1 else "↓" if change < -1 else "→"
            print(f"  Topic {topic_id}: {change:+.1f}pp {arrow}")


# ══════════════════════════════════════════════════════════════════════
# Visualise
# ══════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()

coherence_data = {f"Topic_{i}": {"NPMI": c} for i, c in enumerate(coherences[:10])}
fig = viz.metric_comparison(coherence_data)
fig.update_layout(title="Topic Coherence (NPMI)")
fig.write_html("ex3_topic_coherence.html")
print("\nSaved: ex3_topic_coherence.html")

topic_counts = Counter(t for t in topics if t >= 0)
size_data = {"Topic Size": [topic_counts.get(i, 0) for i in range(min(n_topics, 15))]}
fig_size = viz.training_history(size_data, x_label="Topic ID")
fig_size.update_layout(title="Topic Size Distribution")
fig_size.write_html("ex3_topic_sizes.html")
print("Saved: ex3_topic_sizes.html")

print("\n✓ Exercise 5 complete — topic modeling with BERTopic / NMF")
