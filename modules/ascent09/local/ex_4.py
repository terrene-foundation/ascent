# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT09 — Exercise 4: Advanced RAG and Evaluation
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Implement hybrid search (BM25 + vector), re-ranking, and
#   evaluate RAG quality with RAGAS-style metrics.
#
# TASKS:
#   1. Implement BM25 retrieval
#   2. Combine BM25 + vector search (hybrid)
#   3. Implement cross-encoder re-ranking
#   4. Evaluate with faithfulness, relevance, answer correctness
#   5. Compare basic vs hybrid vs re-ranked RAG
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import math
import os
import re
from collections import Counter

import polars as pl

from kaizen_agents import Delegate
from kaizen import Signature, InputField, OutputField
from kaizen_agents.agents.specialized.simple_qa import SimpleQAAgent

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
regulations = loader.load("ascent09", "sg_regulations.parquet")
texts = regulations.select("text").to_series().to_list()


# Reuse chunk_text from Ex 3 pattern — body provided so you can run immediately
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if end < len(text):
            bp = max(chunk.rfind("."), chunk.rfind("\n"))
            if bp > chunk_size // 2:
                chunk = text[start : start + bp + 1]
                end = start + bp + 1
        chunks.append(chunk.strip())
        start = end - overlap
    return [c for c in chunks if c]


all_chunks = []
for i, text in enumerate(texts):
    for j, chunk in enumerate(chunk_text(text)):
        all_chunks.append({"doc_idx": i, "chunk_idx": j, "text": chunk})
chunk_texts = [c["text"] for c in all_chunks]
print(f"Total chunks: {len(all_chunks)}")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Implement BM25 retrieval
# ══════════════════════════════════════════════════════════════════════


def tokenize_simple(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25:
    """BM25 over a list of documents.
    __init__: tokenize docs, compute avg_dl, build df Counter.
    _idf(term): log((N - df + 0.5) / (df + 0.5) + 1.0)
    score(query): per-doc BM25 sum: idf * tf*(k1+1)/(tf + k1*(1-b+b*dl/avg_dl))
    search(query, top_k): sorted top_k dicts with 'idx','score','text'.
    """

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        # TODO: Store k1/b; tokenize all docs; compute avg_dl; build self.df Counter
        # where df[term] = number of documents containing that term.
        ____
        ____

    def _idf(self, term: str) -> float:
        # TODO: log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
        ____

    def score(self, query: str) -> list[float]:
        # TODO: For each document compute the BM25 score summed over query terms.
        ____
        ____

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        # TODO: score(), sort descending, return top_k dicts {idx, score, text}.
        ____
        ____


bm25 = BM25(chunk_texts)
test_query = "capital adequacy requirements for banks"
bm25_results = bm25.search(test_query, top_k=5)
print(f"\n=== BM25: top result: {bm25_results[0]['text'][:100]}... ===")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Combine BM25 + vector search (hybrid)
# ══════════════════════════════════════════════════════════════════════


def cosine_similarity(a: list[float], b: list[float]) -> float:
    # TODO: dot / (|a|*|b|); 0.0 if either norm is zero.
    ____


async def get_embedding(text: str, delegate: Delegate) -> list[float]:
    # TODO: Prompt Delegate for 8 comma-separated floats in [-1,1] representing
    # [finance, legal, tech, compliance, sentiment, formality, specificity, complexity].
    # Parse; pad to 8; return. (Same pattern as Ex 3 generate_embedding.)
    ____
    ____


def normalize_scores(scores: list[float]) -> list[float]:
    # TODO: Min-max normalize to [0,1]; return [0.5]*n if range is zero.
    ____


async def hybrid_search(query: str, top_k: int = 5, alpha: float = 0.5) -> list[dict]:
    """Fuse BM25 + vector: embed only the top-20 BM25 candidates (cost efficiency).
    Normalize both score lists. Fuse: alpha*bm25_norm + (1-alpha)*vec_norm.
    Each result dict: 'idx','score','text','bm25','vector'.
    """
    # TODO: BM25 scores → normalize; embed top-20 BM25 indices → vector scores →
    # normalize; fuse; sort; return top_k dicts.
    ____
    ____
    ____
    ____


hybrid_results = asyncio.run(hybrid_search(test_query, top_k=5, alpha=0.6))
print(f"\n=== Hybrid top result: hybrid={hybrid_results[0]['score']:.3f} ===")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Implement cross-encoder re-ranking
# ══════════════════════════════════════════════════════════════════════

# TODO: Define RelevanceScore(Signature):
#   InputFields: query (str), passage (str)
#   OutputFields: relevance_score (float 0-1), reasoning (str)
____


async def rerank(query: str, candidates: list[dict], top_k: int = 3) -> list[dict]:
    # TODO: Create SimpleQAAgent(signature=RelevanceScore, model=model, budget_usd=1.0).
    # Score each candidate (passage=text[:500]), attach 'rerank_score'/'rerank_reason',
    # sort descending, return top_k.
    ____
    ____


reranked = asyncio.run(rerank(test_query, hybrid_results, top_k=3))
print(
    f"\n=== Re-ranked top: {reranked[0]['rerank_score']:.3f} — {reranked[0]['text'][:80]}... ==="
)


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Evaluate with faithfulness, relevance, answer correctness
# ══════════════════════════════════════════════════════════════════════

# TODO: Define RAGEvaluation(Signature):
#   InputFields: question (str), context (str), answer (str)
#   OutputFields: faithfulness (float 0-1), relevance (float 0-1),
#                 completeness (float 0-1), evaluation_notes (str)
____


async def evaluate_rag(question: str, context: str, answer: str) -> dict:
    # TODO: Create SimpleQAAgent(signature=RAGEvaluation, model=model, budget_usd=0.5),
    # run it, return dict with faithfulness/relevance/completeness/notes.
    ____
    ____


async def rag_answer(query: str, results: list[dict]) -> tuple[str, str]:
    delegate = Delegate(model=model, budget_usd=0.5)
    context = "\n\n---\n\n".join(r["text"] for r in results)
    prompt = f"Answer using ONLY the provided context.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    response = ""
    async for event in delegate.run(prompt):
        if hasattr(event, "text"):
            response += event.text
    return response.strip(), context


answer, context = asyncio.run(rag_answer(test_query, reranked))
eval_result = asyncio.run(evaluate_rag(test_query, context, answer))
print(f"\n=== RAG Evaluation ===")
print(
    f"  Faithfulness: {eval_result['faithfulness']:.2f}  Relevance: {eval_result['relevance']:.2f}  Completeness: {eval_result['completeness']:.2f}"
)
print(f"  Notes: {eval_result['notes']}")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Compare basic vs hybrid vs re-ranked RAG
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Method Comparison ===")
print(
    pl.DataFrame(
        {
            "method": ["BM25-only", "Hybrid (BM25+Vector)", "Hybrid + Re-ranked"],
            "retrieval_quality": [
                "Keyword match only",
                "Semantic + keyword",
                "LLM-judged relevance",
            ],
            "latency": [
                "Fast (no LLM)",
                "Medium (embeddings)",
                "Slow (per-candidate LLM)",
            ],
            "best_for": [
                "Exact term queries",
                "General questions",
                "High-stakes answers",
            ],
        }
    )
)
print(f"  Production: BM25 first-pass → vector expansion → re-rank top-N only")

print("\n✓ Exercise 4 complete — hybrid RAG with BM25, re-ranking, and evaluation")
