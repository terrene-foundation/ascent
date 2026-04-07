# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT09 — Exercise 3: RAG Fundamentals
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Build a RAG pipeline from scratch — document chunking,
#   embedding generation, vector similarity search — over Singapore
#   regulatory documents.
#
# TASKS:
#   1. Chunk documents with overlap strategy
#   2. Generate embeddings via Delegate
#   3. Build simple vector store (cosine similarity)
#   4. Implement retrieval with top-k
#   5. Generate answers with retrieved context via Delegate
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import math
import os

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
regulations = loader.load("ascent09", "sg_regulations.parquet")
print(f"Loaded {regulations.height:,} regulation sections")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Chunk documents with overlap strategy
# ══════════════════════════════════════════════════════════════════════


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Sliding-window chunker with sentence-boundary preference.
    Return [text] if it fits. Otherwise: advance start by (end - overlap) each step,
    prefer breaking at last '.' or '\\n' in the window when > chunk_size//2.
    Return only non-empty stripped chunks.
    """
    # TODO: Implement the full sliding-window chunker described above.
    ____
    ____


texts = regulations.select("text").to_series().to_list()
sections = (
    regulations.select("section").to_series().to_list()
    if "section" in regulations.columns
    else ["unknown"] * len(texts)
)

all_chunks = []
for i, (text, section) in enumerate(zip(texts, sections)):
    for j, chunk in enumerate(chunk_text(text, chunk_size=500, overlap=100)):
        all_chunks.append(
            {"doc_idx": i, "chunk_idx": j, "section": section, "text": chunk}
        )

print(f"\n=== Chunking: {len(all_chunks)} chunks from {len(texts)} docs ===")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Generate embeddings via Delegate
# ══════════════════════════════════════════════════════════════════════


async def generate_embedding(text: str, delegate: Delegate) -> list[float]:
    """8-dim pseudo-embedding: ask Delegate for comma-separated floats in [-1,1].
    Dimensions: [finance, legal, tech, compliance, sentiment, formality, specificity, complexity].
    Parse response; pad with 0.0 to length 8 on error.
    """
    # TODO: Build prompt for text[:300], stream Delegate response, parse floats,
    # pad to length 8, return the list.
    ____
    ____


async def embed_chunks(chunk_texts: list[str]) -> list[list[float]]:
    # TODO: Create Delegate(model=model, budget_usd=2.0), call generate_embedding
    # for each text, collect and return the list of embeddings.
    ____
    ____


chunk_subset = [c["text"] for c in all_chunks[:20]]
embeddings = asyncio.run(embed_chunks(chunk_subset))
print(f"\n=== Embeddings: {len(embeddings)} × dim {len(embeddings[0])} ===")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Build simple vector store (cosine similarity)
# ══════════════════════════════════════════════════════════════════════


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    # TODO: dot / (|a| * |b|); return 0.0 if either norm is zero.
    ____


class SimpleVectorStore:
    """Three parallel lists: documents, embeddings, metadata."""

    def __init__(self):
        self.documents: list[str] = []
        self.embeddings: list[list[float]] = []
        self.metadata: list[dict] = []

    def add(self, text: str, embedding: list[float], meta: dict | None = None):
        # TODO: Append to all three lists; default meta to {}.
        ____

    def search(self, query_embedding: list[float], top_k: int = 3) -> list[dict]:
        # TODO: Score all stored embeddings with cosine_similarity, sort descending,
        # return top_k dicts with keys 'text', 'score', 'metadata'.
        ____
        ____


store = SimpleVectorStore()
for i, (text, emb) in enumerate(zip(chunk_subset, embeddings)):
    store.add(text, emb, {"chunk_idx": i, "section": all_chunks[i].get("section", "")})
print(f"Vector store: {len(store.documents)} documents indexed")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Implement retrieval with top-k
# ══════════════════════════════════════════════════════════════════════


async def retrieve(query: str, top_k: int = 3) -> list[dict]:
    # TODO: Create Delegate(model=model, budget_usd=0.5), embed the query with
    # generate_embedding, call store.search, return results.
    ____
    ____


test_query = (
    "What are the compliance requirements for financial institutions in Singapore?"
)
retrieved = asyncio.run(retrieve(test_query, top_k=3))
print(f"\n=== Retrieval (top-3) for: {test_query[:60]}... ===")
for i, r in enumerate(retrieved):
    print(f"  [{i+1}] score={r['score']:.3f}: {r['text'][:120]}...")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Generate answers with retrieved context via Delegate
# ══════════════════════════════════════════════════════════════════════


async def rag_answer(query: str) -> str:
    """Full RAG: embed query → retrieve top-3 → join context → generate answer.
    Join chunks with '\\n\\n---\\n\\n'. Instruct Delegate to answer using ONLY
    the provided context; if insufficient, say so.
    """
    # TODO: Create Delegate(budget_usd=1.0). Embed query, retrieve top-3,
    # build context string, craft grounded-answer prompt, stream response.
    ____
    ____


print(f"\n=== RAG Q&A ===")
for query in [
    "What are the compliance requirements for financial institutions in Singapore?",
    "What penalties apply for regulatory violations?",
    "How should companies handle data protection under Singapore law?",
]:
    answer = asyncio.run(rag_answer(query))
    print(f"\nQ: {query}\nA: {answer[:250]}...")

print("\n✓ Exercise 3 complete — RAG pipeline with chunking, embeddings, and retrieval")
