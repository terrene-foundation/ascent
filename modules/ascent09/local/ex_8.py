# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT09 — Exercise 8: Capstone — Agent Deployment via Nexus
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Deploy a complete agent system — RAG + ReActAgent + MCP
#   tools — as a multi-channel service via Nexus (API + CLI + MCP).
#
# TASKS:
#   1. Build RAG-enhanced agent
#   2. Configure Nexus for multi-channel
#   3. Deploy as API endpoint
#   4. Deploy as CLI interface
#   5. Test across channels with session persistence
#   6. Measure latency and cost per query
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import math
import os
import re
import time
from collections import Counter

import polars as pl

from kaizen_agents import Delegate
from kaizen_agents.agents.specialized.simple_qa import SimpleQAAgent
from nexus import Nexus

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()

if not os.environ.get("OPENAI_API_KEY"):
    print("\u26a0 OPENAI_API_KEY not set \u2014 skipping LLM exercises.")
    print("  Set it in .env to run this exercise with real LLM calls.")
    import sys

    sys.exit(0)

model = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))

loader = ASCENTDataLoader()
regulations = loader.load("ascent09", "sg_regulations.parquet")
print(f"=== Singapore Regulations: {regulations.height} documents ===")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Build RAG-enhanced agent
# ══════════════════════════════════════════════════════════════════════

# TODO: Build a TF-IDF retriever over the regulations corpus + a SimpleQAAgent generator.
#
# Retriever pipeline (implement all steps):
#   1. cosine_similarity(a, b) → float  [dot / (|a|*|b| + 1e-10)]
#   2. Tokenise all docs: re.sub(r"[^a-z0-9\\s]", " ", doc.lower()).split()
#   3. Build vocab from top-2000 terms with freq > 1; token_idx mapping
#   4. Compute IDF per term: math.log(n_docs / (1 + vocab_freq[t]))
#   5. Build TF-IDF vector for each doc (tf = count/total; tfidf = tf * idf)
#   6. def retrieve(query, top_k=3): build query TF-IDF vector, cosine-score all
#      doc vectors, sort descending, return top_k doc texts[:500]
#
# Then: rag_agent = SimpleQAAgent(model=model)
____
____
____
____
____
____

rag_agent = SimpleQAAgent(model=model)
print(f"\n=== RAG Agent: TF-IDF retriever + SimpleQAAgent generator ===")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Configure Nexus for multi-channel
# ══════════════════════════════════════════════════════════════════════


async def handle_query(query: str) -> dict:
    """RAG handler: retrieve top-3 context chunks → generate grounded answer.
    Instruct rag_agent to answer ONLY from context; say so if insufficient.
    Return dict: answer (str, max 500 chars), sources (list), confidence (float).
    """
    # TODO: Call retrieve(query, top_k=3), join with '\\n\\n---\\n\\n', run
    # rag_agent with a grounded-answer prompt, return the result dict.
    ____
    ____


# In production: app = Nexus(); app.register("regulatory-rag", workflow); app.start()
print(
    f"\n=== Nexus: same handler → API + CLI + MCP (zero code changes per channel) ==="
)


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Deploy as API endpoint
# ══════════════════════════════════════════════════════════════════════


async def test_api_channel():
    # TODO: Iterate the three queries below. For each: time.time() before/after
    # handle_query, print Q/A[:200]/sources/confidence/latency.
    print(f"\n=== API Channel Test ===")
    queries = [
        "What are Singapore's AI governance requirements for financial services?",
        "How does the EU AI Act classify high-risk AI systems?",
        "What is the timeline for AI Act compliance?",
    ]
    ____
    ____


session = asyncio.run(test_api_channel())


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Deploy as CLI interface
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== CLI Channel ===")
print(f"  $ kailash nexus cli --app regulatory-rag")
print(f"  Same handler, same session state — only the transport differs.")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Session persistence across channels
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Session Persistence ===")
print(
    f"  Nexus sessions persist across channels — API follow-up remembers CLI context."
)
print(f"  Critical for conversational RAG where context accumulates across turns.")


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Measure latency and cost per query
# ══════════════════════════════════════════════════════════════════════


async def benchmark():
    # TODO: Time handle_query for each of the 5 queries below.
    # Compute avg_latency and p95_latency = sorted(latencies)[int(0.95*n)].
    # Print: query count, avg latency, P95, cost estimate.
    print(f"\n=== Performance Benchmark ===")
    queries = [
        "What is MAS TRM?",
        "Explain AI Verify framework",
        "Singapore data protection for AI",
        "Responsible AI principles",
        "AI governance best practices",
    ]
    ____
    ____


latencies = asyncio.run(benchmark())

print(f"\n=== Capstone Summary ===")
print(
    f"  RAG: TF-IDF retriever + SimpleQAAgent | Nexus: API+CLI+MCP | Sessions: persistent"
)
print(f"  This is the Kailash agent lifecycle — from documents to production.")

print("\n✓ Exercise 8 complete — RAG agent deployed via Nexus multi-channel")
