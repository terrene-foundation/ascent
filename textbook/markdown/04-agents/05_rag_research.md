# Chapter 5: RAG Research Agent

## Overview

The **RAGResearchAgent** combines vector search with LLM generation for retrieval-augmented research. It retrieves relevant documents from a vector store, augments the LLM prompt with those documents, and generates answers with source attribution. As an autonomous agent using `MultiCycleStrategy`, it can iteratively refine its search across multiple cycles. This chapter teaches you the RAG pattern, vector store management, and document-grounded generation.

## Prerequisites

- Python 3.10+ installed
- Kailash Kaizen installed (`pip install kailash-kaizen`)
- Completion of Chapters 1-3 (Delegate, SimpleQA, ReAct)
- Understanding of `MultiCycleStrategy` and convergence detection (Chapter 3)

## Concepts

### Concept 1: The RAG Pattern

RAG (Retrieve-Augment-Generate) agents work in three phases: Retrieve relevant documents via semantic search, Augment the LLM context with those documents, and Generate an answer with source attribution. This grounds the LLM's responses in actual data rather than relying solely on parametric knowledge.

- **What**: A pattern that injects retrieved documents into the LLM context before generation
- **Why**: Grounding answers in source documents reduces hallucination and provides verifiable references
- **How**: The agent queries a vector store for semantically similar documents, injects the top-k results into the prompt, and the LLM synthesizes an answer citing sources
- **When**: Use RAG when answers must be based on a specific document corpus -- knowledge bases, documentation, research papers

### Concept 2: Vector Store

The vector store holds document embeddings for semantic search. Documents are added with an ID, title, and content. The store converts text to embeddings using a configurable model (default: `all-MiniLM-L6-v2`) and finds similar documents by cosine similarity.

- **What**: An in-memory store that maps document embeddings to their content for similarity search
- **Why**: Keyword search misses semantically related content -- vector search finds conceptually similar documents even with different wording
- **How**: `agent.add_document(doc_id, title, content)` to add, `agent.clear_documents()` to reset, `agent.get_document_count()` to inspect
- **When**: Before running the agent, populate the vector store with your domain-specific documents

### Concept 3: Iterative Research

Unlike single-shot RAG implementations, `RAGResearchAgent` uses `MultiCycleStrategy` for autonomous research. It can iteratively query, fetch, analyze, and refine -- searching for additional sources when initial results are insufficient. Convergence uses the same ADR-013 `tool_calls` pattern as ReAct.

- **What**: Multi-cycle execution where each cycle can refine the search query or fetch additional documents
- **Why**: Complex research questions often require multiple search passes with refined queries
- **How**: The agent's `tool_calls` field signals whether more research is needed (non-empty) or complete (empty)
- **When**: Automatically -- the agent decides when it has enough sources to synthesize a confident answer

### Concept 4: Source Attribution

RAG outputs include source references (`sources` list) and `relevant_excerpts` with similarity scores. This allows consumers to verify the answer against the original documents and assess retrieval quality via `retrieval_quality` (average similarity).

- **What**: Structured metadata linking each answer to the documents that informed it
- **Why**: Without attribution, RAG answers are indistinguishable from hallucination -- sources make verification possible
- **How**: The result dict includes `sources` (document IDs), `relevant_excerpts` (title, excerpt, similarity), and `retrieval_quality`
- **When**: Always -- source attribution is a core feature of every RAG response

### Key API

| Class / Method               | Parameters                                                        | Returns            | Description                                  |
| ---------------------------- | ----------------------------------------------------------------- | ------------------ | -------------------------------------------- |
| `RAGResearchAgent()`         | `llm_provider`, `model`, `top_k_documents`, ...                   | `RAGResearchAgent` | Create a RAG agent with vector store         |
| `agent.run()`                | `query: str`                                                      | `dict`             | Research a query with retrieval + generation |
| `agent.add_document()`       | `doc_id: str`, `title: str`, `content: str`                       | `None`             | Add a document to the vector store           |
| `agent.clear_documents()`    | --                                                                | `None`             | Remove all documents from the store          |
| `agent.get_document_count()` | --                                                                | `int`              | Number of documents in the store             |
| `agent._check_convergence()` | `result: dict`                                                    | `bool`             | Check if research is complete                |
| `RAGConfig()`                | `top_k_documents`, `similarity_threshold`, `embedding_model`, ... | `RAGConfig`        | Configuration for retrieval behavior         |
| `RAGSignature`               | --                                                                | `Signature`        | Retrieval-augmented I/O definition           |
| `SAMPLE_AI_DOCUMENTS`        | --                                                                | `list[dict]`       | Built-in sample documents for testing        |

## Code Walkthrough

### Imports and Setup

```python
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

from kaizen_agents.agents.specialized.rag_research import (
    RAGResearchAgent,
    RAGConfig,
    RAGSignature,
    SAMPLE_AI_DOCUMENTS,
)
from kaizen.core.base_agent import BaseAgent
from kaizen.strategies.multi_cycle import MultiCycleStrategy

model = os.environ.get("DEFAULT_LLM_MODEL", "claude-sonnet-4-20250514")
```

The RAG module exports the agent, config, signature, and a set of sample documents for testing. `MultiCycleStrategy` is imported to verify autonomous execution.

### RAGSignature -- Retrieval-Augmented Output

```python
assert "query" in RAGSignature._signature_inputs

assert "answer" in RAGSignature._signature_outputs
assert "sources" in RAGSignature._signature_outputs
assert "confidence" in RAGSignature._signature_outputs
assert "relevant_excerpts" in RAGSignature._signature_outputs
assert (
    "tool_calls" in RAGSignature._signature_outputs
), "ADR-013: tool_calls for convergence (web_search, fetch_url, etc.)"
```

The RAG signature takes a `query` input and produces five outputs: the synthesized `answer`, `sources` (document IDs), `confidence`, `relevant_excerpts` (with similarity scores), and `tool_calls` for convergence detection.

### RAGConfig -- Retrieval Settings

```python
config = RAGConfig()

assert config.top_k_documents == 3, "Retrieve top 3 documents by default"
assert config.similarity_threshold == 0.3, "Minimum similarity score"
assert config.embedding_model == "all-MiniLM-L6-v2", "Default embedding model"
assert config.max_cycles == 15, "Research may need many cycles"
assert config.mcp_enabled is True, "Autonomous agents discover MCP tools"

# Custom retrieval configuration
custom_config = RAGConfig(
    top_k_documents=5,
    similarity_threshold=0.4,
    embedding_model="all-mpnet-base-v2",
    max_cycles=20,
)
assert custom_config.top_k_documents == 5
assert custom_config.similarity_threshold == 0.4
```

RAG-specific settings control the retrieval phase: `top_k_documents` limits how many documents are injected into the prompt, `similarity_threshold` filters out low-relevance results, and `embedding_model` selects the encoder. The default `max_cycles=15` is higher than ReAct's 10 because research tasks often need more iterations.

### Sample Documents

```python
assert len(SAMPLE_AI_DOCUMENTS) == 5

doc_ids = {doc["id"] for doc in SAMPLE_AI_DOCUMENTS}
assert "doc1" in doc_ids  # Machine Learning
assert "doc2" in doc_ids  # Deep Learning
assert "doc3" in doc_ids  # NLP
assert "doc4" in doc_ids  # Computer Vision
assert "doc5" in doc_ids  # Reinforcement Learning

# Every document has id, title, content
for doc in SAMPLE_AI_DOCUMENTS:
    assert "id" in doc
    assert "title" in doc
    assert "content" in doc
    assert len(doc["content"]) > 50, "Documents have substantial content"
```

Five sample AI documents ship with the agent for testing. Each has an `id`, `title`, and `content` with substantial text. In production, you replace these with your own domain-specific corpus.

### Agent Instantiation

```python
agent = RAGResearchAgent(
    llm_provider="mock",
    model=model,
)

assert isinstance(agent, RAGResearchAgent)
assert isinstance(agent, BaseAgent)
assert isinstance(
    agent.strategy, MultiCycleStrategy
), "RAG MUST use MultiCycleStrategy for autonomous research"
```

RAGResearchAgent uses `MultiCycleStrategy` -- it is autonomous, not interactive. The agent initializes a vector store with the sample documents automatically.

### Vector Store Management

```python
assert agent.get_document_count() == 5, "Sample docs loaded by default"

# Add a custom document
agent.add_document(
    doc_id="doc6",
    title="Quantum Computing",
    content="Quantum computing uses quantum mechanics principles such as "
    "superposition and entanglement to perform computations that would be "
    "intractable for classical computers.",
)
assert agent.get_document_count() == 6

# Clear all documents
agent.clear_documents()
assert agent.get_document_count() == 0
```

The vector store supports three operations: `add_document()` to insert, `get_document_count()` to inspect, and `clear_documents()` to reset. Documents are embedded on insertion and immediately available for search.

### Convergence Detection for Research

```python
# Research in progress (needs more sources)
result_researching = {
    "tool_calls": [{"name": "web_search", "params": {"query": "deep learning"}}],
}
assert agent._check_convergence(result_researching) is False

# Research complete (no more tools needed)
result_done = {"tool_calls": [], "confidence": 0.9}
assert agent._check_convergence(result_done) is True

# Malformed tool_calls -> stop for safety
result_malformed = {"tool_calls": "not a list"}
assert agent._check_convergence(result_malformed) is True

# Subjective fallback: high confidence + comprehensive depth
result_subjective = {"confidence": 0.9, "research_depth": "comprehensive"}
assert agent._check_convergence(result_subjective) is True

# Default: converged (safe fallback)
assert agent._check_convergence({}) is True
```

RAG uses the same ADR-013 convergence as ReAct. An additional safety check handles malformed `tool_calls` (e.g., a string instead of a list) by defaulting to converged. The subjective fallback also considers `research_depth` for RAG-specific signals.

### Input Validation

```python
empty_result = agent.run(query="")
assert empty_result["error"] == "INVALID_INPUT"
assert empty_result["sources"] == []
assert empty_result["confidence"] == 0.0
```

Empty queries return immediately with an error, empty sources list, and zero confidence.

### Return Structure

```python
# RAGResearchAgent.run() returns:
# {
#     "answer": "Machine learning is a subset of AI...",
#     "sources": ["doc1", "doc2"],
#     "confidence": 0.85,
#     "relevant_excerpts": [
#         {
#             "title": "Introduction to Machine Learning",
#             "excerpt": "Machine learning is a subset of...",
#             "similarity": 0.87,
#         }
#     ],
#     "retrieval_quality": 0.82,  # average similarity
# }
```

The return dict includes the synthesized answer, source document IDs, confidence, detailed excerpts with similarity scores, and an aggregate `retrieval_quality` metric.

### Custom Vector Store

```python
# For production use, provide your own vector store:
#
# from kaizen.retrieval.vector_store import SimpleVectorStore
#
# store = SimpleVectorStore(embedding_model="all-mpnet-base-v2")
# store.add_documents([
#     {"id": "d1", "title": "My Doc", "content": "..."},
# ])
#
# agent = RAGResearchAgent(
#     model=model,
#     vector_store=store,
#     top_k_documents=5,
# )
```

For production, create a `SimpleVectorStore` with your chosen embedding model, populate it with domain documents, and pass it to the agent constructor.

## Common Mistakes

| Mistake                                      | Correct Pattern                         | Why                                                                                            |
| -------------------------------------------- | --------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Using default sample documents in production | Populate with domain-specific documents | Sample documents are for testing only -- they will produce irrelevant answers for real queries |
| Setting `similarity_threshold` too high      | Start with `0.3` (default)              | High thresholds filter out relevant documents that use different vocabulary                    |
| Setting `top_k_documents` too high           | Start with `3-5` documents              | Too many documents overflow the LLM context window and dilute relevance                        |
| Ignoring `retrieval_quality` in the result   | Monitor average similarity scores       | Low `retrieval_quality` signals that the corpus may not cover the query topic                  |

## Exercises

1. Create a `RAGResearchAgent`, clear its default documents, and add three custom documents about a topic of your choice. Verify that `get_document_count()` returns 3 after your additions.
2. Construct three convergence test cases for `_check_convergence()`: one with active `tool_calls` (web_search), one with empty `tool_calls`, and one with malformed `tool_calls` (a string). Verify each returns the expected boolean.
3. Compare `RAGConfig` defaults with `ReActConfig` defaults. Why does RAG have a higher `max_cycles` (15 vs 10)?

## Key Takeaways

- RAG combines vector search with LLM generation for document-grounded answers
- `RAGResearchAgent` uses `MultiCycleStrategy` for autonomous iterative research
- The vector store holds document embeddings for semantic similarity search
- Source attribution (`sources`, `relevant_excerpts`, `retrieval_quality`) makes answers verifiable
- Convergence detection uses ADR-013 `tool_calls` with safety handling for malformed data
- Sample documents are for testing -- replace with domain-specific content for production

## Next Chapter

[Chapter 6: Streaming Chat Agent](06_streaming_chat.md) -- Build a real-time chat agent that streams tokens as they are generated, with configurable chunk sizes and dual-mode operation.
