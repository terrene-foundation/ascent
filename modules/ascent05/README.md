# Module 5: LLMs, AI Agents & RAG Systems

**Kailash**: Kaizen (Delegate, BaseAgent, Signature), kailash-ml (6 ML agents) | **Scaffolding**: 30%

## Lecture (3h)

- **5A** Transformer Architecture & LLMs: encoder-decoder, self/cross-attention, positional encodings (RoPE, ALiBi), tokenization (BPE, WordPiece), scaling laws, inference optimization (KV-cache, speculative decoding, quantization)
- **5B** RAG Architecture: chunking strategies, embedding models, hybrid retrieval, re-ranking, evaluation (faithfulness, relevance, correctness), graph RAG, agentic RAG
- **5C** Agent Architecture: perception-reasoning-action loop, Signature-based contracts (InputField/OutputField), CoT/ReAct/RAG agents, multi-agent A2A, ML agents, agent safety (prompt injection, cost budgets)

## Lab (3h) — 8 Exercises

1. LLM fundamentals: Delegate + SimpleQAAgent with Signature contracts
2. Chain-of-thought reasoning on clustering results from M4
3. ReAct agents with tools (DataExplorer, FeatureEngineer, ModelVisualizer)
4. RAG systems over SDK + Singapore regulatory docs with RAGAS evaluation
5. MCP servers and tool integration for agent-driven ML
6. ML Agent Pipeline: 6 specialized agents (DataScientist → ModelSelector)
7. Multi-agent orchestration: supervisor-worker, sequential, parallel
8. Production deployment with Nexus (API + CLI + MCP)

## Datasets

Same e-commerce + credit datasets (agents reason over familiar data), Kailash SDK docs, Singapore AI Verify corpus
