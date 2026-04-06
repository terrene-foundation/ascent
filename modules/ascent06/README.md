# Module 6: Alignment, Governance, RL & Production Deployment

**Kailash**: Align, PACT, kailash-ml (RLTrainer), Nexus | **Scaffolding**: 20%

## Lecture (3h)

- **6A** Fine-Tuning & Alignment: LoRA/QLoRA/DoRA theory, RLHF pipeline, DPO/GRPO, alignment data quality, evaluation (perplexity, BERTScore, LLM-as-judge), catastrophic forgetting trade-offs
- **6B** AI Governance: EU AI Act (risk tiers, GPAI rules), Singapore AI Verify (ISAGO 2.0), MAS guidelines, PACT D/T/R grammar, operating envelopes, bias/fairness metrics, algorithmic auditing, model cards
- **6C** RL & Advanced Topics: MDPs, Bellman equations, DQN, PPO, SAC, practical RL (dynamic pricing, recommendations), multi-modal AI, federated learning, differential privacy

## Lab (3h) — 8 Exercises

1. SFT fine-tuning with AlignmentPipeline and AdapterRegistry
2. DPO / QLoRA alignment with LLM-as-judge evaluation
3. RL fundamentals: PPO on inventory management environment
4. Model merging and evaluation: SLERP, TIES, adapter export
5. AI governance with PACT: define org in YAML, compile, verify access
6. Governed agents: wrap ReActAgent with PACT enforcement
7. Agent governance at scale: budget cascades, tool restrictions, audit
8. Capstone: full governed ML platform (all packages integrated)

## Datasets

Domain Q&A (SFT, 1000 pairs), Preference pairs (DPO, 500), Singapore Hansard, Gymnasium environments
