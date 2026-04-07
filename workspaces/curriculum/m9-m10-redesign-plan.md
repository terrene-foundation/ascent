# M9 / M10 Exercise Redesign Plan

**Status**: Plan only. No exercise files have been modified.
**Date**: 2026-04-07
**Purpose**: Eliminate the ~70% overlap between (M5/M9) and (M6/M10) by replacing
current M9 and M10 exercises with content that goes deeper than the foundation modules
while building directly on what students already know.

---

## Context: What Students Have When They Reach M9/M10

By the start of the Advanced Certificate (M7-M10), students have completed:

- **M5**: Delegate, SimpleQAAgent, Signature, ReActAgent with tools, RAG from scratch,
  MCP servers, multi-agent Pipeline.router(), Nexus deployment. Full Kaizen lifecycle.
- **M6**: SFT with AlignmentPipeline, DPO alignment, RL (Q-learning + RLTrainer DQN/PPO),
  LoRA adapter merging (linear + SLERP), PACT GovernanceEngine, PactGovernedAgent with
  tool restrictions and clearance levels, capstone governed ML system.
- **M7**: Deep learning internals — CNNs, RNNs, attention — via kailash-ml OnnxBridge and
  TrainingPipeline. ONNX export, InferenceServer deployment.
- **M8**: Transformer internals from scratch, BERT/GPT pre-training, fine-tuning with
  AutoMLEngine, zero-shot/few-shot NLP, sequence-to-sequence.

---

## The Core Problem

### M9 (current): ~70% overlap with M5

| Current M9 exercise                                          | M5 equivalent                             |
| ------------------------------------------------------------ | ----------------------------------------- |
| Ex 1: LLM architecture + first Delegate call                 | M5 Ex 1: LLM fundamentals + Delegate      |
| Ex 2: Zero-shot, few-shot, CoT prompting                     | M5 Ex 2: Chain-of-thought reasoning       |
| Ex 3: RAG from scratch (chunking, embeddings, cosine search) | M5 Ex 4: RAG systems                      |
| Ex 4: Hybrid RAG (BM25 + vector), re-ranking, RAGAS eval     | Extends M5 Ex 4 slightly                  |
| Ex 5: ReActAgent with custom tools                           | M5 Ex 3: ReAct agents with tools          |
| Ex 6: Multi-agent with Pipeline.router() + supervisor        | M5 Ex 7: Multi-agent orchestration        |
| Ex 7: MCP server + ReActAgent                                | M5 Ex 5: MCP servers and tool integration |
| Ex 8: Nexus multi-channel deployment capstone                | M5 Ex 8: Production deployment with Nexus |

Non-overlapping content: only the BM25 implementation in Ex 4 and RAGAS-style metrics
are genuinely new. Everything else repeats M5 with minor variations.

### M10 (current): ~60% overlap with M6

| Current M10 exercise                                                    | M6 equivalent                                |
| ----------------------------------------------------------------------- | -------------------------------------------- |
| Ex 1: LoRA SFT with AlignmentPipeline                                   | M6 Ex 1: SFT fine-tuning with Kailash Align  |
| Ex 2: DPO preference alignment                                          | M6 Ex 2: DPO / QLoRA alignment               |
| Ex 3: RL fundamentals (MDP, value iteration, Q-learning, RLTrainer DQN) | M6 Ex 3: Reinforcement learning              |
| Ex 4: PPO inventory management                                          | Extends M6 Ex 3 slightly                     |
| Ex 5: Model merging (linear + SLERP) + ONNX export                      | M6 Ex 4: LoRA adapter merging and evaluation |
| Ex 6: PACT GovernanceEngine + clearance-based access                    | M6 Ex 5: AI governance with PACT             |
| Ex 7: PactGovernedAgent with tool restrictions + audit                  | M6 Ex 6: Governed agents                     |
| Ex 8: Capstone governed ML system                                       | M6 Ex 8: Capstone — full platform            |

Non-overlapping content: the SLERP implementation is slightly more detailed, the
regulatory mapping (EU AI Act, MAS TRM) is more explicit, but the API surface covered
is identical to M6.

---

## Redesign Principles

1. **M9 assumes M5/M8 done**: students can write a Delegate call and a Signature
   in their sleep. M9 digs into the internals and limits of these tools.
2. **M10 assumes M6/M9 done**: students know LoRA, DPO, PPO, GovernanceEngine.
   M10 covers enterprise scale, regulatory compliance, and adversarial robustness.
3. **Theory over repetition**: Advanced Certificate modules should derive equations,
   not just call APIs. Students implement algorithms first, then validate against
   the kailash wrapper.
4. **Singapore/APAC context**: all new datasets and case studies remain regionally
   grounded (MAS TRM, PDPA, Singapore AI Verify, APAC enterprise scenarios).
5. **Scaffolding ~25% (M9) and ~20% (M10)**: consistent with the progressive
   disclosure table — only imports and high-level comments are provided.

---

## M9 Redesign: Fine-Tuning, Alignment & RL

**New title**: Fine-Tuning, Alignment & RL
**New frameworks**: Align: AlignmentPipeline, AdapterRegistry, LoRAManager;
kailash-ml: TrainingPipeline, OnnxBridge, InferenceServer

### New Exercise Map

---

#### Ex 1 (replaces current Ex 1 + Ex 2): LoRA Internals and Rank Selection

**Current**: LLM architecture survey + first Delegate call (M5 repeat).
**New objective**: Derive LoRA from first principles — low-rank decomposition,
effective rank, gradient flow — and implement rank selection heuristics.

Tasks:

1. Implement low-rank matrix decomposition from scratch (SVD-based). Verify
   that W ≈ B @ A when rank is chosen appropriately.
2. Measure reconstruction error as a function of rank. Plot error vs parameter
   count trade-off curve using ModelVisualizer.
3. Implement the LoRA forward pass: `h = W_0 @ x + (B @ A) @ x * (alpha/r)`.
   Verify output matches `AlignmentPipeline`'s LoRA layer numerically.
4. Use `AlignmentPipeline` with `LoRAConfig` at three ranks (4, 16, 64).
   Compare training loss curves and final task performance.
5. Implement the `LoRAManager.rank_selector()` heuristic (stable rank of
   the weight matrix). Compare against the hand-tuned rank from task 4.

Dataset: `ascent09/sg_company_reports.parquet` (text classification task)
Key insight students derive: LoRA is effective because pre-trained weight
matrices are inherently low-rank on downstream tasks. r=16 captures ~95%
of the useful gradient signal for most domain adaptation tasks.

---

#### Ex 2 (replaces current Ex 3 + Ex 4): DPO Derivation and Reward Modeling

**Current**: RAG from scratch + BM25 hybrid (M5 Ex 4 repeat + minor extension).
**New objective**: Derive DPO from RLHF, implement the DPO loss, train a
reward model, and compare DPO vs reward-model-based RLHF.

Tasks:

1. Implement the Bradley-Terry preference model. Given (prompt, chosen, rejected),
   compute the log-likelihood of the preference under the model.
   Verify against `preference_pairs.parquet`.
2. Derive the DPO loss from the RLHF objective. Show that DPO eliminates
   the separate reward model by expressing reward implicitly through log ratios.
   `L_DPO = -E[log σ(β (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))]`
3. Implement DPO loss numerically on a small tokenized example. Verify the
   gradient direction pushes chosen probability up and rejected probability down.
4. Run `AlignmentPipeline(method="dpo")` on `preference_pairs.parquet`. Compare
   the implicit reward `β log π(y|x)/π_ref(y|x)` at beta=0.01, 0.1, 1.0.
5. Evaluate: compare DPO-aligned model vs base model on a safety test set.
   Compute refusal rate and helpfulness score using a `SimpleQAAgent` judge.

Dataset: `ascent10/preference_pairs.parquet`

---

#### Ex 3 (replaces current Ex 5 + Ex 7): GRPO and Group Relative Policy Optimization

**Current**: ReActAgent with tools (M5 repeat) + MCP server (M5 repeat).
**New objective**: Understand GRPO — the algorithm behind DeepSeek-R1 and recent
reasoning models — and compare it to PPO on a verifiable reward task.

Tasks:

1. Implement the GRPO objective from scratch. Key insight: instead of a value
   network, GRPO uses within-group score normalization.
   `A_i = (r_i - mean(r_group)) / std(r_group)`.
2. Implement a verifiable reward function for Singapore regulatory Q&A: +1 for
   factually correct answers (verified against ground truth), 0 otherwise.
3. Implement the clipped GRPO update: same PPO clip as before, but advantages
   come from group normalization rather than GAE.
4. Compare GRPO vs PPO on the regulatory Q&A task. Measure sample efficiency
   (reward per 1000 steps) and stability (variance of reward across runs).
5. Use `AlignmentPipeline(method="grpo")` to run GRPO at scale. Inspect the
   reasoning traces — does GRPO produce longer, more structured reasoning chains?

Dataset: `ascent09/sg_regulations.parquet` + generated Q&A pairs

---

#### Ex 4 (replaces current Ex 6): Constitutional AI and Self-Critique

**Current**: Multi-agent Pipeline.router() + supervisor (M5 repeat).
**New objective**: Implement Constitutional AI (CAI) — the critique-revision loop
that makes models refuse harmful requests without human labels.

Tasks:

1. Define a constitution: a set of principles (harmlessness, honesty, helpfulness)
   as natural language rules. Use Singapore-specific context (MAS AML guidelines,
   PDPA data protection principles).
2. Implement the CAI critique step using `Delegate`: given a response and a
   principle, generate a critique identifying violations.
3. Implement the CAI revision step: given a critique, revise the response to
   comply with the principle. Chain multiple principles sequentially.
4. Build a critique-revision pipeline as a `Pipeline` with multiple stages.
   Run on 20 test prompts spanning safe and borderline content.
5. Evaluate: compare original vs CAI-revised responses on a safety rubric
   (implemented as a `Signature`). Measure principle violation rate before/after.

Dataset: generated prompts using `sg_company_reports.parquet` as source material
Key insight: CAI shows that safety can be bootstrapped from helpful models using
LLMs as judges, without human preference labels.

---

#### Ex 5 (replaces current Ex 8): Inference Optimization and Quantization

**Current**: Nexus deployment capstone (M5 Ex 8 repeat).
**New objective**: Understand and implement the key inference optimizations:
quantization, KV-cache management, speculative decoding, batching strategies.

Tasks:

1. Implement INT8 quantization from scratch on a small weight matrix.
   Compare absmax vs zero-point quantization. Measure perplexity degradation
   at INT8, INT4, and INT2 on `sg_company_reports.parquet`.
2. Use `OnnxBridge.export()` with FP32, FP16, and INT8 precision. Compare
   model size and inference latency across precisions.
3. Implement a KV cache manager that tracks token positions and evicts
   oldest entries when the cache exceeds a budget (sliding window attention).
   Measure speedup vs re-encoding the full context each turn.
4. Implement speculative decoding: use a small draft model to generate candidate
   tokens, then verify with the target model. Measure acceptance rate and speedup.
5. Use `InferenceServer` with throughput benchmarking: compare latency at
   batch_size=1, 4, 16, 32. Plot the throughput vs latency Pareto curve.

Dataset: `ascent09/sg_company_reports.parquet` (inference benchmark corpus)
Key insight: quantization is the single highest-leverage optimization — INT8
typically costs <1% perplexity loss while halving memory and 1.5-2x latency.

---

#### Ex 6 (new): Reward Hacking and Goodhart's Law

**Current exercise slot**: no direct M5 equivalent.
**Objective**: Study reward misspecification empirically — build environments
where optimizing a proxy reward diverges from the true objective.

Tasks:

1. Define a text quality metric as a proxy reward (e.g., response length,
   keyword density, sentiment score). Optimize a `Delegate`-based policy against it.
2. Show that the policy learns to game the metric: longer responses that are
   less factually accurate, or responses stuffed with positive keywords.
3. Implement KL-regularization to constrain how far the policy drifts from
   the reference model. Show the regularization prevents gaming.
4. Implement reward ensemble: train on two independent proxy rewards. Show
   that gaming is harder when both must be satisfied simultaneously.
5. Design a robust reward that is harder to game for the Singapore regulatory
   Q&A task. Compare: factual accuracy + conciseness + regulatory citation count.

Dataset: `ascent09/sg_regulations.parquet`
Key insight: Goodhart's Law — when a measure becomes a target, it ceases to be a
good measure. This is the root cause of alignment failures in deployed RL systems.

---

#### Ex 7 (new): Safety Evaluation and Red-Teaming

**Current**: MCP server (M5 repeat).
**New objective**: Systematically evaluate model safety using automated red-teaming,
jailbreak detection, and safety benchmarks.

Tasks:

1. Implement a jailbreak classifier using `Signature`: given a prompt, output
   whether it is a jailbreak attempt and which technique (role-play, prefix
   injection, hypothetical framing, base64 encoding).
2. Build an automated red-team agent using `ReActAgent` with a `Delegate`
   as the target model. The red-team agent attempts to elicit harmful outputs
   and classifies successes/failures.
3. Implement adversarial suffix generation (simplified GCG): use gradient-free
   search (random token substitution + beam search) to find suffixes that
   cause the model to comply with refused requests.
4. Run the red-team against a base model and a DPO-aligned model from Ex 2.
   Measure attack success rate (ASR) at each step. Quantify alignment's effect.
5. Use `AlignmentPipeline` to fine-tune on the failed red-team attempts (adversarial
   training). Re-evaluate ASR after adversarial fine-tuning.

Dataset: safety prompts dataset (to be added to `data/ascent09/`)
Key insight: Alignment is not binary. Every aligned model has an attack success
rate; the goal is to make ASR low enough for the deployment context.

---

#### Ex 8 (new): Capstone — Full Fine-Tuning and Alignment Pipeline

**Current**: Nexus multi-channel capstone (M5 repeat with governance layer).
**New objective**: Run a complete alignment pipeline: pre-training simulation →
SFT → reward modeling → DPO → GRPO → safety evaluation → inference-optimized
deployment. Integrate every M9 concept end-to-end.

Tasks:

1. Start with a base `TinyLlama` checkpoint. Run SFT on `sg_domain_qa.parquet`
   using `AlignmentPipeline(method="sft")` with `LoRAConfig(rank=16)`.
2. Train a reward model on `preference_pairs.parquet` using `AlignmentPipeline`
   reward head. Register in `AdapterRegistry`.
3. Run DPO on the SFT model using preference pairs. Compare implicit reward
   vs explicit reward model scores.
4. Run GRPO on top of DPO model using verifiable rewards from Singapore regulatory
   Q&A. Measure reasoning chain quality improvement.
5. Export final model to ONNX with INT8 quantization using `OnnxBridge`.
6. Run safety evaluation from Ex 7. Generate compliance attestation document
   covering: training data lineage, alignment method, safety evaluation results,
   attack success rate.

Dataset: `ascent10/sg_domain_qa.parquet`, `ascent10/preference_pairs.parquet`

---

## M10 Redesign: AI Governance, Safety & Enterprise

**New title**: AI Governance, Safety & Enterprise
**New frameworks**: PACT: GovernanceEngine, PactGovernedAgent, ClearanceManager;
Nexus; kailash-ml: DriftMonitor

### New Exercise Map

---

#### Ex 1 (replaces current Ex 1 + Ex 6): Multi-Organisation Governance

**Current**: LoRA SFT (M6 repeat) + PACT GovernanceEngine (M6 repeat).
**New objective**: Model cross-organisational governance — federated structures,
trust boundaries between organisations, delegation across org boundaries.

Tasks:

1. Define two separate PACT organisations: a regulator (MAS) and a regulated
   entity (a bank). Each has its own `GovernanceEngine`. Compile both.
2. Implement a KSP (Knowledge Sharing Protocol) bridge between the organisations.
   Define what the bank is permitted to share with MAS and vice versa.
3. Model a regulatory inspection: MAS inspector requests access to the bank's
   confidential model cards and audit logs. Verify cross-org access control.
4. Implement delegation chaining: bank CEO delegates to compliance officer who
   delegates to an AI agent. Verify the chain is auditable end-to-end.
5. Run a breach simulation: an agent attempts to access data above its clearance.
   Verify the `GovernanceEngine` blocks, logs, and alerts the delegating officer.

Dataset: `data/ascent10/sg_regulatory_orgs.yaml` (new dataset needed)
Key insight: Cross-org governance is qualitatively different from single-org
governance. Trust boundaries must be explicit and auditable at every crossing.

---

#### Ex 2 (replaces current Ex 2 + Ex 7): Federated Learning with Privacy Guarantees

**Current**: DPO alignment (M6 repeat) + PactGovernedAgent (M6 repeat).
**New objective**: Implement federated learning with differential privacy guarantees
for multi-organisation ML training without sharing raw data.

Tasks:

1. Implement FedAvg from scratch: three virtual clients, each with a local
   dataset partition. Aggregate local gradients into a global model using
   weighted average. Verify convergence matches centralized training.
2. Add Gaussian noise mechanism for differential privacy: clip gradients to
   sensitivity S, add N(0, σ²S²) noise. Implement privacy accounting
   (moments accountant) to track ε-DP budget across rounds.
3. Show the privacy-utility trade-off: run FedAvg at ε = 10, 1, 0.1.
   Plot accuracy vs ε. Find the knee of the curve.
4. Integrate with PACT: each federated client is a `PactGovernedAgent`.
   The aggregation server holds the `GovernanceEngine`. Verify that no client
   can inspect another's local updates.
5. Deploy the federated model via `InferenceServer`. Add drift monitoring with
   `DriftMonitor` to detect when any client's distribution shifts post-training.

Dataset: Partition `ascent02/icu_patients.parquet` into three virtual hospital
clients (no sharing — federated training only).
Key insight: Federated learning with DP is the standard approach for regulated
industries (healthcare, finance) where data residency rules prevent centralisation.

---

#### Ex 3 (replaces current Ex 3 + Ex 4): Privacy-Preserving ML and Data Minimisation

**Current**: RL fundamentals (M6 repeat) + PPO inventory management (M6 slight extension).
**New objective**: Apply privacy-preserving ML techniques: k-anonymity,
l-diversity, data masking, synthetic data generation, and membership inference defence.

Tasks:

1. Implement k-anonymity on `icu_patients.parquet`: generalise quasi-identifiers
   (age → age bracket, postcode → district) until every record is indistinguishable
   from at least k-1 others. Measure utility loss at k=5, 10, 50.
2. Implement a membership inference attack: train a shadow model on half the
   ICU data, then train an attack classifier to determine if a given record
   was in the training set. Report attack accuracy and AUC.
3. Defend against membership inference by adding L2 regularisation and early
   stopping. Re-run the attack and measure AUC reduction.
4. Generate synthetic data using a Gaussian copula model. Verify that the
   synthetic data preserves statistical properties (correlation matrix, marginal
   distributions) but achieves k-anonymity by construction.
5. Implement a PACT-governed inference pipeline: every inference request passes
   through `PactGovernedAgent` with a `ClearanceManager` that checks the requester
   is permitted to receive predictions about the queried individual.

Dataset: `ascent02/icu_patients.parquet`
Regulatory alignment: PDPA (Singapore), GDPR Art. 89 (research exemption), MAS
Notice 655 (data protection for financial institutions).

---

#### Ex 4 (replaces current Ex 5): Model Cards, Datasheets, and Compliance Artefacts

**Current**: Model merging with SLERP (M6 slight extension).
**New objective**: Produce the compliance artefacts required for regulated AI
deployment: model cards, datasheets for datasets, bias audits, and system cards.

Tasks:

1. Implement a `ModelCard` generator using `Signature`. Fields: model name,
   intended use, out-of-scope uses, performance metrics by demographic group,
   known limitations, training data description, evaluation data description.
2. Run a bias audit on a classification model trained on `credit_card_fraud.parquet`.
   Compute demographic parity, equalised odds, and calibration across segments.
   Flag any segment with metric gap > 5%.
3. Generate a `DatasetDatasheet` for `sg_company_reports.parquet`. Fields:
   collection method, consent status, licensing, known biases, recommended uses,
   prohibited uses.
4. Implement a `SystemCard` for a governed agent pipeline (from M10 Ex 1).
   Map every component to an EU AI Act risk category. Flag high-risk components
   that require conformity assessment.
5. Use `DriftMonitor` to generate a post-deployment monitoring report after
   simulated production traffic. Identify which features drifted, PSI scores,
   and recommended retraining triggers.

Dataset: `ascent04/credit_card_fraud.parquet`, `ascent09/sg_company_reports.parquet`
Regulatory alignment: EU AI Act Annex IV (technical documentation), ISO/IEC 42001.

---

#### Ex 5 (replaces current Ex 8 capstone — moves and upgrades): Adversarial Robustness and Attacks

**Current**: Capstone governed ML system (M6 capstone repeat).
**New objective**: Study adversarial attacks on ML models (evasion, poisoning,
model extraction) and implement defences using certified robustness.

Tasks:

1. Implement FGSM (Fast Gradient Sign Method) on a tabular fraud model:
   perturb feature values by ε in the gradient direction to flip the prediction.
   Measure attack success rate at ε = 0.01, 0.1, 0.5.
2. Implement PGD (Projected Gradient Descent) — iterative FGSM with projection
   back to the ε-ball. Compare ASR vs FGSM.
3. Implement adversarial training: augment training data with FGSM/PGD examples.
   Re-evaluate ASR after adversarial training.
4. Implement a data poisoning attack on the fraud model: inject 5% mislabelled
   examples to degrade precision on a specific demographic. Detect the poisoning
   using influence function approximation.
5. Implement certified robustness using randomised smoothing: for a given input,
   certify that the model's prediction is consistent within a radius R with
   probability 1-δ. Report certified accuracy at R=0.1, 0.5, 1.0.

Dataset: `ascent04/credit_card_fraud.parquet`
Key insight: Adversarial robustness certification is required for EU AI Act
high-risk AI systems (Annex III). Certified radius is a regulatory metric, not
just an academic result.

---

#### Ex 6 (new): Regulatory Compliance Automation with PACT

**Current exercise slot**: no direct M6 equivalent at this depth.
**Objective**: Build an automated regulatory compliance checker — given a deployed
model and its governance artefacts, produce a structured compliance report against
multiple regulatory frameworks.

Tasks:

1. Define a compliance schema using `Signature`: for each regulation (EU AI Act,
   MAS TRM, Singapore AI Verify, PDPA), define the required artefacts and
   assertion predicates.
2. Implement a `ComplianceAuditAgent` using `ReActAgent` that checks each
   regulatory requirement: reads model card, queries `DriftMonitor`, inspects
   `GovernanceEngine` audit trail, and generates pass/fail verdicts with evidence.
3. Run the audit on the governed pipeline from Ex 1. For each failed requirement,
   generate a remediation recommendation with estimated effort.
4. Implement continuous compliance monitoring: register a `DriftMonitor` alert
   that triggers the `ComplianceAuditAgent` when PSI exceeds 0.2 on any feature.
5. Deploy the compliance checker as a `Nexus` endpoint. Test via API: submit a
   model ID, receive a structured compliance report within 30 seconds.

Dataset: uses governance artefacts from Ex 1-5

---

#### Ex 7 (new): Enterprise ML Governance at Scale

**Current**: PactGovernedAgent with tools (M6 repeat at slightly more depth).
**New objective**: Design and implement governance for large-scale ML operations —
hundreds of models, multiple teams, automated policy enforcement.

Tasks:

1. Design a model governance registry using `DataFlow`: schema includes model_id,
   training_data_hash, adapter_history, clearance_requirements, drift_metrics,
   compliance_status. Implement CRUD via `db.express`.
2. Implement a governance policy engine: given a model metadata record, apply
   policies (e.g., "models trained on PII data must have clearance level SECRET
   or above") and produce a policy verdict with violation details.
3. Implement model retirement automation: when `DriftMonitor` reports PSI > 0.25
   on two consecutive checks, automatically downgrade the model's PACT clearance
   to RESTRICTED and notify the responsible engineer via a `Nexus` webhook.
4. Implement a multi-tenancy isolation layer: different PACT organisations share
   the same `InferenceServer` but cannot inspect each other's request logs or
   model weights. Verify isolation with cross-tenant access attempt.
5. Run a governance incident simulation: a model returns biased predictions for
   a protected attribute. Trace the incident from alert → audit → root cause →
   remediation using the PACT D/T/R accountability chain.

Dataset: `ascent04/credit_card_fraud.parquet` (simulated production traffic)

---

#### Ex 8 (new): Capstone — Regulated AI System End-to-End

**Current**: Governed ML system capstone (M6 capstone, lightly extended).
**New objective**: Deploy a complete regulated AI system that satisfies all
requirements from M10 Ex 1-7: federated training, privacy preservation, bias
audit, adversarial hardening, compliance reporting, and enterprise governance.

Tasks:

1. Train a fraud detection model using federated learning from Ex 2 (three
   virtual bank clients). Apply DP at ε=1.0 and verify privacy budget spent.
2. Generate model card and dataset datasheet from Ex 4. Run bias audit and
   confirm no demographic group exceeds the 5% metric gap threshold.
3. Apply adversarial training from Ex 5. Certify robustness at R=0.1 with
   probability 0.95 using randomised smoothing.
4. Register the model in the governance registry from Ex 7. Apply compliance
   policies. Generate structured compliance report from Ex 6.
5. Deploy via `Nexus` with `PactGovernedAgent` enforcing clearance levels per
   request. Configure `DriftMonitor` with automated governance alerts.
6. Run the `ComplianceAuditAgent` end-to-end. The final output is a
   machine-readable compliance attestation (JSON) and a human-readable audit
   report that maps every claim to its evidence artefact.

Expected outcome: students produce a complete audit-ready package for a regulated
AI model, suitable for submission to MAS or for EU AI Act conformity assessment.

---

## Dataset Additions Required

The following datasets must be created before M9/M10 exercises can be written:

| Dataset                                 | Format  | Size          | Purpose                   |
| --------------------------------------- | ------- | ------------- | ------------------------- |
| `data/ascent09/safety_prompts.parquet`  | parquet | 500-1000 rows | Ex 7 red-teaming          |
| `data/ascent10/sg_regulatory_orgs.yaml` | yaml    | ~5 KB         | Ex 1 multi-org governance |

All other datasets referenced (sg_domain_qa, preference_pairs, inventory_demand,
sg_company_reports, sg_regulations, icu_patients, credit_card_fraud) already exist.

---

## Implementation Sequence

Execute in this order to ensure dataset and exercise dependencies are satisfied:

1. Create `safety_prompts.parquet` (dataset-curator agent)
2. Create `sg_regulatory_orgs.yaml` (manually authored, ~20 role definitions)
3. Write M9 Ex 1-8 solutions in `modules/ascent09/solutions/`
4. Write M10 Ex 1-8 solutions in `modules/ascent10/solutions/`
5. Run exercise-designer agent to strip each solution to the 25%/20% scaffolding level
6. Convert all exercises to Jupyter + Colab via `python scripts/py_to_notebook.py --module ascent09`
7. Update `modules/ascent09/README.md` and `modules/ascent10/README.md` with new exercise map
8. Update quiz files in `modules/ascent09/quiz/` and `modules/ascent10/quiz/`

Estimated autonomous execution: 2-3 sessions (solution writing is the long pole;
stripping and conversion are automated).

---

## M9/M10 Framework Coverage After Redesign

| Framework        | M9 (current)         | M9 (new)                                                | M10 (current)        | M10 (new)                                                 |
| ---------------- | -------------------- | ------------------------------------------------------- | -------------------- | --------------------------------------------------------- |
| kailash-align    | none                 | Deep (LoRA math, DPO derivation, GRPO, CAI)             | Shallow (repeats M6) | Medium (reward modeling)                                  |
| kailash-pact     | none                 | none                                                    | Shallow (repeats M6) | Deep (multi-org, ClearanceManager, compliance automation) |
| kailash-kaizen   | Shallow (repeats M5) | Medium (red-team agent, CAI pipeline)                   | none                 | Medium (ComplianceAuditAgent)                             |
| kailash-ml       | none                 | Deep (inference optimization, quantization, OnnxBridge) | Shallow (repeats M6) | Deep (DriftMonitor, federated, adversarial robustness)    |
| kailash-nexus    | Shallow (repeats M5) | Medium (InferenceServer benchmarking)                   | Shallow (repeats M6) | Medium (multi-tenant deployment)                          |
| kailash-dataflow | none                 | none                                                    | none                 | Medium (governance registry)                              |
