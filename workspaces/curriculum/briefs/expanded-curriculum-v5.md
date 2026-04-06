# ASCENT Curriculum v5 — Final (10 Modules x 8 Lessons)

**Official name**: Professional Certificate in Machine Learning (Python) — Terrene Open Academy
**Audience**: Working professionals with ZERO Python → Masters-level ML engineering
**Structure**: 10 modules x 8 lessons = 80 lessons (~4 hours each, ~320 contact hours)
**Certification structure**: Foundation Certificate (M1-M6) + Advanced Certificate (M7-M10)

## Design Principles (from 4 rounds of red team)

1. **Tangible results from Lesson 1** — `pl.read_csv()` then `print(df.shape)`, not abstract workflow infrastructure
2. **Python through data** — every concept grounded in data manipulation, never abstract exercises
3. **Kailash engines before Kailash infrastructure** — use DataExplorer/ModelVisualizer (M1) before WorkflowBuilder/custom nodes (M3)
4. **Progressive depth** — M1-M6 cover full ML lifecycle; M7-M10 go deep on DL, NLP, agents, alignment
5. **Each lesson: what WORKS / what the weakest student needs** — progressive scaffolding from 70% (M1) to 20% (M10)
6. **Capstone = integration, not from-scratch** — scaffolding increases for capstones
7. **Feature Engineering Spectrum** — manual (M3) → automated (M4) → architecture-driven (M7) → learned (M8-M9)

## Changes from v4

- Expanded from 6 to 10 modules (48 → 80 lessons)
- M4 split: unsupervised/NLP/DL unbundled into M4 (unsupervised), M7 (DL), M8 (NLP)
- M5 split: LLMs/agents/deployment unbundled into M5 (production ML), M9 (LLMs/agents)
- M6 split: alignment/governance/RL unbundled into M6 (unsupervised/monitoring), M10 (alignment/RL/governance)
- Foundation Certificate now M1-M6, Advanced Certificate M7-M10
- Each deep-dive module gets full 8-lesson treatment vs cramped 2-lesson blocks

---

## Module 1: Foundations — Statistics, Probability & Data Fluency

_Zero to productive. Learn Python by exploring real Singapore data._

| #       | Lesson                  | Theory / Python                                                         | Kailash SDK                                                              | Exercise                                     |
| ------- | ----------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------ | -------------------------------------------- |
| 1.1     | **Polars Deep Dive**    | Polars Arrow backend, lazy evaluation, expression API, window functions | `polars`: `pl.read_csv()`, `df.shape`, `df.filter()`, `join()`, `over()` | HDB resale (15M+ rows) with MRT/school joins |
| 1.2     | **Bayesian Estimation** | Normal, Beta, Poisson, conjugate priors, Bayes' theorem                 | `ModelVisualizer` for posterior plots                                    | Bayesian estimation on HDB prices            |
| 1.3     | **Data Profiling**      | Async profiling, alert types, correlation matrices                      | `DataExplorer`, `AlertConfig`, `DataProfile`, `compare()`                | Profile dirty Singapore economic data        |
| 1.4     | **Hypothesis Testing**  | Neyman-Pearson, power analysis, MDE, Bonferroni/BH-FDR                  | `ExperimentTracker` (create_experiment, log_param, log_metric)           | A/B test with multiple corrections           |
| 1.5     | **Data Cleaning**       | None/null handling, encoding, scaling, imputation                       | `PreprocessingPipeline` (auto-detect, encode, scale, impute)             | Clean messy taxi data → full EDA report      |
| 1.6-1.8 | **(Exercises 6-8)**     | Additional practice and integration                                     | Full M1 toolkit                                                          | Progressive difficulty exercises             |

**Datasets**: Singapore HDB Resale (15M+), Economic Indicators, Taxi Trips, E-commerce A/B Test
**Scaffolding**: ~70% (near-complete code with blanks for key arguments)

---

## Module 2: Feature Engineering & Experiment Design

_Statistical foundations taught through experiment tracking and feature engineering._

| #       | Lesson                           | Theory                                                              | Kailash SDK                                                            | Exercise                                        |
| ------- | -------------------------------- | ------------------------------------------------------------------- | ---------------------------------------------------------------------- | ----------------------------------------------- |
| 2.1     | **Feature Engineering**          | Mutual information, Boruta, VIF, temporal features, target encoding | `FeatureEngineer` (generate + select), `FeatureSchema`, `FeatureField` | Healthcare feature engineering on ICU data      |
| 2.2     | **Feature Store**                | Point-in-time correctness, data lineage, versioning                 | `FeatureStore` lifecycle: define → compute → version → retrieve        | FeatureStore with leakage prevention            |
| 2.3     | **A/B Testing & CUPED**          | Power analysis, SRM check, CUPED variance reduction                 | `ExperimentTracker` full lifecycle                                     | CUPED implementation on experiment data         |
| 2.4     | **Causal Inference**             | DiD, propensity matching, parallel trends, placebo tests            | `ExperimentTracker` for causal logging                                 | DiD on Singapore cooling measures               |
| 2.5     | **Automated Feature Generation** | FeatureEngineer + ExperimentTracker integration                     | `FeatureEngineer`, `ExperimentTracker.compare_runs()`                  | Automated generation with experiment comparison |
| 2.6-2.8 | **(Exercises 6-8)**              | Additional practice and integration                                 | Full M2 toolkit                                                        | Progressive difficulty exercises                |

**Datasets**: Healthcare ICU (60K stays), E-commerce Experiment (500K users), Singapore Housing + Policy
**Scaffolding**: ~60% (arguments + some method calls stripped)

---

## Module 3: Supervised ML — Theory to Production

_From theory to production — workflow orchestration, model registry, governed deployment._

| #       | Lesson                               | Theory                                                      | Kailash SDK                                                              | Exercise                                          |
| ------- | ------------------------------------ | ----------------------------------------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------- |
| 3.1     | **Gradient Boosting**                | XGBoost 2nd-order Taylor, LightGBM GOSS, CatBoost ordered   | `TrainingPipeline`, `ModelSpec`, `EvalSpec`                              | XGBoost vs LightGBM vs CatBoost comparison        |
| 3.2     | **Class Imbalance & Calibration**    | SMOTE failures, cost-sensitive, Focal Loss, Platt/isotonic  | `ModelVisualizer.calibration_curve()`, `precision_recall_curve()`        | Imbalance workshop with calibration               |
| 3.3     | **SHAP Interpretability**            | Shapley axioms, TreeSHAP, KernelSHAP, LIME, PDP, ALE        | `ModelVisualizer.feature_importance()`                                   | Full SHAP analysis with interaction plots         |
| 3.4     | **Workflow Orchestration**           | WorkflowBuilder, nodes, connections, custom nodes           | `WorkflowBuilder`, `LocalRuntime`, `PythonCodeNode`, `ConditionalNode`   | ML workflow: load → preprocess → train → evaluate |
| 3.5     | **Hyperparameter Search & Registry** | Bayesian optimization, model versioning, staging→production | `HyperparameterSearch`, `SearchSpace`, `ModelRegistry`, `ModelSignature` | Bayesian optimization → staging → production      |
| 3.6     | **End-to-End Pipeline**              | Model cards, conformal prediction, full pipeline            | Complete: train → calibrate → register → promote → model card            | Production pipeline with model card               |
| 3.7-3.8 | **(Exercises 7-8)**                  | DataFlow persistence, integration                           | `DataFlow`, `field()`, `db.express`, `ConnectionManager`                 | Persist ML results + integration project          |

**Datasets**: Singapore Credit Scoring (100K apps, 12% default), Lending Club (300K+)
**Scaffolding**: ~50% (method calls + setup stripped)

---

## Module 4: Unsupervised ML & Pattern Discovery

_Clustering, dimensionality reduction, anomaly detection, and production monitoring._

| #       | Lesson                            | Theory                                            | Kailash SDK                                                  | Exercise                                 |
| ------- | --------------------------------- | ------------------------------------------------- | ------------------------------------------------------------ | ---------------------------------------- |
| 4.1     | **Clustering**                    | K-means, spectral, HDBSCAN, GMM, gap statistic    | `AutoMLEngine`, `AutoMLConfig`                               | Clustering comparison on customer data   |
| 4.2     | **Anomaly Detection & Ensembles** | Isolation Forest, LOF, score blending             | `EnsembleEngine` (`blend()`, `stack()`, `bag()`, `boost()`)  | UMAP + anomaly detection on fraud data   |
| 4.3     | **NLP: Text to Topics**           | TF-IDF, BM25, Word2Vec, BERTopic, NPMI coherence  | `ModelVisualizer` for topics                                 | BERTopic on Singapore news corpus        |
| 4.4     | **Drift Monitoring**              | PSI, KS test, performance degradation, monitoring | `DriftMonitor`, `DriftSpec`                                  | Deploy model, simulate drift, detect     |
| 4.5     | **Deep Learning Intro**           | Backprop, gradient flow, CNN, ResBlock, BatchNorm | PyTorch + `ModelVisualizer.training_history()`, `OnnxBridge` | CNN on chest X-ray with LR scheduling    |
| 4.6     | **Inference & Deployment**        | ONNX export, InferenceServer, Nexus multi-channel | `OnnxBridge`, `InferenceServer`, `Nexus`                     | InferenceServer + Nexus: API + CLI + MCP |
| 4.7-4.8 | **(Exercises 7-8)**               | Additional practice and integration               | Full M4 toolkit                                              | Progressive difficulty exercises         |

**Datasets**: E-commerce (200K), Credit Card Fraud (284K), Singapore News (50K), ChestX-ray14 (10K)
**Scaffolding**: ~40% (setup + calls + logic stripped)

---

## Module 5: LLMs, AI Agents & RAG Systems

_Build intelligent agents, then deploy them at scale._

| #       | Lesson                        | Theory / Practice                                     | Kailash SDK                                                       | Exercise                             |
| ------- | ----------------------------- | ----------------------------------------------------- | ----------------------------------------------------------------- | ------------------------------------ |
| 5.1     | **Delegate & Signatures**     | Tokenization, scaling laws, structured output         | `Kaizen`: `Signature`, `InputField`, `OutputField`, `Delegate`    | Delegate + SimpleQAAgent             |
| 5.2     | **Chain-of-Thought**          | Step-by-step reasoning, CoT vs direct answering       | `ChainOfThoughtAgent`                                             | CoT on clustering results from M4    |
| 5.3     | **ReAct Agents**              | Reasoning + action loops, tool selection, cost budget | `ReActAgent`, custom tools                                        | ReAct with DataExplorer/polars tools |
| 5.4     | **RAG Systems**               | Chunking, retrieval, RAGAS evaluation, HyDE           | `RAGResearchAgent`, `MemoryAgent`                                 | RAG over SDK + regulatory docs       |
| 5.5     | **ML Agent Pipeline**         | LLMs augmenting ML lifecycle, double opt-in           | 6 ML agents: DataScientist, FeatureEngineer, ModelSelector, etc.  | Full 6-agent pipeline                |
| 5.6     | **Multi-Agent Orchestration** | A2A, supervisor-worker, sequential, parallel, handoff | `SupervisorWorkerPattern`, `SequentialPattern`, `ParallelPattern` | Multi-agent A2A orchestration        |
| 5.7-5.8 | **(Exercises 7-8)**           | MCP integration, production deployment                | `kailash.mcp_server`, `Nexus`, middleware                         | MCP server + Nexus deployment        |

**Datasets**: E-commerce + credit (familiar data), Kailash SDK docs, Singapore AI Verify corpus
**Scaffolding**: ~30% (most logic stripped, imports + structure given)

---

## Module 6: Alignment, Governance & Production

_Fine-tuning, governance, RL, and the capstone. Masters-level content, coherently grouped._

| #       | Lesson                      | Theory                                          | Kailash SDK                                                   | Exercise                                     |
| ------- | --------------------------- | ----------------------------------------------- | ------------------------------------------------------------- | -------------------------------------------- |
| 6.1     | **SFT Fine-Tuning**         | LoRA/QLoRA/DoRA, adapter lifecycle              | `AlignmentPipeline`, `AlignmentConfig`, `AdapterRegistry`     | SFT on small model with adapter tracking     |
| 6.2     | **Preference Alignment**    | DPO, Bradley-Terry, LLM-as-judge                | `AlignmentPipeline` (method="dpo"), `evaluator`               | DPO with LLM-as-judge evaluation             |
| 6.3     | **AI Governance with PACT** | EU AI Act, Singapore AI Verify, D/T/R grammar   | `GovernanceEngine`, `compile_org()`, `Address`, `CostTracker` | Define org in YAML, compile, verify access   |
| 6.4     | **Governed Agents**         | Monotonic tightening, fail-closed, audit chains | `PactGovernedAgent`, `GovernanceContext`, `RoleEnvelope`      | Wrap ReActAgent with PACT enforcement        |
| 6.5     | **RL Fundamentals**         | Bellman equations, PPO, reward design           | `RLTrainer`, `env_registry`, `policy_registry`                | PPO on inventory management environment      |
| 6.6     | **Capstone: Full Platform** | All packages integrated, production readiness   | Core SDK → DataFlow → ML → Kaizen → PACT → Nexus → Align      | Full governed ML platform (~40% scaffolding) |
| 6.7-6.8 | **(Exercises 7-8)**         | Advanced alignment, agent governance at scale   | `kailash-align` merge, `kaizen_agents.governance`             | Model merging + multi-agent governance       |

**Datasets**: Domain Q&A (SFT, 1000 pairs), Preference pairs (DPO, 500), Gymnasium environments
**Scaffolding**: ~20% for 6.1-6.5, ~40% for 6.6 capstone

---

## Module 7: Deep Learning — Architecture-Driven Feature Engineering

_Deep learning moves feature engineering inside the model. From single neurons to CNNs._

| #   | Lesson                         | Theory                                                           | Kailash SDK                                                          | Exercise                                     |
| --- | ------------------------------ | ---------------------------------------------------------------- | -------------------------------------------------------------------- | -------------------------------------------- |
| 7.1 | **Neural Network Foundations** | Linear regression as neuron, forward pass, MSE, gradient descent | `TrainingPipeline`, `ModelVisualizer`                                | Linear regression as single-neuron network   |
| 7.2 | **Hidden Layers & XOR**        | Multi-layer perceptron, decision boundaries, depth vs width      | `ModelVisualizer` for decision boundaries                            | Hidden layers solving XOR problem            |
| 7.3 | **Activation Functions**       | Sigmoid vs ReLU vs GELU, gradient flow, dead neurons             | `ModelVisualizer.training_history()`                                 | Activation function comparison and analysis  |
| 7.4 | **Loss & Initialization**      | CrossEntropy, Xavier vs He init, vanishing gradients             | `TrainingPipeline`                                                   | Loss functions with weight initialization    |
| 7.5 | **Backpropagation**            | Chain rule, computational graph, gradient checking               | Manual implementation                                                | Backprop from scratch with gradient checking |
| 7.6 | **Optimizers & Scheduling**    | SGD, momentum, Adam, cosine annealing, warm restarts             | `TrainingPipeline`                                                   | Optimizer comparison with LR scheduling      |
| 7.7 | **CNNs**                       | Convolution, pooling, dropout, architectures, ONNX export        | `OnnxBridge` (export, validate)                                      | CNN image classification with ONNX export    |
| 7.8 | **Capstone: DL Pipeline**      | End-to-end: data → train → register → export → serve             | `TrainingPipeline`, `ModelRegistry`, `OnnxBridge`, `InferenceServer` | Full DL pipeline with production serving     |

**Datasets**: MNIST sample, Fashion-MNIST sample, HDB resale prices, synthetic spirals
**Scaffolding**: ~30% (structure + imports given, most implementation stripped)

---

## Module 8: NLP & Transformers

_From raw text to transformers. The journey from bag-of-words to attention._

| #   | Lesson                       | Theory                                                                | Kailash SDK                                   | Exercise                                  |
| --- | ---------------------------- | --------------------------------------------------------------------- | --------------------------------------------- | ----------------------------------------- |
| 8.1 | **Text Preprocessing**       | Tokenization (word, BPE, WordPiece), stemming, lemmatization, n-grams | `PreprocessingPipeline`                       | Text preprocessing pipeline               |
| 8.2 | **BoW & TF-IDF**             | Bag of Words, TF-IDF formula, BM25, document classification           | `ModelVisualizer`                             | BoW and TF-IDF on document corpus         |
| 8.3 | **Word Embeddings**          | Word2Vec (skip-gram, CBOW), GloVe, analogy tasks                      | `ModelVisualizer` for t-SNE                   | Word2Vec training and visualization       |
| 8.4 | **Sequence Models**          | RNNs, LSTMs, gating mechanisms, bidirectional models                  | `TrainingPipeline`                            | RNN/LSTM for sentiment analysis           |
| 8.5 | **Attention Mechanisms**     | Scaled dot-product, multi-head attention, visualization               | `ModelVisualizer`                             | Attention mechanism implementation        |
| 8.6 | **Transformer Architecture** | Positional encoding, encoder/decoder, LayerNorm                       | Manual implementation                         | Build transformer encoder from components |
| 8.7 | **Transfer Learning**        | BERT, GPT, fine-tuning, decoding strategies                           | `AutoMLEngine`, `ModelRegistry`, `OnnxBridge` | Transfer learning with AutoMLEngine       |
| 8.8 | **Capstone: NLP Pipeline**   | End-to-end: preprocess → embed → classify → export                    | Full M8 toolkit                               | Full NLP pipeline with ONNX export        |

**Datasets**: Singapore news articles, Parliament speeches, product reviews
**Scaffolding**: ~30% (structure + imports given, most implementation stripped)

---

## Module 9: LLMs, AI Agents & RAG Systems

_From prediction to reasoning and action. Build production agent systems._

| #   | Lesson                         | Theory                                                            | Kailash SDK                            | Exercise                                  |
| --- | ------------------------------ | ----------------------------------------------------------------- | -------------------------------------- | ----------------------------------------- |
| 9.1 | **LLM Architecture**           | Decoder-only transformers, KV cache, scaling laws, model timeline | `Delegate` (first call)                | LLM architecture and tokenization         |
| 9.2 | **Prompt Engineering**         | Zero-shot, few-shot, chain-of-thought, structured extraction      | `Delegate`, `Signature`                | Prompt engineering with structured output |
| 9.3 | **RAG Fundamentals**           | Document chunking, embeddings, vector similarity, retrieval       | Kaizen RAG components                  | RAG system with document retrieval        |
| 9.4 | **Advanced RAG**               | Hybrid search, re-ranking, RAGAS evaluation metrics               | Kaizen RAG + evaluation                | RAG evaluation with faithfulness metrics  |
| 9.5 | **Building Agents**            | ReAct architecture, custom tools, BaseAgent with Signature        | `ReActAgent`, `BaseAgent`, `Signature` | Kaizen agents with custom tools           |
| 9.6 | **Multi-Agent Orchestration**  | Specialist agents, Pipeline.router(), supervisor pattern          | `Pipeline`, coordination patterns      | Multi-agent specialist system             |
| 9.7 | **MCP Integration**            | MCPServer, tool registration, agent-driven ML via MCP             | `kailash.mcp_server`, MCP tools        | MCP server for ML tool exposure           |
| 9.8 | **Capstone: Agent Deployment** | Multi-channel deployment, session persistence                     | `Nexus`, MCP, `Delegate`               | RAG agent deployed via Nexus              |

**Datasets**: Singapore company reports, Singapore regulations
**Scaffolding**: ~25% (minimal structure, students build from documentation)

---

## Module 10: Alignment, RL & Governance

_Once you build a powerful system, how do you keep it safe, useful, and accountable?_

| #    | Lesson                           | Theory                                                               | Kailash SDK                                                           | Exercise                                         |
| ---- | -------------------------------- | -------------------------------------------------------------------- | --------------------------------------------------------------------- | ------------------------------------------------ |
| 10.1 | **LoRA Fine-Tuning**             | LoRA/QLoRA/DoRA, SFT, parameter savings, adapter lifecycle           | `AlignmentPipeline`, `AlignmentConfig`, `AdapterRegistry`             | LoRA SFT with parameter savings calculation      |
| 10.2 | **Preference Alignment**         | DPO derivation, Bradley-Terry, GRPO, ORPO, SimPO, KTO                | `AlignmentPipeline` (method="dpo")                                    | DPO with beta tuning and safety evaluation       |
| 10.3 | **RL Fundamentals**              | MDPs, Bellman equations, value iteration, Q-learning, DQN            | `RLTrainer`                                                           | MDP + Q-learning + DQN implementation            |
| 10.4 | **PPO Training**                 | Clipped objective, GAE, reward shaping, policy gradient              | `RLTrainer` (PPO)                                                     | PPO on inventory management environment          |
| 10.5 | **Model Merging & Export**       | Linear merge, SLERP, TIES, DARE, quantization                        | `AdapterRegistry`, `OnnxBridge`                                       | Merge adapters and export to ONNX                |
| 10.6 | **AI Governance with PACT**      | EU AI Act, Singapore AI Verify, D/T/R grammar, operating envelopes   | `GovernanceEngine`, `compile_org()`, `Address`                        | YAML org → compile → verify → explain            |
| 10.7 | **Governed Agents**              | Budget cascade, tool restrictions, monotonic tightening, audit chain | `PactGovernedAgent`, `GovernanceContext`                              | Governed agent with budget and tool limits       |
| 10.8 | **Capstone: Governed ML System** | Full integration: alignment + PACT + Nexus deployment                | `AlignmentPipeline`, `GovernanceEngine`, `PactGovernedAgent`, `Nexus` | Governed ML system with multi-channel deployment |

**Datasets**: Singapore domain Q&A (SFT, 1000 pairs), preference pairs (DPO, 500), inventory demand, company reports
**Scaffolding**: ~20% (imports and comments only, students build from API reference)

---

## Exercise Inventory

| Module    | Exercises | Scaffolding | Key Kailash Frameworks                                                  |
| --------- | --------- | ----------- | ----------------------------------------------------------------------- |
| M1        | 8         | 70%         | kailash-ml: DataExplorer, PreprocessingPipeline, ModelVisualizer        |
| M2        | 8         | 60%         | kailash-ml: FeatureEngineer, FeatureStore, ExperimentTracker            |
| M3        | 8         | 50%         | Core SDK, DataFlow, kailash-ml: TrainingPipeline, ModelRegistry         |
| M4        | 8         | 40%         | kailash-ml: AutoMLEngine, EnsembleEngine, DriftMonitor, InferenceServer |
| M5        | 8         | 30%         | Kaizen: Delegate, BaseAgent, 6 ML agents, coordination patterns         |
| M6        | 8         | 20%         | Align, PACT, kailash-ml: RLTrainer, Nexus                               |
| M7        | 8         | 30%         | kailash-ml: OnnxBridge, InferenceServer, TrainingPipeline               |
| M8        | 8         | 30%         | kailash-ml: ModelVisualizer, AutoMLEngine, OnnxBridge                   |
| M9        | 8         | 25%         | Kaizen: Delegate, ReActAgent, Pipeline; Nexus, MCP                      |
| M10       | 8         | 20%         | Align, PACT, kailash-ml: RLTrainer                                      |
| **Total** | **80**    | —           | **All 8 Kailash packages**                                              |

**Exercise formats**: Each exercise exists in 3 formats (local .py, Jupyter .ipynb, Colab .ipynb) = 240 exercise files total.

---

## Complete SDK Coverage

| Package              | Modules         | Key Classes                                                                                        |
| -------------------- | --------------- | -------------------------------------------------------------------------------------------------- |
| **kailash** (core)   | M3, M9          | WorkflowBuilder, LocalRuntime, Node, @register_node, PythonCodeNode, ConnectionManager, MCP server |
| **kailash-ml**       | M1-M8, M10      | All 13 engines + 6 ML agents + RLTrainer + OnnxBridge + interop                                    |
| **kailash-dataflow** | M3              | @db.model, field(), db.express CRUD                                                                |
| **kailash-nexus**    | M4, M5, M9, M10 | Nexus, auth (RBAC/JWT), middleware, plugins                                                        |
| **kailash-kaizen**   | M5, M9          | Signature, InputField/OutputField, BaseAgent, Delegate, ReActAgent, Pipeline                       |
| **kaizen-agents**    | M5, M9          | 6+ specialized agents, coordination patterns, ML agents                                            |
| **kailash-pact**     | M6, M10         | GovernanceEngine, PactGovernedAgent, Address, enforcement, costs                                   |
| **kailash-align**    | M6, M10         | AlignmentPipeline, AlignmentConfig, AdapterRegistry, merge, evaluator                              |

---

## Progression Summary

```
M1:  Zero Python → Data exploration with Polars + Kailash engines
M2:  Statistics → Experiment design with ExperimentTracker + FeatureStore
M3:  Supervised ML → Production with WorkflowBuilder + ModelRegistry + DataFlow
M4:  Unsupervised + NLP intro → Monitoring with AutoML + DriftMonitor + Nexus
M5:  LLM Agents → Deployed at scale with Kaizen + MCP
M6:  Alignment + Governance → Full platform capstone with Align + PACT + RL
M7:  Deep learning from neurons to CNNs → ONNX serving with OnnxBridge
M8:  Text → Transformers → Transfer learning with AutoMLEngine
M9:  LLM reasoning → Agent systems → MCP + Nexus deployment
M10: Fine-tuning → RL → Governance → Governed production systems

Foundation Certificate: M1-M6 (48 lessons, 192 hours)
Advanced Certificate: M7-M10 (32 lessons, 128 hours)
Full Programme: 80 lessons, 320 contact hours
```

---

## Feature Engineering Spectrum

The programme traces the evolution of feature engineering as a unifying narrative:

| Stage                | Module | Approach                                            | Example                                        |
| -------------------- | ------ | --------------------------------------------------- | ---------------------------------------------- |
| Manual               | M2-M3  | Domain expert creates features by hand              | `FeatureEngineer.generate()` with domain rules |
| Automated            | M4     | AutoML discovers features algorithmically           | `AutoMLEngine` with feature search             |
| Architecture-Driven  | M7     | Neural network learns features internally           | CNN filters, learned embeddings                |
| Learned & Contextual | M8-M9  | Attention mechanisms create dynamic features        | Transformer self-attention, RAG context        |
| Aligned              | M10    | Features shaped by human preferences and governance | DPO alignment, PACT operating envelopes        |
