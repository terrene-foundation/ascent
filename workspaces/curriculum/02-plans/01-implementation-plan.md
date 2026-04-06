# ASCENT Implementation Plan — v5 Curriculum

## Execution Summary

All deliverables produced in parallel autonomous agent streams. Expanded from v4 (6 modules) to v5 (10 modules).

## Deliverable Status

### Phase 1: Content Foundation (Complete)

- [x] 80/80 exercise solutions across 10 modules (8 per module)
- [x] v5 curriculum alignment (10 modules x 8 lessons = 80 lessons)
- [x] Infrastructure: shared.run_profile, sg_weather.csv, uv.lock
- [x] ASCENTDataLoader updated for local data/ directory
- [x] 11/11 datasets generated
- [x] pyproject.toml build config (hatchling packages directive)

### Phase 2: Exercise Packaging (Complete)

- [x] 80/80 local exercise files (stripped from solutions)
- [x] 80/80 Jupyter notebooks (converted from local)
- [x] 80/80 Colab notebooks (converted from local)

### Phase 3: Supplementary Materials (Complete)

- [x] 10 Reveal.js lecture decks (~1,398 slides)
- [x] 10 speaker notes files with time budget sections (M1-M9)
- [x] 10 module READMEs with lesson plans (8 exercises each)
- [x] 10 quiz files (246 questions total, 24-25 per module)
- [x] 10 deck PDFs (decktape, 1280x720)
- [x] Curriculum v5 document (expanded-curriculum-v5.md)

### Phase 4: Textbook (Complete)

- [x] 83/83 Python tutorial files across 9 sections
- [x] 83/83 Markdown tutorial files
- [x] 84 HTML tutorial files

### Phase 5: Validation (Complete — Session 10)

- [x] Red team all exercises (RT1: spec coverage, RT2: three-format, RT3: scaffolding)
- [x] Verify three-format consistency (240/240 files, 0 blank mismatches)
- [x] Check no hardcoded keys/secrets (RT4: 0 violations)
- [x] Check no PCML references (RT6: 0 violations, .env.example clean)
- [x] Validate PDFs render correctly (10/10 confirmed)
- [x] Fix deck-audit script (slide bounds, not viewport)
- [x] Fix py_to_notebook.py (triple-quote blanks, package detection, asyncio.run)

### Phase 6: Outstanding (Future Work)

- [ ] Block-level scaffolding redesign for M3-M10 exercises (RT3 finding: all exercises over-scaffolded at 80-97% vs 20-70% target)
- [ ] M7-M10 quiz process_doc question type (RT5 finding: 0 process_doc in advanced modules)
- [ ] M5 speaker notes expansion (18 section-grouped entries vs 150 HTML slides — adequate but less granular than other modules)

## Module Mapping (v5)

### Foundation Certificate (M1-M6)

| v5 Exercise   | Source                                | Status   |
| ------------- | ------------------------------------- | -------- |
| M1 ex_1 (1.1) | Polars deep dive on HDB resale        | Complete |
| M1 ex_2 (1.2) | Bayesian estimation                   | Complete |
| M1 ex_3 (1.3) | DataExplorer profiling                | Complete |
| M1 ex_4 (1.4) | Hypothesis testing                    | Complete |
| M1 ex_5 (1.5) | Data cleaning pipeline                | Complete |
| M1 ex_6-8     | Additional practice + integration     | Complete |
| M2 ex_1 (2.1) | Feature engineering on ICU data       | Complete |
| M2 ex_2 (2.2) | FeatureStore lifecycle                | Complete |
| M2 ex_3 (2.3) | A/B testing + CUPED                   | Complete |
| M2 ex_4 (2.4) | Causal inference (DiD)                | Complete |
| M2 ex_5 (2.5) | Automated feature generation          | Complete |
| M2 ex_6-8     | Additional practice + integration     | Complete |
| M3 ex_1 (3.1) | Gradient boosting comparison          | Complete |
| M3 ex_2 (3.2) | Class imbalance + calibration         | Complete |
| M3 ex_3 (3.3) | SHAP interpretability                 | Complete |
| M3 ex_4 (3.4) | Workflow orchestration                | Complete |
| M3 ex_5 (3.5) | HyperparameterSearch + ModelRegistry  | Complete |
| M3 ex_6 (3.6) | End-to-end pipeline                   | Complete |
| M3 ex_7-8     | DataFlow + integration                | Complete |
| M4 ex_1 (4.1) | Clustering comparison                 | Complete |
| M4 ex_2 (4.2) | Anomaly detection + ensembles         | Complete |
| M4 ex_3 (4.3) | NLP: BERTopic                         | Complete |
| M4 ex_4 (4.4) | Drift monitoring                      | Complete |
| M4 ex_5 (4.5) | Deep learning CNN                     | Complete |
| M4 ex_6 (4.6) | InferenceServer + Nexus               | Complete |
| M4 ex_7-8     | Additional practice + integration     | Complete |
| M5 ex_1 (5.1) | Delegate + SimpleQAAgent              | Complete |
| M5 ex_2 (5.2) | ChainOfThoughtAgent                   | Complete |
| M5 ex_3 (5.3) | ReActAgent with tools                 | Complete |
| M5 ex_4 (5.4) | RAG systems                           | Complete |
| M5 ex_5 (5.5) | ML agent pipeline                     | Complete |
| M5 ex_6 (5.6) | Multi-agent orchestration             | Complete |
| M5 ex_7-8     | MCP + Nexus deployment                | Complete |
| M6 ex_1 (6.1) | SFT fine-tuning                       | Complete |
| M6 ex_2 (6.2) | DPO preference alignment              | Complete |
| M6 ex_3 (6.3) | PACT governance                       | Complete |
| M6 ex_4 (6.4) | Governed agents                       | Complete |
| M6 ex_5 (6.5) | RL fundamentals                       | Complete |
| M6 ex_6 (6.6) | Capstone: full platform               | Complete |
| M6 ex_7-8     | Advanced alignment + agent governance | Complete |

### Advanced Certificate (M7-M10)

| v5 Exercise     | Source                       | Status   |
| --------------- | ---------------------------- | -------- |
| M7 ex_1 (7.1)   | Linear regression as neuron  | Complete |
| M7 ex_2 (7.2)   | Hidden layers + XOR          | Complete |
| M7 ex_3 (7.3)   | Activation functions         | Complete |
| M7 ex_4 (7.4)   | Loss + initialization        | Complete |
| M7 ex_5 (7.5)   | Backpropagation from scratch | Complete |
| M7 ex_6 (7.6)   | Optimizers + LR scheduling   | Complete |
| M7 ex_7 (7.7)   | CNN image classification     | Complete |
| M7 ex_8 (7.8)   | Capstone: DL pipeline        | Complete |
| M8 ex_1 (8.1)   | Text preprocessing           | Complete |
| M8 ex_2 (8.2)   | BoW + TF-IDF                 | Complete |
| M8 ex_3 (8.3)   | Word embeddings              | Complete |
| M8 ex_4 (8.4)   | RNNs + LSTMs                 | Complete |
| M8 ex_5 (8.5)   | Attention mechanisms         | Complete |
| M8 ex_6 (8.6)   | Transformer architecture     | Complete |
| M8 ex_7 (8.7)   | Transfer learning            | Complete |
| M8 ex_8 (8.8)   | Capstone: NLP pipeline       | Complete |
| M9 ex_1 (9.1)   | LLM architecture             | Complete |
| M9 ex_2 (9.2)   | Prompt engineering           | Complete |
| M9 ex_3 (9.3)   | RAG fundamentals             | Complete |
| M9 ex_4 (9.4)   | Advanced RAG                 | Complete |
| M9 ex_5 (9.5)   | Kaizen agents                | Complete |
| M9 ex_6 (9.6)   | Multi-agent orchestration    | Complete |
| M9 ex_7 (9.7)   | MCP integration              | Complete |
| M9 ex_8 (9.8)   | Capstone: agent deployment   | Complete |
| M10 ex_1 (10.1) | LoRA fine-tuning             | Complete |
| M10 ex_2 (10.2) | DPO alignment                | Complete |
| M10 ex_3 (10.3) | RL fundamentals              | Complete |
| M10 ex_4 (10.4) | PPO training                 | Complete |
| M10 ex_5 (10.5) | Model merging + export       | Complete |
| M10 ex_6 (10.6) | PACT governance              | Complete |
| M10 ex_7 (10.7) | Governed agents              | Complete |
| M10 ex_8 (10.8) | Capstone: governed ML system | Complete |
