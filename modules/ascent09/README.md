# Module 9: Fine-Tuning, Alignment & RL

**Kailash**: Align (AlignmentPipeline, AdapterRegistry, LoRAManager), kailash-ml (OnnxBridge, InferenceServer, TrainingPipeline), Kaizen (Delegate, ReActAgent, Signature, Pipeline) | **Scaffolding**: 25%

**Prerequisites**: M5 (Kaizen agents, RAG, MCP, Nexus), M6 (SFT, DPO, RL, PACT), M8 (transformer internals, BERT/GPT, fine-tuning)

## Lecture (3h)

- **9A** LoRA Internals: low-rank decomposition, effective rank, gradient flow, rank selection heuristics
- **9B** Preference Alignment: Bradley-Terry model, DPO derivation from RLHF, reward model elimination
- **9C** GRPO: group-relative advantage estimation, verifiable rewards, comparison with PPO, DeepSeek-R1
- **9D** Constitutional AI: critique-revision loop, principle chaining, Singapore regulatory context
- **9E-H** Inference, Safety & Evaluation: quantization (INT8/FP16), KV cache, speculative decoding, reward hacking, red-teaming, adversarial fine-tuning

## Lab (3h) — 8 Exercises

| Ex  | Title                                              | Objective                                                                                                                                                           | Key Frameworks                                                                                   |
| --- | -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| 1   | LoRA Internals and Rank Selection                  | Derive LoRA from first principles — SVD decomposition, forward pass, stable-rank heuristic — then validate against AlignmentPipeline                                | Align: AlignmentPipeline, LoRAConfig, LoRAManager                                                |
| 2   | DPO Derivation and Reward Modeling                 | Derive DPO from RLHF — implement Bradley-Terry model, DPO loss, gradient verification — and compare aligned vs base at multiple beta values                         | Align: AlignmentPipeline(method="dpo"), AdapterRegistry; Kaizen: Signature                       |
| 3   | GRPO and Group Relative Policy Optimization        | Implement GRPO objective (within-group score normalisation), verifiable reward functions, clipped update, and compare against PPO                                   | Align: AlignmentPipeline(method="grpo"); kailash-ml: TrainingPipeline                            |
| 4   | Constitutional AI and Self-Critique                | Build a CAI critique-revision pipeline with Singapore-specific principles (MAS AML, PDPA) and evaluate principle violation rates                                    | Kaizen: Delegate, Pipeline, Signature                                                            |
| 5   | Inference Optimization and Quantization            | Implement INT8 quantization, KV cache with sliding-window eviction, and speculative decoding; benchmark throughput vs latency                                       | kailash-ml: OnnxBridge, InferenceServer; Align: AlignmentPipeline                                |
| 6   | Reward Hacking and Goodhart's Law                  | Demonstrate proxy reward gaming, implement KL-regularisation and reward ensembles as defences                                                                       | Kaizen: Delegate; Align: AlignmentPipeline                                                       |
| 7   | Safety Evaluation and Red-Teaming                  | Build jailbreak classifier, automated red-team agent, gradient-free adversarial suffix search; measure attack success rate before and after adversarial fine-tuning | Kaizen: Signature, ReActAgent, Delegate; Align: AlignmentPipeline                                |
| 8   | Capstone — Full Fine-Tuning and Alignment Pipeline | Run the complete pipeline: SFT → reward model → DPO → GRPO → INT8 export → safety evaluation → compliance attestation                                               | Align: AlignmentPipeline, AdapterRegistry, LoRAConfig; kailash-ml: OnnxBridge; Kaizen: Signature |

## Learning Outcomes

By the end of this module, students can:

- Derive LoRA mathematically and implement low-rank decomposition from first principles
- Derive the DPO loss from the RLHF objective and explain why it eliminates the separate reward model
- Implement GRPO with group-relative advantage estimation and compare it empirically to PPO
- Build a Constitutional AI pipeline that critiques and revises outputs against regulatory principles
- Implement INT8 quantization, KV cache management, and speculative decoding for inference optimisation
- Identify and defend against proxy reward gaming using KL-regularisation and reward ensembles
- Build automated red-teaming tools that measure model attack success rate across alignment strategies
- Integrate every M9 technique into a full fine-tuning pipeline with compliance attestation output

## Datasets

- `data/ascent09/sg_company_reports.parquet` — SG company text (LoRA fine-tuning, inference benchmarking, CAI)
- `data/ascent09/sg_regulations.parquet` — Singapore regulatory Q&A (GRPO verifiable rewards, reward hacking)
- `data/ascent10/preference_pairs.parquet` — DPO preference pairs (chosen/rejected responses)
- `data/ascent10/sg_domain_qa.parquet` — Singapore domain Q&A (capstone SFT)
- `data/ascent09/safety_prompts.parquet` — Jailbreak and adversarial prompts (red-teaming)
