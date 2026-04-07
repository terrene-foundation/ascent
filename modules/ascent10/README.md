# Module 10: AI Governance, Safety & Enterprise

**Kailash**: PACT (GovernanceEngine, PactGovernedAgent, ClearanceManager), kailash-ml (DriftMonitor, InferenceServer), Kaizen (ReActAgent, Signature, Delegate), DataFlow, Nexus | **Scaffolding**: 20%

**Prerequisites**: M6 (GovernanceEngine, AlignmentPipeline, DPO, RL, PACT basics), M9 (LoRA, DPO derivation, GRPO, safety evaluation, inference optimisation)

## Lecture (3h)

- **10A** Cross-Org Governance: federated PACT structures, KSP bridges, delegation chaining, breach simulation
- **10B** Federated Learning & Privacy: FedAvg, differential privacy (Gaussian mechanism, moments accountant), privacy-utility trade-offs
- **10C** Privacy-Preserving ML: k-anonymity, membership inference attacks and defences, synthetic data, PDPA/GDPR alignment
- **10D** Compliance Artefacts: model cards, dataset datasheets, bias audits, system cards, EU AI Act Annex IV, ISO/IEC 42001
- **10E-H** Adversarial Robustness & Enterprise: FGSM, PGD, adversarial training, randomised smoothing, regulatory compliance automation, multi-tenant ML governance, regulated AI deployment

## Lab (3h) — 8 Exercises

| Ex  | Title                                             | Objective                                                                                                                                                                            | Key Frameworks                                                                               |
| --- | ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------- |
| 1   | Multi-Organisation Governance                     | Build cross-organisation governance with two PACT engines (MAS regulator + bank), KSP bridge, delegation chaining CEO→compliance→agent, and breach simulation                        | PACT: GovernanceEngine, PactGovernedAgent, ClearanceManager                                  |
| 2   | Federated Learning with Privacy Guarantees        | Implement FedAvg from scratch across 3 virtual hospital clients, add Gaussian-noise differential privacy with moments accountant, and integrate with PACT + InferenceServer          | kailash-ml: InferenceServer, DriftMonitor; PACT: PactGovernedAgent, GovernanceEngine         |
| 3   | Privacy-Preserving ML and Data Minimisation       | Implement k-anonymity, membership inference attack and L2/early-stopping defence, Gaussian copula synthetic data, and PACT-governed inference with ClearanceManager                  | PACT: PactGovernedAgent, ClearanceManager; kailash-ml: TrainingPipeline                      |
| 4   | Model Cards, Datasheets, and Compliance Artefacts | Generate model cards, dataset datasheets, bias audits (demographic parity, equalised odds), system cards with EU AI Act risk mapping, and DriftMonitor post-deployment reports       | Kaizen: Signature; kailash-ml: DriftMonitor; PACT: GovernanceEngine                          |
| 5   | Adversarial Robustness and Attacks                | Implement FGSM and PGD evasion attacks, adversarial training, data poisoning with influence-function detection, and certified robustness via randomised smoothing                    | kailash-ml: TrainingPipeline; PACT: GovernanceEngine                                         |
| 6   | Regulatory Compliance Automation with PACT        | Define multi-regulation compliance schemas as Signatures, build a ComplianceAuditAgent, trigger audits from DriftMonitor alerts, and deploy as a Nexus endpoint                      | Kaizen: Signature, ReActAgent; kailash-ml: DriftMonitor; PACT: GovernanceEngine; Nexus       |
| 7   | Enterprise ML Governance at Scale                 | Build a model governance registry via DataFlow, implement automated retirement on drift, enforce multi-tenancy isolation on a shared InferenceServer, and simulate a bias incident   | DataFlow: @db.model, db.express; kailash-ml: DriftMonitor, InferenceServer; PACT; Nexus      |
| 8   | Capstone — Regulated AI System End-to-End         | Combine federated training, compliance artefacts, adversarial hardening, governance registry, and governed Nexus deployment into an audit-ready machine-readable attestation package | PACT: GovernanceEngine, PactGovernedAgent; kailash-ml: DriftMonitor; Kaizen; Nexus; DataFlow |

## Learning Outcomes

By the end of this module, students can:

- Model cross-organisational governance with federated PACT engines, KSP bridges, and delegation chains
- Implement FedAvg with differential privacy and account for privacy budget across training rounds
- Apply k-anonymity, synthetic data generation, and membership inference defences to regulated datasets
- Produce EU AI Act-compliant artefacts: model cards, dataset datasheets, system cards, and bias audits
- Implement FGSM, PGD, and randomised smoothing, and interpret certified robustness as a regulatory metric
- Build a ComplianceAuditAgent that maps governance artefacts to multi-regulation requirements automatically
- Design enterprise-scale ML governance with automated retirement, multi-tenancy isolation, and audit trails
- Deploy a complete regulated AI system that satisfies all M10 requirements and outputs a compliance attestation

## Datasets

- `data/ascent02/icu_patients.parquet` — ICU records partitioned across virtual hospital clients (federated learning, privacy)
- `data/ascent04/credit_card_fraud.parquet` — Fraud detection (adversarial robustness, bias audits, governance registry)
- `data/ascent09/sg_company_reports.parquet` — SG company text (dataset datasheet, compliance artefacts)
- `data/ascent10/preference_pairs.parquet` — DPO preference pairs (reward modeling reference)
- `data/ascent10/sg_domain_qa.parquet` — Singapore domain Q&A (capstone federated model)
- `data/ascent10/sg_regulatory_orgs.yaml` — MAS regulator and bank org definitions (multi-org governance)
