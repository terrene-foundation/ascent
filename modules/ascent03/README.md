# Module 3: Supervised ML — Theory to Production

**Kailash**: Core SDK, DataFlow, kailash-ml (TrainingPipeline, HyperparameterSearch, ModelRegistry) | **Scaffolding**: 50%

## Lecture (3h)

- **3A** Supervised ML Theory: bias-variance decomposition, regularization (L1/L2 geometry), gradient boosting internals (XGBoost/LightGBM/CatBoost), ensemble theory, class imbalance (SMOTE failures, Focal Loss)
- **3B** Evaluation & Interpretability: precision-recall trade-off, AUC-ROC vs AUC-PR, calibration (Platt/isotonic, ECE), SHAP (TreeSHAP, KernelSHAP), LIME, PDP, ALE, model cards
- **3C** Workflow Orchestration: WorkflowBuilder, LocalRuntime, DataFlow (@db.model, db.express)

## Lab (3h) — 8 Exercises

1. Bias-variance tradeoff and regularisation (L1/L2)
2. Gradient boosting deep dive: XGBoost vs LightGBM vs CatBoost
3. Class imbalance and calibration: SMOTE, Focal Loss, Platt scaling
4. SHAP, LIME, and fairness audit with ModelVisualizer
5. Workflow orchestration with WorkflowBuilder and custom nodes
6. DataFlow persistence: store ML results in database
7. HyperparameterSearch + ModelRegistry: Bayesian opt → staging → production
8. Capstone: end-to-end production pipeline with model card

## Datasets

Singapore Credit Scoring (100K apps, 12% default, protected attributes, leakage trap), Lending Club (300K+)
