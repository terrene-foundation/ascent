# Chapter 5: Full Platform (All 8 Packages)

## Overview

This chapter demonstrates all eight Kailash packages working together in a single Python environment. Each package handles a specific layer of the platform: workflow orchestration, ML lifecycle, database operations, multi-channel deployment, agent framework, specialized agents, organizational governance, and LLM fine-tuning. This chapter validates that all packages import correctly and shows how they compose into a complete ML platform.

## Prerequisites

- All Kailash packages installed:
  - `pip install kailash` (Core SDK)
  - `pip install kailash-ml` (ML engines)
  - `pip install kailash-dataflow` (Database)
  - `pip install kailash-nexus` (Deployment)
  - `pip install kailash-kaizen` (Agent framework)
  - `pip install kaizen-agents` (Specialized agents)
  - `pip install kailash-pact` (Governance)
  - `pip install kailash-align` (LLM fine-tuning)

## Concepts

### Concept 1: The Eight-Package Architecture

Each package has a clear responsibility and clean dependency boundaries:

| Package          | Layer                 | Key Exports                                         |
| ---------------- | --------------------- | --------------------------------------------------- |
| kailash          | Orchestration         | WorkflowBuilder, LocalRuntime, Node                 |
| kailash-ml       | ML Lifecycle          | 13 engines (DataExplorer through InferenceServer)   |
| kailash-dataflow | Persistence           | DataFlow                                            |
| kailash-nexus    | Deployment            | Nexus                                               |
| kailash-kaizen   | Agent Framework       | Signature, InputField, OutputField                  |
| kaizen-agents    | Agent Implementations | Delegate, GovernedSupervisor                        |
| kailash-pact     | Governance            | Address, GovernanceEngine, compile_org, can_access  |
| kailash-align    | LLM Fine-Tuning       | AlignmentPipeline, AlignmentConfig, AdapterRegistry |

- **What**: Eight packages that compose into a complete ML platform
- **Why**: Clean separation means you install only what you need -- a training-only deployment does not need Nexus or Kaizen
- **How**: Each package exposes a focused API surface; integration happens through shared types and protocols
- **When**: The full platform is the target for production ML systems; individual packages suffice for development

### Concept 2: Composition Over Monolith

Kailash is not a monolithic framework. Each package is independently versioned, independently installable, and independently useful. They compose through shared conventions (polars DataFrames, async patterns, frozen configs) rather than tight coupling.

### Key API

This chapter validates imports from all eight packages. The key exports are listed below.

| Package       | Import Path                          | Key Classes                                  |
| ------------- | ------------------------------------ | -------------------------------------------- |
| Core SDK      | `from kailash import ...`            | `WorkflowBuilder`, `LocalRuntime`, `Node`    |
| ML            | `from kailash_ml import ...`         | 13 engine classes                            |
| DataFlow      | `from dataflow import ...`           | `DataFlow`                                   |
| Nexus         | `from nexus import ...`              | `Nexus`                                      |
| Kaizen        | `from kaizen import ...`             | `Signature`, `InputField`, `OutputField`     |
| Kaizen Agents | `from kaizen_agents import ...`      | `Delegate`, `GovernedSupervisor`             |
| PACT          | `from kailash.trust.pact import ...` | `Address`, `GovernanceEngine`, `compile_org` |
| Align         | `from kailash_align import ...`      | `AlignmentPipeline`, `AlignmentConfig`       |

## Code Walkthrough

### Package 1: kailash (Core SDK)

```python
from kailash import WorkflowBuilder, LocalRuntime, Node

builder = WorkflowBuilder()
assert builder is not None
print("1/8 kailash (core): WorkflowBuilder, LocalRuntime, Node")
```

The Core SDK provides workflow orchestration -- the foundation that all other packages build upon.

### Package 2: kailash-ml

```python
from kailash_ml import (
    DataExplorer,
    PreprocessingPipeline,
    ModelVisualizer,
    FeatureEngineer,
    FeatureStore,
    ExperimentTracker,
    TrainingPipeline,
    HyperparameterSearch,
    AutoMLEngine,
    EnsembleEngine,
    ModelRegistry,
    DriftMonitor,
    InferenceServer,
)

print("2/8 kailash-ml: 13 engines loaded")
```

All 13 ML engines cover the complete ML lifecycle: from data exploration through training to production inference and drift monitoring.

### Package 3: kailash-dataflow

```python
from dataflow import DataFlow

db = DataFlow(database_url="sqlite:///")
assert isinstance(db, DataFlow)
print("3/8 kailash-dataflow: DataFlow")
```

DataFlow provides zero-config database operations. The `sqlite:///` URL creates an in-memory database for testing.

### Package 4: kailash-nexus

```python
from nexus import Nexus

app = Nexus(enable_durability=False)
assert isinstance(app, Nexus)
print("4/8 kailash-nexus: Nexus")
```

Nexus deploys workflows as multi-channel endpoints (HTTP API + MCP tools simultaneously).

### Package 5: kailash-kaizen

```python
from kaizen import Signature, InputField, OutputField

assert Signature is not None
print("5/8 kailash-kaizen: Signature, InputField, OutputField")
```

Kaizen provides the AI agent framework: signatures define agent behavior, fields declare inputs and outputs.

### Package 6: kaizen-agents

```python
from kaizen_agents import Delegate, GovernedSupervisor

assert Delegate is not None
assert GovernedSupervisor is not None
print("6/8 kaizen-agents: Delegate, GovernedSupervisor")
```

Kaizen Agents provides pre-built agent implementations: Delegate for autonomous tasks, GovernedSupervisor for governance-wrapped multi-agent pipelines.

### Package 7: kailash-pact

```python
from kailash.trust.pact import (
    Address,
    GovernanceEngine,
    compile_org,
    can_access,
    explain_access,
)

addr = Address.parse("D1-R1")
assert addr.depth() == 2
print("7/8 kailash-pact: Address, GovernanceEngine, compile_org")
```

PACT provides organizational governance: D/T/R addressing, clearance levels, constraint envelopes, and access decisions.

### Package 8: kailash-align

```python
from kailash_align import (
    AlignmentPipeline,
    AlignmentConfig,
    AdapterRegistry,
    AlignmentEvaluator,
)

assert AlignmentConfig is not None
print("8/8 kailash-align: AlignmentPipeline, AlignmentConfig, AdapterRegistry")
```

Align handles LLM fine-tuning and alignment: configuration, training, adapter management, evaluation, and serving.

### Platform Summary

```
All 8 Kailash packages imported and validated:
  1. kailash       -- Workflow orchestration
  2. kailash-ml    -- 13 ML engines
  3. kailash-dataflow -- Database operations
  4. kailash-nexus -- Multi-channel deployment
  5. kailash-kaizen -- Agent framework
  6. kaizen-agents -- Specialized agents
  7. kailash-pact  -- Governance
  8. kailash-align -- LLM fine-tuning
```

## Exercises

1. For each of the eight packages, write a one-sentence description of when you would use it in isolation (without the others). Which packages are most commonly used together?
2. Create a minimal integration that uses four packages: Core SDK (workflow), ML (training), DataFlow (storage), and Nexus (deployment). Describe what each package contributes.
3. Which packages would you install for a project that only does LLM fine-tuning and serving, with no classical ML? Which packages are unnecessary?

## Key Takeaways

- Kailash is composed of eight independently installable packages, not a monolith
- Each package has a focused responsibility: orchestration, ML, database, deployment, agents, governance, alignment
- All packages share conventions: polars DataFrames, async patterns, frozen configs, builder patterns
- The full platform combines all eight for a governed ML system
- Install only what you need -- a fine-tuning project does not require DataFlow or Nexus

## Next Section

This concludes the Integration section. Return to the [Align section](../07-align/01_alignment_config.md) or the [PACT section](../06-pact/01_addressing.md) for deeper coverage of individual packages.
